use super::*;

// ── Internal helper methods called by the process_turn orchestrator ──

impl MenteDbServer {
    /// §1: Retrieve relevant context via speculative cache or full cosine scan.
    pub(super) async fn retrieve_context(
        &self,
        user_message: &str,
        query_embedding: &[f32],
        req: &ProcessTurnRequest,
    ) -> (Vec<serde_json::Value>, Vec<MemoryId>, bool) {
        let mut cache = self.speculative_cache.lock().await;
        let cache_result = cache
            .try_hit(user_message, Some(query_embedding))
            .map(|entry| (entry.memory_ids.clone(), entry.topic.clone()));
        drop(cache);
        let cache_hit = cache_result.is_some();

        let mut db = self.db.lock().await;
        let all_raw = recall_all_memories(&mut db);
        drop(db);

        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        let all: Vec<ScoredMemory> = all_raw
            .into_iter()
            .filter(|sm| sm.memory.is_valid_at(now_ts))
            .collect();

        let (context_items, current_ids) = if let Some((ref cached_ids, ref _topic)) = cache_result
        {
            let cached_id_set: std::collections::HashSet<MemoryId> =
                cached_ids.iter().cloned().collect();
            let matched: Vec<&ScoredMemory> = all
                .iter()
                .filter(|sm| cached_id_set.contains(&sm.memory.id))
                .collect();

            if matched.len() >= cached_ids.len() / 2 {
                let ids: Vec<MemoryId> = matched.iter().map(|sm| sm.memory.id).collect();
                let mut delta_tracker = self.delta_tracker.lock().await;
                let delta = delta_tracker.compute_delta(&ids, &delta_tracker.last_served.clone());
                delta_tracker.update(&ids);
                drop(delta_tracker);

                let items: Vec<serde_json::Value> = matched
                    .iter()
                    .take(5)
                    .map(|sm| {
                        let mut val = memory_node_to_json(&sm.memory);
                        val["relevance_score"] = json!(0.9);
                        val["is_new"] = json!(delta.added.contains(&sm.memory.id));
                        val["from_cache"] = json!(true);
                        if let Some(ctx) = &req.project_context {
                            val["same_project"] = json!(sm.memory.tags.iter().any(|t| t == ctx));
                        }
                        val
                    })
                    .collect();

                tracing::debug!(
                    cached_ids = cached_ids.len(),
                    matched = matched.len(),
                    "Speculative cache hit, skipped full scan"
                );
                (items, ids)
            } else {
                tracing::debug!(
                    cached_ids = cached_ids.len(),
                    matched = matched.len(),
                    "Speculative cache hit but IDs stale, falling back to full scan"
                );
                full_context_scan(&all, query_embedding, req, &self.delta_tracker).await
            }
        } else {
            full_context_scan(&all, query_embedding, req, &self.delta_tracker).await
        };

        (context_items, current_ids, cache_hit)
    }

    /// §2: Check pain signals against the current user message.
    pub(super) async fn check_pain_signals(&self, user_message: &str) -> Vec<serde_json::Value> {
        let pain_registry = self.pain_registry.lock().await;
        let context_words: Vec<String> = user_message
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let warnings: Vec<serde_json::Value> = pain_registry
            .get_pain_for_context(&context_words)
            .iter()
            .map(|s| {
                json!({
                    "signal_id": s.id.to_string(),
                    "intensity": s.intensity,
                    "description": &s.description,
                })
            })
            .collect();
        drop(pain_registry);
        warnings
    }

    /// §3: Store the conversation turn as an episodic memory.
    /// Returns (stored ID strings, the MemoryId, the conversation text).
    pub(super) async fn store_episodic_turn(
        &self,
        req: &ProcessTurnRequest,
        agent_id: Uuid,
    ) -> (Vec<String>, MemoryId, String) {
        let assistant_resp = req.assistant_response.as_deref().unwrap_or("");
        let conversation = format!("User: {}\nAssistant: {}", req.user_message, assistant_resp);

        let mut stored_ids = Vec::new();
        let embedding = self
            .embedding_provider
            .embed(&conversation)
            .unwrap_or_default();
        let mut node = MemoryNode::new(
            AgentId(agent_id),
            MemoryType::Episodic,
            conversation.clone(),
            embedding,
        );
        node.tags = vec!["conversation-turn".to_string()];
        if let Some(ctx) = &req.project_context {
            node.tags.push(format!("scope:project:{}", ctx));
        }
        let id = node.id;
        let mut db = self.db.lock().await;
        match db.store(node) {
            Ok(()) => {
                stored_ids.push(id.to_string());
            }
            Err(e) => {
                tracing::error!(error = ?e, "process_turn: failed to store episodic turn");
            }
        }
        drop(db);
        (stored_ids, id, conversation)
    }

    /// §4: Entity resolution via LLM (normalizes names across memories).
    pub(super) async fn resolve_entities_llm(&self, conversation: &str) -> u32 {
        let Some(ref llm) = self.cognitive_llm else {
            return 0;
        };
        let words: Vec<String> = conversation
            .split_whitespace()
            .filter(|w| w.len() >= 3 && w.chars().next().is_some_and(|c| c.is_uppercase()))
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return 0;
        }

        let candidates: Vec<mentedb_cognitive::EntityCandidate> = words
            .iter()
            .map(|w| mentedb_cognitive::EntityCandidate {
                name: w.clone(),
                context: Some(conversation.to_string()),
                memory_id: None,
            })
            .collect();
        match llm.resolve_entities(&candidates).await {
            Ok(groups) => {
                let count = groups.len() as u32;
                tracing::debug!(groups = groups.len(), "entity resolution complete");
                count
            }
            Err(e) => {
                tracing::debug!(error = %e, "entity resolution failed");
                0
            }
        }
    }

    /// §5: Write-time inference on the new memory (contradictions, edges, obsolescence)
    /// plus LLM-powered contradiction verification.
    pub(super) async fn run_write_inference(&self, id: MemoryId, stored_ids: &[String]) -> u32 {
        if stored_ids.is_empty() {
            return 0;
        }
        let mut db = self.db.lock().await;
        let all_memories = recall_all_memories(&mut db);
        let all_nodes: Vec<MemoryNode> = all_memories.iter().map(|sm| sm.memory.clone()).collect();

        let Some(target) = all_nodes.iter().find(|m| m.id == id) else {
            drop(db);
            return 0;
        };

        let engine = WriteInferenceEngine::new();
        let existing: Vec<MemoryNode> = all_nodes.iter().filter(|m| m.id != id).cloned().collect();
        let actions = engine.infer_on_write(target, &existing, &[]);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let mut inference_applied = 0u32;
        for action in &actions {
            match action {
                mentedb_cognitive::InferredAction::FlagContradiction { existing, new, .. } => {
                    let edge = MemoryEdge {
                        source: *new,
                        target: *existing,
                        edge_type: EdgeType::Contradicts,
                        weight: 1.0,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                    };
                    let _ = db.relate(edge);
                    inference_applied += 1;
                }
                mentedb_cognitive::InferredAction::MarkObsolete {
                    memory,
                    superseded_by,
                } => {
                    let edge = MemoryEdge {
                        source: *superseded_by,
                        target: *memory,
                        edge_type: EdgeType::Supersedes,
                        weight: 1.0,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                    };
                    let _ = db.relate(edge);
                    inference_applied += 1;
                }
                mentedb_cognitive::InferredAction::CreateEdge {
                    source,
                    target,
                    edge_type,
                    weight,
                } => {
                    let edge = MemoryEdge {
                        source: *source,
                        target: *target,
                        edge_type: *edge_type,
                        weight: *weight,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                    };
                    let _ = db.relate(edge);
                    inference_applied += 1;
                }
                mentedb_cognitive::InferredAction::UpdateConfidence {
                    memory,
                    new_confidence,
                } => {
                    if let Some(mut mem) = all_nodes.iter().find(|m| m.id == *memory).cloned() {
                        mem.confidence = *new_confidence;
                        let _ = db.store(mem);
                        inference_applied += 1;
                    }
                }
                mentedb_cognitive::InferredAction::PropagateBeliefChange { root, delta } => {
                    tracing::debug!(
                        root = %root,
                        delta = delta,
                        "auto-propagating belief change"
                    );
                    inference_applied += 1;
                }
                mentedb_cognitive::InferredAction::InvalidateMemory {
                    memory,
                    superseded_by: _,
                    valid_until,
                } => {
                    if let Some(mut mem) = all_nodes.iter().find(|m| m.id == *memory).cloned() {
                        mem.valid_until = Some(*valid_until);
                        let _ = db.store(mem);
                        inference_applied += 1;
                    }
                }
                mentedb_cognitive::InferredAction::UpdateContent {
                    memory,
                    new_content,
                    ..
                } => {
                    if let Some(mut mem) = all_nodes.iter().find(|m| m.id == *memory).cloned() {
                        mem.content = new_content.clone();
                        let _ = db.store(mem);
                        inference_applied += 1;
                    }
                }
            }
        }

        // LLM-powered contradiction verification on flagged contradictions
        if let Some(ref llm) = self.cognitive_llm {
            for action in &actions {
                if let mentedb_cognitive::InferredAction::FlagContradiction {
                    existing: existing_id,
                    new: new_id,
                    ..
                } = action
                {
                    let existing_mem = all_nodes.iter().find(|m| m.id == *existing_id);
                    let new_mem = all_nodes.iter().find(|m| m.id == *new_id);
                    if let (Some(em), Some(nm)) = (existing_mem, new_mem) {
                        let summary_a = mentedb_cognitive::MemorySummary {
                            id: em.id,
                            content: em.content.clone(),
                            memory_type: em.memory_type,
                            confidence: em.confidence,
                            created_at: em.created_at,
                        };
                        let summary_b = mentedb_cognitive::MemorySummary {
                            id: nm.id,
                            content: nm.content.clone(),
                            memory_type: nm.memory_type,
                            confidence: nm.confidence,
                            created_at: nm.created_at,
                        };
                        match llm.detect_contradiction(&summary_a, &summary_b).await {
                            Ok(mentedb_cognitive::ContradictionVerdict::Contradicts { reason }) => {
                                let mut old = em.clone();
                                old.valid_until = Some(now);
                                let _ = db.store(old);
                                tracing::info!(
                                    existing = %existing_id,
                                    new = %new_id,
                                    reason = %reason,
                                    "LLM confirmed contradiction, invalidated old memory"
                                );
                            }
                            Ok(mentedb_cognitive::ContradictionVerdict::Supersedes {
                                winner,
                                reason,
                            }) => {
                                let loser = if winner == existing_id.to_string() {
                                    nm
                                } else {
                                    em
                                };
                                let mut old = loser.clone();
                                old.valid_until = Some(now);
                                let _ = db.store(old);
                                tracing::info!(
                                    winner = %winner,
                                    reason = %reason,
                                    "LLM found supersession, invalidated loser"
                                );
                            }
                            Ok(mentedb_cognitive::ContradictionVerdict::Compatible { .. }) => {
                                tracing::debug!("LLM says compatible, keeping both");
                            }
                            Err(e) => {
                                tracing::debug!(error = %e, "LLM contradiction check failed");
                            }
                        }
                    }
                }
            }
        }
        tracing::info!(
            actions = actions.len(),
            applied = inference_applied,
            "write inference on new memory"
        );

        drop(db);
        inference_applied
    }

    /// §4b: Extract facts from the new memory and store as Related edges.
    pub(super) async fn extract_and_store_facts(&self, id: MemoryId, stored_ids: &[String]) {
        if stored_ids.is_empty() {
            return;
        }
        let mut db = self.db.lock().await;
        let all_memories = recall_all_memories(&mut db);
        let all_nodes: Vec<MemoryNode> = all_memories.iter().map(|sm| sm.memory.clone()).collect();

        let extractor = FactExtractor::new();
        let Some(target) = all_nodes.iter().find(|m| m.id == id) else {
            drop(db);
            return;
        };
        let facts = extractor.extract_facts(target);
        let now_facts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        for fact in &facts {
            for other in &all_nodes {
                if other.id != id
                    && (other.content.contains(&fact.subject)
                        || other.content.contains(&fact.object))
                {
                    let edge = MemoryEdge {
                        source: id,
                        target: other.id,
                        edge_type: EdgeType::Related,
                        weight: 0.5,
                        created_at: now_facts,
                        valid_from: None,
                        valid_until: None,
                    };
                    let _ = db.relate(edge);
                }
            }
        }
        if !facts.is_empty() {
            tracing::info!(
                facts = facts.len(),
                "extracted and linked facts from new memory"
            );
        }
        drop(db);
    }
}
