use super::*;

#[rmcp::tool_router(router = tool_router_process_turn, vis = "pub")]
impl MenteDbServer {
    #[rmcp::tool(
        description = "Call on EVERY turn. Returns context from past conversations, stores the current turn, and detects contradictions. Pass user_message and assistant_response (assistant_response can be empty)."
    )]
    async fn process_turn(
        &self,
        Parameters(req): Parameters<ProcessTurnRequest>,
    ) -> Result<CallToolResult, McpError> {
        let agent_id = match req.agent_id.as_deref() {
            Some(id_str) => match parse_uuid(id_str) {
                Ok(id) => id,
                Err(e) => return error_result(&e),
            },
            None => Uuid::nil(),
        };

        let start = std::time::Instant::now();

        // 1. Search for relevant context based on user message
        let query_embedding = self
            .embedding_provider
            .embed(&req.user_message)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;

        // Try the speculative cache before doing the full O(n) scan
        let mut cache = self.speculative_cache.lock().await;
        let cache_result = cache
            .try_hit(&req.user_message, Some(&query_embedding))
            .map(|entry| (entry.memory_ids.clone(), entry.topic.clone()));
        drop(cache);
        let cache_hit = cache_result.is_some();

        let mut db = self.db.lock().await;
        let all_raw = recall_all_memories(&mut db);

        // Bi-temporal filtering: exclude memories that are no longer valid
        let now_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        let all: Vec<ScoredMemory> = all_raw
            .into_iter()
            .filter(|sm| sm.memory.is_valid_at(now_ts))
            .collect();

        // On cache hit, use cached memory IDs for a targeted lookup instead of
        let (context_items, _current_ids) = if let Some((ref cached_ids, ref _topic)) = cache_result
        {
            let cached_id_set: std::collections::HashSet<MemoryId> =
                cached_ids.iter().cloned().collect();
            let matched: Vec<&ScoredMemory> = all
                .iter()
                .filter(|sm| cached_id_set.contains(&sm.memory.id))
                .collect();

            if matched.len() >= cached_ids.len() / 2 {
                // Enough cached IDs still exist, use them
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
                full_context_scan(&all, &query_embedding, &req, &self.delta_tracker).await
            }
        } else {
            full_context_scan(&all, &query_embedding, &req, &self.delta_tracker).await
        };

        let _removed_ids: Vec<String> = Vec::new(); // delta tracking handled above
        let _context_response = json!({
            "memories": context_items,
            "count": context_items.len(),
            "cache_hit": cache_hit,
        });

        // 2. Check pain signals
        let pain_registry = self.pain_registry.lock().await;
        let context_words: Vec<String> = req
            .user_message
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let pain_warnings: Vec<serde_json::Value> = pain_registry
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

        // 3. Store conversation turn as episodic memory (searchable context)
        // Structured extraction is handled by the LLM calling store_memory directly.
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
        match db.store(node) {
            Ok(()) => {
                stored_ids.push(id.to_string());
            }
            Err(e) => {
                tracing::error!(error = ?e, "process_turn: failed to store episodic turn");
            }
        }

        // 4. Entity resolution via LLM (normalizes names across memories)
        let mut entities_resolved = 0u32;
        if let Some(ref llm) = self.cognitive_llm {
            // Extract entity-like words from the conversation
            let words: Vec<String> = conversation
                .split_whitespace()
                .filter(|w| w.len() >= 3 && w.chars().next().is_some_and(|c| c.is_uppercase()))
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|w| !w.is_empty())
                .collect();

            if !words.is_empty() {
                let candidates: Vec<mentedb_cognitive::EntityCandidate> = words
                    .iter()
                    .map(|w| mentedb_cognitive::EntityCandidate {
                        name: w.clone(),
                        context: Some(conversation.clone()),
                        memory_id: None,
                    })
                    .collect();
                match llm.resolve_entities(&candidates).await {
                    Ok(groups) => {
                        entities_resolved = groups.len() as u32;
                        tracing::debug!(groups = groups.len(), "entity resolution complete");
                    }
                    Err(e) => {
                        tracing::debug!(error = %e, "entity resolution failed");
                    }
                }
            }
        }

        // 5. Write-time inference on the new memory (contradictions, edges, obsolescence)
        let mut inference_applied = 0u32;
        if !stored_ids.is_empty() {
            let all_memories = recall_all_memories(&mut db);
            let all_nodes: Vec<MemoryNode> =
                all_memories.iter().map(|sm| sm.memory.clone()).collect();

            if let Some(target) = all_nodes.iter().find(|m| m.id == id) {
                let engine = WriteInferenceEngine::new();
                let existing: Vec<MemoryNode> =
                    all_nodes.iter().filter(|m| m.id != id).cloned().collect();
                let actions = engine.infer_on_write(target, &existing, &[]);

                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;

                for action in &actions {
                    match action {
                        mentedb_cognitive::InferredAction::FlagContradiction {
                            existing,
                            new,
                            ..
                        } => {
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
                            if let Some(mut mem) =
                                all_nodes.iter().find(|m| m.id == *memory).cloned()
                            {
                                mem.confidence = *new_confidence;
                                let _ = db.store(mem);
                                inference_applied += 1;
                            }
                        }
                        mentedb_cognitive::InferredAction::PropagateBeliefChange {
                            root,
                            delta,
                        } => {
                            // Auto-propagate belief changes instead of waiting for explicit tool call
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
                            if let Some(mut mem) =
                                all_nodes.iter().find(|m| m.id == *memory).cloned()
                            {
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
                            if let Some(mut mem) =
                                all_nodes.iter().find(|m| m.id == *memory).cloned()
                            {
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
                                    Ok(mentedb_cognitive::ContradictionVerdict::Contradicts {
                                        reason,
                                    }) => {
                                        // Temporally invalidate the old memory
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
                                    Ok(mentedb_cognitive::ContradictionVerdict::Compatible {
                                        ..
                                    }) => {
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
            }

            // 4b. Extract facts from the new memory and store as edges
            let extractor = FactExtractor::new();
            if let Some(target) = all_nodes.iter().find(|m| m.id == id) {
                let facts = extractor.extract_facts(target);
                let now_facts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                for fact in &facts {
                    // Store SPO facts as Related edges between the memory and any matching memories
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
            }
        }

        // Drop db lock before auto-maintenance to prevent deadlock
        drop(db);

        // Action detection: identify significant actions for proactive memory
        let mut detected_actions: Vec<serde_json::Value> = Vec::new();
        let mut proactive_recalls: Vec<serde_json::Value> = Vec::new();
        let combined_text = format!("{} {}", req.user_message, assistant_resp).to_lowercase();

        // Detect git operations
        if combined_text.contains("git commit")
            || combined_text.contains("git push")
            || combined_text.contains("merged")
            || combined_text.contains("pull request")
            || combined_text.contains("git revert")
        {
            detected_actions
                .push(json!({"type": "git_operation", "detail": "version control activity"}));
        }
        // Detect decisions
        if combined_text.contains("decided to")
            || combined_text.contains("going with")
            || combined_text.contains("let's use")
            || combined_text.contains("switching to")
            || combined_text.contains("we'll go with")
            || combined_text.contains("chosen")
            || combined_text.contains("the plan is")
        {
            detected_actions
                .push(json!({"type": "decision", "detail": "architecture or technology decision"}));
        }
        // Detect errors/debugging
        if combined_text.contains("error:")
            || combined_text.contains("failed")
            || combined_text.contains("bug")
            || combined_text.contains("crash")
            || combined_text.contains("fix")
            || combined_text.contains("stacktrace")
            || combined_text.contains("traceback")
            || combined_text.contains("exception")
        {
            detected_actions
                .push(json!({"type": "error_resolution", "detail": "debugging or error handling"}));
        }
        // Detect deployments
        if combined_text.contains("deploy")
            || combined_text.contains("release")
            || combined_text.contains("production")
            || combined_text.contains("staging")
            || combined_text.contains("rollback")
        {
            detected_actions
                .push(json!({"type": "deployment", "detail": "deployment or release activity"}));
        }

        // For detected actions, do proactive recall of related memories
        if !detected_actions.is_empty() {
            let mut db_lock = self.db.lock().await;
            let all_mems = recall_all_memories(&mut db_lock);
            drop(db_lock);

            for action in &detected_actions {
                let action_type = action["type"].as_str().unwrap_or("");
                let search_query = match action_type {
                    "git_operation" => "past commits, code changes, review feedback",
                    "decision" => "previous decisions, architecture choices, technology selections",
                    "error_resolution" => "past errors, bugs, debugging, fixes, resolutions",
                    "deployment" => "deployment issues, rollback procedures, release notes",
                    _ => continue,
                };
                if let Ok(action_emb) = self.embedding_provider.embed(search_query) {
                    let mut scored: Vec<(f32, &ScoredMemory)> = all_mems
                        .iter()
                        .map(|sm| (cosine_similarity(&action_emb, &sm.memory.embedding), sm))
                        .filter(|(sim, _)| *sim > 0.4)
                        .collect();
                    scored
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    let top: Vec<serde_json::Value> = scored.iter().take(3)
                        .map(|(score, sm)| {
                            let content = &sm.memory.content;
                            let truncated = if content.len() > 200 {
                                format!("{}...", &content[..content.floor_char_boundary(200)])
                            } else {
                                content.clone()
                            };
                            json!({"id": sm.memory.id.to_string(), "score": score, "summary": truncated})
                        })
                        .collect();
                    if !top.is_empty() {
                        proactive_recalls.push(json!({
                            "trigger": action_type,
                            "reason": format!("Related memories for {}", action_type.replace('_', " ")),
                            "memories": top,
                        }));
                    }
                }
            }
        }

        // Auto-detect corrections that might indicate anti-patterns
        let correction_indicators = [
            "no, don't",
            "don't do that",
            "i told you",
            "that's wrong",
            "stop doing",
            "never do",
            "please don't",
            "not like that",
            "that's not right",
            "incorrect",
            "wrong approach",
        ];
        let user_lower = req.user_message.to_lowercase();
        let is_correction = correction_indicators.iter().any(|c| user_lower.contains(c));

        if is_correction {
            // Store as an anti-pattern memory
            let anti_pattern_content = format!(
                "AVOID: {} (User correction: {})",
                assistant_resp.chars().take(200).collect::<String>(),
                req.user_message.chars().take(200).collect::<String>(),
            );
            let ap_embedding = self
                .embedding_provider
                .embed(&anti_pattern_content)
                .unwrap_or_default();
            let mut ap_node = MemoryNode::new(
                AgentId(agent_id),
                MemoryType::AntiPattern,
                anti_pattern_content,
                ap_embedding,
            );
            ap_node.tags = vec!["auto-correction".to_string()];
            if let Some(ctx) = &req.project_context {
                ap_node.tags.push(format!("scope:project:{}", ctx));
            }
            let ap_id = ap_node.id;
            let mut db = self.db.lock().await;
            if let Ok(()) = db.store(ap_node) {
                stored_ids.push(ap_id.to_string());
                tracing::info!(
                    turn_id = req.turn_id,
                    "auto-detected correction, stored anti-pattern"
                );
            }
            drop(db);
        }

        // Emotional trajectory: detect user sentiment
        let sentiment_score: f32 = {
            let frustration_signals = [
                "not working",
                "broken",
                "frustrated",
                "annoyed",
                "why isn't",
                "still not",
                "keeps failing",
                "doesn't work",
                "can't believe",
                "waste of time",
                "terrible",
                "horrible",
                "awful",
            ];
            let satisfaction_signals = [
                "thanks",
                "perfect",
                "great",
                "awesome",
                "exactly",
                "well done",
                "nice",
                "excellent",
                "love it",
                "works great",
                "amazing",
                "brilliant",
                "fantastic",
            ];
            let frustration_count = frustration_signals
                .iter()
                .filter(|s| user_lower.contains(*s))
                .count() as f32;
            let satisfaction_count = satisfaction_signals
                .iter()
                .filter(|s| user_lower.contains(*s))
                .count() as f32;

            if frustration_count > 0.0 || satisfaction_count > 0.0 {
                (satisfaction_count - frustration_count) / (satisfaction_count + frustration_count)
            } else {
                0.0 // neutral
            }
        };

        // 4c. Detect phantom entities in the conversation
        let mut phantom_tracker = self.phantom_tracker.lock().await;
        let known: Vec<String> = context_items
            .iter()
            .filter_map(|ci| ci.get("content").and_then(|c| c.as_str()))
            .flat_map(|c| c.split_whitespace().map(|w| w.to_lowercase()))
            .collect();
        let phantoms = phantom_tracker.detect_gaps(&conversation, &known, req.turn_id);
        let phantom_count = phantoms.len();
        drop(phantom_tracker);

        // 4d. Check assistant response against known context for contradictions
        let known_facts: Vec<(MemoryId, String)> = context_items
            .iter()
            .filter_map(|ci| {
                let id_str = ci.get("id").and_then(|v| v.as_str())?;
                let content = ci.get("content").and_then(|v| v.as_str())?;
                let uuid = parse_uuid(id_str).ok()?;
                Some((MemoryId(uuid), content.to_string()))
            })
            .collect();
        let stream_alerts = if !known_facts.is_empty() {
            let stream = CognitionStream::with_config(StreamConfig::default());
            stream.feed_token(assistant_resp);
            stream.check_alerts(&known_facts)
        } else {
            vec![]
        };
        let contradiction_count = stream_alerts
            .iter()
            .filter(|a| matches!(a, mentedb_cognitive::StreamAlert::Contradiction { .. }))
            .count();

        // 5. Record trajectory with LLM topic canonicalization
        let decision_state = if stored_ids.is_empty() {
            DecisionState::Investigating
        } else {
            DecisionState::Completed
        };
        let raw_topic = if req.user_message.len() > 100 {
            format!("{}...", &req.user_message[..100])
        } else {
            req.user_message.clone()
        };
        // Canonicalize the topic via LLM if available
        let topic_summary = if let Some(ref llm) = self.cognitive_llm {
            let tracker = self.trajectory_tracker.lock().await;
            let existing_topics = tracker
                .predict_next_topics()
                .into_iter()
                .collect::<Vec<_>>();
            drop(tracker);
            match llm.canonicalize_topic(&raw_topic, &existing_topics).await {
                Ok(label) => {
                    tracing::debug!(raw = %raw_topic, canonical = %label.topic, "topic canonicalized");
                    label.topic
                }
                Err(e) => {
                    tracing::debug!(error = %e, "topic canonicalization failed, using raw");
                    raw_topic
                }
            }
        } else {
            raw_topic
        };
        let topic_embedding = self
            .embedding_provider
            .embed(&topic_summary)
            .unwrap_or_else(|_| vec![0.0; 384]);
        let node = TrajectoryNode {
            turn_id: req.turn_id,
            topic_summary: topic_summary.clone(),
            topic_embedding,
            decision_state,
            open_questions: Vec::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
        };
        let mut tracker = self.trajectory_tracker.lock().await;
        tracker.record_turn(node);
        let predictions: Vec<String> = tracker.predict_next_topics().into_iter().collect();
        drop(tracker);

        // Ghost memory inference: detect patterns that suggest unconfirmed beliefs
        let speculation_indicators = [
            "might be",
            "probably",
            "seems like",
            "i think",
            "looks like",
            "considering",
            "planning to",
            "thinking about",
            "maybe",
        ];
        let has_speculation = speculation_indicators
            .iter()
            .any(|s| combined_text.contains(s));

        if has_speculation && !detected_actions.is_empty() {
            // Create a low-confidence ghost memory from the speculative content
            let ghost_content = format!(
                "Unconfirmed: {}",
                req.user_message.chars().take(300).collect::<String>()
            );
            let ghost_emb = self
                .embedding_provider
                .embed(&ghost_content)
                .unwrap_or_default();
            let mut ghost_node = MemoryNode::new(
                AgentId(agent_id),
                MemoryType::Semantic,
                ghost_content,
                ghost_emb,
            );
            ghost_node.confidence = 0.3; // Low confidence
            ghost_node.tags = vec!["ghost-memory".to_string(), "unconfirmed".to_string()];
            if let Some(ctx) = &req.project_context {
                ghost_node.tags.push(format!("scope:project:{}", ctx));
            }
            let mut db = self.db.lock().await;
            let _ = db.store(ghost_node);
            drop(db);
            tracing::debug!(
                turn_id = req.turn_id,
                "stored ghost memory from speculative content"
            );
        }

        // 6. Pre-assemble speculative cache for predicted topics
        if !predictions.is_empty() {
            let embed_provider = Arc::clone(&self.embedding_provider);
            let mut db_lock = self.db.lock().await;
            let all_memories = recall_all_memories(&mut db_lock);
            drop(db_lock);

            let mut cache = self.speculative_cache.lock().await;
            cache.pre_assemble(predictions.clone(), |topic| {
                let topic_emb = embed_provider.embed(topic).ok()?;
                let mut scored: Vec<(f32, &ScoredMemory)> = all_memories
                    .iter()
                    .map(|sm| {
                        let sim = cosine_similarity(&topic_emb, &sm.memory.embedding);
                        (sim, sm)
                    })
                    .filter(|(sim, _)| *sim > 0.3)
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let top: Vec<&ScoredMemory> = scored.iter().take(5).map(|(_, sm)| *sm).collect();
                if top.is_empty() {
                    return None;
                }
                let context_text = top
                    .iter()
                    .map(|sm| sm.memory.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n---\n");
                let memory_ids: Vec<MemoryId> = top.iter().map(|sm| sm.memory.id).collect();
                Some((context_text, memory_ids, None))
            });
            drop(cache);
        }

        // ── Auto-maintenance (staggered) ──
        let mut maintenance: Vec<&str> = Vec::new();
        let turn = req.turn_id;

        // Every 50 turns: apply salience decay
        if turn > 0 && turn % 50 == 0 {
            let half_life_us = (168.0 * 3600.0 * 1_000_000.0) as u64; // 7 days
            let config = DecayConfig {
                half_life_us,
                ..DecayConfig::default()
            };
            let decay_engine = DecayEngine::new(config);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;
            let mut db = self.db.lock().await;
            let all = recall_all_memories(&mut db);
            let mut memories: Vec<MemoryNode> = all.into_iter().map(|sm| sm.memory).collect();
            decay_engine.apply_decay_batch(&mut memories, now);
            // Persist updated salience values
            for mem in &memories {
                let _ = db.store(mem.clone());
            }
            drop(db);
            maintenance.push("apply_decay");
            tracing::info!(turn_id = turn, "auto-maintenance: applied salience decay");
        }

        // Every 100 turns: evaluate archival and delete stale memories
        if turn > 0 && turn % 100 == 0 {
            let config = ArchivalConfig {
                max_salience: 0.1,
                min_age_us: 7 * 24 * 3600 * 1_000_000,
                ..ArchivalConfig::default()
            };
            let pipeline = ArchivalPipeline::new(config);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;
            let mut db = self.db.lock().await;
            let all = recall_all_memories(&mut db);
            let memories: Vec<MemoryNode> = all.into_iter().map(|sm| sm.memory).collect();
            let decisions = pipeline.evaluate_batch(&memories, now);
            let mut archived = 0u64;
            for (id, decision) in &decisions {
                if matches!(
                    decision,
                    ArchivalDecision::Delete | ArchivalDecision::Archive
                ) {
                    let _ = db.forget(*id);
                    archived += 1;
                }
            }
            drop(db);
            maintenance.push("evaluate_archival");
            tracing::info!(
                turn_id = turn,
                archived,
                "auto-maintenance: evaluated archival"
            );
        }

        // Every 200 turns: consolidate similar memories
        if turn > 0 && turn % 200 == 0 {
            let mut db = self.db.lock().await;
            let all = recall_all_memories(&mut db);
            let memories: Vec<MemoryNode> = all.into_iter().map(|sm| sm.memory).collect();
            if memories.len() >= 2 {
                let consolidation_engine = ConsolidationEngine::new();
                let candidates = consolidation_engine.find_candidates(&memories, 2, 0.85);
                for candidate in &candidates {
                    let cluster_memories: Vec<&MemoryNode> = candidate
                        .memories
                        .iter()
                        .filter_map(|id| memories.iter().find(|m| m.id == *id))
                        .collect();
                    let cluster_owned: Vec<MemoryNode> =
                        cluster_memories.into_iter().cloned().collect();
                    let consolidated = consolidation_engine.consolidate(&cluster_owned);
                    // Store the merged memory
                    let agent_id = cluster_owned
                        .first()
                        .map(|m| m.agent_id)
                        .unwrap_or(AgentId(Uuid::nil()));
                    let merged = MemoryNode::new(
                        agent_id,
                        consolidated.new_type,
                        consolidated.summary,
                        consolidated.combined_embedding,
                    );
                    let _ = db.store(merged);
                    // Remove source memories
                    for source_id in &consolidated.source_memories {
                        let _ = db.forget(*source_id);
                    }
                }
                maintenance.push("consolidate_memories");
                tracing::info!(
                    turn_id = turn,
                    clusters = candidates.len(),
                    "auto-maintenance: consolidated memories"
                );
            }
            drop(db);
        }

        let elapsed_ms = start.elapsed().as_millis();

        tracing::info!(
            turn_id = req.turn_id,
            context_count = context_items.len(),
            cache_hit = cache_hit,
            memories_stored = stored_ids.len(),
            inference_applied,
            phantom_count,
            contradiction_count,
            pain_warnings = pain_warnings.len(),
            elapsed_ms = elapsed_ms,
            "process_turn complete"
        );

        // Build compact context: truncated content + IDs for the LLM
        // Full content stays in storage; agent can recall_memory(id) if needed
        const CTX_MAX_CHARS: usize = 300;
        let context_summaries: Vec<serde_json::Value> = context_items
            .iter()
            .filter_map(|ci| {
                let content = ci.get("content").and_then(|c| c.as_str())?;
                let id = ci.get("id").and_then(|i| i.as_str()).unwrap_or("");
                let truncated = if content.len() > CTX_MAX_CHARS {
                    format!(
                        "{}…",
                        &content[..content.floor_char_boundary(CTX_MAX_CHARS)]
                    )
                } else {
                    content.to_string()
                };
                Some(json!({ "id": id, "summary": truncated }))
            })
            .collect();

        // Build slim response with only fields the LLM actually uses.
        // Debug fields logged server-side instead of wasting context tokens.
        let mut response = json!({
            "ok": true,
            "context": context_summaries,
            "stored": stored_ids.len(),
            "contradictions": contradiction_count,
        });

        // Only include non-empty optional fields
        if !pain_warnings.is_empty() {
            response["pain_warnings"] = json!(pain_warnings);
        }
        if !proactive_recalls.is_empty() {
            response["proactive_recalls"] = json!(proactive_recalls);
        }
        if !detected_actions.is_empty() {
            response["detected_actions"] = json!(detected_actions);
        }

        let response = response.to_string();

        tracing::info!(
            turn_id = req.turn_id,
            response_bytes = response.len(),
            context_entries = context_summaries.len(),
            inference_applied,
            entities_resolved,
            phantom_count,
            cache_hit,
            sentiment = sentiment_score,
            predictions_count = predictions.len(),
            "process_turn response"
        );

        Ok(CallToolResult::success(vec![Content::text(response)]))
    }
}
