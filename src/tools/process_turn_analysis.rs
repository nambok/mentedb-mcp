use super::*;

impl MenteDbServer {
    pub(super) fn detect_actions(combined_text: &str) -> Vec<serde_json::Value> {
        let mut detected = Vec::new();
        if combined_text.contains("git commit")
            || combined_text.contains("git push")
            || combined_text.contains("merged")
            || combined_text.contains("pull request")
            || combined_text.contains("git revert")
        {
            detected.push(json!({"type": "git_operation", "detail": "version control activity"}));
        }
        if combined_text.contains("decided to")
            || combined_text.contains("going with")
            || combined_text.contains("let's use")
            || combined_text.contains("switching to")
            || combined_text.contains("we'll go with")
            || combined_text.contains("chosen")
            || combined_text.contains("the plan is")
        {
            detected
                .push(json!({"type": "decision", "detail": "architecture or technology decision"}));
        }
        if combined_text.contains("error:")
            || combined_text.contains("failed")
            || combined_text.contains("bug")
            || combined_text.contains("crash")
            || combined_text.contains("fix")
            || combined_text.contains("stacktrace")
            || combined_text.contains("traceback")
            || combined_text.contains("exception")
        {
            detected
                .push(json!({"type": "error_resolution", "detail": "debugging or error handling"}));
        }
        if combined_text.contains("deploy")
            || combined_text.contains("release")
            || combined_text.contains("production")
            || combined_text.contains("staging")
            || combined_text.contains("rollback")
        {
            detected
                .push(json!({"type": "deployment", "detail": "deployment or release activity"}));
        }
        detected
    }

    /// Proactive recall of related memories for detected actions.
    pub(super) async fn proactive_recall(
        &self,
        detected_actions: &[serde_json::Value],
    ) -> Vec<serde_json::Value> {
        if detected_actions.is_empty() {
            return Vec::new();
        }
        let mut db_lock = self.db.lock().await;
        let all_mems = recall_all_memories(&mut db_lock);
        drop(db_lock);

        let mut proactive_recalls = Vec::new();
        for action in detected_actions {
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
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let top: Vec<serde_json::Value> = scored
                    .iter()
                    .take(3)
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
        proactive_recalls
    }

    /// Auto-detect user corrections and store as anti-pattern memories.
    pub(super) async fn auto_detect_corrections(
        &self,
        user_message: &str,
        assistant_resp: &str,
        agent_id: Uuid,
        project_context: Option<&str>,
    ) -> Option<String> {
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
        let user_lower = user_message.to_lowercase();
        if !correction_indicators.iter().any(|c| user_lower.contains(c)) {
            return None;
        }

        let anti_pattern_content = format!(
            "AVOID: {} (User correction: {})",
            assistant_resp.chars().take(200).collect::<String>(),
            user_message.chars().take(200).collect::<String>(),
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
        if let Some(ctx) = project_context {
            ap_node.tags.push(format!("scope:project:{}", ctx));
        }
        let ap_id = ap_node.id;
        let mut db = self.db.lock().await;
        let result = if db.store(ap_node).is_ok() {
            Some(ap_id.to_string())
        } else {
            None
        };
        drop(db);
        result
    }

    /// Compute a simple keyword-based sentiment score in [-1, 1].
    pub(super) fn analyze_sentiment(user_message: &str) -> f32 {
        let user_lower = user_message.to_lowercase();
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
            0.0
        }
    }

    /// §4c + §4d: Detect phantom entities and check stream contradictions.
    pub(super) async fn detect_phantoms_and_check_stream(
        &self,
        conversation: &str,
        assistant_resp: &str,
        context_items: &[serde_json::Value],
        turn_id: u64,
    ) -> (usize, usize) {
        // Phantom detection
        let mut phantom_tracker = self.phantom_tracker.lock().await;
        let known: Vec<String> = context_items
            .iter()
            .filter_map(|ci| ci.get("content").and_then(|c| c.as_str()))
            .flat_map(|c| c.split_whitespace().map(|w| w.to_lowercase()))
            .collect();
        let phantoms = phantom_tracker.detect_gaps(conversation, &known, turn_id);
        let phantom_count = phantoms.len();
        drop(phantom_tracker);

        // Stream contradiction check
        let known_facts: Vec<(MemoryId, String)> = context_items
            .iter()
            .filter_map(|ci| {
                let id_str = ci.get("id").and_then(|v| v.as_str())?;
                let content = ci.get("content").and_then(|v| v.as_str())?;
                let uuid = parse_uuid(id_str).ok()?;
                Some((MemoryId(uuid), content.to_string()))
            })
            .collect();
        let contradiction_count = if !known_facts.is_empty() {
            let stream = CognitionStream::with_config(StreamConfig::default());
            stream.feed_token(assistant_resp);
            let alerts = stream.check_alerts(&known_facts);
            alerts
                .iter()
                .filter(|a| matches!(a, mentedb_cognitive::StreamAlert::Contradiction { .. }))
                .count()
        } else {
            0
        };

        (phantom_count, contradiction_count)
    }

    /// §5: Record trajectory node with LLM topic canonicalization, store ghost memories.
    /// Returns predicted next topics.
    pub(super) async fn update_trajectory(
        &self,
        req: &ProcessTurnRequest,
        agent_id: Uuid,
        stored_ids: &[String],
        detected_actions: &[serde_json::Value],
        combined_text: &str,
    ) -> Vec<String> {
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
            topic_summary,
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

        // Ghost memory inference
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
            ghost_node.confidence = 0.3;
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

        predictions
    }

    /// §6: Pre-assemble speculative cache for predicted topics.
    pub(super) async fn update_speculative_cache(&self, predictions: &[String]) {
        if predictions.is_empty() {
            return;
        }
        let embed_provider = Arc::clone(&self.embedding_provider);
        let mut db_lock = self.db.lock().await;
        let all_memories = recall_all_memories(&mut db_lock);
        drop(db_lock);

        let mut cache = self.speculative_cache.lock().await;
        cache.pre_assemble(predictions.to_vec(), |topic| {
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

    /// Auto-maintenance: decay, archival, consolidation on staggered intervals.
    pub(super) async fn maybe_run_maintenance(&self, turn_id: u64) {
        if turn_id == 0 {
            return;
        }

        // Every 50 turns: apply salience decay
        if turn_id.is_multiple_of(50) {
            let half_life_us = (168.0 * 3600.0 * 1_000_000.0) as u64;
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
            for mem in &memories {
                let _ = db.store(mem.clone());
            }
            drop(db);
            tracing::info!(turn_id, "auto-maintenance: applied salience decay");
        }

        // Every 100 turns: evaluate archival and delete stale memories
        if turn_id.is_multiple_of(100) {
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
            tracing::info!(turn_id, archived, "auto-maintenance: evaluated archival");
        }

        // Every 200 turns: consolidate similar memories
        if turn_id.is_multiple_of(200) {
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
                    for source_id in &consolidated.source_memories {
                        let _ = db.forget(*source_id);
                    }
                }
                tracing::info!(
                    turn_id,
                    clusters = candidates.len(),
                    "auto-maintenance: consolidated memories"
                );
            }
            drop(db);
        }
    }
}
