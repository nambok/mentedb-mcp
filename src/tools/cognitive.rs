use super::*;

#[rmcp::tool_router(router = tool_router_cognitive, vis = "pub")]
impl MenteDbServer {
    #[rmcp::tool(
        description = "Record a negative experience (pain signal) so MenteDB can warn when similar contexts arise."
    )]
    async fn record_pain(
        &self,
        Parameters(req): Parameters<RecordPainRequest>,
    ) -> Result<CallToolResult, McpError> {
        let memory_id = match parse_uuid(&req.memory_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let signal_id = MemoryId::new();
        let signal = PainSignal {
            id: signal_id,
            memory_id: MemoryId::from(memory_id),
            intensity: req.intensity.clamp(0.0, 1.0),
            trigger_keywords: req.trigger_keywords,
            description: req.description,
            created_at: now,
            decay_rate: 0.1,
        };

        let mut registry = self.pain_registry.lock().await;
        registry.record_pain(signal);
        tracing::info!(signal_id = %signal_id, memory_id = %memory_id, "pain signal recorded");

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "recorded",
                "signal_id": signal_id.to_string(),
                "memory_id": memory_id.to_string(),
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Scan content for knowledge gaps — entities referenced but not present in memory."
    )]
    async fn detect_phantoms(
        &self,
        Parameters(req): Parameters<DetectPhantomsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let known = req.known_entities.unwrap_or_default();
        let turn_id = req.turn_id.unwrap_or(0);

        let mut tracker = self.phantom_tracker.lock().await;
        let phantoms = tracker.detect_gaps(&req.content, &known, turn_id);

        let items: Vec<serde_json::Value> = phantoms
            .iter()
            .map(|p| {
                json!({
                    "id": p.id.to_string(),
                    "gap_description": p.gap_description,
                    "source_reference": p.source_reference,
                    "priority": format!("{:?}", p.priority),
                })
            })
            .collect();

        tracing::info!(
            count = items.len(),
            turn_id = turn_id,
            "phantom detection complete"
        );
        Ok(CallToolResult::success(vec![Content::text(
            json!({ "phantoms": items, "count": items.len() }).to_string(),
        )]))
    }

    #[rmcp::tool(description = "Mark a knowledge gap (phantom memory) as resolved.")]
    async fn resolve_phantom(
        &self,
        Parameters(req): Parameters<ResolvePhantomRequest>,
    ) -> Result<CallToolResult, McpError> {
        let phantom_id = match parse_uuid(&req.phantom_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let mut tracker = self.phantom_tracker.lock().await;
        tracker.resolve(phantom_id);
        tracing::info!(phantom_id = %phantom_id, "phantom resolved");

        Ok(CallToolResult::success(vec![Content::text(
            json!({ "status": "resolved", "phantom_id": phantom_id.to_string() }).to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Record a conversation turn for trajectory tracking. Enables topic prediction and resume context."
    )]
    async fn record_trajectory(
        &self,
        Parameters(req): Parameters<RecordTrajectoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let decision_state = parse_decision_state(&req.decision_state);

        let embedding = self
            .embedding_provider
            .embed(&req.topic_summary)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let node = TrajectoryNode {
            turn_id: req.turn_id,
            topic_embedding: embedding,
            topic_summary: req.topic_summary,
            decision_state,
            open_questions: req.open_questions.unwrap_or_default(),
            timestamp: now,
        };

        let mut tracker = self.trajectory_tracker.lock().await;
        tracker.record_turn(node);
        let trajectory_len = tracker.get_trajectory().len();
        tracing::info!(
            turn_id = req.turn_id,
            trajectory_len,
            "trajectory turn recorded"
        );

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "recorded",
                "turn_id": req.turn_id,
                "trajectory_length": trajectory_len,
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Predict likely next topics based on the current conversation trajectory."
    )]
    async fn predict_topics(&self) -> Result<CallToolResult, McpError> {
        let tracker = self.trajectory_tracker.lock().await;
        let predictions = tracker.predict_next_topics();
        tracing::info!(count = predictions.len(), "topic predictions generated");

        Ok(CallToolResult::success(vec![Content::text(
            json!({ "predictions": predictions, "count": predictions.len() }).to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Detect pairs of memories similar enough to confuse an LLM, with disambiguation hints."
    )]
    async fn detect_interference(
        &self,
        Parameters(req): Parameters<DetectInterferenceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let threshold = req.similarity_threshold.unwrap_or(0.8);
        let detector = InterferenceDetector::new(threshold);

        let mut db = self.db.lock().await;
        let mut memories: Vec<MemoryNode> = Vec::new();
        for id_str in &req.memory_ids {
            let id = match parse_uuid(id_str) {
                Ok(id) => id,
                Err(e) => return error_result(&e),
            };
            match find_memory_by_id(&mut db, id) {
                Ok(Some(sm)) => memories.push(sm.memory),
                Ok(None) => {
                    return error_result(&format!("Memory not found: {id}"));
                }
                Err(e) => {
                    return error_result(&format!("Failed to fetch memory {id}: {e}"));
                }
            }
        }

        let pairs = detector.detect_interference(&memories);
        let items: Vec<serde_json::Value> = pairs
            .iter()
            .map(|p| {
                json!({
                    "memory_a": p.memory_a.to_string(),
                    "memory_b": p.memory_b.to_string(),
                    "similarity": p.similarity,
                    "disambiguation": p.disambiguation,
                })
            })
            .collect();

        tracing::info!(pairs = items.len(), "interference detection complete");
        Ok(CallToolResult::success(vec![Content::text(
            json!({ "interference_pairs": items, "count": items.len() }).to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Register an entity for phantom memory detection. Phantom memories represent knowledge gaps the agent should fill."
    )]
    async fn register_entity(
        &self,
        Parameters(req): Parameters<RegisterEntityRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut tracker = self.phantom_tracker.lock().await;
        tracing::info!(name = %req.name, entity_type = %req.entity_type, "registering entity");
        tracker.register_entity(&req.name);
        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "registered",
                "name": req.name,
                "entity_type": req.entity_type,
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Get the current cognitive state including pain signals, phantom memories, and trajectory predictions."
    )]
    async fn get_cognitive_state(&self) -> Result<CallToolResult, McpError> {
        let pain = self.pain_registry.lock().await;
        let phantom = self.phantom_tracker.lock().await;
        let trajectory = self.trajectory_tracker.lock().await;
        let cache = self.speculative_cache.lock().await;
        let cache_stats = cache.stats();
        drop(cache);

        let active_pain: Vec<serde_json::Value> = pain
            .all_signals()
            .iter()
            .map(|s| {
                json!({
                    "memory_id": s.memory_id.to_string(),
                    "intensity": s.intensity,
                    "description": s.description,
                })
            })
            .collect();

        let phantoms: Vec<serde_json::Value> = phantom
            .get_active_phantoms()
            .iter()
            .map(|p| {
                json!({
                    "gap": p.gap_description,
                    "priority": format!("{:?}", p.priority),
                })
            })
            .collect();

        let trajectory_info = trajectory.get_resume_context();

        let result = json!({
            "pain_signals": active_pain,
            "phantom_memories": phantoms,
            "trajectory": trajectory_info,
            "speculative_cache": {
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "evictions": cache_stats.evictions,
                "cache_size": cache_stats.cache_size,
                "hit_rate": if cache_stats.hits + cache_stats.misses > 0 {
                    cache_stats.hits as f64 / (cache_stats.hits + cache_stats.misses) as f64
                } else {
                    0.0
                },
            },
        });

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }
}
