use super::*;

#[rmcp::tool_router(router = tool_router_inference, vis = "pub")]
impl MenteDbServer {
    #[rmcp::tool(
        description = "Check LLM output text against known facts for contradictions, forgotten facts, and reinforcements."
    )]
    async fn check_stream(
        &self,
        Parameters(req): Parameters<CheckStreamRequest>,
    ) -> Result<CallToolResult, McpError> {
        self.db.feed_stream_token(&req.text);

        let facts: Vec<(MemoryId, String)> = req
            .known_facts
            .iter()
            .filter_map(|f| {
                parse_uuid(&f.memory_id)
                    .ok()
                    .map(|id| (MemoryId::from(id), f.summary.clone()))
            })
            .collect();

        let alerts = self.db.check_stream_alerts(&facts);
        let items: Vec<serde_json::Value> = alerts
            .iter()
            .map(|a| match a {
                mentedb_cognitive::StreamAlert::Contradiction {
                    memory_id,
                    ai_said,
                    stored,
                } => json!({
                    "type": "contradiction",
                    "memory_id": memory_id.to_string(),
                    "ai_said": ai_said,
                    "stored_fact": stored,
                }),
                mentedb_cognitive::StreamAlert::Forgotten { memory_id, summary } => json!({
                    "type": "forgotten",
                    "memory_id": memory_id.to_string(),
                    "summary": summary,
                }),
                mentedb_cognitive::StreamAlert::Correction {
                    memory_id,
                    old,
                    new,
                } => json!({
                    "type": "correction",
                    "memory_id": memory_id.to_string(),
                    "old": old,
                    "new": new,
                }),
                mentedb_cognitive::StreamAlert::Reinforcement { memory_id } => json!({
                    "type": "reinforcement",
                    "memory_id": memory_id.to_string(),
                }),
            })
            .collect();

        tracing::info!(alerts = items.len(), "stream check complete");
        Ok(CallToolResult::success(vec![Content::text(
            json!({ "alerts": items, "count": items.len() }).to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Run write-time inference on a memory to detect contradictions, suggest edges, mark obsolescence, and adjust confidence."
    )]
    async fn write_inference(
        &self,
        Parameters(req): Parameters<WriteInferenceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let target_id = match parse_uuid(&req.memory_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let db = &*self.db;
        let target_memory = match find_memory_by_id(db, target_id) {
            Ok(Some(sm)) => sm.memory,
            Ok(None) => return error_result(&format!("Memory not found: {target_id}")),
            Err(e) => return error_result(&format!("Failed to fetch memory: {e}")),
        };

        // Gather nearby memories via HNSW similarity search for comparison
        let similar_ids = db
            .recall_similar(&target_memory.embedding, 20)
            .unwrap_or_default();
        let existing: Vec<MemoryNode> = resolve_memory_ids(db, &similar_ids)
            .into_iter()
            .filter(|sm| sm.memory.id != MemoryId(target_id))
            .map(|sm| sm.memory)
            .collect();

        let engine = mentedb_cognitive::WriteInferenceEngine::new();
        let actions = engine.infer_on_write(&target_memory, &existing, &[]);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let mut applied = 0u32;
        let mut items: Vec<serde_json::Value> = Vec::new();
        for action in &actions {
            match action {
                mentedb_cognitive::InferredAction::FlagContradiction {
                    existing: ex,
                    new,
                    reason,
                } => {
                    let edge = MemoryEdge {
                        source: *new,
                        target: *ex,
                        edge_type: EdgeType::Contradicts,
                        weight: 1.0,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: None,
                    };
                    let _ = db.relate(edge);
                    applied += 1;
                    items.push(json!({
                        "action": "flag_contradiction",
                        "existing_memory": ex.to_string(),
                        "new_memory": new.to_string(),
                        "reason": reason,
                    }));
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
                        label: None,
                    };
                    let _ = db.relate(edge);
                    applied += 1;
                    items.push(json!({
                        "action": "mark_obsolete",
                        "memory": memory.to_string(),
                        "superseded_by": superseded_by.to_string(),
                    }));
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
                        label: None,
                    };
                    let _ = db.relate(edge);
                    applied += 1;
                    items.push(json!({
                        "action": "create_edge",
                        "source": source.to_string(),
                        "target": target.to_string(),
                        "edge_type": format!("{:?}", edge_type),
                        "weight": weight,
                    }));
                }
                mentedb_cognitive::InferredAction::UpdateConfidence {
                    memory,
                    new_confidence,
                } => {
                    if let Some(mut mem) = existing.iter().find(|m| m.id == *memory).cloned() {
                        mem.confidence = *new_confidence;
                        let _ = db.store(mem);
                        applied += 1;
                    }
                    items.push(json!({
                        "action": "update_confidence",
                        "memory": memory.to_string(),
                        "new_confidence": new_confidence,
                    }));
                }
                mentedb_cognitive::InferredAction::PropagateBeliefChange { root, delta } => {
                    items.push(json!({
                        "action": "propagate_belief_change",
                        "root": root.to_string(),
                        "delta": delta,
                    }));
                }
                mentedb_cognitive::InferredAction::InvalidateMemory {
                    memory,
                    superseded_by,
                    valid_until,
                } => {
                    if let Some(mut mem) = existing.iter().find(|m| m.id == *memory).cloned() {
                        mem.valid_until = Some(*valid_until);
                        let _ = db.store(mem);
                        applied += 1;
                    }
                    items.push(json!({
                        "action": "invalidate_memory",
                        "memory": memory.to_string(),
                        "superseded_by": superseded_by.to_string(),
                        "valid_until": valid_until,
                    }));
                }
                mentedb_cognitive::InferredAction::UpdateContent {
                    memory,
                    new_content,
                    reason,
                } => {
                    if let Some(mut mem) = existing.iter().find(|m| m.id == *memory).cloned() {
                        mem.content = new_content.clone();
                        let _ = db.store(mem);
                        applied += 1;
                    }
                    items.push(json!({
                        "action": "update_content",
                        "memory": memory.to_string(),
                        "reason": reason,
                    }));
                }
            }
        }

        tracing::info!(memory_id = %target_id, actions = items.len(), applied, "write inference complete");
        Ok(CallToolResult::success(vec![Content::text(
            json!({ "inferred_actions": items, "count": items.len(), "applied": applied })
                .to_string(),
        )]))
    }
}
