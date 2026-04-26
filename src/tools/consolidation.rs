use super::*;

#[rmcp::tool_router(router = tool_router_consolidation, vis = "pub")]
impl MenteDbServer {
    #[rmcp::tool(
        description = "Find clusters of similar memories and merge them into consolidated semantic memories. Returns consolidation candidates and merged results."
    )]
    async fn consolidate_memories(
        &self,
        Parameters(req): Parameters<ConsolidateMemoriesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let min_cluster_size = req.min_cluster_size.unwrap_or(2);
        let similarity_threshold = req.similarity_threshold.unwrap_or(0.85);

        let candidates = self
            .db
            .find_consolidation_candidates(min_cluster_size, similarity_threshold)
            .map_err(|e| McpError::internal_error(format!("Consolidation failed: {e}"), None))?;

        if candidates.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                json!({ "status": "no_clusters", "clusters": [], "consolidated": [] }).to_string(),
            )]));
        }

        let mut consolidated_results: Vec<serde_json::Value> = Vec::new();
        for candidate in &candidates {
            match self.db.consolidate_cluster(&candidate.memories) {
                Ok(merged_id) => {
                    consolidated_results.push(json!({
                        "topic": candidate.topic,
                        "avg_similarity": candidate.avg_similarity,
                        "source_memory_ids": candidate.memories.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                        "merged_memory_id": merged_id.to_string(),
                    }));
                }
                Err(e) => {
                    tracing::warn!(topic = %candidate.topic, error = %e, "failed to consolidate cluster");
                }
            }
        }

        tracing::info!(
            clusters = candidates.len(),
            threshold = similarity_threshold,
            "memory consolidation complete"
        );

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "complete",
                "clusters_found": candidates.len(),
                "consolidated": consolidated_results,
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Apply salience decay to all memories based on time and access patterns. Returns count of memories processed and those below archival threshold."
    )]
    async fn apply_decay(&self) -> Result<CallToolResult, McpError> {
        let updated = self
            .db
            .apply_decay_global()
            .map_err(|e| McpError::internal_error(format!("Decay failed: {e}"), None))?;

        tracing::info!(updated, "decay applied globally");

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "complete",
                "memories_updated": updated,
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Compress a memory by extracting key sentences and removing filler. Returns original length, compressed content, and compression ratio."
    )]
    async fn compress_memory(
        &self,
        Parameters(req): Parameters<CompressMemoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let db = &*self.db;
        match find_memory_by_id(db, id) {
            Ok(Some(sm)) => {
                let compressed = self.db.compress_memory(&sm.memory);
                let original_length = sm.memory.content.len();

                // Persist compressed content
                let mut updated = sm.memory.clone();
                updated.content = compressed.compressed_content.clone();
                let _ = db.store(updated);

                tracing::info!(
                    id = %id,
                    original_len = original_length,
                    ratio = compressed.compression_ratio,
                    "memory compressed"
                );

                Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "id": id.to_string(),
                        "original_length": original_length,
                        "compressed_content": compressed.compressed_content,
                        "compression_ratio": compressed.compression_ratio,
                        "key_facts": compressed.key_facts,
                    })
                    .to_string(),
                )]))
            }
            Ok(None) => error_result(&format!("Memory not found: {id}")),
            Err(e) => error_result(&format!("Failed to get memory: {e}")),
        }
    }

    #[rmcp::tool(
        description = "Evaluate all memories for archival, deletion, or consolidation decisions based on salience and age thresholds."
    )]
    async fn evaluate_archival(&self) -> Result<CallToolResult, McpError> {
        let decisions = self
            .db
            .evaluate_archival_global()
            .map_err(|e| McpError::internal_error(format!("Archival eval failed: {e}"), None))?;

        let mut keep = Vec::new();
        let mut archive = Vec::new();
        let mut delete = Vec::new();
        let mut consolidate = Vec::new();

        for (id, decision) in &decisions {
            let id_str = id.to_string();
            match decision {
                ArchivalDecision::Keep => keep.push(id_str),
                ArchivalDecision::Archive | ArchivalDecision::Delete => {
                    let _ = self.db.forget(*id);
                    if matches!(decision, ArchivalDecision::Delete) {
                        delete.push(id_str);
                    } else {
                        archive.push(id_str);
                    }
                }
                ArchivalDecision::Consolidate(ids) => {
                    consolidate.push(json!({
                        "id": id_str,
                        "merge_with": ids.iter().map(|i| i.to_string()).collect::<Vec<_>>(),
                    }));
                }
            }
        }

        tracing::info!(
            total = decisions.len(),
            keep = keep.len(),
            archive = archive.len(),
            delete = delete.len(),
            consolidate = consolidate.len(),
            "archival evaluation complete"
        );

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "complete",
                "total_evaluated": decisions.len(),
                "keep": { "count": keep.len(), "ids": keep },
                "archive": { "count": archive.len(), "ids": archive },
                "delete": { "count": delete.len(), "ids": delete },
                "consolidate": { "count": consolidate.len(), "items": consolidate },
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Extract structured subject-predicate-object facts from a memory using rule-based pattern matching."
    )]
    async fn extract_facts(
        &self,
        Parameters(req): Parameters<ExtractFactsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let db = &*self.db;
        match find_memory_by_id(db, id) {
            Ok(Some(sm)) => {
                let extractor = FactExtractor::new();
                let facts = extractor.extract_facts(&sm.memory);

                // Store extracted facts as Related edges to matching memories
                let all_memories = recall_all_memories(db);
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                let mut edges_created = 0u32;
                for fact in &facts {
                    for other in &all_memories {
                        if other.memory.id != sm.memory.id
                            && (other.memory.content.contains(&fact.subject)
                                || other.memory.content.contains(&fact.object))
                        {
                            let edge = MemoryEdge {
                                source: sm.memory.id,
                                target: other.memory.id,
                                edge_type: EdgeType::Related,
                                weight: 0.5,
                                created_at: now,
                                valid_from: None,
                                valid_until: None,
                                label: None,
                            };
                            let _ = db.relate(edge);
                            edges_created += 1;
                        }
                    }
                }

                let facts_json: Vec<serde_json::Value> = facts
                    .iter()
                    .map(|f| {
                        json!({
                            "subject": f.subject,
                            "predicate": f.predicate,
                            "object": f.object,
                            "confidence": f.confidence,
                            "source_memory": f.source_memory.to_string(),
                        })
                    })
                    .collect();

                tracing::info!(id = %id, facts_count = facts.len(), edges_created, "facts extracted and linked");

                Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "id": id.to_string(),
                        "facts_count": facts.len(),
                        "edges_created": edges_created,
                        "facts": facts_json,
                    })
                    .to_string(),
                )]))
            }
            Ok(None) => error_result(&format!("Memory not found: {id}")),
            Err(e) => error_result(&format!("Failed to get memory: {e}")),
        }
    }

    #[rmcp::tool(
        description = "GDPR-compliant forget: plan and report what would be deleted for a given subject (agent). Returns audit log, count of affected memories and edges."
    )]
    async fn gdpr_forget(
        &self,
        Parameters(req): Parameters<GdprForgetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let agent_id = match parse_uuid(&req.subject) {
            Ok(id) => id,
            Err(e) => return error_result(&format!("Invalid subject UUID: {e}")),
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let db = &*self.db;
        let all = recall_all_memories(db);
        let memories: Vec<MemoryNode> = all.into_iter().map(|sm| sm.memory).collect();

        let forget_request = ForgetRequest {
            agent_id: Some(AgentId(agent_id)),
            space_id: None,
            memory_ids: Vec::new(),
            reason: req.reason.clone(),
            requested_at: now,
        };

        let engine = ForgetEngine::new();
        let result = engine.plan_forget(&forget_request, &memories, &[]);

        // Execute the actual deletion for matching memories
        let mut deleted_count = 0u64;
        for m in &memories {
            if m.agent_id == AgentId(agent_id) {
                if let Err(e) = db.forget(m.id) {
                    tracing::error!(id = %m.id, error = %e, "failed to forget memory during GDPR delete");
                } else {
                    deleted_count += 1;
                }
            }
        }

        tracing::info!(
            subject = %agent_id,
            reason = %req.reason,
            deleted = deleted_count,
            "GDPR forget executed"
        );

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "complete",
                "subject": agent_id.to_string(),
                "reason": req.reason,
                "planned_deletions": result.deleted_memories,
                "actually_deleted": deleted_count,
                "affected_edges": result.deleted_edges,
                "affected_facts": result.deleted_facts,
                "audit_log": result.audit_log_entry,
            })
            .to_string(),
        )]))
    }
}
