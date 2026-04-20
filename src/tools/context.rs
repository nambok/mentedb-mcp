use super::*;

#[rmcp::tool_router(router = tool_router_context, vis = "pub")]
impl MenteDbServer {
    #[rmcp::tool(
        description = "Assemble an optimized context window from memories for a given query and token budget."
    )]
    async fn assemble_context(
        &self,
        Parameters(req): Parameters<AssembleContextRequest>,
    ) -> Result<CallToolResult, McpError> {
        let format = match req.format.as_deref().unwrap_or("structured") {
            "compact" => OutputFormat::Compact,
            "delta" => OutputFormat::Delta,
            _ => OutputFormat::Structured,
        };

        let embedding = self
            .embedding_provider
            .embed(&req.query)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;

        let db = &*self.db;
        match db.recall_similar(&embedding, 50) {
            Ok(results) => {
                let scored_memories: Vec<ScoredMemory> = results
                    .iter()
                    .filter_map(|(id, score)| {
                        find_memory_by_id(db, id.0)
                            .ok()
                            .flatten()
                            .map(|sm| ScoredMemory {
                                memory: sm.memory,
                                score: *score,
                            })
                    })
                    .collect();

                let config = AssemblyConfig {
                    token_budget: req.token_budget,
                    format,
                    include_edges: false,
                    include_metadata: true,
                };

                let window = ContextAssembler::assemble(scored_memories, vec![], &config);

                let blocks_json: Vec<serde_json::Value> = window
                    .blocks
                    .iter()
                    .map(|b| {
                        json!({
                            "zone": format!("{:?}", b.zone),
                            "memory_count": b.memories.len(),
                            "estimated_tokens": b.estimated_tokens,
                            "memories": b.memories.iter().map(|sm| json!({
                                "id": sm.memory.id.to_string(),
                                "content": sm.memory.content,
                                "memory_type": format!("{:?}", sm.memory.memory_type),
                                "score": sm.score,
                            })).collect::<Vec<_>>(),
                        })
                    })
                    .collect();

                let result = json!({
                    "query": req.query,
                    "token_budget": req.token_budget,
                    "total_tokens": window.total_tokens,
                    "format": window.format,
                    "blocks": blocks_json,
                    "metadata": {
                        "total_candidates": window.metadata.total_candidates,
                        "included_count": window.metadata.included_count,
                        "excluded_count": window.metadata.excluded_count,
                        "zones_used": window.metadata.zones_used,
                    },
                });
                Ok(CallToolResult::success(vec![Content::text(
                    result.to_string(),
                )]))
            }
            Err(e) => error_result(&format!("Context assembly failed: {e}")),
        }
    }

    #[rmcp::tool(
        description = "Get database statistics including memory count, edge count, and type breakdown."
    )]
    async fn get_stats(&self) -> Result<CallToolResult, McpError> {
        let db = &*self.db;
        let memory_count = db.memory_count();
        let result = json!({
            "status": "operational",
            "engine": "mentedb",
            "version": env!("CARGO_PKG_VERSION"),
            "memory_count": memory_count,
        });
        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }
}
