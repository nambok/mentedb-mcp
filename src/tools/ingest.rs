use super::*;

#[rmcp::tool_router(router = tool_router_ingest, vis = "pub")]
impl MenteDbServer {
    #[rmcp::tool(
        description = "Ingest a raw conversation, extract structured memories via LLM, run cognitive checks, and store the results. Returns extraction statistics and stored memory IDs."
    )]
    async fn ingest_conversation(
        &self,
        Parameters(req): Parameters<IngestConversationRequest>,
    ) -> Result<CallToolResult, McpError> {
        let provider_name = req.provider.as_deref().unwrap_or(&self.config.llm_provider);
        let api_key = req
            .api_key
            .or_else(|| self.config.llm_api_key.clone())
            .or_else(|| std::env::var("MENTEDB_LLM_API_KEY").ok())
            .or_else(|| std::env::var("OPENAI_API_KEY").ok());

        let agent_id = match req.agent_id.as_deref() {
            Some(id_str) => match parse_uuid(id_str) {
                Ok(id) => id,
                Err(e) => return error_result(&e),
            },
            None => Uuid::nil(),
        };

        let config = match provider_name.to_lowercase().as_str() {
            "openai" => {
                let key = match api_key {
                    Some(k) => k,
                    None => {
                        return error_result(
                            "API key required for OpenAI. Set OPENAI_API_KEY env var or pass api_key.",
                        );
                    }
                };
                ExtractionConfig::openai(key)
            }
            "anthropic" => {
                let key = match api_key {
                    Some(k) => k,
                    None => {
                        return error_result(
                            "API key required for Anthropic. Set MENTEDB_LLM_API_KEY env var or pass api_key.",
                        );
                    }
                };
                ExtractionConfig::anthropic(key)
            }
            "ollama" => ExtractionConfig::ollama(),
            "mock" => ExtractionConfig::default(),
            other => {
                return error_result(&format!(
                    "Unknown provider: {other}. Use openai, anthropic, ollama, or mock."
                ));
            }
        };

        // Gather existing memories for dedup/contradiction checks via HNSW
        let existing_memories = {
            let db = &*self.db;
            let conv_embedding = self
                .embedding_provider
                .embed(&req.conversation)
                .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;
            let similar = db.recall_similar(&conv_embedding, 20).unwrap_or_default();
            resolve_memory_ids(db, &similar)
                .into_iter()
                .map(|sm| sm.memory)
                .collect::<Vec<MemoryNode>>()
        };

        let result = if provider_name == "mock" {
            let mock_provider = MockExtractionProvider::with_realistic_response();
            let pipeline = ExtractionPipeline::new(mock_provider, config);
            pipeline
                .process(
                    &req.conversation,
                    &existing_memories,
                    self.embedding_provider.as_ref(),
                )
                .await
                .map_err(|e| McpError::internal_error(friendly_extraction_error(&e), None))?
        } else {
            let http_provider = mentedb_extraction::HttpExtractionProvider::new(config)
                .map_err(|e| McpError::internal_error(friendly_extraction_error(&e), None))?;
            let pipeline = ExtractionPipeline::new(http_provider, ExtractionConfig::default());
            pipeline
                .process(
                    &req.conversation,
                    &existing_memories,
                    self.embedding_provider.as_ref(),
                )
                .await
                .map_err(|e| McpError::internal_error(friendly_extraction_error(&e), None))?
        };

        let db = &*self.db;
        let stored_ids =
            store_extraction_results(&result, db, self.embedding_provider.as_ref(), agent_id)?;

        let stats = &result.stats;
        tracing::info!(
            stored = stored_ids.len(),
            rejected_quality = stats.rejected_quality,
            rejected_duplicate = stats.rejected_duplicate,
            contradictions = stats.contradictions_found,
            "conversation ingestion complete"
        );

        let response = json!({
            "status": "complete",
            "stats": {
                "total_extracted": stats.total_extracted,
                "accepted": stats.accepted,
                "rejected_quality": stats.rejected_quality,
                "rejected_duplicate": stats.rejected_duplicate,
                "contradictions_found": stats.contradictions_found,
            },
            "stored_ids": stored_ids,
        });

        Ok(CallToolResult::success(vec![Content::text(
            response.to_string(),
        )]))
    }
}
