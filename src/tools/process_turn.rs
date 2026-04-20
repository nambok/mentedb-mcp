use super::*;

impl MenteDbServer {
    /// Build the final JSON response from all computed data.
    fn assemble_response(
        context_items: &[serde_json::Value],
        stored_ids: &[String],
        contradiction_count: usize,
        pain_warnings: &[serde_json::Value],
        proactive_recalls: &[serde_json::Value],
        detected_actions: &[serde_json::Value],
    ) -> String {
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

        let mut response = json!({
            "ok": true,
            "context": context_summaries,
            "stored": stored_ids.len(),
            "contradictions": contradiction_count,
        });

        if !pain_warnings.is_empty() {
            response["pain_warnings"] = json!(pain_warnings);
        }
        if !proactive_recalls.is_empty() {
            response["proactive_recalls"] = json!(proactive_recalls);
        }
        if !detected_actions.is_empty() {
            response["detected_actions"] = json!(detected_actions);
        }

        response.to_string()
    }
}

// ── MCP tool entry point ──

#[rmcp::tool_router(router = tool_router_process_turn, vis = "pub")]
impl MenteDbServer {
    #[rmcp::tool(
        description = "Process a conversation turn. Stores new memories and returns relevant context from past conversations. MUST be called every turn."
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

        // §1: Embedding + context retrieval
        let query_embedding = self
            .embedding_provider
            .embed(&req.user_message)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;
        let (context_items, _current_ids, cache_hit) = self
            .retrieve_context(&req.user_message, &query_embedding, &req)
            .await;

        // §2: Pain signals
        let pain_warnings = self.check_pain_signals(&req.user_message).await;

        // §3: Store episodic turn
        let (mut stored_ids, memory_id, conversation) =
            self.store_episodic_turn(&req, agent_id).await;

        // §4: Entity resolution
        let entities_resolved = self.resolve_entities_llm(&conversation).await;

        // §5: Write-time inference + LLM contradiction verification
        let inference_applied = self.run_write_inference(memory_id, &stored_ids).await;

        // §4b: Fact extraction
        self.extract_and_store_facts(memory_id, &stored_ids).await;

        // Action detection + proactive recall
        let assistant_resp = req.assistant_response.as_deref().unwrap_or("");
        let combined_text = format!("{} {}", req.user_message, assistant_resp).to_lowercase();
        let detected_actions = Self::detect_actions(&combined_text);
        let proactive_recalls = self.proactive_recall(&detected_actions).await;

        // Auto-detect corrections
        if let Some(ap_id) = self
            .auto_detect_corrections(
                &req.user_message,
                assistant_resp,
                agent_id,
                req.project_context.as_deref(),
            )
            .await
        {
            stored_ids.push(ap_id);
            tracing::info!(
                turn_id = req.turn_id,
                "auto-detected correction, stored anti-pattern"
            );
        }

        // Sentiment
        let sentiment_score = Self::analyze_sentiment(&req.user_message);

        // §4c + §4d: Phantoms + stream contradiction check
        let (phantom_count, contradiction_count) = self
            .detect_phantoms_and_check_stream(
                &conversation,
                assistant_resp,
                &context_items,
                req.turn_id,
            )
            .await;

        // §5: Trajectory + ghost memories
        let predictions = self
            .update_trajectory(
                &req,
                agent_id,
                &stored_ids,
                &detected_actions,
                &combined_text,
            )
            .await;

        // §6: Speculative cache
        self.update_speculative_cache(&predictions).await;

        // Auto-maintenance
        self.maybe_run_maintenance(req.turn_id).await;

        let elapsed_ms = start.elapsed().as_millis();
        tracing::info!(
            turn_id = req.turn_id,
            context_count = context_items.len(),
            cache_hit,
            memories_stored = stored_ids.len(),
            inference_applied,
            phantom_count,
            contradiction_count,
            pain_warnings = pain_warnings.len(),
            elapsed_ms = elapsed_ms,
            "process_turn complete"
        );

        let response = Self::assemble_response(
            &context_items,
            &stored_ids,
            contradiction_count,
            &pain_warnings,
            &proactive_recalls,
            &detected_actions,
        );

        tracing::info!(
            turn_id = req.turn_id,
            response_bytes = response.len(),
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
