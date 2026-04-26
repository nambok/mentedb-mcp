use super::*;
use mentedb::process_turn::{ProcessTurnInput, ProcessTurnResult};

impl MenteDbServer {
    /// Build the final JSON response from a ProcessTurnResult.
    fn assemble_response(result: &ProcessTurnResult) -> String {
        const CTX_MAX_CHARS: usize = 300;
        let context_summaries: Vec<serde_json::Value> = result
            .context
            .iter()
            .take(10)
            .map(|sm| {
                let content = &sm.memory.content;
                let truncated = if content.len() > CTX_MAX_CHARS {
                    format!(
                        "{}…",
                        &content[..content.floor_char_boundary(CTX_MAX_CHARS)]
                    )
                } else {
                    content.clone()
                };
                let scope = if sm.memory.tags.iter().any(|t| t == "scope:always") {
                    "always"
                } else {
                    "contextual"
                };
                json!({
                    "id": sm.memory.id.to_string(),
                    "content": truncated,
                    "memory_type": format!("{:?}", sm.memory.memory_type),
                    "scope": scope
                })
            })
            .collect();

        let mut response = json!({
            "ok": true,
            "context": context_summaries,
            "stored": result.stored_ids.len(),
            "contradictions": result.contradiction_count,
        });

        if !result.pain_warnings.is_empty() {
            response["pain_warnings"] = json!(result
                .pain_warnings
                .iter()
                .map(|pw| json!({
                    "signal_id": pw.signal_id.to_string(),
                    "intensity": pw.intensity,
                    "description": &pw.description,
                }))
                .collect::<Vec<_>>());
        }
        if !result.proactive_recalls.is_empty() {
            response["proactive_recalls"] = json!(result
                .proactive_recalls
                .iter()
                .map(|pr| json!({
                    "memory_id": pr.memory_id.to_string(),
                    "content": &pr.content,
                    "relevance": pr.relevance,
                    "action_type": &pr.action_type,
                }))
                .collect::<Vec<_>>());
        }
        if !result.detected_actions.is_empty() {
            response["detected_actions"] = json!(result
                .detected_actions
                .iter()
                .map(|da| json!({
                    "action_type": &da.action_type,
                    "detail": &da.detail,
                }))
                .collect::<Vec<_>>());
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

        // Build engine input
        let input = ProcessTurnInput {
            user_message: req.user_message.clone(),
            assistant_response: req.assistant_response.clone(),
            turn_id: req.turn_id,
            project_context: req.project_context.clone(),
            agent_id: Some(agent_id),
        };

        // Core pipeline: single call handles embedding, context retrieval,
        // pain signals, episodic storage, write inference, fact extraction,
        // action detection, proactive recall, corrections, sentiment,
        // phantoms, trajectory, speculative cache, and maintenance.
        let mut delta_tracker = self.delta_tracker.lock().await;
        let result = self
            .db
            .process_turn(&input, &mut delta_tracker)
            .map_err(|e| McpError::internal_error(format!("process_turn failed: {e}"), None))?;
        drop(delta_tracker);

        // LLM enrichment: entity resolution (if LLM configured)
        let conversation = format!(
            "User: {}\nAssistant: {}",
            req.user_message,
            req.assistant_response.as_deref().unwrap_or("")
        );
        let entities_resolved = self.resolve_entities_llm(&conversation).await;

        // LLM enrichment: contradiction verification on flagged contradictions
        let llm_contradictions = self.verify_contradictions_llm(&result).await;

        let elapsed_ms = start.elapsed().as_millis();
        tracing::info!(
            turn_id = req.turn_id,
            context_count = result.context.len(),
            cache_hit = result.cache_hit,
            memories_stored = result.stored_ids.len(),
            inference_actions = result.inference_actions,
            phantom_count = result.phantom_count,
            contradiction_count = result.contradiction_count,
            pain_warnings = result.pain_warnings.len(),
            entities_resolved,
            llm_contradictions,
            sentiment = result.sentiment,
            predictions_count = result.predicted_topics.len(),
            elapsed_ms = elapsed_ms,
            "process_turn complete (engine)"
        );

        let response = Self::assemble_response(&result);
        Ok(CallToolResult::success(vec![Content::text(response)]))
    }
}
