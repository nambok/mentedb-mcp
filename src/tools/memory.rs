use super::*;

#[rmcp::tool_router(router = tool_router_memory, vis = "pub")]
impl MenteDbServer {
    #[rmcp::tool(
        description = "Store an important fact, preference, decision, or correction. Use types: semantic (facts), procedural (how-to), correction (fixes), anti_pattern (mistakes to avoid). Add tags for retrieval."
    )]
    async fn store_memory(
        &self,
        Parameters(req): Parameters<StoreMemoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Cap content at 4KB to prevent context window abuse
        const MAX_CONTENT_BYTES: usize = 4096;
        if req.content.len() > MAX_CONTENT_BYTES {
            return error_result(&format!(
                "Content too large ({} bytes, max {}). Summarize before storing.",
                req.content.len(),
                MAX_CONTENT_BYTES
            ));
        }

        let memory_type = match parse_memory_type(&req.memory_type) {
            Ok(mt) => mt,
            Err(e) => return error_result(&e),
        };

        let embedding = self
            .embedding_provider
            .embed(&req.content)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;
        let agent_id = match req.agent_id.as_deref() {
            Some(id_str) => match parse_uuid(id_str) {
                Ok(id) => id,
                Err(e) => return error_result(&e),
            },
            None => Uuid::nil(),
        };
        let mut node = MemoryNode::new(AgentId(agent_id), memory_type, req.content, embedding);

        let mut tags = req.tags.unwrap_or_default();
        // Auto-add scope tags based on content analysis
        if !tags.iter().any(|t| t.starts_with("scope:")) {
            tags.push("scope:global".to_string());
        }
        node.tags = tags;

        if let Some(metadata) = req.metadata {
            for (k, v) in metadata {
                let attr = match v {
                    serde_json::Value::String(s) => AttributeValue::String(s),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            AttributeValue::Integer(i)
                        } else if let Some(f) = n.as_f64() {
                            AttributeValue::Float(f)
                        } else {
                            AttributeValue::String(n.to_string())
                        }
                    }
                    serde_json::Value::Bool(b) => AttributeValue::Boolean(b),
                    other => AttributeValue::String(other.to_string()),
                };
                node.attributes.insert(k, attr);
            }
        }

        let id = node.id;
        let db = &*self.db;
        match db.store(node) {
            Ok(()) => {
                tracing::info!(id = %id, memory_type = %req.memory_type, "memory stored");
                Ok(CallToolResult::success(vec![Content::text(
                    json!({ "id": id.to_string(), "status": "stored" }).to_string(),
                )]))
            }
            Err(e) => {
                tracing::error!(error = %e, "store_memory failed");
                error_result(&format!("Failed to store memory: {e}"))
            }
        }
    }

    #[rmcp::tool(
        description = "Recall a specific memory by its UUID. Returns the memory content, type, metadata, and timestamps."
    )]
    async fn recall_memory(
        &self,
        Parameters(req): Parameters<RecallMemoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let db = &*self.db;
        match find_memory_by_id(db, id) {
            Ok(Some(sm)) => {
                tracing::info!(id = %id, "memory recalled");
                Ok(CallToolResult::success(vec![Content::text(
                    memory_node_to_json(&sm.memory).to_string(),
                )]))
            }
            Ok(None) => {
                tracing::warn!(id = %id, "memory not found");
                error_result(&format!("Memory not found: {id}"))
            }
            Err(e) => {
                tracing::error!(id = %id, error = %e, "recall_memory failed");
                error_result(&format!("Recall failed: {e}"))
            }
        }
    }

    #[rmcp::tool(
        description = "Search memories by semantic similarity, or get full content of a specific memory by UUID. Use when you need to look up what you know about a topic, or to get the full text of a truncated context entry."
    )]
    async fn search_memories(
        &self,
        Parameters(req): Parameters<SearchMemoriesRequest>,
    ) -> Result<CallToolResult, McpError> {
        // If the query looks like a UUID, do a direct ID lookup
        if let Ok(uuid) = Uuid::parse_str(req.query.trim()) {
            let db = &*self.db;
            if let Ok(Some(mem)) = find_memory_by_id(db, uuid) {
                return Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "id": uuid.to_string(),
                        "content": mem.memory.content,
                        "memory_type": format!("{:?}", mem.memory.memory_type),
                        "tags": mem.memory.tags,
                        "salience": mem.memory.salience,
                        "created_at": mem.memory.created_at,
                    })
                    .to_string(),
                )]));
            } else {
                return error_result(&format!("Memory not found: {uuid}"));
            }
        }

        let k = req.limit.unwrap_or(10);

        if self.using_hash_fallback {
            tracing::warn!(
                "Search results may be unreliable: embedding model failed to load, using hash fallback"
            );
        }

        let embedding = self
            .embedding_provider
            .embed(&req.query)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;

        let type_filter = match req.memory_type.as_deref() {
            Some(t) => Some(parse_memory_type(t).map_err(|e| {
                McpError::internal_error(format!("Invalid memory_type filter: {e}"), None)
            })?),
            None => None,
        };

        let db = &*self.db;
        // Fetch extra candidates when filtering by type since some will be excluded
        let fetch_k = if type_filter.is_some() { k * 3 } else { k };
        match db.recall_similar(&embedding, fetch_k) {
            Ok(results) => {
                tracing::info!(query = %req.query, k = k, results = results.len(), "search completed");
                let mut items: Vec<serde_json::Value> = Vec::new();
                for (id, score) in &results {
                    if let Ok(Some(mem)) = find_memory_by_id(db, id.0) {
                        if let Some(ref tf) = type_filter
                            && mem.memory.memory_type != *tf
                        {
                            continue;
                        }
                        let boosted_score = if mem.memory.memory_type == MemoryType::AntiPattern {
                            score * 1.5
                        } else {
                            *score
                        };
                        items.push(json!({
                            "id": id.to_string(),
                            "score": boosted_score,
                            "content": mem.memory.content,
                            "memory_type": format!("{:?}", mem.memory.memory_type),
                            "tags": mem.memory.tags,
                            "salience": mem.memory.salience,
                        }));
                    } else {
                        if type_filter.is_some() {
                            continue;
                        }
                        items.push(json!({ "id": id.to_string(), "score": score }));
                    }
                }
                // Re-sort by score so boosted anti-patterns bubble up, then truncate
                items.sort_by(|a, b| {
                    let sa = a["score"].as_f64().unwrap_or(0.0);
                    let sb = b["score"].as_f64().unwrap_or(0.0);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                items.truncate(k);
                Ok(CallToolResult::success(vec![Content::text(
                    json!({ "results": items, "count": items.len() }).to_string(),
                )]))
            }
            Err(e) => {
                tracing::error!(query = %req.query, error = %e, "search_memories failed");
                error_result(&format!("Search failed: {e}"))
            }
        }
    }

    #[rmcp::tool(
        description = "Get a specific memory by UUID. Returns all fields including content, type, tags, metadata, timestamps, salience, and confidence."
    )]
    async fn get_memory(
        &self,
        Parameters(req): Parameters<GetMemoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let db = &*self.db;
        match find_memory_by_id(db, id) {
            Ok(Some(sm)) => {
                tracing::info!(id = %id, "memory retrieved");
                Ok(CallToolResult::success(vec![Content::text(
                    memory_node_to_json(&sm.memory).to_string(),
                )]))
            }
            Ok(None) => {
                tracing::warn!(id = %id, "memory not found");
                error_result(&format!("Memory not found: {id}"))
            }
            Err(e) => {
                tracing::error!(id = %id, error = %e, "get_memory failed");
                error_result(&format!("Failed to get memory: {e}"))
            }
        }
    }

    #[rmcp::tool(
        description = "Delete a memory by ID. Use when the user asks to forget something."
    )]
    async fn forget_memory(
        &self,
        Parameters(req): Parameters<ForgetMemoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        if let Some(reason) = &req.reason {
            tracing::info!(id = %id, reason = %reason, "forgetting memory");
        }

        let db = &*self.db;
        match db.forget(MemoryId(id)) {
            Ok(()) => {
                tracing::info!(id = %id, "memory forgotten");
                Ok(CallToolResult::success(vec![Content::text(
                    json!({ "status": "forgotten", "id": id.to_string() }).to_string(),
                )]))
            }
            Err(e) => {
                tracing::error!(id = %id, error = %e, "forget_memory failed");
                error_result(&format!("Failed to forget memory: {e}"))
            }
        }
    }

    #[rmcp::tool(
        description = "Delete ALL memories from the database permanently. Requires confirm='CONFIRM' as a safety check. Use when the user explicitly asks to reset, clear, or start fresh."
    )]
    async fn forget_all(
        &self,
        Parameters(req): Parameters<ForgetAllRequest>,
    ) -> Result<CallToolResult, McpError> {
        if req.confirm != "CONFIRM" {
            return error_result(
                "Safety check failed. Set confirm to exactly 'CONFIRM' to delete all memories.",
            );
        }

        let reason = req.reason.as_deref().unwrap_or("user requested reset");
        tracing::warn!(reason = %reason, "forgetting ALL memories");

        let db = &*self.db;
        let all = recall_all_memories(db);
        let total = all.len();
        let mut forgotten = 0u64;
        let mut errors = 0u64;

        for mem in &all {
            match db.forget(mem.memory.id) {
                Ok(()) => forgotten += 1,
                Err(e) => {
                    tracing::error!(id = %mem.memory.id.0, error = %e, "forget_all: failed to delete");
                    errors += 1;
                }
            }
        }

        tracing::info!(
            forgotten = forgotten,
            errors = errors,
            "forget_all complete"
        );
        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "reset_complete",
                "total_found": total,
                "forgotten": forgotten,
                "errors": errors,
                "reason": reason,
            })
            .to_string(),
        )]))
    }
}
