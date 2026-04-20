use super::*;

#[rmcp::tool_handler]
impl ServerHandler for MenteDbServer {
    fn get_info(&self) -> ServerInfo {
        let mut instructions = String::from(
            "MenteDB gives you persistent memory across sessions. You have 4 tools:\n\
             \n\
             1. process_turn — Call on EVERY turn. Pass user_message + assistant_response (can be empty). Returns past context, stores the turn, detects contradictions.\n\
             2. store_memory — Save important facts (preferences, decisions, corrections). Add type + tags. Use scope: 'always' for critical rules that must be surfaced every turn (e.g. 'never do X').\n\
             3. search_memories — Look up what you know. Pass a query OR a memory UUID for full content. Use proactively when the user mentions a project or topic.\n\
             4. forget_memory — Delete a memory when the user asks to forget.\n\
             \n\
             SCOPE: Set scope: 'always' for hard rules/constraints the user wants enforced every turn. Default 'contextual' is retrieved by similarity.\n\
             TYPES: semantic (facts/preferences), procedural (how-to), correction (was wrong now right), anti_pattern (never do X), episodic (what happened), reasoning (why a decision was made).\n\
             QUALITY: One fact per memory. Self-contained. Include project context. Under 200 words. Don't store chitchat, temp info, or large code blocks.\n\
             USE THE CONTEXT: process_turn returns summaries with IDs. Reference them. Call search_memories(id) for full text.\n\
             If pain_warnings are returned, WARN the user. If contradictions > 0, flag it.",
        );

        if self.using_hash_fallback {
            instructions.push_str(
                "\n\nWARNING: Embedding model failed to load. Using hash-based fallback. \
                 Search results and context retrieval may be unreliable.",
            );
        }

        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_server_info(Implementation::new(
            "mentedb-mcp",
            env!("CARGO_PKG_VERSION"),
        ))
        .with_instructions(&instructions)
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _cx: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult {
            meta: None,
            next_cursor: None,
            resources: vec![
                RawResource::new("mentedb://stats", "stats".to_string()).no_annotation(),
                RawResource::new("mentedb://memories", "memories".to_string()).no_annotation(),
            ],
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _cx: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let uri = &request.uri;
        let uri_str = uri.as_str();

        if uri_str == "mentedb://stats" {
            let db = self.db.lock().await;
            let memory_count = db.memory_count();
            let stats = json!({
                "status": "operational",
                "engine": "mentedb",
                "version": env!("CARGO_PKG_VERSION"),
                "memory_count": memory_count,
            });
            return Ok(ReadResourceResult::new(vec![ResourceContents::text(
                stats.to_string(),
                uri.clone(),
            )]));
        }

        if uri_str == "mentedb://memories" {
            let tools: Vec<serde_json::Value> = vec![
                json!({ "name": "store_memory", "description": "Store a new memory with content, type, tags, metadata" }),
                json!({ "name": "get_memory", "description": "Retrieve a memory by UUID with full details" }),
                json!({ "name": "recall_memory", "description": "Recall a specific memory by UUID" }),
                json!({ "name": "search_memories", "description": "Semantic similarity search with type filtering" }),
                json!({ "name": "relate_memories", "description": "Create typed edges between memories" }),
                json!({ "name": "forget_memory", "description": "Delete a memory" }),
                json!({ "name": "forget_all", "description": "Delete ALL memories (requires confirm='CONFIRM')" }),
                json!({ "name": "ingest_conversation", "description": "Extract memories from raw conversation via LLM" }),
                json!({ "name": "assemble_context", "description": "Build optimized context window with token budget" }),
                json!({ "name": "get_related", "description": "Traverse relationships from a memory" }),
                json!({ "name": "find_path", "description": "Shortest path between memories" }),
                json!({ "name": "get_subgraph", "description": "Extract local neighborhood subgraph" }),
                json!({ "name": "find_contradictions", "description": "Find contradicting memories via graph edges" }),
                json!({ "name": "propagate_belief", "description": "Propagate confidence changes through graph" }),
                json!({ "name": "consolidate_memories", "description": "Cluster and merge similar memories" }),
                json!({ "name": "apply_decay", "description": "Time-based salience decay" }),
                json!({ "name": "compress_memory", "description": "Extract key sentences from a memory" }),
                json!({ "name": "evaluate_archival", "description": "Categorize memories for keep/archive/delete" }),
                json!({ "name": "extract_facts", "description": "Subject-predicate-object extraction" }),
                json!({ "name": "gdpr_forget", "description": "GDPR-compliant deletion with audit" }),
                json!({ "name": "record_pain", "description": "Record negative experiences for avoidance" }),
                json!({ "name": "detect_phantoms", "description": "Find knowledge gaps in content" }),
                json!({ "name": "resolve_phantom", "description": "Mark a knowledge gap as resolved" }),
                json!({ "name": "record_trajectory", "description": "Track conversation turns for prediction" }),
                json!({ "name": "predict_topics", "description": "Predict next topics from trajectory" }),
                json!({ "name": "detect_interference", "description": "Find confusable memory pairs" }),
                json!({ "name": "check_stream", "description": "Monitor LLM output for contradictions" }),
                json!({ "name": "write_inference", "description": "Write-time contradiction and edge detection" }),
                json!({ "name": "register_entity", "description": "Register entity for phantom detection" }),
                json!({ "name": "get_cognitive_state", "description": "Full cognitive state snapshot" }),
                json!({ "name": "get_stats", "description": "Database statistics" }),
                json!({ "name": "process_turn", "description": "Process a conversation turn. Stores new memories and returns relevant context from past conversations. MUST be called every turn." }),
            ];
            let info = json!({
                "description": "MenteDB memory tools",
                "tool_count": tools.len(),
                "tools": tools,
            });
            return Ok(ReadResourceResult::new(vec![ResourceContents::text(
                info.to_string(),
                uri.clone(),
            )]));
        }

        if uri_str == "mentedb://cognitive/state" {
            let pain = self.pain_registry.lock().await;
            let phantom = self.phantom_tracker.lock().await;
            let trajectory = self.trajectory_tracker.lock().await;

            let active_pain: Vec<serde_json::Value> = pain
                .get_pain_for_context(&[])
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
            });

            return Ok(ReadResourceResult::new(vec![ResourceContents::text(
                result.to_string(),
                uri.clone(),
            )]));
        }

        if let Some(id_str) = uri_str.strip_prefix("mentedb://memories/") {
            let id = Uuid::parse_str(id_str).map_err(|e| {
                McpError::resource_not_found(
                    "invalid_uuid",
                    Some(json!({ "error": e.to_string() })),
                )
            })?;

            let mut db = self.db.lock().await;
            match db.get_memory(MemoryId(id)) {
                Ok(mem) => {
                    let result = memory_node_to_json(&mem);
                    return Ok(ReadResourceResult::new(vec![ResourceContents::text(
                        result.to_string(),
                        uri.clone(),
                    )]));
                }
                Err(e) => {
                    return Err(McpError::resource_not_found(
                        "memory_not_found",
                        Some(json!({ "error": format!("Memory not found: {e}") })),
                    ));
                }
            }
        }

        Err(McpError::resource_not_found(
            "resource_not_found",
            Some(json!({ "uri": uri_str })),
        ))
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _cx: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult {
            meta: None,
            next_cursor: None,
            resource_templates: vec![
                RawResourceTemplate {
                    uri_template: "mentedb://memories/{id}".to_string(),
                    name: "memory".to_string(),
                    title: None,
                    description: Some("Access a specific memory by UUID".to_string()),
                    mime_type: Some("application/json".to_string()),
                    icons: None,
                }
                .no_annotation(),
                RawResourceTemplate {
                    uri_template: "mentedb://cognitive/state".to_string(),
                    name: "cognitive_state".to_string(),
                    title: None,
                    description: Some(
                        "Cognitive state: pain signals, phantom memories, trajectory predictions"
                            .to_string(),
                    ),
                    mime_type: Some("application/json".to_string()),
                    icons: None,
                }
                .no_annotation(),
            ],
        })
    }
}
