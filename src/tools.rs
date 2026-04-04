use std::sync::Arc;

use mentedb::MenteDb;
use mentedb_cognitive::{PainRegistry, PhantomConfig, PhantomTracker, TrajectoryTracker};
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::{AttributeValue, MemoryType};
use mentedb_core::{MemoryEdge, MemoryNode};
use mentedb_embedding::HashEmbeddingProvider;
use mentedb_embedding::provider::EmbeddingProvider;
use rmcp::ErrorData as McpError;
use rmcp::ServerHandler;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config::ServerConfig;

// -- Request types --

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StoreMemoryRequest {
    #[schemars(description = "The text content of the memory to store")]
    pub content: String,
    #[schemars(
        description = "Memory type: episodic, semantic, procedural, anti_pattern, reasoning, or correction"
    )]
    pub memory_type: String,
    #[schemars(description = "Optional tags for categorization")]
    pub tags: Option<Vec<String>>,
    #[schemars(description = "Optional key-value metadata")]
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallMemoryRequest {
    #[schemars(description = "The UUID of the memory to recall")]
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchMemoriesRequest {
    #[schemars(description = "The search query text")]
    pub query: String,
    #[schemars(description = "Maximum number of results to return (default: 10)")]
    pub limit: Option<usize>,
    #[schemars(
        description = "Optional memory type filter: episodic, semantic, procedural, anti_pattern, reasoning, correction"
    )]
    #[allow(dead_code)]
    pub memory_type: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RelateMemoriesRequest {
    #[schemars(description = "UUID of the source memory")]
    pub from_id: String,
    #[schemars(description = "UUID of the target memory")]
    pub to_id: String,
    #[schemars(
        description = "Relationship type: caused, before, related, contradicts, supports, supersedes, derived, part_of"
    )]
    pub edge_type: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ForgetMemoryRequest {
    #[schemars(description = "UUID of the memory to delete")]
    pub id: String,
    #[schemars(description = "Optional reason for deletion")]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AssembleContextRequest {
    #[schemars(description = "The query to assemble context for")]
    pub query: String,
    #[schemars(description = "Maximum token budget for the assembled context")]
    pub token_budget: usize,
    #[schemars(description = "Output format: structured, compact, or delta")]
    pub format: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegisterEntityRequest {
    #[schemars(description = "Name of the entity to register")]
    pub name: String,
    #[schemars(description = "Type classification of the entity (e.g. person, tool, concept)")]
    pub entity_type: String,
}

/// MenteDB MCP server state holding the database and cognitive subsystems.
pub struct MenteDbServer {
    db: Arc<Mutex<MenteDb>>,
    embedding_provider: Arc<HashEmbeddingProvider>,
    pain_registry: Arc<Mutex<PainRegistry>>,
    phantom_tracker: Arc<Mutex<PhantomTracker>>,
    trajectory_tracker: Arc<Mutex<TrajectoryTracker>>,
    #[allow(dead_code)]
    config: ServerConfig,
    pub tool_router: ToolRouter<Self>,
}

impl MenteDbServer {
    pub fn new(db: MenteDb, config: ServerConfig) -> Self {
        let embedding_provider = Arc::new(HashEmbeddingProvider::new(config.embedding_dim));
        Self {
            db: Arc::new(Mutex::new(db)),
            embedding_provider,
            pain_registry: Arc::new(Mutex::new(PainRegistry::new(100))),
            phantom_tracker: Arc::new(Mutex::new(PhantomTracker::new(PhantomConfig::default()))),
            trajectory_tracker: Arc::new(Mutex::new(TrajectoryTracker::new(100))),
            config,
            tool_router: Self::tool_router(),
        }
    }
}

fn parse_memory_type(s: &str) -> Result<MemoryType, String> {
    match s.to_lowercase().as_str() {
        "episodic" => Ok(MemoryType::Episodic),
        "semantic" => Ok(MemoryType::Semantic),
        "procedural" => Ok(MemoryType::Procedural),
        "anti_pattern" | "antipattern" => Ok(MemoryType::AntiPattern),
        "reasoning" => Ok(MemoryType::Reasoning),
        "correction" => Ok(MemoryType::Correction),
        other => Err(format!("Unknown memory type: {other}")),
    }
}

fn parse_edge_type(s: &str) -> Result<EdgeType, String> {
    match s.to_lowercase().as_str() {
        "caused" => Ok(EdgeType::Caused),
        "before" => Ok(EdgeType::Before),
        "related" | "relates_to" => Ok(EdgeType::Related),
        "contradicts" => Ok(EdgeType::Contradicts),
        "supports" => Ok(EdgeType::Supports),
        "supersedes" | "obsoletes" => Ok(EdgeType::Supersedes),
        "derived" | "requires" => Ok(EdgeType::Derived),
        "part_of" => Ok(EdgeType::PartOf),
        other => Err(format!("Unknown edge type: {other}")),
    }
}

fn parse_uuid(s: &str) -> Result<Uuid, String> {
    Uuid::parse_str(s).map_err(|e| format!("Invalid UUID: {e}"))
}

fn error_result(msg: &str) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![Content::text(msg.to_string())]))
}

#[rmcp::tool_router]
impl MenteDbServer {
    #[rmcp::tool(
        description = "Store a new memory in MenteDB. Returns the unique ID of the stored memory."
    )]
    async fn store_memory(
        &self,
        Parameters(req): Parameters<StoreMemoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let memory_type = match parse_memory_type(&req.memory_type) {
            Ok(mt) => mt,
            Err(e) => return error_result(&e),
        };

        let embedding = self
            .embedding_provider
            .embed(&req.content)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;
        let agent_id = Uuid::nil();
        let mut node = MemoryNode::new(agent_id, memory_type, req.content, embedding);

        if let Some(tags) = req.tags {
            node.tags = tags;
        }

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
        let mut db = self.db.lock().await;
        match db.store(node) {
            Ok(()) => Ok(CallToolResult::success(vec![Content::text(
                json!({ "id": id.to_string(), "status": "stored" }).to_string(),
            )])),
            Err(e) => error_result(&format!("Failed to store memory: {e}")),
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

        let mut db = self.db.lock().await;
        let query = format!("LOOKUP {id}");
        match db.recall(&query) {
            Ok(window) => {
                if window.blocks.is_empty() {
                    return error_result(&format!("Memory not found: {id}"));
                }
                let result = json!({
                    "id": id.to_string(),
                    "blocks": window.blocks.len(),
                    "total_tokens": window.total_tokens,
                    "format": window.format,
                });
                Ok(CallToolResult::success(vec![Content::text(
                    result.to_string(),
                )]))
            }
            Err(e) => error_result(&format!("Recall failed: {e}")),
        }
    }

    #[rmcp::tool(
        description = "Search memories by semantic similarity. Returns matching memories ranked by relevance score."
    )]
    async fn search_memories(
        &self,
        Parameters(req): Parameters<SearchMemoriesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let k = req.limit.unwrap_or(10);
        let embedding = self
            .embedding_provider
            .embed(&req.query)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;

        let mut db = self.db.lock().await;
        match db.recall_similar(&embedding, k) {
            Ok(results) => {
                let items: Vec<serde_json::Value> = results
                    .iter()
                    .map(|(id, score)| json!({ "id": id.to_string(), "score": score }))
                    .collect();
                Ok(CallToolResult::success(vec![Content::text(
                    json!({ "results": items, "count": items.len() }).to_string(),
                )]))
            }
            Err(e) => error_result(&format!("Search failed: {e}")),
        }
    }

    #[rmcp::tool(description = "Create a typed relationship edge between two memories.")]
    async fn relate_memories(
        &self,
        Parameters(req): Parameters<RelateMemoriesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let from_id = match parse_uuid(&req.from_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };
        let to_id = match parse_uuid(&req.to_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };
        let edge_type = match parse_edge_type(&req.edge_type) {
            Ok(et) => et,
            Err(e) => return error_result(&e),
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let edge = MemoryEdge {
            source: from_id,
            target: to_id,
            edge_type,
            weight: 1.0,
            created_at: now,
        };

        let mut db = self.db.lock().await;
        match db.relate(edge) {
            Ok(()) => Ok(CallToolResult::success(vec![Content::text(
                json!({
                    "status": "related",
                    "from": from_id.to_string(),
                    "to": to_id.to_string(),
                    "edge_type": req.edge_type,
                })
                .to_string(),
            )])),
            Err(e) => error_result(&format!("Failed to relate memories: {e}")),
        }
    }

    #[rmcp::tool(description = "Delete a memory from the database.")]
    async fn forget_memory(
        &self,
        Parameters(req): Parameters<ForgetMemoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        if let Some(reason) = &req.reason {
            tracing::info!("Forgetting memory {id}: {reason}");
        }

        let mut db = self.db.lock().await;
        match db.forget(id) {
            Ok(()) => Ok(CallToolResult::success(vec![Content::text(
                json!({ "status": "forgotten", "id": id.to_string() }).to_string(),
            )])),
            Err(e) => error_result(&format!("Failed to forget memory: {e}")),
        }
    }

    #[rmcp::tool(
        description = "Assemble an optimized context window from memories for a given query and token budget."
    )]
    async fn assemble_context(
        &self,
        Parameters(req): Parameters<AssembleContextRequest>,
    ) -> Result<CallToolResult, McpError> {
        let format_str = req.format.as_deref().unwrap_or("structured");
        let embedding = self
            .embedding_provider
            .embed(&req.query)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;
        let k = 50;

        let mut db = self.db.lock().await;
        match db.recall_similar(&embedding, k) {
            Ok(results) => {
                let result = json!({
                    "query": req.query,
                    "token_budget": req.token_budget,
                    "format": format_str,
                    "candidate_count": results.len(),
                    "results": results.iter().map(|(id, score)| {
                        json!({ "id": id.to_string(), "score": score })
                    }).collect::<Vec<_>>(),
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
        let result = json!({
            "status": "operational",
            "engine": "mentedb",
            "version": "0.1.1",
        });
        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
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

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }
}

#[rmcp::tool_handler]
impl ServerHandler for MenteDbServer {
    fn get_info(&self) -> ServerInfo {
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
        .with_instructions(
            "MenteDB MCP server provides AI agent memory backed by a cognition aware database. \
             Use store_memory to save knowledge, search_memories for semantic retrieval, \
             relate_memories to build knowledge graphs, and get_cognitive_state to monitor \
             pain signals, phantom memories, and trajectory predictions.",
        )
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
                RawResource::new("mentedb://memories", "memories".to_string()).no_annotation(),
                RawResource::new("mentedb://stats", "stats".to_string()).no_annotation(),
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
            let stats = json!({
                "engine": "mentedb",
                "version": "0.1.1",
                "status": "operational",
            });
            return Ok(ReadResourceResult::new(vec![ResourceContents::text(
                stats.to_string(),
                uri.clone(),
            )]));
        }

        if uri_str == "mentedb://memories" {
            let info = json!({
                "description": "Memory listing",
                "note": "Use search_memories tool for retrieval",
            });
            return Ok(ReadResourceResult::new(vec![ResourceContents::text(
                info.to_string(),
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
            let query = format!("LOOKUP {id}");
            match db.recall(&query) {
                Ok(window) => {
                    let result = json!({
                        "id": id.to_string(),
                        "blocks": window.blocks.len(),
                        "total_tokens": window.total_tokens,
                    });
                    return Ok(ReadResourceResult::new(vec![ResourceContents::text(
                        result.to_string(),
                        uri.clone(),
                    )]));
                }
                Err(e) => {
                    return Err(McpError::resource_not_found(
                        "memory_not_found",
                        Some(json!({ "error": e.to_string() })),
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
            ],
        })
    }
}
