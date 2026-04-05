use std::sync::Arc;

use mentedb::MenteDb;
use mentedb_cognitive::{PainRegistry, PhantomConfig, PhantomTracker, TrajectoryTracker};
use mentedb_context::{AssemblyConfig, ContextAssembler, OutputFormat, ScoredMemory};
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::{AttributeValue, MemoryType};
use mentedb_core::{MemoryEdge, MemoryNode};
use mentedb_embedding::HashEmbeddingProvider;
use mentedb_embedding::provider::EmbeddingProvider;
use mentedb_extraction::{
    ExtractionConfig, ExtractionPipeline, MockExtractionProvider, ProcessedExtractionResult,
};
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
    #[schemars(description = "Optional agent UUID that owns this memory (defaults to nil UUID)")]
    pub agent_id: Option<String>,
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
    #[schemars(description = "Optional edge weight (default: 1.0)")]
    pub weight: Option<f32>,
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

#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestConversationRequest {
    #[schemars(description = "The raw conversation text to extract memories from")]
    pub conversation: String,
    #[schemars(
        description = "LLM provider to use: openai, anthropic, ollama, or mock (default: mock)"
    )]
    pub provider: Option<String>,
    #[schemars(description = "API key for the LLM provider (uses env var if not provided)")]
    pub api_key: Option<String>,
    #[schemars(description = "Optional agent UUID that owns extracted memories (defaults to nil UUID)")]
    pub agent_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetMemoryRequest {
    #[schemars(description = "The UUID of the memory to retrieve")]
    pub id: String,
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

/// Serialize a MemoryNode to a JSON value with all fields.
fn memory_node_to_json(mem: &MemoryNode) -> serde_json::Value {
    let attrs: serde_json::Map<String, serde_json::Value> = mem
        .attributes
        .iter()
        .map(|(k, v)| {
            let val = match v {
                AttributeValue::String(s) => serde_json::Value::String(s.clone()),
                AttributeValue::Integer(i) => json!(*i),
                AttributeValue::Float(f) => json!(*f),
                AttributeValue::Boolean(b) => json!(*b),
                AttributeValue::Bytes(b) => json!(format!("<{} bytes>", b.len())),
            };
            (k.clone(), val)
        })
        .collect();
    json!({
        "id": mem.id.to_string(),
        "agent_id": mem.agent_id.to_string(),
        "content": mem.content,
        "memory_type": format!("{:?}", mem.memory_type),
        "tags": mem.tags,
        "attributes": attrs,
        "created_at": mem.created_at,
        "accessed_at": mem.accessed_at,
        "access_count": mem.access_count,
        "salience": mem.salience,
        "confidence": mem.confidence,
        "space_id": mem.space_id.to_string(),
    })
}

/// Retrieve all memories from the database via a broad MQL recall.
/// Workaround until MenteDb exposes a public get_by_id method.
fn recall_all_memories(db: &mut MenteDb) -> Vec<ScoredMemory> {
    match db.recall("RECALL memories LIMIT 10000") {
        Ok(window) => window
            .blocks
            .into_iter()
            .flat_map(|b| b.memories)
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Find a specific memory by UUID using a broad recall scan.
fn find_memory_by_id(
    db: &mut MenteDb,
    target_id: Uuid,
) -> Result<Option<ScoredMemory>, String> {
    let all = recall_all_memories(db);
    Ok(all.into_iter().find(|sm| sm.memory.id == target_id))
}

/// Store extraction results (accepted + contradictions) into the database.
fn store_extraction_results(
    result: &ProcessedExtractionResult,
    db: &mut MenteDb,
    embedding_provider: &HashEmbeddingProvider,
    agent_id: Uuid,
) -> Result<Vec<String>, McpError> {
    let mut stored_ids = Vec::new();

    for memory in &result.to_store {
        let mem_type =
            mentedb_extraction::map_extraction_type_to_memory_type(&memory.memory_type);
        let embedding = embedding_provider
            .embed(&memory.content)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;
        let mut node = MemoryNode::new(agent_id, mem_type, memory.content.clone(), embedding);
        node.tags = memory.tags.clone();
        let id = node.id;
        if let Err(e) = db.store(node) {
            tracing::error!(error = %e, "failed to store extracted memory");
            continue;
        }
        stored_ids.push(id.to_string());
    }

    for (memory, _findings) in &result.contradictions {
        let mem_type =
            mentedb_extraction::map_extraction_type_to_memory_type(&memory.memory_type);
        let embedding = embedding_provider
            .embed(&memory.content)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;
        let mut node = MemoryNode::new(agent_id, mem_type, memory.content.clone(), embedding);
        node.tags = memory.tags.clone();
        node.tags.push("has_contradiction".to_string());
        let id = node.id;
        if let Err(e) = db.store(node) {
            tracing::error!(error = %e, "failed to store contradicting memory");
            continue;
        }
        stored_ids.push(id.to_string());
    }

    Ok(stored_ids)
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
        let agent_id = match req.agent_id.as_deref() {
            Some(id_str) => match parse_uuid(id_str) {
                Ok(id) => id,
                Err(e) => return error_result(&e),
            },
            None => Uuid::nil(),
        };
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

        let mut db = self.db.lock().await;
        match find_memory_by_id(&mut db, id) {
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

        let type_filter = match req.memory_type.as_deref() {
            Some(t) => Some(parse_memory_type(t).map_err(|e| {
                McpError::internal_error(format!("Invalid memory_type filter: {e}"), None)
            })?),
            None => None,
        };

        let mut db = self.db.lock().await;
        // Fetch extra candidates when filtering by type since some will be excluded
        let fetch_k = if type_filter.is_some() { k * 3 } else { k };
        match db.recall_similar(&embedding, fetch_k) {
            Ok(results) => {
                tracing::info!(query = %req.query, k = k, results = results.len(), "search completed");
                // Retrieve full memory data via broad recall for content enrichment
                let all_memories = recall_all_memories(&mut db);
                let mut items: Vec<serde_json::Value> = Vec::new();
                for (id, score) in &results {
                    if let Some(mem) = all_memories.iter().find(|sm| sm.memory.id == *id) {
                        if let Some(ref tf) = type_filter {
                            if mem.memory.memory_type != *tf {
                                continue;
                            }
                        }
                        items.push(json!({
                            "id": id.to_string(),
                            "score": score,
                            "content": mem.memory.content,
                            "memory_type": format!("{:?}", mem.memory.memory_type),
                            "tags": mem.memory.tags,
                            "salience": mem.memory.salience,
                        }));
                    } else {
                        // Memory not in recall window; include with limited info
                        if type_filter.is_some() {
                            continue; // can't verify type, skip
                        }
                        items.push(json!({ "id": id.to_string(), "score": score }));
                    }
                    if items.len() >= k {
                        break;
                    }
                }
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
            weight: req.weight.unwrap_or(1.0),
            created_at: now,
        };

        let mut db = self.db.lock().await;
        match db.relate(edge) {
            Ok(()) => {
                tracing::info!(from = %from_id, to = %to_id, edge_type = %req.edge_type, "memories related");
                Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "status": "related",
                        "from": from_id.to_string(),
                        "to": to_id.to_string(),
                        "edge_type": req.edge_type,
                    })
                    .to_string(),
                )]))
            }
            Err(e) => {
                tracing::error!(from = %from_id, to = %to_id, error = %e, "relate_memories failed");
                error_result(&format!("Failed to relate memories: {e}"))
            }
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
            tracing::info!(id = %id, reason = %reason, "forgetting memory");
        }

        let mut db = self.db.lock().await;
        match db.forget(id) {
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

        let mut db = self.db.lock().await;
        match db.recall_similar(&embedding, 50) {
            Ok(results) => {
                // Fetch full memory data for scored memories
                let all_memories = recall_all_memories(&mut db);
                let scored_memories: Vec<ScoredMemory> = results
                    .iter()
                    .filter_map(|(id, score)| {
                        all_memories
                            .iter()
                            .find(|sm| sm.memory.id == *id)
                            .map(|sm| ScoredMemory {
                                memory: sm.memory.clone(),
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
        let mut db = self.db.lock().await;
        // Use a broad recall to estimate memory count
        let memory_count = match db.recall("RECALL memories LIMIT 10000") {
            Ok(window) => window
                .blocks
                .iter()
                .map(|b| b.memories.len())
                .sum::<usize>(),
            Err(_) => 0,
        };
        let result = json!({
            "status": "operational",
            "engine": "mentedb",
            "version": env!("CARGO_PKG_VERSION"),
            "memory_count_estimate": memory_count,
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

    #[rmcp::tool(
        description = "Ingest a raw conversation, extract structured memories via LLM, run cognitive checks, and store the results. Returns extraction statistics and stored memory IDs."
    )]
    async fn ingest_conversation(
        &self,
        Parameters(req): Parameters<IngestConversationRequest>,
    ) -> Result<CallToolResult, McpError> {
        let provider_name = req.provider.as_deref().unwrap_or("mock");
        let api_key = req
            .api_key
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

        // Gather existing memories for dedup/contradiction checks
        let existing_memories = {
            let mut db = self.db.lock().await;
            let conv_embedding = self
                .embedding_provider
                .embed(&req.conversation)
                .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;
            let similar = db.recall_similar(&conv_embedding, 20).unwrap_or_default();
            let all_mems = recall_all_memories(&mut db);
            similar
                .iter()
                .filter_map(|(id, _)| {
                    all_mems
                        .iter()
                        .find(|sm| sm.memory.id == *id)
                        .map(|sm| sm.memory.clone())
                })
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
                .map_err(|e| McpError::internal_error(format!("Extraction failed: {e}"), None))?
        } else {
            let http_provider =
                mentedb_extraction::HttpExtractionProvider::new(config).map_err(|e| {
                    McpError::internal_error(format!("Provider init failed: {e}"), None)
                })?;
            let pipeline = ExtractionPipeline::new(http_provider, ExtractionConfig::default());
            pipeline
                .process(
                    &req.conversation,
                    &existing_memories,
                    self.embedding_provider.as_ref(),
                )
                .await
                .map_err(|e| McpError::internal_error(format!("Extraction failed: {e}"), None))?
        };

        let mut db = self.db.lock().await;
        let stored_ids =
            store_extraction_results(&result, &mut db, &self.embedding_provider, agent_id)?;

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

        let mut db = self.db.lock().await;
        match find_memory_by_id(&mut db, id) {
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
                "version": env!("CARGO_PKG_VERSION"),
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
            match find_memory_by_id(&mut db, id) {
                Ok(Some(sm)) => {
                    let result = memory_node_to_json(&sm.memory);
                    return Ok(ReadResourceResult::new(vec![ResourceContents::text(
                        result.to_string(),
                        uri.clone(),
                    )]));
                }
                Ok(None) => {
                    return Err(McpError::resource_not_found(
                        "memory_not_found",
                        Some(json!({ "error": format!("Memory not found: {id}") })),
                    ));
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
