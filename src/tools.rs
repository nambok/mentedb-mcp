use std::sync::Arc;

use mentedb::MenteDb;
use mentedb_cognitive::{
    CognitionStream, InterferenceDetector, PainRegistry, PainSignal, PhantomConfig,
    PhantomTracker, StreamConfig, TrajectoryNode, TrajectoryTracker, WriteInferenceEngine,
};
use mentedb_cognitive::trajectory::DecisionState;
use mentedb_core::types::{AgentId, MemoryId};
use mentedb_consolidation::{
    ArchivalConfig, ArchivalDecision, ArchivalPipeline, ConsolidationEngine, DecayConfig,
    DecayEngine, FactExtractor, ForgetEngine, ForgetRequest, MemoryCompressor,
};
use mentedb_context::{AssemblyConfig, ContextAssembler, OutputFormat, ScoredMemory};
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::{AttributeValue, MemoryType};
use mentedb_core::{MemoryEdge, MemoryNode};
use mentedb_embedding::HashEmbeddingProvider;
use mentedb_embedding::provider::EmbeddingProvider;
use mentedb_extraction::{
    ExtractionConfig, ExtractionPipeline, MockExtractionProvider, ProcessedExtractionResult,
};
use mentedb_graph::{extract_subgraph, find_contradictions, shortest_path};
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

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ConsolidateMemoriesRequest {
    #[schemars(description = "Minimum cluster size for consolidation (default: 2)")]
    pub min_cluster_size: Option<usize>,
    #[schemars(description = "Similarity threshold for clustering (default: 0.85)")]
    pub similarity_threshold: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ApplyDecayRequest {
    #[schemars(description = "Half-life in hours for salience decay (default: 168, i.e. 7 days)")]
    pub half_life_hours: Option<f64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompressMemoryRequest {
    #[schemars(description = "UUID of the memory to compress")]
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EvaluateArchivalRequest {
    #[schemars(description = "Salience threshold below which memories are candidates for archival (default: 0.1)")]
    pub salience_threshold: Option<f32>,
    #[schemars(description = "Maximum age in days before considering archival (default: 7)")]
    pub max_age_days: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExtractFactsRequest {
    #[schemars(description = "UUID of the memory to extract facts from")]
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GdprForgetRequest {
    #[schemars(description = "Subject identifier (agent UUID) whose memories should be forgotten")]
    pub subject: String,
    #[schemars(description = "Reason for the GDPR forget request")]
    pub reason: String,
}

// -- Cognitive tool request types --

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecordPainRequest {
    #[schemars(description = "UUID of the memory associated with this pain signal")]
    pub memory_id: String,
    #[schemars(description = "Pain intensity from 0.0 to 1.0")]
    pub intensity: f32,
    #[schemars(description = "Keywords that should trigger this pain warning")]
    pub trigger_keywords: Vec<String>,
    #[schemars(description = "Human-readable description of the negative experience")]
    pub description: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DetectPhantomsRequest {
    #[schemars(description = "Content text to scan for knowledge gaps")]
    pub content: String,
    #[schemars(description = "Optional list of entities already known to the caller")]
    pub known_entities: Option<Vec<String>>,
    #[schemars(description = "Optional conversation turn ID for tracking")]
    pub turn_id: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResolvePhantomRequest {
    #[schemars(description = "UUID of the phantom memory to mark as resolved")]
    pub phantom_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecordTrajectoryRequest {
    #[schemars(description = "Conversation turn identifier")]
    pub turn_id: u64,
    #[schemars(description = "Summary of the topic discussed in this turn")]
    pub topic_summary: String,
    #[schemars(
        description = "Decision state: investigating, narrowed_to:<choice>, decided:<decision>, interrupted, or completed"
    )]
    pub decision_state: String,
    #[schemars(description = "Optional list of open questions remaining")]
    pub open_questions: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DetectInterferenceRequest {
    #[schemars(description = "List of memory UUIDs to check for interference")]
    pub memory_ids: Vec<String>,
    #[schemars(description = "Optional similarity threshold (default: 0.8)")]
    pub similarity_threshold: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct KnownFact {
    #[schemars(description = "UUID of the memory this fact came from")]
    pub memory_id: String,
    #[schemars(description = "Summary text of the known fact")]
    pub summary: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CheckStreamRequest {
    #[schemars(description = "LLM output text to check against known facts")]
    pub text: String,
    #[schemars(description = "List of known facts to check for contradictions")]
    pub known_facts: Vec<KnownFact>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WriteInferenceRequest {
    #[schemars(description = "UUID of the memory to run write-time inference on")]
    pub memory_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetRelatedRequest {
    #[schemars(description = "UUID of the memory to find relations for")]
    pub id: String,
    #[schemars(
        description = "Optional edge type filter: caused, before, related, contradicts, supports, supersedes, derived, part_of"
    )]
    pub edge_type: Option<String>,
    #[schemars(description = "Maximum traversal depth (default: 1)")]
    pub depth: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindPathRequest {
    #[schemars(description = "UUID of the source memory")]
    pub from_id: String,
    #[schemars(description = "UUID of the target memory")]
    pub to_id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetSubgraphRequest {
    #[schemars(description = "UUID of the center memory")]
    pub center_id: String,
    #[schemars(description = "Maximum hop radius from center (default: 2)")]
    pub radius: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindContradictionsRequest {
    #[schemars(description = "UUID of the memory to find contradictions for")]
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PropagateBeliefRequest {
    #[schemars(description = "UUID of the memory whose confidence changed")]
    pub id: String,
    #[schemars(description = "New confidence value (0.0 to 1.0)")]
    pub new_confidence: f32,
}

/// Parse a decision state string into a DecisionState enum.
fn parse_decision_state(s: &str) -> DecisionState {
    let lower = s.to_lowercase();
    if lower == "investigating" {
        DecisionState::Investigating
    } else if lower == "interrupted" {
        DecisionState::Interrupted
    } else if lower == "completed" {
        DecisionState::Completed
    } else if let Some(choice) = lower.strip_prefix("narrowed_to:") {
        DecisionState::NarrowedTo(choice.trim().to_string())
    } else if let Some(decision) = lower.strip_prefix("decided:") {
        DecisionState::Decided(decision.trim().to_string())
    } else {
        // Fall back to Investigating for unrecognized strings
        DecisionState::Investigating
    }
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
    #[allow(dead_code)]
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
    Ok(all.into_iter().find(|sm| sm.memory.id == MemoryId(target_id)))
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
        let mut node = MemoryNode::new(AgentId(agent_id), mem_type, memory.content.clone(), embedding);
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
        let mut node = MemoryNode::new(AgentId(agent_id), mem_type, memory.content.clone(), embedding);
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
        let mut node = MemoryNode::new(AgentId(agent_id), memory_type, req.content, embedding);

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
                        if let Some(ref tf) = type_filter
                            && mem.memory.memory_type != *tf
                        {
                            continue;
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
            source: MemoryId(from_id),
            target: MemoryId(to_id),
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
        let provider_name = req
            .provider
            .as_deref()
            .unwrap_or(&self.config.llm_provider);
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

    #[rmcp::tool(
        description = "Find clusters of similar memories and merge them into consolidated semantic memories. Returns consolidation candidates and merged results."
    )]
    async fn consolidate_memories(
        &self,
        Parameters(req): Parameters<ConsolidateMemoriesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let min_cluster_size = req.min_cluster_size.unwrap_or(2);
        let similarity_threshold = req.similarity_threshold.unwrap_or(0.85);

        let mut db = self.db.lock().await;
        let all = recall_all_memories(&mut db);
        let memories: Vec<MemoryNode> = all.into_iter().map(|sm| sm.memory).collect();

        if memories.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                json!({ "status": "no_memories", "clusters": [], "consolidated": [] }).to_string(),
            )]));
        }

        let engine = ConsolidationEngine::new();
        let candidates = engine.find_candidates(&memories, min_cluster_size, similarity_threshold);

        let mut consolidated_results: Vec<serde_json::Value> = Vec::new();
        for candidate in &candidates {
            let cluster_memories: Vec<&MemoryNode> = candidate
                .memories
                .iter()
                .filter_map(|id| memories.iter().find(|m| m.id == *id))
                .collect();
            let cluster_owned: Vec<MemoryNode> = cluster_memories.into_iter().cloned().collect();
            let consolidated = engine.consolidate(&cluster_owned);

            consolidated_results.push(json!({
                "topic": candidate.topic,
                "avg_similarity": candidate.avg_similarity,
                "source_memory_ids": candidate.memories.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                "merged_summary": consolidated.summary,
                "new_type": format!("{:?}", consolidated.new_type),
                "combined_confidence": consolidated.combined_confidence,
            }));
        }

        tracing::info!(
            clusters = candidates.len(),
            threshold = similarity_threshold,
            "memory consolidation complete"
        );

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "complete",
                "total_memories_analyzed": memories.len(),
                "clusters_found": candidates.len(),
                "consolidated": consolidated_results,
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Apply salience decay to all memories based on time and access patterns. Returns count of memories processed and those below archival threshold."
    )]
    async fn apply_decay(
        &self,
        Parameters(req): Parameters<ApplyDecayRequest>,
    ) -> Result<CallToolResult, McpError> {
        let half_life_hours = req.half_life_hours.unwrap_or(168.0); // 7 days
        let half_life_us = (half_life_hours * 3600.0 * 1_000_000.0) as u64;

        let config = DecayConfig {
            half_life_us,
            ..DecayConfig::default()
        };
        let engine = DecayEngine::new(config);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let mut db = self.db.lock().await;
        let all = recall_all_memories(&mut db);
        let mut memories: Vec<MemoryNode> = all.into_iter().map(|sm| sm.memory).collect();
        let total = memories.len();

        engine.apply_decay_batch(&mut memories, now);

        let archival_threshold = 0.1;
        let below_threshold = memories
            .iter()
            .filter(|m| DecayEngine::needs_archival(m, archival_threshold))
            .count();

        tracing::info!(
            processed = total,
            below_threshold = below_threshold,
            half_life_hours = half_life_hours,
            "decay applied"
        );

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "complete",
                "memories_processed": total,
                "below_archival_threshold": below_threshold,
                "archival_threshold": archival_threshold,
                "half_life_hours": half_life_hours,
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

        let mut db = self.db.lock().await;
        match find_memory_by_id(&mut db, id) {
            Ok(Some(sm)) => {
                let compressor = MemoryCompressor::new();
                let compressed = compressor.compress(&sm.memory);
                let original_length = sm.memory.content.len();

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
    async fn evaluate_archival(
        &self,
        Parameters(req): Parameters<EvaluateArchivalRequest>,
    ) -> Result<CallToolResult, McpError> {
        let max_salience = req.salience_threshold.unwrap_or(0.1);
        let min_age_us = req
            .max_age_days
            .map(|d| d * 24 * 3600 * 1_000_000)
            .unwrap_or(7 * 24 * 3600 * 1_000_000);

        let config = ArchivalConfig {
            max_salience,
            min_age_us,
            ..ArchivalConfig::default()
        };
        let pipeline = ArchivalPipeline::new(config);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let mut db = self.db.lock().await;
        let all = recall_all_memories(&mut db);
        let memories: Vec<MemoryNode> = all.into_iter().map(|sm| sm.memory).collect();

        let decisions = pipeline.evaluate_batch(&memories, now);

        let mut keep = Vec::new();
        let mut archive = Vec::new();
        let mut delete = Vec::new();
        let mut consolidate = Vec::new();

        for (id, decision) in &decisions {
            let id_str = id.to_string();
            match decision {
                ArchivalDecision::Keep => keep.push(id_str),
                ArchivalDecision::Archive => archive.push(id_str),
                ArchivalDecision::Delete => delete.push(id_str),
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

        let mut db = self.db.lock().await;
        match find_memory_by_id(&mut db, id) {
            Ok(Some(sm)) => {
                let extractor = FactExtractor::new();
                let facts = extractor.extract_facts(&sm.memory);

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

                tracing::info!(id = %id, facts_count = facts.len(), "facts extracted");

                Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "id": id.to_string(),
                        "facts_count": facts.len(),
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

        let mut db = self.db.lock().await;
        let all = recall_all_memories(&mut db);
        let memories: Vec<MemoryNode> = all.into_iter().map(|sm| sm.memory).collect();

        let forget_request = ForgetRequest {
            agent_id: Some(AgentId(agent_id)),
            space_id: None,
            memory_ids: Vec::new(),
            reason: req.reason.clone(),
            requested_at: now,
        };

        let engine = ForgetEngine::new();
        // No edge listing API available; pass empty edges
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

    // -- Cognitive tools --

    #[rmcp::tool(
        description = "Record a negative experience (pain signal) so MenteDB can warn when similar contexts arise."
    )]
    async fn record_pain(
        &self,
        Parameters(req): Parameters<RecordPainRequest>,
    ) -> Result<CallToolResult, McpError> {
        let memory_id = match parse_uuid(&req.memory_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let signal_id = MemoryId::new();
        let signal = PainSignal {
            id: signal_id,
            memory_id: MemoryId::from(memory_id),
            intensity: req.intensity.clamp(0.0, 1.0),
            trigger_keywords: req.trigger_keywords,
            description: req.description,
            created_at: now,
            decay_rate: 0.1,
        };

        let mut registry = self.pain_registry.lock().await;
        registry.record_pain(signal);
        tracing::info!(signal_id = %signal_id, memory_id = %memory_id, "pain signal recorded");

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "recorded",
                "signal_id": signal_id.to_string(),
                "memory_id": memory_id.to_string(),
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Scan content for knowledge gaps — entities referenced but not present in memory."
    )]
    async fn detect_phantoms(
        &self,
        Parameters(req): Parameters<DetectPhantomsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let known = req.known_entities.unwrap_or_default();
        let turn_id = req.turn_id.unwrap_or(0);

        let mut tracker = self.phantom_tracker.lock().await;
        let phantoms = tracker.detect_gaps(&req.content, &known, turn_id);

        let items: Vec<serde_json::Value> = phantoms
            .iter()
            .map(|p| {
                json!({
                    "id": p.id.to_string(),
                    "gap_description": p.gap_description,
                    "source_reference": p.source_reference,
                    "priority": format!("{:?}", p.priority),
                })
            })
            .collect();

        tracing::info!(count = items.len(), turn_id = turn_id, "phantom detection complete");
        Ok(CallToolResult::success(vec![Content::text(
            json!({ "phantoms": items, "count": items.len() }).to_string(),
        )]))
    }

    #[rmcp::tool(description = "Mark a knowledge gap (phantom memory) as resolved.")]
    async fn resolve_phantom(
        &self,
        Parameters(req): Parameters<ResolvePhantomRequest>,
    ) -> Result<CallToolResult, McpError> {
        let phantom_id = match parse_uuid(&req.phantom_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let mut tracker = self.phantom_tracker.lock().await;
        tracker.resolve(phantom_id);
        tracing::info!(phantom_id = %phantom_id, "phantom resolved");

        Ok(CallToolResult::success(vec![Content::text(
            json!({ "status": "resolved", "phantom_id": phantom_id.to_string() }).to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Record a conversation turn for trajectory tracking. Enables topic prediction and resume context."
    )]
    async fn record_trajectory(
        &self,
        Parameters(req): Parameters<RecordTrajectoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let decision_state = parse_decision_state(&req.decision_state);

        let embedding = self
            .embedding_provider
            .embed(&req.topic_summary)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let node = TrajectoryNode {
            turn_id: req.turn_id,
            topic_embedding: embedding,
            topic_summary: req.topic_summary,
            decision_state,
            open_questions: req.open_questions.unwrap_or_default(),
            timestamp: now,
        };

        let mut tracker = self.trajectory_tracker.lock().await;
        tracker.record_turn(node);
        let trajectory_len = tracker.get_trajectory().len();
        tracing::info!(turn_id = req.turn_id, trajectory_len, "trajectory turn recorded");

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "status": "recorded",
                "turn_id": req.turn_id,
                "trajectory_length": trajectory_len,
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Predict likely next topics based on the current conversation trajectory."
    )]
    async fn predict_topics(&self) -> Result<CallToolResult, McpError> {
        let tracker = self.trajectory_tracker.lock().await;
        let predictions = tracker.predict_next_topics();
        tracing::info!(count = predictions.len(), "topic predictions generated");

        Ok(CallToolResult::success(vec![Content::text(
            json!({ "predictions": predictions, "count": predictions.len() }).to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Detect pairs of memories similar enough to confuse an LLM, with disambiguation hints."
    )]
    async fn detect_interference(
        &self,
        Parameters(req): Parameters<DetectInterferenceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let threshold = req.similarity_threshold.unwrap_or(0.8);
        let detector = InterferenceDetector::new(threshold);

        let mut db = self.db.lock().await;
        let mut memories: Vec<MemoryNode> = Vec::new();
        for id_str in &req.memory_ids {
            let id = match parse_uuid(id_str) {
                Ok(id) => id,
                Err(e) => return error_result(&e),
            };
            match find_memory_by_id(&mut db, id) {
                Ok(Some(sm)) => memories.push(sm.memory),
                Ok(None) => {
                    return error_result(&format!("Memory not found: {id}"));
                }
                Err(e) => {
                    return error_result(&format!("Failed to fetch memory {id}: {e}"));
                }
            }
        }

        let pairs = detector.detect_interference(&memories);
        let items: Vec<serde_json::Value> = pairs
            .iter()
            .map(|p| {
                json!({
                    "memory_a": p.memory_a.to_string(),
                    "memory_b": p.memory_b.to_string(),
                    "similarity": p.similarity,
                    "disambiguation": p.disambiguation,
                })
            })
            .collect();

        tracing::info!(pairs = items.len(), "interference detection complete");
        Ok(CallToolResult::success(vec![Content::text(
            json!({ "interference_pairs": items, "count": items.len() }).to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Check LLM output text against known facts for contradictions, forgotten facts, and reinforcements."
    )]
    async fn check_stream(
        &self,
        Parameters(req): Parameters<CheckStreamRequest>,
    ) -> Result<CallToolResult, McpError> {
        let stream = CognitionStream::with_config(StreamConfig::default());
        stream.feed_token(&req.text);

        let facts: Vec<(MemoryId, String)> = req
            .known_facts
            .iter()
            .filter_map(|f| {
                parse_uuid(&f.memory_id)
                    .ok()
                    .map(|id| (MemoryId::from(id), f.summary.clone()))
            })
            .collect();

        let alerts = stream.check_alerts(&facts);
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

        let mut db = self.db.lock().await;
        let target_memory = match find_memory_by_id(&mut db, target_id) {
            Ok(Some(sm)) => sm.memory,
            Ok(None) => return error_result(&format!("Memory not found: {target_id}")),
            Err(e) => return error_result(&format!("Failed to fetch memory: {e}")),
        };

        // Gather nearby memories via similarity search for comparison
        let similar_ids = db
            .recall_similar(&target_memory.embedding, 20)
            .unwrap_or_default();
        let all_memories = recall_all_memories(&mut db);
        let existing: Vec<MemoryNode> = similar_ids
            .iter()
            .filter_map(|(id, _)| {
                if *id == MemoryId(target_id) {
                    return None;
                }
                all_memories
                    .iter()
                    .find(|sm| sm.memory.id == *id)
                    .map(|sm| sm.memory.clone())
            })
            .collect();

        let engine = WriteInferenceEngine::new();
        let actions = engine.infer_on_write(&target_memory, &existing, &[]);

        let items: Vec<serde_json::Value> = actions
            .iter()
            .map(|a| match a {
                mentedb_cognitive::InferredAction::FlagContradiction {
                    existing,
                    new,
                    reason,
                } => json!({
                    "action": "flag_contradiction",
                    "existing_memory": existing.to_string(),
                    "new_memory": new.to_string(),
                    "reason": reason,
                }),
                mentedb_cognitive::InferredAction::MarkObsolete {
                    memory,
                    superseded_by,
                } => json!({
                    "action": "mark_obsolete",
                    "memory": memory.to_string(),
                    "superseded_by": superseded_by.to_string(),
                }),
                mentedb_cognitive::InferredAction::CreateEdge {
                    source,
                    target,
                    edge_type,
                    weight,
                } => json!({
                    "action": "create_edge",
                    "source": source.to_string(),
                    "target": target.to_string(),
                    "edge_type": format!("{:?}", edge_type),
                    "weight": weight,
                }),
                mentedb_cognitive::InferredAction::UpdateConfidence {
                    memory,
                    new_confidence,
                } => json!({
                    "action": "update_confidence",
                    "memory": memory.to_string(),
                    "new_confidence": new_confidence,
                }),
                mentedb_cognitive::InferredAction::PropagateBeliefChange { root, delta } => {
                    json!({
                        "action": "propagate_belief_change",
                        "root": root.to_string(),
                        "delta": delta,
                    })
                }
            })
            .collect();

        tracing::info!(memory_id = %target_id, actions = items.len(), "write inference complete");
        Ok(CallToolResult::success(vec![Content::text(
            json!({ "inferred_actions": items, "count": items.len() }).to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Find all memories directly related to a given memory, with optional edge type filter and traversal depth."
    )]
    async fn get_related(
        &self,
        Parameters(req): Parameters<GetRelatedRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let depth = req.depth.unwrap_or(1);
        let edge_filter: Option<EdgeType> = match req.edge_type.as_deref() {
            Some(et) => match parse_edge_type(et) {
                Ok(et) => Some(et),
                Err(e) => return error_result(&e),
            },
            None => None,
        };

        let db = self.db.lock().await;
        let graph = db.graph();
        let csr = graph.graph();
        let mem_id = MemoryId(id);

        if !csr.contains_node(mem_id) {
            return error_result(&format!("Memory not found in graph: {id}"));
        }

        let mut related: Vec<serde_json::Value> = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut frontier = vec![(mem_id, 0usize)];
        visited.insert(mem_id);

        while let Some((current, current_depth)) = frontier.pop() {
            if current_depth >= depth {
                continue;
            }
            for (target, edge) in csr.outgoing(current) {
                if let Some(ref filter) = edge_filter
                    && edge.edge_type != *filter
                {
                    continue;
                }
                let next_depth = current_depth + 1;
                if visited.insert(target) {
                    related.push(json!({
                        "id": target.to_string(),
                        "edge_type": format!("{:?}", edge.edge_type),
                        "weight": edge.weight,
                        "depth": next_depth,
                        "direction": "outgoing",
                    }));
                    frontier.push((target, next_depth));
                }
            }
            for (source, edge) in csr.incoming(current) {
                if let Some(ref filter) = edge_filter
                    && edge.edge_type != *filter
                {
                    continue;
                }
                let next_depth = current_depth + 1;
                if visited.insert(source) {
                    related.push(json!({
                        "id": source.to_string(),
                        "edge_type": format!("{:?}", edge.edge_type),
                        "weight": edge.weight,
                        "depth": next_depth,
                        "direction": "incoming",
                    }));
                    frontier.push((source, next_depth));
                }
            }
        }

        tracing::info!(id = %id, depth = depth, related_count = related.len(), "get_related completed");
        Ok(CallToolResult::success(vec![Content::text(
            json!({ "id": id.to_string(), "related": related, "count": related.len() }).to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Find the shortest path between two memories in the knowledge graph."
    )]
    async fn find_path(
        &self,
        Parameters(req): Parameters<FindPathRequest>,
    ) -> Result<CallToolResult, McpError> {
        let from_id = match parse_uuid(&req.from_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };
        let to_id = match parse_uuid(&req.to_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let db = self.db.lock().await;
        let csr = db.graph().graph();
        let from_mem = MemoryId(from_id);
        let to_mem = MemoryId(to_id);

        if !csr.contains_node(from_mem) {
            return error_result(&format!("Source memory not found in graph: {from_id}"));
        }
        if !csr.contains_node(to_mem) {
            return error_result(&format!("Target memory not found in graph: {to_id}"));
        }

        match shortest_path(csr, from_mem, to_mem) {
            Some(path) => {
                let path_strs: Vec<String> = path.iter().map(|id| id.to_string()).collect();
                tracing::info!(from = %from_id, to = %to_id, hops = path.len() - 1, "path found");
                Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "from": from_id.to_string(),
                        "to": to_id.to_string(),
                        "path": path_strs,
                        "hops": path.len() - 1,
                    })
                    .to_string(),
                )]))
            }
            None => {
                tracing::info!(from = %from_id, to = %to_id, "no path found");
                Ok(CallToolResult::success(vec![Content::text(
                    json!({
                        "from": from_id.to_string(),
                        "to": to_id.to_string(),
                        "path": null,
                        "message": "no path found",
                    })
                    .to_string(),
                )]))
            }
        }
    }

    #[rmcp::tool(
        description = "Extract all nodes and edges within N hops of a center memory, returning the local subgraph."
    )]
    async fn get_subgraph(
        &self,
        Parameters(req): Parameters<GetSubgraphRequest>,
    ) -> Result<CallToolResult, McpError> {
        let center_id = match parse_uuid(&req.center_id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };
        let radius = req.radius.unwrap_or(2);

        let db = self.db.lock().await;
        let csr = db.graph().graph();
        let center_mem = MemoryId(center_id);

        if !csr.contains_node(center_mem) {
            return error_result(&format!("Center memory not found in graph: {center_id}"));
        }

        let (nodes, edges) = extract_subgraph(csr, center_mem, radius);

        let nodes_json: Vec<String> = nodes.iter().map(|id| id.to_string()).collect();
        let edges_json: Vec<serde_json::Value> = edges
            .iter()
            .map(|e| {
                json!({
                    "source": e.source.to_string(),
                    "target": e.target.to_string(),
                    "edge_type": format!("{:?}", e.edge_type),
                    "weight": e.weight,
                })
            })
            .collect();

        tracing::info!(
            center = %center_id,
            radius = radius,
            nodes = nodes.len(),
            edges = edges.len(),
            "subgraph extracted"
        );
        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "center": center_id.to_string(),
                "radius": radius,
                "nodes": nodes_json,
                "edges": edges_json,
                "node_count": nodes.len(),
                "edge_count": edges.len(),
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Find all memories that contradict a given memory via Contradicts edges in the knowledge graph."
    )]
    async fn find_contradictions(
        &self,
        Parameters(req): Parameters<FindContradictionsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        let db = self.db.lock().await;
        let csr = db.graph().graph();
        let mem_id = MemoryId(id);

        if !csr.contains_node(mem_id) {
            return error_result(&format!("Memory not found in graph: {id}"));
        }

        let contradictions = find_contradictions(csr, mem_id);
        let ids: Vec<String> = contradictions.iter().map(|c| c.to_string()).collect();

        tracing::info!(id = %id, contradictions = contradictions.len(), "contradictions found");
        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "id": id.to_string(),
                "contradictions": ids,
                "count": contradictions.len(),
            })
            .to_string(),
        )]))
    }

    #[rmcp::tool(
        description = "Propagate a confidence change through the knowledge graph. Returns all affected memories and their new confidence values."
    )]
    async fn propagate_belief(
        &self,
        Parameters(req): Parameters<PropagateBeliefRequest>,
    ) -> Result<CallToolResult, McpError> {
        let id = match parse_uuid(&req.id) {
            Ok(id) => id,
            Err(e) => return error_result(&e),
        };

        if req.new_confidence < 0.0 || req.new_confidence > 1.0 {
            return error_result("new_confidence must be between 0.0 and 1.0");
        }

        let db = self.db.lock().await;
        let graph = db.graph();
        let mem_id = MemoryId(id);

        if !graph.graph().contains_node(mem_id) {
            return error_result(&format!("Memory not found in graph: {id}"));
        }

        let affected = graph.propagate_belief_change(mem_id, req.new_confidence);
        let affected_json: Vec<serde_json::Value> = affected
            .iter()
            .map(|(mid, conf)| {
                json!({
                    "id": mid.to_string(),
                    "new_confidence": conf,
                })
            })
            .collect();

        tracing::info!(
            id = %id,
            new_confidence = req.new_confidence,
            affected = affected.len(),
            "belief propagation completed"
        );
        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "id": id.to_string(),
                "new_confidence": req.new_confidence,
                "affected": affected_json,
                "affected_count": affected.len(),
            })
            .to_string(),
        )]))
    }
}
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
            "MenteDB MCP server provides AI agent memory backed by a cognition aware database.\n\
             \n\
             Core tools: store_memory, recall_memory, get_memory, search_memories, \
             relate_memories, forget_memory, ingest_conversation.\n\
             \n\
             Context assembly: assemble_context with real token budgets and format control.\n\
             \n\
             Knowledge graph: get_related, find_path, get_subgraph, find_contradictions, \
             propagate_belief.\n\
             \n\
             Consolidation: consolidate_memories, apply_decay, compress_memory, \
             evaluate_archival, extract_facts, gdpr_forget.\n\
             \n\
             Cognitive systems: record_pain (pain signals), detect_phantoms / resolve_phantom \
             (knowledge gaps), record_trajectory / predict_topics (trajectory tracking), \
             detect_interference, check_stream (LLM output monitoring), write_inference \
             (write time contradiction and edge detection), register_entity, get_cognitive_state.\n\
             \n\
             Extraction: ingest_conversation extracts structured memories from raw conversation \
             text via LLM (openai, anthropic, ollama, or mock).",
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
            let mut db = self.db.lock().await;
            let memory_count = match db.recall("RECALL memories LIMIT 10000") {
                Ok(window) => window
                    .blocks
                    .iter()
                    .map(|b| b.memories.len())
                    .sum::<usize>(),
                Err(_) => 0,
            };
            let stats = json!({
                "status": "operational",
                "engine": "mentedb",
                "version": env!("CARGO_PKG_VERSION"),
                "memory_count_estimate": memory_count,
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
