mod cognitive;
mod consolidation;
mod context;
mod graph;
mod handler;
mod inference;
mod ingest;
mod memory;
mod process_turn;
mod process_turn_analysis;
mod process_turn_helpers;
mod types;

pub use types::*;

use std::sync::Arc;

use mentedb::MenteDb;
use mentedb_cognitive::trajectory::DecisionState;
use mentedb_cognitive::{
    CognitionStream, CognitiveLlmService, InterferenceDetector, PainRegistry, PainSignal,
    PhantomConfig, PhantomTracker, SpeculativeCache, StreamConfig, TrajectoryNode,
    TrajectoryTracker, WriteInferenceEngine,
};
use mentedb_consolidation::{
    ArchivalConfig, ArchivalDecision, ArchivalPipeline, ConsolidationEngine, DecayConfig,
    DecayEngine, FactExtractor, ForgetEngine, ForgetRequest, MemoryCompressor,
};
use mentedb_context::{AssemblyConfig, ContextAssembler, DeltaTracker, OutputFormat, ScoredMemory};
use mentedb_core::edge::EdgeType;
use mentedb_core::memory::{AttributeValue, MemoryType};
use mentedb_core::types::{AgentId, MemoryId};
use mentedb_core::{MemoryEdge, MemoryNode};
use mentedb_embedding::CandleEmbeddingProvider;
use mentedb_embedding::HashEmbeddingProvider;
use mentedb_embedding::provider::EmbeddingProvider;
use mentedb_extraction::{
    ExtractionConfig, ExtractionPipeline, MockExtractionProvider, ProcessedExtractionResult,
};
use mentedb_extraction::{ExtractionLlmJudge, HttpExtractionProvider};
use mentedb_graph::{extract_subgraph, find_contradictions, shortest_path};
use rmcp::ErrorData as McpError;
use rmcp::ServerHandler;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use serde_json::json;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::config::ServerConfig;

/// Parse a decision state string into a DecisionState enum.
pub(crate) fn parse_decision_state(s: &str) -> DecisionState {
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
    db: Arc<MenteDb>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    pain_registry: Arc<Mutex<PainRegistry>>,
    phantom_tracker: Arc<Mutex<PhantomTracker>>,
    trajectory_tracker: Arc<Mutex<TrajectoryTracker>>,
    speculative_cache: Arc<Mutex<SpeculativeCache>>,
    delta_tracker: Arc<Mutex<DeltaTracker>>,
    /// LLM-powered cognitive service for contradiction verification,
    /// entity resolution, and topic canonicalization.
    cognitive_llm: Option<Arc<CognitiveLlmService<ExtractionLlmJudge>>>,
    /// True when Candle embedding model failed to load and hash fallback is active.
    using_hash_fallback: bool,
    config: ServerConfig,
    pub tool_router: ToolRouter<Self>,
}

impl MenteDbServer {
    pub fn new(db: MenteDb, config: ServerConfig) -> Self {
        let (embedding_provider, using_hash_fallback): (Arc<dyn EmbeddingProvider>, bool) =
            match CandleEmbeddingProvider::new() {
                Ok(provider) => {
                    tracing::info!(
                        model = provider.model_name(),
                        dimensions = provider.dimensions(),
                        "Using local Candle embeddings"
                    );
                    (Arc::new(provider), false)
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to load Candle model, falling back to hash embeddings. \
                         Search results will be unreliable."
                    );
                    (
                        Arc::new(HashEmbeddingProvider::new(config.embedding_dim)),
                        true,
                    )
                }
            };
        let full_tools = config.full_tools;
        let mut tool_router = Self::tool_router_memory()
            + Self::tool_router_graph()
            + Self::tool_router_cognitive()
            + Self::tool_router_consolidation()
            + Self::tool_router_inference()
            + Self::tool_router_ingest()
            + Self::tool_router_process_turn()
            + Self::tool_router_context();

        // Initialize LLM-powered cognitive service if a provider is configured
        let cognitive_llm = if config.llm_provider != "mock" {
            let api_key = config
                .llm_api_key
                .clone()
                .or_else(|| std::env::var("MENTEDB_LLM_API_KEY").ok())
                .or_else(|| std::env::var("OPENAI_API_KEY").ok());

            let extraction_config = match config.llm_provider.as_str() {
                "openai" => {
                    if let Some(key) = api_key {
                        let mut cfg = ExtractionConfig::openai(key);
                        if let Some(model) = &config.llm_model {
                            cfg.model = model.clone();
                        }
                        Some(cfg)
                    } else {
                        tracing::warn!("OpenAI provider configured but no API key found");
                        None
                    }
                }
                "anthropic" => {
                    if let Some(key) = api_key {
                        let mut cfg = ExtractionConfig::anthropic(key);
                        if let Some(model) = &config.llm_model {
                            cfg.model = model.clone();
                        }
                        Some(cfg)
                    } else {
                        tracing::warn!("Anthropic provider configured but no API key found");
                        None
                    }
                }
                "ollama" => {
                    let mut cfg = ExtractionConfig::ollama();
                    if let Some(model) = &config.llm_model {
                        cfg.model = model.clone();
                    }
                    Some(cfg)
                }
                _ => None,
            };

            if let Some(cfg) = extraction_config {
                match HttpExtractionProvider::new(cfg) {
                    Ok(provider) => {
                        let judge = ExtractionLlmJudge::new(provider);
                        tracing::info!(
                            provider = %config.llm_provider,
                            "Cognitive LLM service initialized"
                        );
                        Some(Arc::new(CognitiveLlmService::new(judge)))
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to initialize LLM provider");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            tracing::info!("LLM provider is mock, cognitive features using heuristics only");
            None
        };

        // In default mode, only expose essential tools for better agent compliance.
        // All internal tools still run server-side via process_turn.
        if !full_tools {
            let essential = [
                "process_turn",
                "store_memory",
                "search_memories",
                "forget_memory",
            ];
            let all_names: Vec<String> = tool_router
                .list_all()
                .iter()
                .map(|t| t.name.to_string())
                .collect();
            for name in &all_names {
                if !essential.contains(&name.as_str()) {
                    tool_router.remove_route(name);
                }
            }
            tracing::info!(
                exposed = essential.len(),
                hidden = all_names.len() - essential.len(),
                "Slim tool mode: exposing only essential tools"
            );
        }

        Self {
            db: Arc::new(db),
            embedding_provider,
            pain_registry: Arc::new(Mutex::new(PainRegistry::new(100))),
            phantom_tracker: Arc::new(Mutex::new(PhantomTracker::new(PhantomConfig::default()))),
            trajectory_tracker: Arc::new(Mutex::new(TrajectoryTracker::new(100))),
            speculative_cache: Arc::new(Mutex::new(SpeculativeCache::new(64, 0.5, 0.6))),
            delta_tracker: Arc::new(Mutex::new(DeltaTracker::new())),
            cognitive_llm,
            using_hash_fallback,
            config,
            tool_router,
        }
    }

    /// Get a reference to the database for shutdown handling.
    pub fn db_ref(&self) -> Arc<MenteDb> {
        Arc::clone(&self.db)
    }
}

pub(crate) fn parse_memory_type(s: &str) -> Result<MemoryType, String> {
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

pub(crate) fn parse_edge_type(s: &str) -> Result<EdgeType, String> {
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

pub(crate) fn parse_uuid(s: &str) -> Result<Uuid, String> {
    Uuid::parse_str(s).map_err(|e| format!("Invalid UUID: {e}"))
}

pub(crate) fn error_result(msg: &str) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![Content::text(msg.to_string())]))
}

/// Serialize a MemoryNode to a JSON value with all fields.
pub(crate) fn memory_node_to_json(mem: &MemoryNode) -> serde_json::Value {
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

/// Retrieve all memories from the database using direct page_map access.
///
/// Use only when every memory is needed (consolidation, decay, archival, GDPR).
/// For similarity-based retrieval, prefer `recall_similar` / `recall_hybrid_at`.
pub(crate) fn recall_all_memories(db: &MenteDb) -> Vec<ScoredMemory> {
    db.memory_ids()
        .into_iter()
        .filter_map(|id| {
            db.get_memory(id)
                .ok()
                .map(|memory| ScoredMemory { memory, score: 1.0 })
        })
        .collect()
}

/// Resolve HNSW result IDs into full `ScoredMemory` objects.
pub(crate) fn resolve_memory_ids(db: &MenteDb, ids: &[(MemoryId, f32)]) -> Vec<ScoredMemory> {
    ids.iter()
        .filter_map(|(id, score)| {
            db.get_memory(*id).ok().map(|memory| ScoredMemory {
                memory,
                score: *score,
            })
        })
        .collect()
}

/// Find a specific memory by UUID using direct page_map lookup.
pub(crate) fn find_memory_by_id(
    db: &MenteDb,
    target_id: Uuid,
) -> Result<Option<ScoredMemory>, String> {
    match db.get_memory(MemoryId(target_id)) {
        Ok(memory) => Ok(Some(ScoredMemory { memory, score: 1.0 })),
        Err(_) => Ok(None),
    }
}

/// Store extraction results (accepted + contradictions) into the database.
pub(crate) fn store_extraction_results(
    result: &ProcessedExtractionResult,
    db: &MenteDb,
    embedding_provider: &dyn EmbeddingProvider,
    agent_id: Uuid,
) -> Result<Vec<String>, McpError> {
    let mut stored_ids = Vec::new();

    for memory in &result.to_store {
        let mem_type = mentedb_extraction::map_extraction_type_to_memory_type(&memory.memory_type);
        let embedding = embedding_provider
            .embed(&memory.content)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;
        let mut node = MemoryNode::new(
            AgentId(agent_id),
            mem_type,
            memory.content.clone(),
            embedding,
        );
        node.tags = memory.tags.clone();
        let id = node.id;
        if let Err(e) = db.store(node) {
            tracing::error!(error = %e, "failed to store extracted memory");
            continue;
        }
        stored_ids.push(id.to_string());
    }

    for (memory, _findings) in &result.contradictions {
        let mem_type = mentedb_extraction::map_extraction_type_to_memory_type(&memory.memory_type);
        let embedding = embedding_provider
            .embed(&memory.content)
            .map_err(|e| McpError::internal_error(format!("Embedding failed: {e}"), None))?;
        let mut node = MemoryNode::new(
            AgentId(agent_id),
            mem_type,
            memory.content.clone(),
            embedding,
        );
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
