use schemars::JsonSchema;
use serde::Deserialize;

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
    #[schemars(
        description = "Search query text for semantic search, OR a memory UUID to get full content by ID"
    )]
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
pub struct ForgetAllRequest {
    #[schemars(
        description = "Safety confirmation. Must be exactly 'CONFIRM' to proceed. This deletes ALL memories permanently."
    )]
    pub confirm: String,
    #[schemars(description = "Optional reason for resetting the database")]
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
    #[schemars(
        description = "Optional agent UUID that owns extracted memories (defaults to nil UUID)"
    )]
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
pub struct CompressMemoryRequest {
    #[schemars(description = "UUID of the memory to compress")]
    pub id: String,
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

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProcessTurnRequest {
    #[schemars(
        description = "The user's message or question from this turn. Used for context retrieval and memory extraction."
    )]
    pub user_message: String,
    #[schemars(
        description = "The assistant's response from this turn. Can be empty if calling before drafting a response."
    )]
    pub assistant_response: Option<String>,
    #[schemars(
        description = "Conversation turn number (monotonically increasing). Used for trajectory tracking."
    )]
    pub turn_id: u64,
    #[schemars(
        description = "Current project or workspace name for scoping memories, e.g. 'mentedb-mcp' or 'my-app'."
    )]
    pub project_context: Option<String>,
    #[schemars(description = "Optional agent UUID. Defaults to nil UUID if not provided.")]
    pub agent_id: Option<String>,
}

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
    #[schemars(description = "Similarity threshold (reserved for future use)")]
    #[allow(dead_code)]
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
