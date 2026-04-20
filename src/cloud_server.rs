use rmcp::ErrorData as McpError;
use rmcp::RoleServer;
use rmcp::ServerHandler;
use rmcp::ServiceExt;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::service::RequestContext;
use rmcp::transport::io::stdio;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::cloud_client::CloudClient;

// -- Request types (match the cloud API's expected arguments) --

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
    #[allow(dead_code)]
    pub project_context: Option<String>,
    #[schemars(description = "Optional agent UUID. Defaults to nil UUID if not provided.")]
    #[allow(dead_code)]
    pub agent_id: Option<String>,
}

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
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchMemoriesRequest {
    #[schemars(
        description = "Search query text for semantic search, OR a memory UUID to get full content by ID"
    )]
    pub query: String,
    #[schemars(description = "Maximum number of results to return (default: 10)")]
    #[allow(dead_code)]
    pub limit: Option<usize>,
    #[schemars(
        description = "Optional memory type filter: episodic, semantic, procedural, anti_pattern, reasoning, correction"
    )]
    #[allow(dead_code)]
    pub memory_type: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ForgetMemoryRequest {
    #[schemars(description = "UUID of the memory to delete")]
    pub id: String,
    #[schemars(description = "Optional reason for deletion")]
    #[allow(dead_code)]
    pub reason: Option<String>,
}

/// Cloud-backed MCP server that proxies all tool calls to the MenteDB cloud API.
/// No local database is opened — multiple instances can run concurrently.
pub struct CloudMenteDbServer {
    client: CloudClient,
    pub tool_router: ToolRouter<Self>,
}

impl CloudMenteDbServer {
    pub fn new(client: CloudClient) -> Self {
        let tool_router = Self::tool_router();
        Self {
            client,
            tool_router,
        }
    }
}

fn error_result(msg: &str) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![Content::text(msg.to_string())]))
}

#[rmcp::tool_router]
impl CloudMenteDbServer {
    #[rmcp::tool(
        description = "Process a conversation turn. Stores new memories and returns relevant context from past conversations. MUST be called every turn."
    )]
    async fn process_turn(
        &self,
        Parameters(req): Parameters<ProcessTurnRequest>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "user_message": req.user_message,
            "assistant_response": req.assistant_response.unwrap_or_default(),
            "turn_id": req.turn_id,
        });

        match self.client.call_tool("process_turn", args).await {
            Ok(resp) => {
                let text = resp
                    .content
                    .first()
                    .map(|c| c.text.clone())
                    .unwrap_or_default();
                if resp.is_error {
                    error_result(&text)
                } else {
                    Ok(CallToolResult::success(vec![Content::text(text)]))
                }
            }
            Err(e) => error_result(&e),
        }
    }

    #[rmcp::tool(
        description = "Store a new memory. Use for important facts, preferences, decisions, corrections, or procedures worth remembering."
    )]
    async fn store_memory(
        &self,
        Parameters(req): Parameters<StoreMemoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "content": req.content,
            "memory_type": req.memory_type,
            "tags": req.tags.unwrap_or_default(),
        });

        match self.client.call_tool("store_memory", args).await {
            Ok(resp) => {
                let text = resp
                    .content
                    .first()
                    .map(|c| c.text.clone())
                    .unwrap_or_default();
                if resp.is_error {
                    error_result(&text)
                } else {
                    Ok(CallToolResult::success(vec![Content::text(text)]))
                }
            }
            Err(e) => error_result(&e),
        }
    }

    #[rmcp::tool(
        description = "Search memories by semantic similarity or retrieve a specific memory by UUID."
    )]
    async fn search_memories(
        &self,
        Parameters(req): Parameters<SearchMemoriesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "query": req.query,
        });

        match self.client.call_tool("search_memories", args).await {
            Ok(resp) => {
                let text = resp
                    .content
                    .first()
                    .map(|c| c.text.clone())
                    .unwrap_or_default();
                if resp.is_error {
                    error_result(&text)
                } else {
                    Ok(CallToolResult::success(vec![Content::text(text)]))
                }
            }
            Err(e) => error_result(&e),
        }
    }

    #[rmcp::tool(
        description = "Delete a memory by its UUID. Use when the user says 'forget' or 'don't remember that'."
    )]
    async fn forget_memory(
        &self,
        Parameters(req): Parameters<ForgetMemoryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let args = serde_json::json!({
            "id": req.id,
        });

        match self.client.call_tool("forget_memory", args).await {
            Ok(resp) => {
                let text = resp
                    .content
                    .first()
                    .map(|c| c.text.clone())
                    .unwrap_or_default();
                if resp.is_error {
                    error_result(&text)
                } else {
                    Ok(CallToolResult::success(vec![Content::text(text)]))
                }
            }
            Err(e) => error_result(&e),
        }
    }
}

impl ServerHandler for CloudMenteDbServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder().enable_tools().build(),
        )
        .with_server_info(Implementation::new(
            "mentedb-mcp",
            env!("CARGO_PKG_VERSION"),
        ))
        .with_instructions(
            "MenteDB gives you persistent memory across sessions. You have 4 tools:\n\
             \n\
             1. process_turn — Call on EVERY turn. Pass user_message + assistant_response (can be empty). Returns past context, stores the turn, detects contradictions.\n\
             2. store_memory — Save important facts (preferences, decisions, corrections). Add type + tags.\n\
             3. search_memories — Look up what you know. Pass a query OR a memory UUID for full content.\n\
             4. forget_memory — Delete a memory when the user asks to forget.\n\
             \n\
             USE THE CONTEXT: process_turn returns summaries with IDs. Reference them. Call search_memories(id) for full text.\n\
             If pain_warnings are returned, WARN the user. If contradictions > 0, flag it.",
        )
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _cx: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult {
            meta: None,
            tools: self.tool_router.list_all(),
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        cx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let context = ToolCallContext::new(self, request, cx);
        self.tool_router.call(context).await
    }
}

/// Start the cloud-mode MCP server on stdio transport.
pub async fn run(api_url: String, token: String) -> anyhow::Result<()> {
    let client = CloudClient::new(api_url, token);
    let server = CloudMenteDbServer::new(client);

    tracing::info!("Starting cloud MCP server on stdio transport (no local database)");

    let service = server.serve(stdio()).await.inspect_err(|e| {
        tracing::error!("Server error: {:?}", e);
    })?;

    let _ = service.waiting().await;

    tracing::info!("Cloud MCP server shut down");
    Ok(())
}
