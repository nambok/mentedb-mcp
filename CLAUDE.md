# CLAUDE.md

Project instructions for AI coding assistants working on mentedb-mcp.

## Project Overview

MCP (Model Context Protocol) server for MenteDB, the cognition aware database for AI agent memory.
Standalone Rust binary, not part of the mentedb workspace. Published as `mentedb-mcp` on crates.io.

## Build Commands

```bash
cargo build                     # Build
cargo test                      # Run tests
cargo fmt --all                 # Format
cargo clippy -- -D warnings     # Lint (treat warnings as errors)
cargo run -- --help             # Show CLI help
```

## Architecture

- `src/main.rs`: Entry point, CLI args (clap), tracing setup, launches server
- `src/server.rs`: Server lifecycle, opens MenteDb, starts stdio transport
- `src/tools.rs`: MCP tool definitions (store, recall, search, relate, forget, etc.) and ServerHandler impl
- `src/resources.rs`: Reserved for resource utilities
- `src/config.rs`: Server configuration struct

## Dependencies

- `rmcp` v1.x: Official Rust MCP SDK (uses `#[rmcp::tool_router]`, `#[rmcp::tool_handler]`, `Parameters<T>`)
- `mentedb` + sub-crates: Database engine (git dep from github.com/nambok/mentedb)
- `schemars` v1.x: JSON Schema generation for tool parameter types

## Conventions

- No emojis in code, comments, or documentation
- No dashes (em/en dash) in prose, use commas instead
- Conventional commits: `feat:`, `fix:`, `chore:`, single line, no emojis
- NEVER include Co-authored-by or Authored-by trailers in commits
- All clippy warnings treated as errors
- Git config: user.name "Nam Rodriguez", user.email "nambok@gmail.com"
- Edition 2024

## Key Patterns

Tools use the rmcp 1.x pattern:

```rust
#[rmcp::tool_router]
impl MyServer {
    #[rmcp::tool(description = "...")]
    async fn my_tool(&self, Parameters(req): Parameters<MyRequest>) -> Result<CallToolResult, McpError> {
        // ...
    }
}

#[rmcp::tool_handler]
impl ServerHandler for MyServer {
    fn get_info(&self) -> ServerInfo { ... }
}
```

Parameter structs derive `Deserialize` and `schemars::JsonSchema`. Use `#[schemars(description = "...")]`
for field descriptions.

The server struct must have a `pub tool_router: ToolRouter<Self>` field, initialized with `Self::tool_router()`.
