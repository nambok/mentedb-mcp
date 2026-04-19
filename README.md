# MenteDB MCP Server

> **Beta** — MenteDB is under active development. APIs may change between minor versions.

The MCP (Model Context Protocol) server for MenteDB, the mind database for AI agents.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## What is this?

This MCP server lets any AI agent (Claude, GPT, Copilot, or any MCP compatible client) use MenteDB as persistent memory. It connects to MenteDB Cloud by default — no local database, no file locks, works across multiple sessions simultaneously.

## Quick Start

Install and configure in one command:

```bash
npx mentedb-mcp@latest setup copilot
```

Then authenticate:

```bash
npx mentedb-mcp@latest login
```

That's it. Your agent now has persistent memory that works across all your sessions and devices. Replace `copilot` with `cursor` or `claude` for other editors.

### How it works

Once logged in, the MCP server runs as a thin HTTP client — all memory operations (store, search, recall) are handled by MenteDB Cloud. This means:

- No local database locks
- Multiple editor sessions can run simultaneously
- Memories sync across devices automatically
- Embeddings and extraction are handled server-side (no local GPU needed)

### Local mode (offline/self-hosted)

If you prefer to run entirely offline without cloud:

```bash
mentedb-mcp --local
```

In local mode, the server uses an embedded database at `~/.mentedb/`. Only one instance can run at a time due to file locking.

### Alternative: install from source

If you prefer building from source instead of npx:

```bash
cargo install mentedb-mcp
mentedb-mcp setup copilot
mentedb-mcp login
```

### Updating

After upgrading, instructions auto-update on server startup. To manually review and confirm changes:

```bash
mentedb-mcp update copilot
```

The `update` command shows you the exact instructions that will be written and asks for confirmation. If you've customized the MenteDB block, it warns you and creates a `.bak` backup. Your own instructions outside the MenteDB block are always preserved.

## CLI Commands

| Command | Description |
|---------|-------------|
| `setup <client>` | Auto-configure MCP for copilot, cursor, or claude |
| `update <client>` | Update agent instructions (preserves customizations) |
| `login` | Authenticate with MenteDB Cloud via browser |
| `logout` | Remove cloud credentials |
| `status` | Check cloud connection and token validity |

## Authentication

```bash
npx mentedb-mcp@latest login
```

This opens your browser to authorize the CLI. Once authenticated, credentials are saved to `~/.mentedb/cloud.json` and the MCP server connects to MenteDB Cloud on subsequent runs.

To check your connection:

```bash
npx mentedb-mcp@latest status
```

To revoke access:

```bash
npx mentedb-mcp@latest logout
```

You can also revoke sessions from the web dashboard at [app.mentedb.com](https://app.mentedb.com).

## Manual Configuration

### Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json` (macOS/Linux) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "mentedb": {
      "command": "npx",
      "args": ["-y", "mentedb-mcp@latest"]
    }
  }
}
```

### Cursor

Add to your Cursor MCP configuration:

```json
{
  "mcpServers": {
    "mentedb": {
      "command": "npx",
      "args": ["-y", "mentedb-mcp@latest"],
      "transportType": "stdio"
    }
  }
}
```

### GitHub Copilot CLI

Add to `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "mentedb": {
      "command": "npx",
      "args": ["-y", "mentedb-mcp@latest"],
      "alwaysAllow": [
        "process_turn", "store_memory", "search_memories", "forget_memory"
      ]
    }
  }
}
```

The `alwaysAllow` list lets memory tools run without approval prompts.

## Tools

By default, the server exposes 4 essential tools:

| Tool | Description |
|------|-------------|
| `process_turn` | **Call every turn.** Stores conversation, retrieves context, detects contradictions. |
| `store_memory` | Store an important fact with type and tags. |
| `search_memories` | Semantic search by query, or get full content by memory UUID. |
| `forget_memory` | Delete a memory by ID. |

### Local mode: full tools (`--full-tools`)

In local mode (`--local`), you can expose all 32 tools with `--full-tools` for advanced memory operations including knowledge graph traversal, consolidation, cognitive systems, and GDPR forget.

## Configuration

### CLI Arguments

```
mentedb-mcp [OPTIONS]

Options:
  --local                     Force local mode (embedded database, single instance)
  --data-dir <PATH>           Data directory path [default: ~/.mentedb]
  --embedding-dim <DIM>       Embedding vector dimension [default: 128]
  --llm-provider <PROVIDER>   LLM provider for local extraction: openai, anthropic, ollama, mock [default: mock]
  --llm-api-key <KEY>         API key for the LLM provider (overrides env var)
  --llm-model <MODEL>         Model name override for the LLM provider
  --full-tools                Expose all 32 tools (local mode only, default: 4 essential tools)
  -h, --help                  Print help
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MENTEDB_API_URL` | Override cloud API URL (default: https://api.mentedb.com) |
| `MENTEDB_CLOUD_URL` | Override cloud dashboard URL (for login flow) |
| `MENTEDB_LLM_PROVIDER` | LLM provider for local mode: `openai`, `anthropic`, `ollama`, `mock` |
| `MENTEDB_LLM_API_KEY` | API key for local LLM extraction |
| `MENTEDB_LLM_MODEL` | Model name override |

The server writes logs to both stderr and a rolling file at `~/.mentedb/mentedb-mcp.log`.

## Architecture

**Cloud mode (default):** The server runs as a lightweight HTTP proxy on stdio transport. All memory operations are forwarded to MenteDB Cloud which handles embedding generation (via AWS Bedrock Titan), semantic search, LLM extraction (via Claude), and DynamoDB storage. No local state is kept.

**Local mode (`--local`):** The server uses the full MenteDB engine with an embedded fjall database, local Candle embeddings (all-MiniLM-L6-v2), and optional LLM extraction. This mode supports all 32 tools including knowledge graph, consolidation, and cognitive systems.

## Issues

Found a bug or have a feature request? [Open an issue](https://github.com/nambok/mentedb-mcp/issues).

## License

Apache-2.0
