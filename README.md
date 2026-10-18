# MenteDB MCP Server

> **Beta** — MenteDB is under active development. APIs may change between minor versions.

The MCP (Model Context Protocol) server for MenteDB, the mind database for AI agents.

[![Crates.io](https://img.shields.io/crates/v/mentedb-mcp)](https://crates.io/crates/mentedb-mcp) [![CI](https://github.com/nambok/mentedb-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/nambok/mentedb-mcp/actions/workflows/ci.yml) [![dependency status](https://deps.rs/repo/github/nambok/mentedb-mcp/status.svg)](https://deps.rs/repo/github/nambok/mentedb-mcp) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## What is this?

This MCP server lets any AI agent (Claude, GPT, Copilot, or any MCP compatible client) use MenteDB as persistent memory. Fresh installs run a local embedded database by default; once you log in it switches to MenteDB Cloud, which removes local file locks and syncs across sessions and devices.

## Quick Start

Install and configure in one command:

```bash
npx mentedb-mcp@latest setup copilot
```

Then authenticate:

```bash
npx mentedb-mcp@latest login
```

Login is optional: without it, everything runs locally (see Local mode). On a remote or SSH session the browser cannot reach the CLI callback; after you authorize, the dashboard shows a connection code to paste into the waiting terminal.

That's it. Your agent now has persistent memory that works across all your sessions and devices. Replace `copilot` with `cursor` or `claude` for other editors.

### Claude Code: hooks instead of MCP (recommended)

For Claude Code (the CLI), MenteDB integrates through lifecycle hooks rather than MCP tools:

```bash
npx mentedb-mcp@latest setup claude-code
```

This writes six hooks into `~/.claude/settings.json` (and every other Claude Code config directory you use). The npx launcher always runs the latest version, and the hooks keep themselves current: when a new release adds a hook event, the next session start registers it automatically, so setup is a one time step.

| Hook | What it does |
|------|-------------|
| `UserPromptSubmit` | Recalls context for your prompt and injects it before the model responds |
| `PostToolUse` | Captures significant actions (file edits, non-trivial commands) as they happen, so a long agentic session never loses work if it is interrupted |
| `PreToolUse` | Before each shell command, sends the command text to MenteDB and injects the stored rules that bear on it, matched by meaning rather than a fixed list of actions, so commit style and PR format preferences apply at the moment they matter. It never approves or blocks a command, gives up after 1.5 seconds, and prints nothing on failure |
| `Stop` | Stores the completed turn (your prompt plus the assistant's answer) through the full cognitive pipeline |
| `PreCompact` | Makes everything captured so far durable before Claude Code compacts a long session |
| `SessionStart` | Injects your user profile and always-scoped memories at session start, resume, and right after context compaction |

Why hooks beat MCP for memory:

- Zero token overhead: no tool schemas enter the model context (MCP tool definitions cost thousands of tokens per session)
- Deterministic: memory runs on every turn; the model never forgets to call it
- Post-compaction recovery: the SessionStart hook re-injects standing context after Claude Code compacts, which MCP tools cannot do
- Hooks never fail the turn: any error is logged to `~/.mentedb/` and the turn proceeds normally. Recall in `UserPromptSubmit` gives up after 8 seconds (override with `MENTEDB_HOOK_RECALL_BUDGET_MS`)

The hook backend follows your login state: cloud when authenticated (each hook is a single HTTP call), otherwise a local daemon that owns the embedded database and starts automatically on first use (`mentedb-mcp daemon`). The daemon keeps the embedding model loaded and flushes to disk after every stored turn.

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

In local mode, the server uses an embedded database at `~/.mentedb/`. Multiple processes can share it safely: writes are serialized with a cross-process file lock (flock) and reads are lock-free.

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
| `setup <client>` | Auto-configure copilot, cursor, claude (Desktop MCP), or claude-code (hooks) |
| `update <client>` | Update agent instructions (preserves customizations) |
| `login` | Authenticate with MenteDB Cloud via browser |
| `logout` | Remove cloud credentials |
| `status` | Print a health summary from local state: auth, daemon, cloud reachability, retry spool, last sync (exits 1 when unhealthy) |
| `doctor` | Diagnose the full memory pipeline (credentials, hooks, capture, spool) |
| `accounts` | Manage named accounts: `add`, `list`, `use`, `remove` |
| `sync` | Push memories from the local database up to MenteDB Cloud |
| `hook <event>` | Process a lifecycle hook: user-prompt, stop, session-start, post-tool-use, pre-tool-use, pre-compact |
| `daemon` | Run the local hook daemon (started automatically by hooks when needed) |

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

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS), `~/.config/claude/claude_desktop_config.json` (Linux), or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

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

By default the server exposes only the essential tools: 5 in cloud mode (after login), 4 in local mode, where `store_memories` is not available.

| Tool | Description |
|------|-------------|
| `process_turn` | **Call every turn.** Stores conversation, retrieves context, detects contradictions, generates pain warnings. Triggers automatic enrichment when LLM is configured. Accepts `project_context` and `agent_id` for scoping. |
| `store_memory` | Store an important fact with type, tags, and optional scope. |
| `store_memories` | Cloud mode only. Store several memories in one batch transaction (one lock and flush, near duplicate rejection). Accepts optional `agent_id` for scoped ownership. |
| `search_memories` | Semantic search by query, or get full content by memory UUID. Accepts `limit` (default 10, max 50) and `memory_type` filter. |
| `forget_memory` | Delete a memory by ID. Accepts optional `reason` for audit logging. |

The hosted endpoint `https://api.mentedb.com/mcp` (used by the Claude plugin and custom connectors) exposes a larger set: the five above plus `list_memories`, `get_recent_activity`, `get_session_context`, `get_injection_context`, `get_action_rules`, `ingest_agent_file`, `get_ingest_status`, `record_injection_outcome`, and `store_memory_multimodal`.

### Multi agent isolation

Pass an `agent_id` (any stable UUID per agent) to `process_turn` and `store_memories` and each agent recalls only its own memories plus shared ones (stored without an agent). Omit it and everything stays globally visible, matching single agent behavior. A coding agent and a research agent sharing one database no longer contaminate each other's context.

### Plan limits

Hitting a monthly limit never breaks recall: reads keep working, new turns are served read only, and the injected context carries a notice so the assistant can tell the user. Upgrading unblocks instantly.

### What `process_turn` returns

| Field | Description |
|-------|-------------|
| `context` | Top 10 semantically relevant memories + all always-scoped memories |
| `stored` | Number of facts auto-extracted and stored from this turn |
| `contradictions` | Number of contradictions detected |
| `pain_warnings` | Array of `{ signal_id, intensity, description }` from anti_pattern memories matching current context (omitted when empty) |
| `proactive_recalls` | Memories surfaced by detected action keywords (omitted when empty) |
| `detected_actions` | Action keywords recognized in the turn (omitted when empty) |

### Automatic Enrichment

In local mode, when an LLM provider is configured, `process_turn` automatically triggers a background enrichment pipeline that enhances your memory graph over time:

| Phase | What it does |
|-------|-------------|
| **Extraction** | Converts raw conversations into structured semantic facts and entity nodes |
| **Entity Linking** | Resolves duplicates and aliases (e.g., "JS" ↔ "JavaScript") using rules + LLM |
| **Community Detection** | Groups related entities and generates summaries per community |
| **User Model** | Builds an always-available user profile from accumulated knowledge |

Enrichment is **fully automatic** — no additional tools or configuration needed beyond setting an LLM provider. Results feed directly into future `process_turn` context retrieval, improving recall quality over time.

Configure an LLM provider via environment variables (the provider defaults to `mock`, which disables enrichment, so it must be set explicitly):

```bash
# OpenAI
export MENTEDB_LLM_PROVIDER=openai
export MENTEDB_LLM_API_KEY=sk-...   # falls back to OPENAI_API_KEY

# Or Anthropic
export MENTEDB_LLM_PROVIDER=anthropic
export MENTEDB_LLM_API_KEY=sk-ant-...

# Or Ollama (local, no key needed)
export MENTEDB_LLM_PROVIDER=ollama
```

In cloud mode enrichment runs server side and needs no configuration.

Without an LLM provider, the MCP server works perfectly — enrichment simply doesn't run.

### Memory Types

| Type | Use for | Example |
|------|---------|---------|
| `semantic` | Facts, preferences, project details | "User prefers Rust over Go" |
| `episodic` | What happened in a specific interaction | "Debugged OOM in prod on Jan 5" |
| `procedural` | How to do things | "To release: bump version, tag, push" |
| `correction` | Something was wrong and is now right | "API key goes in .env, not config.toml" |
| `anti_pattern` | Things to never do | "Never force-push to main" |
| `reasoning` | Why a decision was made | "Chose DynamoDB over Postgres for scaling" |

### Memory Scope

| Scope | Behavior |
|-------|----------|
| `contextual` (default) | Retrieved by semantic similarity when relevant to the conversation |
| `always` | Returned on every `process_turn` call regardless of topic. Use for critical rules. |

Set `scope: 'always'` when the user says "always remember this" or states a hard constraint.

### Memory Quality Guidelines

- **One fact per memory** — don't combine unrelated facts in a single memory
- **Self-contained** — "User prefers tabs over spaces in Python" not just "prefers tabs"
- **Include context** — "For mentedb-mcp, deploy with..." not just "deploy with..."
- **Keep under 200 words** — summarize if needed
- **Don't store** — greetings, temporary info, large code blocks, chitchat

### Resilience

Even if `process_turn` fails or errors on a turn, always call it again on the next turn. Never skip because of a prior failure.

### Local mode: full tools (`--full-tools`)

In local mode (`--local`), you can expose all 32 tools with `--full-tools` for advanced memory operations including knowledge graph traversal, consolidation, cognitive systems, and GDPR forget.

## Configuration

### CLI Arguments

```
mentedb-mcp [OPTIONS]

Options:
  --local                     Force local mode (embedded database; safe for concurrent processes via file locking)
  --data-dir <PATH>           Data directory path [default: ~/.mentedb]
  --embedding-dim <DIM>       Dimension of the hash fallback used only if the local embedding model fails to load [default: 128]
  --llm-provider <PROVIDER>   LLM provider for local extraction: openai, anthropic, ollama, mock [default: mock]
  --llm-api-key <KEY>         API key for the LLM provider (overrides env var)
  --llm-model <MODEL>         Model name override for the LLM provider
  --full-tools                Expose all 32 tools (local mode only, default: 4 essential tools)
  -h, --help                  Print help
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MENTEDB_API_KEY` | `mdb_` API key; when set it takes precedence over the logged in account |
| `MENTEDB_API_URL` | Override cloud API URL (default: https://api.mentedb.com) |
| `MENTEDB_CLOUD_URL` | Override cloud dashboard URL (for login flow) |
| `MENTEDB_LLM_PROVIDER` | LLM provider: `openai`, `anthropic`, `ollama`, `mock` |
| `MENTEDB_LLM_API_KEY` | API key for LLM extraction (falls back to `OPENAI_API_KEY`) |
| `MENTEDB_LLM_MODEL` | Model name override |
| `MENTEDB_HOOK_RECALL_BUDGET_MS` | Time limit for recall in the `UserPromptSubmit` hook (default 8000) |

The server writes logs to both stderr and a rolling file at `~/.mentedb/mentedb-mcp.log`.

## Architecture

**Cloud mode (after login):** The server runs as a lightweight HTTP proxy on stdio transport. Memory operations are forwarded to MenteDB Cloud, which runs the MenteDB engine on ECS Fargate with per-user data directories on encrypted EFS, embeddings via Cohere embed-english-v3 (1024 dimensions) on AWS Bedrock, and server-side LLM extraction. No memories are stored locally; only credentials, logs, and the hooks' retry spool live under `~/.mentedb/`.

**Local mode (`--local`):** The server uses the full MenteDB engine with its embedded storage engine, local Candle embeddings (all-MiniLM-L6-v2, 384 dimensions), and optional LLM extraction. This mode supports all 32 tools including knowledge graph, consolidation, and cognitive systems.

## Issues

Found a bug or have a feature request? [Open an issue](https://github.com/nambok/mentedb-mcp/issues).

## License

Apache-2.0
