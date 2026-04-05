# MenteDB MCP Server

The MCP (Model Context Protocol) server for MenteDB, the mind database for AI agents.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

## What is this?

This MCP server lets any AI agent (Claude, GPT, Copilot, or any MCP compatible client) use MenteDB as persistent memory. Store, search, relate, and consolidate memories through the standard MCP protocol. The server exposes 30 tools spanning core memory operations, knowledge graph traversal, context assembly, memory consolidation, cognitive systems, and LLM based extraction, all over stdio transport.

## Quick Start

Install from crates.io:

```bash
cargo install mentedb-mcp
```

### Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json` (macOS/Linux) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "mentedb": {
      "command": "mentedb-mcp",
      "args": ["--data-dir", "~/.mentedb"]
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
      "command": "mentedb-mcp",
      "args": ["--data-dir", "~/.mentedb"],
      "transportType": "stdio"
    }
  }
}
```

### GitHub Copilot (VS Code)

Add to `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "mentedb": {
      "type": "stdio",
      "command": "mentedb-mcp",
      "args": ["--data-dir", "~/.mentedb"]
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
      "command": "mentedb-mcp",
      "args": ["--data-dir", "~/.mentedb"],
      "alwaysAllow": [
        "store_memory", "get_memory", "recall_memory", "search_memories",
        "relate_memories", "forget_memory", "forget_all", "ingest_conversation",
        "assemble_context", "get_related", "find_path", "get_subgraph",
        "find_contradictions", "propagate_belief", "consolidate_memories",
        "apply_decay", "compress_memory", "evaluate_archival", "extract_facts",
        "gdpr_forget", "process_turn", "record_pain", "detect_phantoms",
        "resolve_phantom", "record_trajectory", "predict_topics",
        "detect_interference", "check_stream", "write_inference",
        "register_entity", "get_cognitive_state", "get_stats"
      ]
    }
  }
}
```

The `alwaysAllow` list lets memory tools run without approval prompts. Remove it if you prefer manual approval per tool call.

## Recommended Agent Instructions

For the best experience, add a `copilot-instructions.md` file that tells the agent to use memory automatically. Create `~/.copilot/copilot-instructions.md` (global) or `.github/copilot-instructions.md` (per repo):

```markdown
# Memory

You have persistent memory via MenteDB. Use it automatically, never wait to be asked.

## Every conversation start

1. Call `search_memories` with keywords from the user's first message to load relevant context.
2. Call `get_cognitive_state` to check for active pain signals or knowledge gaps.
3. If results come back, use them to inform your responses.

## During conversation

- When the user shares a preference, decision, or project detail, call `store_memory` immediately.
- When a fact changes or the user corrects you, store the new fact and call `relate_memories`
  with `supersedes` pointing from the new memory to the old one.
- When something goes wrong (bad advice, failed approach), call `record_pain`.
- When the user says "forget" or "don't remember", call `forget_memory`.

## Memory types

- `semantic`: facts, decisions, preferences, project details (most common)
- `episodic`: events, meetings, what happened
- `procedural`: how to do things, workflows, commands
- `correction`: when the user corrects you
- `anti_pattern`: mistakes to avoid

## Tags

Always add tags to stored memories. Use lowercase, descriptive tags like:
- Project names: `project-myapp`, `project-backend`
- Topics: `database`, `deployment`, `testing`, `preference`
- Context: `decision`, `architecture`, `bug-fix`
```

## Embeddings

The server uses local Candle embeddings by default with the `all-MiniLM-L6-v2` model (384 dimensions). No API key needed for semantic search. The model auto-downloads from Hugging Face on first use (~80MB) and is cached locally. If the download fails (offline), it falls back to hash embeddings.

## LLM Extraction (Optional)

Embeddings power search. For LLM based memory extraction via `ingest_conversation`, you can optionally configure an API key:

```json
{
  "mcpServers": {
    "mentedb": {
      "command": "mentedb-mcp",
      "args": ["--data-dir", "~/.mentedb", "--llm-provider", "anthropic"],
      "env": {
        "MENTEDB_LLM_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Supported providers: `openai`, `anthropic`, `ollama`, `mock` (default). Without an API key, `ingest_conversation` uses the mock provider which does basic keyword extraction.

## Available Tools (32 tools)

### Core Memory (8 tools)

| Tool | Description |
|------|-------------|
| `store_memory` | Store a new memory with content, type, tags, and metadata. Returns the UUID. |
| `get_memory` | Retrieve a memory by UUID with all fields including salience and confidence. |
| `recall_memory` | Recall a specific memory by UUID. Returns content, type, metadata, timestamps. |
| `search_memories` | Semantic similarity search with optional type filtering and result limit. |
| `relate_memories` | Create a typed edge between two memories (caused, contradicts, supports, etc). |
| `forget_memory` | Delete a memory from the database with optional reason. |
| `forget_all` | Delete ALL memories permanently. Requires `confirm='CONFIRM'` safety check. |
| `ingest_conversation` | Extract structured memories from raw conversation text via LLM provider. |

Memory types: `episodic`, `semantic`, `procedural`, `anti_pattern`, `reasoning`, `correction`.

Edge types: `caused`, `before`, `related`, `contradicts`, `supports`, `supersedes`, `derived`, `part_of`.

### Context Assembly (1 tool)

| Tool | Description |
|------|-------------|
| `assemble_context` | Build an optimized context window from memories for a query with a real token budget. Supports `structured`, `compact`, and `delta` output formats. Returns zone allocations and token usage metadata. |

### Knowledge Graph (5 tools)

| Tool | Description |
|------|-------------|
| `get_related` | Traverse relationships from a memory with optional edge type filter and depth. |
| `find_path` | Find the shortest path between two memories in the knowledge graph. |
| `get_subgraph` | Extract all nodes and edges within N hops of a center memory. |
| `find_contradictions` | Find all memories that contradict a given memory via graph edges. |
| `propagate_belief` | Propagate a confidence change through the graph, returning all affected memories. |

### Memory Consolidation (6 tools)

| Tool | Description |
|------|-------------|
| `consolidate_memories` | Cluster similar memories and merge them into consolidated semantic memories. |
| `apply_decay` | Apply time based salience decay to all memories. Configurable half life. |
| `compress_memory` | Extract key sentences from a memory, removing filler. Returns compression ratio. |
| `evaluate_archival` | Categorize all memories into keep, archive, delete, or consolidate decisions. |
| `extract_facts` | Extract structured subject/predicate/object triples from a memory. |
| `gdpr_forget` | GDPR compliant deletion of all memories for a subject with full audit log. |

### Cognitive Systems (12 tools)

| Tool | Description |
|------|-------------|
| `process_turn` | **One call per turn.** Searches context, extracts memories, stores with embeddings, runs inference, tracks trajectory. Returns relevant context, stored IDs, pain warnings, and topic predictions. |
| `record_pain` | Record a negative experience (pain signal) so MenteDB warns on similar contexts. |
| `detect_phantoms` | Scan content for knowledge gaps, entities referenced but not in memory. |
| `resolve_phantom` | Mark a knowledge gap (phantom memory) as resolved. |
| `record_trajectory` | Record a conversation turn for trajectory tracking and topic prediction. |
| `predict_topics` | Predict likely next topics based on the current conversation trajectory. |
| `detect_interference` | Find pairs of memories similar enough to confuse an LLM, with disambiguation hints. |
| `check_stream` | Check LLM output text against known facts for contradictions and reinforcements. |
| `write_inference` | Run write time inference: contradiction detection, edge suggestion, confidence adjustment. |
| `register_entity` | Register an entity for phantom memory detection. |
| `get_cognitive_state` | Full cognitive state snapshot: pain signals, phantoms, trajectory predictions. |
| `get_stats` | Database statistics: version, memory count estimate, operational status. |

## Configuration

### CLI Arguments

```
mentedb-mcp [OPTIONS]

Options:
  --data-dir <PATH>           Data directory path [default: ~/.mentedb]
  --embedding-dim <DIM>       Embedding vector dimension [default: 384]
  --llm-provider <PROVIDER>   LLM provider for extraction: openai, anthropic, ollama, mock [default: mock]
  --llm-api-key <KEY>         API key for the LLM provider (overrides env var)
  --llm-model <MODEL>         Model name override for the LLM provider
  -h, --help                  Print help
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MENTEDB_LLM_API_KEY` | Default API key for LLM extraction (also read by `--llm-api-key`) |
| `OPENAI_API_KEY` | Fallback API key when using the OpenAI provider |
| `RUST_LOG` | Logging filter, e.g. `RUST_LOG=mentedb_mcp=debug` |

The server writes logs to both stderr (for MCP clients) and a rolling file at `<data-dir>/mentedb-mcp.log`.

## Resources

MCP resources provide read only access to server state.

| URI | Type | Description |
|-----|------|-------------|
| `mentedb://stats` | Static | Database statistics: version, memory count, operational status |
| `mentedb://memories` | Static | JSON listing of all available tools with names and descriptions |
| `mentedb://memories/{id}` | Template | Full memory content by UUID via direct database lookup |
| `mentedb://cognitive/state` | Template | Cognitive state snapshot: pain signals, phantoms, trajectory |

## Architecture

The server runs on stdio transport using the [rmcp](https://crates.io/crates/rmcp) framework. It is backed by the MenteDB engine, which is composed of 13 Rust crates covering storage, indexing, graph, context assembly, consolidation, cognitive systems, embedding, and extraction. Cognitive subsystems (pain registry, phantom tracker, trajectory tracker) are initialized at startup and maintained in memory for the lifetime of the server process.

## Links

- [MenteDB Engine](https://github.com/nambok/mentedb) — the core database (13 crates)
- [MenteDB Site](https://github.com/nambok/mentedb-site) — landing page
- [crates.io](https://crates.io/crates/mentedb-mcp)

## License

Apache-2.0
