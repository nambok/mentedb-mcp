# MenteDB MCP Server

The MCP (Model Context Protocol) server for MenteDB, the mind database for AI agents.

[![crates.io](https://img.shields.io/crates/v/mentedb-mcp)](https://crates.io/crates/mentedb-mcp)
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

## Available Tools (30 tools)

### Core Memory (7 tools)

| Tool | Description |
|------|-------------|
| `store_memory` | Store a new memory with content, type, tags, and metadata. Returns the UUID. |
| `get_memory` | Retrieve a memory by UUID with all fields including salience and confidence. |
| `recall_memory` | Recall a specific memory by UUID. Returns content, type, metadata, timestamps. |
| `search_memories` | Semantic similarity search with optional type filtering and result limit. |
| `relate_memories` | Create a typed edge between two memories (caused, contradicts, supports, etc). |
| `forget_memory` | Delete a memory from the database with optional reason. |
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

### Cognitive Systems (11 tools)

| Tool | Description |
|------|-------------|
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
  --embedding-dim <DIM>       Embedding vector dimension [default: 128]
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
