# MenteDB MCP Server

An MCP (Model Context Protocol) server for [MenteDB](https://github.com/nambok/mentedb), the
cognition aware database engine for AI agent memory. This server exposes MenteDB's storage,
retrieval, graph relationships, context assembly, and cognitive features to any MCP compatible
AI client over stdio transport.

## Installation

```bash
cargo install mentedb-mcp
```

## Configuration

### Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json` (macOS/Linux) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

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

## CLI Options

```
mentedb-mcp [OPTIONS]

Options:
  --data-dir <PATH>         Path to the data directory [default: ~/.mentedb]
  --embedding-dim <DIM>     Embedding vector dimension [default: 128]
  -h, --help                Print help
```

## Available Tools

### store_memory

Store a new memory in MenteDB.

**Parameters:**
- `content` (string, required): The text content of the memory
- `memory_type` (string, required): One of `episodic`, `semantic`, `procedural`, `anti_pattern`, `reasoning`, `correction`
- `tags` (array of strings, optional): Tags for categorization
- `metadata` (object, optional): Key value metadata

**Returns:** The unique UUID of the stored memory.

### recall_memory

Recall a specific memory by its UUID.

**Parameters:**
- `id` (string, required): The UUID of the memory

**Returns:** Memory content, type, metadata, and timestamps.

### search_memories

Search memories by semantic similarity.

**Parameters:**
- `query` (string, required): The search query text
- `limit` (number, optional): Maximum results to return (default: 10)
- `memory_type` (string, optional): Filter by memory type

**Returns:** Array of matching memories with relevance scores.

### relate_memories

Create a typed relationship edge between two memories.

**Parameters:**
- `from_id` (string, required): UUID of the source memory
- `to_id` (string, required): UUID of the target memory
- `edge_type` (string, required): One of `caused`, `before`, `related`, `contradicts`, `supports`, `supersedes`, `derived`, `part_of`

**Returns:** Confirmation with edge details.

### forget_memory

Delete a memory from the database.

**Parameters:**
- `id` (string, required): UUID of the memory to delete
- `reason` (string, optional): Reason for deletion

**Returns:** Confirmation.

### assemble_context

Assemble an optimized context window from memories for a given query and token budget.

**Parameters:**
- `query` (string, required): The query to assemble context for
- `token_budget` (number, required): Maximum token budget
- `format` (string, optional): Output format, one of `structured`, `compact`, `delta`

**Returns:** Assembled context optimized for LLM consumption.

### get_stats

Get database statistics.

**Parameters:** None.

**Returns:** Memory count, edge count, and type breakdown.

### register_entity

Register an entity for phantom memory detection. Phantom memories represent knowledge gaps
the agent should fill.

**Parameters:**
- `name` (string, required): Name of the entity
- `entity_type` (string, required): Type classification (e.g. person, tool, concept)

**Returns:** Confirmation.

### get_cognitive_state

Get the current cognitive state of the database.

**Parameters:** None.

**Returns:** Pain signals, phantom memories (knowledge gaps), and trajectory predictions.

## Available Resources

| URI | Description |
|-----|-------------|
| `mentedb://memories` | List all memory IDs |
| `mentedb://memories/{id}` | Access a specific memory by UUID |
| `mentedb://stats` | Database statistics |

## Usage Examples

An AI agent might use the tools in a conversation like this:

```
Agent: I'll store what we just learned about the deployment process.

[calls store_memory]
  content: "Production deployments require approval from two reviewers and must pass all CI checks before merge."
  memory_type: "procedural"
  tags: ["deployment", "ci", "process"]

Agent: The memory was stored with ID abc123...

Agent: Let me search for anything related to our CI pipeline.

[calls search_memories]
  query: "CI pipeline configuration"
  limit: 5

Agent: Found 3 relevant memories...

Agent: These two memories contradict each other, let me record that.

[calls relate_memories]
  from_id: "abc123..."
  to_id: "def456..."
  edge_type: "contradicts"

Agent: Let me check if there are any knowledge gaps I should address.

[calls get_cognitive_state]

Agent: The cognitive engine detected a phantom memory for "staging environment",
meaning we reference it but have no stored knowledge about it.
```

## MenteDB

MenteDB is a purpose built database engine for AI agent memory. It provides cognitive features
including interference detection, pain signals, phantom memory tracking, speculative context
pre-assembly, stream cognition, trajectory tracking, and write time inference.

Source: [github.com/nambok/mentedb](https://github.com/nambok/mentedb)

## License

Apache 2.0
