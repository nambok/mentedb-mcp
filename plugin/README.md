# MenteDB plugin for Claude

One install gives Claude persistent memory backed by your MenteDB account:
recall before answering, capture after replying, contradiction handling, and
context that survives compaction.

## Install

```
/plugin marketplace add nambok/mentedb-mcp
/plugin install mentedb@mentedb
```

Set your API key (from app.mentedb.com/connect) in the environment Claude
runs with:

```
export MENTEDB_API_KEY=mdb_...
```

## What it bundles

- The MenteDB MCP server connection, wired to your account through
  MENTEDB_API_KEY, no URL pasting.
- The `memory` skill: standing instructions that make Claude recall relevant
  context before every reply and capture every completed turn, without being
  asked.
- Lifecycle hooks, the same handlers Claude Code uses (`mentedb-mcp hook`).
  Where the surface fires hook events (Claude Code today), memory becomes
  fully deterministic: injection and capture happen on every turn whether or
  not the model cooperates. On surfaces that do not fire per turn events yet
  (Claude Cowork), the skill carries the behavior and the hooks are already
  in place for the day the events switch on.

In Claude Code specifically, `npx mentedb-mcp@latest setup claude-code`
remains the recommended path; this plugin is the same machinery in
marketplace form.
