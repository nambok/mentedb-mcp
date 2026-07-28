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

That is the whole setup. The bundled hooks and the MCP connection both
authenticate from `MENTEDB_API_KEY` directly, no login step.

## What it bundles

- The MenteDB MCP server connection, wired to your account through
  `MENTEDB_API_KEY`, no URL pasting.
- The `memory` skill: standing instructions that make Claude recall relevant
  context before every reply and capture every completed turn, without being
  asked. The skill is surface aware: where lifecycle hooks are active it
  steps back and lets them run the loop, so nothing is stored twice.
- Lifecycle hooks, the same handlers Claude Code uses (`mentedb-mcp hook`).
  Where the surface fires hook events, memory is fully deterministic:
  injection and capture happen on every turn whether or not the model
  cooperates. Where per turn events are not fired yet, the skill carries the
  behavior and the hooks are already in place for the day they switch on.

## Using it in Claude Code

Pick ONE of the two install paths, not both:

- This plugin, or
- `npx mentedb-mcp@latest setup claude-code` (the standalone hook setup).

They run the same hook handlers, so either gives the full deterministic
memory loop. Installing both would register the hooks twice; the handlers
detect and collapse duplicate fires, so nothing breaks, but there is no
benefit to a double install.
