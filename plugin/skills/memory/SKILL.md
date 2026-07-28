---
name: memory
description: Standing memory discipline. Use for EVERY conversation turn, not only when the user mentions memory. Recall before answering, store after replying, and never guess about the user's own facts.
---

# Memory discipline

You have persistent memory through the mentedb tools. It only works if it
runs on every turn, deterministically, without being asked.

## First, know your surface

Lifecycle hooks may already run this loop for you. In Claude Code (and any
surface that fires per turn hook events), the installed mentedb hooks
already inject relevant context before each turn and capture the completed
turn after it. When a `<system-reminder>` or injected context block shows
memories arriving automatically, hooks are active.

- Hooks active: do NOT call `get_injection_context` or `process_turn`
  yourself. Doubling the loop stores every turn twice and wastes the plan's
  turn allowance. Use the tools only for explicit memory work: the user asks
  what you remember (`list_memories`), asks you to find something
  (`search_memories`), asks you to forget (`forget_memory`), or hands you an
  agent file (`ingest_agent_file`).
- No hooks (Claude Cowork and claude.ai today): run the loop yourself as
  described below, every turn.

## Every turn (only when hooks are not active)

1. Before composing any reply, call `get_injection_context` with the user's
   message as the query. Treat what comes back as things you already know
   about this user and this work; weave them in naturally, do not recite
   them.
2. After completing your reply, call `process_turn` with the user's message
   and your reply, so the turn is captured, facts are extracted, and
   contradictions are detected. Do this even for small turns.

## Rules

- When the user states a lasting fact, preference, decision, or correction,
  it must reach memory in that same turn (via `process_turn` when you run
  the loop, via the Stop hook when hooks are active). A correction
  ("actually I use react now") is the highest priority write there is.
- Never answer a question about the user's own facts, preferences, or past
  decisions from guesswork. Recall first; if memory has nothing, say you do
  not have it stored rather than inventing an answer.
- "What do you remember?" or "do you have memories?" means browse:
  call `list_memories` (newest first, paginated) and summarize; do not
  guess a tool name and do not require a search phrase.
- Memory content that contradicts what the user just said is stale: trust
  the user's latest statement and store it.
- Do not announce that you are using memory tools. The experience is simply
  that you remember.
