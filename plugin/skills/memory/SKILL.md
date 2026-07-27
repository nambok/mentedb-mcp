---
name: memory
description: Standing memory discipline. Use for EVERY conversation turn, not only when the user mentions memory. Recall before answering, store after replying, and never guess about the user's own facts.
---

# Memory discipline

You have persistent memory through the mentedb tools. It only works if you
use it on every turn, deterministically, without being asked.

## Every turn

1. Before composing any reply, call `get_injection_context` with the user's
   message as the query. Treat what comes back as things you already know
   about this user and this work; weave them in naturally, do not recite
   them.
2. After completing your reply, call `process_turn` with the user's message
   and your reply, so the turn is captured, facts are extracted, and
   contradictions are detected. Do this even for small turns.

## Rules

- When the user states a lasting fact, preference, decision, or correction,
  it must reach `process_turn` in that same turn. A correction ("actually I
  use react now") is the highest priority write there is.
- Never answer a question about the user's own facts, preferences, or past
  decisions from guesswork. Recall first; if memory has nothing, say you do
  not have it stored rather than inventing an answer.
- Memory content that contradicts what the user just said is stale: trust
  the user's latest statement and store it.
- Do not announce that you are using memory tools. The experience is simply
  that you remember.
