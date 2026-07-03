//! Engine-backed helpers for the lifecycle hook daemon.
//!
//! These run inside the daemon process, which is the single owner of the
//! local database, so hooks and any MCP stdio clients never race on the
//! engine's in-memory index and cognitive state.

use super::*;

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

impl MenteDbServer {
    /// Recall context for a user prompt: hybrid search plus always-scoped
    /// memories plus pain warnings. Read-only, used by the UserPromptSubmit
    /// hook so nothing is stored until the turn completes.
    pub(crate) fn hook_context(&self, prompt: &str, k: usize) -> serde_json::Value {
        let db = &*self.db;
        let embedding = self.embedding_provider.embed(prompt).unwrap_or_default();

        let hits = db
            .recall_hybrid_at(&embedding, Some(prompt), k, now_us(), None, None)
            .unwrap_or_default();
        let mut memories = resolve_memory_ids(db, &hits);

        // Always-scoped memories are served every turn regardless of similarity.
        let seen: std::collections::HashSet<MemoryId> =
            memories.iter().map(|sm| sm.memory.id).collect();
        for sm in recall_all_memories(db) {
            if sm.memory.tags.iter().any(|t| t == "scope:always") && !seen.contains(&sm.memory.id) {
                memories.push(sm);
            }
        }

        let keywords: Vec<String> = prompt
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let pains = db.get_pain_warnings(&keywords);

        json!({
            "memories": memories
                .iter()
                .map(|sm| json!({
                    "content": sm.memory.content,
                    "memory_type": format!("{:?}", sm.memory.memory_type),
                    "scope": if sm.memory.tags.iter().any(|t| t == "scope:always") { "always" } else { "contextual" },
                }))
                .collect::<Vec<_>>(),
            "pain": pains
                .iter()
                .map(|p| json!({ "description": p.description, "intensity": p.intensity }))
                .collect::<Vec<_>>(),
        })
    }

    /// Session-start context: the accumulated user profile plus every
    /// always-scoped memory.
    pub(crate) fn hook_session_context(&self) -> serde_json::Value {
        let db = &*self.db;
        let profile = db.user_profile().map(|m| m.content);
        let always: Vec<String> = recall_all_memories(db)
            .into_iter()
            .filter(|sm| sm.memory.tags.iter().any(|t| t == "scope:always"))
            .map(|sm| sm.memory.content)
            .collect();

        json!({ "profile": profile, "always": always })
    }

    /// Store a lightweight episodic "action" note, captured live during a
    /// session (e.g. a file edit or command). Deliberately does NOT run the
    /// full process_turn pipeline: tool actions fire many times per turn, so
    /// this is a direct tagged store. Low salience lets decay and
    /// consolidation fold these into higher-level memories over time.
    pub(crate) fn hook_store_note(&self, content: &str, project: Option<&str>) {
        let embedding = self.embedding_provider.embed(content).unwrap_or_default();
        let mut node = MemoryNode::new(
            AgentId(Uuid::nil()),
            MemoryType::Episodic,
            content.to_string(),
            embedding,
        );
        node.tags.push("action".to_string());
        node.salience = 0.4;
        if let Some(p) = project {
            node.tags.push(format!("scope:project:{p}"));
        }
        if let Err(e) = self.db.store(node) {
            tracing::warn!(error = %e, "hook note store failed");
        }
    }
}
