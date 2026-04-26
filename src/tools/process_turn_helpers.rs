use super::*;
use mentedb::process_turn::ProcessTurnResult;

// ── LLM enrichment methods layered on top of engine process_turn ──

impl MenteDbServer {
    /// Entity resolution via LLM (normalizes names across memories).
    pub(super) async fn resolve_entities_llm(&self, conversation: &str) -> u32 {
        let Some(ref llm) = self.cognitive_llm else {
            return 0;
        };
        let words: Vec<String> = conversation
            .split_whitespace()
            .filter(|w| w.len() >= 3 && w.chars().next().is_some_and(|c| c.is_uppercase()))
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return 0;
        }

        let candidates: Vec<mentedb_cognitive::EntityCandidate> = words
            .iter()
            .map(|w| mentedb_cognitive::EntityCandidate {
                name: w.clone(),
                context: Some(conversation.to_string()),
                memory_id: None,
            })
            .collect();
        match llm.resolve_entities(&candidates).await {
            Ok(groups) => {
                let count = groups.len() as u32;
                tracing::debug!(groups = groups.len(), "entity resolution complete");
                count
            }
            Err(e) => {
                tracing::debug!(error = %e, "entity resolution failed");
                0
            }
        }
    }

    /// LLM-powered contradiction verification on memories flagged by heuristic inference.
    /// The engine's process_turn already creates Contradicts edges heuristically;
    /// this method uses the LLM to confirm/reject and invalidate the older memory.
    pub(super) async fn verify_contradictions_llm(&self, result: &ProcessTurnResult) -> u32 {
        let Some(ref llm) = self.cognitive_llm else {
            return 0;
        };
        if result.contradiction_count == 0 {
            return 0;
        }

        let Some(eid) = result.episodic_id else {
            return 0;
        };
        let db = &*self.db;
        let Ok(new_mem) = db.get_memory(eid) else {
            return 0;
        };

        // Find memories with Contradicts edges to the new episodic memory
        let edges = db.get_edges(eid).unwrap_or_default();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let mut verified = 0u32;
        for edge in &edges {
            if edge.edge_type != EdgeType::Contradicts {
                continue;
            }
            let other_id = if edge.source == eid {
                edge.target
            } else {
                edge.source
            };
            let Ok(other_mem) = db.get_memory(other_id) else {
                continue;
            };

            let summary_a = mentedb_cognitive::MemorySummary {
                id: other_mem.id,
                content: other_mem.content.clone(),
                memory_type: other_mem.memory_type,
                confidence: other_mem.confidence,
                created_at: other_mem.created_at,
            };
            let summary_b = mentedb_cognitive::MemorySummary {
                id: new_mem.id,
                content: new_mem.content.clone(),
                memory_type: new_mem.memory_type,
                confidence: new_mem.confidence,
                created_at: new_mem.created_at,
            };

            match llm.detect_contradiction(&summary_a, &summary_b).await {
                Ok(mentedb_cognitive::ContradictionVerdict::Contradicts { reason }) => {
                    let mut old = other_mem.clone();
                    old.valid_until = Some(now);
                    let _ = db.store(old);
                    tracing::info!(
                        existing = %other_id, new = %eid,
                        reason = %reason,
                        "LLM confirmed contradiction, invalidated old memory"
                    );
                    verified += 1;
                }
                Ok(mentedb_cognitive::ContradictionVerdict::Supersedes { winner, reason }) => {
                    let loser = if winner == other_id.to_string() {
                        &new_mem
                    } else {
                        &other_mem
                    };
                    let mut old = loser.clone();
                    old.valid_until = Some(now);
                    let _ = db.store(old);
                    tracing::info!(
                        winner = %winner, reason = %reason,
                        "LLM found supersession, invalidated loser"
                    );
                    verified += 1;
                }
                Ok(mentedb_cognitive::ContradictionVerdict::Compatible { .. }) => {
                    tracing::debug!("LLM says compatible, keeping both");
                }
                Err(e) => {
                    tracing::debug!(error = %e, "LLM contradiction check failed");
                }
            }
        }
        verified
    }
}
