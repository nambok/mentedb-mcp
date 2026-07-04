//! Offline spool: turns and notes that failed to reach the backend are
//! appended here and retried on later hook invocations, so a network outage
//! or sleeping daemon never silently drops memory.

use std::path::{Path, PathBuf};

const SPOOL_FILE: &str = "spool.jsonl";
const MAX_ENTRIES: usize = 1_000;

pub fn spool_path(data_dir: &Path) -> PathBuf {
    data_dir.join(SPOOL_FILE)
}

/// Number of entries currently waiting for delivery.
pub fn depth(data_dir: &Path) -> usize {
    std::fs::read_to_string(spool_path(data_dir))
        .map(|raw| raw.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0)
}

/// Append an entry, dropping the oldest entries past the cap so a long
/// outage cannot grow the spool without bound.
pub fn push(data_dir: &Path, entry: &serde_json::Value) {
    let path = spool_path(data_dir);
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let line = entry.to_string();
    let mut lines: Vec<&str> = existing.lines().filter(|l| !l.trim().is_empty()).collect();
    lines.push(&line);
    let start = lines.len().saturating_sub(MAX_ENTRIES);
    let mut body = lines[start..].join("\n");
    body.push('\n');
    std::fs::write(&path, body).ok();
}

/// Drain the spool, returning all pending entries. Failed entries must be
/// handed back via `restore` or they are lost.
pub fn take_all(data_dir: &Path) -> Vec<serde_json::Value> {
    let path = spool_path(data_dir);
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    std::fs::remove_file(&path).ok();
    raw.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

pub fn restore(data_dir: &Path, entries: &[serde_json::Value]) {
    for e in entries {
        push(data_dir, e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn push_take_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(depth(dir.path()), 0);
        push(dir.path(), &json!({"kind": "turn", "user_message": "a"}));
        push(dir.path(), &json!({"kind": "note", "content": "b"}));
        assert_eq!(depth(dir.path()), 2);

        let entries = take_all(dir.path());
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["kind"], "turn");
        assert_eq!(depth(dir.path()), 0);
        assert!(take_all(dir.path()).is_empty());
    }

    #[test]
    fn restore_preserves_failed_entries() {
        let dir = tempfile::tempdir().unwrap();
        push(dir.path(), &json!({"kind": "turn", "user_message": "a"}));
        let entries = take_all(dir.path());
        restore(dir.path(), &entries);
        assert_eq!(depth(dir.path()), 1);
    }

    #[test]
    fn caps_spool_size() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..(MAX_ENTRIES + 50) {
            push(dir.path(), &json!({"kind": "note", "content": i}));
        }
        let entries = take_all(dir.path());
        assert_eq!(entries.len(), MAX_ENTRIES);
        // Oldest dropped, newest kept.
        assert_eq!(entries.last().unwrap()["content"], MAX_ENTRIES + 49);
    }
}
