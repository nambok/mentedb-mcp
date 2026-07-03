//! Lifecycle hooks for AI clients, in the style of a PreToolUse rewrite hook:
//! the client invokes this binary on every turn, memory happens
//! deterministically, and no MCP tool schemas enter the model context.
//!
//! Claude Code events supported:
//! - UserPromptSubmit: `hook user-prompt` reads the payload from stdin and
//!   prints recalled context as `hookSpecificOutput.additionalContext`.
//! - Stop: `hook stop` pairs `last_assistant_message` with the prompt stashed
//!   at submit time and stores the completed turn through process_turn.
//! - SessionStart: `hook session-start` prints the user profile and
//!   always-scoped memories as plain text (re-injected after compaction).
//!
//! Hooks never fail the client: every error is logged to the data directory
//! log file and the process exits 0. Output is written only on success.

mod backend;

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::json;

use backend::Backend;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum HookEvent {
    /// UserPromptSubmit: inject recalled context for the prompt
    UserPrompt,
    /// Stop: store the completed turn
    Stop,
    /// SessionStart: inject profile and always-scoped memories
    SessionStart,
    /// PostToolUse: capture a significant tool action live
    PostToolUse,
    /// PreCompact: flush memory before context is compacted away
    PreCompact,
}

/// Per-session state persisted across hook invocations.
///
/// The daemon or cloud holds all memory state; this file only carries what
/// the split hook lifecycle needs: the monotonic turn counter and the prompt
/// waiting for its assistant response.
#[derive(Debug, Default, Serialize, Deserialize)]
struct SessionState {
    turn_id: u64,
    pending_prompt: Option<String>,
}

fn state_path(data_dir: &Path, session_id: &str) -> PathBuf {
    let safe: String = session_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(64)
        .collect();
    let name = if safe.is_empty() {
        "default".to_string()
    } else {
        safe
    };
    data_dir.join("hook_sessions").join(format!("{name}.json"))
}

fn load_state(data_dir: &Path, session_id: &str) -> SessionState {
    std::fs::read_to_string(state_path(data_dir, session_id))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

fn save_state(data_dir: &Path, session_id: &str, state: &SessionState) {
    let path = state_path(data_dir, session_id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Ok(raw) = serde_json::to_string(state) {
        std::fs::write(path, raw).ok();
    }
}

/// Entry point for `mentedb-mcp hook <event>`. Never returns an error to the
/// caller: hooks must not break the client.
pub async fn run(event: HookEvent, data_dir: PathBuf, force_local: bool) -> anyhow::Result<()> {
    if let Err(e) = run_inner(event, &data_dir, force_local).await {
        tracing::warn!(event = ?event, error = %e, "hook failed, client unaffected");
    }
    Ok(())
}

async fn run_inner(event: HookEvent, data_dir: &Path, force_local: bool) -> anyhow::Result<()> {
    let mut raw = String::new();
    std::io::stdin().read_to_string(&mut raw).ok();
    let payload: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|_| json!({}));
    let session_id = payload
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();

    match event {
        HookEvent::UserPrompt => {
            let prompt = payload
                .get("prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if prompt.is_empty() {
                return Ok(());
            }

            // Stash the prompt so the Stop hook can store the complete turn.
            let mut state = load_state(data_dir, &session_id);
            state.pending_prompt = Some(prompt.chars().take(8_000).collect());
            save_state(data_dir, &session_id, &state);

            let backend = Backend::resolve(data_dir, force_local).await?;
            let ctx = backend.context(&prompt).await?;
            if let Some(text) = format_context(&ctx) {
                println!(
                    "{}",
                    json!({
                        "hookSpecificOutput": {
                            "hookEventName": "UserPromptSubmit",
                            "additionalContext": text,
                        }
                    })
                );
            }
        }
        HookEvent::Stop => {
            let assistant = payload
                .get("last_assistant_message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .chars()
                .take(16_000)
                .collect::<String>();

            let mut state = load_state(data_dir, &session_id);
            let Some(user_message) = state.pending_prompt.take() else {
                return Ok(());
            };
            // turn_id starts at 1: the engine treats turn 0 as "no turn" and
            // skips periodic maintenance for it.
            state.turn_id += 1;
            let turn_id = state.turn_id;
            save_state(data_dir, &session_id, &state);

            let project = payload
                .get("cwd")
                .and_then(|v| v.as_str())
                .and_then(|c| Path::new(c).file_name())
                .and_then(|n| n.to_str())
                .map(str::to_string);

            let backend = Backend::resolve(data_dir, force_local).await?;
            backend
                .store_turn(&user_message, &assistant, turn_id, project)
                .await?;
        }
        HookEvent::SessionStart => {
            let backend = Backend::resolve(data_dir, force_local).await?;
            let ctx = backend.session_context().await?;
            if let Some(text) = format_session_context(&ctx) {
                println!("{text}");
            }
        }
        HookEvent::PostToolUse => {
            // Capture significant actions as they happen so an interrupted
            // long session never loses its work, and mid-session decisions
            // become memory immediately rather than only at Stop.
            let Some(note) = summarize_tool_action(&payload) else {
                return Ok(());
            };
            let project = payload
                .get("cwd")
                .and_then(|v| v.as_str())
                .and_then(|c| Path::new(c).file_name())
                .and_then(|n| n.to_str())
                .map(str::to_string);
            let backend = Backend::resolve(data_dir, force_local).await?;
            backend.store_note(&note, project).await?;
        }
        HookEvent::PreCompact => {
            // The long session is about to lose context; make sure everything
            // captured so far is durable.
            let backend = Backend::resolve(data_dir, force_local).await?;
            backend.flush().await?;
        }
    }
    Ok(())
}

/// Bash command prefixes that only read state and are not worth remembering.
const READ_ONLY_BASH: &[&str] = &[
    "ls",
    "cat",
    "grep",
    "rg",
    "find",
    "pwd",
    "echo",
    "which",
    "head",
    "tail",
    "less",
    "more",
    "git status",
    "git diff",
    "git log",
    "git show",
    "git branch",
    "cd",
    "wc",
    "tree",
    "stat",
    "env",
    "printenv",
    "date",
    "whoami",
    "ps",
    "top",
    "df",
    "du",
];

/// Build a compact memory note for a significant tool action, or None for
/// tools not worth remembering (reads, searches, navigation).
fn summarize_tool_action(payload: &serde_json::Value) -> Option<String> {
    let tool = payload.get("tool_name").and_then(|v| v.as_str())?;
    let input = payload.get("tool_input");
    let field = |k: &str| input.and_then(|i| i.get(k)).and_then(|v| v.as_str());

    match tool {
        "Write" | "Edit" | "MultiEdit" | "NotebookEdit" | "Update" => {
            let path = field("file_path").or_else(|| field("notebook_path"))?;
            Some(format!("Edited file: {path}"))
        }
        "Bash" => {
            let cmd = field("command")?.trim();
            if cmd.is_empty() {
                return None;
            }
            let lower = cmd.to_lowercase();
            if READ_ONLY_BASH.iter().any(|p| lower.starts_with(p)) {
                return None;
            }
            let compact: String = cmd.chars().take(200).collect();
            Some(format!("Ran command: {compact}"))
        }
        _ => None,
    }
}

const CONTEXT_CHAR_BUDGET: usize = 4_000;

/// Render backend context JSON ({memories: [...], pain: [...]}) into the text
/// injected into the model context. Returns None when there is nothing worth
/// injecting.
fn format_context(ctx: &serde_json::Value) -> Option<String> {
    let memories = ctx.get("memories").and_then(|v| v.as_array());
    let pains = ctx.get("pain").and_then(|v| v.as_array());

    let mut out = String::new();
    if let Some(memories) = memories {
        for m in memories {
            let content = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if content.is_empty() {
                continue;
            }
            let mtype = m
                .get("memory_type")
                .and_then(|v| v.as_str())
                .unwrap_or("memory");
            let line = format!("- [{}] {}\n", mtype.to_lowercase(), content);
            if out.len() + line.len() > CONTEXT_CHAR_BUDGET {
                break;
            }
            out.push_str(&line);
        }
    }

    let mut warnings = String::new();
    if let Some(pains) = pains {
        for p in pains {
            let desc = p.get("description").and_then(|v| v.as_str()).unwrap_or("");
            if desc.is_empty() {
                continue;
            }
            let line = format!("- WARNING (past failure): {desc}\n");
            if out.len() + warnings.len() + line.len() > CONTEXT_CHAR_BUDGET + 800 {
                break;
            }
            warnings.push_str(&line);
        }
    }

    if out.is_empty() && warnings.is_empty() {
        return None;
    }

    let mut text = String::from("Relevant memories from MenteDB (persistent memory):\n");
    text.push_str(&out);
    if !warnings.is_empty() {
        text.push_str(&warnings);
    }
    Some(text)
}

/// Render session-start JSON ({profile, always: [...]}) as plain stdout text.
fn format_session_context(ctx: &serde_json::Value) -> Option<String> {
    let profile = ctx.get("profile").and_then(|v| v.as_str()).unwrap_or("");
    let always: Vec<&str> = ctx
        .get("always")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    if profile.is_empty() && always.is_empty() {
        return None;
    }

    let mut text = String::from("MenteDB persistent memory for this user:\n");
    if !profile.is_empty() {
        text.push_str("## User profile\n");
        text.push_str(profile);
        text.push('\n');
    }
    if !always.is_empty() {
        text.push_str("## Standing rules (always apply)\n");
        for a in always.iter().take(30) {
            text.push_str(&format!("- {a}\n"));
        }
    }
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_context_renders_memories_and_pain() {
        let ctx = json!({
            "memories": [
                { "content": "User prefers Rust", "memory_type": "Semantic", "scope": "contextual" },
            ],
            "pain": [ { "description": "deploy without tests broke prod", "intensity": 0.9 } ],
        });
        let text = format_context(&ctx).unwrap();
        assert!(text.contains("[semantic] User prefers Rust"));
        assert!(text.contains("WARNING (past failure): deploy without tests broke prod"));
    }

    #[test]
    fn format_context_empty_returns_none() {
        assert!(format_context(&json!({ "memories": [], "pain": [] })).is_none());
        assert!(format_context(&json!({})).is_none());
    }

    #[test]
    fn format_context_respects_budget() {
        let big = "x".repeat(1_000);
        let memories: Vec<serde_json::Value> = (0..20)
            .map(|_| json!({ "content": big, "memory_type": "Semantic" }))
            .collect();
        let text = format_context(&json!({ "memories": memories })).unwrap();
        assert!(text.len() < CONTEXT_CHAR_BUDGET + 1_200);
    }

    #[test]
    fn session_state_roundtrip_and_turn_counter() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = load_state(dir.path(), "abc-123");
        assert_eq!(state.turn_id, 0);
        state.turn_id += 1;
        state.pending_prompt = Some("hello".into());
        save_state(dir.path(), "abc-123", &state);

        let loaded = load_state(dir.path(), "abc-123");
        assert_eq!(loaded.turn_id, 1);
        assert_eq!(loaded.pending_prompt.as_deref(), Some("hello"));
    }

    #[test]
    fn state_path_sanitizes_session_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = state_path(dir.path(), "../../etc/passwd");
        assert!(path.starts_with(dir.path().join("hook_sessions")));
        assert!(!path.to_string_lossy().contains(".."));
    }

    #[test]
    fn summarize_tool_action_captures_edits_and_commands() {
        let edit = json!({
            "tool_name": "Edit",
            "tool_input": { "file_path": "src/main.rs" },
        });
        assert_eq!(
            summarize_tool_action(&edit).as_deref(),
            Some("Edited file: src/main.rs")
        );

        let bash = json!({
            "tool_name": "Bash",
            "tool_input": { "command": "cargo test --workspace" },
        });
        assert_eq!(
            summarize_tool_action(&bash).as_deref(),
            Some("Ran command: cargo test --workspace")
        );
    }

    #[test]
    fn summarize_tool_action_skips_reads_and_unknown() {
        // Read-only bash is noise.
        for cmd in [
            "ls -la",
            "git status",
            "cat foo",
            "grep x .",
            "pwd",
            "cd /tmp",
        ] {
            let p = json!({ "tool_name": "Bash", "tool_input": { "command": cmd } });
            assert!(summarize_tool_action(&p).is_none(), "should skip: {cmd}");
        }
        // Read/search tools are not captured.
        for tool in ["Read", "Grep", "Glob", "WebFetch"] {
            let p = json!({ "tool_name": tool, "tool_input": {} });
            assert!(summarize_tool_action(&p).is_none(), "should skip: {tool}");
        }
        // Missing fields do not panic.
        assert!(summarize_tool_action(&json!({})).is_none());
        assert!(summarize_tool_action(&json!({ "tool_name": "Bash" })).is_none());
    }

    #[test]
    fn format_session_context_renders_profile_and_rules() {
        let ctx = json!({
            "profile": "Works on trading systems",
            "always": ["never commit secrets"],
        });
        let text = format_session_context(&ctx).unwrap();
        assert!(text.contains("Works on trading systems"));
        assert!(text.contains("never commit secrets"));
        assert!(format_session_context(&json!({})).is_none());
    }
}
