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

pub(crate) mod backend;
mod redact;
pub(crate) mod spool;

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::json;

use backend::Backend;
use redact::redact;

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
    /// Previous turn's user message, blended into the recall query so short
    /// follow-up prompts still retrieve on-topic memories.
    #[serde(default)]
    last_user_message: Option<String>,
    /// Memory IDs already injected this session: working memory for the
    /// injector, so the same fact is never re-told turn after turn.
    #[serde(default)]
    injected: Vec<String>,
    /// Memory IDs injected on the most recent prompt, awaiting outcome
    /// reporting when the turn's reply arrives at Stop.
    #[serde(default)]
    last_injected: Vec<String>,
    /// Last stored action note, to collapse consecutive duplicates
    /// (iterating on one file should not produce N identical memories).
    #[serde(default)]
    last_note: Option<String>,
}

/// Claude Code injects background task notifications, system reminders, and
/// local command output as UserPromptSubmit events. They are not user turns, so
/// storing them pollutes memory and the speculative cache with system noise.
/// Remove those blocks; a prompt left empty was pure noise and the caller skips
/// it. Real prompts that merely have a reminder appended keep their content.
fn strip_system_blocks(prompt: &str) -> String {
    static SYSTEM_BLOCK: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(
            r"(?s)<task-notification>.*?</task-notification>|<system-reminder>.*?</system-reminder>|<local-command-stdout>.*?</local-command-stdout>|<local-command-stderr>.*?</local-command-stderr>",
        )
        .expect("valid system-block regex")
    });
    SYSTEM_BLOCK.replace_all(prompt, "").trim().to_string()
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
            // Claude Code injects background task notifications, system
            // reminders, and local command output as UserPromptSubmit events.
            // They are not user turns, so storing them pollutes memory and the
            // speculative cache with system noise. Strip those blocks, and if
            // nothing real is left, skip the turn entirely.
            let prompt = strip_system_blocks(&prompt);
            if prompt.is_empty() {
                return Ok(());
            }
            let prompt = redact(&prompt);

            // Stash the prompt so the Stop hook can store the complete turn.
            let mut state = load_state(data_dir, &session_id);
            state.pending_prompt = Some(prompt.chars().take(8_000).collect());
            save_state(data_dir, &session_id, &state);

            // Short follow-ups ("do it", "and tests?") retrieve nothing on
            // their own; blend in the previous turn for topical grounding.
            let query = match state.last_user_message.as_deref() {
                Some(last) if prompt.chars().count() < 200 => {
                    let last: String = last.chars().take(600).collect();
                    format!("{prompt}\n\n[previous turn] {last}")
                }
                _ => prompt.clone(),
            };

            let backend = Backend::resolve(data_dir, force_local).await?;

            // Native path: the engine owns selection (session exclusion,
            // ledger, knee, MMR, quotas, pinned bypass). Fallback: raw
            // recall shaped by the local heuristic filter, for backends
            // that predate the injection API.
            let native = backend
                .injection_context(&query, &session_id, &state.injected)
                .await;
            let (ctx, injected_ids) = match native {
                Some(ctx) => {
                    record_auth_state(data_dir, None);
                    let ids: Vec<String> = ctx["memories"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|m| m.get("id").and_then(|v| v.as_str()))
                                .map(str::to_string)
                                .collect()
                        })
                        .unwrap_or_default();
                    (Some(ctx), ids)
                }
                None => match backend.context(&query).await {
                    Ok(c) => {
                        record_auth_state(data_dir, None);
                        let (filtered, ids) =
                            filter_for_injection(&c, &state.injected, now_micros());
                        (Some(filtered), ids)
                    }
                    Err(e) => {
                        record_auth_state(data_dir, Some(&e));
                        tracing::warn!(error = %e, "context recall failed");
                        (None, Vec::new())
                    }
                },
            };

            let mut text = String::new();
            if let Some(ctx) = &ctx {
                if !injected_ids.is_empty() {
                    let mut state = load_state(data_dir, &session_id);
                    state.injected.extend(injected_ids.clone());
                    // Bounded ledger: ancient sessions can re-surface things.
                    let excess = state.injected.len().saturating_sub(300);
                    if excess > 0 {
                        state.injected.drain(..excess);
                    }
                    state.last_injected = injected_ids;
                    save_state(data_dir, &session_id, &state);
                }
                text = format_context(ctx).unwrap_or_default();
            }
            if let Some(notice) = auth_notice(data_dir) {
                text = if text.is_empty() {
                    notice
                } else {
                    format!("{notice}\n\n{text}")
                };
            }
            // Surface a stalled write spool in the session itself, so a sync
            // outage is visible immediately instead of silently dropping turns
            // for hours. Prepended above any recalled context so it reads first.
            if let Some(warning) = spool_notice(data_dir) {
                text = if text.is_empty() {
                    warning
                } else {
                    format!("{warning}\n\n{text}")
                };
            }
            if !text.is_empty() {
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
            // The user message carries the signal; the assistant reply is
            // mostly restatement. 2k chars is plenty for distillation and
            // stops verbatim transcripts bloating the store.
            let assistant: String = payload
                .get("last_assistant_message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .chars()
                .take(2_000)
                .collect();
            let assistant = redact(&assistant);

            let mut state = load_state(data_dir, &session_id);
            let Some(user_message) = state.pending_prompt.take() else {
                return Ok(());
            };
            // turn_id starts at 1: the engine treats turn 0 as "no turn" and
            // skips periodic maintenance for it.
            state.turn_id += 1;
            let turn_id = state.turn_id;
            state.last_user_message = Some(user_message.chars().take(600).collect());
            save_state(data_dir, &session_id, &state);

            let project = payload
                .get("cwd")
                .and_then(|v| v.as_str())
                .and_then(|c| Path::new(c).file_name())
                .and_then(|n| n.to_str())
                .map(str::to_string);

            let outcome_ids = std::mem::take(&mut state.last_injected);
            save_state(data_dir, &session_id, &state);

            // Spool the turn (and attention outcome) first so they are durable
            // regardless of the network, then flush synchronously. Each send is
            // bounded by a tight per-send timeout inside flush_spool, so the turn
            // end can never stall on a slow or unreachable Cloud (previously up to
            // the 30s client timeout, worse when flaky). Anything that does not
            // send in time stays spooled and the next hook flushes it, so nothing
            // is lost.
            spool::push(
                data_dir,
                &json!({
                    "kind": "turn",
                    "user_message": user_message,
                    "assistant_response": assistant.clone(),
                    "turn_id": turn_id,
                    "project": project,
                    "session_id": session_id,
                }),
            );
            if !outcome_ids.is_empty() {
                spool::push(
                    data_dir,
                    &json!({
                        "kind": "injection_outcome",
                        "shown_ids": outcome_ids,
                        "assistant_text": assistant,
                    }),
                );
            }
            if let Ok(backend) = Backend::resolve(data_dir, force_local).await {
                flush_spool(data_dir, &backend).await;
                record_auth_state(data_dir, None);
            }
        }
        HookEvent::SessionStart => {
            let backend = Backend::resolve(data_dir, force_local).await?;
            let ctx = match backend.session_context().await {
                Ok(c) => {
                    record_auth_state(data_dir, None);
                    Some(c)
                }
                Err(e) => {
                    record_auth_state(data_dir, Some(&e));
                    tracing::warn!(error = %e, "session context failed");
                    None
                }
            };
            let mut text = ctx
                .as_ref()
                .and_then(format_session_context)
                .unwrap_or_default();
            if let Some(notice) = auth_notice(data_dir) {
                text = if text.is_empty() {
                    notice
                } else {
                    format!("{notice}\n\n{text}")
                };
            }
            if !text.is_empty() {
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
            let note = redact(&note);

            // Iterating on one file fires this hook repeatedly with the
            // identical note; one memory carries all the information.
            let mut state = load_state(data_dir, &session_id);
            if state.last_note.as_deref() == Some(note.as_str()) {
                return Ok(());
            }
            state.last_note = Some(note.clone());
            save_state(data_dir, &session_id, &state);

            let project = payload
                .get("cwd")
                .and_then(|v| v.as_str())
                .and_then(|c| Path::new(c).file_name())
                .and_then(|n| n.to_str())
                .map(str::to_string);
            let store = async {
                let backend = Backend::resolve(data_dir, force_local).await?;
                flush_spool(data_dir, &backend).await;
                backend.store_note(&note, project.clone()).await
            }
            .await;
            if let Err(e) = store {
                tracing::warn!(error = %e, "store_note failed, spooling for retry");
                spool::push(
                    data_dir,
                    &json!({ "kind": "note", "content": note, "project": project }),
                );
            }
        }
        HookEvent::PreCompact => {
            // The injection ledger mirrors what is in the model's context;
            // compaction destroys that context, so the mirror must reset or
            // it would block re-injection of memories the model just lost.
            let mut state = load_state(data_dir, &session_id);
            if !state.injected.is_empty() {
                state.injected.clear();
                save_state(data_dir, &session_id, &state);
            }

            // The long session is about to lose context; make sure everything
            // captured so far is durable.
            let backend = Backend::resolve(data_dir, force_local).await?;
            backend.flush().await?;
        }
    }
    Ok(())
}

/// Track whether the last cloud call authenticated. Hooks must never break
/// the client, so a revoked or expired token would otherwise fail silently
/// forever; the marker lets context injection tell the user to log in again.
fn record_auth_state(data_dir: &Path, err: Option<&anyhow::Error>) {
    let path = data_dir.join("auth_error");
    match err {
        Some(e) if e.to_string().contains("authentication failed") => {
            std::fs::write(&path, "revoked").ok();
        }
        // Transient failures (network, 5xx) say nothing about the token.
        Some(_) => {}
        None => {
            std::fs::remove_file(&path).ok();
        }
    }
}

fn auth_notice(data_dir: &Path) -> Option<String> {
    data_dir.join("auth_error").exists().then(|| {
        "MenteDB: the cloud session is no longer valid (token expired or revoked). \
         Tell the user to run `npx mentedb-mcp@latest login` to reconnect. \
         New memories are spooling locally and will sync automatically after login."
            .to_string()
    })
}

/// When the local write spool has backed up past a small threshold, the cloud
/// is not accepting writes and recent turns are not being remembered. Surface
/// that in the session so a sync outage never stays silent for hours; the hook
/// keeps retrying on its own and the notice clears once the backlog drains.
fn spool_notice(data_dir: &Path) -> Option<String> {
    // A retry or two is normal and self-heals within a prompt, so only warn once
    // a real backlog has formed and this never cries wolf on a transient blip.
    const SPOOL_WARN_THRESHOLD: usize = 10;
    let queued = spool::depth(data_dir);
    (queued >= SPOOL_WARN_THRESHOLD).then(|| {
        format!(
            "MenteDB sync warning: {queued} memories are queued locally and have not synced \
             to the cloud, so recent turns from this session may not be remembered yet. The \
             hook keeps retrying automatically; if this persists, check \
             ~/.mentedb/mentedb-hook.log."
        )
    })
}

/// Retry every spooled entry against the now-reachable backend. Entries that
/// still fail (and everything after the first failure, to preserve order) go
/// back into the spool.
async fn flush_spool(data_dir: &Path, backend: &Backend) {
    // The flush is bounded so a backlog never dominates a hook: at most
    // MAX_FLUSH_PER_CALL entries and at most FLUSH_BUDGET of wall-clock per
    // invocation. Whatever is not reached stays spooled for the next hook, so a
    // large backlog drains over several hooks instead of blocking one prompt for
    // tens of seconds (each store is a full round trip; N of them was the hang).
    const MAX_FLUSH_PER_CALL: usize = 25;
    const FLUSH_BUDGET: std::time::Duration = std::time::Duration::from_secs(6);

    let entries = spool::take_all(data_dir);
    if entries.is_empty() {
        return;
    }
    let start = std::time::Instant::now();
    let mut processed = 0usize;
    let mut failed: Vec<serde_json::Value> = Vec::new();
    let mut hit_failure = false;
    for e in entries {
        // Stop making network calls once we fail, reach the item cap, or exceed
        // the time budget; everything remaining is requeued in order.
        if hit_failure || processed >= MAX_FLUSH_PER_CALL || start.elapsed() >= FLUSH_BUDGET {
            failed.push(e);
            continue;
        }
        processed += 1;
        let ok = match e.get("kind").and_then(|v| v.as_str()) {
            Some("turn") => {
                let user = e.get("user_message").and_then(|v| v.as_str()).unwrap_or("");
                if user.is_empty() {
                    true // malformed, drop
                } else {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(3),
                        backend.store_turn(
                            user,
                            e.get("assistant_response")
                                .and_then(|v| v.as_str())
                                .unwrap_or(""),
                            e.get("turn_id").and_then(|v| v.as_u64()).unwrap_or(1),
                            e.get("project")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            e.get("session_id").and_then(|v| v.as_str()).unwrap_or(""),
                        ),
                    )
                    .await
                    .map(|r| r.is_ok())
                    .unwrap_or(false)
                }
            }
            Some("note") => {
                let content = e.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if content.is_empty() {
                    true
                } else {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(3),
                        backend.store_note(
                            content,
                            e.get("project")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                        ),
                    )
                    .await
                    .map(|r| r.is_ok())
                    .unwrap_or(false)
                }
            }
            Some("injection_outcome") => {
                let shown: Vec<String> = e
                    .get("shown_ids")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let text = e
                    .get("assistant_text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                // Best effort attention signal; bounded so it never blocks the
                // queue, and always dropped (it is non-critical, never requeued).
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    backend.record_injection_outcome(&shown, text),
                )
                .await;
                true
            }
            _ => true,
        };
        if !ok {
            hit_failure = true;
            failed.push(e);
        }
    }
    if failed.is_empty() {
        tracing::info!("offline spool fully flushed");
    } else {
        tracing::warn!(retained = failed.len(), "spool flush incomplete");
        spool::restore(data_dir, &failed);
    }
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

/// Raw turns younger than this are either still in the model's context or
/// fresh in the user's head; injecting them is echo, not memory. Distilled
/// facts take over once the nightly sweep has run.
const FRESH_TURN_HORIZON_US: u64 = 12 * 3600 * 1_000_000;
/// Keep an item only if it scores at least this fraction of the top hit,
/// scale-free so it works for any backend's score range.
const RELATIVE_SCORE_FLOOR: f64 = 0.4;
/// Verbatim episodic memories allowed per injection; semantic knowledge
/// gets the rest of the slots.
const MAX_EPISODIC_INJECTED: usize = 2;
const MAX_INJECTED: usize = 6;
/// Word-overlap ratio above which two candidates are the same information
/// (a distilled fact and its source turn embed almost identically).
const NEAR_DUP_JACCARD: f64 = 0.7;

fn type_rank(memory_type: &str) -> usize {
    match memory_type.to_lowercase().as_str() {
        "anti_pattern" | "antipattern" => 0,
        "correction" => 1,
        "semantic" => 2,
        "procedural" => 3,
        "reasoning" => 4,
        _ => 5, // episodic and unknown last
    }
}

fn word_set(text: &str) -> std::collections::HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_string)
        .collect()
}

fn jaccard(a: &std::collections::HashSet<String>, b: &std::collections::HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = (a.len() + b.len()) as f64 - inter;
    inter / union
}

fn has_tag(m: &serde_json::Value, tag: &str) -> bool {
    m.get("tags")
        .and_then(|v| v.as_array())
        .is_some_and(|arr| arr.iter().any(|v| v.as_str() == Some(tag)))
}

/// The injection policy: storage is a library, this is the librarian.
///
/// Filters the recalled memories down to what is actually worth the model's
/// attention right now: nothing already injected this session, no action
/// notes (they exist for distillation and resume, not topical recall), no
/// raw turns fresh enough to still be in context, nothing far below the top
/// relevance score, no near-duplicates of each other, semantic knowledge
/// before verbatim episodic, and a hard cap. Returns the filtered context
/// and the IDs selected, for the session's working-memory ledger.
///
/// User-pinned scope:always memories bypass every quality filter: the user
/// said always, so they are delivered once per context lifetime (the ledger
/// resets at compaction, when the model's context is destroyed).
fn filter_for_injection(
    ctx: &serde_json::Value,
    already_injected: &[String],
    now_us: u64,
) -> (serde_json::Value, Vec<String>) {
    let Some(memories) = ctx.get("memories").and_then(|v| v.as_array()) else {
        return (ctx.clone(), Vec::new());
    };

    let top_score = memories
        .iter()
        .filter_map(|m| m.get("score").and_then(|s| s.as_f64()))
        .fold(f64::NEG_INFINITY, f64::max);

    let not_yet_injected = |m: &serde_json::Value| {
        let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
        !id.is_empty() && !already_injected.iter().any(|i| i == id)
    };

    // Pinned memories skip every quality filter except the ledger.
    let pinned: Vec<&serde_json::Value> = memories
        .iter()
        .filter(|m| has_tag(m, "scope:always") && not_yet_injected(m))
        .collect();

    let mut candidates: Vec<&serde_json::Value> = memories
        .iter()
        .filter(|m| !has_tag(m, "scope:always") && not_yet_injected(m))
        .filter(|m| {
            if has_tag(m, "action") {
                return false;
            }
            if has_tag(m, "turn") {
                let created = m
                    .get("created_at")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<u64>().ok());
                if let Some(created) = created
                    && now_us.saturating_sub(created) < FRESH_TURN_HORIZON_US
                {
                    return false;
                }
            }
            true
        })
        .filter(|m| {
            match (
                m.get("score").and_then(|s| s.as_f64()),
                top_score.is_finite(),
            ) {
                (Some(s), true) => s >= top_score * RELATIVE_SCORE_FLOOR,
                _ => true, // no scores plumbed: keep
            }
        })
        .collect();

    // Semantic knowledge first, verbatim episodic last.
    candidates
        .sort_by_key(|m| type_rank(m.get("memory_type").and_then(|v| v.as_str()).unwrap_or("")));

    let mut selected: Vec<&serde_json::Value> = pinned;
    let mut selected_words: Vec<std::collections::HashSet<String>> = Vec::new();
    let mut episodic_count = 0usize;
    let budget = MAX_INJECTED + selected.len();

    for m in candidates {
        if selected.len() >= budget {
            break;
        }
        let mtype = m.get("memory_type").and_then(|v| v.as_str()).unwrap_or("");
        let is_episodic = mtype.eq_ignore_ascii_case("episodic");
        if is_episodic && episodic_count >= MAX_EPISODIC_INJECTED {
            continue;
        }
        let content = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let words = word_set(content);
        if selected_words
            .iter()
            .any(|w| jaccard(w, &words) > NEAR_DUP_JACCARD)
        {
            continue;
        }
        if is_episodic {
            episodic_count += 1;
        }
        selected_words.push(words);
        selected.push(m);
    }

    let ids: Vec<String> = selected
        .iter()
        .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(str::to_string))
        .collect();

    let mut filtered = ctx.clone();
    filtered["memories"] = serde_json::json!(selected);
    (filtered, ids)
}

fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

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

    // Plan limit notices ride the injection payload because it reaches the
    // assistant on every prompt; without this the limit was invisible and
    // memory just silently stopped storing.
    let limit_notice = ctx.get("limit_notice").and_then(|v| v.as_str());

    if out.is_empty() && warnings.is_empty() && limit_notice.is_none() {
        return None;
    }

    let mut text = String::new();
    if let Some(notice) = limit_notice {
        text.push_str(&format!(
            "[MenteDB] {notice} Tell the user if they ask about memory.\n\n"
        ));
    }
    text.push_str("Relevant memories from MenteDB (persistent memory):\n");
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
    let update_msg = ctx
        .get("client_update")
        .and_then(|u| u.get("message"))
        .and_then(|v| v.as_str());

    if profile.is_empty() && always.is_empty() && update_msg.is_none() {
        return None;
    }

    let mut text = String::new();
    if let Some(msg) = update_msg {
        // Surfaced inside the assistant's context so the user is told to
        // update without ever checking the dashboard.
        text.push_str(&format!("[MenteDB] {msg}\n\n"));
    }
    text.push_str("MenteDB persistent memory for this user:\n");
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

    fn mem(
        id: &str,
        mtype: &str,
        content: &str,
        tags: &[&str],
        age_us: u64,
        score: f64,
    ) -> serde_json::Value {
        let created = 2_000_000_000_000_000u64 - age_us;
        json!({
            "id": id,
            "memory_type": mtype,
            "content": content,
            "tags": tags,
            "created_at": created.to_string(),
            "score": score,
        })
    }

    const NOW: u64 = 2_000_000_000_000_000;

    #[test]
    fn injection_skips_already_injected_and_actions() {
        let ctx = json!({ "memories": [
            mem("a", "semantic", "user prefers rust", &[], 100, 1.0),
            mem("b", "episodic", "Edited file: src/main.rs", &["action"], 100, 0.9),
            mem("c", "semantic", "deploys use v tags", &[], 100, 0.9),
        ], "pain": [] });
        let (filtered, ids) = filter_for_injection(&ctx, &["a".to_string()], NOW);
        assert_eq!(ids, vec!["c".to_string()]);
        assert_eq!(filtered["memories"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn injection_drops_fresh_turns_keeps_old_ones() {
        let hour = 3_600_000_000u64;
        let ctx = json!({ "memories": [
            mem("fresh", "episodic", "User: hi there Assistant: hello friend", &["turn"], hour, 1.0),
            mem("old", "episodic", "User: deploy plan? Assistant: use tags", &["turn"], 48 * hour, 0.9),
        ], "pain": [] });
        let (_, ids) = filter_for_injection(&ctx, &[], NOW);
        assert_eq!(ids, vec!["old".to_string()]);
    }

    #[test]
    fn injection_applies_relative_score_floor_and_caps() {
        let mut memories: Vec<serde_json::Value> = vec![mem(
            "weak",
            "semantic",
            "barely related trivia entry",
            &[],
            100,
            0.1,
        )];
        let facts = [
            "prefers rust for backend services",
            "uses postgres with logical replication",
            "deploys ride fargate behind cloudfront",
            "billing runs through stripe webhooks",
            "embeddings come from titan on bedrock",
            "frontend built with vite and tailwind",
            "auth handled by cognito user pools",
            "monitoring lives in cloudwatch dashboards",
            "queue processing goes through sqs lambda",
            "dns zones managed inside route fiftythree",
        ];
        for (i, fact) in facts.iter().enumerate() {
            memories.push(mem(&format!("s{i}"), "semantic", fact, &[], 100, 1.0));
        }
        let ctx = json!({ "memories": memories, "pain": [] });
        let (_, ids) = filter_for_injection(&ctx, &[], NOW);
        assert!(
            !ids.contains(&"weak".to_string()),
            "sub-floor item injected"
        );
        assert_eq!(ids.len(), MAX_INJECTED);
    }

    #[test]
    fn injection_prefers_semantic_and_dedups_near_duplicates() {
        let ctx = json!({ "memories": [
            mem("turn1", "episodic", "User: remember the deploy uses v tags on the platform repo", &["turn"], 100_000_000_000, 1.0),
            mem("fact1", "semantic", "the deploy uses v tags on the platform repo", &[], 100, 0.95),
        ], "pain": [] });
        let (_, ids) = filter_for_injection(&ctx, &[], NOW);
        // The distilled fact wins; its verbatim source is a near-duplicate.
        assert_eq!(ids, vec!["fact1".to_string()]);
    }

    #[test]
    fn pinned_always_memories_bypass_all_quality_filters() {
        let hour = 3_600_000_000u64;
        let ctx = json!({ "memories": [
            // A pinned memory that would fail every filter: fresh turn tag,
            // rock-bottom score.
            mem("pin", "episodic", "never deploy on fridays", &["scope:always", "turn"], hour, 0.01),
            mem("top", "semantic", "prefers rust services", &[], 100, 1.0),
        ], "pain": [] });
        let (_, ids) = filter_for_injection(&ctx, &[], NOW);
        assert!(
            ids.contains(&"pin".to_string()),
            "pinned memory must inject"
        );
        assert!(ids.contains(&"top".to_string()));

        // The ledger still applies: once delivered, not repeated within the
        // same context lifetime.
        let (_, ids2) = filter_for_injection(&ctx, &ids, NOW);
        assert!(ids2.is_empty());
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

    #[test]
    fn strips_system_noise_and_keeps_real_prompts() {
        // A pure task notification collapses to empty, so the caller skips it.
        assert_eq!(
            strip_system_blocks(
                "<task-notification>\n<task-id>abc</task-id>\ndone\n</task-notification>"
            ),
            ""
        );
        // A real prompt with an appended system reminder keeps the real text.
        let mixed = "fix the login bug\n<system-reminder>be concise</system-reminder>";
        assert_eq!(strip_system_blocks(mixed), "fix the login bug");
        // An ordinary prompt passes through untouched.
        assert_eq!(
            strip_system_blocks("just a normal question"),
            "just a normal question"
        );
    }
}
