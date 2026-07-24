//! End-to-end tests for the lifecycle hook integration: daemon HTTP surface,
//! hook binary stdin/stdout contract, and the full prompt -> stop -> recall
//! loop against a real embedded database.
#![cfg(feature = "local")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_mentedb-mcp");

#[derive(serde::Deserialize)]
struct DaemonInfo {
    port: u16,
    pid: u32,
    token: String,
}

/// Kills the daemon (spawned directly or by hook auto-spawn) on drop.
struct DaemonGuard {
    child: Option<Child>,
    data_dir: PathBuf,
}

impl Drop for DaemonGuard {
    fn drop(&mut self) {
        if let Some(info) = read_info(&self.data_dir) {
            let _ = Command::new("kill").arg(info.pid.to_string()).status();
        }
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn read_info(data_dir: &Path) -> Option<DaemonInfo> {
    let raw = std::fs::read_to_string(data_dir.join("daemon.json")).ok()?;
    serde_json::from_str(&raw).ok()
}

fn wait_for_daemon(data_dir: &Path, timeout: Duration) -> DaemonInfo {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(info) = read_info(data_dir) {
            let health = ureq_get(&format!("http://127.0.0.1:{}/health", info.port));
            if health.is_some() {
                return info;
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    panic!("daemon did not become healthy within {timeout:?}");
}

fn ureq_get(url: &str) -> Option<String> {
    let output = Command::new("curl")
        .args(["-s", "-f", "-m", "2", url])
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        None
    }
}

fn daemon_post(info: &DaemonInfo, path: &str, body: &serde_json::Value) -> (u32, String) {
    let output = Command::new("curl")
        .args([
            "-s",
            "-m",
            "30",
            "-o",
            "-",
            "-w",
            "\n%{http_code}",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-H",
            &format!("x-mentedb-token: {}", info.token),
            "-d",
            &body.to_string(),
            &format!("http://127.0.0.1:{}{}", info.port, path),
        ])
        .output()
        .expect("curl failed");
    let raw = String::from_utf8_lossy(&output.stdout).to_string();
    let (body, code) = raw.rsplit_once('\n').unwrap_or(("", "0"));
    (code.trim().parse().unwrap_or(0), body.to_string())
}

fn run_hook(data_dir: &Path, event: &str, payload: &serde_json::Value) -> String {
    let mut child = Command::new(BIN)
        .args([
            "--data-dir",
            data_dir.to_str().unwrap(),
            "--local",
            "hook",
            event,
        ])
        // Keep tests hermetic: never let the hook reconcile the developer's
        // real Claude Code settings; the self-update path has its own test
        // with an isolated home.
        .env("MENTEDB_HOOK_NO_SELF_UPDATE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to run hook");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(payload.to_string().as_bytes())
        .unwrap();
    let output = child.wait_with_output().expect("hook did not exit");
    assert!(
        output.status.success(),
        "hook must always exit 0, got {:?}",
        output.status
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn spawn_daemon(data_dir: &Path) -> DaemonGuard {
    let child = Command::new(BIN)
        .args(["--data-dir", data_dir.to_str().unwrap(), "daemon"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn daemon");
    DaemonGuard {
        child: Some(child),
        data_dir: data_dir.to_path_buf(),
    }
}

#[test]
fn daemon_serves_turn_and_context_with_auth() {
    let dir = tempfile::tempdir().unwrap();
    let _guard = spawn_daemon(dir.path());
    let info = wait_for_daemon(dir.path(), Duration::from_secs(120));

    // Unauthorized without the token.
    let output = Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-d",
            "{\"prompt\":\"x\"}",
            &format!("http://127.0.0.1:{}/v1/context", info.port),
        ])
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&output.stdout), "401");

    // Store a turn through the full pipeline.
    let (code, body) = daemon_post(
        &info,
        "/v1/turn",
        &serde_json::json!({
            "user_message": "I always deploy with blue green releases on Fridays",
            "assistant_response": "Noted, blue green on Fridays.",
            "turn_id": 1,
        }),
    );
    assert_eq!(code, 200, "turn failed: {body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["ok"], true);

    // Recall it.
    let (code, body) = daemon_post(
        &info,
        "/v1/context",
        &serde_json::json!({ "prompt": "how do I deploy releases" }),
    );
    assert_eq!(code, 200);
    let ctx: serde_json::Value = serde_json::from_str(&body).unwrap();
    let memories = ctx["memories"].as_array().unwrap();
    assert!(
        memories
            .iter()
            .any(|m| m["content"].as_str().unwrap_or("").contains("blue green")),
        "stored turn must be recallable, got: {ctx}"
    );
}

#[test]
fn hook_full_turn_loop_with_autospawn() {
    let dir = tempfile::tempdir().unwrap();
    // No daemon started: the first hook call must auto-spawn it.
    let guard = DaemonGuard {
        child: None,
        data_dir: dir.path().to_path_buf(),
    };

    let session = "sess-hook-e2e";

    // Turn 1: user prompt (stashes prompt, may return no context on an empty DB).
    let _ = run_hook(
        dir.path(),
        "user-prompt",
        &serde_json::json!({
            "session_id": session,
            "prompt": "remember that my project codename is nightjar",
            "hook_event_name": "UserPromptSubmit",
        }),
    );
    // The auto-spawned daemon may still be loading on the very first call;
    // wait until it registers before the stop hook.
    let _info = wait_for_daemon(dir.path(), Duration::from_secs(120));

    // Retry the prompt hook now that the daemon is up (first call may have
    // timed out waiting on model download).
    let _ = run_hook(
        dir.path(),
        "user-prompt",
        &serde_json::json!({
            "session_id": session,
            "prompt": "remember that my project codename is nightjar",
            "hook_event_name": "UserPromptSubmit",
        }),
    );

    // Stop hook: stores the turn.
    let out = run_hook(
        dir.path(),
        "stop",
        &serde_json::json!({
            "session_id": session,
            "last_assistant_message": "Got it, codename nightjar.",
            "hook_event_name": "Stop",
        }),
    );
    assert!(out.trim().is_empty(), "stop hook must print nothing: {out}");

    // Session state advanced.
    let state_raw =
        std::fs::read_to_string(dir.path().join(format!("hook_sessions/{session}.json"))).unwrap();
    let state: serde_json::Value = serde_json::from_str(&state_raw).unwrap();
    assert_eq!(state["turn_id"], 1);
    assert!(state["pending_prompt"].is_null());

    // Turn 2: the injection policy deliberately suppresses turns this fresh
    // (they are still in the model's context window), so the prompt hook
    // must print nothing.
    let out = run_hook(
        dir.path(),
        "user-prompt",
        &serde_json::json!({
            "session_id": session,
            "prompt": "what is my project codename",
            "hook_event_name": "UserPromptSubmit",
        }),
    );
    assert!(
        out.trim().is_empty(),
        "fresh same-session turns are echo and must not be injected, got: {out}"
    );

    // The turn is stored asynchronously now: the Stop hook spools it and a
    // detached flusher sends it to the daemon, so the turn end never blocks on
    // the write. Poll until it is retrievable (eventual consistency), below the
    // injection policy.
    let info = wait_for_daemon(dir.path(), Duration::from_secs(30));
    let mut recalled = serde_json::Value::Null;
    let mut found = false;
    for _ in 0..40 {
        let (code, body) = daemon_post(
            &info,
            "/v1/context",
            &serde_json::json!({ "prompt": "what is my project codename", "limit": 8 }),
        );
        if code == 200 {
            recalled = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
            let memories = recalled["memories"].as_array().cloned().unwrap_or_default();
            if memories.iter().any(|m| {
                m["content"]
                    .as_str()
                    .is_some_and(|c| c.contains("nightjar"))
            }) {
                found = true;
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    assert!(
        found,
        "stored turn must eventually be retrievable from the daemon, last: {recalled}"
    );

    drop(guard);
}

#[test]
fn post_tool_use_captures_action_live() {
    let dir = tempfile::tempdir().unwrap();
    let _guard = spawn_daemon(dir.path());
    let info = wait_for_daemon(dir.path(), Duration::from_secs(120));

    // A PostToolUse hook for a file edit stores an action note immediately,
    // with no prior user-prompt/stop turn.
    let out = run_hook(
        dir.path(),
        "post-tool-use",
        &serde_json::json!({
            "session_id": "sess-ptu",
            "tool_name": "Edit",
            "tool_input": { "file_path": "src/payments.rs" },
            "cwd": "/tmp/myproj",
            "hook_event_name": "PostToolUse",
        }),
    );
    assert!(out.trim().is_empty(), "post-tool-use prints nothing: {out}");

    // The action is immediately recallable (flushed on write).
    let (code, body) = daemon_post(
        &info,
        "/v1/context",
        &serde_json::json!({ "prompt": "what files were changed for payments" }),
    );
    assert_eq!(code, 200);
    let ctx: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        ctx["memories"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m["content"].as_str().unwrap_or("").contains("payments.rs")),
        "captured action must be recallable, got: {ctx}"
    );

    // A read-only command is not captured (noise filter).
    run_hook(
        dir.path(),
        "post-tool-use",
        &serde_json::json!({
            "session_id": "sess-ptu",
            "tool_name": "Bash",
            "tool_input": { "command": "git status" },
            "hook_event_name": "PostToolUse",
        }),
    );

    // PreCompact flush never errors.
    let out = run_hook(
        dir.path(),
        "pre-compact",
        &serde_json::json!({ "session_id": "sess-ptu", "trigger": "auto" }),
    );
    assert!(out.trim().is_empty());
}

#[test]
fn hook_tolerates_garbage_input() {
    let dir = tempfile::tempdir().unwrap();
    // Malformed JSON, no daemon, no cloud: must still exit 0 with no output.
    let mut child = Command::new(BIN)
        .args([
            "--data-dir",
            dir.path().to_str().unwrap(),
            "--local",
            "hook",
            "stop",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"this is not json")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn pre_tool_use_injects_action_rules_before_commit() {
    use mentedb::prelude::*;
    use mentedb_core::types::AgentId;

    let dir = tempfile::tempdir().unwrap();

    // Seed action rules directly in the engine, then close it so the daemon
    // (single writer) can open the same directory.
    {
        let db = mentedb::MenteDb::open(dir.path()).unwrap();
        let mut commit_rule = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Procedural,
            "never add Co-Authored-By trailers to commits".to_string(),
            vec![],
        );
        commit_rule.tags = vec!["trigger:git-commit".to_string()];
        db.store(commit_rule).unwrap();

        let mut pr_rule = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Procedural,
            "PR descriptions use Summary and Verification sections".to_string(),
            vec![],
        );
        pr_rule.tags = vec!["trigger:pr-create".to_string()];
        db.store(pr_rule).unwrap();

        let ordinary = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            "the user prefers dark mode".to_string(),
            vec![],
        );
        db.store(ordinary).unwrap();
        // Persist the indexes (the tag bitmap is written by close, not drop)
        // so the daemon reopening this directory sees the trigger index.
        db.close().unwrap();
    }

    let _guard = spawn_daemon(dir.path());
    wait_for_daemon(dir.path(), Duration::from_secs(120));

    // The exact command shape agents run, global flag value containing the
    // word commit included.
    let out = run_hook(
        dir.path(),
        "pre-tool-use",
        &serde_json::json!({
            "session_id": "sess-pre",
            "tool_name": "Bash",
            "tool_input": { "command": "git -c commit.gpgsign=false commit -m \"fix: x\"" },
            "cwd": "/tmp/myproj",
            "hook_event_name": "PreToolUse",
        }),
    );
    let parsed: serde_json::Value =
        serde_json::from_str(out.trim()).expect("pre-tool-use must print valid JSON");
    let hso = &parsed["hookSpecificOutput"];
    assert_eq!(hso["hookEventName"], "PreToolUse");
    let ctx = hso["additionalContext"].as_str().unwrap_or_default();
    assert!(
        ctx.contains("never add Co-Authored-By trailers"),
        "commit rule must be injected, got: {ctx}"
    );
    assert!(
        !ctx.contains("Summary and Verification"),
        "pr-create rule must not fire on a commit, got: {ctx}"
    );
    assert!(
        !ctx.contains("dark mode"),
        "ordinary memories must not enter the action channel, got: {ctx}"
    );
    assert!(
        hso.get("permissionDecision").is_none(),
        "the hook must never emit a permission decision"
    );

    // Non-commit git commands stay silent.
    let out = run_hook(
        dir.path(),
        "pre-tool-use",
        &serde_json::json!({
            "session_id": "sess-pre",
            "tool_name": "Bash",
            "tool_input": { "command": "git log --oneline -3" },
            "hook_event_name": "PreToolUse",
        }),
    );
    assert!(
        out.trim().is_empty(),
        "git log must not trigger rules: {out}"
    );

    // Non-Bash tools stay silent even if their input mentions git.
    let out = run_hook(
        dir.path(),
        "pre-tool-use",
        &serde_json::json!({
            "session_id": "sess-pre",
            "tool_name": "Edit",
            "tool_input": { "file_path": "git_commit.rs" },
            "hook_event_name": "PreToolUse",
        }),
    );
    assert!(out.trim().is_empty(), "non-Bash tools are ignored: {out}");
}

#[test]
fn pre_tool_use_is_silent_with_no_rules_and_tolerates_garbage() {
    let dir = tempfile::tempdir().unwrap();
    let _guard = spawn_daemon(dir.path());
    wait_for_daemon(dir.path(), Duration::from_secs(120));

    // A commit with zero stored rules injects nothing.
    let out = run_hook(
        dir.path(),
        "pre-tool-use",
        &serde_json::json!({
            "session_id": "sess-empty",
            "tool_name": "Bash",
            "tool_input": { "command": "git commit -m x" },
            "hook_event_name": "PreToolUse",
        }),
    );
    assert!(out.trim().is_empty(), "no rules means no output: {out}");

    // Garbage stdin never breaks the tool call: exit 0, no output. run_hook
    // asserts the exit status itself.
    let out = run_hook(
        dir.path(),
        "pre-tool-use",
        &serde_json::json!("not an object"),
    );
    assert!(out.trim().is_empty());
}

#[test]
fn session_start_self_updates_hook_registrations() {
    // A settings file written by an older version (five events, no
    // pre-tool-use) must gain the missing hook on session start, while a
    // config dir that never ran setup stays untouched.
    let home = tempfile::tempdir().unwrap();
    let data_dir = tempfile::tempdir().unwrap();

    let ours = home.path().join(".claude");
    std::fs::create_dir_all(&ours).unwrap();
    let old_settings = serde_json::json!({
        "hooks": {
            "UserPromptSubmit": [
                { "hooks": [ { "type": "command", "command": "npx -y mentedb-mcp@latest hook user-prompt" } ] }
            ],
            "Stop": [
                { "hooks": [ { "type": "command", "command": "npx -y mentedb-mcp@latest hook stop" } ] }
            ],
            "SessionStart": [
                { "matcher": "startup|resume|compact",
                  "hooks": [ { "type": "command", "command": "npx -y mentedb-mcp@latest hook session-start" } ] }
            ],
            "PostToolUse": [
                { "matcher": "Write|Edit|MultiEdit|NotebookEdit|Bash",
                  "hooks": [ { "type": "command", "command": "npx -y mentedb-mcp@latest hook post-tool-use" } ] }
            ],
            "PreCompact": [
                { "hooks": [ { "type": "command", "command": "npx -y mentedb-mcp@latest hook pre-compact" } ] }
            ]
        }
    });
    std::fs::write(
        ours.join("settings.json"),
        serde_json::to_string_pretty(&old_settings).unwrap(),
    )
    .unwrap();

    // A second profile with hooks from some other tool and no mentedb:
    // the consent guard must leave it alone.
    let theirs = home.path().join(".claude-other");
    std::fs::create_dir_all(&theirs).unwrap();
    let foreign = r#"{ "hooks": { "Stop": [ { "hooks": [ { "type": "command", "command": "other-tool hook stop" } ] } ] } }"#;
    std::fs::write(theirs.join("settings.json"), foreign).unwrap();

    let run = |label: &str| {
        let mut child = Command::new(BIN)
            .args([
                "--data-dir",
                data_dir.path().to_str().unwrap(),
                "--local",
                "hook",
                "session-start",
            ])
            .env("HOME", home.path())
            .env_remove("CLAUDE_CONFIG_DIR")
            .env_remove("MENTEDB_HOOK_NO_SELF_UPDATE")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to run hook");
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(br#"{"session_id":"self-update"}"#)
            .unwrap();
        let out = child.wait_with_output().expect("hook did not exit");
        assert!(out.status.success(), "{label}: hook must exit 0");
        String::from_utf8_lossy(&out.stdout).to_string()
    };

    let out = run("first");
    let updated = std::fs::read_to_string(ours.join("settings.json")).unwrap();
    assert!(
        updated.contains("hook pre-tool-use"),
        "missing hook must be added: {updated}"
    );
    assert!(
        updated.contains("hook user-prompt"),
        "existing hooks must survive"
    );
    assert!(
        out.contains("refreshed its Claude Code hooks"),
        "session output must mention the refresh: {out}"
    );
    assert_eq!(
        std::fs::read_to_string(theirs.join("settings.json")).unwrap(),
        foreign,
        "profiles without mentedb hooks must never be touched"
    );
    let marker = std::fs::read_to_string(data_dir.path().join("hooks_version")).unwrap();
    assert_eq!(marker.trim(), env!("CARGO_PKG_VERSION"));

    // Second run: marker matches, nothing changes, no refresh notice.
    let before = std::fs::read_to_string(ours.join("settings.json")).unwrap();
    let out = run("second");
    let after = std::fs::read_to_string(ours.join("settings.json")).unwrap();
    assert_eq!(before, after, "reconcile must be once per version");
    assert!(
        !out.contains("refreshed its Claude Code hooks"),
        "no repeat notice: {out}"
    );
}
