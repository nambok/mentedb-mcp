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

    // Turn 2: the prompt hook must now recall the stored turn as context.
    let out = run_hook(
        dir.path(),
        "user-prompt",
        &serde_json::json!({
            "session_id": session,
            "prompt": "what is my project codename",
            "hook_event_name": "UserPromptSubmit",
        }),
    );
    let parsed: serde_json::Value =
        serde_json::from_str(out.trim()).expect("hook must print valid JSON when it has context");
    let ctx = parsed["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .expect("additionalContext present");
    assert_eq!(
        parsed["hookSpecificOutput"]["hookEventName"],
        "UserPromptSubmit"
    );
    assert!(
        ctx.contains("nightjar"),
        "recalled context must contain the stored fact, got: {ctx}"
    );

    drop(guard);
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
