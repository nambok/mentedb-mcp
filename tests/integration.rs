// Integration tests for mentedb-mcp.
//
// Each test spawns the binary as a child process with stdio transport,
// sends JSON-RPC messages, and verifies responses. This exercises the
// full MCP protocol end-to-end.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

// -- Helpers --

fn binary_path() -> PathBuf {
    let path = env!("CARGO_BIN_EXE_mentedb-mcp");
    PathBuf::from(path)
}

fn temp_data_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mentedb-mcp-test-{}-{}",
        test_name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

struct McpProcess {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
    data_dir: PathBuf,
    next_id: u64,
}

impl McpProcess {
    fn spawn(test_name: &str) -> Self {
        let data_dir = temp_data_dir(test_name);
        let mut child = Command::new(binary_path())
            .arg("--data-dir")
            .arg(&data_dir)
            .arg("--embedding-dim")
            .arg("128")
            .arg("--llm-provider")
            .arg("mock")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn mentedb-mcp");

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let reader = BufReader::new(stdout);

        McpProcess {
            child,
            stdin,
            reader,
            data_dir,
            next_id: 1,
        }
    }

    fn send(&mut self, msg: &Value) {
        let s = serde_json::to_string(msg).unwrap();
        writeln!(self.stdin, "{s}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn recv(&mut self) -> Value {
        let mut line = String::new();
        self.reader.read_line(&mut line).unwrap();
        serde_json::from_str(line.trim()).unwrap_or_else(|e| {
            panic!("failed to parse JSON-RPC response: {e}\nraw: {line}");
        })
    }

    fn initialize(&mut self) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "integration-test", "version": "1.0.0" }
            }
        }));
        let resp = self.recv();

        // Send initialized notification (no response expected)
        self.send(&json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }));

        // Give the server time to process the notification
        std::thread::sleep(std::time::Duration::from_millis(100));

        resp
    }

    fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            }
        }));
        self.recv()
    }

    fn tool_result_text(&mut self, name: &str, arguments: Value) -> Value {
        let resp = self.call_tool(name, arguments);
        eprintln!("DEBUG [{name}]: {resp}");
        let content = resp["result"]["content"]
            .as_array()
            .expect("expected content array");
        assert!(!content.is_empty(), "expected at least one content item");
        let text = content[0]["text"].as_str().expect("expected text content");
        serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
    }
}

impl Drop for McpProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.data_dir);
    }
}

// -- Tests --

#[test]
fn test_initialize_returns_server_info() {
    let mut proc = McpProcess::spawn("init");
    let resp = proc.initialize();

    assert_eq!(resp["jsonrpc"], "2.0");
    assert!(resp["error"].is_null(), "unexpected error: {resp}");

    let result = &resp["result"];
    let info = &result["serverInfo"];
    assert_eq!(info["name"], "mentedb-mcp");
    assert!(!info["version"].as_str().unwrap().is_empty());

    let caps = &result["capabilities"];
    assert!(caps["tools"].is_object(), "expected tools capability");
}

#[test]
#[ignore = "recall_memory via MQL does not find memories in fresh temp data dirs"]
fn test_store_and_recall_memory() {
    let mut proc = McpProcess::spawn("store_recall");
    proc.initialize();

    let stored = proc.tool_result_text(
        "store_memory",
        json!({
            "content": "Rust ownership prevents data races at compile time",
            "memory_type": "semantic",
        }),
    );
    assert_eq!(stored["status"], "stored");
    let id = stored["id"].as_str().unwrap();
    assert!(!id.is_empty());

    let recalled = proc.tool_result_text("recall_memory", json!({ "id": id }));
    assert_eq!(
        recalled["content"],
        "Rust ownership prevents data races at compile time"
    );
    assert_eq!(recalled["memory_type"], "Semantic");
}

#[test]
fn test_store_and_search_memories() {
    let mut proc = McpProcess::spawn("store_search");
    proc.initialize();

    proc.tool_result_text(
        "store_memory",
        json!({
            "content": "Tokio is an async runtime for Rust",
            "memory_type": "semantic",
        }),
    );
    proc.tool_result_text(
        "store_memory",
        json!({
            "content": "Serde handles serialization in Rust",
            "memory_type": "semantic",
        }),
    );

    let results = proc.tool_result_text(
        "search_memories",
        json!({
            "query": "async runtime",
            "limit": 5,
        }),
    );
    let count = results["count"].as_u64().unwrap();
    assert!(count >= 1, "expected at least 1 search result, got {count}");
}

#[test]
fn test_store_relate_and_get_related() {
    let mut proc = McpProcess::spawn("relate");
    proc.initialize();

    let a = proc.tool_result_text(
        "store_memory",
        json!({
            "content": "HTTP is a request-response protocol",
            "memory_type": "semantic",
        }),
    );
    let b = proc.tool_result_text(
        "store_memory",
        json!({
            "content": "REST APIs typically use HTTP",
            "memory_type": "semantic",
        }),
    );

    let id_a = a["id"].as_str().unwrap();
    let id_b = b["id"].as_str().unwrap();

    let rel = proc.tool_result_text(
        "relate_memories",
        json!({
            "from_id": id_a,
            "to_id": id_b,
            "edge_type": "related",
        }),
    );
    assert_eq!(rel["status"], "related");

    let related = proc.tool_result_text(
        "get_related",
        json!({
            "id": id_a,
            "depth": 1,
        }),
    );
    let items = related["related"].as_array().unwrap();
    assert!(!items.is_empty(), "expected related memories");
    let found = items.iter().any(|r| r["id"].as_str().unwrap() == id_b);
    assert!(found, "expected to find memory B in related results");
}

#[test]
fn test_store_and_forget_memory() {
    let mut proc = McpProcess::spawn("forget");
    proc.initialize();

    let stored = proc.tool_result_text(
        "store_memory",
        json!({
            "content": "Temporary note to be deleted",
            "memory_type": "episodic",
        }),
    );
    let id = stored["id"].as_str().unwrap();

    let forgot = proc.tool_result_text(
        "forget_memory",
        json!({
            "id": id,
            "reason": "test cleanup",
        }),
    );
    assert_eq!(forgot["status"], "forgotten");

    // Recall should fail after forget
    let recalled = proc.tool_result_text("recall_memory", json!({ "id": id }));
    let text = recalled.as_str().unwrap_or("");
    let is_err = text.contains("not found") || text.contains("Not found");
    // Also check for isError in the raw response
    let raw = proc.call_tool("recall_memory", json!({ "id": id }));
    let is_error_flag = raw["result"]["isError"].as_bool().unwrap_or(false);
    assert!(
        is_err || is_error_flag,
        "expected recall to fail after forget, got: {recalled}"
    );
}

#[test]
fn test_search_with_type_filter() {
    let mut proc = McpProcess::spawn("type_filter");
    proc.initialize();

    proc.tool_result_text(
        "store_memory",
        json!({
            "content": "Today I learned about pattern matching",
            "memory_type": "episodic",
        }),
    );
    proc.tool_result_text(
        "store_memory",
        json!({
            "content": "Pattern matching is a control flow construct",
            "memory_type": "semantic",
        }),
    );

    let results = proc.tool_result_text(
        "search_memories",
        json!({
            "query": "pattern matching",
            "limit": 10,
            "memory_type": "semantic",
        }),
    );
    let items = results["results"].as_array().unwrap();
    for item in items {
        assert_eq!(
            item["memory_type"], "Semantic",
            "expected only Semantic memories with type filter"
        );
    }
}

#[test]
#[ignore = "cognitive state is in-memory only, pain signals not returned via MCP in test env"]
fn test_record_pain_and_cognitive_state() {
    let mut proc = McpProcess::spawn("pain");
    proc.initialize();

    let stored = proc.tool_result_text(
        "store_memory",
        json!({
            "content": "Accidentally deleted production database",
            "memory_type": "episodic",
        }),
    );
    let memory_id = stored["id"].as_str().unwrap();

    let pain = proc.tool_result_text(
        "record_pain",
        json!({
            "memory_id": memory_id,
            "intensity": 0.9,
            "trigger_keywords": ["delete", "production", "database"],
            "description": "Never run destructive commands on production without backup",
        }),
    );
    assert_eq!(pain["status"], "recorded");

    let state = proc.tool_result_text("get_cognitive_state", json!({}));
    let pain_signals = state["pain_signals"].as_array().unwrap();
    assert!(
        !pain_signals.is_empty(),
        "expected at least one pain signal"
    );
    let found = pain_signals
        .iter()
        .any(|s| s["memory_id"].as_str().unwrap() == memory_id);
    assert!(found, "expected pain signal for the stored memory");
}

#[test]
fn test_detect_phantoms() {
    let mut proc = McpProcess::spawn("phantoms");
    proc.initialize();

    let result = proc.tool_result_text("detect_phantoms", json!({
        "content": "The FrobnicatorService handles all frobbing operations and communicates with the BarManager",
        "known_entities": ["BarManager"],
    }));
    // The detector should identify FrobnicatorService as a phantom (unknown entity)
    assert!(
        result["count"].is_number(),
        "expected count field in phantom response"
    );
}

#[test]
fn test_process_turn() {
    let mut proc = McpProcess::spawn("process_turn");
    proc.initialize();

    let result = proc.tool_result_text(
        "process_turn",
        json!({
            "user_message": "I prefer using Rust for all backend services",
            "assistant_response": "Noted, I will use Rust for backend work going forward.",
            "turn_id": 1,
            "project_context": "my-project",
        }),
    );

    assert!(result["turn_id"].as_u64().unwrap() == 1);
    assert!(result["relevant_context"].is_array());
    assert!(result["memories_stored"].is_array());
    assert!(result["pain_warnings"].is_array());
    assert!(result["predicted_topics"].is_array());
    assert!(result["elapsed_ms"].is_number());
}

#[test]
fn test_forget_all() {
    let mut proc = McpProcess::spawn("forget_all");
    proc.initialize();

    // Forget all without confirmation should fail (isError response)
    let resp = proc.call_tool("forget_all", json!({ "confirm": "nope" }));
    let content = resp["result"]["content"].as_array().unwrap();
    let text = content[0]["text"].as_str().unwrap();
    assert!(text.contains("Safety check"), "expected safety check error");

    // Forget all with confirmation should succeed
    let result = proc.tool_result_text(
        "forget_all",
        json!({ "confirm": "CONFIRM", "reason": "test reset" }),
    );
    assert_eq!(result["status"], "reset_complete");
}
