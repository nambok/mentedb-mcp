//! Local hook daemon: a long-running process that owns the embedded database
//! and serves lifecycle hooks over localhost HTTP.
//!
//! Why a daemon: the Candle embedding model takes hundreds of milliseconds to
//! load and the engine's index, graph, and cognitive state persist only on
//! flush. A per-turn hook process would pay the model load on every prompt
//! and race a concurrently running MCP server on flush (last writer wins).
//! One daemon owns the database; hook invocations are thin HTTP calls.
//!
//! Registration: the daemon binds an ephemeral 127.0.0.1 port and writes
//! `daemon.json` (port, pid, token) into the data directory with 0600 perms.
//! Requests must carry the token in `x-mentedb-token`. The engine is flushed
//! after every stored turn so a kill never loses data.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::ServerConfig;
use crate::tools::{MenteDbServer, ProcessTurnRequest};
use rmcp::handler::server::wrapper::Parameters;

const DAEMON_FILE: &str = "daemon.json";

/// Connection info for a running daemon, registered in the data directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonInfo {
    pub port: u16,
    pub pid: u32,
    pub token: String,
}

pub fn info_path(data_dir: &Path) -> PathBuf {
    data_dir.join(DAEMON_FILE)
}

pub fn read_info(data_dir: &Path) -> Option<DaemonInfo> {
    let raw = std::fs::read_to_string(info_path(data_dir)).ok()?;
    serde_json::from_str(&raw).ok()
}

#[derive(Clone)]
struct DaemonState {
    server: Arc<MenteDbServer>,
    token: String,
}

fn authorized(state: &DaemonState, headers: &HeaderMap) -> bool {
    headers
        .get("x-mentedb-token")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|t| t == state.token)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "ok": true, "version": env!("CARGO_PKG_VERSION") }))
}

#[derive(Deserialize)]
struct ContextRequest {
    prompt: String,
    #[serde(default)]
    limit: Option<usize>,
}

async fn context(
    State(state): State<DaemonState>,
    headers: HeaderMap,
    Json(req): Json<ContextRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if !authorized(&state, &headers) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    let ctx = state
        .server
        .hook_context(&req.prompt, req.limit.unwrap_or(8));
    Ok(Json(ctx))
}

#[derive(Deserialize)]
struct TurnRequest {
    user_message: String,
    #[serde(default)]
    assistant_response: Option<String>,
    turn_id: u64,
    #[serde(default)]
    project_context: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
}

async fn turn(
    State(state): State<DaemonState>,
    headers: HeaderMap,
    Json(req): Json<TurnRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if !authorized(&state, &headers) {
        return Err(StatusCode::UNAUTHORIZED);
    }

    let result = state
        .server
        .process_turn(Parameters(ProcessTurnRequest {
            user_message: req.user_message,
            assistant_response: req.assistant_response,
            turn_id: req.turn_id,
            project_context: req.project_context,
            agent_id: None,
            session_id: req.session_id,
        }))
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "daemon process_turn failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Durability: the engine's index, graph, and cognitive state only persist
    // on flush; without this a daemon kill would lose the turn's links.
    if let Err(e) = state.server.db_ref().flush() {
        tracing::warn!(error = %e, "post-turn flush failed");
    }

    let text = result
        .content
        .first()
        .and_then(|c| c.as_text())
        .map(|t| t.text.clone())
        .unwrap_or_default();
    let parsed: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|_| json!({ "raw": text }));
    Ok(Json(parsed))
}

async fn session_context(
    State(state): State<DaemonState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if !authorized(&state, &headers) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(Json(state.server.hook_session_context()))
}

#[derive(Deserialize)]
struct NoteRequest {
    content: String,
    #[serde(default)]
    project: Option<String>,
}

async fn note(
    State(state): State<DaemonState>,
    headers: HeaderMap,
    Json(req): Json<NoteRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if !authorized(&state, &headers) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    state
        .server
        .hook_store_note(&req.content, req.project.as_deref());
    // Durable immediately: an interrupted session must not lose captured work.
    if let Err(e) = state.server.db_ref().flush() {
        tracing::warn!(error = %e, "post-note flush failed");
    }
    Ok(Json(json!({ "ok": true })))
}

async fn flush(
    State(state): State<DaemonState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if !authorized(&state, &headers) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    if let Err(e) = state.server.db_ref().flush() {
        tracing::warn!(error = %e, "flush failed");
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }
    Ok(Json(json!({ "ok": true })))
}

#[derive(Deserialize)]
struct InjectionContextRequest {
    query: String,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    exclude_ids: Vec<String>,
    #[serde(default)]
    max_items: Option<usize>,
    #[serde(default)]
    max_episodic: Option<usize>,
}

/// Injection-ready context through the engine's attention policy.
async fn injection_context(
    State(state): State<DaemonState>,
    headers: HeaderMap,
    Json(req): Json<InjectionContextRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if !authorized(&state, &headers) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    let db = state.server.db_ref();
    let embedding = db.embed_text(&req.query).ok().flatten().unwrap_or_default();
    let exclude: Vec<mentedb::prelude::MemoryId> = req
        .exclude_ids
        .iter()
        .filter_map(|s| uuid::Uuid::parse_str(s).ok())
        .map(mentedb::prelude::MemoryId)
        .collect();

    let query = mentedb::injection::InjectionQuery {
        embedding: &embedding,
        query_text: Some(&req.query),
        session_id: req.session_id.as_deref(),
        agent_id: None,
        exclude_ids: &exclude,
        max_items: req.max_items.unwrap_or(6).min(20),
        max_episodic: req.max_episodic.unwrap_or(2).min(10),
    };
    let selected = db.recall_for_injection(&query).map_err(|e| {
        tracing::error!(error = %e, "injection recall failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let memories: Vec<serde_json::Value> = selected
        .iter()
        .map(|c| {
            json!({
                "id": c.node.id.to_string(),
                "content": c.node.content,
                "memory_type": memory_type_str(&c.node.memory_type),
                "tags": c.node.tags,
                "created_at": c.node.created_at.to_string(),
                "score": c.score,
                "reason": match c.reason {
                    mentedb::injection::SelectionReason::Pinned => "pinned",
                    mentedb::injection::SelectionReason::Relevant => "relevant",
                },
            })
        })
        .collect();
    Ok(Json(json!({ "memories": memories, "pain": [] })))
}

#[derive(Deserialize)]
struct InjectionOutcomeRequest {
    #[serde(default)]
    shown_ids: Vec<String>,
    #[serde(default)]
    assistant_text: String,
}

/// Close the attention loop: usage detection against the reply.
async fn injection_outcome(
    State(state): State<DaemonState>,
    headers: HeaderMap,
    Json(req): Json<InjectionOutcomeRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if !authorized(&state, &headers) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    let db = state.server.db_ref();
    let shown: Vec<mentedb::prelude::MemoryId> = req
        .shown_ids
        .iter()
        .filter_map(|s| uuid::Uuid::parse_str(s).ok())
        .map(mentedb::prelude::MemoryId)
        .collect();
    let reply_embedding = if req.assistant_text.is_empty() {
        None
    } else {
        db.embed_text(&req.assistant_text).ok().flatten()
    };
    let (shown_updated, used) = db
        .record_injection_outcome(&shown, reply_embedding.as_deref())
        .map_err(|e| {
            tracing::error!(error = %e, "injection outcome failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    if let Err(e) = db.flush() {
        tracing::warn!(error = %e, "post-outcome flush failed");
    }
    Ok(Json(json!({ "shown": shown_updated, "used": used })))
}

/// Dump every memory for `mentedb-mcp sync` (local to cloud migration).
async fn export(
    State(state): State<DaemonState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, StatusCode> {
    if !authorized(&state, &headers) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    let db = state.server.db_ref();
    let memories: Vec<serde_json::Value> = db
        .memory_ids()
        .into_iter()
        .filter_map(|id| db.get_memory(id).ok())
        .map(|n| {
            json!({
                "id": n.id.to_string(),
                "content": n.content,
                "memory_type": memory_type_str(&n.memory_type),
                "tags": n.tags,
                "created_at": n.created_at,
            })
        })
        .collect();
    Ok(Json(json!({ "memories": memories })))
}

fn memory_type_str(mt: &mentedb::prelude::MemoryType) -> &'static str {
    use mentedb::prelude::MemoryType;
    match mt {
        MemoryType::Episodic => "episodic",
        MemoryType::Semantic => "semantic",
        MemoryType::Procedural => "procedural",
        MemoryType::Correction => "correction",
        MemoryType::AntiPattern => "anti_pattern",
        MemoryType::Reasoning => "reasoning",
    }
}

async fn daemon_health_ok(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1))
        .build()
    {
        Ok(client) => client
            .get(&url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false),
        Err(_) => false,
    }
}

/// Run the hook daemon until SIGTERM/Ctrl-C. Refuses to start when a healthy
/// daemon is already registered for this data directory.
pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    let data_dir = config.data_dir.clone();
    std::fs::create_dir_all(&data_dir)?;

    if let Some(existing) = read_info(&data_dir)
        && daemon_health_ok(existing.port).await
    {
        anyhow::bail!(
            "daemon already running for {} (pid {}, port {})",
            data_dir.display(),
            existing.pid,
            existing.port
        );
    }

    let db = mentedb::MenteDb::open(&data_dir)
        .map_err(|e| anyhow::anyhow!("failed to open database: {e}"))?;
    let server = Arc::new(MenteDbServer::new(db, config));
    let token = uuid::Uuid::new_v4().to_string();

    let state = DaemonState {
        server: Arc::clone(&server),
        token: token.clone(),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/context", post(context))
        .route("/v1/turn", post(turn))
        .route("/v1/note", post(note))
        .route("/v1/flush", post(flush))
        .route("/v1/session-context", post(session_context))
        .route("/v1/export", post(export))
        .route("/v1/injection-context", post(injection_context))
        .route("/v1/injection-outcome", post(injection_outcome))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();

    let info = DaemonInfo {
        port,
        pid: std::process::id(),
        token,
    };
    // The registration file carries the auth token: it must never exist
    // world-readable, so create it 0600 from the first byte.
    let info_file = info_path(&data_dir);
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&info_file)?;
        f.write_all(serde_json::to_string(&info)?.as_bytes())?;
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&info_file, std::fs::Permissions::from_mode(0o600)).ok();
    }
    #[cfg(not(unix))]
    std::fs::write(&info_file, serde_json::to_string(&info)?)?;

    tracing::info!(port, data_dir = %data_dir.display(), "hook daemon listening");

    let shutdown_server = Arc::clone(&server);
    let shutdown_info = info_file.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            {
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("failed to install SIGTERM handler");
                tokio::select! {
                    _ = ctrl_c => {},
                    _ = sigterm.recv() => {},
                }
            }
            #[cfg(not(unix))]
            {
                let _ = ctrl_c.await;
            }
        })
        .await?;

    if let Err(e) = shutdown_server.db_ref().close() {
        tracing::warn!(error = %e, "database close failed during daemon shutdown");
    }
    std::fs::remove_file(&shutdown_info).ok();
    tracing::info!("hook daemon stopped");
    Ok(())
}
