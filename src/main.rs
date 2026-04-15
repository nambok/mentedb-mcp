mod config;
mod resources;
mod server;
mod tools;

use clap::{Parser, Subcommand};
use tracing_appender::non_blocking;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use config::{Credentials, ServerConfig};

#[derive(Parser)]
#[command(name = "mentedb-mcp", about = "MCP server for MenteDB")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to the data directory
    #[arg(long, default_value = "~/.mentedb")]
    data_dir: String,

    /// Embedding dimension
    #[arg(long, default_value = "128")]
    embedding_dim: usize,

    /// LLM provider: openai, anthropic, ollama, or mock
    #[arg(long, default_value = "mock", env = "MENTEDB_LLM_PROVIDER")]
    llm_provider: String,

    /// API key for the LLM provider
    #[arg(long, env = "MENTEDB_LLM_API_KEY")]
    llm_api_key: Option<String>,

    /// Model name override for the LLM provider
    #[arg(long, env = "MENTEDB_LLM_MODEL")]
    llm_model: Option<String>,

    /// Expose all tools (default: only essential tools for better agent compliance)
    #[arg(long)]
    full_tools: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Auto-configure MenteDB for your MCP client (Copilot CLI, Claude, Cursor, etc.)
    Setup {
        /// Target client
        #[arg(value_enum, default_value = "copilot")]
        client: SetupClient,
    },
    /// Update MCP config and agent instructions (overwrites existing entries)
    Update {
        /// Target client
        #[arg(value_enum, default_value = "copilot")]
        client: SetupClient,
    },
    /// Log in to MenteDB Cloud via browser
    Login {
        /// Target client
        #[arg(value_enum, default_value = "copilot")]
        client: SetupClient,
    },
    /// Log out of MenteDB Cloud and revert to local mode
    Logout {
        /// Target client
        #[arg(value_enum, default_value = "copilot")]
        client: SetupClient,
    },
    /// Show current MenteDB status (local or cloud mode)
    Status,
}

#[derive(Clone, clap::ValueEnum)]
enum SetupClient {
    Copilot,
    Claude,
    Cursor,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Setup { client }) => return run_setup(client, false),
        Some(Commands::Update { client }) => return run_setup(client, true),
        Some(Commands::Login { client }) => return run_login(client).await,
        Some(Commands::Logout { client }) => return run_logout(client),
        Some(Commands::Status) => return run_status().await,
        None => {}
    }

    let data_dir = ServerConfig::resolve_data_dir(&cli.data_dir);

    // Ensure data directory exists for log file
    std::fs::create_dir_all(&data_dir).ok();

    // File logger (persistent, all levels)
    let log_path = data_dir.join("mentedb-mcp.log");
    let file_appender = tracing_appender::rolling::daily(&data_dir, "mentedb-mcp.log");
    let (file_writer, _guard) = non_blocking(file_appender);

    // Stderr logger (for MCP clients that capture stderr)
    let (stderr_writer, _stderr_guard) = non_blocking(std::io::stderr());

    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with(
            fmt::layer()
                .with_writer(file_writer)
                .with_ansi(false)
                .with_target(true)
                .with_thread_ids(true),
        )
        .with(
            fmt::layer()
                .with_writer(stderr_writer)
                .with_ansi(false)
                .with_target(false),
        )
        .init();

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        data_dir = %data_dir.display(),
        log_file = %log_path.display(),
        embedding_dim = cli.embedding_dim,
        llm_provider = %cli.llm_provider,
        "mentedb-mcp starting"
    );

    // Auto-update agent instructions on startup (silent, best-effort)
    auto_update_instructions();

    let config = ServerConfig::new(
        data_dir,
        cli.embedding_dim,
        cli.llm_provider,
        cli.llm_api_key,
        cli.llm_model,
        cli.full_tools,
    );

    server::run(config).await
}

/// Auto-update agent instructions on startup.
/// Checks all known instruction file paths and updates any with stale version markers.
/// Runs silently — errors are logged but never block server startup.
fn auto_update_instructions() {
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => return,
    };

    let instruction_paths = [
        // Copilot CLI
        format!("{home}/.copilot/copilot-instructions.md"),
        // Cursor
        format!("{home}/.cursor/rules/mentedb.md"),
    ];

    for path_str in &instruction_paths {
        let path = std::path::Path::new(path_str);
        if path.exists() {
            match append_instructions(path, false, false) {
                Ok(()) => {}
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "auto-update instructions failed");
                }
            }
        }
    }
}

const TOOL_NAMES: &[&str] = &[
    "process_turn",
    "store_memory",
    "search_memories",
    "forget_memory",
];

const AGENT_INSTRUCTIONS: &str = r#"# Memory

You have persistent memory via MenteDB. You have 4 tools — use them.

## PRIVACY

Memory data is private to this user. Never include memory content in:
- Code commits or pull requests
- Logs, error reports, or diagnostics shared with third parties
- Responses to other users in shared/team contexts
When referencing memories, summarize rather than quoting raw stored content.

## process_turn — CALL EVERY TURN

Call `process_turn` BEFORE responding, on EVERY turn. Pass `user_message` and `assistant_response`.
- It stores the conversation, searches past context, runs inference, and detects contradictions.
- Without this call, nothing is remembered.
- Increment `turn_id` each turn (start at 0).

## USE the returned context

When `process_turn` returns:
- **context**: Truncated summaries with IDs from past conversations. Reference them in your response. Call `search_memories(id)` for full content.
- **pain_warnings**: If non-empty, WARN the user — a similar situation caused problems before.
- **contradictions**: If > 0, flag the inconsistency.

## store_memory — Save important facts

When you notice important information, call `store_memory`:
- **Preferences**: "User prefers Rust" → type: semantic, tags: [preference]
- **Decisions**: "Using PostgreSQL" → type: semantic, tags: [decision]
- **Corrections**: "Deadline is April, not March" → type: correction
- **Procedures**: "Deploy with: cargo build --release" → type: procedural
- **Mistakes**: "Never retag releases" → type: anti_pattern

Don't store chitchat. Store facts, preferences, decisions, corrections.

## search_memories — Look things up

Pass a text query to search by similarity, or a memory UUID to get full content.
Use proactively — if the user mentions a project, search for what you know about it.

## forget_memory — Delete memories

When the user says "forget" or "don't remember that", delete the memory by ID.
"#;

fn run_setup(client: SetupClient, force: bool) -> anyhow::Result<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let binary = which_mentedb_mcp();

    match client {
        SetupClient::Copilot => setup_copilot(&home, &binary, force),
        SetupClient::Claude => setup_claude(&home, &binary, force),
        SetupClient::Cursor => setup_cursor(&home, &binary, force),
    }
}

fn which_mentedb_mcp() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.to_str().map(String::from))
        .unwrap_or_else(|| "mentedb-mcp".to_string())
}

fn write_if_missing(path: &std::path::Path, content: &str, label: &str) -> anyhow::Result<bool> {
    if path.exists() {
        eprintln!("  [skip] {label} already exists: {}", path.display());
        Ok(false)
    } else {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)?;
        eprintln!("  [created] {label}: {}", path.display());
        Ok(true)
    }
}

fn merge_mcp_config(path: &std::path::Path, binary: &str, force: bool) -> anyhow::Result<()> {
    merge_mcp_config_inner(path, binary, force, false, None)
}

/// Write a cloud-mode MCP config entry (SSE transport with auth header).
fn merge_mcp_config_cloud(path: &std::path::Path, api_key: &str) -> anyhow::Result<()> {
    merge_mcp_config_inner(path, "", true, true, Some(api_key))
}

/// Shared implementation for local (stdio) and cloud (SSE) MCP config merging.
fn merge_mcp_config_inner(
    path: &std::path::Path,
    binary: &str,
    force: bool,
    cloud: bool,
    api_key: Option<&str>,
) -> anyhow::Result<()> {
    let mentedb_entry = if cloud {
        let key = api_key.expect("api_key required for cloud config");
        serde_json::json!({
            "url": Credentials::sse_url(),
            "headers": {
                "Authorization": format!("Bearer {key}")
            }
        })
    } else {
        let allow_list: Vec<&str> = TOOL_NAMES.to_vec();
        let mut entry = serde_json::json!({
            "command": binary,
            "args": [],
            "alwaysAllow": allow_list,
        });

        // If LLM env vars are set, include them so MCP clients pass them through
        let mut env_vars = serde_json::Map::new();
        if let Ok(provider) = std::env::var("MENTEDB_LLM_PROVIDER") {
            env_vars.insert(
                "MENTEDB_LLM_PROVIDER".to_string(),
                serde_json::Value::String(provider),
            );
        }
        if let Ok(key) = std::env::var("MENTEDB_LLM_API_KEY") {
            env_vars.insert(
                "MENTEDB_LLM_API_KEY".to_string(),
                serde_json::Value::String(key),
            );
        } else if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            env_vars.insert("OPENAI_API_KEY".to_string(), serde_json::Value::String(key));
        }
        if let Ok(model) = std::env::var("MENTEDB_LLM_MODEL") {
            env_vars.insert(
                "MENTEDB_LLM_MODEL".to_string(),
                serde_json::Value::String(model),
            );
        }
        if !env_vars.is_empty() {
            entry
                .as_object_mut()
                .unwrap()
                .insert("env".to_string(), serde_json::Value::Object(env_vars));
        }
        entry
    };

    let mut mentedb_entry = mentedb_entry;

    let mut config: serde_json::Value = if path.exists() {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content).unwrap_or_else(|_| serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    let servers = config
        .as_object_mut()
        .unwrap()
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));

    let existing = servers.get("mentedb").cloned();
    if existing.is_some() && !force {
        eprintln!("  [skip] mentedb already in MCP config: {}", path.display());
    } else {
        // For local (stdio) mode, preserve user-configured fields (args, env)
        // that we don't generate ourselves. For cloud mode, we replace entirely.
        if !cloud
            && let Some(old) = &existing
            && let Some(old_obj) = old.as_object()
        {
            let new_obj = mentedb_entry.as_object_mut().unwrap();
            // Preserve args if the new entry has none
            if new_obj
                .get("args")
                .and_then(|a| a.as_array())
                .is_none_or(|a| a.is_empty())
                && let Some(old_args) = old_obj.get("args")
            {
                new_obj.insert("args".to_string(), old_args.clone());
            }
            // Merge env: keep old vars, overlay new ones
            if let Some(old_env) = old_obj.get("env").and_then(|e| e.as_object()) {
                let new_env = new_obj
                    .entry("env")
                    .or_insert_with(|| serde_json::json!({}));
                if let Some(new_env_obj) = new_env.as_object_mut() {
                    for (k, v) in old_env {
                        new_env_obj.entry(k.clone()).or_insert(v.clone());
                    }
                }
            }
        }
        servers
            .as_object_mut()
            .unwrap()
            .insert("mentedb".to_string(), mentedb_entry);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&config)?)?;
        let verb = if existing.is_some() {
            "updated"
        } else {
            "created"
        };
        // Transparency: tell the user exactly what was written
        eprintln!("Updating MCP config: {}", path.display());
        if cloud {
            eprintln!(
                "  Setting mcpServers.mentedb -> cloud endpoint ({})",
                Credentials::sse_url()
            );
        } else {
            eprintln!("  Setting mcpServers.mentedb -> local stdio ({binary})");
        }
        eprintln!("  [{verb}] MCP config: {}", path.display());
    }

    Ok(())
}

fn append_instructions(
    path: &std::path::Path,
    interactive: bool,
    force: bool,
) -> anyhow::Result<()> {
    let version_marker = format!(
        "<!-- mentedb-instructions-v{} -->",
        env!("CARGO_PKG_VERSION")
    );

    if path.exists() {
        let content = std::fs::read_to_string(path)?;
        if content.contains(&version_marker) && !force {
            eprintln!(
                "  [skip] instructions already up-to-date: {}",
                path.display()
            );
            return Ok(());
        }

        // Extract old version for display
        let old_version = content
            .lines()
            .find(|l| l.contains("<!-- mentedb-instructions-v"))
            .and_then(|l| {
                l.trim()
                    .strip_prefix("<!-- mentedb-instructions-v")
                    .and_then(|s| s.strip_suffix(" -->"))
            })
            .unwrap_or("unknown");

        // Extract the current MenteDB block to detect user edits
        let has_mentedb_block =
            content.contains("# Memory\n\nYou have persistent memory via MenteDB");

        // Check for user customizations inside the MenteDB block
        let user_customized = if has_mentedb_block {
            let start = content
                .find("# Memory\n\nYou have persistent memory via MenteDB")
                .unwrap();
            let rest = &content[start..];
            let end = rest
                .find("\n# ")
                .map(|i| start + i)
                .unwrap_or(content.len());
            let old_block = &content[start..end];
            // Compare with what the old version would have had
            // If user edited the block, the content won't match a clean install
            // Simple heuristic: check for lines not in AGENT_INSTRUCTIONS
            let agent_lines: std::collections::HashSet<&str> =
                AGENT_INSTRUCTIONS.lines().map(|l| l.trim()).collect();
            old_block
                .lines()
                .any(|l| !l.trim().is_empty() && !agent_lines.contains(l.trim()))
        } else {
            false
        };

        if interactive {
            eprintln!(
                "\n  Agent instructions update: v{} → v{}",
                old_version,
                env!("CARGO_PKG_VERSION")
            );
            eprintln!("  File: {}", path.display());

            // Show the full instructions that will be written
            eprintln!("\n  ┌─── Instructions to be written ───────────────────────────");
            for line in AGENT_INSTRUCTIONS.lines() {
                eprintln!("  │ {line}");
            }
            eprintln!("  └─────────────────────────────────────────────────────────");

            if user_customized {
                eprintln!("\n  ⚠️  Your file has custom edits inside the MenteDB block.");
                eprintln!(
                    "     These will be replaced. Your content OUTSIDE the block is preserved."
                );
                eprintln!("     A backup will be saved to: {}.bak", path.display());
            }

            eprint!("\n  Apply these instructions? [Y/n] ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let input = input.trim().to_lowercase();
            if input == "n" || input == "no" {
                eprintln!("  [skip] user declined update");
                return Ok(());
            }

            // Create backup if user has customizations
            if user_customized {
                let backup_path = path.with_extension("md.bak");
                std::fs::copy(path, &backup_path)?;
                eprintln!("  [backup] saved to: {}", backup_path.display());
            }
        }

        // Remove old MenteDB instructions block, preserve everything else
        let cleaned = if has_mentedb_block {
            let start = content
                .find("# Memory\n\nYou have persistent memory via MenteDB")
                .unwrap();
            let rest = &content[start..];
            let end = rest
                .find("\n# ")
                .map(|i| start + i)
                .unwrap_or(content.len());
            // Also remove old version marker line above the block
            let mut prefix_end = start;
            if let Some(marker_start) = content[..start].rfind("<!-- mentedb-instructions-v") {
                prefix_end = marker_start;
            }
            let prefix = content[..prefix_end].trim_end().to_string();
            let suffix = content[end..].to_string();
            format!("{prefix}\n{suffix}")
        } else {
            content
        };

        // Create backup before writing
        let backup_path = path.with_extension("md.bak");
        std::fs::copy(path, &backup_path)?;

        let updated = format!(
            "{}\n\n{version_marker}\n{AGENT_INSTRUCTIONS}",
            cleaned.trim_end()
        );
        std::fs::write(path, updated)?;
        // Transparency: tell the user exactly what was written
        eprintln!("Updating instructions: {}", path.display());
        eprintln!(
            "  Writing mentedb memory instructions v{}",
            env!("CARGO_PKG_VERSION")
        );
        eprintln!("  Backup saved: {}", backup_path.display());
    } else {
        eprintln!("Updating instructions: {}", path.display());
        eprintln!(
            "  Writing mentedb memory instructions v{}",
            env!("CARGO_PKG_VERSION")
        );
        let content = format!("{version_marker}\n{AGENT_INSTRUCTIONS}");
        write_if_missing(path, &content, "agent instructions")?;
    }
    Ok(())
}

fn setup_copilot(home: &str, binary: &str, force: bool) -> anyhow::Result<()> {
    println!("\nSetting up MenteDB for GitHub Copilot CLI...\n");

    let copilot_dir = std::path::PathBuf::from(home).join(".copilot");
    merge_mcp_config(&copilot_dir.join("mcp-config.json"), binary, force)?;
    append_instructions(&copilot_dir.join("copilot-instructions.md"), true, force)?;

    println!("\nDone! Restart Copilot CLI to activate MenteDB memory.");
    Ok(())
}

fn setup_claude(home: &str, binary: &str, force: bool) -> anyhow::Result<()> {
    println!("\nSetting up MenteDB for Claude Desktop...\n");

    let config_dir = std::path::PathBuf::from(home).join("Library/Application Support/Claude");
    merge_mcp_config(
        &config_dir.join("claude_desktop_config.json"),
        binary,
        force,
    )?;

    println!("\nDone! Restart Claude Desktop to activate MenteDB memory.");
    println!("Note: Claude Desktop reads server instructions automatically from the MCP server.");
    Ok(())
}

fn setup_cursor(home: &str, binary: &str, force: bool) -> anyhow::Result<()> {
    println!("\nSetting up MenteDB for Cursor...\n");

    let cursor_dir = std::path::PathBuf::from(home).join(".cursor");
    merge_mcp_config(&cursor_dir.join("mcp.json"), binary, force)?;
    append_instructions(&cursor_dir.join("rules/mentedb.md"), true, force)?;

    println!("\nDone! Restart Cursor to activate MenteDB memory.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Login / Logout / Status
// ---------------------------------------------------------------------------

/// Returns (mcp_config_path, instruction_path_or_none) for a given client.
fn client_paths(
    client: &SetupClient,
    home: &str,
) -> (std::path::PathBuf, Option<std::path::PathBuf>) {
    match client {
        SetupClient::Copilot => {
            let dir = std::path::PathBuf::from(home).join(".copilot");
            (
                dir.join("mcp-config.json"),
                Some(dir.join("copilot-instructions.md")),
            )
        }
        SetupClient::Claude => {
            let dir = std::path::PathBuf::from(home).join("Library/Application Support/Claude");
            (dir.join("claude_desktop_config.json"), None)
        }
        SetupClient::Cursor => {
            let dir = std::path::PathBuf::from(home).join(".cursor");
            (dir.join("mcp.json"), Some(dir.join("rules/mentedb.md")))
        }
    }
}

fn client_label(client: &SetupClient) -> &'static str {
    match client {
        SetupClient::Copilot => "copilot",
        SetupClient::Claude => "claude",
        SetupClient::Cursor => "cursor",
    }
}

/// Callback payload sent by the browser after authentication.
#[derive(serde::Deserialize)]
struct AuthCallback {
    api_key: String,
    user_id: String,
}

async fn run_login(client: SetupClient) -> anyhow::Result<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let (mcp_config_path, instructions_path) = client_paths(&client, &home);
    let label = client_label(&client);

    // 1. Start a temporary local HTTP server on a random port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();

    let url = format!("https://app.mentedb.com/auth/device?callback_port={port}&client={label}");

    println!("Opening browser for login... If it doesn't open, visit:");
    println!("  {url}");

    // Try to open the browser (best effort)
    let _ = webbrowser::open(&url);

    // 2. Wait for the callback (5 minute timeout)
    let creds = tokio::time::timeout(
        std::time::Duration::from_secs(300),
        wait_for_callback(listener),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Login timed out after 5 minutes"))??;

    // 3. Save credentials
    let credentials = Credentials {
        api_key: creds.api_key.clone(),
        user_id: creds.user_id.clone(),
        endpoint: Credentials::cloud_endpoint().to_string(),
    };
    credentials.save()?;
    let cred_path = Credentials::path()?;
    eprintln!("Updating credentials: {}", cred_path.display());
    eprintln!("  Saved API key and user ID (permissions 0600)");

    // 4. Update MCP config to cloud mode
    merge_mcp_config_cloud(&mcp_config_path, &creds.api_key)?;

    // 5. Update instructions
    let mut modified_files: Vec<(String, String)> = vec![
        (
            "~/.mentedb/credentials.json".to_string(),
            "API key saved".to_string(),
        ),
        (
            mcp_config_path.display().to_string(),
            "MCP endpoint -> cloud".to_string(),
        ),
    ];

    if let Some(instr_path) = &instructions_path {
        append_instructions(instr_path, false, true)?;
        modified_files.push((
            instr_path.display().to_string(),
            format!("memory instructions v{}", env!("CARGO_PKG_VERSION")),
        ));
    }

    // 6. Print summary
    println!("\nLogin successful!\n");
    println!("Files modified:");
    for (path, desc) in &modified_files {
        println!("  {path}  ({desc})");
    }
    println!();
    println!(
        "Your memory is now cloud-synced at {}",
        Credentials::cloud_endpoint()
    );
    println!("Data is encrypted and isolated to your account.");

    Ok(())
}

/// Wait for the auth callback POST on the local server.
async fn wait_for_callback(listener: tokio::net::TcpListener) -> anyhow::Result<AuthCallback> {
    use bytes::Bytes;
    use http_body_util::{BodyExt, Full};
    use hyper::body::Incoming;
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use hyper::{Method, Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;

    let (tx, rx) = tokio::sync::oneshot::channel::<AuthCallback>();
    let tx = std::sync::Arc::new(tokio::sync::Mutex::new(Some(tx)));

    // Accept exactly one connection
    let (stream, _) = listener.accept().await?;
    let io = TokioIo::new(stream);

    let tx_clone = tx.clone();
    let service = service_fn(move |req: Request<Incoming>| {
        let tx = tx_clone.clone();
        async move {
            if req.method() == Method::POST && req.uri().path() == "/callback" {
                let body = req.collect().await?.to_bytes();
                match serde_json::from_slice::<AuthCallback>(&body) {
                    Ok(cb) => {
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(cb);
                        }
                        let resp = Response::builder()
                            .status(StatusCode::OK)
                            .header("Content-Type", "text/plain")
                            .header("Access-Control-Allow-Origin", "*")
                            .body(Full::new(Bytes::from(
                                "Login successful! You can close this tab.",
                            )))?;
                        Ok::<_, anyhow::Error>(resp)
                    }
                    Err(e) => {
                        let resp = Response::builder()
                            .status(StatusCode::BAD_REQUEST)
                            .body(Full::new(Bytes::from(format!("Invalid payload: {e}"))))?;
                        Ok(resp)
                    }
                }
            } else if req.method() == Method::OPTIONS {
                // CORS preflight
                let resp = Response::builder()
                    .status(StatusCode::OK)
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "POST, OPTIONS")
                    .header("Access-Control-Allow-Headers", "Content-Type")
                    .body(Full::new(Bytes::new()))?;
                Ok(resp)
            } else {
                let resp = Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .body(Full::new(Bytes::from("Not found")))?;
                Ok(resp)
            }
        }
    });

    // Serve the single connection in the background
    tokio::spawn(async move {
        if let Err(e) = http1::Builder::new().serve_connection(io, service).await {
            eprintln!("HTTP server error: {e}");
        }
    });

    // Wait for the callback
    let result = rx
        .await
        .map_err(|_| anyhow::anyhow!("Callback channel closed without receiving credentials"))?;
    Ok(result)
}

fn run_logout(client: SetupClient) -> anyhow::Result<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let binary = which_mentedb_mcp();
    let (mcp_config_path, instructions_path) = client_paths(&client, &home);

    let mut modified: Vec<(String, String)> = Vec::new();

    // 1. Remove credentials
    let cred_path = Credentials::path()?;
    if Credentials::remove()? {
        eprintln!("Removing credentials: {}", cred_path.display());
        modified.push((
            "~/.mentedb/credentials.json".to_string(),
            "removed".to_string(),
        ));
    } else {
        eprintln!("No credentials file found, skipping.");
    }

    // 2. Revert MCP config to local stdio mode
    merge_mcp_config(&mcp_config_path, &binary, true)?;
    modified.push((
        mcp_config_path.display().to_string(),
        "MCP endpoint -> local stdio".to_string(),
    ));

    // 3. Update instructions (same as update does)
    if let Some(instr_path) = &instructions_path {
        append_instructions(instr_path, false, true)?;
        modified.push((
            instr_path.display().to_string(),
            format!("memory instructions v{}", env!("CARGO_PKG_VERSION")),
        ));
    }

    println!("\nLogout complete.\n");
    println!("Files modified:");
    for (path, desc) in &modified {
        println!("  {path}  ({desc})");
    }
    println!();
    println!(
        "MenteDB is now in local mode. Your local memories in ~/.mentedb are still available."
    );

    Ok(())
}

async fn run_status() -> anyhow::Result<()> {
    println!("MenteDB Memory Status");

    match Credentials::load()? {
        Some(creds) => {
            println!("  Mode: cloud ({})", creds.endpoint);

            // Try to fetch user info from the API
            let client = reqwest::Client::new();
            match client
                .get(format!("{}/api/me", creds.endpoint))
                .header("Authorization", format!("Bearer {}", creds.api_key))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    if let Ok(body) = resp.json::<serde_json::Value>().await {
                        if let Some(email) = body.get("email").and_then(|v| v.as_str()) {
                            println!("  User: {email}");
                        } else {
                            println!("  User: {}", creds.user_id);
                        }
                        if let Some(plan) = body.get("plan").and_then(|v| v.as_str()) {
                            println!("  Plan: {plan}");
                        }
                    }
                }
                Ok(resp) => {
                    println!("  User: {} (API returned {})", creds.user_id, resp.status());
                }
                Err(e) => {
                    println!("  User: {} (could not reach API: {e})", creds.user_id);
                }
            }
        }
        None => {
            let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
            let data_dir = std::path::PathBuf::from(&home).join(".mentedb");
            println!("  Mode: local (~/.mentedb)");

            // Try to open the DB and count memories
            let db_path = data_dir.join("mentedb");
            if db_path.exists() {
                println!("  Database: present");
            } else {
                println!("  Database: not initialized");
            }
        }
    }

    Ok(())
}
