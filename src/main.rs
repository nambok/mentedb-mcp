mod cloud_client;
mod cloud_server;
mod config;
#[cfg(feature = "local")]
mod resources;
#[cfg(feature = "local")]
mod server;
#[cfg(feature = "local")]
mod tools;

use clap::{Parser, Subcommand};
use tracing_appender::non_blocking;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[cfg(feature = "local")]
use config::ServerConfig;
use config::resolve_data_dir;

fn home_dir() -> Option<String> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
}

#[derive(Parser)]
#[command(name = "mentedb-mcp", about = "MCP server for MenteDB")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to the data directory
    #[arg(long, default_value = "~/.mentedb")]
    data_dir: String,

    /// Embedding dimension (local mode only)
    #[cfg(feature = "local")]
    #[arg(long, default_value = "128")]
    embedding_dim: usize,

    /// LLM provider: openai, anthropic, ollama, or mock (local mode only)
    #[cfg(feature = "local")]
    #[arg(long, default_value = "mock", env = "MENTEDB_LLM_PROVIDER")]
    llm_provider: String,

    /// API key for the LLM provider (local mode only)
    #[cfg(feature = "local")]
    #[arg(long, env = "MENTEDB_LLM_API_KEY")]
    llm_api_key: Option<String>,

    /// Model name override for the LLM provider (local mode only)
    #[cfg(feature = "local")]
    #[arg(long, env = "MENTEDB_LLM_MODEL")]
    llm_model: Option<String>,

    /// Expose all tools (local mode only)
    #[cfg(feature = "local")]
    #[arg(long)]
    full_tools: bool,

    /// Force local mode (use embedded database even when cloud credentials exist)
    #[arg(long)]
    local: bool,
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
    /// Authenticate with MenteDB Cloud
    Login,
    /// Remove cloud credentials
    Logout,
    /// Check cloud connection status
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
        Some(Commands::Login) => return run_login().await,
        Some(Commands::Logout) => return run_logout(),
        Some(Commands::Status) => return run_status().await,
        None => {}
    }

    let data_dir = resolve_data_dir(&cli.data_dir);

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
        "mentedb-mcp starting"
    );

    // Auto-update agent instructions on startup (silent, best-effort)
    auto_update_instructions();

    // Check for cloud credentials — if present, use cloud mode (HTTP proxy, no local DB).
    // This allows multiple MCP server instances to run concurrently.
    if !cli.local
        && let Some((api_url, token)) = load_cloud_credentials()
    {
        tracing::info!("Cloud credentials found, starting in cloud mode (no local database)");
        return cloud_server::run(api_url, token).await;
    }

    // No cloud credentials or --local flag — fall back to local embedded database mode.
    #[cfg(feature = "local")]
    {
        tracing::info!("Starting in local mode (embedded database)");

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

    #[cfg(not(feature = "local"))]
    {
        tracing::error!(
            "No cloud credentials found. Run `mentedb-mcp login` to connect to MenteDB Cloud, or rebuild with --features local for embedded mode."
        );
        anyhow::bail!(
            "No cloud credentials found. Run `mentedb-mcp login` to connect to MenteDB Cloud."
        )
    }
}

/// Auto-update agent instructions on startup.
/// Checks all known instruction file paths and updates any with stale version markers.
/// Runs silently — errors are logged but never block server startup.
fn auto_update_instructions() {
    let home = match home_dir() {
        Some(h) => h,
        None => return,
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

## process_turn — CALL EVERY TURN

Call `process_turn` on EVERY turn. Pass `user_message` and `assistant_response`.
- It returns relevant context from past conversations, stores the current turn, and detects contradictions.
- Without this call, nothing is remembered.
- Increment `turn_id` each turn (start at 0).
- `assistant_response` can be brief or empty if you have not yet drafted a response.

## USE the returned context

When `process_turn` returns:
- **context**: Summaries with IDs from past conversations. Reference them in your response.
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

## Scope

Set `scope: 'always'` for critical rules that must be surfaced every turn regardless of topic. Default is `scope: 'contextual'` (retrieved by semantic similarity). Use `'always'` when the user says "always remember this" or states a hard rule.

## Writing good memories

- One fact per memory — self-contained with project context
- Keep under 200 words. Summarize if needed.
- If a fact changes, store as a `correction` — contradictions are detected automatically.
- Do NOT store: greetings, temporary info, large code blocks, one-off details
"#;

async fn run_login() -> anyhow::Result<()> {
    use std::net::TcpListener;
    use tokio::sync::oneshot;

    println!("\nAuthenticating with MenteDB Cloud...\n");

    // Find available port
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);

    // Channel to receive the token
    let (tx, rx) = oneshot::channel::<String>();
    let tx = std::sync::Arc::new(tokio::sync::Mutex::new(Some(tx)));

    let cloud_url = std::env::var("MENTEDB_CLOUD_URL")
        .unwrap_or_else(|_| "https://app.mentedb.com".to_string());

    // Build CORS headers with the cloud dashboard as allowed origin
    let origin_value = axum::http::HeaderValue::from_str(&cloud_url)
        .unwrap_or_else(|_| axum::http::HeaderValue::from_static("https://app.mentedb.com"));
    let origin_for_post = origin_value.clone();

    // Start local HTTP server
    let tx_clone = tx.clone();
    let server = axum::Router::new()
        .route(
            "/callback",
            axum::routing::post(move |body: axum::Json<serde_json::Value>| {
                let tx = tx_clone.clone();
                let origin = origin_for_post.clone();
                async move {
                    let api_key = body
                        .get("api_key")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();

                    if let Some(sender) = tx.lock().await.take() {
                        let _ = sender.send(api_key);
                    }

                    let headers = [
                        (axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN, origin),
                        (
                            axum::http::header::ACCESS_CONTROL_ALLOW_HEADERS,
                            axum::http::HeaderValue::from_static("content-type"),
                        ),
                        (
                            axum::http::header::ACCESS_CONTROL_ALLOW_METHODS,
                            axum::http::HeaderValue::from_static("POST, OPTIONS"),
                        ),
                    ];
                    (
                        axum::http::StatusCode::OK,
                        headers,
                        axum::Json(serde_json::json!({"ok": true})),
                    )
                }
            }),
        )
        .route(
            "/callback",
            axum::routing::options(move || {
                let origin = origin_value.clone();
                async move {
                    let headers = [
                        (axum::http::header::ACCESS_CONTROL_ALLOW_ORIGIN, origin),
                        (
                            axum::http::header::ACCESS_CONTROL_ALLOW_HEADERS,
                            axum::http::HeaderValue::from_static("content-type"),
                        ),
                        (
                            axum::http::header::ACCESS_CONTROL_ALLOW_METHODS,
                            axum::http::HeaderValue::from_static("POST, OPTIONS"),
                        ),
                    ];
                    (axum::http::StatusCode::OK, headers, "")
                }
            }),
        );

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    let server_handle = tokio::spawn(async move {
        axum::serve(listener, server).await.ok();
    });

    // Open browser
    let url = format!("{cloud_url}/auth/device?callback_port={port}");
    println!("  Opening browser: {url}");
    println!("  Waiting for authorization...\n");

    if open::that(&url).is_err() {
        println!("  Could not open browser automatically.");
        println!("  Please visit: {url}\n");
    }

    // Wait for token (with timeout)
    let token = tokio::time::timeout(std::time::Duration::from_secs(300), rx)
        .await
        .map_err(|_| anyhow::anyhow!("Login timed out after 5 minutes"))?
        .map_err(|_| anyhow::anyhow!("Login cancelled"))?;

    if token.is_empty() {
        anyhow::bail!("Received empty token");
    }

    // Save to ~/.mentedb/cloud.json
    let home = home_dir().unwrap_or_else(|| "~".to_string());
    let mentedb_dir = std::path::PathBuf::from(&home).join(".mentedb");
    std::fs::create_dir_all(&mentedb_dir)?;

    let cloud_config = serde_json::json!({
        "api_url": std::env::var("MENTEDB_API_URL")
            .unwrap_or_else(|_| "https://api.mentedb.com".to_string()),
        "token": token,
    });

    let config_path = mentedb_dir.join("cloud.json");
    std::fs::write(&config_path, serde_json::to_string_pretty(&cloud_config)?)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&config_path, std::fs::Permissions::from_mode(0o600))?;
    }

    server_handle.abort();

    println!("  Authenticated successfully!");
    println!("  Credentials saved to: {}", config_path.display());
    println!("\n  Your MCP server will now sync with MenteDB Cloud.");

    Ok(())
}

fn run_logout() -> anyhow::Result<()> {
    let home = home_dir().unwrap_or_else(|| "~".to_string());
    let mentedb_dir = std::path::PathBuf::from(&home).join(".mentedb");
    let config_path = mentedb_dir.join("cloud.json");

    if config_path.exists() {
        std::fs::remove_file(&config_path)?;
        println!("  Logged out. Cloud credentials removed.");
    } else {
        println!("  Not logged in (no credentials found).");
    }
    Ok(())
}

async fn run_status() -> anyhow::Result<()> {
    let home = home_dir().unwrap_or_else(|| "~".to_string());
    let mentedb_dir = std::path::PathBuf::from(&home).join(".mentedb");
    let config_path = mentedb_dir.join("cloud.json");

    if !config_path.exists() {
        println!("  Status: Not logged in");
        println!("  Run `mentedb-mcp login` to authenticate.");
        return Ok(());
    }

    let config_str = std::fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_str)?;

    let api_url = config["api_url"]
        .as_str()
        .unwrap_or("https://api.mentedb.com");
    let token = config["token"].as_str().unwrap_or("");

    if token.is_empty() {
        println!("  Status: Invalid credentials (empty token)");
        println!("  Run `mentedb-mcp login` to re-authenticate.");
        return Ok(());
    }

    println!("  Checking cloud connection...");

    let client = reqwest::Client::new();
    let res = client
        .get(format!("{api_url}/api/sessions"))
        .header("Authorization", format!("Bearer {token}"))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await;

    match res {
        Ok(resp) => match resp.status().as_u16() {
            200 => {
                println!("  Status: Connected");
                println!("  Cloud URL: {api_url}");
                let masked = format!("mdb_{}...", &token[4..12]);
                println!("  Token: {masked}");
            }
            401 | 403 => {
                println!("  Status: Session revoked");
                println!();
                println!("  Your session has been revoked (possibly from the web dashboard).");
                println!("  Run `mentedb-mcp login` to create a new session.");
                // Remove stale credentials
                std::fs::remove_file(&config_path).ok();
            }
            code => {
                println!("  Status: Error (HTTP {code})");
                println!("  The cloud service may be temporarily unavailable.");
            }
        },
        Err(e) => {
            if e.is_timeout() {
                println!("  Status: Timeout");
                println!("  Could not reach {api_url} within 10 seconds.");
            } else {
                println!("  Status: Connection failed");
                println!("  Error: {e}");
            }
        }
    }

    Ok(())
}

/// Load cloud credentials from ~/.mentedb/cloud.json.
/// Returns (api_url, token) if valid credentials exist.
/// Also checks the MENTEDB_API_URL env var as an override for the API URL.
fn load_cloud_credentials() -> Option<(String, String)> {
    let home = home_dir()?;
    let config_path = std::path::PathBuf::from(&home)
        .join(".mentedb")
        .join("cloud.json");
    if !config_path.exists() {
        return None;
    }

    let config_str = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config_str).ok()?;

    let token = config["token"].as_str().unwrap_or_default();
    if token.is_empty() {
        return None;
    }

    // MENTEDB_API_URL env var takes precedence over stored api_url
    let api_url = std::env::var("MENTEDB_API_URL")
        .ok()
        .or_else(|| config["api_url"].as_str().map(String::from))
        .unwrap_or_else(|| "https://api.mentedb.com".to_string());

    Some((api_url, token.to_string()))
}

fn run_setup(client: SetupClient, force: bool) -> anyhow::Result<()> {
    let home = home_dir().unwrap_or_else(|| "~".to_string());
    let binary = which_mentedb_mcp();

    match client {
        SetupClient::Copilot => setup_copilot(&home, &binary, force),
        SetupClient::Claude => setup_claude(&home, &binary, force),
        SetupClient::Cursor => setup_cursor(&home, &binary, force),
    }
}

fn which_mentedb_mcp() -> String {
    // If running via npx, use "npx" as the command so it stays version-proof.
    // npx sets npm_execpath and npm_lifecycle_event env vars.
    let is_npx = std::env::var("npm_execpath").is_ok()
        || std::env::current_exe()
            .ok()
            .and_then(|p| p.to_str().map(String::from))
            .is_some_and(|p| p.contains(".npm/_npx") || p.contains("npx"));

    if is_npx {
        return "npx".to_string();
    }

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
    let allow_list: Vec<&str> = TOOL_NAMES.to_vec();

    // When running via npx, use "npx" as command and prepend package args
    let (command, base_args) = if binary == "npx" {
        ("npx", vec!["-y", "mentedb-mcp@latest"])
    } else {
        (binary, vec![])
    };

    let mut mentedb_entry = serde_json::json!({
        "command": command,
        "args": base_args,
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
        mentedb_entry
            .as_object_mut()
            .unwrap()
            .insert("env".to_string(), serde_json::Value::Object(env_vars));
    }

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
        // When updating, preserve user-configured args and env.
        if let Some(old) = &existing
            && let Some(old_obj) = old.as_object()
        {
            let new_obj = mentedb_entry.as_object_mut().unwrap();

            // Extract user args from old config (skip npx prefix args like "-y", "mentedb-mcp@latest")
            if let Some(old_args) = old_obj.get("args").and_then(|a| a.as_array()) {
                let user_args: Vec<&serde_json::Value> = old_args
                    .iter()
                    .filter(|a| {
                        let s = a.as_str().unwrap_or_default();
                        // Skip npx meta-args; keep user args like --data-dir, --llm-provider
                        s != "-y" && !s.starts_with("mentedb-mcp")
                    })
                    .collect();

                if !user_args.is_empty() {
                    let new_args = new_obj
                        .get_mut("args")
                        .and_then(|a| a.as_array_mut())
                        .unwrap();
                    for arg in user_args {
                        new_args.push(arg.clone());
                    }
                }
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
                eprintln!("\n  [warn] Your file has custom edits inside the MenteDB block.");
                eprintln!(
                    "         These will be replaced. Your content OUTSIDE the block is preserved."
                );
                eprintln!("         A backup will be saved to: {}.bak", path.display());
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

        let updated = format!(
            "{}\n\n{version_marker}\n{AGENT_INSTRUCTIONS}",
            cleaned.trim_end()
        );
        std::fs::write(path, updated)?;
        eprintln!(
            "  [updated] refreshed MenteDB instructions: {}",
            path.display()
        );
    } else {
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
    println!("\nTo sync memories across devices, run: mentedb-mcp login");
    Ok(())
}

fn setup_claude(home: &str, binary: &str, force: bool) -> anyhow::Result<()> {
    println!("\nSetting up MenteDB for Claude Desktop...\n");

    let config_dir = if cfg!(target_os = "macos") {
        std::path::PathBuf::from(home).join("Library/Application Support/Claude")
    } else if cfg!(target_os = "windows") {
        std::env::var("APPDATA")
            .map(|d| std::path::PathBuf::from(d).join("Claude"))
            .unwrap_or_else(|_| std::path::PathBuf::from(home).join("AppData/Roaming/Claude"))
    } else {
        // Linux: ~/.config/Claude
        std::path::PathBuf::from(home).join(".config/Claude")
    };
    merge_mcp_config(
        &config_dir.join("claude_desktop_config.json"),
        binary,
        force,
    )?;

    println!("\nDone! Restart Claude Desktop to activate MenteDB memory.");
    println!("Note: Claude Desktop reads server instructions automatically from the MCP server.");
    println!("\nTo sync memories across devices, run: mentedb-mcp login");
    Ok(())
}

fn setup_cursor(home: &str, binary: &str, force: bool) -> anyhow::Result<()> {
    println!("\nSetting up MenteDB for Cursor...\n");

    let cursor_dir = std::path::PathBuf::from(home).join(".cursor");
    merge_mcp_config(&cursor_dir.join("mcp.json"), binary, force)?;
    append_instructions(&cursor_dir.join("rules/mentedb.md"), true, force)?;

    println!("\nDone! Restart Cursor to activate MenteDB memory.");
    println!("\nTo sync memories across devices, run: mentedb-mcp login");
    Ok(())
}
