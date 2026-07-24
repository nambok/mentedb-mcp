mod cloud_client;
mod cloud_server;
mod config;
#[cfg(feature = "local")]
mod daemon;
mod hook;
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

/// Every Claude Code settings directory on this machine. Users running
/// separate configs (work vs personal) set CLAUDE_CONFIG_DIR or keep
/// multiple ~/.claude-* directories; writing hooks to only ~/.claude then
/// configures an instance they never run. Order: CLAUDE_CONFIG_DIR first,
/// then ~/.claude, then any ~/.claude-* directory that already has a
/// settings.json.
fn claude_config_dirs(home: &str) -> Vec<std::path::PathBuf> {
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(v) = std::env::var("CLAUDE_CONFIG_DIR")
        && !v.trim().is_empty()
    {
        dirs.push(std::path::PathBuf::from(v));
    }

    let default = std::path::PathBuf::from(home).join(".claude");
    if !dirs.contains(&default) {
        dirs.push(default);
    }

    if let Ok(entries) = std::fs::read_dir(home) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(".claude-")
                && entry.path().is_dir()
                && entry.path().join("settings.json").exists()
                && !dirs.contains(&entry.path())
            {
                dirs.push(entry.path());
            }
        }
    }
    dirs
}

#[derive(Parser)]
#[command(name = "mentedb-mcp", about = "MCP server for MenteDB")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to the data directory
    #[arg(long, default_value = "~/.mentedb", global = true)]
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
        #[arg(value_enum, default_value = "claude-code")]
        client: SetupClient,
    },
    /// Update MCP config and agent instructions (overwrites existing entries)
    Update {
        /// Target client
        #[arg(value_enum, default_value = "claude-code")]
        client: SetupClient,
    },
    /// Authenticate with MenteDB Cloud
    Login,
    /// Remove cloud credentials
    Logout,
    /// Manage named MenteDB accounts (add, list, use, remove)
    Accounts {
        #[command(subcommand)]
        command: AccountsCommand,
    },
    /// Print a compact health block from local state (exit 1 when unhealthy)
    Status,
    /// Diagnose the full memory pipeline (credentials, hooks, capture, spool)
    Doctor,
    /// Push memories from the local database up to MenteDB Cloud
    #[cfg(feature = "local")]
    Sync,
    /// Process a client lifecycle hook (reads JSON payload from stdin)
    Hook {
        /// Hook event to process
        #[arg(value_enum)]
        event: hook::HookEvent,
    },
    /// Run the local hook daemon (owns the embedded database)
    #[cfg(feature = "local")]
    Daemon,
}

#[derive(Subcommand)]
enum AccountsCommand {
    /// Register a named account with an mdb_ API key (prompts if --key omitted)
    Add {
        /// Account name, e.g. "work" or "personal"
        name: String,
        /// The mdb_ API key. If omitted, you are prompted (key is not echoed).
        #[arg(long)]
        key: Option<String>,
        /// Optional per-account cloud API URL override
        #[arg(long)]
        cloud_url: Option<String>,
    },
    /// List accounts, showing which is active (keys are masked)
    List,
    /// Set the active account (all cloud operations use its key)
    Use {
        /// Account name to activate
        name: String,
    },
    /// Remove a named account
    Remove {
        /// Account name to remove
        name: String,
    },
}

#[derive(Clone, clap::ValueEnum)]
enum SetupClient {
    Copilot,
    Claude,
    Cursor,
    /// Claude Code (the CLI): lifecycle hooks, no MCP tool schemas
    ClaudeCode,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Setup { client }) => return run_setup(client, false),
        Some(Commands::Update { client }) => return run_setup(client, true),
        Some(Commands::Login) => return run_login().await,
        Some(Commands::Logout) => return run_logout(),
        Some(Commands::Accounts { command }) => return run_accounts(command).await,
        Some(Commands::Status) => {
            let data_dir = resolve_data_dir(&cli.data_dir);
            return run_status(&data_dir).await;
        }
        Some(Commands::Doctor) => {
            let data_dir = resolve_data_dir(&cli.data_dir);
            return run_doctor(&data_dir).await;
        }
        #[cfg(feature = "local")]
        Some(Commands::Sync) => {
            let data_dir = resolve_data_dir(&cli.data_dir);
            return run_sync(&data_dir).await;
        }
        Some(Commands::Hook { event }) => {
            // Hooks print only their payload on stdout; logging goes to the
            // data directory so a noisy logger can never corrupt the output
            // the client parses.
            let data_dir = resolve_data_dir(&cli.data_dir);
            std::fs::create_dir_all(&data_dir).ok();
            let file_appender = tracing_appender::rolling::daily(&data_dir, "mentedb-hook.log");
            let (file_writer, _guard) = non_blocking(file_appender);
            tracing_subscriber::registry()
                .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
                .with(fmt::layer().with_writer(file_writer).with_ansi(false))
                .init();
            return hook::run(event, data_dir, cli.local).await;
        }
        #[cfg(feature = "local")]
        Some(Commands::Daemon) => {
            let data_dir = resolve_data_dir(&cli.data_dir);
            std::fs::create_dir_all(&data_dir).ok();
            let file_appender = tracing_appender::rolling::daily(&data_dir, "mentedb-daemon.log");
            let (file_writer, _guard) = non_blocking(file_appender);
            tracing_subscriber::registry()
                .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
                .with(fmt::layer().with_writer(file_writer).with_ansi(false))
                .init();
            let config = ServerConfig::new(
                data_dir,
                cli.embedding_dim,
                cli.llm_provider,
                cli.llm_api_key,
                cli.llm_model,
                cli.full_tools,
            );
            return daemon::run(config).await;
        }
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

    // If stdin is a TTY, the user ran this directly — show help instead of hanging.
    if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        eprintln!("mentedb-mcp v{}", env!("CARGO_PKG_VERSION"));
        eprintln!();
        eprintln!("This is an MCP (Model Context Protocol) server that communicates over stdio.");
        eprintln!("It's designed to be launched by an MCP client, not run directly.");
        eprintln!();
        eprintln!("Quick start:");
        eprintln!(
            "  mentedb-mcp setup     Set up MenteDB in your editor (Copilot CLI, Cursor, etc.)"
        );
        eprintln!("  mentedb-mcp login     Authenticate with MenteDB Cloud");
        eprintln!("  mentedb-mcp status    Check connection and configuration");
        eprintln!();
        eprintln!("For more info: https://mentedb.com/docs");
        std::process::exit(0);
    }

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

    let current_version = env!("CARGO_PKG_VERSION");
    let version_marker = format!("<!-- mentedb-instructions-v{current_version} -->");

    let instruction_paths = [
        // Copilot CLI
        format!("{home}/.copilot/copilot-instructions.md"),
        // Cursor
        format!("{home}/.cursor/rules/mentedb.md"),
    ];

    for path_str in &instruction_paths {
        let path = std::path::Path::new(path_str);
        if !path.exists() {
            continue;
        }

        // Check if update is needed before calling append_instructions
        let needs_update = std::fs::read_to_string(path)
            .map(|content| !content.contains(&version_marker))
            .unwrap_or(false);

        if !needs_update {
            continue;
        }

        let old_version = std::fs::read_to_string(path)
            .ok()
            .and_then(|c| {
                c.lines()
                    .find(|l| l.contains("<!-- mentedb-instructions-v"))
                    .and_then(|l| {
                        l.trim()
                            .strip_prefix("<!-- mentedb-instructions-v")
                            .and_then(|s| s.strip_suffix(" -->"))
                            .map(String::from)
                    })
            })
            .unwrap_or_else(|| "unknown".to_string());

        match append_instructions(path, false, false) {
            Ok(()) => {
                tracing::info!(
                    path = %path.display(),
                    from = %old_version,
                    to = %current_version,
                    "auto-updated agent instructions"
                );
            }
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "auto-update instructions failed");
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

When the user explicitly says "remember this", "always remember", "don't forget", or similar — call `store_memory` immediately in that same turn. Do not rely on process_turn extraction for explicit requests.

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

## Resilience

Even if `process_turn` fails or errors on a turn, ALWAYS call it again on the next turn. Never skip because of a prior failure.
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

    // One-time nonce: the callback only accepts a token from the browser tab
    // this login opened, so another local process or a malicious page cannot
    // plant its own credentials.
    let state_nonce = uuid::Uuid::new_v4().to_string();
    let expected_state = state_nonce.clone();

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
                let expected_state = expected_state.clone();
                async move {
                    let api_key = body
                        .get("api_key")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let state = body.get("state").and_then(|v| v.as_str()).unwrap_or("");

                    if state == expected_state {
                        if let Some(sender) = tx.lock().await.take() {
                            let _ = sender.send(api_key);
                        }
                    } else {
                        eprintln!("  [warn] callback with wrong state nonce rejected");
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
    let url = format!("{cloud_url}/auth/device?callback_port={port}&state={state_nonce}");
    println!("  Opening browser: {url}");

    if open::that(&url).is_err() {
        println!("  Could not open browser automatically.");
        println!("  Please visit: {url}");
    }

    let api_url =
        std::env::var("MENTEDB_API_URL").unwrap_or_else(|_| "https://api.mentedb.com".to_string());

    println!("\n  Waiting for authorization...");
    println!("  Approve in the browser and this finishes on its own.");
    println!();
    println!("  Only on SSH or a remote machine, where the browser cannot reach this");
    println!("  terminal, the page shows a code after you approve; paste it here:");
    println!();

    // Whichever arrives first wins: the browser callback (same machine) or a
    // manually pasted connection code (SSH and remote sessions).
    async fn read_pasted_code() -> Option<String> {
        use tokio::io::AsyncBufReadExt;
        let mut lines = tokio::io::BufReader::new(tokio::io::stdin()).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    let code = line.trim().to_string();
                    if !code.is_empty() {
                        return Some(code);
                    }
                }
                _ => return None,
            }
        }
    }

    let token = tokio::select! {
        cb = rx => {
            let t = cb.map_err(|_| anyhow::anyhow!("Login cancelled"))?;
            if t.is_empty() {
                anyhow::bail!("Received empty token");
            }
            t
        }
        pasted = read_pasted_code() => {
            let Some(code) = pasted else {
                anyhow::bail!("Login cancelled");
            };
            // A pasted code is typed by hand; verify it against the API
            // before persisting so typos fail loudly here.
            print!("  Verifying code... ");
            let client = reqwest::Client::new();
            let ok = client
                .get(format!("{api_url}/api/usage"))
                .bearer_auth(&code)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            if !ok {
                anyhow::bail!("The pasted code was rejected by the API. Run login again.");
            }
            println!("ok");
            code
        }
        _ = tokio::time::sleep(std::time::Duration::from_secs(300)) => {
            anyhow::bail!("Login timed out after 5 minutes");
        }
    };

    // Save to ~/.mentedb/cloud.json, merging into the multi-account config.
    // `login` targets the active account (or "default" when none is set), so
    // it keeps behaving like a single-account login while `accounts add`
    // manages the rest.
    let dir = mentedb_dir().unwrap_or_else(|| std::path::PathBuf::from("~/.mentedb"));
    std::fs::create_dir_all(&dir)?;

    let target = {
        let config = config::load_accounts(&dir)?;
        config
            .active_name()
            .unwrap_or_else(|| config::DEFAULT_ACCOUNT.to_string())
    };
    let cloud_url = std::env::var("MENTEDB_API_URL").ok();
    let config_path = upsert_account(&dir, &target, &token, cloud_url, true)?;

    server_handle.abort();

    println!("  Authenticated successfully!");
    println!("  Account: {target}");
    println!("  Credentials saved to: {}", config_path.display());
    println!("\n  Your MCP server will now sync with MenteDB Cloud.");

    Ok(())
}

/// Insert or update a named account in `<dir>/cloud.json`, writing the file
/// with 0600 permissions. When `make_active` is set (or no account is active
/// yet) the account is also made active. Returns the credentials file path.
fn upsert_account(
    dir: &std::path::Path,
    name: &str,
    api_key: &str,
    cloud_url: Option<String>,
    make_active: bool,
) -> anyhow::Result<std::path::PathBuf> {
    let mut config = config::load_accounts(dir)?;
    let entry = config.accounts.entry(name.to_string()).or_default();
    entry.api_key = api_key.to_string();
    // Only overwrite the stored URL when a new one was supplied, so re-running
    // login does not clobber an account's pinned cloud_url.
    if cloud_url.is_some() {
        entry.cloud_url = cloud_url;
    }
    if make_active || config.active_account.is_none() {
        config.active_account = Some(name.to_string());
    }

    let config_path = config::credentials_path(dir);
    write_secret_file(&config_path, &config.to_json_string()?)?;
    Ok(config_path)
}

/// Dispatch `mentedb-mcp accounts <subcommand>`.
async fn run_accounts(cmd: AccountsCommand) -> anyhow::Result<()> {
    let dir = mentedb_dir().unwrap_or_else(|| std::path::PathBuf::from("~/.mentedb"));
    match cmd {
        AccountsCommand::Add {
            name,
            key,
            cloud_url,
        } => run_accounts_add(&dir, &name, key, cloud_url).await,
        AccountsCommand::List => run_accounts_list(&dir),
        AccountsCommand::Use { name } => run_accounts_use(&dir, &name),
        AccountsCommand::Remove { name } => run_accounts_remove(&dir, &name),
    }
}

/// Register (or update) a named account with an `mdb_` API key.
async fn run_accounts_add(
    dir: &std::path::Path,
    name: &str,
    key: Option<String>,
    cloud_url: Option<String>,
) -> anyhow::Result<()> {
    if name.trim().is_empty() {
        anyhow::bail!("Account name cannot be empty.");
    }

    // Take the key from the flag, else prompt. Never echo it back.
    let key = match key {
        Some(k) => k.trim().to_string(),
        None => {
            print!("  Paste the mdb_ API key for account '{name}': ");
            use std::io::Write;
            std::io::stdout().flush().ok();
            let mut line = String::new();
            std::io::stdin().read_line(&mut line)?;
            line.trim().to_string()
        }
    };

    if key.is_empty() {
        anyhow::bail!("No API key provided.");
    }
    if !key.starts_with("mdb_") {
        // A warning, not an error: keys are opaque and the prefix may change.
        eprintln!("  [warn] key does not start with 'mdb_'; storing it anyway.");
    }

    std::fs::create_dir_all(dir)?;
    let existed = config::load_accounts(dir)?.accounts.contains_key(name);
    let config_path = upsert_account(dir, name, &key, cloud_url, false)?;

    let verb = if existed { "Updated" } else { "Added" };
    println!("  {verb} account '{name}' ({}).", config::mask_secret(&key));
    // Report whether this became the active account.
    let active = config::load_accounts(dir)?.active_name();
    if active.as_deref() == Some(name) {
        println!("  Active account is now '{name}'.");
    } else if let Some(active) = active {
        println!(
            "  Active account remains '{active}'. Switch with `mentedb-mcp accounts use {name}`."
        );
    }
    println!("  Saved to: {}", config_path.display());
    Ok(())
}

/// List every account, marking the active one and masking keys.
fn run_accounts_list(dir: &std::path::Path) -> anyhow::Result<()> {
    let config = config::load_accounts(dir)?;
    if config.accounts.is_empty() {
        println!("  No accounts configured.");
        println!("  Add one with `mentedb-mcp accounts add <name>` or `mentedb-mcp login`.");
        return Ok(());
    }

    let active = config.active_name();
    println!("  Accounts:");
    for (name, account) in &config.accounts {
        let marker = if active.as_deref() == Some(name.as_str()) {
            "*"
        } else {
            " "
        };
        let email = account.email.as_deref().unwrap_or("-");
        let url = account
            .cloud_url
            .as_deref()
            .unwrap_or(config::DEFAULT_CLOUD_URL);
        println!(
            "  {marker} {name}  key={}  email={email}  url={url}",
            config::mask_secret(&account.api_key)
        );
    }
    println!("\n  (* = active)");
    Ok(())
}

/// Set the active account.
fn run_accounts_use(dir: &std::path::Path, name: &str) -> anyhow::Result<()> {
    let mut config = config::load_accounts(dir)?;
    if !config.accounts.contains_key(name) {
        anyhow::bail!(
            "No account named '{name}'. Run `mentedb-mcp accounts list` to see configured accounts."
        );
    }
    config.active_account = Some(name.to_string());
    std::fs::create_dir_all(dir)?;
    write_secret_file(&config::credentials_path(dir), &config.to_json_string()?)?;
    println!("  Active account is now '{name}'.");
    Ok(())
}

/// Remove an account. Clears the active pointer if it named the removed one.
fn run_accounts_remove(dir: &std::path::Path, name: &str) -> anyhow::Result<()> {
    let mut config = config::load_accounts(dir)?;
    if config.accounts.remove(name).is_none() {
        anyhow::bail!(
            "No account named '{name}'. Run `mentedb-mcp accounts list` to see configured accounts."
        );
    }
    let was_active = config.active_account.as_deref() == Some(name);
    if was_active {
        config.active_account = None;
    }
    std::fs::create_dir_all(dir)?;
    write_secret_file(&config::credentials_path(dir), &config.to_json_string()?)?;

    println!("  Removed account '{name}'.");
    if was_active {
        if let Some(next) = config.active_name() {
            // A single remaining account is resolved implicitly; make that
            // explicit so later commands are unambiguous.
            config.active_account = Some(next.clone());
            write_secret_file(&config::credentials_path(dir), &config.to_json_string()?)?;
            println!("  Active account is now '{next}'.");
        } else if !config.accounts.is_empty() {
            println!("  No active account set. Choose one with `mentedb-mcp accounts use <name>`.");
        }
    }
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

/// Compact health block built from local state only: files under the data
/// directory plus two cheap liveness probes (daemon TCP port, cloud /health).
/// No authenticated calls, no new server endpoints. Exits 1 when unhealthy
/// (spool backlog or auth error) so scripts can gate on it.
async fn run_status(data_dir: &std::path::Path) -> anyhow::Result<()> {
    // A liveness probe, not a real API call; anything slower than this is
    // effectively down for interactive use.
    const CLOUD_HEALTH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);
    // The daemon is on loopback, so a live one accepts a TCP connection
    // near-instantly; a longer wait only slows down the "not running" answer.
    const DAEMON_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

    println!("\nMenteDB status v{}\n", env!("CARGO_PKG_VERSION"));
    println!("  Data dir:   {}", data_dir.display());

    let depth = hook::spool::depth(data_dir);
    if depth == 0 {
        println!("  Spool:      empty");
    } else {
        println!("  Spool:      {depth} queued");
    }

    // The hook writes this marker on authentication failure and clears it on
    // the next successful call (see record_auth_state in src/hook/mod.rs).
    let auth_ok = !data_dir.join("auth_error").exists();
    if auth_ok {
        println!("  Auth:       ok");
    } else {
        println!("  Auth:       session invalid, run `mentedb-mcp login`");
    }

    // Daemon liveness from daemon.json, parsed directly (not via the daemon
    // module) so this also works in cloud-only builds without local mode.
    let daemon_port = std::fs::read_to_string(data_dir.join("daemon.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|info| info.get("port").and_then(|p| p.as_u64()))
        .and_then(|p| u16::try_from(p).ok());
    match daemon_port {
        Some(port) => {
            let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
            if std::net::TcpStream::connect_timeout(&addr, DAEMON_CONNECT_TIMEOUT).is_ok() {
                println!("  Daemon:     running (port {port})");
            } else {
                println!("  Daemon:     not running (stale daemon.json, port {port})");
            }
        }
        None => println!("  Daemon:     not started"),
    }

    // Cloud reachability: unauthenticated GET /health on the configured base
    // URL. Answers "is the service up", not "is the token valid"; the auth
    // line above covers token validity from local state.
    if config::credentials_path(data_dir).exists() {
        let api_url = std::env::var("MENTEDB_API_URL")
            .ok()
            .or_else(|| {
                config::load_accounts(data_dir)
                    .ok()
                    .and_then(|c| c.active_account().and_then(|(_, a)| a.cloud_url.clone()))
            })
            .unwrap_or_else(|| config::DEFAULT_CLOUD_URL.to_string());
        let reachable = reqwest::Client::new()
            .get(format!("{api_url}/health"))
            .timeout(CLOUD_HEALTH_TIMEOUT)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        if reachable {
            println!("  Cloud:      reachable ({api_url})");
        } else {
            println!("  Cloud:      unreachable ({api_url})");
        }
    } else {
        println!("  Cloud:      not configured (local mode)");
    }

    match sync_report(data_dir) {
        SyncReport::LastSuccess(ts) => println!("  Sync:       last successful sync {ts}"),
        SyncReport::FailingSince(ts) => println!("  Sync:       failing since {ts}"),
        SyncReport::NoEvents => println!("  Sync:       no sync events logged yet"),
    }

    println!();
    let healthy = depth < hook::SPOOL_WARN_THRESHOLD && auth_ok;
    if healthy {
        println!("  Health:     ok");
        Ok(())
    } else {
        let mut reasons: Vec<String> = Vec::new();
        if depth >= hook::SPOOL_WARN_THRESHOLD {
            reasons.push(format!("spool backlog {depth}"));
        }
        if !auth_ok {
            reasons.push("auth error".to_string());
        }
        println!("  Health:     degraded ({})", reasons.join(", "));
        std::process::exit(1);
    }
}

/// Sync recency derived from the newest daily hook log file.
#[derive(Debug, PartialEq)]
enum SyncReport {
    /// Newest sync event is a full spool flush at this timestamp.
    LastSuccess(String),
    /// Newest sync event is a store failure at this timestamp.
    FailingSince(String),
    /// No hook log or no sync events in it yet.
    NoEvents,
}

/// Marker lines written by the hook logger (see flush_spool and the
/// PostToolUse handler in src/hook/mod.rs). Matched by substring so the
/// structured fields around them do not matter.
const SYNC_OK_MARKER: &str = "offline spool fully flushed";
const SYNC_FAIL_MARKER: &str = "store_note failed";

fn sync_report(data_dir: &std::path::Path) -> SyncReport {
    // tracing_appender writes daily files named mentedb-hook.log.YYYY-MM-DD;
    // the date suffix sorts lexically, so max by path is the newest file.
    let newest_log = std::fs::read_dir(data_dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("mentedb-hook.log."))
        })
        .max();
    let Some(path) = newest_log else {
        return SyncReport::NoEvents;
    };
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return SyncReport::NoEvents;
    };
    // Lines are appended chronologically, so the last hit of each marker is
    // the newest, and its leading token is the tracing timestamp.
    let newest_ts = |marker: &str| -> Option<String> {
        raw.lines()
            .rev()
            .find(|l| l.contains(marker))
            .and_then(|l| l.split_whitespace().next())
            .map(str::to_string)
    };
    match (newest_ts(SYNC_OK_MARKER), newest_ts(SYNC_FAIL_MARKER)) {
        // ISO-8601 timestamps compare chronologically as strings.
        (Some(ok), Some(fail)) if fail > ok => SyncReport::FailingSince(fail),
        (Some(ok), _) => SyncReport::LastSuccess(ok),
        (None, Some(fail)) => SyncReport::FailingSince(fail),
        (None, None) => SyncReport::NoEvents,
    }
}

/// Directory holding MenteDB credentials and state (`~/.mentedb`).
fn mentedb_dir() -> Option<std::path::PathBuf> {
    Some(std::path::PathBuf::from(home_dir()?).join(".mentedb"))
}

/// Load cloud credentials for the ACTIVE account from ~/.mentedb/cloud.json.
/// Returns (api_url, token) if a valid active account exists.
///
/// The legacy single-key shape is migrated on read (see
/// `config::AccountsConfig`), so pre-existing single-key installs keep working.
/// Also checks the MENTEDB_API_URL env var as an override for the API URL.
fn load_cloud_credentials() -> Option<(String, String)> {
    let dir = mentedb_dir()?;
    let config = config::load_accounts(&dir).ok()?;
    let (_, account) = config.active_account()?;
    if account.api_key.is_empty() {
        return None;
    }

    // MENTEDB_API_URL env var takes precedence over the account's cloud_url.
    let api_url = std::env::var("MENTEDB_API_URL")
        .ok()
        .or_else(|| account.cloud_url.clone())
        .unwrap_or_else(|| config::DEFAULT_CLOUD_URL.to_string());

    Some((api_url, account.api_key.clone()))
}

/// Write a credentials file created with 0600 from the first byte, instead
/// of writing world-readable and tightening afterwards.
fn write_secret_file(path: &std::path::Path, contents: &str) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        f.write_all(contents.as_bytes())?;
        // mode() only applies on create; correct pre-existing files too.
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, contents)?;
    }
    Ok(())
}

/// End-to-end pipeline diagnosis: answers "is memory actually working?"
/// in one command.
async fn run_doctor(data_dir: &std::path::Path) -> anyhow::Result<()> {
    println!("\nMenteDB doctor v{}\n", env!("CARGO_PKG_VERSION"));
    let mut problems = 0usize;

    // 1. Backend credentials.
    let creds = load_cloud_credentials();
    match &creds {
        Some((api_url, token)) => {
            println!(
                "  [ok]   Cloud credentials: {} ({})",
                config::mask_secret(token),
                api_url
            );
            let client = reqwest::Client::new();
            match client
                .get(format!("{api_url}/api/me"))
                .header("Authorization", format!("Bearer {token}"))
                .timeout(std::time::Duration::from_secs(10))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let me: serde_json::Value = resp.json().await.unwrap_or_default();
                    let email = me.get("email").and_then(|v| v.as_str()).unwrap_or("?");
                    let plan = me.get("plan").and_then(|v| v.as_str()).unwrap_or("?");
                    println!("  [ok]   Cloud reachable, account {email} (plan: {plan})");
                }
                Ok(resp) if resp.status().as_u16() == 401 || resp.status().as_u16() == 403 => {
                    problems += 1;
                    println!("  [FAIL] Token revoked or expired. Run `mentedb-mcp login`.");
                }
                Ok(resp) => {
                    problems += 1;
                    println!("  [warn] Cloud returned HTTP {}", resp.status());
                }
                Err(e) => {
                    problems += 1;
                    println!("  [FAIL] Cloud unreachable: {e}");
                }
            }
        }
        None => {
            println!("  [info] Not logged in: hooks use the local daemon backend.");
            #[cfg(not(feature = "local"))]
            {
                problems += 1;
                println!("  [FAIL] This build has no local mode. Run `mentedb-mcp login`.");
            }
        }
    }

    // 2. Claude Code hooks registered, in every config directory.
    let home = home_dir().unwrap_or_else(|| "~".to_string());
    let expected = [
        "UserPromptSubmit",
        "Stop",
        "SessionStart",
        "PostToolUse",
        "PreToolUse",
        "PreCompact",
    ];
    for dir in claude_config_dirs(&home) {
        let settings_path = dir.join("settings.json");
        match std::fs::read_to_string(&settings_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        {
            Some(settings) => {
                let mut missing = Vec::new();
                for event in expected {
                    let present = settings["hooks"][event].as_array().is_some_and(|groups| {
                        groups.iter().any(|g| {
                            g["hooks"].as_array().is_some_and(|hs| {
                                hs.iter().any(|h| {
                                    h["command"]
                                        .as_str()
                                        .is_some_and(|c| c.contains("mentedb-mcp"))
                                })
                            })
                        })
                    });
                    if !present {
                        missing.push(event);
                    }
                }
                if missing.is_empty() {
                    println!("  [ok]   Hooks in {}: all 5 registered", dir.display());
                } else {
                    problems += 1;
                    println!(
                        "  [FAIL] Hooks in {} missing: {}. Run `npx mentedb-mcp@latest setup claude-code`.",
                        dir.display(),
                        missing.join(", ")
                    );
                }
            }
            None => {
                problems += 1;
                println!(
                    "  [warn] No Claude Code settings at {}. Run `npx mentedb-mcp@latest setup claude-code`.",
                    settings_path.display()
                );
            }
        }
    }

    // 3. Recent capture activity.
    let sessions_dir = data_dir.join("hook_sessions");
    let newest = std::fs::read_dir(&sessions_dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.metadata().ok().and_then(|m| m.modified().ok()))
        .max();
    match newest {
        Some(t) => {
            let ago = t.elapsed().map(|d| d.as_secs()).unwrap_or(0);
            let human = if ago < 120 {
                format!("{ago}s ago")
            } else if ago < 7_200 {
                format!("{}m ago", ago / 60)
            } else {
                format!("{}h ago", ago / 3_600)
            };
            println!("  [ok]   Last hook activity: {human}");
        }
        None => {
            println!(
                "  [info] No hook activity yet. Hooks fire in sessions started after setup; start a new Claude Code session."
            );
        }
    }

    // 4. Offline spool depth.
    let depth = hook::spool::depth(data_dir);
    if depth == 0 {
        println!("  [ok]   Offline spool: empty");
    } else {
        println!("  [warn] Offline spool: {depth} undelivered entries (retried on next turn)");
    }

    // 5. Recent hook errors from the log.
    let today = chrono_free_today();
    let log_path = data_dir.join(format!("mentedb-hook.log.{today}"));
    let warns: Vec<String> = std::fs::read_to_string(&log_path)
        .map(|raw| {
            raw.lines()
                .filter(|l| l.contains("WARN") || l.contains("ERROR"))
                .rev()
                .take(3)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if warns.is_empty() {
        println!("  [ok]   Hook log: no recent errors");
    } else {
        println!("  [warn] Recent hook errors ({}):", log_path.display());
        for w in warns.iter().rev() {
            let trimmed: String = w.chars().take(160).collect();
            println!("           {trimmed}");
        }
    }

    // 6. Local daemon, when that is the active backend.
    #[cfg(feature = "local")]
    if creds.is_none() {
        match crate::daemon::read_info(data_dir) {
            Some(info) => {
                let url = format!("http://127.0.0.1:{}/health", info.port);
                let healthy = reqwest::Client::new()
                    .get(&url)
                    .timeout(std::time::Duration::from_secs(2))
                    .send()
                    .await
                    .map(|r| r.status().is_success())
                    .unwrap_or(false);
                if healthy {
                    println!("  [ok]   Local daemon: running on port {}", info.port);
                } else {
                    println!("  [info] Local daemon: not running (auto-starts on first hook)");
                }
            }
            None => println!("  [info] Local daemon: not started yet (auto-starts on first hook)"),
        }
    }

    println!();
    if problems == 0 {
        println!("  Everything looks healthy.");
    } else {
        println!("  {problems} problem(s) found. Fix the [FAIL] items above.");
    }
    Ok(())
}

/// Today's UTC date as YYYY-MM-DD without pulling in a date crate: the
/// tracing_appender daily logs use this suffix.
fn chrono_free_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    // Civil-from-days (Howard Hinnant's algorithm), valid for the era we run in.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Push every memory in the local database up to MenteDB Cloud, so a user
/// who started local and later logs in keeps their history. Idempotent via
/// a ledger of already-pushed memory IDs.
#[cfg(feature = "local")]
async fn run_sync(data_dir: &std::path::Path) -> anyhow::Result<()> {
    let Some((api_url, token)) = load_cloud_credentials() else {
        anyhow::bail!("Not logged in. Run `mentedb-mcp login` first, then `mentedb-mcp sync`.");
    };

    println!("\nSyncing local memories to MenteDB Cloud...\n");

    let daemon = hook::backend::LocalDaemonClient::connect_or_spawn(data_dir).await?;
    let export = daemon.post("/v1/export", serde_json::json!({})).await?;
    let memories = export
        .get("memories")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    if memories.is_empty() {
        println!("  Local database has no memories to sync.");
        return Ok(());
    }

    let ledger_path = data_dir.join("sync_state.json");
    let mut pushed: std::collections::HashSet<String> = std::fs::read_to_string(&ledger_path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default();

    let client = cloud_client::CloudClient::new(api_url, token);
    let total = memories.len();
    let mut sent = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;

    for m in &memories {
        let id = m
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let content = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
        if id.is_empty() || content.is_empty() || pushed.contains(&id) {
            skipped += 1;
            continue;
        }
        let mut tags: Vec<String> = m
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| t.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        tags.push("synced-from-local".to_string());

        let args = serde_json::json!({
            "content": content,
            "memory_type": m.get("memory_type").and_then(|v| v.as_str()).unwrap_or("episodic"),
            "tags": tags,
        });
        match client.call_tool("store_memory", args).await {
            Ok(_) => {
                pushed.insert(id);
                sent += 1;
                if sent.is_multiple_of(25) {
                    println!("  {sent}/{total} pushed...");
                    std::fs::write(&ledger_path, serde_json::to_string(&pushed)?).ok();
                }
            }
            Err(e) => {
                failed += 1;
                if failed <= 3 {
                    eprintln!("  [warn] failed to push {id}: {e}");
                }
            }
        }
    }
    std::fs::write(&ledger_path, serde_json::to_string(&pushed)?)?;

    println!("\n  Done: {sent} pushed, {skipped} already synced or empty, {failed} failed.");
    if failed > 0 {
        println!("  Re-run `mentedb-mcp sync` to retry failures.");
    }
    Ok(())
}

fn run_setup(client: SetupClient, force: bool) -> anyhow::Result<()> {
    let home = home_dir().unwrap_or_else(|| "~".to_string());
    let binary = which_mentedb_mcp();

    match client {
        SetupClient::Copilot => setup_copilot(&home, &binary, force),
        SetupClient::Claude => setup_claude(&home, &binary, force),
        SetupClient::Cursor => setup_cursor(&home, &binary, force),
        SetupClient::ClaudeCode => setup_claude_code(&home, &binary, force),
    }
}

/// Configure Claude Code (the CLI) with lifecycle hooks instead of MCP.
///
/// Hooks cost zero tool-schema tokens and run deterministically every turn:
/// UserPromptSubmit injects recalled context, Stop stores the completed turn,
/// SessionStart injects the user profile (including right after compaction).
fn setup_claude_code(home: &str, binary: &str, force: bool) -> anyhow::Result<()> {
    println!("\nSetting up MenteDB hooks for Claude Code...\n");

    let hook_command = if binary == "npx" {
        "npx -y mentedb-mcp@latest hook".to_string()
    } else {
        format!("{binary} hook")
    };

    // Install into every Claude config directory on the machine, so users
    // with separate work and personal configs get hooks in all of them.
    for dir in claude_config_dirs(home) {
        eprintln!("  Claude config: {}", dir.display());
        install_claude_code_hooks(&dir.join("settings.json"), &hook_command, force)?;
    }

    println!("\nDone. MenteDB now runs on every Claude Code turn via hooks:");
    println!("  UserPromptSubmit  recalls context for the prompt");
    println!("  PostToolUse       captures significant actions as they happen");
    println!("  PreToolUse        surfaces action rules before git commits and PRs");
    println!("  Stop              stores the completed turn");
    println!("  PreCompact        flushes memory before context is compacted");
    println!("  SessionStart      injects your profile and standing rules");
    println!("\nBackend: cloud when logged in (mentedb-mcp login), otherwise a");
    println!("local daemon that starts automatically on first use.");
    println!("\nRestart any open Claude Code sessions to activate.");
    Ok(())
}

/// The hook subcommands a current install registers. Used by the self-update
/// reconcile to detect a settings file written by an older version.
const HOOK_SUBCOMMANDS: [&str; 6] = [
    "user-prompt",
    "stop",
    "session-start",
    "post-tool-use",
    "pre-tool-use",
    "pre-compact",
];

/// Reconcile installed Claude Code hooks with this binary version.
///
/// The npx launcher auto-updates the binary on every hook invocation, but
/// the hook registrations in settings.json only change when something
/// rewrites them, so a release that adds a hook event never reaches users
/// who set up earlier. This runs from the SessionStart hook and closes that
/// gap: once per binary version (marker file in the data dir), only in
/// settings files that already carry mentedb hooks (running setup was the
/// consent), only when a registration is actually missing, and add-only
/// through the idempotent installer which never touches non mentedb hooks.
/// Returns the settings paths it changed; every failure is skipped silently
/// because this runs inside a hook that must never break the session.
pub(crate) fn ensure_hooks_current(data_dir: &std::path::Path) -> Vec<String> {
    let mut updated = Vec::new();
    let marker = data_dir.join("hooks_version");
    let current = env!("CARGO_PKG_VERSION");
    if std::fs::read_to_string(&marker)
        .map(|v| v.trim() == current)
        .unwrap_or(false)
    {
        return updated;
    }
    let Some(home) = home_dir() else {
        return updated;
    };
    let binary = which_mentedb_mcp();
    let hook_command = if binary == "npx" {
        "npx -y mentedb-mcp@latest hook".to_string()
    } else {
        format!("{binary} hook")
    };
    for dir in claude_config_dirs(&home) {
        let settings_path = dir.join("settings.json");
        let Ok(raw) = std::fs::read_to_string(&settings_path) else {
            continue;
        };
        // Consent guard: reconcile only where the user already installed
        // mentedb hooks; never add hooks to a profile setup never touched.
        if !raw.contains("mentedb-mcp") {
            continue;
        }
        // Rewrite only when a registration is actually missing, so an
        // up-to-date settings file is never touched (rewriting reorders
        // keys and churns diffs for no reason).
        let missing = HOOK_SUBCOMMANDS
            .iter()
            .any(|sub| !raw.contains(&format!("hook {sub}")));
        if !missing {
            continue;
        }
        if install_claude_code_hooks(&settings_path, &hook_command, false).is_ok() {
            updated.push(settings_path.display().to_string());
        }
    }
    let _ = std::fs::write(&marker, current);
    updated
}

fn install_claude_code_hooks(
    settings_path: &std::path::Path,
    hook_command: &str,
    force: bool,
) -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(settings_path).unwrap_or_else(|_| "{}".to_string());
    let mut settings: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({}));
    if !settings.is_object() {
        settings = serde_json::json!({});
    }

    let hooks = settings
        .as_object_mut()
        .expect("settings is an object")
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}));
    if !hooks.is_object() {
        *hooks = serde_json::json!({});
    }

    // (event, subcommand, matcher, timeout seconds). The timeout matters for
    // PreToolUse: it runs directly in front of the user's tool call and the
    // client's default hook timeout is far too long to risk there; 5 seconds
    // caps the worst case while the hook's internal budget keeps the normal
    // case near-instant and fail-open.
    let events: [(&str, &str, &str, Option<u64>); 6] = [
        ("UserPromptSubmit", "user-prompt", "", None),
        ("Stop", "stop", "", None),
        (
            "SessionStart",
            "session-start",
            "startup|resume|compact",
            None,
        ),
        (
            "PostToolUse",
            "post-tool-use",
            "Write|Edit|MultiEdit|NotebookEdit|Bash",
            None,
        ),
        ("PreToolUse", "pre-tool-use", "Bash", Some(5)),
        ("PreCompact", "pre-compact", "", None),
    ];

    for (event, subcommand, matcher, timeout) in events {
        let command = format!("{hook_command} {subcommand}");
        let entries = hooks
            .as_object_mut()
            .expect("hooks is an object")
            .entry(event)
            .or_insert_with(|| serde_json::json!([]));
        if !entries.is_array() {
            *entries = serde_json::json!([]);
        }
        let arr = entries.as_array_mut().expect("entries is an array");

        let already = arr.iter().any(|group| {
            group["hooks"].as_array().is_some_and(|hs| {
                hs.iter().any(|h| {
                    h["command"]
                        .as_str()
                        .is_some_and(|c| c.contains("mentedb-mcp") && c.contains(subcommand))
                })
            })
        });
        if already && !force {
            eprintln!("  [skip] {event} hook already configured");
            continue;
        }
        if already && force {
            arr.retain(|group| {
                !group["hooks"].as_array().is_some_and(|hs| {
                    hs.iter().any(|h| {
                        h["command"]
                            .as_str()
                            .is_some_and(|c| c.contains("mentedb-mcp"))
                    })
                })
            });
        }

        let mut hook_entry = serde_json::json!({ "type": "command", "command": command });
        if let Some(secs) = timeout {
            hook_entry["timeout"] = serde_json::json!(secs);
        }
        let mut group = serde_json::json!({ "hooks": [hook_entry] });
        if !matcher.is_empty() {
            group["matcher"] = serde_json::json!(matcher);
        }
        arr.push(group);
        eprintln!("  [added] {event} hook");
    }

    if let Some(parent) = settings_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(settings_path, serde_json::to_string_pretty(&settings)?)?;
    eprintln!("  [updated] {}", settings_path.display());
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_report_no_events_without_logs() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(sync_report(dir.path()), SyncReport::NoEvents);
        // A log with neither marker also reports no events.
        std::fs::write(
            dir.path().join("mentedb-hook.log.2026-07-22"),
            "2026-07-22T09:00:00.000000Z  INFO hook started\n",
        )
        .unwrap();
        assert_eq!(sync_report(dir.path()), SyncReport::NoEvents);
    }

    #[test]
    fn sync_report_success_when_flush_is_newest() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("mentedb-hook.log.2026-07-22"),
            concat!(
                "2026-07-22T09:00:00.000000Z  WARN store_note failed, spooling for retry\n",
                "2026-07-22T11:00:00.000000Z  INFO offline spool fully flushed\n",
            ),
        )
        .unwrap();
        assert_eq!(
            sync_report(dir.path()),
            SyncReport::LastSuccess("2026-07-22T11:00:00.000000Z".to_string())
        );
    }

    #[test]
    fn sync_report_reads_newest_log_and_reports_failure() {
        let dir = tempfile::tempdir().unwrap();
        // An older file reports success; only the newest file may be read.
        std::fs::write(
            dir.path().join("mentedb-hook.log.2026-07-20"),
            "2026-07-20T10:00:00.000000Z  INFO offline spool fully flushed\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("mentedb-hook.log.2026-07-22"),
            concat!(
                "2026-07-22T09:00:00.000000Z  INFO offline spool fully flushed\n",
                "2026-07-22T11:00:00.000000Z  WARN store_note failed, spooling for retry\n",
            ),
        )
        .unwrap();
        assert_eq!(
            sync_report(dir.path()),
            SyncReport::FailingSince("2026-07-22T11:00:00.000000Z".to_string())
        );
    }
}
