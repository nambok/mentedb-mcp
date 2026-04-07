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

use config::ServerConfig;

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

## ⚠️ process_turn — CALL EVERY TURN

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
    let allow_list: Vec<&str> = TOOL_NAMES.to_vec();
    let mut mentedb_entry = serde_json::json!({
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
        // When updating, preserve user-configured fields (args, env) that we
        // don't generate ourselves.  Only overwrite command and alwaysAllow.
        if let Some(old) = &existing {
            if let Some(old_obj) = old.as_object() {
                let new_obj = mentedb_entry.as_object_mut().unwrap();
                // Preserve args if the new entry has none
                if new_obj
                    .get("args")
                    .and_then(|a| a.as_array())
                    .map_or(true, |a| a.is_empty())
                {
                    if let Some(old_args) = old_obj.get("args") {
                        new_obj.insert("args".to_string(), old_args.clone());
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
