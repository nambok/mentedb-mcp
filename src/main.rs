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

    /// Default LLM provider for extraction: openai, anthropic, ollama, or mock
    #[arg(long, default_value = "mock")]
    llm_provider: String,

    /// API key for the LLM provider (overrides MENTEDB_LLM_API_KEY env var)
    #[arg(long, env = "MENTEDB_LLM_API_KEY")]
    llm_api_key: Option<String>,

    /// Model name override for the LLM provider
    #[arg(long)]
    llm_model: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Auto-configure MenteDB for your MCP client (Copilot CLI, Claude, Cursor, etc.)
    Setup {
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

    if let Some(Commands::Setup { client }) = cli.command {
        return run_setup(client);
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

    let config = ServerConfig::new(
        data_dir,
        cli.embedding_dim,
        cli.llm_provider,
        cli.llm_api_key,
        cli.llm_model,
    );

    server::run(config).await
}

const TOOL_NAMES: &[&str] = &[
    "store_memory", "search_memories", "recall_memory", "get_memory",
    "forget_memory", "forget_all", "relate_memories", "get_related",
    "find_path", "get_subgraph", "find_contradictions", "write_inference",
    "propagate_belief", "extract_facts", "compress_memory",
    "consolidate_memories", "detect_interference", "assemble_context",
    "check_stream", "detect_phantoms", "register_entity", "resolve_phantom",
    "record_pain", "get_cognitive_state", "record_trajectory",
    "predict_topics", "ingest_conversation", "get_stats",
    "evaluate_archival", "apply_decay", "process_turn", "gdpr_forget",
];

const AGENT_INSTRUCTIONS: &str = r#"# Memory

You have persistent memory via MenteDB. Use it automatically, never wait to be asked.

## Every conversation start

1. Call `search_memories` with keywords from the user's first message to load relevant context.
2. Call `get_cognitive_state` to check for active pain signals or knowledge gaps.
3. If results come back, use them to inform your responses. Mention what you remember when relevant.

## During conversation

- When the user shares a preference, decision, or project detail, call `store_memory` immediately. Use descriptive tags.
- When a fact changes or the user corrects you, store the new fact and call `relate_memories` with `supersedes` pointing from the new memory to the old one.
- When something goes wrong (bad advice, failed approach), call `record_pain` so you can warn about it in the future.
- When the user says "forget" or "don't remember", call `forget_memory`.

## Memory types

- `semantic`: facts, decisions, preferences, project details (most common)
- `episodic`: events, meetings, what happened
- `procedural`: how to do things, workflows, commands
- `correction`: when the user corrects you
- `anti_pattern`: mistakes to avoid

## Tags

Always add tags to stored memories. Use lowercase, descriptive tags like:
- Project names: `project-mentedb`, `project-trading-bot`
- Topics: `database`, `deployment`, `testing`, `preference`
- Context: `decision`, `architecture`, `bug-fix`
"#;

fn run_setup(client: SetupClient) -> anyhow::Result<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let binary = which_mentedb_mcp();

    match client {
        SetupClient::Copilot => setup_copilot(&home, &binary),
        SetupClient::Claude => setup_claude(&home, &binary),
        SetupClient::Cursor => setup_cursor(&home, &binary),
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
        println!("  [skip] {label} already exists: {}", path.display());
        Ok(false)
    } else {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)?;
        println!("  [created] {label}: {}", path.display());
        Ok(true)
    }
}

fn merge_mcp_config(path: &std::path::Path, binary: &str) -> anyhow::Result<()> {
    let allow_list: Vec<&str> = TOOL_NAMES.to_vec();
    let mentedb_entry = serde_json::json!({
        "command": binary,
        "args": [],
        "alwaysAllow": allow_list,
    });

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

    if servers.get("mentedb").is_some() {
        println!("  [skip] mentedb already in MCP config: {}", path.display());
    } else {
        servers
            .as_object_mut()
            .unwrap()
            .insert("mentedb".to_string(), mentedb_entry);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&config)?)?;
        println!("  [created] MCP config: {}", path.display());
    }

    Ok(())
}

fn append_instructions(path: &std::path::Path) -> anyhow::Result<()> {
    if path.exists() {
        let content = std::fs::read_to_string(path)?;
        if content.contains("MenteDB") {
            println!("  [skip] instructions already contain MenteDB: {}", path.display());
            return Ok(());
        }
        let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
        use std::io::Write;
        write!(file, "\n{AGENT_INSTRUCTIONS}")?;
        println!("  [updated] appended MenteDB instructions: {}", path.display());
    } else {
        write_if_missing(path, AGENT_INSTRUCTIONS, "agent instructions")?;
    }
    Ok(())
}

fn setup_copilot(home: &str, binary: &str) -> anyhow::Result<()> {
    println!("\nSetting up MenteDB for GitHub Copilot CLI...\n");

    let copilot_dir = std::path::PathBuf::from(home).join(".copilot");
    merge_mcp_config(&copilot_dir.join("mcp-config.json"), binary)?;
    append_instructions(&copilot_dir.join("copilot-instructions.md"))?;

    println!("\nDone! Restart Copilot CLI to activate MenteDB memory.");
    Ok(())
}

fn setup_claude(home: &str, binary: &str) -> anyhow::Result<()> {
    println!("\nSetting up MenteDB for Claude Desktop...\n");

    let config_dir = std::path::PathBuf::from(home)
        .join("Library/Application Support/Claude");
    merge_mcp_config(&config_dir.join("claude_desktop_config.json"), binary)?;

    println!("\nDone! Restart Claude Desktop to activate MenteDB memory.");
    println!("Note: Claude Desktop reads server instructions automatically from the MCP server.");
    Ok(())
}

fn setup_cursor(home: &str, binary: &str) -> anyhow::Result<()> {
    println!("\nSetting up MenteDB for Cursor...\n");

    let cursor_dir = std::path::PathBuf::from(home).join(".cursor");
    merge_mcp_config(&cursor_dir.join("mcp.json"), binary)?;
    append_instructions(&cursor_dir.join("rules/mentedb.md"))?;

    println!("\nDone! Restart Cursor to activate MenteDB memory.");
    Ok(())
}
