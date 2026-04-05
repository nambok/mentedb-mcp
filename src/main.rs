mod config;
mod resources;
mod server;
mod tools;

use clap::Parser;
use tracing_appender::non_blocking;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use config::ServerConfig;

#[derive(Parser)]
#[command(name = "mentedb-mcp", about = "MCP server for MenteDB")]
struct Cli {
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

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
