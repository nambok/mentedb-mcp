use std::path::PathBuf;

/// Server configuration parsed from CLI arguments and environment.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Path to the MenteDB data directory.
    pub data_dir: PathBuf,
    /// Embedding vector dimension.
    pub embedding_dim: usize,
    /// Default LLM provider for extraction (e.g. "mock", "openai", "anthropic", "ollama").
    pub llm_provider: String,
    /// Optional API key for the LLM provider.
    pub llm_api_key: Option<String>,
    /// Optional model name override for the LLM provider.
    pub llm_model: Option<String>,
}

impl ServerConfig {
    pub fn new(
        data_dir: PathBuf,
        embedding_dim: usize,
        llm_provider: String,
        llm_api_key: Option<String>,
        llm_model: Option<String>,
    ) -> Self {
        Self {
            data_dir,
            embedding_dim,
            llm_provider,
            llm_api_key,
            llm_model,
        }
    }

    /// Resolve the data directory, expanding `~` to the user home.
    pub fn resolve_data_dir(raw: &str) -> PathBuf {
        if raw.starts_with('~')
            && let Some(home) = dirs_home()
        {
            return home.join(raw.trim_start_matches("~/"));
        }
        PathBuf::from(raw)
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}
