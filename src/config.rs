use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Server configuration parsed from CLI arguments and environment.
#[derive(Debug, Clone)]
#[allow(dead_code)]
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
    /// Expose all tools (for power users). Default: false (only essential tools).
    pub full_tools: bool,
}

impl ServerConfig {
    pub fn new(
        data_dir: PathBuf,
        embedding_dim: usize,
        llm_provider: String,
        llm_api_key: Option<String>,
        llm_model: Option<String>,
        full_tools: bool,
    ) -> Self {
        Self {
            data_dir,
            embedding_dim,
            llm_provider,
            llm_api_key,
            llm_model,
            full_tools,
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

/// Cloud credentials stored in ~/.mentedb/credentials.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credentials {
    pub api_key: String,
    pub user_id: String,
    pub endpoint: String,
}

const CLOUD_ENDPOINT: &str = "https://api.mentedb.com";

impl Credentials {
    /// Path to the credentials file (~/.mentedb/credentials.json).
    pub fn path() -> anyhow::Result<PathBuf> {
        let home = std::env::var("HOME")?;
        Ok(PathBuf::from(home).join(".mentedb/credentials.json"))
    }

    /// Load credentials from disk. Returns None if the file does not exist.
    pub fn load() -> anyhow::Result<Option<Self>> {
        let path = Self::path()?;
        if !path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(&path)?;
        let creds: Self = serde_json::from_str(&content)?;
        Ok(Some(creds))
    }

    /// Save credentials to disk with restricted permissions (0600).
    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, &json)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }

        Ok(())
    }

    /// Remove the credentials file from disk.
    pub fn remove() -> anyhow::Result<bool> {
        let path = Self::path()?;
        if path.exists() {
            std::fs::remove_file(&path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// The cloud API endpoint.
    pub fn cloud_endpoint() -> &'static str {
        CLOUD_ENDPOINT
    }

    /// The SSE URL for MCP cloud transport.
    pub fn sse_url() -> String {
        format!("{CLOUD_ENDPOINT}/mcp/v1/sse")
    }
}
