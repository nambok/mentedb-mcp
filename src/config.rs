use std::path::PathBuf;

/// Server configuration parsed from CLI arguments and environment.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Path to the MenteDB data directory.
    pub data_dir: PathBuf,
    /// Embedding vector dimension.
    pub embedding_dim: usize,
}

impl ServerConfig {
    pub fn new(data_dir: PathBuf, embedding_dim: usize) -> Self {
        Self {
            data_dir,
            embedding_dim,
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
