use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Resolve the data directory, expanding `~` to the user home.
pub fn resolve_data_dir(raw: &str) -> PathBuf {
    if raw.starts_with('~')
        && let Some(home) = dirs_home()
    {
        return home.join(raw.trim_start_matches("~/"));
    }
    PathBuf::from(raw)
}

/// The default cloud API URL, used when an account does not pin its own.
pub const DEFAULT_CLOUD_URL: &str = "https://api.mentedb.com";

/// The name a migrated legacy single-key config is filed under.
pub const DEFAULT_ACCOUNT: &str = "default";

/// A single named MenteDB account. Each account is a distinct `mdb_` API key,
/// which on the hosted side maps to a distinct engine and storage, so accounts
/// are structurally isolated and never mixed.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Account {
    /// The `mdb_` API key used as the bearer token for every request.
    pub api_key: String,
    /// Optional per-account cloud URL override. When absent, the process-wide
    /// default (or the `MENTEDB_API_URL` env var) applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cloud_url: Option<String>,
    /// Optional cached account email, shown by `accounts list` / `status`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
}

/// The credentials file (`~/.mentedb/cloud.json`) as a set of named accounts.
///
/// Backward compatibility: the legacy shape `{ "api_url", "token" }` is read via
/// [`AccountsConfig::from_json_str`], which migrates a lone legacy key into an
/// account named [`DEFAULT_ACCOUNT`]. The migrated form is persisted on the next
/// write, so old single-key installs keep working untouched until then.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AccountsConfig {
    /// The account whose key the connector and all cloud operations use.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_account: Option<String>,
    /// Named accounts, keyed by user-chosen name. `BTreeMap` keeps `list`
    /// output and on-disk order stable.
    #[serde(default)]
    pub accounts: BTreeMap<String, Account>,
}

impl AccountsConfig {
    /// Parse the credentials file, migrating the legacy single-key shape.
    ///
    /// - New shape (`accounts` present): parsed as-is.
    /// - Legacy shape (`token`/`api_url`, no `accounts`): the key becomes the
    ///   [`DEFAULT_ACCOUNT`] account and is made active.
    /// - Empty / whitespace input: an empty config.
    pub fn from_json_str(raw: &str) -> anyhow::Result<Self> {
        if raw.trim().is_empty() {
            return Ok(Self::default());
        }

        let value: serde_json::Value = serde_json::from_str(raw)?;

        // New shape: an `accounts` object exists. Deserialize directly.
        if value.get("accounts").is_some_and(|a| a.is_object()) {
            let mut config: AccountsConfig = serde_json::from_value(value)?;
            // Drop an active pointer that names a since-removed account so
            // callers never resolve to a missing key.
            if let Some(active) = &config.active_account
                && !config.accounts.contains_key(active)
            {
                config.active_account = None;
            }
            return Ok(config);
        }

        // Legacy shape: a bare token (with optional api_url). Migrate it.
        let mut config = AccountsConfig::default();
        if let Some(token) = value.get("token").and_then(|t| t.as_str())
            && !token.is_empty()
        {
            let cloud_url = value
                .get("api_url")
                .and_then(|u| u.as_str())
                .filter(|u| !u.is_empty())
                .map(str::to_string);
            config.accounts.insert(
                DEFAULT_ACCOUNT.to_string(),
                Account {
                    api_key: token.to_string(),
                    cloud_url,
                    email: None,
                },
            );
            config.active_account = Some(DEFAULT_ACCOUNT.to_string());
        }
        Ok(config)
    }

    /// The active account name, falling back to a lone account or
    /// [`DEFAULT_ACCOUNT`] when no explicit pointer is set.
    pub fn active_name(&self) -> Option<String> {
        if let Some(active) = &self.active_account
            && self.accounts.contains_key(active)
        {
            return Some(active.clone());
        }
        // No explicit (or stale) pointer: a single account is unambiguous.
        if self.accounts.len() == 1 {
            return self.accounts.keys().next().cloned();
        }
        if self.accounts.contains_key(DEFAULT_ACCOUNT) {
            return Some(DEFAULT_ACCOUNT.to_string());
        }
        None
    }

    /// The active account's name and credentials, if one resolves.
    pub fn active_account(&self) -> Option<(&str, &Account)> {
        let name = self.active_name()?;
        // Re-borrow the key from the map so the returned `&str` is tied to the
        // config, not the local `name`.
        self.accounts
            .get_key_value(&name)
            .map(|(k, v)| (k.as_str(), v))
    }

    /// Serialize to the on-disk JSON (pretty, new multi-account shape).
    pub fn to_json_string(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

/// Mask a secret for display without ever panicking on short values.
/// Shows the first 12 characters (enough to see the `mdb_` prefix) then `...`.
pub fn mask_secret(secret: &str) -> String {
    let prefix: String = secret.chars().take(12).collect();
    format!("{prefix}...")
}

/// Path to the credentials file (`<home>/.mentedb/cloud.json`) under a given
/// config directory. Taking the directory as an argument keeps the config
/// logic testable against a temp dir.
pub fn credentials_path(config_dir: &Path) -> PathBuf {
    config_dir.join("cloud.json")
}

/// Load the accounts config from `<config_dir>/cloud.json`, migrating the
/// legacy single-key shape. A missing file yields an empty config.
pub fn load_accounts(config_dir: &Path) -> anyhow::Result<AccountsConfig> {
    let path = credentials_path(config_dir);
    if !path.exists() {
        return Ok(AccountsConfig::default());
    }
    let raw = std::fs::read_to_string(&path)?;
    AccountsConfig::from_json_str(&raw)
}

/// Server configuration parsed from CLI arguments and environment.
#[cfg(feature = "local")]
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
    /// Expose all tools (for power users). Default: false (only essential tools).
    pub full_tools: bool,
}

#[cfg(feature = "local")]
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
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirror what `upsert_account` in main.rs does: add or update an account
    // on disk, optionally making it active. Kept here so the config round-trip
    // (add -> reload -> use -> reload) is exercised end to end against a temp
    // dir without pulling in the binary crate.
    fn upsert(dir: &Path, name: &str, api_key: &str, cloud_url: Option<&str>, make_active: bool) {
        let mut config = load_accounts(dir).unwrap();
        let entry = config.accounts.entry(name.to_string()).or_default();
        entry.api_key = api_key.to_string();
        if let Some(url) = cloud_url {
            entry.cloud_url = Some(url.to_string());
        }
        if make_active || config.active_account.is_none() {
            config.active_account = Some(name.to_string());
        }
        std::fs::write(credentials_path(dir), config.to_json_string().unwrap()).unwrap();
    }

    #[test]
    fn migrates_legacy_single_key_to_default_account() {
        // The old on-disk shape: a bare token plus api_url, no `accounts` map.
        let legacy = r#"{ "api_url": "https://api.example.com", "token": "mdb_legacy_key_123" }"#;
        let config = AccountsConfig::from_json_str(legacy).unwrap();

        // Migrated into exactly one account named "default", made active.
        assert_eq!(config.accounts.len(), 1);
        let account = config
            .accounts
            .get(DEFAULT_ACCOUNT)
            .expect("default account");
        assert_eq!(account.api_key, "mdb_legacy_key_123");
        assert_eq!(
            account.cloud_url.as_deref(),
            Some("https://api.example.com")
        );
        assert_eq!(config.active_account.as_deref(), Some(DEFAULT_ACCOUNT));

        // active_account() resolves the migrated key.
        let (name, active) = config.active_account().unwrap();
        assert_eq!(name, DEFAULT_ACCOUNT);
        assert_eq!(active.api_key, "mdb_legacy_key_123");
    }

    #[test]
    fn migrates_legacy_key_without_api_url() {
        let legacy = r#"{ "token": "mdb_only_token" }"#;
        let config = AccountsConfig::from_json_str(legacy).unwrap();
        let account = config.accounts.get(DEFAULT_ACCOUNT).unwrap();
        assert_eq!(account.api_key, "mdb_only_token");
        assert!(account.cloud_url.is_none());
        assert_eq!(config.active_name().as_deref(), Some(DEFAULT_ACCOUNT));
    }

    #[test]
    fn legacy_migration_persists_on_first_write() {
        // A file written in the legacy shape is transparently upgraded to the
        // multi-account shape the next time it is saved.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            credentials_path(dir.path()),
            r#"{ "api_url": "https://api.example.com", "token": "mdb_legacy" }"#,
        )
        .unwrap();

        // Load (migrates in memory) then add a second account (persists).
        upsert(dir.path(), "work", "mdb_work_key", None, true);

        // Re-read the raw file: it is now the new shape, with both accounts.
        let raw = std::fs::read_to_string(credentials_path(dir.path())).unwrap();
        assert!(raw.contains("\"accounts\""));
        assert!(!raw.contains("\"token\"")); // legacy field gone
        let reloaded = load_accounts(dir.path()).unwrap();
        assert_eq!(reloaded.accounts.len(), 2);
        assert!(reloaded.accounts.contains_key(DEFAULT_ACCOUNT));
        assert!(reloaded.accounts.contains_key("work"));
        assert_eq!(reloaded.active_name().as_deref(), Some("work"));
    }

    #[test]
    fn add_use_list_roundtrip() {
        let dir = tempfile::tempdir().unwrap();

        // add "work" (first account becomes active implicitly).
        upsert(
            dir.path(),
            "work",
            "mdb_work_key",
            Some("https://work.example.com"),
            false,
        );
        // add "personal" (do not switch active).
        upsert(dir.path(), "personal", "mdb_personal_key", None, false);

        // list: both present, "work" is active.
        let config = load_accounts(dir.path()).unwrap();
        assert_eq!(config.accounts.len(), 2);
        assert_eq!(config.active_name().as_deref(), Some("work"));
        assert_eq!(
            config.accounts["work"].cloud_url.as_deref(),
            Some("https://work.example.com")
        );
        assert!(config.accounts["personal"].cloud_url.is_none());

        // use "personal": set active, persist, reload.
        let mut config = load_accounts(dir.path()).unwrap();
        config.active_account = Some("personal".to_string());
        std::fs::write(
            credentials_path(dir.path()),
            config.to_json_string().unwrap(),
        )
        .unwrap();

        let reloaded = load_accounts(dir.path()).unwrap();
        assert_eq!(reloaded.active_name().as_deref(), Some("personal"));
        let (name, active) = reloaded.active_account().unwrap();
        assert_eq!(name, "personal");
        assert_eq!(active.api_key, "mdb_personal_key");
        // The keys are structurally isolated: switching active never mixes them.
        assert_ne!(active.api_key, reloaded.accounts["work"].api_key);
    }

    #[test]
    fn remove_active_clears_pointer() {
        let dir = tempfile::tempdir().unwrap();
        upsert(dir.path(), "work", "mdb_work", None, true);
        upsert(dir.path(), "personal", "mdb_personal", None, false);

        // Remove the active account "work".
        let mut config = load_accounts(dir.path()).unwrap();
        config.accounts.remove("work");
        config.active_account = None;
        std::fs::write(
            credentials_path(dir.path()),
            config.to_json_string().unwrap(),
        )
        .unwrap();

        let reloaded = load_accounts(dir.path()).unwrap();
        assert!(!reloaded.accounts.contains_key("work"));
        // With one account left, active resolves to it implicitly.
        assert_eq!(reloaded.active_name().as_deref(), Some("personal"));
    }

    #[test]
    fn active_name_falls_back_and_ignores_stale_pointer() {
        // No explicit pointer, single account -> that account.
        let mut config = AccountsConfig::default();
        config.accounts.insert(
            "solo".into(),
            Account {
                api_key: "k".into(),
                ..Default::default()
            },
        );
        assert_eq!(config.active_name().as_deref(), Some("solo"));

        // A pointer naming a removed account is treated as unset (via parse).
        let json = r#"{
            "active_account": "gone",
            "accounts": { "a": { "api_key": "ka" }, "b": { "api_key": "kb" } }
        }"#;
        let config = AccountsConfig::from_json_str(json).unwrap();
        assert!(config.active_account.is_none());
        // Two accounts, no valid pointer, no "default" -> ambiguous -> None.
        assert!(config.active_name().is_none());
    }

    #[test]
    fn empty_or_missing_config_is_empty() {
        assert_eq!(
            AccountsConfig::from_json_str("").unwrap(),
            AccountsConfig::default()
        );
        assert_eq!(
            AccountsConfig::from_json_str("   ").unwrap(),
            AccountsConfig::default()
        );
        // Empty legacy token -> no account created.
        let config = AccountsConfig::from_json_str(r#"{ "token": "" }"#).unwrap();
        assert!(config.accounts.is_empty());
        assert!(config.active_name().is_none());

        // Missing file -> empty config.
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            load_accounts(dir.path()).unwrap(),
            AccountsConfig::default()
        );
    }

    #[test]
    fn mask_secret_never_panics_and_hides_tail() {
        assert_eq!(mask_secret("mdb_1234567890abcdef"), "mdb_12345678...");
        // Short and empty inputs must not panic.
        assert_eq!(mask_secret("mdb_"), "mdb_...");
        assert_eq!(mask_secret(""), "...");
        // The full secret is never present in the masked form.
        let secret = "mdb_supersecrettail_zzzzzz";
        assert!(!mask_secret(secret).contains("zzzzzz"));
    }
}
