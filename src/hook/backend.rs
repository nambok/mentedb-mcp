//! Hook backends: cloud (single HTTP call per hook, no local engine) and
//! local (thin client of the hook daemon, which owns the embedded database).
//!
//! Selection mirrors the MCP server: cloud when credentials exist and
//! `--local` was not passed, otherwise the local daemon, auto-spawned on
//! first use.

use std::path::Path;

use serde_json::json;

use crate::cloud_client::CloudClient;

pub enum Backend {
    Cloud(CloudClient),
    #[cfg(feature = "local")]
    Local(LocalDaemonClient),
}

impl Backend {
    pub async fn resolve(data_dir: &Path, force_local: bool) -> anyhow::Result<Self> {
        if !force_local && let Some((api_url, token)) = crate::load_cloud_credentials() {
            return Ok(Backend::Cloud(CloudClient::new(api_url, token)));
        }

        #[cfg(feature = "local")]
        {
            let client = LocalDaemonClient::connect_or_spawn(data_dir).await?;
            Ok(Backend::Local(client))
        }

        #[cfg(not(feature = "local"))]
        {
            let _ = data_dir;
            anyhow::bail!(
                "no cloud credentials and this build has no local mode; run `mentedb-mcp login`"
            )
        }
    }

    /// Context for a user prompt: {memories: [...], pain: [...]}.
    pub async fn context(&self, prompt: &str) -> anyhow::Result<serde_json::Value> {
        match self {
            Backend::Cloud(client) => {
                let resp = client
                    .call_tool("search_memories", json!({ "query": prompt, "limit": 8 }))
                    .await
                    .map_err(|e| anyhow::anyhow!(e))?;
                let text = resp
                    .content
                    .first()
                    .map(|c| c.text.clone())
                    .unwrap_or_default();
                let parsed: serde_json::Value =
                    serde_json::from_str(&text).unwrap_or_else(|_| json!({}));
                // Normalize search_memories {results: [...]} to the hook shape.
                let memories = parsed.get("results").cloned().unwrap_or_else(|| json!([]));
                Ok(json!({ "memories": memories, "pain": [] }))
            }
            #[cfg(feature = "local")]
            Backend::Local(client) => {
                client
                    .post("/v1/context", json!({ "prompt": prompt, "limit": 8 }))
                    .await
            }
        }
    }

    /// Store a completed turn through the full process_turn pipeline.
    pub async fn store_turn(
        &self,
        user_message: &str,
        assistant_response: &str,
        turn_id: u64,
        project_context: Option<String>,
    ) -> anyhow::Result<()> {
        let args = json!({
            "user_message": user_message,
            "assistant_response": assistant_response,
            "turn_id": turn_id,
            "project_context": project_context,
        });
        match self {
            Backend::Cloud(client) => {
                client
                    .call_tool("process_turn", args)
                    .await
                    .map_err(|e| anyhow::anyhow!(e))?;
                Ok(())
            }
            #[cfg(feature = "local")]
            Backend::Local(client) => {
                client.post("/v1/turn", args).await?;
                Ok(())
            }
        }
    }

    /// Store a lightweight action note captured live during a session.
    pub async fn store_note(&self, content: &str, project: Option<String>) -> anyhow::Result<()> {
        match self {
            Backend::Cloud(client) => {
                // Cloud has no low-level note endpoint; store_memory is the
                // closest and is already persisted server-side.
                let mut tags = vec!["action".to_string()];
                if let Some(p) = &project {
                    tags.push(format!("scope:project:{p}"));
                }
                client
                    .call_tool(
                        "store_memory",
                        json!({
                            "content": content,
                            "memory_type": "episodic",
                            "tags": tags,
                        }),
                    )
                    .await
                    .map_err(|e| anyhow::anyhow!(e))?;
                Ok(())
            }
            #[cfg(feature = "local")]
            Backend::Local(client) => {
                client
                    .post(
                        "/v1/note",
                        json!({ "content": content, "project": project }),
                    )
                    .await?;
                Ok(())
            }
        }
    }

    /// Flush in-memory engine state to disk before context loss. No-op for
    /// cloud, where every write is already persisted server-side.
    pub async fn flush(&self) -> anyhow::Result<()> {
        match self {
            Backend::Cloud(_) => Ok(()),
            #[cfg(feature = "local")]
            Backend::Local(client) => {
                client.post("/v1/flush", json!({})).await?;
                Ok(())
            }
        }
    }

    /// Session-start context: {profile, always: [...]}.
    pub async fn session_context(&self) -> anyhow::Result<serde_json::Value> {
        match self {
            Backend::Cloud(client) => {
                // The cloud API has no profile endpoint; surface the most
                // relevant standing memories via search instead.
                let resp = client
                    .call_tool(
                        "search_memories",
                        json!({ "query": "user profile preferences standing rules", "limit": 10 }),
                    )
                    .await
                    .map_err(|e| anyhow::anyhow!(e))?;
                let text = resp
                    .content
                    .first()
                    .map(|c| c.text.clone())
                    .unwrap_or_default();
                let parsed: serde_json::Value =
                    serde_json::from_str(&text).unwrap_or_else(|_| json!({}));
                let always: Vec<serde_json::Value> = parsed
                    .get("results")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|m| m.get("content").cloned())
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(json!({ "profile": null, "always": always }))
            }
            #[cfg(feature = "local")]
            Backend::Local(client) => client.post("/v1/session-context", json!({})).await,
        }
    }
}

/// Thin HTTP client for the local hook daemon, spawning it when absent.
#[cfg(feature = "local")]
pub struct LocalDaemonClient {
    port: u16,
    token: String,
    http: reqwest::Client,
}

#[cfg(feature = "local")]
impl LocalDaemonClient {
    /// Total budget for daemon startup. First-ever start downloads the
    /// embedding model and can exceed this; the hook then serves nothing for
    /// that turn and connects on the next one.
    const SPAWN_WAIT: std::time::Duration = std::time::Duration::from_secs(20);

    pub async fn connect_or_spawn(data_dir: &Path) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(25))
            .build()?;

        if let Some(client) = Self::try_connect(data_dir, &http).await {
            return Ok(client);
        }

        Self::spawn_daemon(data_dir)?;

        let deadline = std::time::Instant::now() + Self::SPAWN_WAIT;
        while std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            if let Some(client) = Self::try_connect(data_dir, &http).await {
                return Ok(client);
            }
        }
        anyhow::bail!(
            "hook daemon did not become healthy within {:?} (first start downloads the embedding model; it will be ready next turn)",
            Self::SPAWN_WAIT
        )
    }

    async fn try_connect(data_dir: &Path, http: &reqwest::Client) -> Option<Self> {
        let info = crate::daemon::read_info(data_dir)?;
        let url = format!("http://127.0.0.1:{}/health", info.port);
        let ok = http
            .get(&url)
            .timeout(std::time::Duration::from_secs(1))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        if !ok {
            return None;
        }
        Some(Self {
            port: info.port,
            token: info.token,
            http: http.clone(),
        })
    }

    fn spawn_daemon(data_dir: &Path) -> anyhow::Result<()> {
        let exe: std::path::PathBuf = std::env::current_exe()?;
        std::process::Command::new(exe)
            .arg("--data-dir")
            .arg(data_dir)
            .arg("daemon")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| anyhow::anyhow!("failed to spawn hook daemon: {e}"))?;
        Ok(())
    }

    pub async fn post(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        let url = format!("http://127.0.0.1:{}{}", self.port, path);
        let resp = self
            .http
            .post(&url)
            .header("x-mentedb-token", &self.token)
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            anyhow::bail!("daemon returned HTTP {}", resp.status());
        }
        Ok(resp.json().await?)
    }
}
