use reqwest::Client;
use serde::{Deserialize, Serialize};

/// HTTP client for the MenteDB cloud API.
/// Proxies MCP tool calls to the cloud backend, eliminating the need for a local database.
pub struct CloudClient {
    client: Client,
    api_url: String,
    token: String,
}

#[derive(Serialize)]
struct ToolCallRequest {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Deserialize)]
pub struct ToolCallResponse {
    pub content: Vec<ToolContent>,
    pub is_error: bool,
}

#[derive(Deserialize)]
pub struct ToolContent {
    #[serde(rename = "type")]
    pub _content_type: String,
    pub text: String,
}

impl CloudClient {
    pub fn new(api_url: String, token: String) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client");

        Self {
            client,
            api_url,
            token,
        }
    }

    /// Call a tool on the cloud API.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResponse, String> {
        let url = format!("{}/mcp/v1/tools/call", self.api_url);

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .json(&ToolCallRequest {
                name: name.to_string(),
                arguments,
            })
            .send()
            .await
            .map_err(|e| format!("cloud request failed: {e}"))?;

        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err("authentication failed: token may be revoked. Run `mentedb-mcp login` to re-authenticate.".to_string());
        }

        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("cloud API error (HTTP {status}): {body}"));
        }

        resp.json::<ToolCallResponse>()
            .await
            .map_err(|e| format!("failed to parse cloud response: {e}"))
    }
}
