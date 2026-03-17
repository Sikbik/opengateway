use super::adapters::AgentKind;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionParams {
    pub cwd: PathBuf,
    pub mcp_servers: Vec<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptParams {
    pub session_id: String,
    pub prompt: Vec<PromptBlock>,
    #[serde(default)]
    pub message_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelParams {
    pub session_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptBlock {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InProcessMockSession {
    pub id: String,
    pub agent: AgentKind,
    pub cwd: PathBuf,
    pub mcp_server_count: usize,
    pub prompt_count: u64,
}

impl InProcessMockSession {
    pub fn new(id: String, agent: AgentKind, params: NewSessionParams) -> Self {
        Self {
            id,
            agent,
            cwd: params.cwd,
            mcp_server_count: params.mcp_servers.len(),
            prompt_count: 0,
        }
    }

    pub fn build_mock_reply(&mut self, prompt: &PromptParams) -> String {
        self.prompt_count += 1;
        let text_blocks = prompt
            .prompt
            .iter()
            .filter(|block| block.kind == "text")
            .filter_map(|block| block.text.as_deref())
            .collect::<Vec<_>>();
        let resource_links = prompt
            .prompt
            .iter()
            .filter(|block| block.kind == "resource_link")
            .count();

        if !text_blocks.is_empty() {
            return format!(
                "Mock {} session {} in {} received: {}",
                self.agent.as_str(),
                self.id,
                self.cwd.display(),
                text_blocks.join(" ")
            );
        }

        format!(
            "Mock {} session {} received {} prompt block(s) with {} resource link(s); MCP servers attached: {}.",
            self.agent.as_str(),
            self.id,
            prompt.prompt.len(),
            resource_links,
            self.mcp_server_count
        )
    }
}

pub fn session_states() -> Vec<String> {
    vec![
        "starting".to_string(),
        "running".to_string(),
        "cancelling".to_string(),
        "exited".to_string(),
        "failed".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{InProcessMockSession, NewSessionParams, PromptBlock, PromptParams};
    use crate::acp::adapters::AgentKind;
    use std::path::PathBuf;

    #[test]
    fn mock_reply_prefers_text_content() {
        let mut session = InProcessMockSession {
            id: "session-000001".to_string(),
            agent: AgentKind::Codex,
            cwd: PathBuf::from("/workspace"),
            mcp_server_count: 1,
            prompt_count: 0,
        };
        let reply = session.build_mock_reply(&PromptParams {
            session_id: session.id.clone(),
            prompt: vec![
                PromptBlock {
                    kind: "text".to_string(),
                    text: Some("hello".to_string()),
                },
                PromptBlock {
                    kind: "resource_link".to_string(),
                    text: None,
                },
            ],
            message_id: Some("123".to_string()),
        });
        assert!(reply.contains("received: hello"));
        assert_eq!(session.prompt_count, 1);
    }

    #[test]
    fn new_mock_session_tracks_mcp_server_count() {
        let session = InProcessMockSession::new(
            "session-000001".to_string(),
            AgentKind::Codex,
            NewSessionParams {
                cwd: PathBuf::from("/tmp"),
                mcp_servers: vec![serde_json::json!({"name": "demo"})],
            },
        );
        assert_eq!(session.mcp_server_count, 1);
    }
}
