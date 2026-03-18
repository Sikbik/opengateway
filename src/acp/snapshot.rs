use super::adapters::{codex, supported_agents, AgentKind};
use super::capabilities::{planned_capabilities, PlannedCapabilities};
use super::journal::{collect_metrics_summary, collect_session_summaries, AcpMetricsSummary, SessionSummary};
use super::protocol::JSONRPC_VERSION;
use super::supervisor::PROCESS_MODEL;
use super::transport::TRANSPORT_NAME;
use crate::paths::AppPaths;
use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct AcpSnapshot {
    pub experimental: bool,
    pub command: &'static str,
    pub transport: &'static str,
    pub jsonrpc: &'static str,
    pub process_model: &'static str,
    pub paths: AcpSnapshotPaths,
    pub capabilities: PlannedCapabilities,
    pub metrics: AcpMetricsSummary,
    pub recorded_session_count: usize,
    pub agents: Vec<AcpAgentSnapshot>,
    pub sessions: Vec<SessionSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AcpSnapshotPaths {
    pub acp_root: String,
    pub sessions: String,
    pub logs: String,
    pub tmp: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AcpAgentSnapshot {
    pub kind: &'static str,
    pub runtime_name: &'static str,
    pub status: &'static str,
    pub note: &'static str,
    pub ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issue: Option<String>,
    pub guidance: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executable_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_json: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_ephemeral: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_skip_git_repo_check: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_cwd_flag: Option<bool>,
}

pub fn build_snapshot(paths: &AppPaths) -> Result<AcpSnapshot> {
    let sessions = collect_session_summaries(paths)?;
    let metrics = collect_metrics_summary(paths)?;
    let codex_runtime = codex::inspect_runtime();
    let agents = supported_agents()
        .iter()
        .map(|agent| match agent.kind {
            AgentKind::Codex => AcpAgentSnapshot {
                kind: agent.kind.as_str(),
                runtime_name: agent.runtime_name,
                status: agent.status,
                note: agent.note,
                ready: codex_runtime.ready,
                issue: codex_runtime.issue.clone(),
                guidance: codex::doctor_guidance(&codex_runtime),
                executable_path: codex_runtime
                    .executable_path
                    .as_ref()
                    .map(|path| path.display().to_string()),
                version: codex_runtime.version.clone(),
                supports_json: Some(codex_runtime.supports_json),
                supports_ephemeral: Some(codex_runtime.supports_ephemeral),
                supports_skip_git_repo_check: Some(codex_runtime.supports_skip_git_repo_check),
                supports_cwd_flag: Some(codex_runtime.supports_cwd_flag),
            },
            AgentKind::Claude => AcpAgentSnapshot {
                kind: agent.kind.as_str(),
                runtime_name: agent.runtime_name,
                status: agent.status,
                note: agent.note,
                ready: false,
                issue: Some("adapter not implemented".to_string()),
                guidance: vec![
                    "Wait for the Claude adapter before configuring a Claude ACP harness."
                        .to_string(),
                ],
                executable_path: None,
                version: None,
                supports_json: None,
                supports_ephemeral: None,
                supports_skip_git_repo_check: None,
                supports_cwd_flag: None,
            },
        })
        .collect();

    Ok(AcpSnapshot {
        experimental: true,
        command: "opengateway acp",
        transport: TRANSPORT_NAME,
        jsonrpc: JSONRPC_VERSION,
        process_model: PROCESS_MODEL,
        paths: AcpSnapshotPaths {
            acp_root: paths.acp_dir.display().to_string(),
            sessions: paths.acp_sessions_dir.display().to_string(),
            logs: paths.acp_logs_dir.display().to_string(),
            tmp: paths.acp_tmp_dir.display().to_string(),
        },
        capabilities: planned_capabilities(),
        metrics,
        recorded_session_count: sessions.len(),
        agents,
        sessions,
    })
}

#[cfg(test)]
mod tests {
    use super::build_snapshot;
    use crate::acp::journal::append_journal_event;
    use crate::paths::build_paths;
    use serde_json::json;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn snapshot_includes_agents_metrics_and_sessions() {
        let mut paths = build_paths().expect("paths");
        let test_root = std::env::temp_dir().join(format!(
            "opengateway-acp-snapshot-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        paths.acp_dir = test_root.clone();
        paths.acp_sessions_dir = test_root.join("sessions");
        paths.acp_logs_dir = test_root.join("logs");
        paths.acp_tmp_dir = test_root.join("tmp");
        fs::create_dir_all(&paths.acp_sessions_dir).expect("sessions dir");
        fs::create_dir_all(&paths.acp_logs_dir).expect("logs dir");
        append_journal_event(&paths, "session-000001", "session.created", json!({"cwd": "/tmp"}))
            .expect("journal");

        let snapshot = build_snapshot(&paths).expect("snapshot");
        assert!(snapshot.experimental);
        assert_eq!(snapshot.command, "opengateway acp");
        assert_eq!(snapshot.recorded_session_count, 1);
        assert_eq!(snapshot.metrics.sessions_created, 1);
        assert_eq!(snapshot.sessions.len(), 1);
        assert_eq!(snapshot.agents[0].kind, "codex");
        assert_eq!(snapshot.agents[1].kind, "claude");

        let _ = fs::remove_dir_all(test_root);
    }
}
