use super::adapters::supported_agents;
use super::capabilities::planned_capabilities;
use super::errors::error_categories;
use super::journal::{collect_session_summaries, scaffold_session_files};
use super::protocol::JSONRPC_VERSION;
use super::redact::redact_text;
use super::session::session_states;
use super::supervisor::PROCESS_MODEL;
use super::transport::TRANSPORT_NAME;
use crate::paths::AppPaths;

pub fn render_doctor_report(paths: &AppPaths) -> String {
    let capabilities = planned_capabilities();
    let sample_files = scaffold_session_files(paths);
    let agents = supported_agents()
        .iter()
        .map(|agent| format!("{} ({})", agent.runtime_name, agent.status))
        .collect::<Vec<_>>()
        .join(", ");
    let states = session_states().join(", ");
    let redaction_ready =
        redact_text("Authorization: Bearer acp-scaffold-secret") != "Authorization: Bearer acp-scaffold-secret";
    let session_summaries = collect_session_summaries(paths).unwrap_or_default();
    let latest_session = session_summaries.first();

    format!(
        "\
ACP doctor
experimental: yes
command: opengateway acp
transport: {transport}
jsonrpc: {jsonrpc}
process-model: {process_model}
supported-agents: {agents}
planned-capabilities:
  initialize: {initialize}
  session/new: {new_session}
  session/prompt: {prompt}
  session/cancel: {cancel}
  loadSession: {load_session}
  streaming: {streaming}
paths:
  acp-root: {acp_root}
  sessions: {sessions}
  logs: {logs}
  tmp: {tmp}
  scaffold-journal: {journal}
  scaffold-log: {session_log}
recorded-sessions: {session_count}
latest-session: {latest_session}
latest-event: {latest_event}
session-states: {states}
error-categories: {error_categories}
redaction-ready: {redaction_ready}
",
        transport = TRANSPORT_NAME,
        jsonrpc = JSONRPC_VERSION,
        process_model = PROCESS_MODEL,
        agents = agents,
        initialize = yes_no(capabilities.initialize),
        new_session = yes_no(capabilities.sessions.new_session),
        prompt = yes_no(capabilities.sessions.prompt),
        cancel = yes_no(capabilities.sessions.cancel),
        load_session = yes_no(capabilities.sessions.load_session),
        streaming = yes_no(capabilities.sessions.streaming_updates),
        acp_root = paths.acp_dir.display(),
        sessions = paths.acp_sessions_dir.display(),
        logs = paths.acp_logs_dir.display(),
        tmp = paths.acp_tmp_dir.display(),
        journal = sample_files.journal_path.display(),
        session_log = sample_files.log_path.display(),
        session_count = session_summaries.len(),
        latest_session = latest_session
            .map(|session| session.session_id.as_str())
            .unwrap_or("none"),
        latest_event = latest_session
            .and_then(|session| session.last_event.as_deref())
            .unwrap_or("none"),
        states = states,
        error_categories = error_categories().join(", "),
        redaction_ready = yes_no(redaction_ready),
    )
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

#[cfg(test)]
mod tests {
    use super::render_doctor_report;
    use crate::paths::build_paths;

    #[test]
    fn doctor_report_mentions_experimental_status() {
        let paths = build_paths().expect("paths");
        let report = render_doctor_report(&paths);
        assert!(report.contains("experimental: yes"));
        assert!(report.contains("loadSession: no"));
        assert!(report.contains("recorded-sessions:"));
    }
}
