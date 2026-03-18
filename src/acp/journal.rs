use anyhow::{Context, Result};
use crate::paths::AppPaths;
use serde_json::{json, Value};
use std::cmp::Reverse;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SessionFiles {
    pub journal_path: PathBuf,
    pub log_path: PathBuf,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SessionSummary {
    pub session_id: String,
    pub prompt_count: usize,
    pub cwd: Option<String>,
    pub last_event: Option<String>,
    pub journal_path: PathBuf,
    pub log_path: PathBuf,
    pub last_timestamp_ms: Option<u128>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SessionEventRecord {
    pub timestamp_ms: Option<u128>,
    pub event: Option<String>,
    pub data: Value,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SessionDetail {
    pub summary: SessionSummary,
    pub recent_events: Vec<SessionEventRecord>,
    pub recent_logs: Vec<String>,
}

pub fn scaffold_session_files(paths: &AppPaths) -> SessionFiles {
    session_files(paths, "scaffold")
}

pub fn session_files(paths: &AppPaths, session_id: &str) -> SessionFiles {
    let safe_session_id = sanitize_session_id(session_id);
    SessionFiles {
        journal_path: paths
            .acp_sessions_dir
            .join(format!("{safe_session_id}.jsonl")),
        log_path: paths.acp_logs_dir.join(format!("{safe_session_id}.log")),
    }
}

pub fn collect_session_summaries(paths: &AppPaths) -> Result<Vec<SessionSummary>> {
    let mut sessions = Vec::new();
    for entry in fs::read_dir(&paths.acp_sessions_dir)
        .with_context(|| format!("failed to read {}", paths.acp_sessions_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
            continue;
        }

        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };

        let files = session_files(paths, stem);
        let mut summary = SessionSummary {
            session_id: stem.to_string(),
            prompt_count: 0,
            cwd: None,
            last_event: None,
            journal_path: files.journal_path,
            log_path: files.log_path,
            last_timestamp_ms: None,
        };

        let contents = fs::read_to_string(&summary.journal_path)
            .with_context(|| format!("failed to read {}", summary.journal_path.display()))?;
        for line in contents.lines().filter(|line| !line.trim().is_empty()) {
            let value: Value = serde_json::from_str(line)
                .with_context(|| format!("invalid ACP journal line in {}", summary.journal_path.display()))?;
            if let Some(ts) = value.get("ts").and_then(Value::as_u64) {
                summary.last_timestamp_ms = Some(ts as u128);
            }
            if let Some(event) = value.get("event").and_then(Value::as_str) {
                summary.last_event = Some(event.to_string());
                if event == "prompt.completed" {
                    summary.prompt_count += 1;
                }
            }
            if summary.cwd.is_none() {
                summary.cwd = value
                    .get("data")
                    .and_then(|data| data.get("cwd"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
        }

        sessions.push(summary);
    }

    sessions.sort_by_key(|session| {
        Reverse((
            session.last_timestamp_ms.unwrap_or_default(),
            session.session_id.clone(),
        ))
    });
    Ok(sessions)
}

pub fn inspect_session(
    paths: &AppPaths,
    session_id: &str,
    limit: usize,
) -> Result<Option<SessionDetail>> {
    let Some(summary) = collect_session_summaries(paths)?
        .into_iter()
        .find(|summary| summary.session_id == session_id)
    else {
        return Ok(None);
    };

    let events = read_recent_session_events(&summary.journal_path, limit)?;
    let logs = read_recent_log_lines(&summary.log_path, limit)?;
    Ok(Some(SessionDetail {
        summary,
        recent_events: events,
        recent_logs: logs,
    }))
}

pub fn append_journal_event(
    paths: &AppPaths,
    session_id: &str,
    event: &str,
    data: Value,
) -> std::io::Result<()> {
    let files = session_files(paths, session_id);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(files.journal_path)?;
    serde_json::to_writer(
        &mut file,
        &json!({
            "ts": unix_timestamp_ms(),
            "event": event,
            "data": data,
        }),
    )?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

pub fn append_session_log(
    paths: &AppPaths,
    session_id: &str,
    message: &str,
) -> std::io::Result<()> {
    let files = session_files(paths, session_id);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(files.log_path)?;
    writeln!(file, "[{}] {}", unix_timestamp_ms(), message)?;
    file.flush()?;
    Ok(())
}

fn sanitize_session_id(session_id: &str) -> String {
    let mut out = String::with_capacity(session_id.len());
    for ch in session_id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }

    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "session".to_string()
    } else {
        trimmed.to_string()
    }
}

fn read_recent_session_events(path: &PathBuf, limit: usize) -> Result<Vec<SessionEventRecord>> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let lines = tail_lines(&contents, limit);
    let mut events = Vec::with_capacity(lines.len());
    for line in lines {
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("invalid ACP journal line in {}", path.display()))?;
        events.push(SessionEventRecord {
            timestamp_ms: value.get("ts").and_then(Value::as_u64).map(u128::from),
            event: value
                .get("event")
                .and_then(Value::as_str)
                .map(str::to_string),
            data: value.get("data").cloned().unwrap_or(Value::Null),
        });
    }
    Ok(events)
}

fn read_recent_log_lines(path: &PathBuf, limit: usize) -> Result<Vec<String>> {
    match fs::read_to_string(path) {
        Ok(contents) => Ok(tail_lines(&contents, limit)
            .into_iter()
            .map(str::to_string)
            .collect()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(error).with_context(|| format!("failed to read {}", path.display())),
    }
}

fn tail_lines(contents: &str, limit: usize) -> Vec<&str> {
    if limit == 0 {
        return Vec::new();
    }
    let lines = contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    let start = lines.len().saturating_sub(limit);
    lines.into_iter().skip(start).collect()
}

fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::{
        append_journal_event, append_session_log, collect_session_summaries, inspect_session,
        session_files, tail_lines,
    };
    use crate::paths::build_paths;
    use serde_json::json;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn session_files_sanitize_unfriendly_ids() {
        let paths = build_paths().expect("paths");
        let files = session_files(&paths, "session:01/example");
        assert!(files.journal_path.to_string_lossy().contains("session_01_example"));
        assert!(files.log_path.to_string_lossy().contains("session_01_example"));
    }

    #[test]
    fn append_helpers_create_files() {
        let mut paths = build_paths().expect("paths");
        let test_root = std::env::temp_dir().join(format!(
            "opengateway-acp-journal-test-{}",
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

        append_journal_event(&paths, "session-1", "session.created", json!({"cwd": "/tmp"}))
            .expect("journal event");
        append_session_log(&paths, "session-1", "session created").expect("session log");

        let files = session_files(&paths, "session-1");
        let journal = fs::read_to_string(files.journal_path).expect("read journal");
        let log = fs::read_to_string(files.log_path).expect("read log");

        assert!(journal.contains("\"event\":\"session.created\""));
        assert!(log.contains("session created"));

        let _ = fs::remove_dir_all(test_root);
    }

    #[test]
    fn collect_session_summaries_reads_prompt_count_and_cwd() {
        let mut paths = build_paths().expect("paths");
        let test_root = std::env::temp_dir().join(format!(
            "opengateway-acp-summary-test-{}",
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

        append_journal_event(&paths, "session-1", "session.created", json!({"cwd": "/tmp"}))
            .expect("journal event");
        append_journal_event(
            &paths,
            "session-1",
            "prompt.completed",
            json!({"promptBlocks": 1}),
        )
        .expect("prompt event");

        let summaries = collect_session_summaries(&paths).expect("summaries");
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].session_id, "session-1");
        assert_eq!(summaries[0].prompt_count, 1);
        assert_eq!(summaries[0].cwd.as_deref(), Some("/tmp"));
        assert_eq!(summaries[0].last_event.as_deref(), Some("prompt.completed"));

        let _ = fs::remove_dir_all(test_root);
    }

    #[test]
    fn inspect_session_reads_recent_events_and_logs() {
        let mut paths = build_paths().expect("paths");
        let test_root = std::env::temp_dir().join(format!(
            "opengateway-acp-inspect-test-{}",
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

        append_journal_event(&paths, "session-1", "session.created", json!({"cwd": "/tmp"}))
            .expect("journal event");
        append_journal_event(&paths, "session-1", "prompt.completed", json!({"promptBlocks": 1}))
            .expect("prompt event");
        append_session_log(&paths, "session-1", "line one").expect("log line one");
        append_session_log(&paths, "session-1", "line two").expect("log line two");

        let detail = inspect_session(&paths, "session-1", 1)
            .expect("detail")
            .expect("session present");
        assert_eq!(detail.summary.session_id, "session-1");
        assert_eq!(detail.recent_events.len(), 1);
        assert_eq!(
            detail.recent_events[0].event.as_deref(),
            Some("prompt.completed")
        );
        assert_eq!(detail.recent_logs.len(), 1);
        assert!(detail.recent_logs[0].contains("line two"));

        let _ = fs::remove_dir_all(test_root);
    }

    #[test]
    fn tail_lines_returns_last_non_empty_lines() {
        let lines = tail_lines("a\n\nb\nc\n", 2);
        assert_eq!(lines, vec!["b", "c"]);
    }
}
