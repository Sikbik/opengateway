use crate::paths::AppPaths;
use serde_json::{json, Value};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SessionFiles {
    pub journal_path: PathBuf,
    pub log_path: PathBuf,
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

fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::{append_journal_event, append_session_log, session_files};
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
}
