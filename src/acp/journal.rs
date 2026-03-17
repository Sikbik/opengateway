use crate::paths::AppPaths;
use std::path::PathBuf;

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

#[cfg(test)]
mod tests {
    use super::session_files;
    use crate::paths::build_paths;

    #[test]
    fn session_files_sanitize_unfriendly_ids() {
        let paths = build_paths().expect("paths");
        let files = session_files(&paths, "session:01/example");
        assert!(files.journal_path.to_string_lossy().contains("session_01_example"));
        assert!(files.log_path.to_string_lossy().contains("session_01_example"));
    }
}
