use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const PROBE_COMMAND_TIMEOUT: Duration = Duration::from_secs(2);
const PROBE_COMMAND_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DroidReadiness {
    pub executable: Option<String>,
    pub version: Option<String>,
    pub supports_exec: bool,
    pub supports_stream_jsonrpc: bool,
    pub supports_daemon_ipc: bool,
    pub issue: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CodexReadiness {
    pub executable: Option<String>,
    pub version: Option<String>,
    pub supports_app_server: bool,
    pub supports_generate_schema: bool,
    pub issue: Option<String>,
}

pub fn probe_droid_cli(preferred: Option<PathBuf>) -> DroidReadiness {
    let executable = preferred.unwrap_or_else(default_droid_executable);
    let executable_label = executable.display().to_string();

    let version = command_stdout(&executable, &["--version"]).ok();
    let exec_help = command_stdout(&executable, &["exec", "--help"]).ok();
    let daemon_help = command_stdout(&executable, &["daemon", "--help"]).ok();

    let supports_exec = exec_help.as_deref().is_some_and(droid_exec_supported);
    let supports_stream_jsonrpc = exec_help
        .as_deref()
        .is_some_and(droid_stream_jsonrpc_supported);
    let supports_daemon_ipc = daemon_help
        .as_deref()
        .is_some_and(droid_daemon_ipc_supported);
    let issue = if version.is_none() {
        Some("Droid CLI was not found or could not be executed".to_string())
    } else if !supports_exec {
        Some("Droid CLI does not expose exec mode".to_string())
    } else if !supports_stream_jsonrpc {
        Some("Droid exec does not expose stream-jsonrpc input".to_string())
    } else if !supports_daemon_ipc {
        Some("Droid daemon does not expose IPC listener support".to_string())
    } else {
        None
    };

    DroidReadiness {
        executable: Some(executable_label),
        version: version.map(clean_version),
        supports_exec,
        supports_stream_jsonrpc,
        supports_daemon_ipc,
        issue,
    }
}

pub fn probe_codex_cli() -> CodexReadiness {
    let executable = default_codex_executable();
    let version = command_stdout(&executable, &["--version"]).ok();
    let app_server_help = command_stdout(&executable, &["app-server", "--help"]).ok();

    let supports_app_server = app_server_help
        .as_deref()
        .is_some_and(codex_app_server_supported);
    let supports_generate_schema = app_server_help
        .as_deref()
        .is_some_and(codex_generate_schema_supported);
    let issue = if version.is_none() {
        Some("Codex CLI was not found or could not be executed".to_string())
    } else if !supports_app_server {
        Some("Codex CLI does not expose app-server".to_string())
    } else {
        None
    };

    CodexReadiness {
        executable: Some(executable.display().to_string()),
        version: version.map(clean_version),
        supports_app_server,
        supports_generate_schema,
        issue,
    }
}

fn default_droid_executable() -> PathBuf {
    PathBuf::from(if cfg!(windows) { "droid.exe" } else { "droid" })
}

fn default_codex_executable() -> PathBuf {
    PathBuf::from(if cfg!(windows) { "codex.exe" } else { "codex" })
}

fn command_stdout(executable: &Path, args: &[&str]) -> Result<String, String> {
    command_stdout_with_env(executable, args, None)
}

fn command_stdout_with_env(
    executable: &Path,
    args: &[&str],
    env: Option<(&str, &str)>,
) -> Result<String, String> {
    let mut command = Command::new(executable);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some((name, value)) = env {
        command.env(name, value);
    }

    let mut child = command.spawn().map_err(|err| err.to_string())?;

    let started_at = Instant::now();
    loop {
        if child.try_wait().map_err(|err| err.to_string())?.is_some() {
            break;
        }

        if started_at.elapsed() >= PROBE_COMMAND_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "{} timed out after {} seconds",
                executable.display(),
                PROBE_COMMAND_TIMEOUT.as_secs()
            ));
        }

        thread::sleep(PROBE_COMMAND_POLL_INTERVAL);
    }

    let output = child.wait_with_output().map_err(|err| err.to_string())?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

fn clean_version(raw: String) -> String {
    raw.trim().to_string()
}

fn droid_exec_supported(help: &str) -> bool {
    help.contains("Run a prompt") || help.contains("exec")
}

fn droid_stream_jsonrpc_supported(help: &str) -> bool {
    help.contains("--input-format") && help.contains("stream-jsonrpc")
}

fn droid_daemon_ipc_supported(help: &str) -> bool {
    help.contains("--listen") && help.contains("ipc") && help.contains("--enable-child-ipc")
}

fn codex_app_server_supported(help: &str) -> bool {
    help.contains("app-server") || help.contains("generate-schema")
}

fn codex_generate_schema_supported(help: &str) -> bool {
    help.contains("generate-json-schema") || help.contains("generate-schema")
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SLEEP_ENV: &str = "OPENGATEWAY_TOOL_PROBE_TEST_SLEEP_MS";

    const DROID_EXEC_HELP: &str = r#"
Run a prompt
      --input-format <INPUT_FORMAT>
          Input format. One of: text, json, stream-jsonrpc
"#;

    const DROID_DAEMON_HELP: &str = r#"
Usage: droid daemon --listen <LISTEN>
      --listen <LISTEN>
      --enable-child-ipc
      ipc
"#;

    const CODEX_APP_SERVER_HELP: &str = r#"
Usage: codex app-server [OPTIONS]
      --generate-json-schema
"#;

    #[test]
    fn parses_droid_exec_stream_jsonrpc_support() {
        assert!(droid_exec_supported(DROID_EXEC_HELP));
        assert!(droid_stream_jsonrpc_supported(DROID_EXEC_HELP));
    }

    #[test]
    fn parses_droid_daemon_ipc_support() {
        assert!(droid_daemon_ipc_supported(DROID_DAEMON_HELP));
    }

    #[test]
    fn parses_codex_app_server_support() {
        assert!(codex_app_server_supported(CODEX_APP_SERVER_HELP));
        assert!(codex_generate_schema_supported(CODEX_APP_SERVER_HELP));
    }

    #[test]
    fn rejects_codex_app_server_help_without_schema_support() {
        assert!(!codex_generate_schema_supported(
            "Usage: codex app-server [OPTIONS]"
        ));
    }

    #[test]
    fn rejects_incomplete_droid_exec_help() {
        assert!(!droid_stream_jsonrpc_supported("--input-format text"));
    }

    #[test]
    fn command_stdout_times_out_long_running_commands() {
        let current_exe = std::env::current_exe().unwrap();
        let err = command_stdout_with_env(
            &current_exe,
            &[
                "--exact",
                "tool_probe::tests::tool_probe_timeout_sleep_helper",
                "--nocapture",
            ],
            Some((TEST_SLEEP_ENV, "5000")),
        )
        .unwrap_err();

        assert!(err.contains("timed out"));
    }

    #[test]
    fn tool_probe_timeout_sleep_helper() {
        if let Ok(raw) = std::env::var(TEST_SLEEP_ENV) {
            let sleep_ms = raw.parse::<u64>().unwrap();
            thread::sleep(Duration::from_millis(sleep_ms));
        }
    }

    #[test]
    fn defaults_to_windows_executable_names_on_windows() {
        if cfg!(windows) {
            assert_eq!(default_droid_executable(), PathBuf::from("droid.exe"));
            assert_eq!(default_codex_executable(), PathBuf::from("codex.exe"));
        } else {
            assert_eq!(default_droid_executable(), PathBuf::from("droid"));
            assert_eq!(default_codex_executable(), PathBuf::from("codex"));
        }
    }
}
