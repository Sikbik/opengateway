use anyhow::{anyhow, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

const DROID_SMOKE_TIMEOUT: Duration = Duration::from_secs(90);
const DROID_SMOKE_POLL_INTERVAL: Duration = Duration::from_millis(100);
pub const DROID_SMOKE_PROMPT: &str = "Reply exactly OK. Do not inspect or modify files.";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DroidSmokeResult {
    pub executable: String,
    pub cwd: String,
    pub model: String,
    pub output: String,
    pub session_path: String,
}

pub fn run_factory_droid_smoke(
    factory_home: &Path,
    executable: &Path,
    workspace: &Path,
    model: &str,
) -> Result<DroidSmokeResult> {
    let started_at = SystemTime::now();
    let cwd = droid_cwd_for_workspace(workspace);
    let output = run_droid_exec(executable, &cwd, model)?;
    let trimmed = output.trim();
    if trimmed != "OK" {
        return Err(anyhow!("Droid smoke expected `OK`, got `{trimmed}`"));
    }

    let session_path = find_matching_session(
        &factory_home.join("sessions"),
        &cwd,
        DROID_SMOKE_PROMPT,
        started_at,
    )
    .ok_or_else(|| {
        anyhow!("Droid smoke completed, but no matching Factory session file was found for {cwd}")
    })?
    .display()
    .to_string();

    Ok(DroidSmokeResult {
        executable: executable.display().to_string(),
        cwd,
        model: model.to_string(),
        output: trimmed.to_string(),
        session_path,
    })
}

pub fn format_droid_smoke_result(result: &DroidSmokeResult) -> String {
    format!(
        "Factory Droid probe passed.\n- executable: {}\n- cwd: {}\n- model: {}\n- output: {}\n- session: {}",
        result.executable, result.cwd, result.model, result.output, result.session_path
    )
}

fn run_droid_exec(executable: &Path, cwd: &str, model: &str) -> Result<String> {
    let mut child = Command::new(executable)
        .args(["exec", "--cwd", cwd, "-m", model, DROID_SMOKE_PROMPT])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to launch Droid at {}", executable.display()))?;

    let started_at = Instant::now();
    loop {
        if child
            .try_wait()
            .context("failed to poll Droid smoke")?
            .is_some()
        {
            break;
        }
        if started_at.elapsed() >= DROID_SMOKE_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            return Err(anyhow!(
                "Droid smoke timed out after {} seconds",
                DROID_SMOKE_TIMEOUT.as_secs()
            ));
        }
        thread::sleep(DROID_SMOKE_POLL_INTERVAL);
    }

    let output = child
        .wait_with_output()
        .context("failed to collect Droid smoke output")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr).trim().to_string();
    if output.status.success() {
        Ok(stdout.to_string())
    } else if combined.is_empty() {
        Err(anyhow!("Droid smoke failed with status {}", output.status))
    } else {
        Err(anyhow!(combined))
    }
}

fn droid_cwd_for_workspace(path: &Path) -> String {
    let distro = env::var("WSL_DISTRO_NAME").ok();
    droid_cwd_for_workspace_with_distro(path, distro.as_deref())
}

fn droid_cwd_for_workspace_with_distro(path: &Path, distro: Option<&str>) -> String {
    let raw = path.display().to_string();
    let Some(distro) = distro.filter(|value| !value.trim().is_empty()) else {
        return raw;
    };
    if !raw.starts_with('/') {
        return raw;
    }

    let suffix = raw.trim_start_matches('/').replace('/', r"\");
    format!(r"\\wsl.localhost\{distro}\{suffix}")
}

fn find_matching_session(
    sessions_dir: &Path,
    expected_cwd: &str,
    prompt: &str,
    started_at: SystemTime,
) -> Option<PathBuf> {
    let entries = fs::read_dir(sessions_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        for file_entry in fs::read_dir(&path).ok()?.flatten() {
            let file_path = file_entry.path();
            if file_path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
                continue;
            }
            if file_entry
                .metadata()
                .ok()
                .and_then(|metadata| metadata.modified().ok())
                .map(|modified| modified < started_at)
                .unwrap_or(true)
            {
                continue;
            }
            if session_file_matches(&file_path, expected_cwd, prompt) {
                return Some(file_path);
            }
        }
    }
    None
}

fn session_file_matches(path: &Path, expected_cwd: &str, prompt: &str) -> bool {
    let Ok(raw) = fs::read_to_string(path) else {
        return false;
    };

    let mut cwd_matches = false;
    let mut prompt_matches = false;
    for line in raw.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if value
            .get("cwd")
            .and_then(Value::as_str)
            .map(|cwd| cwd == expected_cwd)
            .unwrap_or(false)
        {
            cwd_matches = true;
        }
        if message_contains_prompt(&value, prompt) {
            prompt_matches = true;
        }
        if cwd_matches && prompt_matches {
            return true;
        }
    }
    false
}

fn message_contains_prompt(value: &Value, prompt: &str) -> bool {
    value
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_array)
        .map(|items| {
            items.iter().any(|item| {
                item.get("text")
                    .and_then(Value::as_str)
                    .map(|text| text == prompt)
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn wsl_workspace_uses_windows_unc_cwd_for_droid() {
        let cwd = droid_cwd_for_workspace_with_distro(
            Path::new("/home/stache/projects/droidproxy"),
            Some("Ubuntu"),
        );

        assert_eq!(
            cwd,
            r"\\wsl.localhost\Ubuntu\home\stache\projects\droidproxy"
        );
    }

    #[test]
    fn finds_recent_session_for_expected_cwd_and_prompt() {
        let root = temp_root("droid-smoke-session");
        let sessions_dir = root.join("sessions");
        let project_dir =
            sessions_dir.join("--wsl.localhost-Ubuntu-home-stache-projects-droidproxy");
        fs::create_dir_all(&project_dir).unwrap();
        let session_file = project_dir.join("400801bb.jsonl");
        fs::write(
            &session_file,
            concat!(
                "{\"type\":\"session_start\",\"id\":\"400801bb\",\"cwd\":\"\\\\\\\\wsl.localhost\\\\Ubuntu\\\\home\\\\stache\\\\projects\\\\droidproxy\"}\n",
                "{\"type\":\"message\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Reply exactly OK. Do not inspect or modify files.\"}]}}\n"
            ),
        )
        .unwrap();

        let match_path = find_matching_session(
            &sessions_dir,
            r"\\wsl.localhost\Ubuntu\home\stache\projects\droidproxy",
            "Reply exactly OK. Do not inspect or modify files.",
            UNIX_EPOCH,
        );

        assert_eq!(match_path.as_deref(), Some(session_file.as_path()));
        fs::remove_dir_all(root).unwrap();
    }

    fn temp_root(label: &str) -> PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_nanos();
        std::env::temp_dir().join(format!("opengateway-{label}-{id}"))
    }
}
