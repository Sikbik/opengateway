use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::path::PathBuf;
use std::process::Command;
use tauri::AppHandle;
#[cfg(target_os = "windows")]
use tauri_plugin_shell::ShellExt;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandResult {
    success: bool,
    output: String,
}

#[derive(Debug)]
struct RawOutput {
    code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl RawOutput {
    fn success(&self) -> bool {
        self.code == Some(0)
    }
}

#[tauri::command]
pub async fn load_snapshot(app: AppHandle) -> Result<Value, String> {
    run_json_command(&app, &["gui-snapshot"]).await
}

#[tauri::command]
pub async fn tail_logs(app: AppHandle, limit: Option<usize>) -> Result<Vec<String>, String> {
    let args = vec![
        "gui-logs".to_string(),
        "--limit".to_string(),
        limit.unwrap_or(160).to_string(),
    ];
    run_json_command_owned(&app, args).await
}

#[tauri::command]
pub async fn start_gateway(app: AppHandle) -> Result<CommandResult, String> {
    run_json_command(&app, &["gui-start"]).await
}

#[tauri::command]
pub async fn stop_gateway(app: AppHandle) -> Result<CommandResult, String> {
    run_json_command(&app, &["gui-stop"]).await
}

#[tauri::command]
pub async fn run_doctor(app: AppHandle) -> Result<CommandResult, String> {
    run_json_command(&app, &["gui-doctor"]).await
}

#[tauri::command]
pub async fn sync_factory(app: AppHandle) -> Result<CommandResult, String> {
    run_json_command(&app, &["gui-sync-factory"]).await
}

#[tauri::command]
pub async fn set_droid_model(app: AppHandle, path: String, model: String) -> Result<Value, String> {
    run_json_command_owned(
        &app,
        vec![
            "gui-set-droid-model".to_string(),
            "--path".to_string(),
            path,
            "--model".to_string(),
            model,
        ],
    )
    .await
}

async fn run_json_command<T>(app: &AppHandle, args: &[&str]) -> Result<T, String>
where
    T: serde::de::DeserializeOwned,
{
    run_json_command_owned(app, args.iter().map(|value| (*value).to_string()).collect()).await
}

async fn run_json_command_owned<T>(app: &AppHandle, args: Vec<String>) -> Result<T, String>
where
    T: serde::de::DeserializeOwned,
{
    let output = match resolve_runtime_target() {
        RuntimeTarget::Bundled => run_sidecar_command(app, &args).await?,
        RuntimeTarget::Local(binary) => run_local_command(&binary, &args)?,
        RuntimeTarget::Wsl(bridge) => run_wsl_command(&bridge, &args)?,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.success() {
        let combined = format!("{}{}", stdout, stderr).trim().to_string();
        return Err(if combined.is_empty() {
            format!(
                "command failed{}",
                output
                    .code
                    .map(|code| format!(" with exit code {code}"))
                    .unwrap_or_default()
            )
        } else {
            combined
        });
    }

    serde_json::from_str(stdout.trim()).map_err(|err| {
        format!(
            "failed to decode opengateway JSON output: {err}; output was: {}",
            stdout.trim()
        )
    })
}

enum RuntimeTarget {
    Bundled,
    Local(PathBuf),
    Wsl(WslBridge),
}

struct WslBridge {
    distro: Option<String>,
    workspace: Option<String>,
    binary: String,
}

fn resolve_runtime_target() -> RuntimeTarget {
    #[cfg(target_os = "windows")]
    {
        if let Some(bridge) = resolve_wsl_bridge() {
            return RuntimeTarget::Wsl(bridge);
        }

        if should_use_bundled_backend() {
            return RuntimeTarget::Bundled;
        }
    }

    RuntimeTarget::Local(resolve_local_binary())
}

#[cfg(target_os = "windows")]
fn should_use_bundled_backend() -> bool {
    !cfg!(debug_assertions) && env_nonempty("OPENGATEWAY_BIN").is_none()
}

#[cfg(not(target_os = "windows"))]
fn should_use_bundled_backend() -> bool {
    false
}

#[cfg(target_os = "windows")]
fn resolve_wsl_bridge() -> Option<WslBridge> {
    let bridge_forced = env_flag("OPENGATEWAY_WSL_BRIDGE");
    let distro = env_nonempty("OPENGATEWAY_WSL_DISTRO");
    let workspace = env_nonempty("OPENGATEWAY_WSL_WORKSPACE").or_else(|| {
        env_nonempty("OPENGATEWAY_WORKSPACE").filter(|value| looks_like_linux_path(value))
    });
    let wsl_binary = env_nonempty("OPENGATEWAY_WSL_BIN");

    if bridge_forced || distro.is_some() || workspace.is_some() || wsl_binary.is_some() {
        return Some(WslBridge {
            distro,
            workspace,
            binary: wsl_binary.unwrap_or_else(|| "opengateway".to_string()),
        });
    }

    None
}

#[cfg(not(target_os = "windows"))]
fn resolve_wsl_bridge() -> Option<WslBridge> {
    None
}

async fn run_sidecar_command(app: &AppHandle, args: &[String]) -> Result<RawOutput, String> {
    #[cfg(target_os = "windows")]
    {
        let output = app
            .shell()
            .sidecar("opengateway")
            .map_err(|err| format!("failed to resolve bundled opengateway sidecar: {err}"))?
            .args(args.iter().map(|value| value.as_str()))
            .output()
            .await
            .map_err(|err| format!("failed to run bundled opengateway sidecar: {err}"))?;

        return Ok(RawOutput {
            code: output.status.code(),
            stdout: output.stdout,
            stderr: output.stderr,
        });
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = (app, args);
        Err("bundled backend is only available on Windows".to_string())
    }
}

fn run_local_command(binary: &PathBuf, args: &[String]) -> Result<RawOutput, String> {
    let output = Command::new(binary)
        .args(args)
        .output()
        .map_err(|err| format!("failed to run opengateway: {err}"))?;

    Ok(RawOutput {
        code: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

#[cfg(target_os = "windows")]
fn run_wsl_command(bridge: &WslBridge, args: &[String]) -> Result<RawOutput, String> {
    let mut command = Command::new("wsl.exe");
    if let Some(distro) = bridge.distro.as_ref() {
        command.arg("-d").arg(distro);
    }
    if let Some(workspace) = bridge.workspace.as_ref() {
        command.arg("--cd").arg(workspace);
    }
    let output = command
        .arg("-e")
        .arg(&bridge.binary)
        .args(args)
        .output()
        .map_err(|err| format!("failed to run opengateway through WSL: {err}"))?;

    Ok(RawOutput {
        code: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

#[cfg(not(target_os = "windows"))]
fn run_wsl_command(_bridge: &WslBridge, _args: &[String]) -> Result<RawOutput, String> {
    Err("WSL bridge is only available on Windows".to_string())
}

fn resolve_local_binary() -> PathBuf {
    if let Some(raw) = env_nonempty("OPENGATEWAY_BIN") {
        return PathBuf::from(raw);
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for candidate in local_binary_candidates(&manifest_dir) {
        if candidate.exists() {
            return candidate;
        }
    }

    #[cfg(target_os = "windows")]
    {
        return PathBuf::from("opengateway.exe");
    }

    #[cfg(not(target_os = "windows"))]
    {
        PathBuf::from("opengateway")
    }
}

fn local_binary_candidates(manifest_dir: &std::path::Path) -> [PathBuf; 2] {
    #[cfg(target_os = "windows")]
    {
        return [
            manifest_dir.join("../../target/debug/opengateway.exe"),
            manifest_dir.join("../../target/release/opengateway.exe"),
        ];
    }

    #[cfg(not(target_os = "windows"))]
    {
        [
            manifest_dir.join("../../target/debug/opengateway"),
            manifest_dir.join("../../target/release/opengateway"),
        ]
    }
}

fn env_nonempty(name: &str) -> Option<String> {
    env::var(name).ok().filter(|raw| !raw.trim().is_empty())
}

fn env_flag(name: &str) -> bool {
    matches!(
        env_nonempty(name)
            .as_deref()
            .map(|value| value.to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
    )
}

fn looks_like_linux_path(value: &str) -> bool {
    value.starts_with('/') || value.starts_with("~/")
}
