use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
#[cfg(target_os = "windows")]
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;
use tauri::AppHandle;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
#[cfg(target_os = "windows")]
use std::sync::OnceLock;
#[cfg(target_os = "windows")]
use tauri_plugin_shell::ShellExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;
#[cfg(target_os = "windows")]
static WSL_BRIDGE_CACHE: OnceLock<WslBridge> = OnceLock::new();

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

async fn run_json_command_owned<T>(_app: &AppHandle, args: Vec<String>) -> Result<T, String>
where
    T: serde::de::DeserializeOwned,
{
    let output = match resolve_runtime_target() {
        #[cfg(target_os = "windows")]
        RuntimeTarget::Bundled(wsl_bridge) => {
            run_sidecar_command(_app, &args, wsl_bridge.as_ref()).await?
        }
        RuntimeTarget::Local(binary) => run_local_command(&binary, &args)?,
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
    #[cfg(target_os = "windows")]
    Bundled(Option<WslBridge>),
    Local(PathBuf),
}

#[cfg(target_os = "windows")]
#[derive(Clone, Debug)]
struct WslBridge {
    distro: String,
    home: String,
    factory_home: String,
    config_dir: String,
    workspace: Option<String>,
}

fn resolve_runtime_target() -> RuntimeTarget {
    #[cfg(target_os = "windows")]
    {
        if should_use_bundled_backend() {
            return RuntimeTarget::Bundled(resolve_wsl_bridge());
        }
    }

    RuntimeTarget::Local(resolve_local_binary())
}

#[cfg(target_os = "windows")]
fn should_use_bundled_backend() -> bool {
    !cfg!(debug_assertions) && env_nonempty("OPENGATEWAY_BIN").is_none()
}

#[cfg(target_os = "windows")]
fn resolve_wsl_bridge() -> Option<WslBridge> {
    let bridge_forced = env_flag("OPENGATEWAY_WSL_BRIDGE");
    let distro = env_nonempty("OPENGATEWAY_WSL_DISTRO");
    let workspace = env_nonempty("OPENGATEWAY_WSL_WORKSPACE").or_else(|| {
        env_nonempty("OPENGATEWAY_WORKSPACE").filter(|value| looks_like_linux_path(value))
    });

    if bridge_forced || distro.is_some() || workspace.is_some() {
        let distro = distro.or_else(detect_default_wsl_distro)?;
        return probe_wsl_bridge(&distro, workspace);
    }

    if let Some(cached) = WSL_BRIDGE_CACHE.get() {
        return Some(cached.clone());
    }

    let detected = detect_default_wsl_bridge().or_else(detect_any_wsl_bridge)?;
    let _ = WSL_BRIDGE_CACHE.set(detected.clone());
    Some(detected)
}

#[cfg(target_os = "windows")]
async fn run_sidecar_command(
    app: &AppHandle,
    args: &[String],
    wsl_bridge: Option<&WslBridge>,
) -> Result<RawOutput, String> {
    let mut command = app
        .shell()
        .sidecar("opengateway")
        .map_err(|err| format!("failed to resolve bundled opengateway sidecar: {err}"))?;

    if let Some(bridge) = wsl_bridge {
        command = command.envs(wsl_env_overrides(bridge));
    }

    let output = command
        .args(args.iter().map(|value| value.as_str()))
        .output()
        .await
        .map_err(|err| format!("failed to run bundled opengateway sidecar: {err}"))?;

    Ok(RawOutput {
        code: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    })
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
fn detect_default_wsl_bridge() -> Option<WslBridge> {
    let distro = detect_default_wsl_distro()?;
    probe_wsl_bridge(&distro, None)
}

#[cfg(target_os = "windows")]
fn detect_any_wsl_bridge() -> Option<WslBridge> {
    let mut command = windows_system_command("wsl.exe");
    hide_windows_command(&mut command);
    let output = command.arg("--list").arg("--quiet").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = decode_wsl_output(&output.stdout)?;
    for distro in stdout.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let Some(bridge) = probe_wsl_bridge(distro, None) else {
            continue;
        };
        return Some(bridge);
    }

    None
}

#[cfg(target_os = "windows")]
fn detect_default_wsl_distro() -> Option<String> {
    let mut command = windows_system_command("wsl.exe");
    hide_windows_command(&mut command);
    let output = command.arg("--list").arg("--verbose").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = decode_wsl_output(&output.stdout)?;
    stdout
        .lines()
        .find_map(|line| line.trim_start().strip_prefix('*').map(str::trim))
        .filter(|line| !line.is_empty() && !line.eq_ignore_ascii_case("NAME"))
        .map(|line| {
            line.split_whitespace()
                .next()
                .unwrap_or_default()
                .to_string()
        })
        .filter(|line| !line.is_empty())
}

#[cfg(target_os = "windows")]
fn probe_wsl_bridge(distro: &str, workspace: Option<String>) -> Option<WslBridge> {
    let mut command = windows_system_command("wsl.exe");
    hide_windows_command(&mut command);
    command.arg("-d").arg(distro);
    let output = command
        .arg("-e")
        .arg("bash")
        .arg("-lc")
        .arg(
            r#"config_dir="${OPENGATEWAY_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/opengateway}"
factory_home="${FACTORY_HOME:-$HOME/.factory}"
if [ ! -d "$factory_home" ]; then
  exit 1
fi
printf '%s\n%s\n%s\n' "$HOME" "$factory_home" "$config_dir""#,
        )
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = decode_wsl_output(&output.stdout)?;
    let mut lines = stdout.lines().map(str::trim);
    let home = lines.next()?.to_string();
    let factory_home = lines.next()?.to_string();
    let config_dir = lines.next()?.to_string();
    if home.is_empty() || factory_home.is_empty() || config_dir.is_empty() {
        None
    } else {
        Some(WslBridge {
            distro: distro.to_string(),
            home,
            factory_home,
            config_dir,
            workspace,
        })
    }
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

#[cfg(target_os = "windows")]
fn env_flag(name: &str) -> bool {
    matches!(
        env_nonempty(name)
            .as_deref()
            .map(|value| value.to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
    )
}

#[cfg(target_os = "windows")]
fn looks_like_linux_path(value: &str) -> bool {
    value.starts_with('/') || value.starts_with("~/")
}

#[cfg(target_os = "windows")]
fn decode_wsl_output(bytes: &[u8]) -> Option<String> {
    if bytes.is_empty() {
        return Some(String::new());
    }

    if bytes.len() % 2 == 0 && bytes.iter().skip(1).step_by(2).any(|byte| *byte == 0) {
        let words = bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        return String::from_utf16(&words).ok();
    }

    String::from_utf8(bytes.to_vec()).ok()
}

#[cfg(target_os = "windows")]
fn wsl_env_overrides(bridge: &WslBridge) -> Vec<(String, String)> {
    let mut envs = vec![
        (
            "OPENGATEWAY_FACTORY_HOME".to_string(),
            wsl_unc_path(&bridge.distro, &bridge.factory_home),
        ),
        (
            "OPENGATEWAY_CONFIG_DIR".to_string(),
            wsl_unc_path(&bridge.distro, &bridge.config_dir),
        ),
        (
            "OPENGATEWAY_AUTH_DIR".to_string(),
            wsl_unc_path(
                &bridge.distro,
                &format!("{}/auth", bridge.config_dir.trim_end_matches('/')),
            ),
        ),
    ];

    if let Some(workspace) = bridge.workspace.as_deref() {
        envs.push((
            "OPENGATEWAY_WORKSPACE".to_string(),
            wsl_unc_path(&bridge.distro, &expand_linux_path(&bridge.home, workspace)),
        ));
    }

    envs
}

#[cfg(target_os = "windows")]
fn expand_linux_path(home: &str, path: &str) -> String {
    if path == "~" {
        home.to_string()
    } else if let Some(rest) = path.strip_prefix("~/") {
        format!("{home}/{rest}")
    } else {
        path.to_string()
    }
}

#[cfg(target_os = "windows")]
fn wsl_unc_path(distro: &str, linux_path: &str) -> String {
    let normalized = expand_linux_path("", linux_path)
        .trim_start_matches('/')
        .replace('/', "\\");
    format!(r"\\wsl$\{distro}\{normalized}")
}

#[cfg(target_os = "windows")]
fn windows_system_command(executable: &str) -> Command {
    if let Some(path) = windows_system_executable(executable) {
        return Command::new(path);
    }
    Command::new(executable)
}

#[cfg(target_os = "windows")]
fn windows_system_executable(executable: &str) -> Option<OsString> {
    let system_root = env_nonempty("SystemRoot").or_else(|| env_nonempty("WINDIR"))?;
    Some(
        PathBuf::from(system_root)
            .join("System32")
            .join(executable)
            .into_os_string(),
    )
}

#[cfg(target_os = "windows")]
fn hide_windows_command(command: &mut Command) {
    command.creation_flags(CREATE_NO_WINDOW);
}
