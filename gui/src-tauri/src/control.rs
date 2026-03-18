use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
#[cfg(target_os = "windows")]
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;
#[cfg(target_os = "windows")]
use std::process::Stdio;
use tauri::AppHandle;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
#[cfg(target_os = "windows")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(target_os = "windows")]
use std::sync::OnceLock;
#[cfg(target_os = "windows")]
use std::time::{Duration, Instant};
#[cfg(target_os = "windows")]
use tauri_plugin_shell::ShellExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;
#[cfg(target_os = "windows")]
static WSL_BRIDGE_CACHE: OnceLock<WslBridge> = OnceLock::new();
#[cfg(target_os = "windows")]
static WSL_GATEWAY_STARTED_BY_APP: AtomicBool = AtomicBool::new(false);

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
pub async fn load_acp_snapshot(app: AppHandle) -> Result<Value, String> {
    run_json_command(&app, &["gui-acp-snapshot"]).await
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
    #[cfg(target_os = "windows")]
    if let RuntimeTarget::Wsl(bridge) = resolve_runtime_target() {
        return start_wsl_gateway(&bridge);
    }

    run_json_command(&app, &["gui-start"]).await
}

#[tauri::command]
pub async fn stop_gateway(app: AppHandle) -> Result<CommandResult, String> {
    let result = run_json_command(&app, &["gui-stop"]).await?;
    #[cfg(target_os = "windows")]
    if let RuntimeTarget::Wsl(bridge) = resolve_runtime_target() {
        let _ = clear_managed_marker(&bridge);
        WSL_GATEWAY_STARTED_BY_APP.store(false, Ordering::Relaxed);
    }
    Ok(result)
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
        RuntimeTarget::Wsl(bridge) => run_wsl_command(&bridge, &args)?,
        #[cfg(target_os = "windows")]
        RuntimeTarget::Bundled => run_sidecar_command(_app, &args).await?,
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
    Wsl(WslBridge),
    #[cfg(target_os = "windows")]
    Bundled,
    Local(PathBuf),
}

#[cfg(target_os = "windows")]
#[derive(Clone, Debug)]
struct WslBridge {
    distro: String,
    binary: String,
    home: String,
    factory_home: String,
    config_dir: String,
    state_dir: String,
    workspace: Option<String>,
}

#[cfg(target_os = "windows")]
pub fn cleanup_managed_gateway_on_startup() {
    let Some(bridge) = resolve_wsl_bridge() else {
        return;
    };
    if !has_managed_marker(&bridge) {
        return;
    }
    let _ = run_wsl_command(&bridge, &["gui-stop".to_string()]);
    let _ = clear_managed_marker(&bridge);
    WSL_GATEWAY_STARTED_BY_APP.store(false, Ordering::Relaxed);
}

#[cfg(not(target_os = "windows"))]
pub fn cleanup_managed_gateway_on_startup() {}

#[cfg(target_os = "windows")]
pub fn stop_managed_gateway_on_exit() {
    if !WSL_GATEWAY_STARTED_BY_APP.load(Ordering::Relaxed) {
        return;
    }
    let Some(bridge) = resolve_wsl_bridge() else {
        return;
    };
    let _ = run_wsl_command(&bridge, &["gui-stop".to_string()]);
    let _ = clear_managed_marker(&bridge);
    WSL_GATEWAY_STARTED_BY_APP.store(false, Ordering::Relaxed);
}

#[cfg(not(target_os = "windows"))]
pub fn stop_managed_gateway_on_exit() {}

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
async fn run_sidecar_command(app: &AppHandle, args: &[String]) -> Result<RawOutput, String> {
    let mut command = app
        .shell()
        .sidecar("opengateway")
        .map_err(|err| format!("failed to resolve bundled opengateway sidecar: {err}"))?;

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
state_dir="${OPENGATEWAY_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/opengateway}"
factory_home="${FACTORY_HOME:-$HOME/.factory}"
if [ ! -d "$factory_home" ]; then
  exit 1
fi
if [ -x "$HOME/.local/bin/opengateway" ]; then
  binary="$HOME/.local/bin/opengateway"
elif [ -x "$HOME/.cargo/bin/opengateway" ]; then
  binary="$HOME/.cargo/bin/opengateway"
else
  binary="$(command -v opengateway 2>/dev/null || true)"
fi
if [ -z "$binary" ]; then
  exit 1
fi
printf '%s\n%s\n%s\n%s\n%s\n' "$HOME" "$factory_home" "$config_dir" "$state_dir" "$binary""#,
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
    let state_dir = lines.next()?.to_string();
    let binary = lines.next()?.to_string();
    if home.is_empty()
        || factory_home.is_empty()
        || config_dir.is_empty()
        || state_dir.is_empty()
        || binary.is_empty()
    {
        None
    } else {
        Some(WslBridge {
            distro: distro.to_string(),
            binary,
            home,
            factory_home,
            config_dir,
            state_dir,
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

#[cfg(target_os = "windows")]
fn run_wsl_command(bridge: &WslBridge, args: &[String]) -> Result<RawOutput, String> {
    let mut command = windows_system_command("wsl.exe");
    hide_windows_command(&mut command);
    let output = command
        .arg("-d")
        .arg(&bridge.distro)
        .arg("-e")
        .arg("bash")
        .arg("-lc")
        .arg(
            r#"factory_home="$1"
config_dir="$2"
state_dir="$3"
workspace="$4"
binary="$5"
shift 5
export OPENGATEWAY_FACTORY_HOME="$factory_home"
export OPENGATEWAY_CONFIG_DIR="$config_dir"
export OPENGATEWAY_AUTH_DIR="$config_dir/auth"
export OPENGATEWAY_STATE_DIR="$state_dir"
if [ -n "$workspace" ]; then
  export OPENGATEWAY_WORKSPACE="$workspace"
else
  unset OPENGATEWAY_WORKSPACE
fi
exec "$binary" "$@""#,
        )
        .arg("opengateway-wsl")
        .arg(&bridge.factory_home)
        .arg(&bridge.config_dir)
        .arg(&bridge.state_dir)
        .arg(bridge.workspace.as_deref().unwrap_or(""))
        .arg(&bridge.binary)
        .args(args)
        .output()
        .map_err(|err| format!("failed to run WSL opengateway command: {err}"))?;

    Ok(RawOutput {
        code: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

#[cfg(target_os = "windows")]
fn start_wsl_gateway(bridge: &WslBridge) -> Result<CommandResult, String> {
    if let Ok(snapshot) = run_wsl_json_command::<Value>(bridge, &["gui-snapshot".to_string()]) {
        if snapshot
            .get("gateway")
            .and_then(|gateway| gateway.get("running"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let pid = snapshot
                .get("gateway")
                .and_then(|gateway| gateway.get("pid"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            return Ok(CommandResult {
                success: true,
                output: format!("opengateway already running (pid={pid})"),
            });
        }
    }

    let workspace = bridge.workspace.as_deref().unwrap_or("");
    let log_file = format!("{}/opengateway.log", bridge.state_dir.trim_end_matches('/'));

    let mut command = windows_system_command("wsl.exe");
    hide_windows_command(&mut command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .arg("-d")
        .arg(&bridge.distro)
        .arg("-e")
        .arg("bash")
        .arg("-lc")
        .arg(
            r#"factory_home="$1"
config_dir="$2"
state_dir="$3"
workspace="$4"
binary="$5"
log_file="$6"
mkdir -p "$state_dir"
export OPENGATEWAY_FACTORY_HOME="$factory_home"
export OPENGATEWAY_CONFIG_DIR="$config_dir"
export OPENGATEWAY_AUTH_DIR="$config_dir/auth"
export OPENGATEWAY_STATE_DIR="$state_dir"
if [ -n "$workspace" ]; then
  export OPENGATEWAY_WORKSPACE="$workspace"
  cd "$workspace" || exit 1
else
  unset OPENGATEWAY_WORKSPACE
fi
exec "$binary" run >>"$log_file" 2>&1"#,
        )
        .arg("opengateway-wsl-start")
        .arg(&bridge.factory_home)
        .arg(&bridge.config_dir)
        .arg(&bridge.state_dir)
        .arg(workspace)
        .arg(&bridge.binary)
        .arg(&log_file);

    command
        .spawn()
        .map_err(|err| format!("failed to launch WSL gateway host: {err}"))?;

    let deadline = Instant::now() + Duration::from_secs(12);
    while Instant::now() < deadline {
        if let Ok(snapshot) = run_wsl_json_command::<Value>(bridge, &["gui-snapshot".to_string()]) {
            let gateway = snapshot.get("gateway").cloned().unwrap_or(Value::Null);
            let running = gateway
                .get("running")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let health = gateway
                .get("health")
                .and_then(Value::as_str)
                .unwrap_or("offline");
            if running || health != "offline" {
                let pid = gateway.get("pid").and_then(Value::as_u64).unwrap_or(0);
                let _ = write_managed_marker(bridge);
                WSL_GATEWAY_STARTED_BY_APP.store(true, Ordering::Relaxed);
                return Ok(CommandResult {
                    success: true,
                    output: format!("opengateway started (pid={pid}) on http://127.0.0.1:42069"),
                });
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }

    let logs = run_wsl_json_command::<Vec<String>>(
        bridge,
        &["gui-logs".to_string(), "--limit".to_string(), "20".to_string()],
    )
    .unwrap_or_default()
    .join("\n");
    Err(format!("opengateway failed to start. recent logs:\n{logs}"))
}

#[cfg(target_os = "windows")]
fn run_wsl_json_command<T>(bridge: &WslBridge, args: &[String]) -> Result<T, String>
where
    T: serde::de::DeserializeOwned,
{
    let output = run_wsl_command(bridge, args)?;
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
            "failed to decode WSL opengateway JSON output: {err}; output was: {}",
            stdout.trim()
        )
    })
}

#[cfg(target_os = "windows")]
fn has_managed_marker(bridge: &WslBridge) -> bool {
    let mut command = windows_system_command("wsl.exe");
    hide_windows_command(&mut command);
    command
        .arg("-d")
        .arg(&bridge.distro)
        .arg("-e")
        .arg("bash")
        .arg("-lc")
        .arg(r#"test -f "$1""#)
        .arg("factory-control-marker")
        .arg(managed_marker_path(bridge))
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn write_managed_marker(bridge: &WslBridge) -> Result<(), String> {
    let mut command = windows_system_command("wsl.exe");
    hide_windows_command(&mut command);
    let status = command
        .arg("-d")
        .arg(&bridge.distro)
        .arg("-e")
        .arg("bash")
        .arg("-lc")
        .arg(r#"mkdir -p "$(dirname "$1")" && : > "$1""#)
        .arg("factory-control-marker")
        .arg(managed_marker_path(bridge))
        .status()
        .map_err(|err| format!("failed to write WSL managed marker: {err}"))?;

    if status.success() {
        Ok(())
    } else {
        Err("failed to write WSL managed marker".to_string())
    }
}

#[cfg(target_os = "windows")]
fn clear_managed_marker(bridge: &WslBridge) -> Result<(), String> {
    let mut command = windows_system_command("wsl.exe");
    hide_windows_command(&mut command);
    let status = command
        .arg("-d")
        .arg(&bridge.distro)
        .arg("-e")
        .arg("bash")
        .arg("-lc")
        .arg(r#"rm -f "$1""#)
        .arg("factory-control-marker")
        .arg(managed_marker_path(bridge))
        .status()
        .map_err(|err| format!("failed to clear WSL managed marker: {err}"))?;

    if status.success() {
        Ok(())
    } else {
        Err("failed to clear WSL managed marker".to_string())
    }
}

#[cfg(target_os = "windows")]
fn managed_marker_path(bridge: &WslBridge) -> String {
    format!(
        "{}/factory-control-managed",
        bridge.state_dir.trim_end_matches('/')
    )
}
