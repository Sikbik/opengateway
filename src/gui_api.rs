use anyhow::{anyhow, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::droid_files::{
    merge_droids, read_droids, set_droid_model as set_droid_record_model, DroidRecord,
    WORKSPACE_DROIDS_RELATIVE,
};
use crate::factory_desktop::FactoryDesktopReadiness;
use crate::paths::{build_factory_paths, build_paths};
use crate::tool_probe::{CodexReadiness, DroidReadiness};

const GATEWAY_URL: &str = "http://127.0.0.1:42069";

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSnapshot {
    generated_at: i64,
    workspace_path: Option<String>,
    environment: EnvironmentSnapshot,
    gateway: GatewaySnapshot,
    factory: FactorySnapshot,
    native_harness: NativeHarnessSnapshot,
    models: Vec<ModelOption>,
    droids: Vec<DroidRecord>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentSnapshot {
    os: String,
    is_wsl: bool,
    desktop_shell_supported: bool,
    preferred_runtime: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewaySnapshot {
    running: bool,
    pid: Option<u32>,
    health: &'static str,
    api_base_url: String,
    log_path: String,
    preferred_model: Option<String>,
    auth: AuthSnapshot,
    last_log_line: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthSnapshot {
    account_count: usize,
    active_account: Option<String>,
    expires_at_ms: Option<i64>,
    expires_in_minutes: Option<i64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FactorySnapshot {
    home_path: String,
    config_path: String,
    settings_path: String,
    machine_droids_path: String,
    legacy_custom_model_count: usize,
    settings_custom_model_count: usize,
    session_default_model: Option<String>,
    mission_models: MissionModels,
    issues: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeHarnessSnapshot {
    factory_desktop: FactoryDesktopReadiness,
    droid: DroidReadiness,
    codex: CodexReadiness,
    mode_recommendation: &'static str,
    byok_required: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MissionModels {
    orchestrator: Option<String>,
    worker: Option<String>,
    validation_worker: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelOption {
    display_name: String,
    model: String,
    id: Option<String>,
    source: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandResult {
    success: bool,
    output: String,
}

pub fn print_snapshot_json() -> Result<()> {
    let snapshot = load_snapshot()?;
    println!(
        "{}",
        serde_json::to_string(&snapshot).context("failed to encode snapshot")?
    );
    Ok(())
}

pub fn print_logs_json(limit: usize) -> Result<()> {
    let paths = build_paths()?;
    let logs = crate::tail_file(&paths.log_file, limit).unwrap_or_default();
    println!(
        "{}",
        serde_json::to_string(&logs).context("failed to encode logs")?
    );
    Ok(())
}

pub fn print_command_result_json(command: &[&str]) -> Result<()> {
    let result = run_self_command(command)?;
    println!(
        "{}",
        serde_json::to_string(&result).context("failed to encode command result")?
    );
    Ok(())
}

pub fn print_droid_model_update_json(path: &Path, model: &str) -> Result<()> {
    let factory_paths = build_factory_paths()?;
    let workspace_droids_dir =
        detect_workspace_root().map(|workspace| workspace.join(WORKSPACE_DROIDS_RELATIVE));
    let record = set_droid_record_model(
        path,
        model,
        &factory_paths.machine_droids_dir,
        workspace_droids_dir.as_deref(),
    )?;
    println!(
        "{}",
        serde_json::to_string(&record).context("failed to encode droid record")?
    );
    Ok(())
}

fn load_snapshot() -> Result<AppSnapshot> {
    let paths = build_paths()?;
    let workspace_path = detect_workspace_root();
    let factory_paths = build_factory_paths()?;
    let machine_droids = read_droids(&factory_paths.machine_droids_dir, "machine");
    let workspace_droids = workspace_path
        .as_ref()
        .map(|path| read_droids(&path.join(WORKSPACE_DROIDS_RELATIVE), "workspace"))
        .unwrap_or_default();

    let factory = read_factory_snapshot(&factory_paths);
    let factory_desktop = crate::factory_desktop::probe_factory_desktop();
    let preferred_droid = factory_desktop
        .bundled_droid_path
        .as_ref()
        .map(PathBuf::from);
    let droid = crate::tool_probe::probe_droid_cli(preferred_droid);
    let codex = crate::tool_probe::probe_codex_cli();
    let native_harness = build_native_harness_snapshot(factory_desktop, droid, codex);

    Ok(AppSnapshot {
        generated_at: now_millis(),
        workspace_path: workspace_path.map(|path| path.display().to_string()),
        environment: environment_snapshot(),
        gateway: read_gateway_snapshot(&paths, &factory),
        factory,
        native_harness,
        models: read_model_catalog(),
        droids: merge_droids(workspace_droids, machine_droids),
    })
}

fn build_native_harness_snapshot(
    factory_desktop: FactoryDesktopReadiness,
    droid: DroidReadiness,
    codex: CodexReadiness,
) -> NativeHarnessSnapshot {
    let factory_droid_ready = factory_desktop.installed
        && factory_desktop.bundled_droid_path.is_some()
        && droid.issue.is_none();
    let mode_recommendation = if factory_droid_ready {
        "factory-droid-native"
    } else if droid.issue.is_none() {
        "droid-cli-native"
    } else if codex.issue.is_none() {
        "codex-app-server"
    } else {
        "setup-required"
    };

    NativeHarnessSnapshot {
        factory_desktop,
        droid,
        codex,
        mode_recommendation,
        byok_required: false,
    }
}

fn environment_snapshot() -> EnvironmentSnapshot {
    let is_wsl = detect_wsl();
    EnvironmentSnapshot {
        os: std::env::consts::OS.to_string(),
        is_wsl,
        desktop_shell_supported: !is_wsl,
        preferred_runtime: if is_wsl {
            "browser".to_string()
        } else {
            "desktop".to_string()
        },
    }
}

fn read_gateway_snapshot(
    paths: &crate::paths::AppPaths,
    factory: &FactorySnapshot,
) -> GatewaySnapshot {
    let pid = crate::read_pid(&paths.pid_file).map(|value| value as u32);
    let http_ready = crate::is_http_ready("127.0.0.1", 42069, Duration::from_millis(500));
    let port_ready = crate::is_port_open("127.0.0.1", 42069, Duration::from_millis(300));
    let health = match (http_ready, port_ready, pid.is_some()) {
        (true, _, true) => "online",
        (true, _, false) => "degraded",
        (false, true, _) => "degraded",
        (false, false, _) => "offline",
    };

    GatewaySnapshot {
        running: health != "offline",
        pid,
        health,
        api_base_url: format!("{GATEWAY_URL}/v1"),
        log_path: paths.log_file.display().to_string(),
        preferred_model: factory.session_default_model.clone(),
        auth: read_auth_snapshot(&paths.auth_dir.join("auth.json")),
        last_log_line: crate::tail_file(&paths.log_file, 1)
            .ok()
            .and_then(|mut items| items.pop()),
    }
}

fn read_auth_snapshot(path: &Path) -> AuthSnapshot {
    let raw = fs::read_to_string(path).ok();
    let json = raw
        .as_deref()
        .and_then(|content| serde_json::from_str::<Value>(content).ok());

    let active_account = json
        .as_ref()
        .and_then(|value| value.get("active_account"))
        .and_then(Value::as_str)
        .map(str::to_string);

    let accounts = json
        .as_ref()
        .and_then(|value| value.get("accounts"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let account_count = accounts
        .values()
        .filter(|entry| entry.get("provider").and_then(Value::as_str) == Some("openai"))
        .count();

    let active_entry = active_account
        .as_ref()
        .and_then(|key| accounts.get(key))
        .or_else(|| accounts.values().next());

    let expires_at_ms = active_entry
        .and_then(|entry| entry.get("expires_at_ms"))
        .and_then(Value::as_i64);
    let expires_in_minutes =
        expires_at_ms.map(|expiry| ((expiry - now_millis()) / 1000 / 60).max(0));

    AuthSnapshot {
        account_count,
        active_account,
        expires_at_ms,
        expires_in_minutes,
    }
}

fn read_factory_snapshot(factory_paths: &crate::paths::FactoryPaths) -> FactorySnapshot {
    let config_path = factory_paths.config_path.clone();
    let settings_path = factory_paths.settings_path.clone();
    let config = read_json_file(&config_path);
    let settings = read_json_file(&settings_path);

    let legacy_custom_model_count = config
        .get("custom_models")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);

    let settings_custom_model_count = settings
        .get("customModels")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);

    let session_default_model = settings
        .get("sessionDefaultSettings")
        .and_then(|value| value.get("model"))
        .and_then(Value::as_str)
        .map(str::to_string);

    let mission_models = MissionModels {
        orchestrator: settings
            .get("missionModelSettings")
            .and_then(|value| value.get("orchestratorModel"))
            .and_then(Value::as_str)
            .map(str::to_string),
        worker: settings
            .get("missionModelSettings")
            .and_then(|value| value.get("workerModel"))
            .and_then(Value::as_str)
            .map(str::to_string),
        validation_worker: settings
            .get("missionModelSettings")
            .and_then(|value| value.get("validationWorkerModel"))
            .and_then(Value::as_str)
            .map(str::to_string),
    };

    let mut issues = Vec::new();
    if legacy_custom_model_count == 0 {
        issues.push("Legacy Factory config has no custom models.".to_string());
    }
    if settings_custom_model_count == 0 {
        issues.push("Factory settings have no custom models.".to_string());
    }
    if !session_default_model
        .as_deref()
        .map(|value| value.starts_with("custom:"))
        .unwrap_or(false)
    {
        issues.push("Session default is not pointing at a custom model.".to_string());
    }
    for (label, value) in [
        ("orchestrator", mission_models.orchestrator.as_deref()),
        ("worker", mission_models.worker.as_deref()),
        (
            "validation worker",
            mission_models.validation_worker.as_deref(),
        ),
    ] {
        if !value
            .map(|item| item.starts_with("custom:"))
            .unwrap_or(false)
        {
            issues.push(format!(
                "{label} mission model is not using a custom model."
            ));
        }
    }

    FactorySnapshot {
        home_path: factory_paths.home_dir.display().to_string(),
        config_path: config_path.display().to_string(),
        settings_path: settings_path.display().to_string(),
        machine_droids_path: factory_paths.machine_droids_dir.display().to_string(),
        legacy_custom_model_count,
        settings_custom_model_count,
        session_default_model,
        mission_models,
        issues,
    }
}

fn read_model_catalog() -> Vec<ModelOption> {
    let settings_path = match build_factory_paths() {
        Ok(paths) => paths.settings_path,
        Err(_) => return Vec::new(),
    };
    let settings = read_json_file(&settings_path);
    let mut models = settings
        .get("customModels")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(model_option_from_factory_settings_item)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    models.sort_by(|left, right| left.display_name.cmp(&right.display_name));
    models
}

fn model_option_from_factory_settings_item(item: &Value) -> Option<ModelOption> {
    let raw_model = item.get("model")?.as_str()?.to_string();
    let factory_model_id = item.get("id").and_then(Value::as_str).map(str::to_string);
    Some(ModelOption {
        display_name: item.get("displayName")?.as_str()?.to_string(),
        model: factory_model_id
            .clone()
            .unwrap_or_else(|| format!("custom:{raw_model}")),
        id: factory_model_id,
        source: "factory-settings".to_string(),
    })
}

fn read_json_file(path: &Path) -> Value {
    fs::read_to_string(path)
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok())
        .unwrap_or_else(|| Value::Object(Default::default()))
}

fn detect_workspace_root() -> Option<PathBuf> {
    if let Ok(raw) = env::var("OPENGATEWAY_WORKSPACE") {
        let path = PathBuf::from(raw);
        if path.exists() {
            return Some(path);
        }
    }

    let current = env::current_dir().ok()?;
    current
        .ancestors()
        .map(Path::to_path_buf)
        .find(|path| path.join(".git").exists() && path.join("Cargo.toml").exists())
}

fn detect_wsl() -> bool {
    if env::var("WSL_DISTRO_NAME")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
    {
        return true;
    }

    fs::read_to_string("/proc/version")
        .map(|content| content.to_ascii_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

fn run_self_command(args: &[&str]) -> Result<CommandResult> {
    let current_exe = env::current_exe().context("failed to resolve current executable")?;
    let output = Command::new(current_exe)
        .args(args)
        .output()
        .context("failed to execute opengateway command")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr).trim().to_string();

    if output.status.success() {
        Ok(CommandResult {
            success: true,
            output: if combined.is_empty() {
                "Command completed.".to_string()
            } else {
                combined
            },
        })
    } else {
        Err(anyhow!(if combined.is_empty() {
            format!("command failed with status {}", output.status)
        } else {
            combined
        }))
    }
}

fn now_millis() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommends_factory_droid_when_droid_is_ready() {
        let snapshot = build_native_harness_snapshot(
            FactoryDesktopReadiness {
                installed: true,
                install_dir: Some("C:/Users/example/AppData/Local/Factory/app-0.116.1".to_string()),
                version: Some("0.116.1".to_string()),
                bundled_droid_path: Some(
                    "C:/Users/example/AppData/Local/Factory/app-0.116.1/resources/bin/droid.exe"
                        .to_string(),
                ),
                issue: None,
            },
            DroidReadiness {
                executable: Some("droid".to_string()),
                version: Some("0.159.1".to_string()),
                supports_exec: true,
                supports_stream_jsonrpc: true,
                supports_daemon_ipc: true,
                issue: None,
            },
            CodexReadiness {
                executable: Some("codex".to_string()),
                version: Some("codex 26.623.5546".to_string()),
                supports_app_server: true,
                supports_generate_schema: true,
                issue: None,
            },
        );

        assert_eq!(snapshot.mode_recommendation, "factory-droid-native");
        assert!(!snapshot.byok_required);
    }

    #[test]
    fn recommends_codex_when_droid_is_not_ready() {
        let snapshot = build_native_harness_snapshot(
            FactoryDesktopReadiness {
                installed: false,
                install_dir: None,
                version: None,
                bundled_droid_path: None,
                issue: Some("Factory Desktop install was not found".to_string()),
            },
            DroidReadiness {
                executable: Some("droid".to_string()),
                version: None,
                supports_exec: false,
                supports_stream_jsonrpc: false,
                supports_daemon_ipc: false,
                issue: Some("Droid CLI was not found or could not be executed".to_string()),
            },
            CodexReadiness {
                executable: Some("codex".to_string()),
                version: Some("codex 26.623.5546".to_string()),
                supports_app_server: true,
                supports_generate_schema: true,
                issue: None,
            },
        );

        assert_eq!(snapshot.mode_recommendation, "codex-app-server");
        assert!(!snapshot.byok_required);
    }

    #[test]
    fn recommends_droid_cli_when_factory_desktop_is_missing() {
        let snapshot = build_native_harness_snapshot(
            FactoryDesktopReadiness {
                installed: false,
                install_dir: None,
                version: None,
                bundled_droid_path: None,
                issue: Some("Factory Desktop install was not found".to_string()),
            },
            DroidReadiness {
                executable: Some("droid".to_string()),
                version: Some("0.159.1".to_string()),
                supports_exec: true,
                supports_stream_jsonrpc: true,
                supports_daemon_ipc: true,
                issue: None,
            },
            CodexReadiness {
                executable: Some("codex".to_string()),
                version: Some("codex 26.623.5546".to_string()),
                supports_app_server: true,
                supports_generate_schema: true,
                issue: None,
            },
        );

        assert_eq!(snapshot.mode_recommendation, "droid-cli-native");
        assert!(!snapshot.byok_required);
    }

    #[test]
    fn recommends_setup_when_droid_and_codex_are_not_ready() {
        let snapshot = build_native_harness_snapshot(
            FactoryDesktopReadiness {
                installed: false,
                install_dir: None,
                version: None,
                bundled_droid_path: None,
                issue: Some("Factory Desktop install was not found".to_string()),
            },
            DroidReadiness {
                executable: Some("droid".to_string()),
                version: None,
                supports_exec: false,
                supports_stream_jsonrpc: false,
                supports_daemon_ipc: false,
                issue: Some("Droid CLI was not found or could not be executed".to_string()),
            },
            CodexReadiness {
                executable: Some("codex".to_string()),
                version: None,
                supports_app_server: false,
                supports_generate_schema: false,
                issue: Some("Codex CLI was not found or could not be executed".to_string()),
            },
        );

        assert_eq!(snapshot.mode_recommendation, "setup-required");
        assert!(!snapshot.byok_required);
    }

    #[test]
    fn model_catalog_uses_factory_custom_model_id() {
        let item = serde_json::json!({
            "model": "gpt-5.4(xhigh)",
            "id": "custom:GPT-5.4-(XHigh)-24",
            "displayName": "GPT-5.4 (XHigh)"
        });

        let option = model_option_from_factory_settings_item(&item).unwrap();

        assert_eq!(option.model, "custom:GPT-5.4-(XHigh)-24");
        assert_eq!(option.id.as_deref(), Some("custom:GPT-5.4-(XHigh)-24"));
        assert_eq!(option.display_name, "GPT-5.4 (XHigh)");
    }
}
