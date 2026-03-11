mod auth_store;
mod gui_api;
mod oauth;
mod paths;
mod service;

use anyhow::{anyhow, Context, Result};
use auth_store::{AuthStore, NewCredential};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use clap::{Parser, Subcommand, ValueEnum};
use oauth::LoginMode;
use paths::build_paths;
use rand::Rng;
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_FRONT_HOST: &str = "127.0.0.1";
const DEFAULT_FRONT_PORT: u16 = 42069;
const DEFAULT_BACKEND_HOST: &str = "127.0.0.1";
const DEFAULT_BACKEND_PORT: u16 = 42069;
const DEFAULT_MAX_BODY_BYTES: usize = 10 * 1024 * 1024;
const DEFAULT_MAX_INFLIGHT: usize = 32;
const DEFAULT_MAX_QUEUE: usize = 128;
const DEFAULT_QUEUE_TIMEOUT_SECONDS: f64 = 1.5;
const DEFAULT_PER_CLIENT_RATE: f64 = 0.0;
const DEFAULT_PER_CLIENT_BURST: usize = 0;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;
const FACTORY_PREFERRED_MODEL: &str = "gpt-5.4(xhigh)";
const FACTORY_PREFERRED_REASONING_EFFORT: &str = "xhigh";
const DEFAULT_OPENAI_MODEL_CATALOG: [(&str, &str); 26] = [
    ("gpt-5.4", "GPT-5.4"),
    ("gpt-5.4(low)", "GPT-5.4 (Low)"),
    ("gpt-5.4(medium)", "GPT-5.4 (Medium)"),
    ("gpt-5.4(high)", "GPT-5.4 (High)"),
    ("gpt-5.4(xhigh)", "GPT-5.4 (XHigh)"),
    ("gpt-5.3-codex", "GPT-5.3 Codex"),
    ("gpt-5.3-codex-spark", "GPT-5.3 Codex Spark"),
    ("gpt-5.3-codex(high)", "GPT-5.3 Codex (High)"),
    ("gpt-5.3-codex(xhigh)", "GPT-5.3 Codex (XHigh)"),
    ("gpt-5.2-codex", "GPT-5.2 Codex"),
    ("gpt-5.2-codex(high)", "GPT-5.2 Codex (High)"),
    ("gpt-5.2-codex(xhigh)", "GPT-5.2 Codex (XHigh)"),
    ("gpt-5.1-codex-max", "GPT-5.1 Codex Max"),
    ("gpt-5.1-codex-max(high)", "GPT-5.1 Codex Max (High)"),
    ("gpt-5.1-codex-max(xhigh)", "GPT-5.1 Codex Max (XHigh)"),
    ("gpt-5.1-codex", "GPT-5.1 Codex"),
    ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini"),
    ("gpt-5-codex", "GPT-5 Codex"),
    ("gpt-5-codex-mini", "GPT-5 Codex Mini"),
    ("gpt-5.2", "GPT-5.2"),
    ("gpt-5.2(high)", "GPT-5.2 (High)"),
    ("gpt-5.2(xhigh)", "GPT-5.2 (XHigh)"),
    ("gpt-5.1", "GPT-5.1"),
    ("gpt-5.1(high)", "GPT-5.1 (High)"),
    ("gpt-5", "GPT-5"),
    ("gpt-5(high)", "GPT-5 (High)"),
];

#[derive(Debug, Parser)]
#[command(name = "opengateway")]
#[command(version)]
#[command(about = "Rust-first local OAuth proxy")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Init(InitArgs),
    Setup(SetupArgs),
    SyncFactory(SyncFactoryArgs),
    #[command(alias = "gui")]
    Control(ControlArgs),
    Start(StartArgs),
    Run(RunArgs),
    Stop(StopArgs),
    Status(StatusArgs),
    Logs(LogsArgs),
    Login(LoginArgs),
    ShowKey(ShowKeyArgs),
    SelfTest(SelfTestArgs),
    FactoryConfig(FactoryConfigArgs),
    Doctor(DoctorArgs),
    #[command(name = "gui-snapshot", hide = true)]
    GuiSnapshot,
    #[command(name = "gui-logs", hide = true)]
    GuiLogs(GuiLogsArgs),
    #[command(name = "gui-start", hide = true)]
    GuiStart,
    #[command(name = "gui-stop", hide = true)]
    GuiStop,
    #[command(name = "gui-doctor", hide = true)]
    GuiDoctor,
    #[command(name = "gui-sync-factory", hide = true)]
    GuiSyncFactory,
    #[command(name = "gui-set-droid-model", hide = true)]
    GuiSetDroidModel(GuiSetDroidModelArgs),
    InstallBackend,
}

#[derive(Debug, clap::Args)]
struct InitArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_BACKEND_HOST)]
    backend_host: String,
    #[arg(long, default_value_t = DEFAULT_BACKEND_PORT)]
    backend_port: u16,
    #[arg(long)]
    force: bool,
}

#[derive(Debug, clap::Args)]
struct SetupArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_FRONT_HOST)]
    host: String,
    #[arg(long, default_value_t = DEFAULT_FRONT_PORT)]
    port: u16,
    #[arg(long, default_value = DEFAULT_BACKEND_HOST)]
    backend_host: String,
    #[arg(long, default_value_t = DEFAULT_BACKEND_PORT)]
    backend_port: u16,
    #[arg(long, default_value_t = 12.0)]
    start_timeout: f64,
    #[arg(long)]
    headless: bool,
    #[arg(long, help = "Open the OAuth URL in your default browser")]
    open_browser: bool,
    #[arg(long, hide = true)]
    no_browser: bool,
    #[arg(long)]
    relogin: bool,
    #[arg(long, default_value = "")]
    base_url: String,
    #[arg(long, help = "Path to Factory legacy config.json")]
    factory_config: Option<PathBuf>,
    #[arg(long, default_value = "")]
    api_key: String,
    #[arg(long, default_value = "")]
    models: String,
    #[arg(long, default_value_t = DEFAULT_MAX_BODY_BYTES)]
    max_body_bytes: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_INFLIGHT)]
    max_inflight: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_QUEUE)]
    max_queue: usize,
    #[arg(long, default_value_t = DEFAULT_QUEUE_TIMEOUT_SECONDS)]
    queue_timeout_seconds: f64,
    #[arg(long, default_value_t = DEFAULT_PER_CLIENT_RATE)]
    per_client_rate: f64,
    #[arg(long, default_value_t = DEFAULT_PER_CLIENT_BURST)]
    per_client_burst: usize,
    #[arg(long)]
    verbose: bool,
}

#[derive(Debug, clap::Args)]
struct SyncFactoryArgs {
    #[arg(long, default_value = "")]
    base_url: String,
    #[arg(long, help = "Path to Factory legacy config.json")]
    factory_config: Option<PathBuf>,
    #[arg(long, help = "Path to Factory settings.json")]
    factory_settings: Option<PathBuf>,
    #[arg(long, default_value = "")]
    api_key: String,
    #[arg(long, default_value = "")]
    models: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ControlModeArg {
    Auto,
    Web,
    Desktop,
    Check,
    Build,
}

#[derive(Debug, clap::Args)]
struct ControlArgs {
    #[arg(default_value = "auto")]
    mode: ControlModeArg,
    #[arg(
        long,
        help = "Path to the repo root that contains gui/ and bin/factory-control"
    )]
    workspace: Option<PathBuf>,
}

#[derive(Debug, clap::Args, Clone)]
struct StartArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_FRONT_HOST)]
    host: String,
    #[arg(long, default_value_t = DEFAULT_FRONT_PORT)]
    port: u16,
    #[arg(long, default_value = DEFAULT_BACKEND_HOST)]
    backend_host: String,
    #[arg(long, default_value_t = DEFAULT_BACKEND_PORT)]
    backend_port: u16,
    #[arg(long, default_value_t = 12.0)]
    timeout: f64,
    #[arg(long)]
    no_auto_install: bool,
    #[arg(long, default_value = "")]
    api_key: String,
    #[arg(long, default_value = "")]
    models: String,
    #[arg(long, default_value_t = DEFAULT_MAX_BODY_BYTES)]
    max_body_bytes: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_INFLIGHT)]
    max_inflight: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_QUEUE)]
    max_queue: usize,
    #[arg(long, default_value_t = DEFAULT_QUEUE_TIMEOUT_SECONDS)]
    queue_timeout_seconds: f64,
    #[arg(long, default_value_t = DEFAULT_PER_CLIENT_RATE)]
    per_client_rate: f64,
    #[arg(long, default_value_t = DEFAULT_PER_CLIENT_BURST)]
    per_client_burst: usize,
}

#[derive(Debug, clap::Args, Clone)]
struct RunArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_FRONT_HOST)]
    host: String,
    #[arg(long, default_value_t = DEFAULT_FRONT_PORT)]
    port: u16,
    #[arg(long, default_value = DEFAULT_BACKEND_HOST)]
    backend_host: String,
    #[arg(long, default_value_t = DEFAULT_BACKEND_PORT)]
    backend_port: u16,
    #[arg(long)]
    no_auto_install: bool,
    #[arg(long, default_value = "")]
    api_key: String,
    #[arg(long, default_value = "")]
    models: String,
    #[arg(long, default_value_t = DEFAULT_MAX_BODY_BYTES)]
    max_body_bytes: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_INFLIGHT)]
    max_inflight: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_QUEUE)]
    max_queue: usize,
    #[arg(long, default_value_t = DEFAULT_QUEUE_TIMEOUT_SECONDS)]
    queue_timeout_seconds: f64,
    #[arg(long, default_value_t = DEFAULT_PER_CLIENT_RATE)]
    per_client_rate: f64,
    #[arg(long, default_value_t = DEFAULT_PER_CLIENT_BURST)]
    per_client_burst: usize,
}

#[derive(Debug, clap::Args)]
struct StopArgs {
    #[arg(long, default_value_t = 8.0)]
    timeout: f64,
    #[arg(long)]
    force: bool,
}

#[derive(Debug, clap::Args)]
struct StatusArgs {
    #[arg(long, default_value = DEFAULT_FRONT_HOST)]
    host: String,
    #[arg(long, default_value_t = DEFAULT_FRONT_PORT)]
    port: u16,
    #[arg(long, default_value = DEFAULT_BACKEND_HOST)]
    backend_host: String,
    #[arg(long, default_value_t = DEFAULT_BACKEND_PORT)]
    backend_port: u16,
}

#[derive(Debug, clap::Args)]
struct LogsArgs {
    #[arg(short = 'n', long, default_value_t = 60)]
    lines: usize,
    #[arg(short = 'f', long)]
    follow: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LoginModeArg {
    Browser,
    Headless,
}

#[derive(Debug, clap::Args)]
struct LoginArgs {
    #[arg(default_value = "browser")]
    mode: LoginModeArg,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_BACKEND_HOST)]
    backend_host: String,
    #[arg(long, default_value_t = DEFAULT_BACKEND_PORT)]
    backend_port: u16,
    #[arg(long, help = "Open the OAuth URL in your default browser")]
    open_browser: bool,
    #[arg(long, hide = true)]
    no_browser: bool,
    #[arg(long)]
    no_auto_install: bool,
    #[arg(long)]
    verbose: bool,
}

#[derive(Debug, clap::Args)]
struct ShowKeyArgs {
    #[arg(long)]
    quiet: bool,
}

#[derive(Debug, clap::Args)]
struct SelfTestArgs {
    #[arg(long, default_value = DEFAULT_FRONT_HOST)]
    host: String,
    #[arg(long, default_value_t = DEFAULT_FRONT_PORT)]
    port: u16,
    #[arg(long, default_value = "")]
    api_key: String,
    #[arg(long, default_value_t = 6.0)]
    timeout: f64,
}

#[derive(Debug, clap::Args)]
struct FactoryConfigArgs {
    #[arg(long, default_value = "http://localhost:42069")]
    base_url: String,
    #[arg(long, default_value = "")]
    api_key: String,
    #[arg(long, default_value = "")]
    models: String,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, clap::Args)]
struct DoctorArgs {
    #[arg(long, default_value = DEFAULT_FRONT_HOST)]
    host: String,
    #[arg(long, default_value_t = DEFAULT_FRONT_PORT)]
    port: u16,
    #[arg(long, default_value = DEFAULT_BACKEND_HOST)]
    backend_host: String,
    #[arg(long, default_value_t = DEFAULT_BACKEND_PORT)]
    backend_port: u16,
}

#[derive(Debug, clap::Args)]
struct GuiLogsArgs {
    #[arg(long, default_value_t = 160)]
    limit: usize,
}

#[derive(Debug, clap::Args)]
struct GuiSetDroidModelArgs {
    #[arg(long)]
    path: PathBuf,
    #[arg(long)]
    model: String,
}

impl StartArgs {
    fn to_run_args(&self, persisted_api_key: String) -> RunArgs {
        RunArgs {
            config: self.config.clone(),
            host: self.host.clone(),
            port: self.port,
            backend_host: self.backend_host.clone(),
            backend_port: self.backend_port,
            no_auto_install: self.no_auto_install,
            api_key: persisted_api_key,
            models: self.models.clone(),
            max_body_bytes: self.max_body_bytes,
            max_inflight: self.max_inflight,
            max_queue: self.max_queue,
            queue_timeout_seconds: self.queue_timeout_seconds,
            per_client_rate: self.per_client_rate,
            per_client_burst: self.per_client_burst,
        }
    }
}

fn main() {
    if let Err(err) = run_cli() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

pub(crate) fn runtime_log_info(message: impl AsRef<str>) {
    println!("[{}] {}", runtime_log_timestamp_ms(), message.as_ref());
}

pub(crate) fn runtime_log_error(message: impl AsRef<str>) {
    eprintln!("[{}] {}", runtime_log_timestamp_ms(), message.as_ref());
}

fn runtime_log_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis())
        .unwrap_or(0)
}

fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init(args) => command_init(args),
        Commands::Setup(args) => command_setup(args),
        Commands::SyncFactory(args) => command_sync_factory(args),
        Commands::Control(args) => command_control(args),
        Commands::Start(args) => command_start(args),
        Commands::Run(args) => command_run(args),
        Commands::Stop(args) => command_stop(args),
        Commands::Status(args) => command_status(args),
        Commands::Logs(args) => command_logs(args),
        Commands::Login(args) => command_login(args),
        Commands::ShowKey(args) => command_show_key(args),
        Commands::SelfTest(args) => command_self_test(args),
        Commands::FactoryConfig(args) => command_factory_config(args),
        Commands::Doctor(args) => command_doctor(args),
        Commands::GuiSnapshot => gui_api::print_snapshot_json(),
        Commands::GuiLogs(args) => gui_api::print_logs_json(args.limit),
        Commands::GuiStart => gui_api::print_command_result_json(&["start"]),
        Commands::GuiStop => gui_api::print_command_result_json(&["stop"]),
        Commands::GuiDoctor => gui_api::print_command_result_json(&["doctor"]),
        Commands::GuiSyncFactory => gui_api::print_command_result_json(&["sync-factory"]),
        Commands::GuiSetDroidModel(args) => {
            gui_api::print_droid_model_update_json(&args.path, &args.model)
        }
        Commands::InstallBackend => command_install_backend(),
    }
}

fn command_install_backend() -> Result<()> {
    println!("install-backend is deprecated in Rust mode (no external backend binary required).");
    Ok(())
}

fn command_control(args: ControlArgs) -> Result<()> {
    let workspace = resolve_control_workspace(args.workspace.as_deref())?;
    let mut command = build_control_launcher_command(&workspace, args.mode)?;
    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to launch Factory Control")?;

    if status.success() {
        return Ok(());
    }

    Err(anyhow!(
        "Factory Control exited with status {}",
        status
            .code()
            .map(|code| code.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    ))
}

fn build_control_launcher_command(workspace: &Path, mode: ControlModeArg) -> Result<Command> {
    let mode = control_mode_name(mode);

    #[cfg(windows)]
    {
        let launcher = workspace.join("bin/factory-control.cmd");
        if !launcher.exists() {
            return Err(anyhow!(
                "Factory Control launcher not found at {}",
                launcher.display()
            ));
        }

        let mut command = Command::new("cmd");
        command
            .current_dir(workspace)
            .arg("/C")
            .arg(launcher)
            .arg(mode);
        return Ok(command);
    }

    #[cfg(not(windows))]
    {
        let launcher = workspace.join("bin/factory-control");
        if !launcher.exists() {
            return Err(anyhow!(
                "Factory Control launcher not found at {}",
                launcher.display()
            ));
        }

        let mut command = Command::new(launcher);
        command.current_dir(workspace).arg(mode);
        Ok(command)
    }
}

fn control_mode_name(mode: ControlModeArg) -> &'static str {
    match mode {
        ControlModeArg::Auto => "auto",
        ControlModeArg::Web => "web",
        ControlModeArg::Desktop => "desktop",
        ControlModeArg::Check => "check",
        ControlModeArg::Build => "build",
    }
}

fn resolve_control_workspace(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        let workspace = expand_user_path(path);
        if is_control_workspace(&workspace) {
            return Ok(workspace);
        }
        return Err(anyhow!(
            "workspace does not contain gui/ and bin/factory-control: {}",
            workspace.display()
        ));
    }

    if let Ok(raw) = std::env::var("OPENGATEWAY_WORKSPACE") {
        if !raw.trim().is_empty() {
            let workspace = expand_user_path(Path::new(raw.trim()));
            if is_control_workspace(&workspace) {
                return Ok(workspace);
            }
        }
    }

    if let Ok(current) = std::env::current_dir() {
        if let Some(found) = find_control_workspace_from(&current) {
            return Ok(found);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            if let Some(found) = find_control_workspace_from(parent) {
                return Ok(found);
            }
        }
    }

    Err(anyhow!(
        "could not find the Factory Control workspace; run this inside the repo or pass --workspace"
    ))
}

fn find_control_workspace_from(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|candidate| is_control_workspace(candidate))
        .map(Path::to_path_buf)
}

fn is_control_workspace(path: &Path) -> bool {
    path.join("gui/package.json").exists()
        && (path.join("bin/factory-control").exists()
            || path.join("bin/factory-control.cmd").exists())
}

fn command_init(args: InitArgs) -> Result<()> {
    let paths = build_paths()?;
    paths.ensure_runtime_dirs()?;

    let config_path = resolve_config_path(&paths.default_config, args.config.as_ref());
    let existed = config_path.exists();

    ensure_config_exists(
        &config_path,
        &args.backend_host,
        args.backend_port,
        &paths.auth_dir,
        args.force,
    )?;

    if existed && !args.force {
        println!("config already exists: {}", config_path.display());
    } else {
        println!("wrote config: {}", config_path.display());
    }
    println!("auth directory: {}", paths.auth_dir.display());
    Ok(())
}

fn command_setup(args: SetupArgs) -> Result<()> {
    let paths = build_paths()?;
    paths.ensure_runtime_dirs()?;

    println!("Step 1/5: Preparing config...");
    let config_path = resolve_config_path(&paths.default_config, args.config.as_ref());
    ensure_config_exists(
        &config_path,
        &args.backend_host,
        args.backend_port,
        &paths.auth_dir,
        false,
    )?;

    println!("Step 2/5: Starting local proxy...");
    command_start(StartArgs {
        config: Some(config_path.clone()),
        host: args.host.clone(),
        port: args.port,
        backend_host: args.backend_host.clone(),
        backend_port: args.backend_port,
        timeout: args.start_timeout,
        no_auto_install: false,
        api_key: args.api_key.clone(),
        models: args.models.clone(),
        max_body_bytes: args.max_body_bytes,
        max_inflight: args.max_inflight,
        max_queue: args.max_queue,
        queue_timeout_seconds: args.queue_timeout_seconds,
        per_client_rate: args.per_client_rate,
        per_client_burst: args.per_client_burst,
    })?;

    println!("Step 3/5: OAuth login...");
    let auth_store = AuthStore::new(paths.auth_dir.join("auth.json"));
    let existing = auth_store.count_openai_accounts().unwrap_or(0);
    if existing > 0 && !args.relogin {
        println!("Found existing credentials ({existing}), skipping login.");
    } else {
        command_login(LoginArgs {
            mode: if args.headless {
                LoginModeArg::Headless
            } else {
                LoginModeArg::Browser
            },
            config: Some(config_path.clone()),
            backend_host: args.backend_host.clone(),
            backend_port: args.backend_port,
            open_browser: args.open_browser,
            no_browser: args.no_browser,
            no_auto_install: false,
            verbose: args.verbose,
        })?;
    }

    println!("Step 4/5: Writing config...");
    let api_key = resolve_proxy_api_key(&paths.api_key_file, &args.api_key)?;
    let base_url = if args.base_url.trim().is_empty() {
        format!("http://{}:{}", args.host, args.port)
    } else {
        args.base_url.clone()
    };
    let model_ids = resolve_model_ids(&args.models);
    let factory_path = resolve_factory_config_path(args.factory_config.as_ref())?;
    let (added, updated, backup) = merge_factory_config(
        &factory_path,
        base_url.trim_end_matches('/'),
        &api_key,
        &model_ids,
    )?;
    println!("Legacy config updated: {}", factory_path.display());
    println!("Legacy custom models added: {added}, updated: {updated}");
    if let Some(backup_path) = backup {
        println!("Legacy config backup saved: {}", backup_path.display());
    }

    let factory_settings_path = resolve_factory_settings_path(None)?;
    let (settings_added, settings_updated, settings_backup, defaults_updated) =
        merge_factory_settings(
            &factory_settings_path,
            base_url.trim_end_matches('/'),
            &api_key,
            &model_ids,
        )?;
    println!(
        "Factory settings updated: {}",
        factory_settings_path.display()
    );
    println!("Factory custom models added: {settings_added}, updated: {settings_updated}");
    if defaults_updated {
        println!("Factory session and mission defaults now point to GPT-5.4 (XHigh).");
    }
    if let Some(backup_path) = settings_backup {
        println!("Factory settings backup saved: {}", backup_path.display());
    }

    println!("Step 5/5: Complete.");
    println!("Ready to use.");
    println!("Local API key: {}", mask_secret(&api_key));
    println!("Try:");
    println!("  - opengateway self-test");
    println!("  - opengateway show-key");
    println!("  - opengateway logs -f");
    Ok(())
}

fn command_sync_factory(args: SyncFactoryArgs) -> Result<()> {
    let paths = build_paths()?;
    paths.ensure_runtime_dirs()?;
    let api_key = resolve_proxy_api_key(&paths.api_key_file, &args.api_key)?;
    let model_ids = resolve_model_ids(&args.models);
    let base_url = if args.base_url.trim().is_empty() {
        format!("http://{}:{}", DEFAULT_FRONT_HOST, DEFAULT_FRONT_PORT)
    } else {
        args.base_url.clone()
    };
    let factory_path = resolve_factory_config_path(args.factory_config.as_ref())?;
    let factory_settings_path = resolve_factory_settings_path(args.factory_settings.as_ref())?;

    let (added, updated, backup) = merge_factory_config(
        &factory_path,
        base_url.trim_end_matches('/'),
        &api_key,
        &model_ids,
    )?;
    println!("Legacy config updated: {}", factory_path.display());
    println!("Legacy custom models added: {added}, updated: {updated}");
    if let Some(backup_path) = backup {
        println!("Legacy config backup saved: {}", backup_path.display());
    }

    let (settings_added, settings_updated, settings_backup, defaults_updated) =
        merge_factory_settings(
            &factory_settings_path,
            base_url.trim_end_matches('/'),
            &api_key,
            &model_ids,
        )?;
    println!(
        "Factory settings updated: {}",
        factory_settings_path.display()
    );
    println!("Factory custom models added: {settings_added}, updated: {settings_updated}");
    if defaults_updated {
        println!("Factory session and mission defaults now point to GPT-5.4 (XHigh).");
    }
    if let Some(backup_path) = settings_backup {
        println!("Factory settings backup saved: {}", backup_path.display());
    }

    Ok(())
}

fn command_start(args: StartArgs) -> Result<()> {
    let paths = build_paths()?;
    paths.ensure_runtime_dirs()?;
    clear_stale_pid_file(&paths.pid_file);

    if let Some(pid) = read_pid(&paths.pid_file) {
        if pid_running(pid) {
            println!("opengateway already running (pid={pid})");
            return Ok(());
        }
    }

    let config_path = resolve_config_path(&paths.default_config, args.config.as_ref());
    ensure_config_exists(
        &config_path,
        &args.backend_host,
        args.backend_port,
        &paths.auth_dir,
        false,
    )?;

    let api_key = resolve_proxy_api_key(&paths.api_key_file, &args.api_key)?;
    let run_args = args.to_run_args(api_key.clone());

    let mut command =
        Command::new(std::env::current_exe().context("failed to resolve executable path")?);
    command
        .arg("run")
        .arg("--host")
        .arg(&run_args.host)
        .arg("--port")
        .arg(run_args.port.to_string())
        .arg("--backend-host")
        .arg(&run_args.backend_host)
        .arg("--backend-port")
        .arg(run_args.backend_port.to_string())
        .arg("--api-key")
        .arg(&run_args.api_key)
        .arg("--max-body-bytes")
        .arg(run_args.max_body_bytes.to_string())
        .arg("--max-inflight")
        .arg(run_args.max_inflight.to_string())
        .arg("--max-queue")
        .arg(run_args.max_queue.to_string())
        .arg("--queue-timeout-seconds")
        .arg(run_args.queue_timeout_seconds.to_string())
        .arg("--per-client-rate")
        .arg(run_args.per_client_rate.to_string())
        .arg("--per-client-burst")
        .arg(run_args.per_client_burst.to_string());

    if !run_args.models.trim().is_empty() {
        command.arg("--models").arg(&run_args.models);
    }

    if let Some(config) = run_args.config.as_ref() {
        command.arg("--config").arg(expand_user_path(config));
    }
    if run_args.no_auto_install {
        command.arg("--no-auto-install");
    }

    let log_handle = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.log_file)
        .with_context(|| format!("failed to open log file {}", paths.log_file.display()))?;
    let err_handle = log_handle
        .try_clone()
        .context("failed to clone log handle")?;

    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_handle))
        .stderr(Stdio::from(err_handle));
    configure_background_command(&mut child);

    let mut child = child
        .spawn()
        .context("failed to start background process")?;

    let timeout = args.timeout.max(0.1);
    let deadline = Instant::now() + Duration::from_secs_f64(timeout);
    while Instant::now() < deadline {
        if is_http_ready(&args.host, args.port, Duration::from_millis(500)) {
            let pid = read_pid(&paths.pid_file).unwrap_or_else(|| child.id() as i32);
            println!(
                "opengateway started (pid={pid}) on http://{}:{}",
                args.host, args.port
            );
            println!("log file: {}", paths.log_file.display());
            return Ok(());
        }

        if child
            .try_wait()
            .context("failed checking child process state")?
            .is_some()
        {
            break;
        }
        thread::sleep(Duration::from_millis(200));
    }

    let _ = child.kill();
    println!("opengateway failed to start. recent logs:");
    if paths.log_file.exists() {
        for line in tail_file(&paths.log_file, 20)? {
            println!("{line}");
        }
    }
    Err(anyhow!("startup failed"))
}

fn command_run(args: RunArgs) -> Result<()> {
    let paths = build_paths()?;
    paths.ensure_runtime_dirs()?;
    clear_stale_pid_file(&paths.pid_file);

    let config_path = resolve_config_path(&paths.default_config, args.config.as_ref());
    ensure_config_exists(
        &config_path,
        &args.backend_host,
        args.backend_port,
        &paths.auth_dir,
        false,
    )?;

    if let Some(existing_pid) = read_pid(&paths.pid_file) {
        if pid_running(existing_pid) && existing_pid != std::process::id() as i32 {
            return Err(anyhow!("opengateway already running (pid={existing_pid})"));
        }
    }

    let api_key = resolve_proxy_api_key(&paths.api_key_file, &args.api_key)?;
    let model_ids = resolve_model_ids(&args.models);
    let pid = std::process::id() as i32;
    write_pid(&paths.pid_file, pid)?;
    runtime_log_info(format!("service starting (pid={pid})"));
    runtime_log_info(format!("using config: {}", config_path.display()));

    let auth_store = AuthStore::new(paths.auth_dir.join("auth.json"));
    let run_result = service::run_service(
        service::RunConfig {
            host: args.host,
            port: args.port,
            api_key,
            max_body_bytes: args.max_body_bytes,
            max_inflight: args.max_inflight,
            max_queue: args.max_queue,
            queue_timeout: Duration::from_secs_f64(args.queue_timeout_seconds.max(0.1)),
            per_client_rate: args.per_client_rate,
            per_client_burst: args.per_client_burst,
            models: model_ids,
        },
        auth_store,
    );

    remove_pid(&paths.pid_file);
    run_result
}

fn command_stop(args: StopArgs) -> Result<()> {
    let paths = build_paths()?;

    let Some(pid) = read_pid(&paths.pid_file) else {
        println!("opengateway is not running");
        return Ok(());
    };

    let health_up = is_http_ready(
        DEFAULT_FRONT_HOST,
        DEFAULT_FRONT_PORT,
        Duration::from_millis(500),
    );
    if !pid_running(pid) && !health_up {
        remove_pid(&paths.pid_file);
        println!("opengateway is not running");
        return Ok(());
    }

    send_signal(pid, "-TERM")?;
    let deadline = Instant::now() + Duration::from_secs_f64(args.timeout.max(0.1));
    while Instant::now() < deadline {
        if !pid_running(pid)
            && !is_http_ready(
                DEFAULT_FRONT_HOST,
                DEFAULT_FRONT_PORT,
                Duration::from_millis(500),
            )
        {
            remove_pid(&paths.pid_file);
            println!("opengateway stopped");
            return Ok(());
        }
        thread::sleep(Duration::from_millis(200));
    }

    if args.force {
        send_signal(pid, "-KILL")?;
        remove_pid(&paths.pid_file);
        println!("opengateway force-stopped");
        return Ok(());
    }

    Err(anyhow!(
        "opengateway did not stop before timeout (use --force)"
    ))
}

fn command_status(args: StatusArgs) -> Result<()> {
    let paths = build_paths()?;
    clear_stale_pid_file(&paths.pid_file);
    let pid = read_pid(&paths.pid_file);
    let front_up = is_http_ready(&args.host, args.port, Duration::from_millis(500));
    let running = front_up || pid.map(pid_running).unwrap_or(false);

    println!("running: {running}");
    println!(
        "pid: {}",
        pid.map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!(
        "front_proxy: {} ({}:{})",
        if front_up {
            "up"
        } else {
            "down"
        },
        args.host,
        args.port
    );
    println!(
        "backend: {} ({}:{})",
        if is_port_open(
            &args.backend_host,
            args.backend_port,
            Duration::from_millis(300)
        ) {
            "up"
        } else {
            "down"
        },
        args.backend_host,
        args.backend_port
    );
    println!("log_file: {}", paths.log_file.display());
    Ok(())
}

fn command_logs(args: LogsArgs) -> Result<()> {
    let paths = build_paths()?;
    if !paths.log_file.exists() {
        return Err(anyhow!("log file not found: {}", paths.log_file.display()));
    }

    for line in tail_file(&paths.log_file, args.lines)? {
        println!("{line}");
    }

    if args.follow {
        follow_file(&paths.log_file)?;
    }

    Ok(())
}

fn command_login(args: LoginArgs) -> Result<()> {
    let paths = build_paths()?;
    paths.ensure_runtime_dirs()?;

    let config_path = resolve_config_path(&paths.default_config, args.config.as_ref());
    ensure_config_exists(
        &config_path,
        &args.backend_host,
        args.backend_port,
        &paths.auth_dir,
        false,
    )?;

    let auth_store = AuthStore::new(paths.auth_dir.join("auth.json"));
    let before = auth_store.count_openai_accounts().unwrap_or(0);

    let mode = match args.mode {
        LoginModeArg::Browser => LoginMode::Browser,
        LoginModeArg::Headless => LoginMode::Headless,
    };

    let should_open_browser = args.open_browser && !args.no_browser;

    let login_result = oauth::login(mode, !should_open_browser, args.verbose)?;
    auth_store.upsert_openai(NewCredential {
        refresh_token: login_result.refresh_token,
        access_token: login_result.access_token,
        expires_at_ms: login_result.expires_at_ms,
        account_id: login_result.account_id,
    })?;

    let after = auth_store.count_openai_accounts().unwrap_or(before);
    if after > before {
        println!("Login completed and credentials were saved.");
    } else {
        println!("Login successful.");
    }
    println!("Auth store: {}", auth_store.path().display());
    Ok(())
}

fn command_show_key(args: ShowKeyArgs) -> Result<()> {
    let paths = build_paths()?;
    paths.ensure_runtime_dirs()?;
    let key = resolve_proxy_api_key(&paths.api_key_file, "")?;

    if args.quiet {
        println!("{key}");
        return Ok(());
    }

    println!("API key: {key}");
    println!("File: {}", paths.api_key_file.display());
    println!("Use it as: Authorization: Bearer {}", mask_secret(&key));
    Ok(())
}

fn command_self_test(args: SelfTestArgs) -> Result<()> {
    let paths = build_paths()?;
    paths.ensure_runtime_dirs()?;

    if !is_port_open(&args.host, args.port, Duration::from_millis(400)) {
        return Err(anyhow!(
            "proxy is not reachable on {}:{} (run `opengateway start` first)",
            args.host,
            args.port
        ));
    }

    let api_key = resolve_proxy_api_key(&paths.api_key_file, &args.api_key)?;
    let timeout = Duration::from_secs_f64(args.timeout.max(0.1));
    let client = Client::builder()
        .timeout(timeout)
        .build()
        .context("failed to create HTTP client for self-test")?;

    let base_url = format!("http://{}:{}", args.host, args.port);
    let health_url = format!("{base_url}/healthz");
    let models_url = format!("{base_url}/v1/models");

    let health = client
        .get(&health_url)
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .with_context(|| format!("failed request to {health_url}"))?;
    if !health.status().is_success() {
        return Err(anyhow!("healthz failed with status {}", health.status()));
    }

    let models = client
        .get(&models_url)
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .with_context(|| format!("failed request to {models_url}"))?;
    if !models.status().is_success() {
        return Err(anyhow!("models failed with status {}", models.status()));
    }

    println!("Self-test passed on {}:{}", args.host, args.port);
    println!("- /healthz: ok");
    println!("- /v1/models: ok");
    Ok(())
}

fn command_factory_config(args: FactoryConfigArgs) -> Result<()> {
    let paths = build_paths()?;
    paths.ensure_runtime_dirs()?;
    let api_key = resolve_proxy_api_key(&paths.api_key_file, &args.api_key)?;
    let model_ids = resolve_model_ids(&args.models);

    let data = build_factory_config(args.base_url.trim_end_matches('/'), &api_key, &model_ids);
    let rendered =
        serde_json::to_string_pretty(&data).context("failed to encode factory config")?;

    if let Some(path) = args.output.as_ref() {
        let output_path = expand_user_path(path);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
        fs::write(&output_path, format!("{rendered}\n"))
            .with_context(|| format!("failed to write {}", output_path.display()))?;
        println!("wrote {}", output_path.display());
    } else {
        println!("{rendered}");
    }

    Ok(())
}

fn command_doctor(args: DoctorArgs) -> Result<()> {
    let paths = build_paths()?;
    paths.ensure_runtime_dirs()?;
    let auth_store = AuthStore::new(paths.auth_dir.join("auth.json"));

    let env_api_key = std::env::var("OPENGATEWAY_API_KEY").unwrap_or_default();
    let stored_api_key = read_secret_file(&paths.api_key_file).unwrap_or_default();

    println!("version: {}", env!("CARGO_PKG_VERSION"));
    println!(
        "platform: {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!(
        "default_config: {} ({})",
        if paths.default_config.exists() {
            "present"
        } else {
            "missing"
        },
        paths.default_config.display()
    );
    println!("auth_dir: {}", paths.auth_dir.display());
    println!(
        "auth_store: {} ({})",
        if auth_store.path().exists() {
            "present"
        } else {
            "missing"
        },
        auth_store.path().display()
    );
    println!(
        "openai_accounts: {}",
        auth_store.count_openai_accounts().unwrap_or(0)
    );
    println!(
        "active_openai_account: {}",
        match auth_store.active_openai().ok().flatten() {
            Some(entry) => entry.account_id.unwrap_or_else(|| "default".to_string()),
            None => "none".to_string(),
        }
    );
    println!("log_file: {}", paths.log_file.display());
    println!(
        "pid_file: {} ({})",
        paths.pid_file.display(),
        if paths.pid_file.exists() {
            "present"
        } else {
            "missing"
        }
    );

    if !env_api_key.trim().is_empty() {
        println!("proxy_api_key: set via OPENGATEWAY_API_KEY");
    } else {
        println!(
            "proxy_api_key_file: {} ({})",
            if !stored_api_key.trim().is_empty() {
                "present"
            } else {
                "missing"
            },
            paths.api_key_file.display()
        );
    }

    println!(
        "front_port_{}: {}",
        args.port,
        if is_http_ready(&args.host, args.port, Duration::from_millis(500)) {
            "open"
        } else {
            "closed"
        }
    );
    println!(
        "backend_port_{}: {}",
        args.backend_port,
        if is_port_open(
            &args.backend_host,
            args.backend_port,
            Duration::from_millis(300)
        ) {
            "open"
        } else {
            "closed"
        }
    );
    Ok(())
}

fn resolve_config_path(default_path: &Path, raw: Option<&PathBuf>) -> PathBuf {
    match raw {
        Some(path) => expand_user_path(path),
        None => default_path.to_path_buf(),
    }
}

fn resolve_factory_config_path(raw: Option<&PathBuf>) -> Result<PathBuf> {
    match raw {
        Some(path) => Ok(expand_user_path(path)),
        None => Ok(paths::build_factory_paths()?.config_path),
    }
}

fn resolve_factory_settings_path(raw: Option<&PathBuf>) -> Result<PathBuf> {
    match raw {
        Some(path) => Ok(expand_user_path(path)),
        None => Ok(paths::build_factory_paths()?.settings_path),
    }
}

fn ensure_config_exists(
    config_path: &Path,
    host: &str,
    port: u16,
    auth_dir: &Path,
    force: bool,
) -> Result<()> {
    if config_path.exists() && !force {
        return Ok(());
    }
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(config_path, default_config_text(host, port, auth_dir))
        .with_context(|| format!("failed to write {}", config_path.display()))?;
    Ok(())
}

fn default_config_text(host: &str, port: u16, auth_dir: &Path) -> String {
    format!(
        "# Backend port used by opengateway Rust backend.\n\
port: {port}\n\
\n\
# Bind backend to localhost for security.\n\
host: \"{host}\"\n\
\n\
# Store OAuth credential files in this directory.\n\
auth-dir: \"{}\"\n\
\n\
debug: false\n\
logging-to-file: false\n\
usage-statistics-enabled: true\n\
proxy-url: \"\"\n\
request-retry: 3\n\
request-timeout: \"10m\"\n\
\n\
quota-exceeded:\n\
  switch-project: true\n\
  switch-preview-model: true\n\
\n\
# Management API disabled by default.\n\
remote-management:\n\
  allow-remote: false\n\
  secret-key: \"\"\n",
        auth_dir.display()
    )
}

fn resolve_model_ids(explicit_models: &str) -> Vec<String> {
    let explicit_models = explicit_models.trim();
    if !explicit_models.is_empty() {
        return parse_model_list(explicit_models);
    }

    let env_models = std::env::var("OPENGATEWAY_MODELS").unwrap_or_default();
    if !env_models.trim().is_empty() {
        return parse_model_list(&env_models);
    }

    DEFAULT_OPENAI_MODEL_CATALOG
        .iter()
        .map(|(model, _)| (*model).to_string())
        .collect()
}

fn parse_model_list(raw: &str) -> Vec<String> {
    let mut models = Vec::new();
    let mut seen = HashSet::new();

    for entry in raw.split(',') {
        let model = entry.trim();
        if model.is_empty() {
            continue;
        }
        if seen.insert(model.to_string()) {
            models.push(model.to_string());
        }
    }

    if models.is_empty() {
        DEFAULT_OPENAI_MODEL_CATALOG
            .iter()
            .map(|(model, _)| (*model).to_string())
            .collect()
    } else {
        models
    }
}

fn model_display_name(model_id: &str) -> String {
    DEFAULT_OPENAI_MODEL_CATALOG
        .iter()
        .find(|(candidate, _)| *candidate == model_id)
        .map(|(_, display_name)| (*display_name).to_string())
        .unwrap_or_else(|| model_id.to_string())
}

fn build_factory_config(base_url: &str, api_key: &str, model_ids: &[String]) -> Value {
    let models = model_ids
        .iter()
        .map(|model_id| {
            json!({
              "model_display_name": model_display_name(model_id),
              "model": model_id,
              "base_url": format!("{base_url}/v1"),
              "api_key": api_key,
              "provider": "openai"
            })
        })
        .collect::<Vec<_>>();

    json!({ "custom_models": models })
}

fn merge_factory_config(
    output_path: &Path,
    base_url: &str,
    api_key: &str,
    model_ids: &[String],
) -> Result<(usize, usize, Option<PathBuf>)> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let incoming_models = build_factory_config(base_url, api_key, model_ids)
        .get("custom_models")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut existing = json!({});
    let mut backup: Option<PathBuf> = None;
    if output_path.exists() {
        let original_name = output_path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_else(|| "config.json".to_string());
        let backup_path =
            output_path.with_file_name(format!("{original_name}.bak-{}", epoch_seconds()));
        fs::copy(output_path, &backup_path).with_context(|| {
            format!(
                "failed to create backup {} from {}",
                backup_path.display(),
                output_path.display()
            )
        })?;
        backup = Some(backup_path);

        existing = fs::read_to_string(output_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .unwrap_or_else(|| json!({}));
    }

    if !existing.is_object() {
        existing = json!({});
    }

    let object = existing
        .as_object_mut()
        .ok_or_else(|| anyhow!("internal error: expected JSON object"))?;

    let current_models = object
        .entry("custom_models")
        .or_insert_with(|| Value::Array(Vec::new()));
    if !current_models.is_array() {
        *current_models = Value::Array(Vec::new());
    }
    let current_models = current_models
        .as_array_mut()
        .ok_or_else(|| anyhow!("internal error: custom_models should be array"))?;

    let mut index_by_model: HashMap<String, usize> = HashMap::new();
    for (index, model) in current_models.iter().enumerate() {
        if let Some(name) = model.get("model").and_then(Value::as_str) {
            index_by_model.insert(name.to_string(), index);
        }
    }

    let mut added = 0;
    let mut updated = 0;
    for model in incoming_models {
        let Some(model_name) = model.get("model").and_then(Value::as_str) else {
            continue;
        };
        if let Some(index) = index_by_model.get(model_name).copied() {
            current_models[index] = model;
            updated += 1;
        } else {
            index_by_model.insert(model_name.to_string(), current_models.len());
            current_models.push(model);
            added += 1;
        }
    }

    let rendered =
        serde_json::to_string_pretty(&existing).context("failed to encode merged config")?;
    fs::write(output_path, format!("{rendered}\n"))
        .with_context(|| format!("failed to write {}", output_path.display()))?;

    Ok((added, updated, backup))
}

fn build_factory_settings_model(
    model_id: &str,
    base_url: &str,
    api_key: &str,
    index: usize,
) -> Value {
    let display_name = model_display_name(model_id);
    json!({
        "model": model_id,
        "id": factory_custom_model_id(&display_name, index),
        "index": index,
        "baseUrl": format!("{base_url}/v1"),
        "apiKey": api_key,
        "displayName": display_name,
        "noImageSupport": false,
        "provider": "openai"
    })
}

fn factory_custom_model_id(display_name: &str, index: usize) -> String {
    format!("custom:{}-{index}", display_name.replace(' ', "-"))
}

fn merge_factory_settings(
    output_path: &Path,
    base_url: &str,
    api_key: &str,
    model_ids: &[String],
) -> Result<(usize, usize, Option<PathBuf>, bool)> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let (existing, backup) = read_json_with_backup(output_path)?;
    let (merged, added, updated, defaults_updated) =
        merge_factory_settings_document(existing, base_url, api_key, model_ids)?;

    let rendered =
        serde_json::to_string_pretty(&merged).context("failed to encode merged settings")?;
    fs::write(output_path, format!("{rendered}\n"))
        .with_context(|| format!("failed to write {}", output_path.display()))?;

    Ok((added, updated, backup, defaults_updated))
}

fn merge_factory_settings_document(
    mut existing: Value,
    base_url: &str,
    api_key: &str,
    model_ids: &[String],
) -> Result<(Value, usize, usize, bool)> {
    if !existing.is_object() {
        existing = json!({});
    }

    let object = existing
        .as_object_mut()
        .ok_or_else(|| anyhow!("internal error: expected settings JSON object"))?;

    let managed_ids_before;
    let preferred_model_id;
    let added;
    let updated;
    {
        let current_models = object
            .entry("customModels")
            .or_insert_with(|| Value::Array(Vec::new()));
        if !current_models.is_array() {
            *current_models = Value::Array(Vec::new());
        }
        let current_models = current_models
            .as_array_mut()
            .ok_or_else(|| anyhow!("internal error: customModels should be array"))?;

        managed_ids_before = collect_managed_factory_model_ids(current_models, base_url, api_key);
        let merge_result =
            merge_factory_settings_models(current_models, base_url, api_key, model_ids);
        added = merge_result.0;
        updated = merge_result.1;
        preferred_model_id = merge_result.2;
    }

    let defaults_updated = update_factory_settings_defaults(
        object,
        &managed_ids_before,
        preferred_model_id.as_deref(),
    );

    Ok((existing, added, updated, defaults_updated))
}

fn collect_managed_factory_model_ids(
    current_models: &[Value],
    base_url: &str,
    api_key: &str,
) -> HashSet<String> {
    let expected_base_url = format!("{base_url}/v1");

    current_models
        .iter()
        .filter_map(Value::as_object)
        .filter(|model| {
            model.get("provider").and_then(Value::as_str) == Some("openai")
                && model.get("baseUrl").and_then(Value::as_str) == Some(expected_base_url.as_str())
                && model.get("apiKey").and_then(Value::as_str) == Some(api_key)
        })
        .filter_map(|model| model.get("id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn merge_factory_settings_models(
    current_models: &mut Vec<Value>,
    base_url: &str,
    api_key: &str,
    model_ids: &[String],
) -> (usize, usize, Option<String>) {
    let mut index_by_model: HashMap<String, usize> = HashMap::new();
    for (index, model) in current_models.iter().enumerate() {
        if let Some(name) = model.get("model").and_then(Value::as_str) {
            index_by_model.insert(name.to_string(), index);
        }
    }

    let mut added = 0;
    let mut updated = 0;
    let mut preferred_model_id = None;

    for model_id in model_ids {
        if let Some(index) = index_by_model.get(model_id).copied() {
            let mut replacement = build_factory_settings_model(model_id, base_url, api_key, index);
            if let (Some(existing), Some(replacement_object)) = (
                current_models[index].as_object(),
                replacement.as_object_mut(),
            ) {
                if let Some(existing_id) = existing.get("id").and_then(Value::as_str) {
                    replacement_object
                        .insert("id".to_string(), Value::String(existing_id.to_string()));
                }
                if let Some(existing_index) = existing.get("index").and_then(Value::as_u64) {
                    replacement_object
                        .insert("index".to_string(), Value::Number(existing_index.into()));
                }
            }
            if model_id == FACTORY_PREFERRED_MODEL {
                preferred_model_id = replacement
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
            current_models[index] = replacement;
            updated += 1;
        } else {
            let index = current_models.len();
            let model = build_factory_settings_model(model_id, base_url, api_key, index);
            if model_id == FACTORY_PREFERRED_MODEL {
                preferred_model_id = model.get("id").and_then(Value::as_str).map(str::to_string);
            }
            index_by_model.insert(model_id.to_string(), index);
            current_models.push(model);
            added += 1;
        }
    }

    (added, updated, preferred_model_id)
}

fn update_factory_settings_defaults(
    settings: &mut serde_json::Map<String, Value>,
    managed_ids_before: &HashSet<String>,
    preferred_model_id: Option<&str>,
) -> bool {
    let Some(preferred_model_id) = preferred_model_id else {
        return false;
    };

    let mut updated = false;

    let session_defaults = ensure_object_entry(settings, "sessionDefaultSettings");
    if should_update_factory_default_model(
        session_defaults.get("model").and_then(Value::as_str),
        managed_ids_before,
        preferred_model_id,
    ) {
        session_defaults.insert(
            "model".to_string(),
            Value::String(preferred_model_id.to_string()),
        );
        updated = true;
    }
    if session_defaults
        .get("model")
        .and_then(Value::as_str)
        .map(|value| value == preferred_model_id)
        .unwrap_or(false)
        && session_defaults
            .get("reasoningEffort")
            .and_then(Value::as_str)
            != Some(FACTORY_PREFERRED_REASONING_EFFORT)
    {
        session_defaults.insert(
            "reasoningEffort".to_string(),
            Value::String(FACTORY_PREFERRED_REASONING_EFFORT.to_string()),
        );
        updated = true;
    }

    let mission_defaults = ensure_object_entry(settings, "missionModelSettings");
    for (model_key, effort_key) in [
        ("orchestratorModel", "orchestratorReasoningEffort"),
        ("workerModel", "workerReasoningEffort"),
        ("validationWorkerModel", "validationWorkerReasoningEffort"),
    ] {
        if should_update_factory_default_model(
            mission_defaults.get(model_key).and_then(Value::as_str),
            managed_ids_before,
            preferred_model_id,
        ) {
            mission_defaults.insert(
                model_key.to_string(),
                Value::String(preferred_model_id.to_string()),
            );
            updated = true;
        }

        if mission_defaults
            .get(model_key)
            .and_then(Value::as_str)
            .map(|value| value == preferred_model_id)
            .unwrap_or(false)
            && mission_defaults.get(effort_key).and_then(Value::as_str)
                != Some(FACTORY_PREFERRED_REASONING_EFFORT)
        {
            mission_defaults.insert(
                effort_key.to_string(),
                Value::String(FACTORY_PREFERRED_REASONING_EFFORT.to_string()),
            );
            updated = true;
        }
    }

    updated
}

fn should_update_factory_default_model(
    current_model: Option<&str>,
    managed_ids_before: &HashSet<String>,
    preferred_model_id: &str,
) -> bool {
    match current_model {
        None => true,
        Some(value) if value == preferred_model_id => false,
        Some(value) => managed_ids_before.contains(value),
    }
}

fn ensure_object_entry<'a>(
    object: &'a mut serde_json::Map<String, Value>,
    key: &str,
) -> &'a mut serde_json::Map<String, Value> {
    let value = object
        .entry(key.to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !value.is_object() {
        *value = Value::Object(serde_json::Map::new());
    }
    value
        .as_object_mut()
        .expect("object entry should be an object after normalization")
}

fn read_json_with_backup(path: &Path) -> Result<(Value, Option<PathBuf>)> {
    let mut existing = json!({});
    let mut backup = None;

    if path.exists() {
        let original_name = path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_else(|| "config.json".to_string());
        let backup_path = path.with_file_name(format!("{original_name}.bak-{}", epoch_seconds()));
        fs::copy(path, &backup_path).with_context(|| {
            format!(
                "failed to create backup {} from {}",
                backup_path.display(),
                path.display()
            )
        })?;
        backup = Some(backup_path);

        existing = fs::read_to_string(path)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .unwrap_or_else(|| json!({}));
    }

    Ok((existing, backup))
}

fn expand_user_path(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        return paths::home_dir().unwrap_or_else(|_| path.to_path_buf());
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Ok(home) = paths::home_dir() {
            return home.join(rest);
        }
    }
    path.to_path_buf()
}

fn resolve_proxy_api_key(api_key_path: &Path, explicit: &str) -> Result<String> {
    let explicit = explicit.trim();
    if !explicit.is_empty() {
        write_secret_file(api_key_path, explicit)?;
        return Ok(explicit.to_string());
    }

    let env_key = std::env::var("OPENGATEWAY_API_KEY").unwrap_or_default();
    if !env_key.trim().is_empty() {
        return Ok(env_key.trim().to_string());
    }

    if let Some(existing) = read_secret_file(api_key_path) {
        if !existing.trim().is_empty() {
            return Ok(existing.trim().to_string());
        }
    }

    let generated = generate_secret(32);
    write_secret_file(api_key_path, &generated)?;
    Ok(generated)
}

fn read_secret_file(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|text| text.trim().to_string())
}

fn mask_secret(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "(empty)".to_string();
    }
    if trimmed.len() <= 8 {
        return "********".to_string();
    }
    format!("{}...{}", &trimmed[..4], &trimmed[trimmed.len() - 4..])
}

fn write_secret_file(path: &Path, value: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create secret directory {}", parent.display()))?;
    }

    let mut file = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .with_context(|| format!("failed to write secret file {}", path.display()))?;

    #[cfg(unix)]
    {
        file.set_permissions(fs::Permissions::from_mode(0o600))
            .with_context(|| format!("failed to set permissions on {}", path.display()))?;
    }

    writeln!(file, "{}", value.trim())
        .with_context(|| format!("failed to persist secret value into {}", path.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

fn generate_secret(byte_len: usize) -> String {
    let mut bytes = vec![0_u8; byte_len];
    rand::thread_rng().fill(&mut bytes[..]);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn read_pid(path: &Path) -> Option<i32> {
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| raw.trim().parse::<i32>().ok())
}

fn write_pid(path: &Path, pid: i32) -> Result<()> {
    let tmp_path = path.with_extension("pid.tmp");
    fs::write(&tmp_path, format!("{pid}\n"))
        .with_context(|| format!("failed to write pid temp file {}", tmp_path.display()))?;
    fs::rename(&tmp_path, path)
        .with_context(|| format!("failed to move pid file {}", path.display()))?;
    Ok(())
}

fn remove_pid(path: &Path) {
    let _ = fs::remove_file(path);
}

fn clear_stale_pid_file(path: &Path) {
    if let Some(pid) = read_pid(path) {
        if !pid_running(pid) {
            remove_pid(path);
        }
    }
}

fn pid_running(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }

    #[cfg(windows)]
    {
        let filter = format!("PID eq {pid}");
        let mut command = Command::new("tasklist");
        hidden_command(&mut command);
        let output = command
            .arg("/FO")
            .arg("CSV")
            .arg("/NH")
            .arg("/FI")
            .arg(filter)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output();

        return output
            .ok()
            .filter(|result| result.status.success())
            .and_then(|result| String::from_utf8(result.stdout).ok())
            .map(|stdout| stdout.trim_start().starts_with('"'))
            .unwrap_or(false);
    }

    #[cfg(not(windows))]
    {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
    }
}

fn send_signal(pid: i32, signal: &str) -> Result<()> {
    #[cfg(windows)]
    {
        let mut command = Command::new("taskkill");
        hidden_command(&mut command);
        command.arg("/PID").arg(pid.to_string()).arg("/T");
        if signal == "-KILL" {
            command.arg("/F");
        }

        let status = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .with_context(|| format!("failed to execute taskkill for pid {pid}"))?;

        return if status.success() {
            Ok(())
        } else {
            Err(anyhow!("taskkill {} {} failed", signal, pid))
        };
    }

    #[cfg(not(windows))]
    {
    let status = Command::new("kill")
        .arg(signal)
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("failed to execute kill for pid {pid}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("kill {} {} failed", signal, pid))
    }
    }
}

fn tail_file(path: &Path, lines: usize) -> Result<Vec<String>> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open log file {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut deque: VecDeque<String> = VecDeque::new();

    for line in reader.lines() {
        let line = line.with_context(|| format!("failed reading {}", path.display()))?;
        if lines == 0 {
            continue;
        }
        if deque.len() >= lines {
            deque.pop_front();
        }
        deque.push_back(line);
    }

    Ok(deque.into_iter().collect())
}

fn follow_file(path: &Path) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .read(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;

    let mut position = file
        .metadata()
        .with_context(|| format!("failed reading metadata for {}", path.display()))?
        .len();

    loop {
        let length = file
            .metadata()
            .with_context(|| format!("failed reading metadata for {}", path.display()))?
            .len();

        if length < position {
            position = 0;
        }

        if length > position {
            file.seek(SeekFrom::Start(position))
                .with_context(|| format!("failed seeking {}", path.display()))?;
            let mut chunk = String::new();
            file.read_to_string(&mut chunk)
                .with_context(|| format!("failed reading {}", path.display()))?;
            print!("{chunk}");
            std::io::stdout().flush().ok();
            position = length;
        }

        thread::sleep(Duration::from_millis(300));
    }
}

fn is_port_open(host: &str, port: u16, timeout: Duration) -> bool {
    let address = format!("{host}:{port}");
    if let Ok(socket_addr) = address.parse() {
        return std::net::TcpStream::connect_timeout(&socket_addr, timeout).is_ok();
    }
    false
}

fn is_http_ready(host: &str, port: u16, timeout: Duration) -> bool {
    let Ok(client) = Client::builder().timeout(timeout).build() else {
        return false;
    };

    let url = format!("http://{host}:{port}/healthz");
    client
        .get(url)
        .send()
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

fn epoch_seconds() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_secs() as i64
}

#[cfg(windows)]
fn configure_background_command(command: &mut Command) {
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn configure_background_command(_command: &mut Command) {}

#[cfg(windows)]
fn hidden_command(command: &mut Command) -> &mut Command {
    command.creation_flags(CREATE_NO_WINDOW);
    command
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_factory_custom_model_ids_from_display_names() {
        assert_eq!(
            factory_custom_model_id("GPT-5.4 (XHigh)", 24),
            "custom:GPT-5.4-(XHigh)-24"
        );
    }

    #[test]
    fn merges_factory_settings_and_upgrades_managed_defaults() {
        let existing = json!({
            "customModels": [
                {
                    "model": "gpt-5.3-codex(xhigh)",
                    "id": "custom:GPT-5.3-Codex-(XHigh)-3",
                    "index": 3,
                    "baseUrl": "http://127.0.0.1:42069/v1",
                    "apiKey": "secret",
                    "displayName": "GPT-5.3 Codex (XHigh)",
                    "noImageSupport": false,
                    "provider": "openai"
                }
            ],
            "sessionDefaultSettings": {
                "model": "custom:GPT-5.3-Codex-(XHigh)-3",
                "reasoningEffort": "xhigh"
            },
            "missionModelSettings": {
                "orchestratorModel": "custom:GPT-5.3-Codex-(XHigh)-3",
                "orchestratorReasoningEffort": "none",
                "workerModel": "custom:GPT-5.3-Codex-(XHigh)-3",
                "workerReasoningEffort": "none",
                "validationWorkerModel": "custom:GPT-5.3-Codex-(XHigh)-3",
                "validationWorkerReasoningEffort": "none"
            }
        });

        let model_ids = vec![
            "gpt-5.3-codex(xhigh)".to_string(),
            "gpt-5.4(xhigh)".to_string(),
        ];

        let (merged, added, updated, defaults_updated) = merge_factory_settings_document(
            existing,
            "http://127.0.0.1:42069",
            "secret",
            &model_ids,
        )
        .expect("settings merge should succeed");

        assert_eq!(added, 1);
        assert_eq!(updated, 1);
        assert!(defaults_updated);

        let custom_models = merged
            .get("customModels")
            .and_then(Value::as_array)
            .expect("customModels should be an array");
        let preferred_model = custom_models
            .iter()
            .find(|entry| entry.get("model").and_then(Value::as_str) == Some("gpt-5.4(xhigh)"))
            .expect("gpt-5.4(xhigh) should be present");
        let preferred_model_id = preferred_model
            .get("id")
            .and_then(Value::as_str)
            .expect("preferred custom model should have an id");

        assert_eq!(
            merged
                .get("sessionDefaultSettings")
                .and_then(Value::as_object)
                .and_then(|settings| settings.get("model"))
                .and_then(Value::as_str),
            Some(preferred_model_id)
        );
        assert_eq!(
            merged
                .get("missionModelSettings")
                .and_then(Value::as_object)
                .and_then(|settings| settings.get("workerReasoningEffort"))
                .and_then(Value::as_str),
            Some("xhigh")
        );
    }
}
