use super::adapters::AgentKind;
use super::snapshot::AcpBridgeSnapshot;
use super::transport::ServeConfig;
use super::transport::serve_stdio;
use super::supervisor::RuntimeMode;
use crate::paths::AppPaths;
use anyhow::{anyhow, Context, Result};
use clap::{Args, Subcommand};
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::BufReader;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

pub const DEFAULT_BRIDGE_HOST: &str = "127.0.0.1";
const CODEX_BRIDGE_PORT: u16 = 42170;
const CLAUDE_BRIDGE_PORT: u16 = 42171;

#[derive(Debug, Args)]
pub struct AcpBridgeArgs {
    #[command(subcommand)]
    pub command: AcpBridgeCommand,
}

#[derive(Debug, Subcommand)]
pub enum AcpBridgeCommand {
    Start(AcpBridgeStartArgs),
    Stop(AcpBridgeStopArgs),
    Status(AcpBridgeStatusArgs),
}

#[derive(Debug, Args)]
pub struct AcpBridgeStartArgs {
    #[arg(long, value_enum)]
    pub agent: AgentKind,
    #[arg(long, default_value = DEFAULT_BRIDGE_HOST)]
    pub host: String,
    #[arg(long)]
    pub port: Option<u16>,
    #[arg(long)]
    pub workspace: Option<PathBuf>,
    #[arg(long, default_value_t = 8.0)]
    pub timeout: f64,
}

#[derive(Debug, Args)]
pub struct AcpBridgeStopArgs {
    #[arg(long, value_enum)]
    pub agent: AgentKind,
    #[arg(long, default_value_t = 6.0)]
    pub timeout: f64,
    #[arg(long)]
    pub force: bool,
}

#[derive(Debug, Args, Default)]
pub struct AcpBridgeStatusArgs {}

#[derive(Debug, Args)]
pub struct AcpBridgeRunArgs {
    #[arg(long, value_enum)]
    pub agent: AgentKind,
    #[arg(long, default_value = DEFAULT_BRIDGE_HOST)]
    pub host: String,
    #[arg(long)]
    pub port: Option<u16>,
    #[arg(long)]
    pub workspace: Option<PathBuf>,
}

#[derive(Debug)]
struct BridgeRuntimeFiles {
    pid_path: PathBuf,
    log_path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct BridgeStatus {
    pub agent: &'static str,
    pub running: bool,
    pub pid: Option<i32>,
    pub host: String,
    pub port: u16,
    pub endpoint: String,
    pub log_path: String,
    pub last_log_line: Option<String>,
}

pub fn command_bridge(args: AcpBridgeArgs) -> Result<()> {
    match args.command {
        AcpBridgeCommand::Start(args) => command_bridge_start(args),
        AcpBridgeCommand::Stop(args) => command_bridge_stop(args),
        AcpBridgeCommand::Status(_) => command_bridge_status(),
    }
}

pub fn command_bridge_run(args: AcpBridgeRunArgs) -> Result<()> {
    let paths = crate::paths::build_paths()?;
    paths.ensure_runtime_dirs()?;
    let files = bridge_runtime_files(&paths, args.agent);
    crate::clear_stale_pid_file(&files.pid_path);

    let pid = std::process::id() as i32;
    crate::write_pid(&files.pid_path, pid)?;
    let host = args.host;
    let port = args.port.unwrap_or(default_bridge_port(args.agent));
    let listener = TcpListener::bind((host.as_str(), port)).with_context(|| {
        format!(
            "failed to bind ACP {} bridge on {}:{}",
            args.agent.as_str(),
            host,
            port
        )
    })?;

    crate::runtime_log_info(format!(
        "acp {} bridge listening on tcp://{}:{}",
        args.agent.as_str(),
        host,
        port
    ));

    let serve_config = ServeConfig {
        agent: args.agent,
        workspace: args.workspace,
        paths: Some(paths.clone()),
        runtime_mode: RuntimeMode::Subprocess,
        debug_trace_path: None,
    };

    let run_result = run_bridge_listener(listener, serve_config, files.log_path.clone());
    crate::remove_pid(&files.pid_path);
    run_result
}

pub fn collect_bridge_statuses(paths: &AppPaths) -> Vec<AcpBridgeSnapshot> {
    [AgentKind::Codex, AgentKind::Claude]
        .into_iter()
        .map(|agent| {
            let status = read_bridge_status(paths, agent);
            AcpBridgeSnapshot {
                agent: status.agent.to_string(),
                running: status.running,
                pid: status.pid,
                host: status.host,
                port: status.port,
                endpoint: status.endpoint,
                log_path: status.log_path,
                last_log_line: status.last_log_line,
            }
        })
        .collect()
}

fn command_bridge_start(args: AcpBridgeStartArgs) -> Result<()> {
    let paths = crate::paths::build_paths()?;
    paths.ensure_runtime_dirs()?;
    let files = bridge_runtime_files(&paths, args.agent);
    crate::clear_stale_pid_file(&files.pid_path);

    if let Some(pid) = crate::read_pid(&files.pid_path) {
        if crate::pid_running(pid) {
            println!(
                "ACP {} bridge already running (pid={pid}) at tcp://{}:{}",
                args.agent.as_str(),
                args.host,
                args.port.unwrap_or(default_bridge_port(args.agent))
            );
            println!("log file: {}", files.log_path.display());
            return Ok(());
        }
    }

    let background_executable = crate::resolve_background_executable(&paths)?;
    let mut command = std::process::Command::new(background_executable);
    command
        .arg("acp-bridge-run")
        .arg("--agent")
        .arg(args.agent.as_str())
        .arg("--host")
        .arg(&args.host)
        .arg("--port")
        .arg(args.port.unwrap_or(default_bridge_port(args.agent)).to_string());
    if let Some(workspace) = args.workspace.as_ref() {
        command.arg("--workspace").arg(crate::expand_user_path(workspace));
    }

    let mut child = crate::spawn_background_command(command, &files.log_path)?;
    let timeout = args.timeout.max(0.1);
    let deadline = Instant::now() + Duration::from_secs_f64(timeout);
    let port = args.port.unwrap_or(default_bridge_port(args.agent));

    while Instant::now() < deadline {
        let pid = crate::read_pid(&files.pid_path)
            .or_else(|| child.as_ref().map(|process| process.id() as i32));
        let bridge_ready = is_bridge_ready(&args.host, port, Duration::from_millis(500));
        if bridge_ready && pid.map(crate::pid_running).unwrap_or(false) {
            let pid = pid.unwrap_or_default();
            println!(
                "ACP {} bridge started (pid={pid}) at tcp://{}:{}",
                args.agent.as_str(),
                args.host,
                port
            );
            println!("log file: {}", files.log_path.display());
            return Ok(());
        }

        if let Some(child) = child.as_mut() {
            if child
                .try_wait()
                .context("failed checking ACP bridge child process state")?
                .is_some()
            {
                break;
            }
        }
        thread::sleep(Duration::from_millis(200));
    }

    if let Some(child) = child.as_mut() {
        let _ = child.kill();
    } else if let Some(pid) = crate::read_pid(&files.pid_path) {
        let _ = crate::send_signal(pid, "-TERM");
    }
    println!("ACP {} bridge failed to start. recent logs:", args.agent.as_str());
    if files.log_path.exists() {
        for line in crate::tail_file(&files.log_path, 20)? {
            println!("{line}");
        }
    }
    Err(anyhow!("startup failed"))
}

fn command_bridge_stop(args: AcpBridgeStopArgs) -> Result<()> {
    let paths = crate::paths::build_paths()?;
    let files = bridge_runtime_files(&paths, args.agent);

    let Some(pid) = crate::read_pid(&files.pid_path) else {
        println!("ACP {} bridge is not running", args.agent.as_str());
        return Ok(());
    };

    let host = DEFAULT_BRIDGE_HOST.to_string();
    let port = default_bridge_port(args.agent);
    if !crate::pid_running(pid) && !is_bridge_ready(&host, port, Duration::from_millis(500)) {
        crate::remove_pid(&files.pid_path);
        println!("ACP {} bridge is not running", args.agent.as_str());
        return Ok(());
    }

    crate::send_signal(pid, "-TERM")?;
    let deadline = Instant::now() + Duration::from_secs_f64(args.timeout.max(0.1));
    while Instant::now() < deadline {
        if !crate::pid_running(pid) && !is_bridge_ready(&host, port, Duration::from_millis(500)) {
            crate::remove_pid(&files.pid_path);
            println!("ACP {} bridge stopped", args.agent.as_str());
            return Ok(());
        }
        thread::sleep(Duration::from_millis(200));
    }

    if args.force {
        crate::send_signal(pid, "-KILL")?;
        crate::remove_pid(&files.pid_path);
        println!("ACP {} bridge force-stopped", args.agent.as_str());
        return Ok(());
    }

    Err(anyhow!(
        "ACP {} bridge did not stop before timeout",
        args.agent.as_str()
    ))
}

fn command_bridge_status() -> Result<()> {
    let paths = crate::paths::build_paths()?;
    paths.ensure_runtime_dirs()?;
    println!("ACP bridges");
    for bridge in [AgentKind::Codex, AgentKind::Claude]
        .into_iter()
        .map(|agent| read_bridge_status(&paths, agent))
    {
        println!("{}", bridge.agent);
        println!("  running: {}", if bridge.running { "yes" } else { "no" });
        println!(
            "  pid: {}",
            bridge
                .pid
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string())
        );
        println!("  endpoint: {}", bridge.endpoint);
        println!("  log: {}", bridge.log_path);
    }
    Ok(())
}

fn run_bridge_listener(
    listener: TcpListener,
    serve_config: ServeConfig,
    log_path: PathBuf,
) -> Result<()> {
    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let config = serve_config.clone();
                let connection_log_path = log_path.clone();
                thread::spawn(move || {
                    if let Err(error) = handle_bridge_connection(stream, config, &connection_log_path)
                    {
                        let _ = append_bridge_log(
                            &connection_log_path,
                            &format!("bridge connection error: {error:#}"),
                        );
                    }
                });
            }
            Err(error) => {
                append_bridge_log(&log_path, &format!("bridge accept error: {error}"))?;
            }
        }
    }

    Ok(())
}

fn handle_bridge_connection(
    stream: TcpStream,
    serve_config: ServeConfig,
    log_path: &PathBuf,
) -> Result<()> {
    let peer = stream
        .peer_addr()
        .map(format_peer)
        .unwrap_or_else(|_| "unknown".to_string());
    append_bridge_log(
        log_path,
        &format!(
            "accepted ACP {} bridge client from {peer}",
            serve_config.agent.as_str()
        ),
    )?;

    let input = BufReader::new(
        stream
            .try_clone()
            .context("failed to clone ACP bridge stream")?,
    );
    let mut output = stream;
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to open bridge log {}", log_path.display()))?;

    serve_stdio(serve_config, input, &mut output, &mut log)
}

fn read_bridge_status(paths: &AppPaths, agent: AgentKind) -> BridgeStatus {
    let files = bridge_runtime_files(paths, agent);
    crate::clear_stale_pid_file(&files.pid_path);
    let host = DEFAULT_BRIDGE_HOST.to_string();
    let port = default_bridge_port(agent);
    let pid = crate::read_pid(&files.pid_path);
    let running = pid.map(crate::pid_running).unwrap_or(false)
        && is_bridge_ready(&host, port, Duration::from_millis(250));

    BridgeStatus {
        agent: agent.as_str(),
        running,
        pid,
        host: host.clone(),
        port,
        endpoint: format!("tcp://{host}:{port}"),
        log_path: files.log_path.display().to_string(),
        last_log_line: crate::tail_file(&files.log_path, 1)
            .ok()
            .and_then(|lines| lines.into_iter().last()),
    }
}

fn bridge_runtime_files(paths: &AppPaths, agent: AgentKind) -> BridgeRuntimeFiles {
    let stem = match agent {
        AgentKind::Codex => "codex",
        AgentKind::Claude => "claude",
    };
    BridgeRuntimeFiles {
        pid_path: paths.acp_bridges_dir.join(format!("{stem}.pid")),
        log_path: paths.acp_bridges_dir.join(format!("{stem}.log")),
    }
}

pub fn default_bridge_port(agent: AgentKind) -> u16 {
    match agent {
        AgentKind::Codex => CODEX_BRIDGE_PORT,
        AgentKind::Claude => CLAUDE_BRIDGE_PORT,
    }
}

fn is_bridge_ready(host: &str, port: u16, timeout: Duration) -> bool {
    let Ok(address) = format!("{host}:{port}").parse::<SocketAddr>() else {
        return false;
    };
    TcpStream::connect_timeout(&address, timeout).is_ok()
}

fn append_bridge_log(path: &PathBuf, line: &str) -> Result<()> {
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open bridge log {}", path.display()))?;
    use std::io::Write;
    writeln!(log, "{line}")
        .with_context(|| format!("failed to append bridge log {}", path.display()))?;
    Ok(())
}

fn format_peer(address: SocketAddr) -> String {
    address.to_string()
}
