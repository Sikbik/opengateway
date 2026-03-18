use super::adapters::AgentKind;
use super::capabilities::planned_capabilities;
use super::doctor;
use super::supervisor::RuntimeMode;
use super::transport::{serve_stdio, ServeConfig};
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use std::env;
use std::io::{stdin, stdout, BufReader};
use std::path::{Path, PathBuf};

#[derive(Debug, Args)]
#[command(about = "Experimental ACP command surface")]
pub struct AcpArgs {
    #[command(subcommand)]
    pub command: AcpCommand,
}

#[derive(Debug, Subcommand)]
pub enum AcpCommand {
    Serve(AcpServeArgs),
    Doctor(AcpDoctorArgs),
    Sessions(AcpSessionsArgs),
    Inspect(AcpInspectArgs),
    Snapshot(AcpSnapshotArgs),
}

#[derive(Debug, Args)]
pub struct AcpServeArgs {
    #[arg(long, value_enum)]
    pub agent: AgentKind,
    #[arg(long)]
    pub workspace: Option<PathBuf>,
}

#[derive(Debug, Args, Default)]
pub struct AcpDoctorArgs {}

#[derive(Debug, Args, Default)]
pub struct AcpSessionsArgs {}

#[derive(Debug, Args)]
pub struct AcpInspectArgs {
    pub session_id: String,
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
}

#[derive(Debug, Args, Default)]
pub struct AcpSnapshotArgs {}

pub fn command_acp(args: AcpArgs) -> Result<()> {
    match args.command {
        AcpCommand::Serve(args) => command_serve(args),
        AcpCommand::Doctor(_) => command_doctor(),
        AcpCommand::Sessions(_) => command_sessions(),
        AcpCommand::Inspect(args) => command_inspect(args),
        AcpCommand::Snapshot(_) => command_snapshot(),
    }
}

fn command_serve(args: AcpServeArgs) -> Result<()> {
    let paths = crate::paths::build_paths()?;
    paths.ensure_runtime_dirs()?;

    let workspace = args
        .workspace
        .as_deref()
        .map(resolve_workspace)
        .transpose()?;

    eprintln!("ACP is experimental in this branch.");
    eprintln!("Agent: {}", args.agent.as_str());
    eprintln!("Transport: stdio");
    eprintln!("ACP state root: {}", paths.acp_dir.display());
    if let Some(workspace) = &workspace {
        eprintln!("Workspace: {}", workspace.display());
    }
    eprintln!(
        "Planned next capabilities: session/new={}, session/prompt={}, session/cancel={}, loadSession={}",
        planned_capabilities().sessions.new_session,
        planned_capabilities().sessions.prompt,
        planned_capabilities().sessions.cancel,
        planned_capabilities().sessions.load_session
    );

    let stdin = stdin();
    let stdout = stdout();
    let mut stderr = std::io::stderr().lock();

    serve_stdio(
        ServeConfig {
            agent: args.agent,
            workspace,
            paths: Some(paths.clone()),
            runtime_mode: RuntimeMode::Subprocess,
        },
        BufReader::new(stdin),
        &mut stdout.lock(),
        &mut stderr,
    )
}

fn command_doctor() -> Result<()> {
    let paths = crate::paths::build_paths()?;
    paths.ensure_runtime_dirs()?;
    print!("{}", doctor::render_doctor_report(&paths));
    Ok(())
}

fn command_sessions() -> Result<()> {
    let paths = crate::paths::build_paths()?;
    paths.ensure_runtime_dirs()?;
    let sessions = super::journal::collect_session_summaries(&paths)?;

    if sessions.is_empty() {
        println!("No ACP session journals found.");
        return Ok(());
    }

    println!("ACP sessions");
    for session in sessions {
        println!("{}", session.session_id);
        println!("  prompts: {}", session.prompt_count);
        println!(
            "  cwd: {}",
            session.cwd.as_deref().unwrap_or("unknown")
        );
        println!(
            "  model: {}",
            session.selected_model.as_deref().unwrap_or("default")
        );
        println!(
            "  last-event: {}",
            session.last_event.as_deref().unwrap_or("unknown")
        );
        println!("  journal: {}", session.journal_path.display());
        println!("  log: {}", session.log_path.display());
    }
    Ok(())
}

fn command_inspect(args: AcpInspectArgs) -> Result<()> {
    let paths = crate::paths::build_paths()?;
    paths.ensure_runtime_dirs()?;
    let detail = super::journal::inspect_session(&paths, &args.session_id, args.limit)?;

    let Some(detail) = detail else {
        anyhow::bail!("unknown ACP session: {}", args.session_id);
    };

    println!("ACP session inspect");
    println!("session: {}", detail.summary.session_id);
    println!("metrics:");
    println!("  sessions-created: {}", detail.metrics.sessions_created);
    println!("  prompts-completed: {}", detail.metrics.prompts_completed);
    println!("  prompts-cancelled: {}", detail.metrics.prompts_cancelled);
    println!("  runtime-failures: {}", detail.metrics.runtime_failures);
    println!("prompts: {}", detail.summary.prompt_count);
    println!(
        "cwd: {}",
        detail.summary.cwd.as_deref().unwrap_or("unknown")
    );
    println!(
        "model: {}",
        detail
            .summary
            .selected_model
            .as_deref()
            .unwrap_or("default")
    );
    println!(
        "last-event: {}",
        detail.summary.last_event.as_deref().unwrap_or("unknown")
    );
    println!(
        "last-ts-ms: {}",
        detail
            .summary
            .last_timestamp_ms
            .map(|timestamp| timestamp.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!("journal: {}", detail.summary.journal_path.display());
    println!("log: {}", detail.summary.log_path.display());

    println!("recent-events:");
    if detail.recent_events.is_empty() {
        println!("  none");
    } else {
        for event in &detail.recent_events {
            println!(
                "  {} {} {}",
                event
                    .timestamp_ms
                    .map(|timestamp| timestamp.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                event.event.as_deref().unwrap_or("unknown"),
                serde_json::to_string(&event.data)?,
            );
        }
    }

    println!("recent-log-lines:");
    if detail.recent_logs.is_empty() {
        println!("  none");
    } else {
        for line in &detail.recent_logs {
            println!("  {}", line);
        }
    }

    Ok(())
}

fn command_snapshot() -> Result<()> {
    let paths = crate::paths::build_paths()?;
    paths.ensure_runtime_dirs()?;
    let snapshot = super::snapshot::build_snapshot(&paths)?;
    println!("{}", serde_json::to_string_pretty(&snapshot)?);
    Ok(())
}

fn resolve_workspace(path: &Path) -> Result<PathBuf> {
    let workspace = expand_user_path(path);
    if workspace.is_dir() {
        return Ok(workspace);
    }
    Err(anyhow::anyhow!(
        "workspace path does not exist or is not a directory: {}",
        workspace.display()
    ))
}

fn expand_user_path(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }

    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }

    if path.is_absolute() {
        return path.to_path_buf();
    }

    env::current_dir()
        .context("failed to resolve current directory")
        .map(|cwd| cwd.join(path))
        .unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::{
        command_acp, expand_user_path, AcpArgs, AcpCommand, AcpInspectArgs, AcpSnapshotArgs,
    };
    use crate::acp::journal::append_journal_event;
    use crate::paths::build_paths;
    use serde_json::json;
    use std::fs;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn relative_workspace_paths_are_resolved() {
        let expanded = expand_user_path(Path::new("src"));
        assert!(expanded.ends_with("src"));
    }

    #[test]
    fn inspect_unknown_session_returns_error() {
        let error = command_acp(AcpArgs {
            command: AcpCommand::Inspect(AcpInspectArgs {
                session_id: "missing".to_string(),
                limit: 5,
            }),
        })
        .expect_err("inspect should fail");
        assert!(error.to_string().contains("unknown ACP session"));
    }

    #[test]
    fn inspect_command_succeeds_for_recorded_session() {
        let mut paths = build_paths().expect("paths");
        let test_root = std::env::temp_dir().join(format!(
            "opengateway-acp-cli-inspect-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        paths.state_dir = test_root.clone();
        paths.acp_dir = test_root.join("acp");
        paths.acp_sessions_dir = paths.acp_dir.join("sessions");
        paths.acp_logs_dir = paths.acp_dir.join("logs");
        paths.acp_tmp_dir = paths.acp_dir.join("tmp");
        fs::create_dir_all(&paths.acp_sessions_dir).expect("sessions dir");
        fs::create_dir_all(&paths.acp_logs_dir).expect("logs dir");
        append_journal_event(&paths, "session-1", "session.created", json!({"cwd": "/tmp"}))
            .expect("journal event");

        let prior = std::env::var_os("OPENGATEWAY_STATE_DIR");
        std::env::set_var("OPENGATEWAY_STATE_DIR", &test_root);
        let result = command_acp(AcpArgs {
            command: AcpCommand::Inspect(AcpInspectArgs {
                session_id: "session-1".to_string(),
                limit: 5,
            }),
        });
        match prior {
            Some(value) => std::env::set_var("OPENGATEWAY_STATE_DIR", value),
            None => std::env::remove_var("OPENGATEWAY_STATE_DIR"),
        }
        let _ = fs::remove_dir_all(test_root);

        result.expect("inspect command");
    }

    #[test]
    fn snapshot_command_succeeds() {
        let result = command_acp(AcpArgs {
            command: AcpCommand::Snapshot(AcpSnapshotArgs {}),
        });
        result.expect("snapshot command");
    }
}
