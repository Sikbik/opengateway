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

pub fn command_acp(args: AcpArgs) -> Result<()> {
    match args.command {
        AcpCommand::Serve(args) => command_serve(args),
        AcpCommand::Doctor(_) => command_doctor(),
        AcpCommand::Sessions(_) => command_sessions(),
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
            "  last-event: {}",
            session.last_event.as_deref().unwrap_or("unknown")
        );
        println!("  journal: {}", session.journal_path.display());
        println!("  log: {}", session.log_path.display());
    }
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
    use super::expand_user_path;
    use std::path::Path;

    #[test]
    fn relative_workspace_paths_are_resolved() {
        let expanded = expand_user_path(Path::new("src"));
        assert!(expanded.ends_with("src"));
    }
}
