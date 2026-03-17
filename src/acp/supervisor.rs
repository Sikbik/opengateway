use super::adapters::AgentKind;
use super::session::{CancelParams, InProcessMockSession, NewSessionParams, PromptBlock, PromptParams};
use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::io::{BufRead, BufReader, Write};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub const PROCESS_MODEL: &str = "one-session-one-subprocess";
const CANCEL_TIMEOUT: Duration = Duration::from_millis(400);
const MAX_ACTIVE_SESSIONS: usize = 8;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RuntimeMode {
    ProcessMock,
    InProcessMock,
}

#[derive(Debug, Args)]
pub struct MockRuntimeArgs {
    #[arg(long, value_enum)]
    pub agent: AgentKind,
    #[arg(long)]
    pub session_id: String,
    #[arg(long)]
    pub cwd: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub mcp_server_count: usize,
}

#[derive(Debug)]
pub struct SessionSupervisor {
    runtime_mode: RuntimeMode,
    next_id: u64,
    sessions: HashMap<String, SessionHandle>,
}

#[derive(Debug)]
enum SessionHandle {
    InProcess(InProcessMockSession),
    Process(ProcessSession),
}

#[derive(Debug)]
struct ProcessSession {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SessionCreated {
    pub session_id: String,
    pub cwd: PathBuf,
    pub mcp_server_count: usize,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PromptOutcome {
    pub session_id: String,
    pub reply_text: String,
    pub prompt_count: u64,
    pub message_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CancelOutcome {
    Cancelled,
    UnknownSession,
}

#[derive(Debug, Deserialize, Serialize)]
struct MockRuntimeRequest {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    prompt: Vec<PromptBlock>,
    #[serde(rename = "messageId", default)]
    message_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MockRuntimeResponse {
    text: String,
    prompt_count: u64,
}

impl SessionSupervisor {
    pub fn new(runtime_mode: RuntimeMode) -> Self {
        Self {
            runtime_mode,
            next_id: 0,
            sessions: HashMap::new(),
        }
    }

    pub fn create(&mut self, agent: AgentKind, params: NewSessionParams) -> Result<SessionCreated> {
        self.reap_exited()?;
        validate_session_cwd(&params.cwd)?;
        if self.sessions.len() >= MAX_ACTIVE_SESSIONS {
            bail!("ACP session limit reached ({MAX_ACTIVE_SESSIONS})");
        }

        self.next_id += 1;
        let session_id = format!("session-{:06}", self.next_id);
        let created = SessionCreated {
            session_id: session_id.clone(),
            cwd: params.cwd.clone(),
            mcp_server_count: params.mcp_servers.len(),
        };

        let handle = match self.runtime_mode {
            RuntimeMode::InProcessMock => {
                SessionHandle::InProcess(InProcessMockSession::new(session_id.clone(), agent, params))
            }
            RuntimeMode::ProcessMock => {
                SessionHandle::Process(ProcessSession::spawn(agent, &session_id, &params)?)
            }
        };

        self.sessions.insert(session_id, handle);
        Ok(created)
    }

    pub fn contains(&mut self, session_id: &str) -> bool {
        let _ = self.reap_exited();
        self.sessions.contains_key(session_id)
    }

    pub fn prompt(&mut self, params: PromptParams) -> Result<PromptOutcome> {
        let session_id = params.session_id.clone();
        let mut handle = self
            .sessions
            .remove(&session_id)
            .ok_or_else(|| anyhow!("unknown session: {}", session_id))?;

        let result = match &mut handle {
            SessionHandle::InProcess(session) => {
                let reply_text = session.build_mock_reply(&params);
                Ok(PromptOutcome {
                    session_id: session.id.clone(),
                    reply_text,
                    prompt_count: session.prompt_count,
                    message_id: params.message_id,
                })
            }
            SessionHandle::Process(session) => session.prompt(params),
        };

        if result.is_ok() {
            self.sessions.insert(session_id, handle);
        }
        result
    }

    pub fn cancel(&mut self, params: CancelParams) -> Result<CancelOutcome> {
        let Some(mut handle) = self.sessions.remove(&params.session_id) else {
            return Ok(CancelOutcome::UnknownSession);
        };

        match &mut handle {
            SessionHandle::InProcess(_) => Ok(CancelOutcome::Cancelled),
            SessionHandle::Process(session) => {
                session.cancel()?;
                Ok(CancelOutcome::Cancelled)
            }
        }
    }

    pub fn shutdown_all(&mut self) -> Result<()> {
        for (_, mut handle) in self.sessions.drain() {
            if let SessionHandle::Process(session) = &mut handle {
                session.cancel()?;
            }
        }
        Ok(())
    }

    fn reap_exited(&mut self) -> Result<()> {
        let mut exited = Vec::new();
        for (session_id, handle) in &mut self.sessions {
            if let SessionHandle::Process(session) = handle {
                if session
                    .child
                    .try_wait()
                    .with_context(|| format!("failed to poll ACP runtime for {}", session_id))?
                    .is_some()
                {
                    exited.push(session_id.clone());
                }
            }
        }

        for session_id in exited {
            self.sessions.remove(&session_id);
        }
        Ok(())
    }
}

impl ProcessSession {
    fn spawn(agent: AgentKind, session_id: &str, params: &NewSessionParams) -> Result<Self> {
        let mut command = Command::new(
            env::current_exe().context("failed to resolve current executable for ACP runtime")?,
        );
        command
            .arg("acp-mock-runtime")
            .arg("--agent")
            .arg(agent.as_str())
            .arg("--session-id")
            .arg(session_id)
            .arg("--cwd")
            .arg(&params.cwd)
            .arg("--mcp-server-count")
            .arg(params.mcp_servers.len().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);
        apply_allowed_env(&mut command);

        let mut child = command.spawn().context("failed to spawn ACP mock runtime")?;
        let stdin = child
            .stdin
            .take()
            .context("spawned ACP mock runtime has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("spawned ACP mock runtime has no stdout")?;

        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    fn prompt(&mut self, params: PromptParams) -> Result<PromptOutcome> {
        self.ensure_alive("prompt")?;
        write_json_line(
            &mut self.stdin,
            &MockRuntimeRequest {
                kind: "prompt".to_string(),
                prompt: params.prompt.clone(),
                message_id: params.message_id.clone(),
            },
        )?;
        let response: MockRuntimeResponse = read_json_line(&mut self.stdout)?;

        Ok(PromptOutcome {
            session_id: params.session_id,
            reply_text: response.text,
            prompt_count: response.prompt_count,
            message_id: params.message_id,
        })
    }

    fn cancel(&mut self) -> Result<()> {
        let _ = write_json_line(
            &mut self.stdin,
            &MockRuntimeRequest {
                kind: "cancel".to_string(),
                prompt: Vec::new(),
                message_id: None,
            },
        );

        let started = Instant::now();
        while started.elapsed() < CANCEL_TIMEOUT {
            if self
                .child
                .try_wait()
                .context("failed to poll ACP mock runtime during cancel")?
                .is_some()
            {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(25));
        }

        self.child
            .kill()
            .context("failed to kill ACP mock runtime after cancel timeout")?;
        let _ = self.child.wait();
        Ok(())
    }

    fn ensure_alive(&mut self, operation: &str) -> Result<()> {
        if let Some(status) = self
            .child
            .try_wait()
            .with_context(|| format!("failed to poll ACP mock runtime before {operation}"))?
        {
            bail!("ACP mock runtime exited unexpectedly before {operation}: {status}");
        }
        Ok(())
    }
}

pub fn command_mock_runtime(args: MockRuntimeArgs) -> Result<()> {
    let mut session = InProcessMockSession {
        id: args.session_id,
        agent: args.agent,
        cwd: args.cwd,
        mcp_server_count: args.mcp_server_count,
        prompt_count: 0,
    };
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    let mut reader = BufReader::new(stdin.lock());
    let mut frame = String::new();

    loop {
        frame.clear();
        if reader
            .read_line(&mut frame)
            .context("failed to read ACP mock runtime stdin")?
            == 0
        {
            break;
        }

        let trimmed = frame.trim();
        if trimmed.is_empty() {
            continue;
        }

        let request: MockRuntimeRequest = serde_json::from_str(trimmed)
            .with_context(|| format!("invalid ACP mock runtime frame: {trimmed}"))?;
        match request.kind.as_str() {
            "prompt" => {
                let params = PromptParams {
                    session_id: session.id.clone(),
                    prompt: request.prompt,
                    message_id: request.message_id,
                };
                let text = session.build_mock_reply(&params);
                write_json_line(
                    &mut stdout,
                    &MockRuntimeResponse {
                        text,
                        prompt_count: session.prompt_count,
                    },
                )?;
            }
            "cancel" => break,
            other => bail!("unsupported ACP mock runtime command: {other}"),
        }
    }

    Ok(())
}

fn write_json_line<W, T>(writer: &mut W, value: &T) -> Result<()>
where
    W: Write,
    T: Serialize,
{
    serde_json::to_writer(&mut *writer, value).context("failed to write ACP runtime message")?;
    writer
        .write_all(b"\n")
        .context("failed to terminate ACP runtime message")?;
    writer
        .flush()
        .context("failed to flush ACP runtime message")?;
    Ok(())
}

fn read_json_line<R, T>(reader: &mut R) -> Result<T>
where
    R: BufRead,
    T: for<'de> Deserialize<'de>,
{
    let mut line = String::new();
    if reader
        .read_line(&mut line)
        .context("failed to read ACP runtime response")?
        == 0
    {
        bail!("ACP mock runtime closed stdout unexpectedly");
    }
    serde_json::from_str(line.trim()).context("failed to parse ACP runtime response")
}

fn validate_session_cwd(path: &Path) -> Result<()> {
    if !path.is_absolute() {
        bail!("session cwd must be absolute: {}", path.display());
    }
    if !path.is_dir() {
        bail!("session cwd does not exist or is not a directory: {}", path.display());
    }
    Ok(())
}

fn apply_allowed_env(command: &mut Command) {
    command.env_clear();
    for key in [
        "HOME",
        "PATH",
        "TMPDIR",
        "TMP",
        "TEMP",
        "USERPROFILE",
        "SYSTEMROOT",
    ] {
        if let Ok(value) = env::var(key) {
            command.env(key, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CancelOutcome, RuntimeMode, SessionSupervisor, MAX_ACTIVE_SESSIONS};
    use crate::acp::adapters::AgentKind;
    use crate::acp::session::{CancelParams, NewSessionParams, PromptBlock, PromptParams};
    use std::path::PathBuf;

    #[test]
    fn in_process_supervisor_runs_prompt_and_cancel() {
        let mut supervisor = SessionSupervisor::new(RuntimeMode::InProcessMock);
        let created = supervisor
            .create(
                AgentKind::Codex,
                NewSessionParams {
                    cwd: PathBuf::from("/tmp"),
                    mcp_servers: vec![],
                },
            )
            .expect("create");

        let prompt = supervisor
            .prompt(PromptParams {
                session_id: created.session_id.clone(),
                prompt: vec![PromptBlock {
                    kind: "text".to_string(),
                    text: Some("hello".to_string()),
                }],
                message_id: Some("user-1".to_string()),
            })
            .expect("prompt");
        assert_eq!(prompt.prompt_count, 1);
        assert!(prompt.reply_text.contains("hello"));

        let cancelled = supervisor
            .cancel(CancelParams {
                session_id: created.session_id,
            })
            .expect("cancel");
        assert_eq!(cancelled, CancelOutcome::Cancelled);
    }

    #[test]
    fn unknown_cancel_returns_unknown_session() {
        let mut supervisor = SessionSupervisor::new(RuntimeMode::InProcessMock);
        let cancelled = supervisor
            .cancel(CancelParams {
                session_id: "missing".to_string(),
            })
            .expect("cancel");
        assert_eq!(cancelled, CancelOutcome::UnknownSession);
    }

    #[test]
    fn supervisor_enforces_active_session_cap() {
        let mut supervisor = SessionSupervisor::new(RuntimeMode::InProcessMock);
        for _ in 0..MAX_ACTIVE_SESSIONS {
            supervisor
                .create(
                    AgentKind::Codex,
                    NewSessionParams {
                        cwd: PathBuf::from("/tmp"),
                        mcp_servers: vec![],
                    },
                )
                .expect("create within cap");
        }

        let error = supervisor
            .create(
                AgentKind::Codex,
                NewSessionParams {
                    cwd: PathBuf::from("/tmp"),
                    mcp_servers: vec![],
                },
            )
            .expect_err("cap should reject");
        assert!(error.to_string().contains("session limit reached"));
    }
}
