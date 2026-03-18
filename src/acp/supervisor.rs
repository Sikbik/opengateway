use super::adapters::{codex, AgentKind};
use super::journal::highest_session_sequence;
use super::session::{
    CancelParams, ChildRuntimeEnvelope, ChildRuntimeRequest, ChildRuntimeResponse,
    ChildRuntimeStopReason, ChildRuntimeUpdate, InProcessMockSession, NewSessionParams, PromptParams,
};
use anyhow::{anyhow, bail, Context, Result};
use crate::paths::AppPaths;
use clap::{Args, ValueEnum};
use std::collections::HashMap;
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread::JoinHandle;
use std::thread;
use std::time::{Duration, Instant};

pub const PROCESS_MODEL: &str = "one-session-one-subprocess";
const CANCEL_TIMEOUT: Duration = Duration::from_millis(400);
const MAX_ACTIVE_SESSIONS: usize = 8;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum RuntimeMode {
    Subprocess,
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
    stdout: Option<BufReader<ChildStdout>>,
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
    pub stop_reason: ChildRuntimeStopReason,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CancelOutcome {
    Cancelled,
    UnknownSession,
}

impl SessionSupervisor {
    pub fn new(runtime_mode: RuntimeMode) -> Self {
        Self::with_seed(runtime_mode, 0)
    }

    pub fn new_with_persisted_counter(
        runtime_mode: RuntimeMode,
        paths: &AppPaths,
    ) -> Result<Self> {
        Ok(Self::with_seed(
            runtime_mode,
            highest_session_sequence(paths)?,
        ))
    }

    fn with_seed(runtime_mode: RuntimeMode, next_id: u64) -> Self {
        Self {
            runtime_mode,
            next_id,
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
            RuntimeMode::Subprocess | RuntimeMode::ProcessMock => {
                SessionHandle::Process(ProcessSession::spawn(
                    self.runtime_mode,
                    agent,
                    &session_id,
                    &params,
                )?)
            }
        };

        self.sessions.insert(session_id, handle);
        Ok(created)
    }

    pub fn contains(&mut self, session_id: &str) -> bool {
        let _ = self.reap_exited();
        self.sessions.contains_key(session_id)
    }

    pub fn prompt<F>(
        &mut self,
        params: PromptParams,
        cancel_rx: &Receiver<()>,
        mut on_update: F,
    ) -> Result<PromptOutcome>
    where
        F: FnMut(ChildRuntimeUpdate) -> Result<()>,
    {
        let session_id = params.session_id.clone();
        let mut handle = self
            .sessions
            .remove(&session_id)
            .ok_or_else(|| anyhow!("unknown session: {}", session_id))?;

        let result = match &mut handle {
            SessionHandle::InProcess(session) => {
                if cancel_rx.try_recv().is_ok() {
                    on_update(ChildRuntimeUpdate::TurnCancelled)?;
                    Ok(PromptOutcome {
                        session_id: session.id.clone(),
                        reply_text: String::new(),
                        prompt_count: session.prompt_count,
                        message_id: params.message_id,
                        stop_reason: ChildRuntimeStopReason::Cancelled,
                    })
                } else {
                    let reply_text = session.build_mock_reply(&params);
                    on_update(ChildRuntimeUpdate::AgentMessage {
                        text: reply_text.clone(),
                    })?;
                    Ok(PromptOutcome {
                        session_id: session.id.clone(),
                        reply_text,
                        prompt_count: session.prompt_count,
                        message_id: params.message_id,
                        stop_reason: ChildRuntimeStopReason::EndTurn,
                    })
                }
            }
            SessionHandle::Process(session) => session.prompt(params, cancel_rx, &mut on_update),
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
    fn spawn(
        runtime_mode: RuntimeMode,
        agent: AgentKind,
        session_id: &str,
        params: &NewSessionParams,
    ) -> Result<Self> {
        let mut command = Command::new(
            env::current_exe().context("failed to resolve current executable for ACP runtime")?,
        );
        match runtime_mode {
            RuntimeMode::Subprocess => match agent {
                AgentKind::Codex => {
                    let readiness = codex::inspect_runtime();
                    if !readiness.ready {
                        let issue = readiness
                            .issue
                            .as_deref()
                            .unwrap_or("unknown Codex runtime issue");
                        bail!(
                            "ACP Codex runtime is not ready: {issue}. {}",
                            codex::spawn_guidance(&readiness)
                        );
                    }
                    command
                        .arg("acp-codex-runtime")
                        .arg("--session-id")
                        .arg(session_id)
                        .arg("--cwd")
                        .arg(&params.cwd)
                        .arg("--mcp-server-count")
                        .arg(params.mcp_servers.len().to_string());
                }
                AgentKind::Claude => bail!("ACP Claude runtime is not implemented yet"),
            },
            RuntimeMode::ProcessMock => {
                command
                    .arg("acp-mock-runtime")
                    .arg("--agent")
                    .arg(agent.as_str())
                    .arg("--session-id")
                    .arg(session_id)
                    .arg("--cwd")
                    .arg(&params.cwd)
                    .arg("--mcp-server-count")
                    .arg(params.mcp_servers.len().to_string());
            }
            RuntimeMode::InProcessMock => bail!("in-process mock runtime cannot spawn subprocesses"),
        }
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);
        apply_allowed_env(&mut command);

        let mut child = command.spawn().with_context(|| match agent {
            AgentKind::Codex => {
                let readiness = codex::inspect_runtime();
                let issue = readiness
                    .issue
                    .as_deref()
                    .unwrap_or("failed to start the Codex child runtime");
                format!(
                    "failed to spawn ACP session runtime: {issue}. {}",
                    codex::spawn_guidance(&readiness)
                )
            }
            AgentKind::Claude => "failed to spawn ACP session runtime".to_string(),
        })?;
        let stdin = child
            .stdin
            .take()
            .context("spawned ACP session runtime has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("spawned ACP session runtime has no stdout")?;

        Ok(Self {
            child,
            stdin,
            stdout: Some(BufReader::new(stdout)),
        })
    }

    fn prompt<F>(
        &mut self,
        params: PromptParams,
        cancel_rx: &Receiver<()>,
        on_update: &mut F,
    ) -> Result<PromptOutcome>
    where
        F: FnMut(ChildRuntimeUpdate) -> Result<()>,
    {
        self.ensure_alive("prompt")?;
        write_json_line(
            &mut self.stdin,
            &ChildRuntimeRequest {
                kind: "prompt".to_string(),
                prompt: params.prompt.clone(),
                message_id: params.message_id.clone(),
            },
        )?;
        let stdout = self
            .stdout
            .take()
            .context("ACP session runtime stdout already in use")?;
        let (envelope_tx, envelope_rx) = mpsc::channel();
        let reader = spawn_runtime_reader(stdout, envelope_tx);
        let mut cancelled = false;
        loop {
            if !cancelled && cancel_rx.try_recv().is_ok() {
                self.cancel_prompt()?;
                cancelled = true;
            }

            match envelope_rx.recv_timeout(Duration::from_millis(25)) {
                Ok(ChildRuntimeEnvelope::Update { update }) => on_update(update)?,
                Ok(ChildRuntimeEnvelope::Result { result }) => {
                    self.stdout = Some(join_runtime_reader(reader)?);
                    return Ok(PromptOutcome {
                        session_id: params.session_id,
                        reply_text: result.text,
                        prompt_count: result.prompt_count,
                        message_id: params.message_id,
                        stop_reason: result.stop_reason,
                    });
                }
                Err(RecvTimeoutError::Timeout) => {
                    self.ensure_alive("prompt")?;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let read_error = match reader.join() {
                        Ok(result) => result.err(),
                        Err(_) => Some(anyhow!("ACP session runtime reader thread panicked")),
                    };
                    if let Some(error) = read_error {
                        bail!("failed to read ACP session runtime output: {error}");
                    }
                    bail!("ACP session runtime output closed unexpectedly during prompt");
                }
            }
        }
    }

    fn cancel_prompt(&mut self) -> Result<()> {
        write_json_line(
            &mut self.stdin,
            &ChildRuntimeRequest {
                kind: "cancel_prompt".to_string(),
                prompt: Vec::new(),
                message_id: None,
            },
        )
    }

    fn cancel(&mut self) -> Result<()> {
        let _ = write_json_line(
            &mut self.stdin,
            &ChildRuntimeRequest {
                kind: "shutdown".to_string(),
                prompt: Vec::new(),
                message_id: None,
            },
        );

        let started = Instant::now();
        while started.elapsed() < CANCEL_TIMEOUT {
            if self
                .child
                .try_wait()
                .context("failed to poll ACP session runtime during cancel")?
                .is_some()
            {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(25));
        }

        self.child
            .kill()
            .context("failed to kill ACP session runtime after cancel timeout")?;
        let _ = self.child.wait();
        Ok(())
    }

    fn ensure_alive(&mut self, operation: &str) -> Result<()> {
        if let Some(status) = self
            .child
            .try_wait()
            .with_context(|| format!("failed to poll ACP session runtime before {operation}"))?
        {
            bail!("ACP session runtime exited unexpectedly before {operation}: {status}");
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

        let request: ChildRuntimeRequest = serde_json::from_str(trimmed)
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
                    &ChildRuntimeEnvelope::Update {
                        update: ChildRuntimeUpdate::AgentMessage { text: text.clone() },
                    },
                )?;
                write_json_line(
                    &mut stdout,
                    &ChildRuntimeEnvelope::Result {
                        result: ChildRuntimeResponse {
                            text,
                            prompt_count: session.prompt_count,
                            stop_reason: ChildRuntimeStopReason::EndTurn,
                        },
                    },
                )?;
            }
            "cancel_prompt" | "shutdown" => break,
            other => bail!("unsupported ACP mock runtime command: {other}"),
        }
    }

    Ok(())
}

fn write_json_line<W, T>(writer: &mut W, value: &T) -> Result<()>
where
    W: Write,
    T: serde::Serialize,
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
    T: for<'de> serde::Deserialize<'de>,
{
    let mut line = String::new();
    if reader
        .read_line(&mut line)
        .context("failed to read ACP runtime response")?
        == 0
    {
        bail!("ACP session runtime closed stdout unexpectedly");
    }
    serde_json::from_str(line.trim()).context("failed to parse ACP runtime response")
}

fn spawn_runtime_reader(
    mut stdout: BufReader<ChildStdout>,
    envelope_tx: Sender<ChildRuntimeEnvelope>,
) -> JoinHandle<Result<BufReader<ChildStdout>>> {
    thread::spawn(move || {
        loop {
            let envelope: ChildRuntimeEnvelope = read_json_line(&mut stdout)?;
            let finished = matches!(envelope, ChildRuntimeEnvelope::Result { .. });
            if envelope_tx.send(envelope).is_err() {
                break;
            }
            if finished {
                break;
            }
        }
        Ok(stdout)
    })
}

fn join_runtime_reader(reader: JoinHandle<Result<BufReader<ChildStdout>>>) -> Result<BufReader<ChildStdout>> {
    match reader.join() {
        Ok(result) => result,
        Err(_) => bail!("ACP session runtime reader thread panicked"),
    }
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
    use crate::acp::journal::append_journal_event;
    use crate::acp::session::{
        CancelParams, ChildRuntimeStopReason, NewSessionParams, PromptBlock, PromptParams,
    };
    use crate::paths::build_paths;
    use serde_json::json;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let (_cancel_tx, cancel_rx) = mpsc::channel();

        let prompt = supervisor
            .prompt(
                PromptParams {
                    session_id: created.session_id.clone(),
                    prompt: vec![PromptBlock {
                        kind: "text".to_string(),
                        text: Some("hello".to_string()),
                    }],
                    message_id: Some("user-1".to_string()),
                },
                &cancel_rx,
                |_| Ok(()),
            )
            .expect("prompt");
        assert_eq!(prompt.prompt_count, 1);
        assert!(prompt.reply_text.contains("hello"));
        assert_eq!(prompt.stop_reason, ChildRuntimeStopReason::EndTurn);

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

    #[test]
    fn real_subprocess_mode_rejects_claude_until_adapter_exists() {
        let mut supervisor = SessionSupervisor::new(RuntimeMode::Subprocess);
        let error = supervisor
            .create(
                AgentKind::Claude,
                NewSessionParams {
                    cwd: PathBuf::from("/tmp"),
                    mcp_servers: vec![],
                },
        )
        .expect_err("claude should not create yet");
        assert!(error.to_string().contains("Claude runtime is not implemented"));
    }

    #[test]
    fn persisted_supervisor_continues_session_numbering() {
        let mut paths = build_paths().expect("paths");
        let test_root = std::env::temp_dir().join(format!(
            "opengateway-acp-supervisor-seed-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        paths.acp_dir = test_root.join("acp");
        paths.acp_sessions_dir = paths.acp_dir.join("sessions");
        paths.acp_logs_dir = paths.acp_dir.join("logs");
        paths.acp_tmp_dir = paths.acp_dir.join("tmp");
        fs::create_dir_all(&paths.acp_sessions_dir).expect("sessions dir");
        fs::create_dir_all(&paths.acp_logs_dir).expect("logs dir");
        append_journal_event(&paths, "session-000041", "session.created", json!({"cwd": "/tmp"}))
            .expect("journal event");

        let mut supervisor = SessionSupervisor::new_with_persisted_counter(
            RuntimeMode::InProcessMock,
            &paths,
        )
        .expect("seeded supervisor");
        let created = supervisor
            .create(
                AgentKind::Codex,
                NewSessionParams {
                    cwd: PathBuf::from("/tmp"),
                    mcp_servers: vec![],
                },
            )
            .expect("create");
        assert_eq!(created.session_id, "session-000042");

        let _ = fs::remove_dir_all(test_root);
    }
}
