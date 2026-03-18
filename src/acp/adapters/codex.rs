use crate::acp::session::{
    ChildRuntimeEnvelope, ChildRuntimeRequest, ChildRuntimeResponse, ChildRuntimeStopReason,
    ChildRuntimeUpdate, PromptBlock,
};
use anyhow::{bail, Context, Result};
use clap::Args;
use serde_json::Value;
use std::env;
#[cfg(windows)]
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::process::CommandExt as _;
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub const RUNTIME_NAME: &str = "codex exec";
pub const SCAFFOLD_NOTE: &str =
    "MVP session runtime launches `codex exec --json` for each ACP prompt.";
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;
const RUNTIME_POLL_INTERVAL: Duration = Duration::from_millis(25);
const CODEX_EXECUTABLE: &str = "codex";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexRuntimeReadiness {
    pub executable_path: Option<PathBuf>,
    pub version: Option<String>,
    pub supports_json: bool,
    pub supports_ephemeral: bool,
    pub supports_skip_git_repo_check: bool,
    pub supports_cwd_flag: bool,
    pub ready: bool,
    pub issue: Option<String>,
}

#[derive(Debug, Args)]
pub struct CodexRuntimeArgs {
    #[arg(long)]
    pub session_id: String,
    #[arg(long)]
    pub cwd: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub mcp_server_count: usize,
}

#[derive(Debug)]
struct CodexRuntimeSession {
    cwd: PathBuf,
    mcp_server_count: usize,
    prompt_count: u64,
}

struct CodexPromptExecution {
    child: Child,
    stream_rx: Receiver<CodexStreamEvent>,
    reader: JoinHandle<Result<()>>,
    agent_messages: Vec<String>,
    cancel_requested: bool,
}

struct CodexPromptResult {
    text: String,
    stop_reason: ChildRuntimeStopReason,
}

enum RuntimeInput {
    Request(ChildRuntimeRequest),
    Eof,
    Error(String),
}

enum CodexStreamEvent {
    Update(ChildRuntimeUpdate),
}

pub fn command_codex_runtime(args: CodexRuntimeArgs) -> Result<()> {
    let mut session = CodexRuntimeSession {
        cwd: args.cwd,
        mcp_server_count: args.mcp_server_count,
        prompt_count: 0,
    };
    let (input_tx, input_rx) = mpsc::channel();
    let _input_reader = spawn_runtime_input_reader(input_tx);
    let mut stdout = std::io::stdout().lock();
    let mut active_prompt: Option<CodexPromptExecution> = None;

    loop {
        if let Some(prompt) = active_prompt.as_mut() {
            drain_prompt_updates(prompt, &mut stdout)?;
            if let Some(result) = prompt.try_finish()? {
                session.prompt_count += 1;
                write_json_line(
                    &mut stdout,
                    &ChildRuntimeEnvelope::Result {
                        result: ChildRuntimeResponse {
                            text: result.text,
                            prompt_count: session.prompt_count,
                            stop_reason: result.stop_reason,
                        },
                    },
                )?;
                active_prompt = None;
                continue;
            }
        }

        match input_rx.recv_timeout(RUNTIME_POLL_INTERVAL) {
            Ok(RuntimeInput::Request(request)) => match request.kind.as_str() {
                "prompt" => {
                    if active_prompt.is_some() {
                        bail!("ACP Codex runtime received prompt while another prompt was active");
                    }
                    active_prompt = Some(CodexPromptExecution::spawn(
                        &session.cwd,
                        &request.prompt,
                        session.mcp_server_count,
                    )?);
                }
                "cancel_prompt" => {
                    if let Some(prompt) = active_prompt.as_mut() {
                        prompt.cancel()?;
                        write_json_line(
                            &mut stdout,
                            &ChildRuntimeEnvelope::Update {
                                update: ChildRuntimeUpdate::TurnCancelled,
                            },
                        )?;
                    }
                }
                "shutdown" => {
                    if let Some(mut prompt) = active_prompt.take() {
                        prompt.cancel()?;
                        let _ = prompt.finish_blocking();
                    }
                    break;
                }
                other => bail!("unsupported ACP Codex runtime command: {other}"),
            },
            Ok(RuntimeInput::Eof) => {
                if let Some(mut prompt) = active_prompt.take() {
                    prompt.cancel()?;
                    let _ = prompt.finish_blocking();
                }
                break;
            }
            Ok(RuntimeInput::Error(error)) => bail!("{error}"),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    Ok(())
}

pub fn inspect_runtime() -> CodexRuntimeReadiness {
    let Some(executable_path) = resolve_executable_path(CODEX_EXECUTABLE) else {
        return CodexRuntimeReadiness {
            executable_path: None,
            version: None,
            supports_json: false,
            supports_ephemeral: false,
            supports_skip_git_repo_check: false,
            supports_cwd_flag: false,
            ready: false,
            issue: Some("`codex` was not found on PATH".to_string()),
        };
    };

    let version_output = match Command::new(&executable_path).arg("--version").output() {
        Ok(output) => output,
        Err(error) => {
            return CodexRuntimeReadiness {
                executable_path: Some(executable_path),
                version: None,
                supports_json: false,
                supports_ephemeral: false,
                supports_skip_git_repo_check: false,
                supports_cwd_flag: false,
                ready: false,
                issue: Some(format!("failed to run `codex --version`: {error}")),
            };
        }
    };
    if !version_output.status.success() {
        return CodexRuntimeReadiness {
            executable_path: Some(executable_path),
            version: None,
            supports_json: false,
            supports_ephemeral: false,
            supports_skip_git_repo_check: false,
            supports_cwd_flag: false,
            ready: false,
            issue: Some(format!(
                "`codex --version` exited with {}",
                version_output.status
            )),
        };
    }
    let version = command_output_text(&version_output.stdout, &version_output.stderr);

    let help_output = match Command::new(&executable_path).args(["exec", "--help"]).output() {
        Ok(output) => output,
        Err(error) => {
            return CodexRuntimeReadiness {
                executable_path: Some(executable_path),
                version,
                supports_json: false,
                supports_ephemeral: false,
                supports_skip_git_repo_check: false,
                supports_cwd_flag: false,
                ready: false,
                issue: Some(format!("failed to run `codex exec --help`: {error}")),
            };
        }
    };
    if !help_output.status.success() {
        return CodexRuntimeReadiness {
            executable_path: Some(executable_path),
            version,
            supports_json: false,
            supports_ephemeral: false,
            supports_skip_git_repo_check: false,
            supports_cwd_flag: false,
            ready: false,
            issue: Some(format!(
                "`codex exec --help` exited with {}",
                help_output.status
            )),
        };
    }

    let help_text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&help_output.stdout),
        String::from_utf8_lossy(&help_output.stderr)
    );
    let capabilities = parse_exec_help_capabilities(&help_text);
    let mut missing_flags = Vec::new();
    if !capabilities.supports_json {
        missing_flags.push("--json");
    }
    if !capabilities.supports_ephemeral {
        missing_flags.push("--ephemeral");
    }
    if !capabilities.supports_skip_git_repo_check {
        missing_flags.push("--skip-git-repo-check");
    }
    if !capabilities.supports_cwd_flag {
        missing_flags.push("-C/--cd");
    }

    CodexRuntimeReadiness {
        executable_path: Some(executable_path),
        version,
        supports_json: capabilities.supports_json,
        supports_ephemeral: capabilities.supports_ephemeral,
        supports_skip_git_repo_check: capabilities.supports_skip_git_repo_check,
        supports_cwd_flag: capabilities.supports_cwd_flag,
        ready: missing_flags.is_empty(),
        issue: (!missing_flags.is_empty()).then(|| {
            format!(
                "`codex exec --help` is missing required flags: {}",
                missing_flags.join(", ")
            )
        }),
    }
}

pub fn doctor_guidance(readiness: &CodexRuntimeReadiness) -> Vec<String> {
    if readiness.executable_path.is_none() {
        return vec![
            "Install the Codex CLI and make sure `codex` is on PATH.".to_string(),
            "Verify the install with `codex --version` before starting an ACP harness."
                .to_string(),
        ];
    }

    if readiness.version.is_none() {
        return vec![
            "Run `codex --version` manually. If it fails, repair or reinstall the Codex CLI."
                .to_string(),
            "After that, rerun `opengateway acp doctor` to confirm the runtime probe is clean."
                .to_string(),
        ];
    }

    if !readiness.ready {
        return vec![
            "Upgrade the Codex CLI so `codex exec --help` exposes `--json`, `--ephemeral`, `--skip-git-repo-check`, and `-C/--cd`."
                .to_string(),
            "Validate the flags with `codex exec --help` before starting an ACP harness."
                .to_string(),
        ];
    }

    vec![
        "Start the harness with `opengateway acp serve --agent codex`.".to_string(),
        "If prompts still fail, run `codex exec --json --ephemeral --skip-git-repo-check -C <dir> -` manually to inspect the local runtime."
            .to_string(),
        "This preflight does not verify Codex account auth; it only checks local runtime readiness."
            .to_string(),
    ]
}

pub fn spawn_guidance(readiness: &CodexRuntimeReadiness) -> String {
    doctor_guidance(readiness).join(" ")
}

pub fn build_codex_prompt(prompt: &[PromptBlock], mcp_server_count: usize) -> Result<String> {
    let text_blocks = prompt
        .iter()
        .filter(|block| block.kind == "text")
        .filter_map(|block| block.text.as_deref())
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    if text_blocks.is_empty() {
        bail!("ACP Codex prompt requires at least one text block");
    }

    let non_text_blocks = prompt
        .iter()
        .filter(|block| block.kind != "text")
        .map(|block| block.kind.as_str())
        .collect::<Vec<_>>();

    let mut rendered = text_blocks.join("\n\n");
    if !non_text_blocks.is_empty() || mcp_server_count > 0 {
        rendered.push_str("\n\n[ACP session context]\n");
        if !non_text_blocks.is_empty() {
            rendered.push_str(&format!(
                "- non-text prompt blocks omitted in MVP: {}\n",
                non_text_blocks.join(", ")
            ));
        }
        if mcp_server_count > 0 {
            rendered.push_str(&format!("- attached MCP server stubs: {mcp_server_count}\n"));
        }
    }

    Ok(rendered)
}

fn spawn_runtime_input_reader(input_tx: Sender<RuntimeInput>) -> JoinHandle<()> {
    thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut reader = BufReader::new(stdin.lock());
        let mut frame = String::new();

        loop {
            frame.clear();
            match reader.read_line(&mut frame) {
                Ok(0) => {
                    let _ = input_tx.send(RuntimeInput::Eof);
                    break;
                }
                Ok(_) => {
                    let trimmed = frame.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<ChildRuntimeRequest>(trimmed) {
                        Ok(request) => {
                            if input_tx.send(RuntimeInput::Request(request)).is_err() {
                                break;
                            }
                        }
                        Err(error) => {
                            let _ = input_tx.send(RuntimeInput::Error(format!(
                                "invalid ACP Codex runtime frame: {trimmed}; {error}"
                            )));
                            break;
                        }
                    }
                }
                Err(error) => {
                    let _ = input_tx.send(RuntimeInput::Error(format!(
                        "failed to read ACP Codex runtime stdin: {error}"
                    )));
                    break;
                }
            }
        }
    })
}

fn command_output_text(stdout: &[u8], stderr: &[u8]) -> Option<String> {
    let stdout = String::from_utf8_lossy(stdout);
    let trimmed_stdout = stdout.trim();
    if !trimmed_stdout.is_empty() {
        return Some(trimmed_stdout.to_string());
    }

    let stderr = String::from_utf8_lossy(stderr);
    let trimmed_stderr = stderr.trim();
    if !trimmed_stderr.is_empty() {
        return Some(trimmed_stderr.to_string());
    }

    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExecHelpCapabilities {
    supports_json: bool,
    supports_ephemeral: bool,
    supports_skip_git_repo_check: bool,
    supports_cwd_flag: bool,
}

fn parse_exec_help_capabilities(help: &str) -> ExecHelpCapabilities {
    ExecHelpCapabilities {
        supports_json: help.contains("--json"),
        supports_ephemeral: help.contains("--ephemeral"),
        supports_skip_git_repo_check: help.contains("--skip-git-repo-check"),
        supports_cwd_flag: help.contains("-C, --cd") || help.contains("--cd <DIR>"),
    }
}

fn resolve_executable_path(command: &str) -> Option<PathBuf> {
    let command_path = Path::new(command);
    if command_path.components().count() > 1 {
        return is_existing_file(command_path).then(|| command_path.to_path_buf());
    }

    let path_var = env::var_os("PATH")?;
    for directory in env::split_paths(&path_var) {
        #[cfg(windows)]
        {
            for candidate in windows_command_candidates(&directory, command) {
                if is_existing_file(&candidate) {
                    return Some(candidate);
                }
            }
        }
        #[cfg(not(windows))]
        {
            let candidate = directory.join(command);
            if is_existing_file(&candidate) {
                return Some(candidate);
            }
        }
    }

    None
}

fn is_existing_file(path: &Path) -> bool {
    path.metadata().map(|metadata| metadata.is_file()).unwrap_or(false)
}

#[cfg(windows)]
fn windows_command_candidates(directory: &Path, command: &str) -> Vec<PathBuf> {
    let command_path = Path::new(command);
    if command_path.extension().is_some() {
        return vec![directory.join(command)];
    }

    let path_ext = env::var_os("PATHEXT")
        .unwrap_or_else(|| OsString::from(".COM;.EXE;.BAT;.CMD"));
    path_ext
        .to_string_lossy()
        .split(';')
        .filter(|ext| !ext.is_empty())
        .map(|ext| {
            let trimmed = ext.trim();
            let suffix = if trimmed.starts_with('.') {
                trimmed.to_string()
            } else {
                format!(".{trimmed}")
            };
            directory.join(format!("{command}{suffix}"))
        })
        .collect()
}

impl CodexPromptExecution {
    fn spawn(cwd: &Path, prompt: &[PromptBlock], mcp_server_count: usize) -> Result<Self> {
        let prompt_text = build_codex_prompt(prompt, mcp_server_count)?;
        let mut command = Command::new("codex");
        command
            .arg("exec")
            .arg("--json")
            .arg("--ephemeral")
            .arg("--skip-git-repo-check")
            .arg("-C")
            .arg(cwd)
            .arg("-");
        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(unix)]
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        let mut child = command.spawn().context("failed to spawn `codex exec`")?;
        let mut stdin = child
            .stdin
            .take()
            .context("spawned `codex exec` has no stdin")?;
        stdin
            .write_all(prompt_text.as_bytes())
            .context("failed to write ACP prompt to `codex exec`")?;
        drop(stdin);

        let stdout = child
            .stdout
            .take()
            .context("spawned `codex exec` has no stdout")?;
        let (stream_tx, stream_rx) = mpsc::channel();
        let reader = spawn_codex_stream_reader(stdout, stream_tx);

        Ok(Self {
            child,
            stream_rx,
            reader,
            agent_messages: Vec::new(),
            cancel_requested: false,
        })
    }

    fn cancel(&mut self) -> Result<()> {
        if self.cancel_requested {
            return Ok(());
        }
        self.cancel_requested = true;
        #[cfg(unix)]
        {
            let pid = i32::try_from(self.child.id()).unwrap_or(i32::MAX);
            let rc = unsafe { libc::killpg(pid, libc::SIGKILL) };
            if rc == 0 {
                return Ok(());
            }
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ESRCH) {
                return Ok(());
            }
        }
        match self.child.kill() {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => Ok(()),
            Err(error) => Err(error).context("failed to kill in-flight `codex exec`"),
        }
    }

    fn try_drain_updates<W>(&mut self, output: &mut W) -> Result<()>
    where
        W: Write,
    {
        while let Ok(event) = self.stream_rx.try_recv() {
            let CodexStreamEvent::Update(update) = event;
            if let ChildRuntimeUpdate::AgentMessage { text } = &update {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    self.agent_messages.push(trimmed.to_string());
                }
            }
            write_json_line(output, &ChildRuntimeEnvelope::Update { update })?;
        }
        Ok(())
    }

    fn try_finish(&mut self) -> Result<Option<CodexPromptResult>> {
        let Some(status) = self
            .child
            .try_wait()
            .context("failed to poll in-flight `codex exec`")?
        else {
            return Ok(None);
        };

        join_codex_stream_reader(std::mem::replace(
            &mut self.reader,
            thread::spawn(|| Ok(())),
        ))?;

        if self.cancel_requested {
            return Ok(Some(CodexPromptResult {
                text: String::new(),
                stop_reason: ChildRuntimeStopReason::Cancelled,
            }));
        }
        if !status.success() {
            bail!("`codex exec` failed with status {status}");
        }
        if self.agent_messages.is_empty() {
            bail!("`codex exec --json` completed without an agent_message item");
        }

        Ok(Some(CodexPromptResult {
            text: self.agent_messages.join("\n\n"),
            stop_reason: ChildRuntimeStopReason::EndTurn,
        }))
    }

    fn finish_blocking(mut self) -> Result<()> {
        let _ = self.child.wait();
        join_codex_stream_reader(self.reader)
    }
}

fn spawn_codex_stream_reader(
    stdout: ChildStdout,
    stream_tx: Sender<CodexStreamEvent>,
) -> JoinHandle<Result<()>> {
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut frame = String::new();

        loop {
            frame.clear();
            if reader
                .read_line(&mut frame)
                .context("failed to read `codex exec --json` output")?
                == 0
            {
                break;
            }

            let line = frame.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(update) = parse_codex_update(line)? {
                if stream_tx.send(CodexStreamEvent::Update(update)).is_err() {
                    break;
                }
            }
        }

        Ok(())
    })
}

fn join_codex_stream_reader(reader: JoinHandle<Result<()>>) -> Result<()> {
    match reader.join() {
        Ok(result) => result,
        Err(_) => bail!("Codex stream reader thread panicked"),
    }
}

fn drain_prompt_updates<W>(prompt: &mut CodexPromptExecution, output: &mut W) -> Result<()>
where
    W: Write,
{
    prompt.try_drain_updates(output)
}

fn parse_codex_update(line: &str) -> Result<Option<ChildRuntimeUpdate>> {
    let event: Value = serde_json::from_str(line)
        .with_context(|| format!("invalid `codex exec --json` event: {line}"))?;

    match event.get("type").and_then(Value::as_str) {
        Some("turn.started") => Ok(Some(ChildRuntimeUpdate::TurnStarted)),
        Some("turn.completed") => Ok(Some(ChildRuntimeUpdate::TurnCompleted {
            input_tokens: event
                .get("usage")
                .and_then(|usage| usage.get("input_tokens"))
                .and_then(Value::as_u64),
            output_tokens: event
                .get("usage")
                .and_then(|usage| usage.get("output_tokens"))
                .and_then(Value::as_u64),
        })),
        Some("item.started") => Ok(parse_item_update(event.get("item"), true)),
        Some("item.completed") => Ok(parse_item_update(event.get("item"), false)),
        _ => Ok(None),
    }
}

fn parse_item_update(item: Option<&Value>, started: bool) -> Option<ChildRuntimeUpdate> {
    let item = item?;
    match item.get("type").and_then(Value::as_str) {
        Some("agent_message") if !started => item
            .get("text")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(|text| ChildRuntimeUpdate::AgentMessage {
                text: text.to_string(),
            }),
        Some("command_execution") => {
            let command = item.get("command").and_then(Value::as_str)?.trim();
            if command.is_empty() {
                return None;
            }
            if started {
                Some(ChildRuntimeUpdate::CommandStarted {
                    command: command.to_string(),
                })
            } else {
                Some(ChildRuntimeUpdate::CommandCompleted {
                    command: command.to_string(),
                    output: item
                        .get("aggregated_output")
                        .and_then(Value::as_str)
                        .map(truncate_output)
                        .filter(|output| !output.is_empty()),
                    exit_code: item
                        .get("exit_code")
                        .and_then(Value::as_i64)
                        .and_then(|code| i32::try_from(code).ok()),
                })
            }
        }
        _ => None,
    }
}

fn truncate_output(output: &str) -> String {
    const MAX_CHARS: usize = 800;
    let trimmed = output.trim();
    let truncated = trimmed.chars().take(MAX_CHARS).collect::<String>();
    if trimmed.chars().count() > MAX_CHARS {
        format!("{truncated}\n…")
    } else {
        truncated
    }
}

fn write_json_line<W, T>(writer: &mut W, value: &T) -> Result<()>
where
    W: Write,
    T: serde::Serialize,
{
    serde_json::to_writer(&mut *writer, value)
        .context("failed to write ACP Codex runtime message")?;
    writer
        .write_all(b"\n")
        .context("failed to terminate ACP Codex runtime message")?;
    writer
        .flush()
        .context("failed to flush ACP Codex runtime message")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_codex_prompt, doctor_guidance, parse_codex_update, parse_exec_help_capabilities,
        truncate_output, CodexRuntimeReadiness,
    };
    use crate::acp::session::ChildRuntimeUpdate;
    use crate::acp::session::PromptBlock;
    use std::path::PathBuf;

    #[test]
    fn codex_prompt_requires_text_blocks() {
        let error = build_codex_prompt(
            &[PromptBlock {
                kind: "resource_link".to_string(),
                text: None,
            }],
            0,
        )
        .expect_err("prompt should require text");
        assert!(error.to_string().contains("requires at least one text block"));
    }

    #[test]
    fn codex_prompt_includes_context_summary() {
        let prompt = build_codex_prompt(
            &[
                PromptBlock {
                    kind: "text".to_string(),
                    text: Some("hello".to_string()),
                },
                PromptBlock {
                    kind: "resource_link".to_string(),
                    text: None,
                },
            ],
            2,
        )
        .expect("prompt");
        assert!(prompt.contains("hello"));
        assert!(prompt.contains("resource_link"));
        assert!(prompt.contains("attached MCP server stubs: 2"));
    }

    #[test]
    fn parse_codex_update_handles_agent_message_and_commands() {
        let turn_started = parse_codex_update(r#"{"type":"turn.started"}"#)
            .expect("parse turn")
            .expect("turn update");
        assert_eq!(turn_started, ChildRuntimeUpdate::TurnStarted);

        let command_started = parse_codex_update(
            r#"{"type":"item.started","item":{"id":"item_1","type":"command_execution","command":"echo hi","status":"in_progress"}}"#,
        )
        .expect("parse command started")
        .expect("command started update");
        assert_eq!(
            command_started,
            ChildRuntimeUpdate::CommandStarted {
                command: "echo hi".to_string(),
            }
        );

        let command_completed = parse_codex_update(
            r#"{"type":"item.completed","item":{"id":"item_1","type":"command_execution","command":"echo hi","aggregated_output":"hi\n","exit_code":0,"status":"completed"}}"#,
        )
        .expect("parse command completed")
        .expect("command completed update");
        assert_eq!(
            command_completed,
            ChildRuntimeUpdate::CommandCompleted {
                command: "echo hi".to_string(),
                output: Some("hi".to_string()),
                exit_code: Some(0),
            }
        );

        let agent_message = parse_codex_update(
            r#"{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"hello"}}"#,
        )
        .expect("parse message")
        .expect("message update");
        assert_eq!(
            agent_message,
            ChildRuntimeUpdate::AgentMessage {
                text: "hello".to_string(),
            }
        );
    }

    #[test]
    fn truncate_output_limits_large_command_output() {
        let large = "x".repeat(900);
        let preview = truncate_output(&large);
        assert!(preview.ends_with('…'));
        assert!(preview.len() < large.len());
    }

    #[test]
    fn parse_exec_help_capabilities_detects_required_flags() {
        let help = "\
Usage: codex exec [OPTIONS]
  -C, --cd <DIR>
      --skip-git-repo-check
      --ephemeral
      --json
";
        let capabilities = parse_exec_help_capabilities(help);
        assert!(capabilities.supports_json);
        assert!(capabilities.supports_ephemeral);
        assert!(capabilities.supports_skip_git_repo_check);
        assert!(capabilities.supports_cwd_flag);
    }

    #[test]
    fn doctor_guidance_for_missing_codex_is_actionable() {
        let readiness = CodexRuntimeReadiness {
            executable_path: None,
            version: None,
            supports_json: false,
            supports_ephemeral: false,
            supports_skip_git_repo_check: false,
            supports_cwd_flag: false,
            ready: false,
            issue: Some("`codex` was not found on PATH".to_string()),
        };

        let guidance = doctor_guidance(&readiness);
        assert!(guidance[0].contains("Install the Codex CLI"));
        assert!(guidance[1].contains("codex --version"));
    }

    #[test]
    fn doctor_guidance_for_ready_codex_recommends_next_step() {
        let readiness = CodexRuntimeReadiness {
            executable_path: Some(PathBuf::from("/tmp/codex")),
            version: Some("codex-cli 0.114.0".to_string()),
            supports_json: true,
            supports_ephemeral: true,
            supports_skip_git_repo_check: true,
            supports_cwd_flag: true,
            ready: true,
            issue: None,
        };

        let guidance = doctor_guidance(&readiness);
        assert!(guidance[0].contains("opengateway acp serve --agent codex"));
        assert!(guidance[2].contains("does not verify Codex account auth"));
    }
}
