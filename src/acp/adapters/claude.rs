use crate::acp::session::{
    ChildRuntimeEnvelope, ChildRuntimeRequest, ChildRuntimeResponse, ChildRuntimeStopReason,
    ChildRuntimeUpdate, PromptBlock,
};
use anyhow::{bail, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
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

pub const RUNTIME_NAME: &str = "claude -p";
pub const SCAFFOLD_NOTE: &str =
    "MVP session runtime launches `claude -p --output-format stream-json` per ACP prompt.";
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;
const RUNTIME_POLL_INTERVAL: Duration = Duration::from_millis(25);
const CLAUDE_EXECUTABLE: &str = "claude";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClaudeRuntimeReadiness {
    pub executable_path: Option<PathBuf>,
    pub version: Option<String>,
    pub supports_print: bool,
    pub supports_stream_json: bool,
    pub supports_partial_messages: bool,
    pub supports_permission_mode: bool,
    pub supports_model_flag: bool,
    pub supports_no_session_persistence: bool,
    pub supports_verbose: bool,
    pub auth_ready: bool,
    pub auth_method: Option<String>,
    pub api_provider: Option<String>,
    pub subscription_type: Option<String>,
    pub ready: bool,
    pub issue: Option<String>,
}

#[derive(Debug, Args)]
pub struct ClaudeRuntimeArgs {
    #[arg(long)]
    pub session_id: String,
    #[arg(long)]
    pub cwd: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub mcp_server_count: usize,
    #[arg(long)]
    pub model: Option<String>,
}

#[derive(Debug)]
struct ClaudeRuntimeSession {
    cwd: PathBuf,
    mcp_server_count: usize,
    model: Option<String>,
    prompt_count: u64,
}

struct ClaudePromptExecution {
    child: Child,
    stream_rx: Receiver<ClaudeStreamEvent>,
    reader: JoinHandle<Result<()>>,
    agent_message_chunks: Vec<String>,
    final_text: Option<String>,
    final_input_tokens: Option<u64>,
    final_output_tokens: Option<u64>,
    final_issue: Option<String>,
    cancel_requested: bool,
}

struct ClaudePromptResult {
    text: String,
    stop_reason: ChildRuntimeStopReason,
}

enum RuntimeInput {
    Request(ChildRuntimeRequest),
    Eof,
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ClaudeStreamEvent {
    Update(ChildRuntimeUpdate),
    Final(ClaudeFinalEvent),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClaudeFinalEvent {
    text: Option<String>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    issue: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrintHelpCapabilities {
    supports_print: bool,
    supports_stream_json: bool,
    supports_partial_messages: bool,
    supports_permission_mode: bool,
    supports_model_flag: bool,
    supports_no_session_persistence: bool,
    supports_verbose: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudeAuthStatus {
    #[serde(default)]
    logged_in: bool,
    #[serde(default)]
    auth_method: Option<String>,
    #[serde(default)]
    api_provider: Option<String>,
    #[serde(default)]
    subscription_type: Option<String>,
}

pub fn command_claude_runtime(args: ClaudeRuntimeArgs) -> Result<()> {
    let mut session = ClaudeRuntimeSession {
        cwd: args.cwd,
        mcp_server_count: args.mcp_server_count,
        model: args.model,
        prompt_count: 0,
    };
    let (input_tx, input_rx) = mpsc::channel();
    let _input_reader = spawn_runtime_input_reader(input_tx);
    let mut stdout = std::io::stdout().lock();
    let mut active_prompt: Option<ClaudePromptExecution> = None;

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
                        bail!("ACP Claude runtime received prompt while another prompt was active");
                    }
                    active_prompt = Some(ClaudePromptExecution::spawn(
                        &session.cwd,
                        &request.prompt,
                        session.mcp_server_count,
                        session.model.as_deref(),
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
                other => bail!("unsupported ACP Claude runtime command: {other}"),
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

pub fn inspect_runtime() -> ClaudeRuntimeReadiness {
    let Some(executable_path) = resolve_executable_path(CLAUDE_EXECUTABLE) else {
        return ClaudeRuntimeReadiness {
            executable_path: None,
            version: None,
            supports_print: false,
            supports_stream_json: false,
            supports_partial_messages: false,
            supports_permission_mode: false,
            supports_model_flag: false,
            supports_no_session_persistence: false,
            supports_verbose: false,
            auth_ready: false,
            auth_method: None,
            api_provider: None,
            subscription_type: None,
            ready: false,
            issue: Some("`claude` was not found on PATH".to_string()),
        };
    };

    let version_output = match Command::new(&executable_path).arg("--version").output() {
        Ok(output) => output,
        Err(error) => {
            return ClaudeRuntimeReadiness {
                executable_path: Some(executable_path),
                version: None,
                supports_print: false,
                supports_stream_json: false,
                supports_partial_messages: false,
                supports_permission_mode: false,
                supports_model_flag: false,
                supports_no_session_persistence: false,
                supports_verbose: false,
                auth_ready: false,
                auth_method: None,
                api_provider: None,
                subscription_type: None,
                ready: false,
                issue: Some(format!("failed to run `claude --version`: {error}")),
            };
        }
    };
    if !version_output.status.success() {
        return ClaudeRuntimeReadiness {
            executable_path: Some(executable_path),
            version: None,
            supports_print: false,
            supports_stream_json: false,
            supports_partial_messages: false,
            supports_permission_mode: false,
            supports_model_flag: false,
            supports_no_session_persistence: false,
            supports_verbose: false,
            auth_ready: false,
            auth_method: None,
            api_provider: None,
            subscription_type: None,
            ready: false,
            issue: Some(format!(
                "`claude --version` exited with {}",
                version_output.status
            )),
        };
    }
    let version = command_output_text(&version_output.stdout, &version_output.stderr);

    let help_output = match Command::new(&executable_path).args(["-p", "--help"]).output() {
        Ok(output) => output,
        Err(error) => {
            return ClaudeRuntimeReadiness {
                executable_path: Some(executable_path),
                version,
                supports_print: false,
                supports_stream_json: false,
                supports_partial_messages: false,
                supports_permission_mode: false,
                supports_model_flag: false,
                supports_no_session_persistence: false,
                supports_verbose: false,
                auth_ready: false,
                auth_method: None,
                api_provider: None,
                subscription_type: None,
                ready: false,
                issue: Some(format!("failed to run `claude -p --help`: {error}")),
            };
        }
    };
    if !help_output.status.success() {
        return ClaudeRuntimeReadiness {
            executable_path: Some(executable_path),
            version,
            supports_print: false,
            supports_stream_json: false,
            supports_partial_messages: false,
            supports_permission_mode: false,
            supports_model_flag: false,
            supports_no_session_persistence: false,
            supports_verbose: false,
            auth_ready: false,
            auth_method: None,
            api_provider: None,
            subscription_type: None,
            ready: false,
            issue: Some(format!(
                "`claude -p --help` exited with {}",
                help_output.status
            )),
        };
    }

    let help_text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&help_output.stdout),
        String::from_utf8_lossy(&help_output.stderr)
    );
    let capabilities = parse_print_help_capabilities(&help_text);
    let mut issues = Vec::new();
    let mut missing_flags = Vec::new();
    if !capabilities.supports_print {
        missing_flags.push("-p/--print");
    }
    if !capabilities.supports_stream_json {
        missing_flags.push("--output-format");
    }
    if !capabilities.supports_partial_messages {
        missing_flags.push("--include-partial-messages");
    }
    if !capabilities.supports_permission_mode {
        missing_flags.push("--permission-mode");
    }
    if !capabilities.supports_model_flag {
        missing_flags.push("--model");
    }
    if !capabilities.supports_no_session_persistence {
        missing_flags.push("--no-session-persistence");
    }
    if !capabilities.supports_verbose {
        missing_flags.push("--verbose");
    }
    if !missing_flags.is_empty() {
        issues.push(format!(
            "`claude -p --help` is missing required flags: {}",
            missing_flags.join(", ")
        ));
    }

    let api_key_ready = env::var("ANTHROPIC_API_KEY")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    let (auth_ready, auth_method, api_provider, subscription_type, auth_issue) =
        inspect_auth_status(&executable_path, api_key_ready);
    if let Some(issue) = auth_issue {
        issues.push(issue);
    }

    ClaudeRuntimeReadiness {
        executable_path: Some(executable_path),
        version,
        supports_print: capabilities.supports_print,
        supports_stream_json: capabilities.supports_stream_json,
        supports_partial_messages: capabilities.supports_partial_messages,
        supports_permission_mode: capabilities.supports_permission_mode,
        supports_model_flag: capabilities.supports_model_flag,
        supports_no_session_persistence: capabilities.supports_no_session_persistence,
        supports_verbose: capabilities.supports_verbose,
        auth_ready,
        auth_method,
        api_provider,
        subscription_type,
        ready: issues.is_empty(),
        issue: (!issues.is_empty()).then(|| issues.join("; ")),
    }
}

pub fn doctor_guidance(readiness: &ClaudeRuntimeReadiness) -> Vec<String> {
    if readiness.executable_path.is_none() {
        return vec![
            "Install Claude Code and make sure `claude` is on PATH.".to_string(),
            "Verify the install with `claude --version` before starting an ACP harness."
                .to_string(),
        ];
    }

    if readiness.version.is_none() {
        return vec![
            "Run `claude --version` manually. If it fails, repair or reinstall Claude Code."
                .to_string(),
            "After that, rerun `opengateway acp doctor` to confirm the runtime probe is clean."
                .to_string(),
        ];
    }

    if !readiness.supports_print
        || !readiness.supports_stream_json
        || !readiness.supports_partial_messages
        || !readiness.supports_permission_mode
        || !readiness.supports_model_flag
        || !readiness.supports_no_session_persistence
        || !readiness.supports_verbose
    {
        return vec![
            "Upgrade Claude Code so `claude -p --help` exposes `--output-format`, `--include-partial-messages`, `--permission-mode`, `--model`, `--no-session-persistence`, and `--verbose`."
                .to_string(),
            "Validate the flags with `claude -p --help` before starting an ACP harness."
                .to_string(),
        ];
    }

    if !readiness.auth_ready {
        return vec![
            "Authenticate Claude Code with `claude auth login`, or set `ANTHROPIC_API_KEY` for API-key-backed use."
                .to_string(),
            "Confirm the local auth state with `claude auth status` before starting an ACP harness."
                .to_string(),
        ];
    }

    vec![
        "Start the harness with `opengateway acp serve --agent claude`.".to_string(),
        "If prompts still fail, run `claude -p 'hello' --output-format stream-json --include-partial-messages --verbose --permission-mode default --no-session-persistence` manually to inspect the local runtime."
            .to_string(),
    ]
}

pub fn spawn_guidance(readiness: &ClaudeRuntimeReadiness) -> String {
    doctor_guidance(readiness).join(" ")
}

pub fn build_claude_prompt(prompt: &[PromptBlock], mcp_server_count: usize) -> Result<String> {
    let text_blocks = prompt
        .iter()
        .filter(|block| block.kind == "text")
        .filter_map(|block| block.text.as_deref())
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>();
    if text_blocks.is_empty() {
        bail!("ACP Claude prompt requires at least one text block");
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
                                "invalid ACP Claude runtime frame: {trimmed}; {error}"
                            )));
                            break;
                        }
                    }
                }
                Err(error) => {
                    let _ = input_tx.send(RuntimeInput::Error(format!(
                        "failed to read ACP Claude runtime stdin: {error}"
                    )));
                    break;
                }
            }
        }
    })
}

fn inspect_auth_status(
    executable_path: &Path,
    api_key_ready: bool,
) -> (
    bool,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    if api_key_ready {
        return (
            true,
            Some("api-key".to_string()),
            Some("anthropic".to_string()),
            None,
            None,
        );
    }

    let status_output = match Command::new(executable_path).args(["auth", "status"]).output() {
        Ok(output) => output,
        Err(error) => {
            return (
                false,
                None,
                None,
                None,
                Some(format!("failed to run `claude auth status`: {error}")),
            );
        }
    };
    if !status_output.status.success() {
        return (
            false,
            None,
            None,
            None,
            Some(format!(
                "`claude auth status` exited with {}",
                status_output.status
            )),
        );
    }

    let text = command_output_text(&status_output.stdout, &status_output.stderr).unwrap_or_default();
    if text.trim().is_empty() {
        return (
            false,
            None,
            None,
            None,
            Some("`claude auth status` returned no auth data".to_string()),
        );
    }

    let status: ClaudeAuthStatus = match serde_json::from_str(&text) {
        Ok(status) => status,
        Err(error) => {
            return (
                false,
                None,
                None,
                None,
                Some(format!("failed to parse `claude auth status`: {error}")),
            );
        }
    };

    if status.logged_in {
        (
            true,
            status.auth_method,
            status.api_provider,
            status.subscription_type,
            None,
        )
    } else {
        (
            false,
            status.auth_method,
            status.api_provider,
            status.subscription_type,
            Some("Claude auth is not ready; run `claude auth login` or set `ANTHROPIC_API_KEY`".to_string()),
        )
    }
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

fn parse_print_help_capabilities(help: &str) -> PrintHelpCapabilities {
    PrintHelpCapabilities {
        supports_print: help.contains("-p, --print") || help.contains("--print"),
        supports_stream_json: help.contains("--output-format"),
        supports_partial_messages: help.contains("--include-partial-messages"),
        supports_permission_mode: help.contains("--permission-mode"),
        supports_model_flag: help.contains("--model"),
        supports_no_session_persistence: help.contains("--no-session-persistence"),
        supports_verbose: help.contains("--verbose"),
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

impl ClaudePromptExecution {
    fn spawn(
        cwd: &Path,
        prompt: &[PromptBlock],
        mcp_server_count: usize,
        model: Option<&str>,
    ) -> Result<Self> {
        let prompt_text = build_claude_prompt(prompt, mcp_server_count)?;
        let mut command = Command::new(CLAUDE_EXECUTABLE);
        command
            .arg("-p")
            .arg(prompt_text)
            .arg("--output-format")
            .arg("stream-json")
            .arg("--include-partial-messages")
            .arg("--verbose")
            .arg("--permission-mode")
            .arg("default")
            .arg("--no-session-persistence")
            .current_dir(cwd);
        if let Some(model) = model {
            if !model.trim().is_empty() {
                command.arg("--model").arg(model);
            }
        }
        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);
        command
            .stdin(Stdio::null())
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

        let mut child = command.spawn().context("failed to spawn `claude -p`")?;
        let stdout = child
            .stdout
            .take()
            .context("spawned `claude -p` has no stdout")?;
        let (stream_tx, stream_rx) = mpsc::channel();
        let reader = spawn_claude_stream_reader(stdout, stream_tx);

        Ok(Self {
            child,
            stream_rx,
            reader,
            agent_message_chunks: Vec::new(),
            final_text: None,
            final_input_tokens: None,
            final_output_tokens: None,
            final_issue: None,
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
            Err(error) => Err(error).context("failed to kill in-flight `claude -p`"),
        }
    }

    fn try_drain_updates<W>(&mut self, output: &mut W) -> Result<()>
    where
        W: Write,
    {
        while let Ok(event) = self.stream_rx.try_recv() {
            match event {
                ClaudeStreamEvent::Update(update) => {
                    if let ChildRuntimeUpdate::AgentMessage { text } = &update {
                        if !text.is_empty() {
                            self.agent_message_chunks.push(text.clone());
                        }
                    }
                    if let ChildRuntimeUpdate::TurnCompleted {
                        input_tokens,
                        output_tokens,
                    } = &update
                    {
                        self.final_input_tokens = *input_tokens;
                        self.final_output_tokens = *output_tokens;
                    }
                    write_json_line(output, &ChildRuntimeEnvelope::Update { update })?;
                }
                ClaudeStreamEvent::Final(final_event) => {
                    if let Some(text) = final_event.text {
                        self.final_text = Some(text);
                    }
                    if let Some(input_tokens) = final_event.input_tokens {
                        self.final_input_tokens = Some(input_tokens);
                    }
                    if let Some(output_tokens) = final_event.output_tokens {
                        self.final_output_tokens = Some(output_tokens);
                    }
                    if let Some(issue) = final_event.issue {
                        self.final_issue = Some(issue);
                    }
                }
            }
        }
        Ok(())
    }

    fn try_finish(&mut self) -> Result<Option<ClaudePromptResult>> {
        let Some(status) = self
            .child
            .try_wait()
            .context("failed to poll in-flight `claude -p`")?
        else {
            return Ok(None);
        };

        join_claude_stream_reader(std::mem::replace(
            &mut self.reader,
            thread::spawn(|| Ok(())),
        ))?;

        if self.cancel_requested {
            return Ok(Some(ClaudePromptResult {
                text: String::new(),
                stop_reason: ChildRuntimeStopReason::Cancelled,
            }));
        }
        if let Some(issue) = &self.final_issue {
            bail!("`claude -p` failed: {issue}");
        }
        if !status.success() {
            bail!("`claude -p` failed with status {status}");
        }

        let text = self
            .final_text
            .clone()
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| self.agent_message_chunks.join(""));
        if text.trim().is_empty() {
            bail!("`claude -p` completed without a final response");
        }

        Ok(Some(ClaudePromptResult {
            text,
            stop_reason: ChildRuntimeStopReason::EndTurn,
        }))
    }

    fn finish_blocking(mut self) -> Result<()> {
        let _ = self.child.wait();
        join_claude_stream_reader(self.reader)
    }
}

fn spawn_claude_stream_reader(
    stdout: ChildStdout,
    stream_tx: Sender<ClaudeStreamEvent>,
) -> JoinHandle<Result<()>> {
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut frame = String::new();
        let mut tool_commands = HashMap::new();

        loop {
            frame.clear();
            if reader
                .read_line(&mut frame)
                .context("failed to read `claude -p` output")?
                == 0
            {
                break;
            }

            let line = frame.trim();
            if line.is_empty() {
                continue;
            }

            for event in parse_claude_events(line, &mut tool_commands)? {
                if stream_tx.send(event).is_err() {
                    return Ok(());
                }
            }
        }

        Ok(())
    })
}

fn join_claude_stream_reader(reader: JoinHandle<Result<()>>) -> Result<()> {
    match reader.join() {
        Ok(result) => result,
        Err(_) => bail!("Claude stream reader thread panicked"),
    }
}

fn drain_prompt_updates<W>(prompt: &mut ClaudePromptExecution, output: &mut W) -> Result<()>
where
    W: Write,
{
    prompt.try_drain_updates(output)
}

fn parse_claude_events(
    line: &str,
    tool_commands: &mut HashMap<String, String>,
) -> Result<Vec<ClaudeStreamEvent>> {
    let event: Value = serde_json::from_str(line)
        .with_context(|| format!("invalid `claude -p` stream event: {line}"))?;

    match event.get("type").and_then(Value::as_str) {
        Some("system") | Some("rate_limit_event") => Ok(Vec::new()),
        Some("stream_event") => Ok(parse_stream_event(event.get("event"))),
        Some("assistant") => Ok(parse_assistant_event(event.get("message"), tool_commands)),
        Some("user") => Ok(parse_user_event(
            event.get("message"),
            event.get("tool_use_result"),
            tool_commands,
        )),
        Some("result") => Ok(parse_result_event(&event)),
        _ => Ok(Vec::new()),
    }
}

fn parse_stream_event(event: Option<&Value>) -> Vec<ClaudeStreamEvent> {
    let Some(event) = event else {
        return Vec::new();
    };

    match event.get("type").and_then(Value::as_str) {
        Some("message_start") => vec![ClaudeStreamEvent::Update(ChildRuntimeUpdate::TurnStarted)],
        Some("content_block_delta") => event
            .get("delta")
            .and_then(|delta| match delta.get("type").and_then(Value::as_str) {
                Some("text_delta") => delta.get("text").and_then(Value::as_str),
                _ => None,
            })
            .filter(|text| !text.is_empty())
            .map(|text| {
                vec![ClaudeStreamEvent::Update(ChildRuntimeUpdate::AgentMessage {
                    text: text.to_string(),
                })]
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn parse_assistant_event(
    message: Option<&Value>,
    tool_commands: &mut HashMap<String, String>,
) -> Vec<ClaudeStreamEvent> {
    let Some(content) = message
        .and_then(|message| message.get("content"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };

    let mut events = Vec::new();
    for block in content {
        if block.get("type").and_then(Value::as_str) != Some("tool_use") {
            continue;
        }

        let Some(tool_id) = block.get("id").and_then(Value::as_str) else {
            continue;
        };
        let command = describe_tool_call(block);
        tool_commands.insert(tool_id.to_string(), command.clone());
        events.push(ClaudeStreamEvent::Update(ChildRuntimeUpdate::CommandStarted {
            command,
        }));
    }

    events
}

fn parse_user_event(
    message: Option<&Value>,
    tool_use_result: Option<&Value>,
    tool_commands: &mut HashMap<String, String>,
) -> Vec<ClaudeStreamEvent> {
    let Some(content) = message
        .and_then(|message| message.get("content"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };

    let mut events = Vec::new();
    for block in content {
        if block.get("type").and_then(Value::as_str) != Some("tool_result") {
            continue;
        }

        let command = block
            .get("tool_use_id")
            .and_then(Value::as_str)
            .and_then(|tool_id| tool_commands.remove(tool_id))
            .unwrap_or_else(|| "tool result".to_string());
        events.push(ClaudeStreamEvent::Update(
            ChildRuntimeUpdate::CommandCompleted {
                command,
                output: tool_result_output(block, tool_use_result),
                exit_code: None,
            },
        ));
    }

    events
}

fn parse_result_event(event: &Value) -> Vec<ClaudeStreamEvent> {
    if event.get("is_error").and_then(Value::as_bool) == Some(true) {
        let message = event
            .get("result")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| "Claude prompt failed".to_string());
        return vec![ClaudeStreamEvent::Final(ClaudeFinalEvent {
            text: None,
            input_tokens: None,
            output_tokens: None,
            issue: Some(message),
        })];
    }

    let input_tokens = event
        .get("usage")
        .and_then(|usage| usage.get("input_tokens"))
        .and_then(Value::as_u64);
    let output_tokens = event
        .get("usage")
        .and_then(|usage| usage.get("output_tokens"))
        .and_then(Value::as_u64);
    let text = event
        .get("result")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    vec![
        ClaudeStreamEvent::Update(ChildRuntimeUpdate::TurnCompleted {
            input_tokens,
            output_tokens,
        }),
        ClaudeStreamEvent::Final(ClaudeFinalEvent {
            text,
            input_tokens,
            output_tokens,
            issue: None,
        }),
    ]
}

fn describe_tool_call(block: &Value) -> String {
    let name = block.get("name").and_then(Value::as_str).unwrap_or("tool");
    let input = block.get("input");
    let command = input
        .and_then(|input| input.get("command"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if let Some(command) = command {
        return command.to_string();
    }

    let description = input
        .and_then(|input| input.get("description"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    match description {
        Some(description) => format!("{name}: {description}"),
        None => name.to_string(),
    }
}

fn tool_result_output(block: &Value, tool_use_result: Option<&Value>) -> Option<String> {
    let stdout = tool_use_result
        .and_then(|value| value.get("stdout"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let stderr = tool_use_result
        .and_then(|value| value.get("stderr"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let content = block
        .get("content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());

    stdout
        .or(stderr)
        .or(content)
        .map(truncate_output)
        .filter(|value| !value.is_empty())
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
        .context("failed to write ACP Claude runtime message")?;
    writer
        .write_all(b"\n")
        .context("failed to terminate ACP Claude runtime message")?;
    writer
        .flush()
        .context("failed to flush ACP Claude runtime message")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_claude_prompt, doctor_guidance, parse_claude_events, parse_print_help_capabilities,
        truncate_output, ClaudeRuntimeReadiness,
    };
    use crate::acp::session::ChildRuntimeUpdate;
    use crate::acp::session::PromptBlock;
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn claude_prompt_requires_text_blocks() {
        let error = build_claude_prompt(
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
    fn claude_prompt_includes_context_summary() {
        let prompt = build_claude_prompt(
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
    fn parse_print_help_capabilities_detects_required_flags() {
        let help = "\
Usage: claude [options] [command] [prompt]
  -p, --print
      --output-format
      --include-partial-messages
      --permission-mode
      --model
      --no-session-persistence
      --verbose
";
        let capabilities = parse_print_help_capabilities(help);
        assert!(capabilities.supports_print);
        assert!(capabilities.supports_stream_json);
        assert!(capabilities.supports_partial_messages);
        assert!(capabilities.supports_permission_mode);
        assert!(capabilities.supports_model_flag);
        assert!(capabilities.supports_no_session_persistence);
        assert!(capabilities.supports_verbose);
    }

    #[test]
    fn parse_claude_events_handles_text_commands_and_result() {
        let mut tool_commands = HashMap::new();

        let turn_started = parse_claude_events(
            r#"{"type":"stream_event","event":{"type":"message_start","message":{"id":"msg_1"}}}"#,
            &mut tool_commands,
        )
        .expect("parse turn");
        assert_eq!(turn_started.len(), 1);
        assert_eq!(
            turn_started[0],
            super::ClaudeStreamEvent::Update(ChildRuntimeUpdate::TurnStarted)
        );

        let command_started = parse_claude_events(
            r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"pwd","description":"Print working directory"}}]}}"#,
            &mut tool_commands,
        )
        .expect("parse assistant");
        assert_eq!(
            command_started,
            vec![super::ClaudeStreamEvent::Update(
                ChildRuntimeUpdate::CommandStarted {
                    command: "pwd".to_string(),
                }
            )]
        );

        let command_completed = parse_claude_events(
            r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"/tmp"}]},"tool_use_result":{"stdout":"/tmp","stderr":"","interrupted":false}}"#,
            &mut tool_commands,
        )
        .expect("parse tool result");
        assert_eq!(
            command_completed,
            vec![super::ClaudeStreamEvent::Update(
                ChildRuntimeUpdate::CommandCompleted {
                    command: "pwd".to_string(),
                    output: Some("/tmp".to_string()),
                    exit_code: None,
                }
            )]
        );

        let text_delta = parse_claude_events(
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}}"#,
            &mut tool_commands,
        )
        .expect("parse text delta");
        assert_eq!(
            text_delta,
            vec![super::ClaudeStreamEvent::Update(ChildRuntimeUpdate::AgentMessage {
                text: "hello".to_string(),
            })]
        );

        let result = parse_claude_events(
            r#"{"type":"result","subtype":"success","is_error":false,"result":"hello","stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":9}}"#,
            &mut tool_commands,
        )
        .expect("parse result");
        assert_eq!(
            result[0],
            super::ClaudeStreamEvent::Update(ChildRuntimeUpdate::TurnCompleted {
                input_tokens: Some(4),
                output_tokens: Some(9),
            })
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
    fn doctor_guidance_for_missing_claude_is_actionable() {
        let readiness = ClaudeRuntimeReadiness {
            executable_path: None,
            version: None,
            supports_print: false,
            supports_stream_json: false,
            supports_partial_messages: false,
            supports_permission_mode: false,
            supports_model_flag: false,
            supports_no_session_persistence: false,
            supports_verbose: false,
            auth_ready: false,
            auth_method: None,
            api_provider: None,
            subscription_type: None,
            ready: false,
            issue: Some("`claude` was not found on PATH".to_string()),
        };

        let guidance = doctor_guidance(&readiness);
        assert!(guidance[0].contains("Install Claude Code"));
        assert!(guidance[1].contains("claude --version"));
    }

    #[test]
    fn doctor_guidance_for_ready_claude_recommends_next_step() {
        let readiness = ClaudeRuntimeReadiness {
            executable_path: Some(PathBuf::from("/tmp/claude")),
            version: Some("2.1.69".to_string()),
            supports_print: true,
            supports_stream_json: true,
            supports_partial_messages: true,
            supports_permission_mode: true,
            supports_model_flag: true,
            supports_no_session_persistence: true,
            supports_verbose: true,
            auth_ready: true,
            auth_method: Some("claude.ai".to_string()),
            api_provider: Some("firstParty".to_string()),
            subscription_type: Some("pro".to_string()),
            ready: true,
            issue: None,
        };

        let guidance = doctor_guidance(&readiness);
        assert!(guidance[0].contains("opengateway acp serve --agent claude"));
        assert!(guidance[1].contains("claude -p 'hello'"));
    }
}
