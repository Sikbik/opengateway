use crate::acp::session::{
    ChildRuntimeEnvelope, ChildRuntimeRequest, ChildRuntimeResponse, ChildRuntimeUpdate,
    PromptBlock,
};
use anyhow::{bail, Context, Result};
use clap::Args;
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const RUNTIME_NAME: &str = "codex exec";
pub const SCAFFOLD_NOTE: &str =
    "MVP session runtime launches `codex exec --json` for each ACP prompt.";
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

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

pub fn command_codex_runtime(args: CodexRuntimeArgs) -> Result<()> {
    let mut session = CodexRuntimeSession {
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
            .context("failed to read ACP Codex runtime stdin")?
            == 0
        {
            break;
        }

        let trimmed = frame.trim();
        if trimmed.is_empty() {
            continue;
        }

        let request: ChildRuntimeRequest = serde_json::from_str(trimmed)
            .with_context(|| format!("invalid ACP Codex runtime frame: {trimmed}"))?;
        match request.kind.as_str() {
            "prompt" => {
                session.prompt_count += 1;
                let text = run_codex_prompt(
                    &session.cwd,
                    &request.prompt,
                    session.mcp_server_count,
                    &mut stdout,
                )?;
                write_json_line(
                    &mut stdout,
                    &ChildRuntimeEnvelope::Result {
                        result: ChildRuntimeResponse {
                            text,
                            prompt_count: session.prompt_count,
                        },
                    },
                )?;
            }
            "cancel" => break,
            other => bail!("unsupported ACP Codex runtime command: {other}"),
        }
    }

    Ok(())
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

fn run_codex_prompt<W>(
    cwd: &Path,
    prompt: &[PromptBlock],
    mcp_server_count: usize,
    runtime_output: &mut W,
) -> Result<String>
where
    W: Write,
{
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
    let mut reader = BufReader::new(stdout);
    let mut frame = String::new();
    let mut agent_messages = Vec::new();

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
            if let ChildRuntimeUpdate::AgentMessage { text } = &update {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    agent_messages.push(trimmed.to_string());
                }
            }

            write_json_line(
                runtime_output,
                &ChildRuntimeEnvelope::Update { update },
            )?;
        }
    }

    let status = child
        .wait()
        .context("failed while waiting for `codex exec`")?;
    if !status.success() {
        bail!("`codex exec` failed with status {status}");
    }

    if agent_messages.is_empty() {
        bail!("`codex exec --json` completed without an agent_message item");
    }

    Ok(agent_messages.join("\n\n"))
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
    use super::{build_codex_prompt, parse_codex_update, truncate_output};
    use crate::acp::session::ChildRuntimeUpdate;
    use crate::acp::session::PromptBlock;

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
}
