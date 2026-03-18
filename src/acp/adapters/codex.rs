use crate::acp::session::{ChildRuntimeRequest, ChildRuntimeResponse, PromptBlock};
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
                let text = run_codex_prompt(&session.cwd, &request.prompt, session.mcp_server_count)?;
                write_json_line(
                    &mut stdout,
                    &ChildRuntimeResponse {
                        text,
                        prompt_count: session.prompt_count,
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

fn run_codex_prompt(cwd: &Path, prompt: &[PromptBlock], mcp_server_count: usize) -> Result<String> {
    let prompt_text = build_codex_prompt(prompt, mcp_server_count)?;
    let mut command = Command::new("codex");
    command
        .arg("exec")
        .arg("--json")
        .arg("--skip-git-repo-check")
        .arg("-C")
        .arg(cwd)
        .arg("-");
    #[cfg(windows)]
    command.creation_flags(CREATE_NO_WINDOW);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn().context("failed to spawn `codex exec`")?;
    let mut stdin = child
        .stdin
        .take()
        .context("spawned `codex exec` has no stdin")?;
    stdin
        .write_all(prompt_text.as_bytes())
        .context("failed to write ACP prompt to `codex exec`")?;
    drop(stdin);

    let output = child
        .wait_with_output()
        .context("failed while waiting for `codex exec`")?;
    if !output.status.success() {
        bail!(
            "`codex exec` failed: {}",
            stderr_tail(&output.stderr).unwrap_or_else(|| "no stderr output".to_string())
        );
    }

    parse_codex_exec_output(&output.stdout)
}

fn parse_codex_exec_output(stdout: &[u8]) -> Result<String> {
    let output = String::from_utf8_lossy(stdout);
    let mut agent_messages = Vec::new();

    for line in output.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let event: Value = serde_json::from_str(line)
            .with_context(|| format!("invalid `codex exec --json` event: {line}"))?;
        if event.get("type").and_then(Value::as_str) != Some("item.completed") {
            continue;
        }

        let Some(item) = event.get("item") else {
            continue;
        };
        if item.get("type").and_then(Value::as_str) != Some("agent_message") {
            continue;
        }
        if let Some(text) = item.get("text").and_then(Value::as_str) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                agent_messages.push(trimmed.to_string());
            }
        }
    }

    if agent_messages.is_empty() {
        bail!("`codex exec --json` completed without an agent_message item");
    }

    Ok(agent_messages.join("\n\n"))
}

fn stderr_tail(stderr: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(stderr);
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return None;
    }
    let tail = lines
        .into_iter()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join(" | ");
    Some(tail)
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
    use super::{build_codex_prompt, parse_codex_exec_output};
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
    fn parse_codex_exec_output_uses_completed_agent_messages() {
        let text = parse_codex_exec_output(
            br#"{"type":"thread.started","thread_id":"abc"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"first"}}
{"type":"item.completed","item":{"id":"item_1","type":"tool_call"}}
{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"second"}}"#,
        )
        .expect("parse output");
        assert_eq!(text, "first\n\nsecond");
    }
}
