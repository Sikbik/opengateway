use super::adapters::AgentKind;
use super::errors::{
    INVALID_REQUEST, METHOD_NOT_FOUND, PARSE_ERROR, SERVER_NOT_INITIALIZED,
};
use super::protocol::{
    error_response, initialize_result, success_response, RpcError, RpcRequest, JSONRPC_VERSION,
};
use super::redact::redact_text;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::PathBuf;

pub const TRANSPORT_NAME: &str = "stdio";

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub agent: AgentKind,
    pub workspace: Option<PathBuf>,
}

#[derive(Debug, Default)]
struct ServerState {
    initialized: bool,
    shutdown_requested: bool,
}

enum LoopControl {
    Continue,
    Exit,
}

pub fn serve_stdio<R, W, L>(
    config: ServeConfig,
    mut input: R,
    output: &mut W,
    log: &mut L,
) -> Result<()>
where
    R: BufRead,
    W: Write,
    L: Write,
{
    log_line(
        log,
        &format!(
            "acp stdio server ready (agent={}, workspace={})",
            config.agent.as_str(),
            config
                .workspace
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "unset".to_string())
        ),
    )?;

    let mut state = ServerState::default();
    let mut frame = String::new();
    loop {
        frame.clear();
        if input
            .read_line(&mut frame)
            .context("failed to read ACP frame from stdin")?
            == 0
        {
            log_line(log, "acp stdio server stopping on EOF")?;
            break;
        }

        let trimmed = frame.trim();
        if trimmed.is_empty() {
            continue;
        }

        let raw_value = match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => value,
            Err(error) => {
                log_line(
                    log,
                    &format!("acp parse error: {error}; frame={}", redact_text(trimmed)),
                )?;
                write_message(
                    output,
                    &error_response(
                        None,
                        RpcError::new(
                            PARSE_ERROR,
                            "invalid JSON-RPC frame",
                            Some(json!({ "detail": error.to_string() })),
                        ),
                    ),
                )?;
                continue;
            }
        };

        if raw_value.is_array() {
            write_message(
                output,
                &error_response(
                    None,
                    RpcError::new(INVALID_REQUEST, "batch requests are not supported", None),
                ),
            )?;
            continue;
        }

        let request: RpcRequest = match serde_json::from_value(raw_value) {
            Ok(request) => request,
            Err(error) => {
                write_message(
                    output,
                    &error_response(
                        None,
                        RpcError::new(
                            INVALID_REQUEST,
                            "request frame is missing required JSON-RPC fields",
                            Some(json!({ "detail": error.to_string() })),
                        ),
                    ),
                )?;
                continue;
            }
        };

        match handle_request(&mut state, request, output, log)? {
            LoopControl::Continue => {}
            LoopControl::Exit => {
                log_line(log, "acp stdio server exiting after exit notification")?;
                break;
            }
        }
    }

    Ok(())
}

fn handle_request<W, L>(
    state: &mut ServerState,
    request: RpcRequest,
    output: &mut W,
    log: &mut L,
) -> Result<LoopControl>
where
    W: Write,
    L: Write,
{
    if request.jsonrpc.as_deref() != Some(JSONRPC_VERSION) {
        write_message(
            output,
            &error_response(
                request.id,
                RpcError::new(INVALID_REQUEST, "jsonrpc must be \"2.0\"", None),
            ),
        )?;
        return Ok(LoopControl::Continue);
    }

    let Some(method) = request.method.as_deref() else {
        write_message(
            output,
            &error_response(
                request.id,
                RpcError::new(INVALID_REQUEST, "request is missing method", None),
            ),
        )?;
        return Ok(LoopControl::Continue);
    };

    match method {
        "initialize" => {
            let Some(id) = request.id else {
                write_message(
                    output,
                    &error_response(
                        None,
                        RpcError::new(INVALID_REQUEST, "initialize must be sent as a request", None),
                    ),
                )?;
                return Ok(LoopControl::Continue);
            };

            state.initialized = true;
            write_message(output, &success_response(id, initialize_result()))?;
        }
        "shutdown" => {
            let Some(id) = request.id else {
                write_message(
                    output,
                    &error_response(
                        None,
                        RpcError::new(INVALID_REQUEST, "shutdown must be sent as a request", None),
                    ),
                )?;
                return Ok(LoopControl::Continue);
            };

            state.shutdown_requested = true;
            write_message(output, &success_response(id, json!({})))?;
        }
        "exit" => {
            return Ok(LoopControl::Exit);
        }
        other if !state.initialized => {
            if request.id.is_some() {
                write_message(
                    output,
                    &error_response(
                        request.id,
                        RpcError::new(
                            SERVER_NOT_INITIALIZED,
                            "initialize must complete before other ACP methods are used",
                            Some(json!({ "method": other })),
                        ),
                    ),
                )?;
            } else {
                log_line(
                    log,
                    &format!("ignored ACP notification before initialize: {other}"),
                )?;
            }
        }
        other => {
            if request.id.is_some() {
                write_message(
                    output,
                    &error_response(
                        request.id,
                        RpcError::new(
                            METHOD_NOT_FOUND,
                            format!("unsupported ACP method: {other}"),
                            None,
                        ),
                    ),
                )?;
            } else {
                log_line(log, &format!("ignored unsupported ACP notification: {other}"))?;
            }
        }
    }

    Ok(LoopControl::Continue)
}

fn write_message<W: Write>(output: &mut W, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *output, value).context("failed to write ACP response")?;
    output
        .write_all(b"\n")
        .context("failed to terminate ACP response")?;
    output.flush().context("failed to flush ACP response")?;
    Ok(())
}

fn log_line<L: Write>(log: &mut L, message: &str) -> Result<()> {
    writeln!(log, "{}", redact_text(message)).context("failed to write ACP stderr log")?;
    log.flush().context("failed to flush ACP stderr log")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{serve_stdio, ServeConfig};
    use crate::acp::adapters::AgentKind;
    use serde_json::Value;
    use std::io::Cursor;

    fn run_server(input: &str) -> (Vec<Value>, String) {
        let mut output = Vec::new();
        let mut log = Vec::new();

        serve_stdio(
            ServeConfig {
                agent: AgentKind::Codex,
                workspace: None,
            },
            Cursor::new(input.as_bytes()),
            &mut output,
            &mut log,
        )
        .expect("server should run");

        let messages = String::from_utf8(output)
            .expect("utf8 output")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str::<Value>(line).expect("valid JSON-RPC output"))
            .collect::<Vec<_>>();
        let log = String::from_utf8(log).expect("utf8 log");

        (messages, log)
    }

    #[test]
    fn initialize_returns_conservative_capability_set() {
        let (messages, log) =
            run_server("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n");

        assert!(log.contains("acp stdio server ready"));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["id"], 1);
        assert_eq!(messages[0]["result"]["protocolVersion"], 1);
        assert_eq!(messages[0]["result"]["capabilities"]["loadSession"], false);
        assert_eq!(
            messages[0]["result"]["capabilities"]["promptCapabilities"]["embeddedContext"],
            false
        );
    }

    #[test]
    fn malformed_frames_do_not_crash_the_server() {
        let (messages, log) = run_server(
            "not-json\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"initialize\",\"params\":{}}\n",
        );

        assert!(log.contains("acp parse error"));
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["error"]["code"], -32700);
        assert_eq!(messages[1]["id"], 2);
    }

    #[test]
    fn unknown_methods_return_method_not_found_after_initialize() {
        let (messages, _) = run_server(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n\
             {\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"session/new\",\"params\":{}}\n",
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["error"]["code"], -32601);
        assert_eq!(
            messages[1]["error"]["message"],
            "unsupported ACP method: session/new"
        );
    }

    #[test]
    fn exit_notification_stops_the_loop_cleanly() {
        let (messages, log) = run_server(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n\
             {\"jsonrpc\":\"2.0\",\"method\":\"exit\"}\n\
             {\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"initialize\",\"params\":{}}\n",
        );

        assert_eq!(messages.len(), 1);
        assert!(log.contains("acp stdio server exiting after exit notification"));
    }
}
