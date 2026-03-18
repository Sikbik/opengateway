use super::adapters::AgentKind;
use super::errors::{
    INTERNAL_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, PARSE_ERROR, SERVER_NOT_INITIALIZED,
};
use super::journal::{append_journal_event, append_session_log};
use super::protocol::{
    error_response, initialize_result, notification, success_response, RpcError, RpcRequest,
    JSONRPC_VERSION,
};
use super::redact::redact_text;
use super::session::{CancelParams, ChildRuntimeUpdate, NewSessionParams, PromptParams};
use super::supervisor::{CancelOutcome, RuntimeMode, SessionSupervisor};
use crate::paths::AppPaths;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub const TRANSPORT_NAME: &str = "stdio";

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub agent: AgentKind,
    pub workspace: Option<PathBuf>,
    pub paths: Option<AppPaths>,
    pub runtime_mode: RuntimeMode,
}

struct ServerState {
    initialized: bool,
    shutdown_requested: bool,
    supervisor: Arc<Mutex<SessionSupervisor>>,
    active_prompt: Option<ActivePrompt>,
}

enum LoopControl {
    Continue,
    Exit,
}

struct ActivePrompt {
    session_id: String,
    request_id: Value,
    prompt_len: usize,
    event_rx: mpsc::Receiver<PromptWorkerEvent>,
    cancel_tx: mpsc::Sender<()>,
    worker: JoinHandle<()>,
}

enum PromptWorkerEvent {
    Update(ChildRuntimeUpdate),
    Completed(PromptResultPayload),
    Failed(String),
}

struct PromptResultPayload {
    outcome: super::supervisor::PromptOutcome,
}

enum InputFrame {
    Line(String),
    Eof,
    Error(String),
}

pub fn serve_stdio<R, W, L>(
    config: ServeConfig,
    input: R,
    output: &mut W,
    log: &mut L,
) -> Result<()>
where
    R: BufRead + Send + 'static,
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

    let mut state = ServerState {
        initialized: false,
        shutdown_requested: false,
        supervisor: Arc::new(Mutex::new(match config.paths.as_ref() {
            Some(paths) => {
                SessionSupervisor::new_with_persisted_counter(config.runtime_mode, paths)?
            }
            None => SessionSupervisor::new(config.runtime_mode),
        })),
        active_prompt: None,
    };
    let (input_tx, input_rx) = mpsc::channel();
    let _input_reader = spawn_input_reader(input, input_tx);
    let mut input_closed = false;
    loop {
        if matches!(
            handle_active_prompt(&config, &mut state, output, log)?,
            LoopControl::Exit
        ) {
            break;
        }
        if input_closed && state.active_prompt.is_none() {
            log_line(log, "acp stdio server stopping after EOF and prompt drain")?;
            break;
        }

        match input_rx.recv_timeout(Duration::from_millis(25)) {
            Ok(InputFrame::Line(frame)) => {
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

                match handle_request(&config, &mut state, request, output, log)? {
                    LoopControl::Continue => {}
                    LoopControl::Exit => {
                        log_line(log, "acp stdio server exiting after exit notification")?;
                        break;
                    }
                }
            }
            Ok(InputFrame::Eof) => {
                input_closed = true;
            }
            Ok(InputFrame::Error(error)) => return Err(anyhow::anyhow!(error)),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                input_closed = true;
            }
        }
    }

    if let Some(active) = state.active_prompt.take() {
        let _ = active.cancel_tx.send(());
        let _ = active.worker.join();
    }
    if let Err(error) = state
        .supervisor
        .lock()
        .expect("ACP supervisor mutex poisoned")
        .shutdown_all()
    {
        log_line(log, &format!("failed to clean up ACP sessions on shutdown: {error}"))?;
    }
    Ok(())
}

fn handle_request<W, L>(
    config: &ServeConfig,
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
        "session/new" => {
            let Some(id) = request.id else {
                write_message(
                    output,
                    &error_response(
                        None,
                        RpcError::new(INVALID_REQUEST, "session/new must be sent as a request", None),
                    ),
                )?;
                return Ok(LoopControl::Continue);
            };

            let params: NewSessionParams = match decode_params(request.params, "session/new") {
                Ok(params) => params,
                Err(error) => {
                    write_message(
                        output,
                        &error_response(
                            Some(id),
                            RpcError::new(
                                INVALID_REQUEST,
                                "invalid params for session/new",
                                Some(json!({ "detail": error.to_string() })),
                            ),
                        ),
                    )?;
                    return Ok(LoopControl::Continue);
                }
            };
            let session = match state
                .supervisor
                .lock()
                .expect("ACP supervisor mutex poisoned")
                .create(config.agent, params)
            {
                Ok(session) => session,
                Err(error) => {
                    write_message(
                        output,
                        &error_response(
                            Some(id),
                            RpcError::new(
                                INTERNAL_ERROR,
                                "failed to create ACP session",
                                Some(json!({ "detail": error.to_string() })),
                            ),
                        ),
                    )?;
                    return Ok(LoopControl::Continue);
                }
            };
            log_line(
                log,
                &format!("acp session created: {} ({})", session.session_id, session.cwd.display()),
            )?;
            record_session_event(
                config,
                log,
                &session.session_id,
                "session.created",
                json!({
                    "agent": config.agent.as_str(),
                    "cwd": session.cwd.display().to_string(),
                    "mcpServers": session.mcp_server_count,
                }),
            )?;
            record_session_log(
                config,
                log,
                &session.session_id,
                &format!("created session in {}", session.cwd.display()),
            )?;
            write_message(
                output,
                &success_response(
                    id,
                    json!({
                        "sessionId": session.session_id,
                    }),
                ),
            )?;
        }
        "session/prompt" => {
            let Some(id) = request.id else {
                write_message(
                    output,
                    &error_response(
                        None,
                        RpcError::new(
                            INVALID_REQUEST,
                            "session/prompt must be sent as a request",
                            None,
                        ),
                    ),
                )?;
                return Ok(LoopControl::Continue);
            };

            let params: PromptParams = match decode_params(request.params, "session/prompt") {
                Ok(params) => params,
                Err(error) => {
                    write_message(
                        output,
                        &error_response(
                            Some(id),
                            RpcError::new(
                                INVALID_REQUEST,
                                "invalid params for session/prompt",
                                Some(json!({ "detail": error.to_string() })),
                            ),
                        ),
                    )?;
                    return Ok(LoopControl::Continue);
                }
            };
            if state.active_prompt.is_some() {
                write_message(
                    output,
                    &error_response(
                        Some(id),
                        RpcError::new(
                            INVALID_REQUEST,
                            "only one active ACP prompt is supported at a time in this MVP",
                            None,
                        ),
                    ),
                )?;
                return Ok(LoopControl::Continue);
            }

            if !state
                .supervisor
                .lock()
                .expect("ACP supervisor mutex poisoned")
                .contains(&params.session_id)
            {
                write_message(
                    output,
                    &error_response(
                        Some(id),
                        RpcError::new(
                            INVALID_REQUEST,
                            "session/prompt references an unknown sessionId",
                            Some(json!({ "sessionId": params.session_id })),
                        ),
                    ),
                )?;
                return Ok(LoopControl::Continue);
            }

            state.active_prompt = Some(spawn_prompt_worker(
                state.supervisor.clone(),
                params,
                id,
            ));
        }
        "session/cancel" => {
            if request.id.is_some() {
                write_message(
                    output,
                    &error_response(
                        request.id,
                        RpcError::new(
                            INVALID_REQUEST,
                            "session/cancel must be sent as a notification",
                            None,
                        ),
                    ),
                )?;
                return Ok(LoopControl::Continue);
            }

            let params: CancelParams = match decode_params(request.params, "session/cancel") {
                Ok(params) => params,
                Err(error) => {
                    log_line(log, &format!("invalid session/cancel notification: {error}"))?;
                    return Ok(LoopControl::Continue);
                }
            };
            if let Some(active) = state.active_prompt.as_ref() {
                if active.session_id == params.session_id {
                    let _ = active.cancel_tx.send(());
                    log_line(log, &format!("acp prompt cancel requested {}", params.session_id))?;
                    record_session_event(
                        config,
                        log,
                        &params.session_id,
                        "session.cancel_requested",
                        json!({}),
                    )?;
                    record_session_log(
                        config,
                        log,
                        &params.session_id,
                        "cancel notification received for active prompt",
                    )?;
                    return Ok(LoopControl::Continue);
                }
            }

            match state
                .supervisor
                .lock()
                .expect("ACP supervisor mutex poisoned")
                .cancel(params.clone())
            {
                Ok(CancelOutcome::Cancelled) => {
                    log_line(log, &format!("acp session cancelled {}", params.session_id))?;
                    record_session_event(
                        config,
                        log,
                        &params.session_id,
                        "session.cancel",
                        json!({}),
                    )?;
                    record_session_log(
                        config,
                        log,
                        &params.session_id,
                        "cancel notification received",
                    )?;
                }
                Ok(CancelOutcome::UnknownSession) => {
                    log_line(
                        log,
                        &format!("acp cancel ignored for unknown session {}", params.session_id),
                    )?;
                }
                Err(error) => {
                    log_line(
                        log,
                        &format!("failed to cancel ACP session {}: {error}", params.session_id),
                    )?;
                }
            }
        }
        "exit" => {
            if !state.shutdown_requested {
                log_line(log, "acp exit received without prior shutdown request")?;
            }
            return Ok(LoopControl::Exit);
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

fn decode_params<T>(params: Option<Value>, method: &str) -> Result<T>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_value(params.unwrap_or(Value::Null)).with_context(|| {
        format!("invalid params for {method}")
    })
}

fn log_line<L: Write>(log: &mut L, message: &str) -> Result<()> {
    writeln!(log, "{}", redact_text(message)).context("failed to write ACP stderr log")?;
    log.flush().context("failed to flush ACP stderr log")?;
    Ok(())
}

fn record_session_event<L: Write>(
    config: &ServeConfig,
    log: &mut L,
    session_id: &str,
    event: &str,
    data: Value,
) -> Result<()> {
    let Some(paths) = &config.paths else {
        return Ok(());
    };

    if let Err(error) = append_journal_event(paths, session_id, event, data) {
        log_line(
            log,
            &format!("failed to append ACP journal for {session_id}: {error}"),
        )?;
    }
    Ok(())
}

fn record_session_log<L: Write>(
    config: &ServeConfig,
    log: &mut L,
    session_id: &str,
    message: &str,
) -> Result<()> {
    let Some(paths) = &config.paths else {
        return Ok(());
    };

    if let Err(error) = append_session_log(paths, session_id, message) {
        log_line(
            log,
            &format!("failed to append ACP session log for {session_id}: {error}"),
        )?;
    }
    Ok(())
}

fn spawn_input_reader<R>(input: R, input_tx: mpsc::Sender<InputFrame>) -> JoinHandle<()>
where
    R: BufRead + Send + 'static,
{
    thread::spawn(move || {
        let mut input = input;
        let mut frame = String::new();

        loop {
            frame.clear();
            match input.read_line(&mut frame) {
                Ok(0) => {
                    let _ = input_tx.send(InputFrame::Eof);
                    break;
                }
                Ok(_) => {
                    if input_tx.send(InputFrame::Line(frame.clone())).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let _ = input_tx.send(InputFrame::Error(error.to_string()));
                    break;
                }
            }
        }
    })
}

fn spawn_prompt_worker(
    supervisor: Arc<Mutex<SessionSupervisor>>,
    params: PromptParams,
    request_id: Value,
) -> ActivePrompt {
    let (event_tx, event_rx) = mpsc::channel();
    let (cancel_tx, cancel_rx) = mpsc::channel();
    let session_id = params.session_id.clone();
    let prompt_len = params.prompt.len();
    let worker = thread::spawn(move || {
        let result = supervisor
            .lock()
            .expect("ACP supervisor mutex poisoned")
            .prompt(params, &cancel_rx, |update| {
                event_tx
                    .send(PromptWorkerEvent::Update(update))
                    .map_err(|_| anyhow::anyhow!("ACP prompt event channel closed"))
            });

        match result {
            Ok(outcome) => {
                let _ = event_tx.send(PromptWorkerEvent::Completed(PromptResultPayload { outcome }));
            }
            Err(error) => {
                let _ = event_tx.send(PromptWorkerEvent::Failed(error.to_string()));
            }
        }
    });

    ActivePrompt {
        session_id,
        request_id,
        prompt_len,
        event_rx,
        cancel_tx,
        worker,
    }
}

fn handle_active_prompt<W, L>(
    config: &ServeConfig,
    state: &mut ServerState,
    output: &mut W,
    log: &mut L,
) -> Result<LoopControl>
where
    W: Write,
    L: Write,
{
    let Some(active) = state.active_prompt.as_mut() else {
        return Ok(LoopControl::Continue);
    };

    loop {
        match active.event_rx.try_recv() {
            Ok(PromptWorkerEvent::Update(update)) => {
                emit_prompt_update(config, output, log, &active.session_id, &update)?;
            }
            Ok(PromptWorkerEvent::Completed(payload)) => {
                let outcome = payload.outcome;
                record_session_event(
                    config,
                    log,
                    &outcome.session_id,
                    "prompt.completed",
                    json!({
                        "promptCount": outcome.prompt_count,
                        "userMessageId": outcome.message_id.clone(),
                        "promptBlocks": active.prompt_len,
                        "stopReason": match outcome.stop_reason {
                            super::session::ChildRuntimeStopReason::EndTurn => "end_turn",
                            super::session::ChildRuntimeStopReason::Cancelled => "cancelled",
                        },
                    }),
                )?;
                record_session_log(
                    config,
                    log,
                    &outcome.session_id,
                    &format!(
                        "completed prompt {} with {} block(s), stop_reason={}",
                        outcome.prompt_count,
                        active.prompt_len,
                        match outcome.stop_reason {
                            super::session::ChildRuntimeStopReason::EndTurn => "end_turn",
                            super::session::ChildRuntimeStopReason::Cancelled => "cancelled",
                        }
                    ),
                )?;
                write_message(
                    output,
                    &success_response(
                        active.request_id.clone(),
                        json!({
                            "stopReason": match outcome.stop_reason {
                                super::session::ChildRuntimeStopReason::EndTurn => "end_turn",
                                super::session::ChildRuntimeStopReason::Cancelled => "cancelled",
                            },
                            "userMessageId": outcome.message_id,
                        }),
                    ),
                )?;
                let finished = state.active_prompt.take().expect("active prompt missing");
                let _ = finished.worker.join();
                return Ok(LoopControl::Continue);
            }
            Ok(PromptWorkerEvent::Failed(error)) => {
                record_session_event(
                    config,
                    log,
                    &active.session_id,
                    "session.runtime_failed",
                    json!({
                        "reason": "worker_failed",
                        "detail": error,
                    }),
                )?;
                record_session_log(config, log, &active.session_id, "ACP prompt worker failed")?;
                write_message(
                    output,
                    &error_response(
                        Some(active.request_id.clone()),
                        RpcError::new(
                            INTERNAL_ERROR,
                            "failed to run ACP session prompt",
                            Some(json!({ "detail": error })),
                        ),
                    ),
                )?;
                let finished = state.active_prompt.take().expect("active prompt missing");
                let _ = finished.worker.join();
                return Ok(LoopControl::Continue);
            }
            Err(mpsc::TryRecvError::Empty) => return Ok(LoopControl::Continue),
            Err(mpsc::TryRecvError::Disconnected) => {
                record_session_event(
                    config,
                    log,
                    &active.session_id,
                    "session.runtime_failed",
                    json!({
                        "reason": "worker_disconnected",
                    }),
                )?;
                record_session_log(
                    config,
                    log,
                    &active.session_id,
                    "ACP prompt worker disconnected unexpectedly",
                )?;
                write_message(
                    output,
                    &error_response(
                        Some(active.request_id.clone()),
                        RpcError::new(
                            INTERNAL_ERROR,
                            "ACP prompt worker disconnected unexpectedly",
                            None,
                        ),
                    ),
                )?;
                let finished = state.active_prompt.take().expect("active prompt missing");
                let _ = finished.worker.join();
                return Ok(LoopControl::Continue);
            }
        }
    }
}

fn emit_prompt_update<W, L>(
    config: &ServeConfig,
    output: &mut W,
    log: &mut L,
    session_id: &str,
    update: &ChildRuntimeUpdate,
) -> Result<()>
where
    W: Write,
    L: Write,
{
    let payload = update_payload(update);
    write_message(
        output,
        &notification(
            "session/update",
            json!({
                "sessionId": session_id,
                "update": payload,
            }),
        ),
    )?;

    record_session_event(config, log, session_id, update_event_name(update), payload.clone())?;
    record_session_log(config, log, session_id, &update_log_line(update))
}

fn update_payload(update: &ChildRuntimeUpdate) -> Value {
    match update {
        ChildRuntimeUpdate::TurnStarted => json!({
            "sessionUpdate": "turn_started",
        }),
        ChildRuntimeUpdate::TurnCancelled => json!({
            "sessionUpdate": "turn_cancelled",
        }),
        ChildRuntimeUpdate::TurnCompleted {
            input_tokens,
            output_tokens,
        } => json!({
            "sessionUpdate": "turn_completed",
            "usage": {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
            }
        }),
        ChildRuntimeUpdate::AgentMessage { text } => json!({
            "sessionUpdate": "agent_message_chunk",
            "content": {
                "type": "text",
                "text": text,
            }
        }),
        ChildRuntimeUpdate::CommandStarted { command } => json!({
            "sessionUpdate": "tool_started",
            "tool": "shell",
            "title": command,
        }),
        ChildRuntimeUpdate::CommandCompleted {
            command,
            output,
            exit_code,
        } => {
            let mut value = json!({
                "sessionUpdate": "tool_completed",
                "tool": "shell",
                "title": command,
                "exitCode": exit_code,
            });
            if let Some(command_output) = output {
                value["content"] = json!({
                    "type": "text",
                    "text": command_output,
                });
            }
            value
        }
    }
}

fn update_event_name(update: &ChildRuntimeUpdate) -> &'static str {
    match update {
        ChildRuntimeUpdate::TurnStarted => "prompt.turn_started",
        ChildRuntimeUpdate::TurnCancelled => "prompt.turn_cancelled",
        ChildRuntimeUpdate::TurnCompleted { .. } => "prompt.turn_completed",
        ChildRuntimeUpdate::AgentMessage { .. } => "prompt.agent_message",
        ChildRuntimeUpdate::CommandStarted { .. } => "prompt.tool_started",
        ChildRuntimeUpdate::CommandCompleted { .. } => "prompt.tool_completed",
    }
}

fn update_log_line(update: &ChildRuntimeUpdate) -> String {
    match update {
        ChildRuntimeUpdate::TurnStarted => "Codex turn started".to_string(),
        ChildRuntimeUpdate::TurnCancelled => "Codex turn cancelled".to_string(),
        ChildRuntimeUpdate::TurnCompleted {
            input_tokens,
            output_tokens,
        } => format!(
            "Codex turn completed (input_tokens={}, output_tokens={})",
            input_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            output_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ),
        ChildRuntimeUpdate::AgentMessage { text } => {
            format!("Codex agent message: {}", summarize_update_text(text))
        }
        ChildRuntimeUpdate::CommandStarted { command } => {
            format!("Codex command started: {}", summarize_update_text(command))
        }
        ChildRuntimeUpdate::CommandCompleted {
            command,
            output,
            exit_code,
        } => format!(
            "Codex command completed: {} (exit_code={}, output={})",
            summarize_update_text(command),
            exit_code
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            output
                .as_deref()
                .map(summarize_update_text)
                .unwrap_or_else(|| "no output".to_string())
        ),
    }
}

fn summarize_update_text(text: &str) -> String {
    const MAX_CHARS: usize = 120;
    let trimmed = text.trim();
    let summary = trimmed.chars().take(MAX_CHARS).collect::<String>();
    if trimmed.chars().count() > MAX_CHARS {
        format!("{summary}…")
    } else {
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::{serve_stdio, update_payload, RuntimeMode, ServeConfig};
    use crate::acp::adapters::AgentKind;
    use crate::acp::session::ChildRuntimeUpdate;
    use serde_json::Value;
    use std::io::Cursor;

    fn run_server(input: &str) -> (Vec<Value>, String) {
        let mut output = Vec::new();
        let mut log = Vec::new();

        serve_stdio(
            ServeConfig {
                agent: AgentKind::Codex,
                workspace: None,
                paths: None,
                runtime_mode: RuntimeMode::InProcessMock,
            },
            Cursor::new(input.as_bytes().to_vec()),
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
        assert_eq!(messages[0]["result"]["agentCapabilities"]["loadSession"], false);
        assert_eq!(
            messages[0]["result"]["agentCapabilities"]["promptCapabilities"]["embeddedContext"],
            false
        );
        assert_eq!(messages[0]["result"]["agentInfo"]["name"], "opengateway");
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
    fn session_new_and_prompt_use_mock_lifecycle() {
        let (messages, _) = run_server(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n\
             {\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"session/new\",\"params\":{\"cwd\":\"/tmp\",\"mcpServers\":[]}}\n\
             {\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"session/prompt\",\"params\":{\"sessionId\":\"session-000001\",\"prompt\":[{\"type\":\"text\",\"text\":\"hello world\"}],\"messageId\":\"user-1\"}}\n",
        );

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[1]["result"]["sessionId"], "session-000001");
        assert_eq!(messages[2]["method"], "session/update");
        assert_eq!(
            messages[2]["params"]["update"]["sessionUpdate"],
            "agent_message_chunk"
        );
        assert_eq!(messages[3]["result"]["stopReason"], "end_turn");
        assert_eq!(messages[3]["result"]["userMessageId"], "user-1");
    }

    #[test]
    fn prompt_rejects_unknown_session_ids() {
        let (messages, _) = run_server(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n\
             {\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"session/prompt\",\"params\":{\"sessionId\":\"missing\",\"prompt\":[{\"type\":\"text\",\"text\":\"hello\"}]}}\n",
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["error"]["code"], -32600);
        assert_eq!(
            messages[1]["error"]["message"],
            "session/prompt references an unknown sessionId"
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

    #[test]
    fn tool_completion_updates_include_shell_details() {
        let payload = update_payload(&ChildRuntimeUpdate::CommandCompleted {
            command: "echo hi".to_string(),
            output: Some("hi".to_string()),
            exit_code: Some(0),
        });

        assert_eq!(payload["sessionUpdate"], "tool_completed");
        assert_eq!(payload["tool"], "shell");
        assert_eq!(payload["title"], "echo hi");
        assert_eq!(payload["exitCode"], 0);
        assert_eq!(payload["content"]["text"], "hi");
    }
}
