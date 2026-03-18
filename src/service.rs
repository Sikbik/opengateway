use crate::acp::adapters::AgentKind;
use crate::acp::bridge::{default_bridge_port, DEFAULT_BRIDGE_HOST};
use crate::auth_store::{AuthStore, NewCredential, OAuthCredential};
use crate::oauth;
use anyhow::{anyhow, bail, Context, Result};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_HEADER_BYTES: usize = 64 * 1024;
const REFRESH_SKEW_MS: i64 = 5 * 60 * 1000;
const UPSTREAM_RESPONSES_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex/responses";
const DEFAULT_INSTRUCTIONS: &str = "You are a helpful assistant.";
const RETRY_AFTER_SECONDS: u64 = 1;
const RATE_LIMITER_ENTRY_TTL: Duration = Duration::from_secs(15 * 60);
const RATE_LIMITER_CLEANUP_INTERVAL: Duration = Duration::from_secs(60);
const ACP_MODEL_PREFIX: &str = "acp:";
const ACP_WORKSPACE_HEADER: &str = "x-opengateway-workspace";

const HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "proxy-connection",
];

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub host: String,
    pub port: u16,
    pub api_key: String,
    pub models: Vec<String>,
    pub max_body_bytes: usize,
    pub max_inflight: usize,
    pub max_queue: usize,
    pub queue_timeout: Duration,
    pub per_client_rate: f64,
    pub per_client_burst: usize,
    pub default_acp_workspace: Option<PathBuf>,
}

#[derive(Debug)]
struct ServiceState {
    api_key: String,
    models: Vec<String>,
    max_body_bytes: usize,
    default_acp_workspace: Option<PathBuf>,
    auth_store: AuthStore,
    upstream_client: Client,
    request_gate: Arc<RequestGate>,
    rate_limiter: Option<Arc<RateLimiter>>,
}

#[derive(Debug)]
struct RequestGate {
    max_inflight: usize,
    max_queue: usize,
    queue_timeout: Duration,
    state: Mutex<GateState>,
    notifier: Condvar,
}

#[derive(Debug, Default)]
struct GateState {
    inflight: usize,
    queued: usize,
}

#[derive(Debug)]
enum GateAcquireError {
    QueueFull,
    QueueTimeout,
}

#[derive(Debug)]
struct GatePermit {
    gate: Arc<RequestGate>,
}

#[derive(Debug)]
struct RateLimiter {
    tokens_per_second: f64,
    burst: f64,
    entry_ttl: Duration,
    cleanup_interval: Duration,
    state: Mutex<RateLimiterState>,
}

#[derive(Debug)]
struct RateLimiterState {
    buckets: HashMap<String, ClientBucket>,
    last_cleanup: Instant,
}

#[derive(Debug)]
struct ClientBucket {
    tokens: f64,
    last_refill: Instant,
    last_seen: Instant,
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    header_list: Vec<(String, String)>,
    body: Vec<u8>,
}

#[derive(Debug)]
struct HttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

#[derive(Debug)]
struct RequestError {
    status: u16,
    message: String,
}

impl RequestError {
    fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

pub fn run_service(config: RunConfig, auth_store: AuthStore) -> Result<()> {
    let listener = TcpListener::bind((config.host.as_str(), config.port)).with_context(|| {
        format!(
            "failed to bind local service on {}:{}",
            config.host, config.port
        )
    })?;

    let upstream_client = Client::builder()
        .timeout(Duration::from_secs(10 * 60))
        .build()
        .context("failed to create upstream HTTP client")?;

    let request_gate = Arc::new(RequestGate::new(
        config.max_inflight.max(1),
        config.max_queue,
        config.queue_timeout,
    ));

    let rate_limiter =
        RateLimiter::new(config.per_client_rate, config.per_client_burst).map(Arc::new);

    let state = Arc::new(ServiceState {
        api_key: config.api_key,
        models: config.models,
        max_body_bytes: config.max_body_bytes,
        default_acp_workspace: config.default_acp_workspace,
        auth_store,
        upstream_client,
        request_gate,
        rate_limiter,
    });

    crate::runtime_log_info(format!(
        "opengateway service listening on http://{}:{}",
        config.host, config.port
    ));

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let shared_state = Arc::clone(&state);
                thread::spawn(move || {
                    if let Err(err) = handle_connection(stream, shared_state) {
                        crate::runtime_log_error(format!("connection handling error: {err:#}"));
                    }
                });
            }
            Err(err) => {
                crate::runtime_log_error(format!("listener accept error: {err}"));
            }
        }
    }

    Ok(())
}

fn handle_connection(mut stream: TcpStream, state: Arc<ServiceState>) -> Result<()> {
    let peer_addr = stream.peer_addr().ok();

    let _permit = match state.request_gate.acquire() {
        Ok(permit) => permit,
        Err(GateAcquireError::QueueFull) => {
            let response = retry_after_response(503, "server overloaded", RETRY_AFTER_SECONDS);
            write_http_response(&mut stream, &response)?;
            return Ok(());
        }
        Err(GateAcquireError::QueueTimeout) => {
            let response = retry_after_response(429, "request queue timeout", RETRY_AFTER_SECONDS);
            write_http_response(&mut stream, &response)?;
            return Ok(());
        }
    };

    stream
        .set_read_timeout(Some(Duration::from_secs(20)))
        .context("failed to set socket read timeout")?;
    stream
        .set_write_timeout(Some(Duration::from_secs(20)))
        .context("failed to set socket write timeout")?;

    let request = match read_http_request(&mut stream, state.max_body_bytes) {
        Ok(request) => request,
        Err(err) => {
            if err.message == "empty request" {
                return Ok(());
            }
            let response = plain_text_response(err.status, &err.message);
            write_http_response(&mut stream, &response)?;
            return Ok(());
        }
    };

    if let Some(rate_limiter) = state.rate_limiter.as_ref() {
        let client_key = client_identifier(&request, peer_addr);
        if !rate_limiter.allow(&client_key) {
            let response =
                retry_after_response(429, "rate limit exceeded for client", RETRY_AFTER_SECONDS);
            write_http_response(&mut stream, &response)?;
            return Ok(());
        }
    }

    let path = request.path.split('?').next().unwrap_or_default();
    if request.method == "GET" && path == "/healthz" {
        let response = json_response(200, json!({ "status": "ok" }));
        write_http_response(&mut stream, &response)?;
        return Ok(());
    }

    if !is_authorized(&request, &state.api_key) {
        let response = HttpResponse {
            status: 401,
            headers: vec![(
                "WWW-Authenticate".to_string(),
                "Bearer realm=\"opengateway\"".to_string(),
            )],
            body: b"unauthorized".to_vec(),
        };
        write_http_response(&mut stream, &response)?;
        return Ok(());
    }

    if request.method == "GET" && path == "/v1/models" {
        let response = build_models_response(&state.models);
        write_http_response(&mut stream, &response)?;
        return Ok(());
    }

    if request.method == "POST"
        && (path == "/v1/chat/completions"
            || path == "/v1/responses"
            || path == "/v1/messages"
            || path == "/messages")
    {
        if let Err(err) = proxy_upstream(&mut stream, request, &state) {
            crate::runtime_log_error(format!("upstream proxy error: {err:#}"));
            let response = plain_text_response(502, "bad gateway");
            write_http_response(&mut stream, &response)?;
        }
        return Ok(());
    }

    let response = plain_text_response(404, "not found");
    write_http_response(&mut stream, &response)?;
    Ok(())
}

impl RequestGate {
    fn new(max_inflight: usize, max_queue: usize, queue_timeout: Duration) -> Self {
        Self {
            max_inflight,
            max_queue,
            queue_timeout,
            state: Mutex::new(GateState::default()),
            notifier: Condvar::new(),
        }
    }

    fn acquire(self: &Arc<Self>) -> Result<GatePermit, GateAcquireError> {
        let mut state = self.state.lock().unwrap_or_else(|err| err.into_inner());
        if state.inflight < self.max_inflight {
            state.inflight += 1;
            return Ok(GatePermit {
                gate: Arc::clone(self),
            });
        }

        if state.queued >= self.max_queue {
            return Err(GateAcquireError::QueueFull);
        }

        state.queued += 1;
        let deadline = Instant::now() + self.queue_timeout;

        loop {
            let now = Instant::now();
            if now >= deadline {
                state.queued = state.queued.saturating_sub(1);
                return Err(GateAcquireError::QueueTimeout);
            }

            let wait_for = deadline.saturating_duration_since(now);
            let wait_result = self
                .notifier
                .wait_timeout(state, wait_for)
                .unwrap_or_else(|err| err.into_inner());
            state = wait_result.0;

            if state.inflight < self.max_inflight {
                state.queued = state.queued.saturating_sub(1);
                state.inflight += 1;
                return Ok(GatePermit {
                    gate: Arc::clone(self),
                });
            }

            if wait_result.1.timed_out() {
                state.queued = state.queued.saturating_sub(1);
                return Err(GateAcquireError::QueueTimeout);
            }
        }
    }

    fn release(&self) {
        let mut state = self.state.lock().unwrap_or_else(|err| err.into_inner());
        state.inflight = state.inflight.saturating_sub(1);
        self.notifier.notify_one();
    }
}

impl Drop for GatePermit {
    fn drop(&mut self) {
        self.gate.release();
    }
}

impl RateLimiter {
    fn new(tokens_per_second: f64, burst: usize) -> Option<Self> {
        if tokens_per_second <= 0.0 || burst == 0 {
            return None;
        }

        Some(Self {
            tokens_per_second,
            burst: burst as f64,
            entry_ttl: RATE_LIMITER_ENTRY_TTL,
            cleanup_interval: RATE_LIMITER_CLEANUP_INTERVAL,
            state: Mutex::new(RateLimiterState {
                buckets: HashMap::new(),
                last_cleanup: Instant::now(),
            }),
        })
    }

    fn allow(&self, client_key: &str) -> bool {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap_or_else(|err| err.into_inner());

        if now.duration_since(state.last_cleanup) >= self.cleanup_interval {
            let entry_ttl = self.entry_ttl;
            state
                .buckets
                .retain(|_, bucket| now.duration_since(bucket.last_seen) <= entry_ttl);
            state.last_cleanup = now;
        }

        let bucket = state
            .buckets
            .entry(client_key.to_string())
            .or_insert_with(|| ClientBucket {
                tokens: self.burst,
                last_refill: now,
                last_seen: now,
            });

        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            bucket.tokens = (bucket.tokens + elapsed * self.tokens_per_second).min(self.burst);
            bucket.last_refill = now;
        }
        bucket.last_seen = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

fn proxy_upstream(
    client_stream: &mut TcpStream,
    request: HttpRequest,
    state: &Arc<ServiceState>,
) -> Result<()> {
    if let Some(route) =
        parse_requested_model_id(&request.body).and_then(|model_id| parse_acp_model_route(&model_id))
    {
        return proxy_acp(client_stream, request, state, route);
    }

    let client_requested_stream = request_prefers_stream(&request.body);
    let credential = active_or_refreshed_credential(&state.auth_store)?;
    let normalized_body = normalize_model_alias_in_request_body(request.body);

    let mut outbound = state
        .upstream_client
        .post(UPSTREAM_RESPONSES_ENDPOINT)
        .body(normalized_body);

    for (name, value) in &request.header_list {
        let lower = name.to_ascii_lowercase();
        if should_skip_upstream_header(&lower) {
            continue;
        }
        outbound = outbound.header(name, value);
    }

    outbound = outbound.header(
        "Authorization",
        format!("Bearer {}", credential.access_token),
    );
    if let Some(account_id) = credential.account_id.as_deref() {
        outbound = outbound.header("ChatGPT-Account-Id", account_id);
    }
    outbound = outbound.header("originator", "opengateway");

    let mut upstream_response = outbound
        .send()
        .context("failed to call upstream endpoint")?;
    let status = upstream_response.status().as_u16();

    let mut headers = Vec::new();
    for (name, value) in upstream_response.headers() {
        let lower = name.as_str().to_ascii_lowercase();
        if should_skip_response_header(&lower) {
            continue;
        }
        if let Ok(parsed) = value.to_str() {
            headers.push((name.as_str().to_string(), parsed.to_string()));
        }
    }

    let is_event_stream = headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("content-type")
            && value.to_ascii_lowercase().contains("text/event-stream")
    });

    if client_requested_stream {
        if is_event_stream
            && !headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        {
            headers.push((
                "Content-Type".to_string(),
                "text/event-stream; charset=utf-8".to_string(),
            ));
        }

        write_http_response_head(client_stream, status, &headers, None)?;
        stream_upstream_response_body(client_stream, &mut upstream_response)?;
        return Ok(());
    }

    let body = upstream_response
        .bytes()
        .context("failed to read upstream response body")?
        .to_vec();
    let is_event_stream = is_event_stream || is_sse_payload(&body);

    let mut response = HttpResponse {
        status,
        headers,
        body,
    };

    if !client_requested_stream && status < 400 && is_event_stream {
        if let Some(decoded_body) = extract_response_object_from_sse(&response.body) {
            response
                .headers
                .retain(|(name, _)| !name.eq_ignore_ascii_case("content-type"));
            response.headers.push((
                "Content-Type".to_string(),
                "application/json; charset=utf-8".to_string(),
            ));
            response.body = decoded_body;
        } else {
            let response = json_response(
                502,
                json!({
                    "error": {
                        "message": "failed to decode upstream streaming response",
                        "type": "bad_gateway"
                    }
                }),
            );
            write_http_response(client_stream, &response)?;
            return Ok(());
        }
    }

    write_http_response(client_stream, &response)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AcpModelRoute {
    agent: AgentKind,
    requested_model: String,
    runtime_model: Option<String>,
}

struct AcpBridgeClient {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    next_request_id: u64,
}

struct AcpPromptResult {
    reply_text: String,
    stop_reason: String,
}

struct AcpResponseMeta {
    response_id: String,
    message_id: String,
    model: String,
    created: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AcpHttpSurface {
    OpenAiChatCompletions,
    OpenAiResponses,
    AnthropicMessages,
}

fn proxy_acp(
    client_stream: &mut TcpStream,
    request: HttpRequest,
    state: &Arc<ServiceState>,
    route: AcpModelRoute,
) -> Result<()> {
    let client_requested_stream = request_prefers_stream(&request.body);
    let path = request.path.split('?').next().unwrap_or_default();
    let surface = match classify_acp_http_surface(path) {
        Ok(surface) => surface,
        Err(error) => {
            let response =
                openai_error_response(400, error.to_string(), "invalid_request_error");
            write_http_response(client_stream, &response)?;
            return Ok(());
        }
    };
    let payload: Value = match serde_json::from_slice(&request.body) {
        Ok(payload) => payload,
        Err(error) => {
            let response = openai_error_response(
                400,
                format!("invalid JSON request body: {error}"),
                "invalid_request_error",
            );
            write_http_response(client_stream, &response)?;
            return Ok(());
        }
    };

    let workspace = match resolve_acp_workspace(&request, state.default_acp_workspace.as_deref()) {
        Ok(workspace) => workspace,
        Err(error) => {
            let response =
                openai_error_response(400, error.to_string(), "invalid_request_error");
            write_http_response(client_stream, &response)?;
            return Ok(());
        }
    };
    let prompt_text = build_acp_prompt_text(&payload);
    if prompt_text.trim().is_empty() {
        let response = openai_error_response(
            400,
            "ACP model requests require text input or messages",
            "invalid_request_error",
        );
        write_http_response(client_stream, &response)?;
        return Ok(());
    }

    let mut bridge = match AcpBridgeClient::connect(route.agent) {
        Ok(bridge) => bridge,
        Err(error) => {
            let response = openai_error_response(
                503,
                format!(
                    "ACP {} bridge is unavailable; start it from the ACP panel or `opengateway acp bridge start --agent {}`: {error}",
                    route.agent.as_str(),
                    route.agent.as_str()
                ),
                "service_unavailable",
            );
            write_http_response(client_stream, &response)?;
            return Ok(());
        }
    };

    if let Err(error) = bridge.initialize() {
        let response = openai_error_response(
            502,
            format!("failed to initialize ACP {} bridge: {error}", route.agent.as_str()),
            "bad_gateway",
        );
        write_http_response(client_stream, &response)?;
        return Ok(());
    }

    let session_id = match bridge.new_session(&workspace, route.runtime_model.as_deref()) {
        Ok(session_id) => session_id,
        Err(error) => {
            let response = openai_error_response(
                502,
                format!("failed to create ACP session: {error}"),
                "bad_gateway",
            );
            write_http_response(client_stream, &response)?;
            return Ok(());
        }
    };

    let response_meta = AcpResponseMeta {
        response_id: format!("acp_{}", now_millis()),
        message_id: format!("msg_{}", now_millis()),
        model: route.requested_model.clone(),
        created: (now_millis() / 1000) as u64,
    };

    if client_requested_stream {
        let mut headers = vec![
            (
                "Content-Type".to_string(),
                "text/event-stream; charset=utf-8".to_string(),
            ),
            ("Cache-Control".to_string(), "no-cache".to_string()),
            ("X-Accel-Buffering".to_string(), "no".to_string()),
        ];
        if surface == AcpHttpSurface::AnthropicMessages {
            headers.push((
                "anthropic-version".to_string(),
                "2023-06-01".to_string(),
            ));
        }
        write_http_response_head(client_stream, 200, &headers, None)?;
        emit_acp_stream_prelude(client_stream, surface, &response_meta)?;
    }

    let prompt_result = bridge.prompt(&session_id, &prompt_text, |chunk| {
        if client_requested_stream {
            emit_acp_stream_chunk(client_stream, surface, &response_meta, chunk)?;
        }
        Ok(())
    });

    match prompt_result {
        Ok(result) => {
            if client_requested_stream {
                emit_acp_stream_completion(
                    client_stream,
                    surface,
                    &response_meta,
                    &result.reply_text,
                    &result.stop_reason,
                )?;
            } else {
                let response = build_acp_http_response(
                    surface,
                    &response_meta,
                    &result.reply_text,
                    &result.stop_reason,
                );
                write_http_response(client_stream, &response)?;
            }
        }
        Err(error) => {
            if client_requested_stream {
                emit_acp_stream_error(client_stream, surface, &response_meta, &error.to_string())?;
            } else {
                let response = match surface {
                    AcpHttpSurface::AnthropicMessages => anthropic_error_response(
                        502,
                        format!("ACP prompt failed: {error}"),
                        "api_error",
                    ),
                    _ => openai_error_response(
                        502,
                        format!("ACP prompt failed: {error}"),
                        "bad_gateway",
                    ),
                };
                write_http_response(client_stream, &response)?;
            }
        }
    }

    Ok(())
}

fn parse_requested_model_id(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|payload| payload.get("model").and_then(Value::as_str).map(str::to_string))
}

fn parse_acp_model_route(model_id: &str) -> Option<AcpModelRoute> {
    let trimmed = model_id.trim();
    let suffix = trimmed.strip_prefix(ACP_MODEL_PREFIX)?;
    let (agent_name, runtime_model) = match suffix.split_once(':') {
        Some((agent_name, runtime_model)) => (agent_name, Some(runtime_model.trim().to_string())),
        None => (suffix, None),
    };

    let agent = match agent_name {
        "codex" => AgentKind::Codex,
        "claude" => AgentKind::Claude,
        _ => return None,
    };

    Some(AcpModelRoute {
        agent,
        requested_model: trimmed.to_string(),
        runtime_model: runtime_model.filter(|value| !value.is_empty()),
    })
}

fn classify_acp_http_surface(path: &str) -> Result<AcpHttpSurface> {
    match path {
        "/v1/chat/completions" => Ok(AcpHttpSurface::OpenAiChatCompletions),
        "/v1/responses" => Ok(AcpHttpSurface::OpenAiResponses),
        "/v1/messages" | "/messages" => Ok(AcpHttpSurface::AnthropicMessages),
        other => bail!("ACP model requests are not supported on {other}"),
    }
}

fn resolve_acp_workspace(request: &HttpRequest, default_workspace: Option<&Path>) -> Result<PathBuf> {
    if let Some(header_value) = request.headers.get(ACP_WORKSPACE_HEADER) {
        let candidate = PathBuf::from(header_value.trim());
        let resolved = if candidate.is_absolute() {
            candidate
        } else if let Some(base) = default_workspace {
            base.join(candidate)
        } else {
            bail!("ACP workspace header must be an absolute path when no default ACP workspace is configured");
        };
        if !resolved.is_dir() {
            bail!("ACP workspace does not exist: {}", resolved.display());
        }
        return Ok(resolved);
    }

    let Some(default_workspace) = default_workspace else {
        bail!(
            "ACP model requests require either the X-OpenGateway-Workspace header or OPENGATEWAY_WORKSPACE/current repo root"
        );
    };
    if !default_workspace.is_dir() {
        bail!("configured ACP workspace does not exist: {}", default_workspace.display());
    }
    Ok(default_workspace.to_path_buf())
}

fn build_acp_prompt_text(payload: &Value) -> String {
    let Some(object) = payload.as_object() else {
        return String::new();
    };

    let mut sections = Vec::new();
    if let Some(instructions) = object.get("instructions") {
        let text = collect_text_from_value(instructions);
        if !text.is_empty() {
            sections.push(format!("Instructions:\n{text}"));
        }
    }
    if let Some(system) = object.get("system") {
        let text = collect_text_from_value(system);
        if !text.is_empty() {
            sections.push(format!("System:\n{text}"));
        }
    }
    if let Some(messages) = object.get("messages").and_then(Value::as_array) {
        let conversation = render_acp_messages(messages);
        if !conversation.is_empty() {
            sections.push(format!("Conversation:\n{conversation}"));
        }
    } else if let Some(input) = object.get("input") {
        let conversation = render_acp_input(input);
        if !conversation.is_empty() {
            sections.push(format!("Conversation:\n{conversation}"));
        }
    }

    sections.join("\n\n")
}

fn render_acp_messages(messages: &[Value]) -> String {
    messages
        .iter()
        .filter_map(|message| {
            let object = message.as_object()?;
            let role = object.get("role").and_then(Value::as_str).unwrap_or("user");
            let content = object.get("content").unwrap_or(&Value::Null);
            let text = collect_text_from_value(content);
            if text.is_empty() {
                None
            } else {
                Some(format!("{}: {}", format_message_role(role), text))
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_acp_input(input: &Value) -> String {
    match input {
        Value::String(text) => text.trim().to_string(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| match item {
                Value::Object(object) => {
                    let role = object.get("role").and_then(Value::as_str).unwrap_or("user");
                    let content = object
                        .get("content")
                        .or_else(|| object.get("text"))
                        .unwrap_or(&Value::Null);
                    let text = collect_text_from_value(content);
                    if text.is_empty() {
                        None
                    } else {
                        Some(format!("{}: {}", format_message_role(role), text))
                    }
                }
                Value::String(text) => {
                    let text = text.trim();
                    if text.is_empty() {
                        None
                    } else {
                        Some(format!("User: {text}"))
                    }
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n"),
        Value::Object(object) => {
            let role = object.get("role").and_then(Value::as_str).unwrap_or("user");
            let content = object
                .get("content")
                .or_else(|| object.get("text"))
                .unwrap_or(&Value::Null);
            let text = collect_text_from_value(content);
            if text.is_empty() {
                String::new()
            } else {
                format!("{}: {}", format_message_role(role), text)
            }
        }
        _ => String::new(),
    }
}

fn format_message_role(role: &str) -> &'static str {
    match role {
        "assistant" => "Assistant",
        "system" => "System",
        "developer" => "Developer",
        "tool" => "Tool",
        _ => "User",
    }
}

impl AcpBridgeClient {
    fn connect(agent: AgentKind) -> Result<Self> {
        let endpoint = format!("{}:{}", DEFAULT_BRIDGE_HOST, default_bridge_port(agent));
        let address = endpoint
            .parse::<SocketAddr>()
            .with_context(|| format!("invalid ACP bridge endpoint {endpoint}"))?;
        let writer = TcpStream::connect_timeout(&address, Duration::from_secs(2))
            .with_context(|| format!("failed connecting to ACP {} bridge", agent.as_str()))?;
        writer
            .set_read_timeout(Some(Duration::from_secs(10 * 60)))
            .ok();
        writer
            .set_write_timeout(Some(Duration::from_secs(30)))
            .ok();
        let reader = BufReader::new(
            writer
                .try_clone()
                .context("failed to clone ACP bridge stream")?,
        );
        Ok(Self {
            reader,
            writer,
            next_request_id: 1,
        })
    }

    fn initialize(&mut self) -> Result<()> {
        self.request("initialize", json!({}))?;
        Ok(())
    }

    fn new_session(&mut self, cwd: &Path, model: Option<&str>) -> Result<String> {
        let mut params = json!({
            "cwd": cwd.display().to_string(),
            "mcpServers": [],
        });
        if let Some(model) = model.filter(|value| !value.trim().is_empty()) {
            params["model"] = Value::String(model.to_string());
        }
        let result = self.request("session/new", params)?;
        result
            .get("sessionId")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| anyhow!("ACP session/new did not return sessionId"))
    }

    fn prompt<F>(
        &mut self,
        session_id: &str,
        prompt_text: &str,
        mut on_chunk: F,
    ) -> Result<AcpPromptResult>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        self.write_frame(&json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    }
                ],
                "messageId": format!("http-{}", now_millis()),
            }
        }))?;

        let mut reply_text = String::new();
        loop {
            let frame = self.read_frame()?;
            if frame
                .get("method")
                .and_then(Value::as_str)
                == Some("session/update")
            {
                if frame
                    .get("params")
                    .and_then(|params| params.get("sessionId"))
                    .and_then(Value::as_str)
                    != Some(session_id)
                {
                    continue;
                }
                if let Some(text) = frame
                    .get("params")
                    .and_then(|params| params.get("update"))
                    .and_then(|update| update.get("content"))
                    .and_then(|content| content.get("text"))
                    .and_then(Value::as_str)
                {
                    reply_text.push_str(text);
                    on_chunk(text)?;
                }
                continue;
            }

            if frame.get("id").and_then(Value::as_u64) != Some(request_id) {
                continue;
            }
            if let Some(error) = frame.get("error") {
                bail!("ACP prompt request failed: {}", rpc_error_message(error));
            }

            let stop_reason = frame
                .get("result")
                .and_then(|result| result.get("stopReason"))
                .and_then(Value::as_str)
                .unwrap_or("end_turn")
                .to_string();
            return Ok(AcpPromptResult {
                reply_text,
                stop_reason,
            });
        }
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        self.write_frame(&json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }))?;

        loop {
            let frame = self.read_frame()?;
            if frame.get("id").and_then(Value::as_u64) != Some(request_id) {
                continue;
            }
            if let Some(error) = frame.get("error") {
                bail!("ACP {method} failed: {}", rpc_error_message(error));
            }
            return Ok(frame.get("result").cloned().unwrap_or(Value::Null));
        }
    }

    fn write_frame(&mut self, value: &Value) -> Result<()> {
        serde_json::to_writer(&mut self.writer, value).context("failed writing ACP frame")?;
        self.writer
            .write_all(b"\n")
            .context("failed terminating ACP frame")?;
        self.writer.flush().context("failed flushing ACP frame")
    }

    fn read_frame(&mut self) -> Result<Value> {
        let mut line = String::new();
        loop {
            line.clear();
            let read = self
                .reader
                .read_line(&mut line)
                .context("failed reading ACP frame")?;
            if read == 0 {
                bail!("ACP bridge connection closed unexpectedly");
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            return serde_json::from_str(trimmed)
                .with_context(|| format!("failed decoding ACP frame: {trimmed}"));
        }
    }
}

fn rpc_error_message(error: &Value) -> String {
    let code = error.get("code").and_then(Value::as_i64).unwrap_or_default();
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unknown ACP error");
    format!("{message} (code {code})")
}

fn build_acp_http_response(
    surface: AcpHttpSurface,
    meta: &AcpResponseMeta,
    text: &str,
    stop_reason: &str,
) -> HttpResponse {
    match surface {
        AcpHttpSurface::OpenAiChatCompletions => json_response(
            200,
            json!({
                "id": meta.response_id,
                "object": "chat.completion",
                "created": meta.created,
                "model": meta.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": chat_finish_reason(stop_reason)
                    }
                ]
            }),
        ),
        AcpHttpSurface::OpenAiResponses => json_response(
            200,
            json!({
                "id": meta.response_id,
                "object": "response",
                "created_at": meta.created,
                "status": "completed",
                "model": meta.model,
                "output": [
                    {
                        "type": "message",
                        "id": meta.message_id,
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": []
                            }
                        ]
                    }
                ],
                "output_text": text
            }),
        ),
        AcpHttpSurface::AnthropicMessages => json_response(
            200,
            json!({
                "id": meta.message_id,
                "type": "message",
                "role": "assistant",
                "model": meta.model,
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    }
                ],
                "stop_reason": anthropic_stop_reason(stop_reason),
                "stop_sequence": Value::Null,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            }),
        ),
    }
}

fn emit_acp_stream_prelude(
    client_stream: &mut TcpStream,
    surface: AcpHttpSurface,
    meta: &AcpResponseMeta,
) -> Result<()> {
    match surface {
        AcpHttpSurface::OpenAiChatCompletions => write_sse_data(
            client_stream,
            &json!({
                "id": meta.response_id,
                "object": "chat.completion.chunk",
                "created": meta.created,
                "model": meta.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant"
                        },
                        "finish_reason": Value::Null
                    }
                ]
            }),
        ),
        AcpHttpSurface::OpenAiResponses => write_sse_data(
            client_stream,
            &json!({
                "type": "response.created",
                "response": {
                    "id": meta.response_id,
                    "object": "response",
                    "created_at": meta.created,
                    "status": "in_progress",
                    "model": meta.model,
                    "output": []
                }
            }),
        ),
        AcpHttpSurface::AnthropicMessages => {
            write_anthropic_sse_event(
                client_stream,
                "message_start",
                &json!({
                    "type": "message_start",
                    "message": {
                        "id": meta.message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": meta.model,
                        "content": [],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                        }
                    }
                }),
            )?;
            write_anthropic_sse_event(
                client_stream,
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "text",
                        "text": "",
                    }
                }),
            )
        }
    }
}

fn emit_acp_stream_chunk(
    client_stream: &mut TcpStream,
    surface: AcpHttpSurface,
    meta: &AcpResponseMeta,
    text: &str,
) -> Result<()> {
    match surface {
        AcpHttpSurface::OpenAiChatCompletions => write_sse_data(
            client_stream,
            &json!({
                "id": meta.response_id,
                "object": "chat.completion.chunk",
                "created": meta.created,
                "model": meta.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": text
                        },
                        "finish_reason": Value::Null
                    }
                ]
            }),
        ),
        AcpHttpSurface::OpenAiResponses => write_sse_data(
            client_stream,
            &json!({
                "type": "response.output_text.delta",
                "response_id": meta.response_id,
                "delta": text
            }),
        ),
        AcpHttpSurface::AnthropicMessages => write_anthropic_sse_event(
            client_stream,
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": text,
                }
            }),
        ),
    }
}

fn emit_acp_stream_completion(
    client_stream: &mut TcpStream,
    surface: AcpHttpSurface,
    meta: &AcpResponseMeta,
    text: &str,
    stop_reason: &str,
) -> Result<()> {
    match surface {
        AcpHttpSurface::OpenAiChatCompletions => {
            write_sse_data(
                client_stream,
                &json!({
                    "id": meta.response_id,
                    "object": "chat.completion.chunk",
                    "created": meta.created,
                    "model": meta.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": chat_finish_reason(stop_reason)
                        }
                    ]
                }),
            )?;
            client_stream
                .write_all(b"data: [DONE]\n\n")
                .context("failed writing ACP SSE terminator")?;
            client_stream.flush().ok();
            Ok(())
        }
        AcpHttpSurface::OpenAiResponses => {
            write_sse_data(
                client_stream,
                &json!({
                    "type": "response.completed",
                    "response": {
                        "id": meta.response_id,
                        "object": "response",
                        "created_at": meta.created,
                        "status": "completed",
                        "model": meta.model,
                        "output": [
                            {
                                "type": "message",
                                "id": meta.message_id,
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": text,
                                        "annotations": []
                                    }
                                ]
                            }
                        ],
                        "output_text": text
                    }
                }),
            )?;
            client_stream
                .write_all(b"data: [DONE]\n\n")
                .context("failed writing ACP SSE terminator")?;
            client_stream.flush().ok();
            Ok(())
        }
        AcpHttpSurface::AnthropicMessages => {
            write_anthropic_sse_event(
                client_stream,
                "content_block_stop",
                &json!({
                    "type": "content_block_stop",
                    "index": 0,
                }),
            )?;
            write_anthropic_sse_event(
                client_stream,
                "message_delta",
                &json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": anthropic_stop_reason(stop_reason),
                        "stop_sequence": Value::Null,
                    },
                    "usage": {
                        "output_tokens": 0,
                    }
                }),
            )?;
            write_anthropic_sse_event(
                client_stream,
                "message_stop",
                &json!({
                    "type": "message_stop",
                }),
            )
        }
    }
}

fn emit_acp_stream_error(
    client_stream: &mut TcpStream,
    surface: AcpHttpSurface,
    meta: &AcpResponseMeta,
    error: &str,
) -> Result<()> {
    match surface {
        AcpHttpSurface::AnthropicMessages => write_anthropic_sse_event(
            client_stream,
            "error",
            &json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": error,
                },
                "message": {
                    "id": meta.message_id,
                }
            }),
        ),
        _ => {
            write_sse_data(
                client_stream,
                &json!({
                    "type": "response.failed",
                    "response": {
                        "id": meta.response_id,
                        "object": "response",
                        "created_at": meta.created,
                        "status": "failed",
                        "model": meta.model,
                        "error": {
                            "message": error,
                            "type": "bad_gateway"
                        }
                    }
                }),
            )?;
            client_stream
                .write_all(b"data: [DONE]\n\n")
                .context("failed writing ACP SSE terminator")?;
            client_stream.flush().ok();
            Ok(())
        }
    }
}

fn write_sse_data(client_stream: &mut TcpStream, payload: &Value) -> Result<()> {
    let rendered = serde_json::to_string(payload).context("failed to encode SSE payload")?;
    write!(client_stream, "data: {rendered}\n\n").context("failed writing SSE payload")?;
    client_stream.flush().ok();
    Ok(())
}

fn write_anthropic_sse_event(
    client_stream: &mut TcpStream,
    event: &str,
    payload: &Value,
) -> Result<()> {
    let rendered = serde_json::to_string(payload).context("failed to encode anthropic SSE payload")?;
    write!(client_stream, "event: {event}\ndata: {rendered}\n\n")
        .context("failed writing anthropic SSE payload")?;
    client_stream.flush().ok();
    Ok(())
}

fn chat_finish_reason(stop_reason: &str) -> &str {
    match stop_reason {
        "cancelled" => "stop",
        _ => "stop",
    }
}

fn anthropic_stop_reason(stop_reason: &str) -> &str {
    match stop_reason {
        "cancelled" => "end_turn",
        _ => "end_turn",
    }
}

fn active_or_refreshed_credential(auth_store: &AuthStore) -> Result<OAuthCredential> {
    let mut credential = auth_store
        .active_openai()?
        .ok_or_else(|| anyhow!("no OpenAI OAuth credentials available; run `opengateway login`"))?;

    if credential.expires_at_ms > now_millis() + REFRESH_SKEW_MS {
        return Ok(credential);
    }

    let refreshed =
        oauth::refresh_access_token(&credential.refresh_token, credential.account_id.as_deref())
            .context("failed to refresh access token")?;

    let refreshed_account_id = refreshed.account_id.clone();
    let merged_account_id = refreshed_account_id.or_else(|| credential.account_id.clone());
    auth_store.upsert_openai(NewCredential {
        refresh_token: refreshed.refresh_token.clone(),
        access_token: refreshed.access_token.clone(),
        expires_at_ms: refreshed.expires_at_ms,
        account_id: merged_account_id.clone(),
    })?;

    credential.refresh_token = refreshed.refresh_token;
    credential.access_token = refreshed.access_token;
    credential.expires_at_ms = refreshed.expires_at_ms;
    credential.account_id = merged_account_id;
    Ok(credential)
}

fn read_http_request(
    stream: &mut TcpStream,
    max_body_bytes: usize,
) -> Result<HttpRequest, RequestError> {
    let mut buffer = Vec::with_capacity(4096);
    let mut chunk = [0_u8; 4096];
    let header_end = loop {
        let read_bytes = stream
            .read(&mut chunk)
            .map_err(|_| RequestError::new(400, "failed to read request"))?;
        if read_bytes == 0 {
            return Err(RequestError::new(400, "empty request"));
        }
        buffer.extend_from_slice(&chunk[..read_bytes]);
        if let Some(index) = find_header_end(&buffer) {
            break index;
        }
        if buffer.len() > MAX_HEADER_BYTES {
            return Err(RequestError::new(431, "request header too large"));
        }
    };

    let header_text = std::str::from_utf8(&buffer[..header_end])
        .map_err(|_| RequestError::new(400, "request headers must be valid UTF-8"))?;

    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| RequestError::new(400, "missing request line"))?;

    let mut request_line_parts = request_line.split_whitespace();
    let method = request_line_parts
        .next()
        .ok_or_else(|| RequestError::new(400, "missing method"))?
        .to_string();
    let path = request_line_parts
        .next()
        .ok_or_else(|| RequestError::new(400, "missing path"))?
        .to_string();

    let mut headers = HashMap::new();
    let mut header_list = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| RequestError::new(400, "invalid header"))?;
        let header_name = name.trim().to_string();
        let header_value = value.trim().to_string();
        headers.insert(header_name.to_ascii_lowercase(), header_value.clone());
        header_list.push((header_name, header_value));
    }

    if headers
        .get("transfer-encoding")
        .map(|value| value.to_ascii_lowercase().contains("chunked"))
        .unwrap_or(false)
    {
        return Err(RequestError::new(
            501,
            "chunked request bodies are not supported",
        ));
    }

    let content_length = headers
        .get("content-length")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| RequestError::new(400, "invalid content-length"))
        })
        .transpose()?
        .unwrap_or(0);

    if content_length > max_body_bytes {
        return Err(RequestError::new(413, "payload too large"));
    }

    let body_start = header_end + 4;
    let mut body = if buffer.len() > body_start {
        buffer[body_start..].to_vec()
    } else {
        Vec::new()
    };

    if body.len() > content_length {
        body.truncate(content_length);
    }

    while body.len() < content_length {
        let mut temp = vec![0_u8; content_length - body.len()];
        let n = stream
            .read(&mut temp)
            .map_err(|_| RequestError::new(400, "failed reading request body"))?;
        if n == 0 {
            return Err(RequestError::new(400, "incomplete request body"));
        }
        body.extend_from_slice(&temp[..n]);
    }

    Ok(HttpRequest {
        method,
        path,
        headers,
        header_list,
        body,
    })
}

fn write_http_response(stream: &mut TcpStream, response: &HttpResponse) -> Result<()> {
    write_http_response_head(
        stream,
        response.status,
        &response.headers,
        Some(response.body.len()),
    )?;
    stream
        .write_all(&response.body)
        .context("failed writing HTTP response body")?;
    stream.flush().ok();
    Ok(())
}

fn write_http_response_head(
    stream: &mut TcpStream,
    status: u16,
    headers: &[(String, String)],
    content_length: Option<usize>,
) -> Result<()> {
    let mut payload = Vec::new();

    write!(payload, "HTTP/1.1 {} {}\r\n", status, status_reason(status))
        .context("failed to serialize response status line")?;

    let mut content_type_present = false;
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("content-length") || name.eq_ignore_ascii_case("connection") {
            continue;
        }
        if name.eq_ignore_ascii_case("content-type") {
            content_type_present = true;
        }
        write!(payload, "{}: {}\r\n", name, value)
            .context("failed to serialize response headers")?;
    }

    if !content_type_present {
        write!(payload, "Content-Type: text/plain; charset=utf-8\r\n")
            .context("failed to serialize content-type")?;
    }

    if let Some(content_length) = content_length {
        write!(payload, "Content-Length: {}\r\n", content_length)
            .context("failed to serialize content-length")?;
    }

    write!(payload, "Connection: close\r\n\r\n")
        .context("failed to serialize connection header")?;
    stream
        .write_all(&payload)
        .context("failed writing HTTP response head")?;
    stream.flush().ok();
    Ok(())
}

fn stream_upstream_response_body(
    client_stream: &mut TcpStream,
    upstream_response: &mut reqwest::blocking::Response,
) -> Result<()> {
    let mut buffer = [0_u8; 16 * 1024];

    loop {
        let read_bytes = upstream_response
            .read(&mut buffer)
            .context("failed to read upstream response chunk")?;
        if read_bytes == 0 {
            break;
        }

        client_stream
            .write_all(&buffer[..read_bytes])
            .context("failed writing streamed response chunk")?;
        client_stream.flush().ok();
    }

    Ok(())
}

fn build_models_response(models: &[String]) -> HttpResponse {
    let data = models
        .iter()
        .map(|model| json!({"id": model, "object": "model", "owned_by": "openai"}))
        .collect::<Vec<_>>();

    json_response(
        200,
        json!({
          "object": "list",
          "data": data
        }),
    )
}

fn normalize_model_alias_in_request_body(body: Vec<u8>) -> Vec<u8> {
    let mut payload: Value = match serde_json::from_slice(&body) {
        Ok(payload) => payload,
        Err(_) => return body,
    };

    let Some(payload_object) = payload.as_object_mut() else {
        return body;
    };

    normalize_model_alias_and_reasoning(payload_object);
    normalize_tool_shapes(payload_object);
    convert_messages_into_input(payload_object);
    normalize_input_shape(payload_object);
    ensure_instructions(payload_object);
    strip_unsupported_fields(payload_object);

    payload_object.insert("store".to_string(), Value::Bool(false));
    payload_object.insert("stream".to_string(), Value::Bool(true));

    serde_json::to_vec(&payload).unwrap_or(body)
}

fn request_prefers_stream(body: &[u8]) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|payload| payload.get("stream").and_then(Value::as_bool))
        .unwrap_or(false)
}

fn is_sse_payload(body: &[u8]) -> bool {
    let text = std::str::from_utf8(body).unwrap_or_default();
    let trimmed = text.trim_start();
    trimmed.starts_with("event:") || trimmed.starts_with("data:")
}

fn extract_response_object_from_sse(body: &[u8]) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(body).ok()?;
    let mut response_object: Option<Value> = None;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if !line.starts_with("data:") {
            continue;
        }

        let data = line.trim_start_matches("data:").trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }

        let Ok(event_payload) = serde_json::from_str::<Value>(data) else {
            continue;
        };

        let Some(event_type) = event_payload.get("type").and_then(Value::as_str) else {
            continue;
        };

        let response = event_payload.get("response").cloned();

        match event_type {
            "response.created" => {
                if response_object.is_none() {
                    response_object = response;
                }
            }
            "response.completed" | "response.failed" | "response.incomplete" => {
                if response.is_some() {
                    response_object = response;
                }
            }
            _ => {}
        }
    }

    response_object
        .filter(Value::is_object)
        .and_then(|response| serde_json::to_vec(&response).ok())
}

fn strip_unsupported_fields(payload_object: &mut serde_json::Map<String, Value>) {
    payload_object.retain(|key, _| {
        matches!(
            key.as_str(),
            "model"
                | "instructions"
                | "input"
                | "tools"
                | "tool_choice"
                | "reasoning"
                | "parallel_tool_calls"
                | "text"
                | "include"
                | "stream"
                | "store"
        )
    });
}

fn normalize_tool_shapes(payload_object: &mut serde_json::Map<String, Value>) {
    if let Some(Value::Array(tools)) = payload_object.get_mut("tools") {
        for tool in tools {
            let Some(tool_object) = tool.as_object_mut() else {
                continue;
            };

            let Some(function_value) = tool_object.remove("function") else {
                continue;
            };

            let Value::Object(function_object) = function_value else {
                continue;
            };

            if !tool_object.contains_key("name") {
                if let Some(name) = function_object.get("name").and_then(Value::as_str) {
                    tool_object.insert("name".to_string(), Value::String(name.to_string()));
                }
            }

            if !tool_object.contains_key("description") {
                if let Some(description) =
                    function_object.get("description").and_then(Value::as_str)
                {
                    tool_object.insert(
                        "description".to_string(),
                        Value::String(description.to_string()),
                    );
                }
            }

            if !tool_object.contains_key("parameters") {
                if let Some(parameters) = function_object.get("parameters") {
                    tool_object.insert("parameters".to_string(), parameters.clone());
                }
            }

            if !tool_object.contains_key("strict") {
                if let Some(strict) = function_object.get("strict") {
                    tool_object.insert("strict".to_string(), strict.clone());
                }
            }
        }
    }

    let Some(tool_choice) = payload_object.get_mut("tool_choice") else {
        return;
    };

    let Some(choice_object) = tool_choice.as_object_mut() else {
        return;
    };

    let Some(function_value) = choice_object.remove("function") else {
        return;
    };

    let Value::Object(function_object) = function_value else {
        return;
    };

    if !choice_object.contains_key("name") {
        if let Some(name) = function_object.get("name").and_then(Value::as_str) {
            choice_object.insert("name".to_string(), Value::String(name.to_string()));
        }
    }

    if !choice_object.contains_key("type") {
        choice_object.insert("type".to_string(), Value::String("function".to_string()));
    }
}

fn normalize_model_alias_and_reasoning(payload_object: &mut serde_json::Map<String, Value>) {
    let Some(model_id) = payload_object.get("model").and_then(Value::as_str) else {
        return;
    };

    let Some((canonical_model, effort)) = parse_model_effort_alias(model_id) else {
        return;
    };

    payload_object.insert("model".to_string(), Value::String(canonical_model));

    match payload_object.get_mut("reasoning") {
        Some(existing_reasoning) if existing_reasoning.is_object() => {
            if let Some(reasoning) = existing_reasoning.as_object_mut() {
                reasoning.insert("effort".to_string(), Value::String(effort));
            }
        }
        _ => {
            let mut reasoning = serde_json::Map::new();
            reasoning.insert("effort".to_string(), Value::String(effort));
            payload_object.insert("reasoning".to_string(), Value::Object(reasoning));
        }
    }
}

fn convert_messages_into_input(payload_object: &mut serde_json::Map<String, Value>) {
    let Some(messages_value) = payload_object.remove("messages") else {
        return;
    };

    let Some(messages) = messages_value.as_array() else {
        return;
    };

    let had_input = payload_object.contains_key("input");
    let mut input_items = Vec::new();
    let mut instruction_blocks = Vec::new();
    let mut known_tool_call_ids = HashSet::new();

    for message in messages {
        let Some(message_object) = message.as_object() else {
            continue;
        };

        let role = message_object
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");

        if role == "system" {
            let instruction_text = message_object
                .get("content")
                .map(collect_text_from_value)
                .unwrap_or_default();
            if !instruction_text.is_empty() {
                instruction_blocks.push(instruction_text);
            }
            continue;
        }

        if had_input {
            continue;
        }

        if role == "assistant" {
            let function_calls = convert_assistant_tool_calls_to_input_items(message_object);
            for function_call in &function_calls {
                if let Some(call_id) = function_call.get("call_id").and_then(Value::as_str) {
                    if !call_id.is_empty() {
                        known_tool_call_ids.insert(call_id.to_string());
                    }
                }
            }
            input_items.extend(function_calls);
        }

        if role == "tool" {
            if let Some((call_id, tool_output_item)) =
                convert_tool_message_to_input_item(message_object)
            {
                if known_tool_call_ids.contains(&call_id) {
                    input_items.push(tool_output_item);
                }
            }
            continue;
        }

        let content = message_object
            .get("content")
            .cloned()
            .unwrap_or(Value::Null);
        let normalized_content = normalize_message_content_for_role(content, role);
        if normalized_content.is_empty() {
            continue;
        }

        let mut normalized_message = serde_json::Map::new();
        normalized_message.insert(
            "role".to_string(),
            Value::String(normalize_message_role(role).to_string()),
        );
        normalized_message.insert("content".to_string(), Value::Array(normalized_content));
        input_items.push(Value::Object(normalized_message));
    }

    if !had_input && !input_items.is_empty() {
        payload_object.insert("input".to_string(), Value::Array(input_items));
    }

    if is_missing_or_empty_instructions(payload_object) && !instruction_blocks.is_empty() {
        payload_object.insert(
            "instructions".to_string(),
            Value::String(instruction_blocks.join("\n\n")),
        );
    }
}

fn normalize_input_shape(payload_object: &mut serde_json::Map<String, Value>) {
    let Some(input_value) = payload_object.remove("input") else {
        return;
    };

    let normalized_input = match input_value {
        Value::String(text) => Value::Array(vec![build_text_message("user", text)]),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(normalize_input_item)
                .collect::<Vec<_>>(),
        ),
        Value::Object(object) => Value::Array(vec![normalize_input_item(Value::Object(object))]),
        other => other,
    };

    payload_object.insert("input".to_string(), normalized_input);
}

fn normalize_input_item(item: Value) -> Value {
    match item {
        Value::String(text) => build_text_message("user", text),
        Value::Object(mut object) => {
            if object.get("type").and_then(Value::as_str).is_some() && !object.contains_key("role")
            {
                return Value::Object(object);
            }

            let role = object
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user")
                .to_string();
            let content_value = object
                .remove("content")
                .or_else(|| object.remove("text"))
                .unwrap_or(Value::Null);

            object.insert(
                "content".to_string(),
                Value::Array(normalize_message_content_for_role(content_value, &role)),
            );

            object.insert(
                "role".to_string(),
                Value::String(normalize_message_role(&role).to_string()),
            );

            Value::Object(object)
        }
        other => other,
    }
}

fn normalize_message_content_for_role(content: Value, role: &str) -> Vec<Value> {
    let default_text_type = default_content_text_type(role);
    match content {
        Value::String(text) => vec![json!({"type": default_text_type, "text": text})],
        Value::Array(parts) => parts
            .into_iter()
            .flat_map(|part| normalize_content_part(part, default_text_type))
            .collect::<Vec<_>>(),
        Value::Object(part) => normalize_content_part(Value::Object(part), default_text_type),
        Value::Null => Vec::new(),
        other => vec![json!({"type": default_text_type, "text": other.to_string()})],
    }
}

fn normalize_content_part(part: Value, default_text_type: &str) -> Vec<Value> {
    match part {
        Value::String(text) => vec![json!({"type": default_text_type, "text": text})],
        Value::Object(mut object) => {
            let part_type = object
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();

            if part_type == default_text_type || part_type == "refusal" {
                return vec![Value::Object(object)];
            }

            if matches!(part_type, "" | "text" | "input_text" | "output_text") {
                if let Some(text) = object
                    .remove("text")
                    .and_then(|value| value.as_str().map(str::to_string))
                {
                    return vec![json!({"type": default_text_type, "text": text})];
                }
                if let Some(text) = object
                    .remove("content")
                    .and_then(|value| value.as_str().map(str::to_string))
                {
                    return vec![json!({"type": default_text_type, "text": text})];
                }
            }

            vec![Value::Object(object)]
        }
        _ => Vec::new(),
    }
}

fn normalize_message_role(role: &str) -> &str {
    match role {
        "assistant" => "assistant",
        "developer" => "developer",
        "user" => "user",
        _ => "user",
    }
}

fn default_content_text_type(role: &str) -> &str {
    if role == "assistant" {
        "output_text"
    } else {
        "input_text"
    }
}

fn convert_assistant_tool_calls_to_input_items(
    message_object: &serde_json::Map<String, Value>,
) -> Vec<Value> {
    let Some(tool_calls) = message_object.get("tool_calls").and_then(Value::as_array) else {
        return Vec::new();
    };

    tool_calls
        .iter()
        .filter_map(|tool_call| {
            let tool_call_object = tool_call.as_object()?;
            let function = tool_call_object.get("function")?.as_object()?;
            let name = function.get("name")?.as_str()?;
            let call_id = tool_call_object
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if call_id.is_empty() {
                return None;
            }

            let arguments = function
                .get("arguments")
                .map(stringify_json_value)
                .unwrap_or_else(|| "{}".to_string());

            Some(json!({
                "type": "function_call",
                "name": name,
                "arguments": arguments,
                "call_id": call_id
            }))
        })
        .collect::<Vec<_>>()
}

fn convert_tool_message_to_input_item(
    message_object: &serde_json::Map<String, Value>,
) -> Option<(String, Value)> {
    let call_id = message_object.get("tool_call_id")?.as_str()?;
    let output = message_object
        .get("content")
        .map(collect_text_from_value)
        .unwrap_or_default();

    Some((
        call_id.to_string(),
        json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output
        }),
    ))
}

fn stringify_json_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()),
    }
}

fn build_text_message(role: &str, text: String) -> Value {
    json!({
        "role": role,
        "content": [
            {
                "type": "input_text",
                "text": text
            }
        ]
    })
}

fn ensure_instructions(payload_object: &mut serde_json::Map<String, Value>) {
    if !is_missing_or_empty_instructions(payload_object) {
        return;
    }

    if let Some(system_value) = payload_object.remove("system") {
        let system_text = collect_text_from_value(&system_value);
        if !system_text.is_empty() {
            payload_object.insert("instructions".to_string(), Value::String(system_text));
            return;
        }
    }

    payload_object.insert(
        "instructions".to_string(),
        Value::String(DEFAULT_INSTRUCTIONS.to_string()),
    );
}

fn is_missing_or_empty_instructions(payload_object: &serde_json::Map<String, Value>) -> bool {
    match payload_object.get("instructions") {
        Some(Value::String(value)) => value.trim().is_empty(),
        Some(_) => true,
        None => true,
    }
}

fn collect_text_from_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.trim().to_string(),
        Value::Array(items) => items
            .iter()
            .map(collect_text_from_value)
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        Value::Object(object) => {
            if let Some(text) = object.get("text").and_then(Value::as_str) {
                return text.trim().to_string();
            }
            if let Some(content) = object.get("content") {
                return collect_text_from_value(content);
            }
            String::new()
        }
        _ => String::new(),
    }
}

fn parse_model_effort_alias(model_id: &str) -> Option<(String, String)> {
    if let Some((base, effort)) = parse_parenthesized_effort_alias(model_id) {
        return Some((base.to_string(), effort.to_string()));
    }

    for suffix in ["-xhigh", "-high", "-medium", "-low", "-minimal", "-none"] {
        if let Some(base) = model_id.strip_suffix(suffix) {
            let effort = &suffix[1..];
            if !base.is_empty() && is_reasoning_effort(effort) {
                return Some((base.to_string(), effort.to_string()));
            }
        }
    }

    None
}

fn parse_parenthesized_effort_alias(model_id: &str) -> Option<(&str, &str)> {
    if !model_id.ends_with(')') {
        return None;
    }

    let open_index = model_id.rfind('(')?;
    if open_index == 0 || open_index >= model_id.len().saturating_sub(2) {
        return None;
    }

    let base = &model_id[..open_index];
    let effort = &model_id[open_index + 1..model_id.len() - 1];
    if base.is_empty() || !is_reasoning_effort(effort) {
        return None;
    }

    Some((base, effort))
}

fn is_reasoning_effort(effort: &str) -> bool {
    matches!(
        effort,
        "none" | "minimal" | "low" | "medium" | "high" | "xhigh"
    )
}

fn plain_text_response(status: u16, message: &str) -> HttpResponse {
    HttpResponse {
        status,
        headers: vec![(
            "Content-Type".to_string(),
            "text/plain; charset=utf-8".to_string(),
        )],
        body: message.as_bytes().to_vec(),
    }
}

fn retry_after_response(status: u16, message: &str, seconds: u64) -> HttpResponse {
    HttpResponse {
        status,
        headers: vec![
            (
                "Content-Type".to_string(),
                "text/plain; charset=utf-8".to_string(),
            ),
            ("Retry-After".to_string(), seconds.to_string()),
        ],
        body: message.as_bytes().to_vec(),
    }
}

fn json_response(status: u16, payload: serde_json::Value) -> HttpResponse {
    let body = serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec());
    HttpResponse {
        status,
        headers: vec![(
            "Content-Type".to_string(),
            "application/json; charset=utf-8".to_string(),
        )],
        body,
    }
}

fn openai_error_response(status: u16, message: impl Into<String>, error_type: &str) -> HttpResponse {
    json_response(
        status,
        json!({
            "error": {
                "message": message.into(),
                "type": error_type,
            }
        }),
    )
}

fn anthropic_error_response(
    status: u16,
    message: impl Into<String>,
    error_type: &str,
) -> HttpResponse {
    json_response(
        status,
        json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message.into(),
            }
        }),
    )
}

fn is_authorized(request: &HttpRequest, expected_api_key: &str) -> bool {
    if expected_api_key.is_empty() {
        return true;
    }

    let provided = if let Some(value) = request.headers.get("authorization") {
        if value.to_ascii_lowercase().starts_with("bearer ") {
            value[7..].trim().to_string()
        } else {
            String::new()
        }
    } else {
        request
            .headers
            .get("x-api-key")
            .cloned()
            .unwrap_or_default()
    };

    secure_eq(provided.trim(), expected_api_key)
}

fn secure_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut diff = 0_u8;
    for (a, b) in left.as_bytes().iter().zip(right.as_bytes().iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

fn should_skip_upstream_header(lower_name: &str) -> bool {
    lower_name == "host"
        || lower_name == "content-length"
        || lower_name == "authorization"
        || lower_name == "x-api-key"
        || HOP_BY_HOP_HEADERS.contains(&lower_name)
}

fn should_skip_response_header(lower_name: &str) -> bool {
    lower_name == "content-length"
        || lower_name == "connection"
        || HOP_BY_HOP_HEADERS.contains(&lower_name)
}

fn client_identifier(request: &HttpRequest, peer_addr: Option<SocketAddr>) -> String {
    if let Some(value) = request.headers.get("x-forwarded-for") {
        let first = value.split(',').next().unwrap_or_default().trim();
        if !first.is_empty() {
            return first.to_string();
        }
    }

    peer_addr
        .map(|addr| addr.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        431 => "Request Header Fields Too Large",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn now_millis() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_parenthesized_effort_alias() {
        assert_eq!(
            parse_model_effort_alias("gpt-5.2(xhigh)"),
            Some(("gpt-5.2".to_string(), "xhigh".to_string()))
        );
    }

    #[test]
    fn parses_dash_effort_alias() {
        assert_eq!(
            parse_model_effort_alias("gpt-5.2-high"),
            Some(("gpt-5.2".to_string(), "high".to_string()))
        );
    }

    #[test]
    fn does_not_treat_canonical_model_as_alias() {
        assert_eq!(parse_model_effort_alias("gpt-5.1-codex-max"), None);
    }

    #[test]
    fn normalizes_model_alias_and_injects_reasoning() {
        let body = serde_json::to_vec(&json!({"model":"gpt-5.2(xhigh)", "input": "hi"}))
            .expect("body serialization should succeed");
        let normalized = normalize_model_alias_in_request_body(body);
        let payload: Value =
            serde_json::from_slice(&normalized).expect("normalized payload must be valid json");

        assert_eq!(
            payload.get("model").and_then(Value::as_str),
            Some("gpt-5.2")
        );
        assert_eq!(
            payload
                .get("reasoning")
                .and_then(Value::as_object)
                .and_then(|reasoning| reasoning.get("effort"))
                .and_then(Value::as_str),
            Some("xhigh")
        );
    }

    #[test]
    fn model_alias_effort_overrides_existing_reasoning_effort() {
        let body = serde_json::to_vec(&json!({
            "model": "gpt-5.2(high)",
            "reasoning": {"effort": "low"}
        }))
        .expect("body serialization should succeed");
        let normalized = normalize_model_alias_in_request_body(body);
        let payload: Value =
            serde_json::from_slice(&normalized).expect("normalized payload must be valid json");

        assert_eq!(
            payload.get("model").and_then(Value::as_str),
            Some("gpt-5.2")
        );
        assert_eq!(
            payload
                .get("reasoning")
                .and_then(Value::as_object)
                .and_then(|reasoning| reasoning.get("effort"))
                .and_then(Value::as_str),
            Some("high")
        );
    }

    #[test]
    fn injects_required_codex_defaults() {
        let body = serde_json::to_vec(&json!({
            "model": "gpt-5.3-codex",
            "input": "hello"
        }))
        .expect("body serialization should succeed");

        let normalized = normalize_model_alias_in_request_body(body);
        let payload: Value =
            serde_json::from_slice(&normalized).expect("normalized payload must be valid json");

        assert_eq!(payload.get("store").and_then(Value::as_bool), Some(false));
        assert_eq!(payload.get("stream").and_then(Value::as_bool), Some(true));
        assert_eq!(
            payload.get("instructions").and_then(Value::as_str),
            Some(DEFAULT_INSTRUCTIONS)
        );

        let input = payload
            .get("input")
            .and_then(Value::as_array)
            .expect("input should be normalized into an array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0].get("role").and_then(Value::as_str), Some("user"));
        assert_eq!(
            input[0]
                .get("content")
                .and_then(Value::as_array)
                .and_then(|content| content.first())
                .and_then(|part| part.get("type"))
                .and_then(Value::as_str),
            Some("input_text")
        );
    }

    #[test]
    fn converts_chat_messages_into_codex_input() {
        let body = serde_json::to_vec(&json!({
            "model": "gpt-5.3-codex",
            "messages": [
                {"role": "system", "content": "Follow the repo conventions"},
                {"role": "user", "content": "Say hi"}
            ]
        }))
        .expect("body serialization should succeed");

        let normalized = normalize_model_alias_in_request_body(body);
        let payload: Value =
            serde_json::from_slice(&normalized).expect("normalized payload must be valid json");

        assert!(payload.get("messages").is_none());
        assert_eq!(
            payload.get("instructions").and_then(Value::as_str),
            Some("Follow the repo conventions")
        );
        assert_eq!(
            payload
                .get("input")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(|item| item.get("role"))
                .and_then(Value::as_str),
            Some("user")
        );
    }

    #[test]
    fn normalizes_chat_tools_schema() {
        let body = serde_json::to_vec(&json!({
            "model": "gpt-5.3-codex",
            "messages": [{"role": "user", "content": "Call ping once"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Return pong",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
                }
            }],
            "tool_choice": "auto"
        }))
        .expect("body serialization should succeed");

        let normalized = normalize_model_alias_in_request_body(body);
        let payload: Value =
            serde_json::from_slice(&normalized).expect("normalized payload must be valid json");

        let tool = payload
            .get("tools")
            .and_then(Value::as_array)
            .and_then(|tools| tools.first())
            .and_then(Value::as_object)
            .expect("tools[0] should exist");

        assert!(tool.get("function").is_none());
        assert_eq!(tool.get("name").and_then(Value::as_str), Some("ping"));
        assert_eq!(
            tool.get("description").and_then(Value::as_str),
            Some("Return pong")
        );
    }

    #[test]
    fn normalizes_chat_tool_choice_schema() {
        let body = serde_json::to_vec(&json!({
            "model": "gpt-5.3-codex",
            "input": "Call ping once",
            "tools": [{
                "type": "function",
                "name": "ping",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
            }],
            "tool_choice": {"type": "function", "function": {"name": "ping"}}
        }))
        .expect("body serialization should succeed");

        let normalized = normalize_model_alias_in_request_body(body);
        let payload: Value =
            serde_json::from_slice(&normalized).expect("normalized payload must be valid json");

        let tool_choice = payload
            .get("tool_choice")
            .and_then(Value::as_object)
            .expect("tool_choice should be an object");

        assert!(tool_choice.get("function").is_none());
        assert_eq!(
            tool_choice.get("type").and_then(Value::as_str),
            Some("function")
        );
        assert_eq!(
            tool_choice.get("name").and_then(Value::as_str),
            Some("ping")
        );
    }

    #[test]
    fn normalizes_assistant_and_tool_message_roles() {
        let body = serde_json::to_vec(&json!({
            "model": "gpt-5.3-codex",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Calling ping",
                    "tool_calls": [
                        {"id": "call_123", "type": "function", "function": {"name": "ping", "arguments": "{}"}}
                    ]
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "pong"},
                {"role": "user", "content": "continue"}
            ]
        }))
        .expect("body serialization should succeed");

        let normalized = normalize_model_alias_in_request_body(body);
        let payload: Value =
            serde_json::from_slice(&normalized).expect("normalized payload must be valid json");

        let input = payload
            .get("input")
            .and_then(Value::as_array)
            .expect("input should be normalized into an array");

        assert_eq!(input.len(), 4);

        assert_eq!(
            input[0].get("type").and_then(Value::as_str),
            Some("function_call")
        );
        assert_eq!(input[0].get("name").and_then(Value::as_str), Some("ping"));
        assert_eq!(
            input[0].get("call_id").and_then(Value::as_str),
            Some("call_123")
        );

        assert_eq!(
            input[1].get("role").and_then(Value::as_str),
            Some("assistant")
        );
        assert_eq!(
            input[1]
                .get("content")
                .and_then(Value::as_array)
                .and_then(|parts| parts.first())
                .and_then(|part| part.get("type"))
                .and_then(Value::as_str),
            Some("output_text")
        );

        assert_eq!(
            input[2].get("type").and_then(Value::as_str),
            Some("function_call_output")
        );
        assert_eq!(
            input[2].get("call_id").and_then(Value::as_str),
            Some("call_123")
        );
        assert_eq!(input[2].get("output").and_then(Value::as_str), Some("pong"));

        assert_eq!(input[3].get("role").and_then(Value::as_str), Some("user"));
        assert_eq!(
            input[3]
                .get("content")
                .and_then(Value::as_array)
                .and_then(|parts| parts.first())
                .and_then(|part| part.get("type"))
                .and_then(Value::as_str),
            Some("input_text")
        );
    }

    #[test]
    fn drops_orphan_tool_messages_without_prior_tool_call() {
        let body = serde_json::to_vec(&json!({
            "model": "gpt-5.3-codex",
            "messages": [
                {"role": "tool", "tool_call_id": "call_404", "content": "orphan output"},
                {"role": "user", "content": "continue"}
            ]
        }))
        .expect("body serialization should succeed");

        let normalized = normalize_model_alias_in_request_body(body);
        let payload: Value =
            serde_json::from_slice(&normalized).expect("normalized payload must be valid json");

        let input = payload
            .get("input")
            .and_then(Value::as_array)
            .expect("input should be normalized into an array");

        assert_eq!(input.len(), 1);
        assert_eq!(input[0].get("role").and_then(Value::as_str), Some("user"));
        assert_eq!(
            input[0]
                .get("content")
                .and_then(Value::as_array)
                .and_then(|parts| parts.first())
                .and_then(|part| part.get("type"))
                .and_then(Value::as_str),
            Some("input_text")
        );
    }

    #[test]
    fn preserves_typed_input_items() {
        let body = serde_json::to_vec(&json!({
            "model": "gpt-5.3-codex",
            "input": [{"type": "function_call_output", "call_id": "call_abc", "output": "done"}]
        }))
        .expect("body serialization should succeed");

        let normalized = normalize_model_alias_in_request_body(body);
        let payload: Value =
            serde_json::from_slice(&normalized).expect("normalized payload must be valid json");

        let item = payload
            .get("input")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .and_then(Value::as_object)
            .expect("first input item should be an object");

        assert_eq!(
            item.get("type").and_then(Value::as_str),
            Some("function_call_output")
        );
        assert_eq!(
            item.get("call_id").and_then(Value::as_str),
            Some("call_abc")
        );
        assert_eq!(item.get("output").and_then(Value::as_str), Some("done"));
        assert!(item.get("role").is_none());
    }

    #[test]
    fn strips_unsupported_chat_style_fields() {
        let body = serde_json::to_vec(&json!({
            "model": "gpt-5.3-codex",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Return pong",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
                }
            }],
            "tool_choice": "auto",
            "stream_options": {"include_usage": true},
            "temperature": 1,
            "top_p": 1,
            "max_tokens": 512,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "response_format": {"type": "json_object"},
            "metadata": {"trace_id": "abc"}
        }))
        .expect("body serialization should succeed");

        let normalized = normalize_model_alias_in_request_body(body);
        let payload: Value =
            serde_json::from_slice(&normalized).expect("normalized payload must be valid json");
        let object = payload
            .as_object()
            .expect("normalized payload should remain an object");

        assert!(object.contains_key("model"));
        assert!(object.contains_key("input"));
        assert!(object.contains_key("tools"));
        assert!(object.contains_key("tool_choice"));
        assert!(object.contains_key("instructions"));
        assert!(object.contains_key("stream"));
        assert!(object.contains_key("store"));

        assert!(!object.contains_key("messages"));
        assert!(!object.contains_key("stream_options"));
        assert!(!object.contains_key("temperature"));
        assert!(!object.contains_key("top_p"));
        assert!(!object.contains_key("max_tokens"));
        assert!(!object.contains_key("frequency_penalty"));
        assert!(!object.contains_key("presence_penalty"));
        assert!(!object.contains_key("response_format"));
        assert!(!object.contains_key("metadata"));
    }

    #[test]
    fn detects_stream_preference_flag() {
        assert!(request_prefers_stream(br#"{"stream":true}"#));
        assert!(!request_prefers_stream(br#"{"stream":false}"#));
        assert!(!request_prefers_stream(br#"{"model":"gpt-5.3-codex"}"#));
    }

    #[test]
    fn detects_sse_payloads() {
        assert!(is_sse_payload(b"event: response.created\n"));
        assert!(is_sse_payload(
            b"\n\ndata: {\"type\":\"response.created\"}\n"
        ));
        assert!(!is_sse_payload(b"{\"object\":\"response\"}"));
    }

    #[test]
    fn extracts_final_response_object_from_sse() {
        let sse = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\"}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"output\":[]}}\n\n"
        );

        let extracted =
            extract_response_object_from_sse(sse.as_bytes()).expect("should decode response");
        let payload: Value = serde_json::from_slice(&extracted).expect("valid decoded json");

        assert_eq!(payload.get("id").and_then(Value::as_str), Some("resp_1"));
        assert_eq!(
            payload.get("object").and_then(Value::as_str),
            Some("response")
        );
        assert_eq!(
            payload.get("status").and_then(Value::as_str),
            Some("completed")
        );
    }

    #[test]
    fn parses_acp_model_route_without_runtime_override() {
        let route = parse_acp_model_route("acp:codex").expect("route should parse");

        assert_eq!(route.agent, AgentKind::Codex);
        assert_eq!(route.requested_model, "acp:codex");
        assert_eq!(route.runtime_model, None);
    }

    #[test]
    fn parses_acp_model_route_with_runtime_override() {
        let route = parse_acp_model_route("acp:claude:sonnet-4.5")
            .expect("route should parse");

        assert_eq!(route.agent, AgentKind::Claude);
        assert_eq!(route.requested_model, "acp:claude:sonnet-4.5");
        assert_eq!(route.runtime_model.as_deref(), Some("sonnet-4.5"));
    }

    #[test]
    fn ignores_unknown_acp_agent_models() {
        assert!(parse_acp_model_route("acp:unknown").is_none());
    }

    #[test]
    fn builds_acp_prompt_text_from_messages_and_instructions() {
        let payload = json!({
            "instructions": "Follow repo conventions",
            "messages": [
                {"role": "system", "content": "Be terse"},
                {"role": "user", "content": "Reply with OK"},
                {"role": "assistant", "content": "Working on it"}
            ]
        });

        let prompt = build_acp_prompt_text(&payload);

        assert!(prompt.contains("Instructions:\nFollow repo conventions"));
        assert!(prompt.contains("Conversation:\nSystem: Be terse"));
        assert!(prompt.contains("User: Reply with OK"));
        assert!(prompt.contains("Assistant: Working on it"));
    }

    #[test]
    fn resolves_relative_acp_workspace_header_against_default_workspace() {
        let workspace_root = std::env::temp_dir().join(format!("opengateway-acp-workspace-{}", now_millis()));
        let nested = workspace_root.join(".factory");
        std::fs::create_dir_all(&nested).expect("workspace fixture should be created");
        let request = HttpRequest {
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            headers: HashMap::from([(
                ACP_WORKSPACE_HEADER.to_string(),
                ".factory".to_string(),
            )]),
            header_list: vec![(
                ACP_WORKSPACE_HEADER.to_string(),
                ".factory".to_string(),
            )],
            body: Vec::new(),
        };

        let resolved = resolve_acp_workspace(&request, Some(&workspace_root))
            .expect("workspace should resolve");

        assert_eq!(resolved, nested);
        let _ = std::fs::remove_dir_all(workspace_root);
    }

    #[test]
    fn classifies_anthropic_messages_surface() {
        assert_eq!(
            classify_acp_http_surface("/v1/messages").expect("surface should parse"),
            AcpHttpSurface::AnthropicMessages
        );
    }

    #[test]
    fn builds_anthropic_acp_response_shape() {
        let response = build_acp_http_response(
            AcpHttpSurface::AnthropicMessages,
            &AcpResponseMeta {
                response_id: "resp_1".to_string(),
                message_id: "msg_1".to_string(),
                model: "acp:claude:claude-sonnet-4-6".to_string(),
                created: 123,
            },
            "Hello",
            "end_turn",
        );

        let payload: Value =
            serde_json::from_slice(&response.body).expect("anthropic payload should decode");
        assert_eq!(payload.get("type").and_then(Value::as_str), Some("message"));
        assert_eq!(payload.get("role").and_then(Value::as_str), Some("assistant"));
        assert_eq!(
            payload
                .get("content")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(|item| item.get("type"))
                .and_then(Value::as_str),
            Some("text")
        );
        assert_eq!(
            payload
                .get("content")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(|item| item.get("text"))
                .and_then(Value::as_str),
            Some("Hello")
        );
    }
}
