use anyhow::{anyhow, bail, Context, Result};
use base64::engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD};
use base64::Engine as _;
use rand::Rng;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use url::{form_urlencoded, Url};

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const ISSUER: &str = "https://auth.openai.com";
const OAUTH_CALLBACK_PORT: u16 = 1455;
const OAUTH_CALLBACK_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const OAUTH_POLLING_SAFETY_MARGIN: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, Copy)]
pub enum LoginMode {
    Browser,
    Headless,
}

#[derive(Debug, Clone)]
pub struct OAuthLoginResult {
    pub refresh_token: String,
    pub access_token: String,
    pub expires_at_ms: i64,
    pub account_id: Option<String>,
}

pub fn login(mode: LoginMode, no_browser: bool, verbose: bool) -> Result<OAuthLoginResult> {
    let client = Client::builder()
        .timeout(Duration::from_secs(45))
        .build()
        .context("failed to initialize OAuth HTTP client")?;

    match mode {
        LoginMode::Browser => login_browser(&client, no_browser, verbose),
        LoginMode::Headless => login_headless(&client, verbose),
    }
}

pub fn refresh_access_token(
    refresh_token: &str,
    account_hint: Option<&str>,
) -> Result<OAuthLoginResult> {
    let client = Client::builder()
        .timeout(Duration::from_secs(45))
        .build()
        .context("failed to initialize OAuth HTTP client")?;

    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "refresh_token")
        .append_pair("refresh_token", refresh_token)
        .append_pair("client_id", CLIENT_ID)
        .finish();

    let response = client
        .post(format!("{ISSUER}/oauth/token"))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .context("failed requesting refresh token")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        bail!("oauth token refresh failed ({status}): {body}");
    }

    let token_response: TokenResponse = response
        .json()
        .context("failed to decode refresh token response")?;
    build_login_result(token_response, Some(refresh_token), account_hint)
}

fn login_browser(client: &Client, no_browser: bool, verbose: bool) -> Result<OAuthLoginResult> {
    let pkce_verifier = generate_pkce_verifier(43);
    let pkce_challenge = generate_pkce_challenge(&pkce_verifier);
    let state = generate_state();

    let (listener, port) = bind_callback_listener()?;
    let redirect_uri = format!("http://localhost:{port}/auth/callback");
    let auth_url = build_authorize_url(&redirect_uri, &pkce_challenge, &state)?;

    println!("Starting browser authorization...");
    println!("Open this URL: {auth_url}");

    if !no_browser {
        if let Err(err) = open_browser(&auth_url) {
            eprintln!("Could not auto-open browser ({err}). Continue manually with the URL above.");
        }
    }

    let state_for_callback = state.clone();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = wait_for_callback(listener, &state_for_callback);
        let _ = tx.send(result);
    });

    let authorization_code = rx
        .recv_timeout(OAUTH_CALLBACK_TIMEOUT)
        .map_err(|_| anyhow!("OAuth callback timeout after 5 minutes"))??;

    let tokens =
        exchange_code_for_tokens(client, &authorization_code, &redirect_uri, &pkce_verifier)?;
    if verbose {
        eprintln!("Browser OAuth succeeded.");
    }
    build_login_result(tokens, None, None)
}

fn login_headless(client: &Client, verbose: bool) -> Result<OAuthLoginResult> {
    println!("Starting device authorization...");

    let device_response = client
        .post(format!("{ISSUER}/api/accounts/deviceauth/usercode"))
        .header("Content-Type", "application/json")
        .header(
            "User-Agent",
            format!("opengateway/{}", env!("CARGO_PKG_VERSION")),
        )
        .json(&serde_json::json!({ "client_id": CLIENT_ID }))
        .send()
        .context("failed to initiate device authorization")?;

    if !device_response.status().is_success() {
        let status = device_response.status();
        let body = device_response.text().unwrap_or_default();
        bail!("device authorization start failed ({status}): {body}");
    }

    let device_data: DeviceCodeResponse = device_response
        .json()
        .context("failed to decode device authorization response")?;
    let poll_interval = parse_poll_interval(device_data.interval.as_deref());

    println!("Open this URL: {ISSUER}/codex/device");
    println!("Enter code: {}", device_data.user_code);

    let deadline = Instant::now() + Duration::from_secs(15 * 60);
    loop {
        if Instant::now() >= deadline {
            bail!("device authorization timed out");
        }

        let poll_response = client
            .post(format!("{ISSUER}/api/accounts/deviceauth/token"))
            .header("Content-Type", "application/json")
            .header(
                "User-Agent",
                format!("opengateway/{}", env!("CARGO_PKG_VERSION")),
            )
            .json(&serde_json::json!({
                "device_auth_id": device_data.device_auth_id,
                "user_code": device_data.user_code,
            }))
            .send()
            .context("failed polling device authorization")?;

        if poll_response.status().is_success() {
            let device_token: DeviceTokenResponse = poll_response
                .json()
                .context("failed to decode device token response")?;

            let tokens = exchange_code_for_tokens(
                client,
                &device_token.authorization_code,
                &format!("{ISSUER}/deviceauth/callback"),
                &device_token.code_verifier,
            )?;

            if verbose {
                eprintln!("Headless OAuth succeeded.");
            }
            return build_login_result(tokens, None, None);
        }

        let status = poll_response.status();
        let body = poll_response.text().unwrap_or_default();
        let lower_body = body.to_lowercase();

        if status.as_u16() == 403
            || status.as_u16() == 404
            || (status.as_u16() == 400
                && (lower_body.contains("authorization_pending")
                    || lower_body.contains("slow_down")))
        {
            thread::sleep(poll_interval);
            continue;
        }

        bail!("device authorization poll failed ({status}): {body}");
    }
}

fn bind_callback_listener() -> Result<(TcpListener, u16)> {
    for port in OAUTH_CALLBACK_PORT..(OAUTH_CALLBACK_PORT + 25) {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(listener) => return Ok((listener, port)),
            Err(_) => continue,
        }
    }
    bail!("could not bind OAuth callback listener on localhost ports 1455-1479")
}

fn wait_for_callback(listener: TcpListener, expected_state: &str) -> Result<String> {
    let (mut stream, _) = listener
        .accept()
        .context("failed waiting for OAuth callback connection")?;

    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .context("failed to set callback read timeout")?;

    let mut buffer = [0_u8; 8192];
    let bytes_read = stream
        .read(&mut buffer)
        .context("failed reading OAuth callback request")?;
    if bytes_read == 0 {
        bail!("empty OAuth callback request");
    }

    let request = String::from_utf8_lossy(&buffer[..bytes_read]);
    let request_line = request.lines().next().unwrap_or_default();
    let target = request_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow!("invalid OAuth callback request line"))?;

    let callback_url = Url::parse(&format!("http://localhost{target}"))
        .context("failed to parse OAuth callback URL")?;

    let params: HashMap<String, String> = callback_url.query_pairs().into_owned().collect();

    if let Some(error) = params.get("error") {
        let detail = params
            .get("error_description")
            .cloned()
            .unwrap_or_else(|| error.clone());
        write_html_response(&mut stream, 400, "Authorization Failed", &detail)?;
        bail!("oauth authorization failed: {detail}");
    }

    let state = params.get("state").cloned().unwrap_or_default();
    if state != expected_state {
        write_html_response(
            &mut stream,
            400,
            "Authorization Failed",
            "Invalid state value (possible CSRF or stale callback).",
        )?;
        bail!("oauth callback state mismatch");
    }

    let code = params
        .get("code")
        .cloned()
        .ok_or_else(|| anyhow!("oauth callback missing authorization code"))?;

    write_html_response(
        &mut stream,
        200,
        "Authorization Successful",
        "You can close this tab and return to opengateway.",
    )?;

    Ok(code)
}

fn write_html_response(
    stream: &mut TcpStream,
    status: u16,
    title: &str,
    message: &str,
) -> Result<()> {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        _ => "Error",
    };

    let escaped_message = html_escape(message);
    let body = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>{title}</title></head><body><h1>{title}</h1><p>{escaped_message}</p></body></html>"
    );

    let response = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );

    stream
        .write_all(response.as_bytes())
        .context("failed writing OAuth callback response")?;
    stream.flush().ok();
    Ok(())
}

fn build_authorize_url(redirect_uri: &str, code_challenge: &str, state: &str) -> Result<String> {
    let mut url =
        Url::parse(&format!("{ISSUER}/oauth/authorize")).context("invalid authorize URL")?;
    let mut qp = url.query_pairs_mut();
    qp.append_pair("response_type", "code");
    qp.append_pair("client_id", CLIENT_ID);
    qp.append_pair("redirect_uri", redirect_uri);
    qp.append_pair("scope", "openid profile email offline_access");
    qp.append_pair("code_challenge", code_challenge);
    qp.append_pair("code_challenge_method", "S256");
    qp.append_pair("id_token_add_organizations", "true");
    qp.append_pair("codex_cli_simplified_flow", "true");
    qp.append_pair("state", state);
    qp.append_pair("originator", "opengateway");
    drop(qp);
    Ok(url.to_string())
}

fn exchange_code_for_tokens(
    client: &Client,
    code: &str,
    redirect_uri: &str,
    code_verifier: &str,
) -> Result<TokenResponse> {
    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "authorization_code")
        .append_pair("code", code)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("client_id", CLIENT_ID)
        .append_pair("code_verifier", code_verifier)
        .finish();

    let response = client
        .post(format!("{ISSUER}/oauth/token"))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .context("failed requesting oauth token")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        bail!("oauth token exchange failed ({status}): {body}");
    }

    response
        .json::<TokenResponse>()
        .context("failed to decode oauth token response")
}

fn build_login_result(
    tokens: TokenResponse,
    fallback_refresh_token: Option<&str>,
    account_hint: Option<&str>,
) -> Result<OAuthLoginResult> {
    let expires_in_seconds = tokens.expires_in.unwrap_or(3600).max(60);
    let expires_at_ms = now_millis() + (expires_in_seconds * 1000);

    let refresh_token = tokens
        .refresh_token
        .or_else(|| fallback_refresh_token.map(ToOwned::to_owned))
        .ok_or_else(|| anyhow!("oauth token response missing refresh token"))?;

    let account_id = tokens
        .id_token
        .as_deref()
        .and_then(extract_account_id_from_jwt)
        .or_else(|| extract_account_id_from_jwt(tokens.access_token.as_str()))
        .or_else(|| account_hint.map(ToOwned::to_owned));

    Ok(OAuthLoginResult {
        refresh_token,
        access_token: tokens.access_token,
        expires_at_ms,
        account_id,
    })
}

fn parse_poll_interval(raw: Option<&str>) -> Duration {
    let seconds = raw
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(5)
        .max(1);
    Duration::from_secs(seconds) + OAUTH_POLLING_SAFETY_MARGIN
}

fn generate_pkce_verifier(length: usize) -> String {
    let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    let mut rng = rand::thread_rng();
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..chars.len());
            chars[idx] as char
        })
        .collect()
}

fn generate_pkce_challenge(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(digest)
}

fn generate_state() -> String {
    let mut bytes = [0_u8; 32];
    rand::thread_rng().fill(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn open_browser(url: &str) -> Result<()> {
    if cfg!(target_os = "linux") {
        let candidates = ["xdg-open", "gio", "gnome-open", "kde-open"];
        for cmd in candidates {
            let mut process = Command::new(cmd);
            if cmd == "gio" {
                process.arg("open").arg(url);
            } else {
                process.arg(url);
            }
            if process.status().is_ok() {
                return Ok(());
            }
        }
    }

    if cfg!(target_os = "macos") {
        if Command::new("open").arg(url).status().is_ok() {
            return Ok(());
        }
    }

    if cfg!(target_os = "windows") {
        if Command::new("cmd")
            .args(["/C", "start", "", url])
            .status()
            .is_ok()
        {
            return Ok(());
        }
    }

    bail!("no supported browser launcher found")
}

fn extract_account_id_from_jwt(token: &str) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    let decoded = decode_base64url(payload)?;
    let claims: Value = serde_json::from_slice(&decoded).ok()?;

    claims
        .get("chatgpt_account_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| {
            claims
                .get("https://api.openai.com/auth")
                .and_then(Value::as_object)
                .and_then(|obj| obj.get("chatgpt_account_id"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .or_else(|| {
            claims
                .get("organizations")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(Value::as_object)
                .and_then(|obj| obj.get("id"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
}

fn decode_base64url(payload: &str) -> Option<Vec<u8>> {
    URL_SAFE_NO_PAD.decode(payload).ok().or_else(|| {
        let mut padded = payload.to_string();
        while padded.len() % 4 != 0 {
            padded.push('=');
        }
        URL_SAFE.decode(padded.as_bytes()).ok()
    })
}

fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn now_millis() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_millis() as i64
}

#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    device_auth_id: String,
    user_code: String,
    interval: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeviceTokenResponse {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    id_token: Option<String>,
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_account_id_from_auth_claims() {
        let claims = serde_json::json!({
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_123"
            }
        });

        let encoded = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap());
        let token = format!("aaa.{encoded}.bbb");
        assert_eq!(
            extract_account_id_from_jwt(&token),
            Some("acct_123".to_string())
        );
    }

    #[test]
    fn parse_poll_interval_has_safety_margin() {
        assert_eq!(parse_poll_interval(Some("5")), Duration::from_secs(8));
        assert_eq!(parse_poll_interval(Some("0")), Duration::from_secs(4));
    }
}
