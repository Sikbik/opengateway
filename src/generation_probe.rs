use anyhow::{anyhow, Context, Result};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use std::time::Duration;

pub const PROBE_PROMPT: &str = "Reply with exactly: ok";

pub struct ProbeConfig<'a> {
    pub host: &'a str,
    pub port: u16,
    pub api_key: &'a str,
    pub model: &'a str,
    pub timeout: Duration,
}

pub fn run_probe(config: ProbeConfig<'_>) -> Result<String> {
    let base_url = format!("http://{}:{}", config.host, config.port);
    let url = format!("{base_url}/v1/responses");
    let client = Client::builder()
        .timeout(config.timeout)
        .build()
        .context("failed to create HTTP client for generation probe")?;

    let response = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", config.api_key))
        .json(&build_probe_request(config.model))
        .send()
        .with_context(|| format!("failed request to {url}"))?;

    let status = response.status();
    let body = response
        .text()
        .context("failed to read generation probe response")?;
    if !status.is_success() {
        return Err(anyhow!(
            "generation probe failed with status {}: {}",
            status,
            compact_text(&body, 360)
        ));
    }

    let payload: Value =
        serde_json::from_str(&body).context("failed to decode generation probe response")?;
    let output = extract_response_output_text(&payload)
        .ok_or_else(|| anyhow!("generation probe completed but response text was empty"))?;

    Ok(success_summary(
        config.host,
        config.port,
        config.model,
        output.as_str(),
    ))
}

pub fn resolve_probe_model(explicit: &str, factory_settings_path: &Path) -> Result<String> {
    let explicit = explicit.trim();
    if !explicit.is_empty() {
        return Ok(explicit.to_string());
    }

    let raw = fs::read_to_string(factory_settings_path).with_context(|| {
        format!(
            "failed to read Factory settings from {}",
            factory_settings_path.display()
        )
    })?;
    let settings: Value = serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to decode Factory settings from {}",
            factory_settings_path.display()
        )
    })?;

    factory_session_default_model(&settings).ok_or_else(|| {
        anyhow!("Factory session default model is not set; run `opengateway sync-factory` first")
    })
}

fn build_probe_request(model: &str) -> Value {
    json!({
        "model": model,
        "input": PROBE_PROMPT,
    })
}

fn factory_session_default_model(settings: &Value) -> Option<String> {
    settings
        .get("sessionDefaultSettings")
        .and_then(|value| value.get("model"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn extract_response_output_text(response: &Value) -> Option<String> {
    if let Some(text) = response
        .get("output_text")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(text.to_string());
    }

    let mut chunks = Vec::new();
    for item in response.get("output").and_then(Value::as_array)? {
        if let Some(text) = item
            .get("text")
            .and_then(Value::as_str)
            .or_else(|| item.get("output_text").and_then(Value::as_str))
        {
            chunks.push(text);
        }

        if let Some(content) = item.get("content").and_then(Value::as_array) {
            for part in content {
                if let Some(text) = part
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.get("output_text").and_then(Value::as_str))
                {
                    chunks.push(text);
                }
            }
        }
    }

    let joined = chunks.join("");
    let trimmed = joined.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn success_summary(host: &str, port: u16, model: &str, output: &str) -> String {
    format!(
        "Generation probe passed on {host}:{port}. - model: {model} - /v1/responses: ok - output: {}",
        compact_text(output, 160)
    )
}

fn compact_text(value: &str, max_chars: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        return compact;
    }

    compact.chars().take(max_chars).collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn probe_request_uses_model_and_avoids_token_limit_fields() {
        let body = build_probe_request("custom:GPT-5.4-(XHigh)-4");

        assert_eq!(
            body.get("model").and_then(Value::as_str),
            Some("custom:GPT-5.4-(XHigh)-4")
        );
        assert_eq!(
            body.get("input").and_then(Value::as_str),
            Some(PROBE_PROMPT)
        );
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("max_output_tokens").is_none());
    }

    #[test]
    fn extracts_text_from_responses_output_content() {
        let response = json!({
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "ok"}
                    ]
                }
            ]
        });

        assert_eq!(
            extract_response_output_text(&response).as_deref(),
            Some("ok")
        );
    }

    #[test]
    fn resolves_factory_session_default_model() {
        let settings = json!({
            "sessionDefaultSettings": {
                "model": "custom:GPT-5.4-(XHigh)-4"
            }
        });

        assert_eq!(
            factory_session_default_model(&settings).as_deref(),
            Some("custom:GPT-5.4-(XHigh)-4")
        );
    }

    #[test]
    fn success_summary_keeps_output_compact() {
        let summary = success_summary(
            "127.0.0.1",
            42069,
            "custom:GPT-5.4-(XHigh)-4",
            "ok\nwith extra words",
        );

        assert!(summary.contains("Generation probe passed on 127.0.0.1:42069."));
        assert!(summary.contains("- model: custom:GPT-5.4-(XHigh)-4"));
        assert!(summary.contains("- /v1/responses: ok"));
        assert!(summary.contains("- output: ok with extra words"));
        assert!(!summary.contains('\n'));
    }
}
