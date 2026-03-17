pub fn redact_text(input: &str) -> String {
    let mut output = input.to_string();
    for prefix in [
        "Authorization: Bearer ",
        "authorization: Bearer ",
        "access_token=",
        "refresh_token=",
        "\"access_token\":\"",
        "\"refresh_token\":\"",
        "\"apiKey\":\"",
        "api_key=",
    ] {
        output = redact_after_prefix(&output, prefix);
    }
    redact_sk_tokens(&output)
}

fn redact_after_prefix(input: &str, prefix: &str) -> String {
    let mut remaining = input;
    let mut output = String::with_capacity(input.len());

    while let Some(index) = remaining.find(prefix) {
        output.push_str(&remaining[..index + prefix.len()]);
        remaining = &remaining[index + prefix.len()..];

        let end = remaining
            .find(|ch: char| ch.is_whitespace() || ch == '"' || ch == '\'' || ch == ',' || ch == '&')
            .unwrap_or(remaining.len());

        output.push_str("[REDACTED]");
        output.push_str(&remaining[end..]);
        remaining = "";
    }

    if output.is_empty() {
        input.to_string()
    } else {
        output
    }
}

fn redact_sk_tokens(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == 's' && chars.peek() == Some(&'k') {
            let mut candidate = String::from(ch);
            while let Some(next) = chars.peek().copied() {
                if next.is_ascii_alphanumeric() || next == '-' || next == '_' {
                    candidate.push(next);
                    chars.next();
                    continue;
                }
                break;
            }
            if candidate.starts_with("sk-") && candidate.len() > 6 {
                output.push_str("[REDACTED]");
                continue;
            }
            output.push_str(&candidate);
            continue;
        }
        output.push(ch);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::redact_text;

    #[test]
    fn redacts_bearer_and_refresh_tokens() {
        let input =
            "Authorization: Bearer abc123 refresh_token=xyz456 \"apiKey\":\"token-value\"";
        let output = redact_text(input);
        assert!(!output.contains("abc123"));
        assert!(!output.contains("xyz456"));
        assert!(!output.contains("token-value"));
        assert!(output.contains("[REDACTED]"));
    }

    #[test]
    fn redacts_sk_style_tokens() {
        let output = redact_text("use sk-test-secret-here please");
        assert_eq!(output, "use [REDACTED] please");
    }
}
