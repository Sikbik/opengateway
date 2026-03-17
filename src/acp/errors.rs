pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INTERNAL_ERROR: i64 = -32603;
pub const SERVER_NOT_INITIALIZED: i64 = -32002;

pub fn error_categories() -> &'static [&'static str] {
    &[
        "parse-error",
        "invalid-request",
        "unsupported-method",
        "internal-error",
        "workspace-invalid",
        "runtime-missing",
        "auth-required",
        "session-crashed",
        "server-not-initialized",
    ]
}
