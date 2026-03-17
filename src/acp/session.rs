pub fn session_states() -> Vec<String> {
    vec![
        "starting".to_string(),
        "running".to_string(),
        "cancelling".to_string(),
        "exited".to_string(),
        "failed".to_string(),
    ]
}
