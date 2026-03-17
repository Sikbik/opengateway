#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PlannedCapabilities {
    pub initialize: bool,
    pub sessions: SessionCapabilities,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SessionCapabilities {
    pub new_session: bool,
    pub prompt: bool,
    pub cancel: bool,
    pub load_session: bool,
    pub streaming_updates: bool,
}

pub fn planned_capabilities() -> PlannedCapabilities {
    PlannedCapabilities {
        initialize: true,
        sessions: SessionCapabilities {
            new_session: true,
            prompt: true,
            cancel: true,
            load_session: false,
            streaming_updates: true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::planned_capabilities;

    #[test]
    fn planned_capabilities_keep_load_session_disabled() {
        let capabilities = planned_capabilities();
        assert!(capabilities.initialize);
        assert!(!capabilities.sessions.load_session);
    }
}
