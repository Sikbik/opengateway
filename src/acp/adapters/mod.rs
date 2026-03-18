pub(crate) mod claude;
pub(crate) mod codex;

use clap::ValueEnum;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum AgentKind {
    Codex,
    Claude,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentDescriptor {
    pub kind: AgentKind,
    pub runtime_name: &'static str,
    pub status: &'static str,
    pub note: &'static str,
}

pub fn supported_agents() -> &'static [AgentDescriptor] {
    &[
        AgentDescriptor {
            kind: AgentKind::Codex,
            runtime_name: codex::RUNTIME_NAME,
            status: "mvp-session-runtime",
            note: codex::SCAFFOLD_NOTE,
        },
        AgentDescriptor {
            kind: AgentKind::Claude,
            runtime_name: claude::RUNTIME_NAME,
            status: "placeholder",
            note: claude::SCAFFOLD_NOTE,
        },
    ]
}

impl AgentKind {
    pub fn as_str(self) -> &'static str {
        match self {
            AgentKind::Codex => "codex",
            AgentKind::Claude => "claude",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{supported_agents, AgentKind};

    #[test]
    fn supported_agents_include_codex_first() {
        let agents = supported_agents();
        assert_eq!(agents[0].kind, AgentKind::Codex);
        assert_eq!(agents[0].runtime_name, "codex exec");
    }
}
