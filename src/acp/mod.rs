pub mod adapters;
pub mod bridge;
pub mod capabilities;
pub mod cli;
pub mod doctor;
pub mod journal;
pub mod redact;
pub mod snapshot;
pub use bridge::{command_bridge_run, AcpBridgeRunArgs};
pub use adapters::claude::{command_claude_runtime, ClaudeRuntimeArgs};
pub use adapters::codex::{command_codex_runtime, CodexRuntimeArgs};
pub use supervisor::{command_mock_runtime, MockRuntimeArgs};

mod errors;
mod protocol;
mod session;
mod supervisor;
mod transport;
