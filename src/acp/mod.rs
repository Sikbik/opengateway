pub mod adapters;
pub mod capabilities;
pub mod cli;
pub mod doctor;
pub mod journal;
pub mod redact;
pub mod snapshot;
pub use adapters::claude::{command_claude_runtime, ClaudeRuntimeArgs};
pub use adapters::codex::{command_codex_runtime, CodexRuntimeArgs};
pub use supervisor::{command_mock_runtime, MockRuntimeArgs};

mod errors;
mod protocol;
mod session;
mod supervisor;
mod transport;
