use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub const APP_NAME: &str = "opengateway";

#[derive(Debug, Clone)]
pub struct AppPaths {
    pub state_dir: PathBuf,
    pub log_file: PathBuf,
    pub pid_file: PathBuf,
    pub data_dir: PathBuf,
    pub config_dir: PathBuf,
    pub default_config: PathBuf,
    pub auth_dir: PathBuf,
    pub api_key_file: PathBuf,
}

pub fn build_paths() -> Result<AppPaths> {
    let home = home_dir()?;
    let xdg_state = env_path("XDG_STATE_HOME", &home.join(".local/state"), &home);
    let xdg_data = env_path("XDG_DATA_HOME", &home.join(".local/share"), &home);
    let xdg_config = env_path("XDG_CONFIG_HOME", &home.join(".config"), &home);

    let state_dir = env_path("OPENGATEWAY_STATE_DIR", &xdg_state.join(APP_NAME), &home);
    let data_dir = env_path("OPENGATEWAY_DATA_DIR", &xdg_data.join(APP_NAME), &home);
    let config_dir = env_path("OPENGATEWAY_CONFIG_DIR", &xdg_config.join(APP_NAME), &home);
    let auth_dir = env_path("OPENGATEWAY_AUTH_DIR", &config_dir.join("auth"), &home);

    Ok(AppPaths {
        log_file: state_dir.join("opengateway.log"),
        pid_file: state_dir.join("opengateway.pid"),
        default_config: config_dir.join("config.yaml"),
        api_key_file: config_dir.join("proxy_api_key"),
        state_dir,
        data_dir,
        config_dir,
        auth_dir,
    })
}

impl AppPaths {
    pub fn ensure_runtime_dirs(&self) -> Result<()> {
        fs::create_dir_all(&self.state_dir)
            .with_context(|| format!("failed to create state dir {}", self.state_dir.display()))?;
        fs::create_dir_all(&self.data_dir)
            .with_context(|| format!("failed to create data dir {}", self.data_dir.display()))?;
        fs::create_dir_all(&self.config_dir).with_context(|| {
            format!("failed to create config dir {}", self.config_dir.display())
        })?;
        fs::create_dir_all(&self.auth_dir)
            .with_context(|| format!("failed to create auth dir {}", self.auth_dir.display()))?;
        Ok(())
    }
}

fn home_dir() -> Result<PathBuf> {
    let raw = env::var("HOME").context("HOME environment variable is not set")?;
    Ok(PathBuf::from(raw))
}

fn env_path(name: &str, default: &Path, home: &Path) -> PathBuf {
    match env::var(name) {
        Ok(raw) if !raw.trim().is_empty() => expand_tilde(raw.trim(), home),
        _ => default.to_path_buf(),
    }
}

fn expand_tilde(raw: &str, home: &Path) -> PathBuf {
    if raw == "~" {
        return home.to_path_buf();
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return home.join(rest);
    }
    PathBuf::from(raw)
}
