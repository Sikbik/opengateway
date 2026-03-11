use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub const APP_NAME: &str = "opengateway";
const DEFAULT_FACTORY_DIR: &str = ".factory";

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

#[derive(Debug, Clone)]
pub struct FactoryPaths {
    pub home_dir: PathBuf,
    pub config_path: PathBuf,
    pub settings_path: PathBuf,
    pub machine_droids_dir: PathBuf,
}

pub fn build_paths() -> Result<AppPaths> {
    let home = home_dir()?;
    let state_root = default_state_root(&home);
    let data_root = default_data_root(&home);
    let config_root = default_config_root(&home);

    let xdg_state = env_path(state_env_name(), &state_root, &home);
    let xdg_data = env_path(data_env_name(), &data_root, &home);
    let xdg_config = env_path(config_env_name(), &config_root, &home);

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

pub fn build_factory_paths() -> Result<FactoryPaths> {
    let home = home_dir()?;
    Ok(factory_paths_from_env(&home, |name| env::var(name).ok()))
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

fn factory_paths_from_env<F>(home: &Path, env_lookup: F) -> FactoryPaths
where
    F: Fn(&str) -> Option<String>,
{
    let explicit_home = first_env_path(
        &["OPENGATEWAY_FACTORY_HOME", "FACTORY_HOME"],
        &env_lookup,
        home,
    );
    let explicit_config = first_env_path(&["OPENGATEWAY_FACTORY_CONFIG"], &env_lookup, home);
    let explicit_settings = first_env_path(&["OPENGATEWAY_FACTORY_SETTINGS"], &env_lookup, home);
    let explicit_droids = first_env_path(&["OPENGATEWAY_FACTORY_DROIDS_DIR"], &env_lookup, home);

    let home_dir = explicit_home
        .or_else(|| derive_factory_home(explicit_droids.as_deref()))
        .or_else(|| derive_factory_home(explicit_settings.as_deref()))
        .or_else(|| derive_factory_home(explicit_config.as_deref()))
        .unwrap_or_else(|| home.join(DEFAULT_FACTORY_DIR));

    FactoryPaths {
        config_path: explicit_config.unwrap_or_else(|| home_dir.join("config.json")),
        settings_path: explicit_settings.unwrap_or_else(|| home_dir.join("settings.json")),
        machine_droids_dir: explicit_droids.unwrap_or_else(|| home_dir.join("droids")),
        home_dir,
    }
}

pub fn home_dir() -> Result<PathBuf> {
    if let Some(path) = nonempty_env("HOME") {
        return Ok(PathBuf::from(path));
    }

    #[cfg(windows)]
    {
        if let Some(path) = nonempty_env("USERPROFILE") {
            return Ok(PathBuf::from(path));
        }

        if let (Some(drive), Some(path)) = (nonempty_env("HOMEDRIVE"), nonempty_env("HOMEPATH")) {
            return Ok(PathBuf::from(format!("{drive}{path}")));
        }
    }

    Err(anyhow::anyhow!(
        "home directory environment variable is not set"
    ))
}

fn env_path(name: &str, default: &Path, home: &Path) -> PathBuf {
    match env::var(name) {
        Ok(raw) if !raw.trim().is_empty() => expand_tilde(raw.trim(), home),
        _ => default.to_path_buf(),
    }
}

fn first_env_path<F>(names: &[&str], env_lookup: &F, home: &Path) -> Option<PathBuf>
where
    F: Fn(&str) -> Option<String>,
{
    names.iter().find_map(|name| {
        env_lookup(name)
            .filter(|raw| !raw.trim().is_empty())
            .map(|raw| expand_tilde(raw.trim(), home))
    })
}

fn derive_factory_home(path: Option<&Path>) -> Option<PathBuf> {
    let path = path?;
    path.parent().map(Path::to_path_buf)
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

fn nonempty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|raw| !raw.trim().is_empty())
}

#[cfg(windows)]
fn state_env_name() -> &'static str {
    "LOCALAPPDATA"
}

#[cfg(not(windows))]
fn state_env_name() -> &'static str {
    "XDG_STATE_HOME"
}

#[cfg(windows)]
fn data_env_name() -> &'static str {
    "APPDATA"
}

#[cfg(not(windows))]
fn data_env_name() -> &'static str {
    "XDG_DATA_HOME"
}

#[cfg(windows)]
fn config_env_name() -> &'static str {
    "APPDATA"
}

#[cfg(not(windows))]
fn config_env_name() -> &'static str {
    "XDG_CONFIG_HOME"
}

#[cfg(windows)]
fn default_state_root(home: &Path) -> PathBuf {
    home.join("AppData/Local")
}

#[cfg(not(windows))]
fn default_state_root(home: &Path) -> PathBuf {
    home.join(".local/state")
}

#[cfg(windows)]
fn default_data_root(home: &Path) -> PathBuf {
    home.join("AppData/Roaming")
}

#[cfg(not(windows))]
fn default_data_root(home: &Path) -> PathBuf {
    home.join(".local/share")
}

#[cfg(windows)]
fn default_config_root(home: &Path) -> PathBuf {
    home.join("AppData/Roaming")
}

#[cfg(not(windows))]
fn default_config_root(home: &Path) -> PathBuf {
    home.join(".config")
}

#[cfg(test)]
mod tests {
    use super::factory_paths_from_env;
    use std::collections::HashMap;
    use std::path::Path;

    #[test]
    fn factory_paths_default_to_home_factory_dir() {
        let home = Path::new("/tmp/test-home");
        let env = HashMap::<String, String>::new();
        let paths = factory_paths_from_env(home, |name| env.get(name).cloned());

        assert_eq!(paths.home_dir, home.join(".factory"));
        assert_eq!(paths.config_path, home.join(".factory/config.json"));
        assert_eq!(paths.settings_path, home.join(".factory/settings.json"));
        assert_eq!(paths.machine_droids_dir, home.join(".factory/droids"));
    }

    #[test]
    fn factory_paths_follow_explicit_factory_home() {
        let home = Path::new("/tmp/test-home");
        let env = HashMap::from([(
            "OPENGATEWAY_FACTORY_HOME".to_string(),
            "~/custom-factory".to_string(),
        )]);
        let paths = factory_paths_from_env(home, |name| env.get(name).cloned());

        assert_eq!(paths.home_dir, home.join("custom-factory"));
        assert_eq!(paths.config_path, home.join("custom-factory/config.json"));
        assert_eq!(
            paths.settings_path,
            home.join("custom-factory/settings.json")
        );
        assert_eq!(paths.machine_droids_dir, home.join("custom-factory/droids"));
    }

    #[test]
    fn factory_paths_derive_home_from_explicit_settings_path() {
        let home = Path::new("/tmp/test-home");
        let env = HashMap::from([(
            "OPENGATEWAY_FACTORY_SETTINGS".to_string(),
            "/opt/factory/settings.json".to_string(),
        )]);
        let paths = factory_paths_from_env(home, |name| env.get(name).cloned());

        assert_eq!(paths.home_dir, Path::new("/opt/factory"));
        assert_eq!(paths.config_path, Path::new("/opt/factory/config.json"));
        assert_eq!(paths.settings_path, Path::new("/opt/factory/settings.json"));
        assert_eq!(paths.machine_droids_dir, Path::new("/opt/factory/droids"));
    }

    #[test]
    fn factory_paths_preserve_explicit_droids_dir() {
        let home = Path::new("/tmp/test-home");
        let env = HashMap::from([(
            "OPENGATEWAY_FACTORY_DROIDS_DIR".to_string(),
            "/srv/factory-state/droids".to_string(),
        )]);
        let paths = factory_paths_from_env(home, |name| env.get(name).cloned());

        assert_eq!(paths.home_dir, Path::new("/srv/factory-state"));
        assert_eq!(
            paths.machine_droids_dir,
            Path::new("/srv/factory-state/droids")
        );
    }
}
