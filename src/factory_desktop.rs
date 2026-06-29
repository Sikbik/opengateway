use serde::Serialize;
use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct FactoryDesktopReadiness {
    pub installed: bool,
    pub install_dir: Option<String>,
    pub version: Option<String>,
    pub bundled_droid_path: Option<String>,
    pub issue: Option<String>,
}

pub fn probe_factory_desktop() -> FactoryDesktopReadiness {
    probe_factory_desktop_from_candidates(factory_roots())
}

fn probe_factory_desktop_from_candidates(roots: Vec<PathBuf>) -> FactoryDesktopReadiness {
    let mut app_dirs = Vec::new();

    for root in roots {
        if let Ok(entries) = fs::read_dir(&root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && factory_app_version(&path).is_some() {
                    app_dirs.push(path);
                }
            }
        }
    }

    app_dirs.sort_by(|left, right| compare_factory_app_dirs(left, right));
    let Some(install_dir) = app_dirs.pop() else {
        return FactoryDesktopReadiness {
            installed: false,
            install_dir: None,
            version: None,
            bundled_droid_path: None,
            issue: Some("Factory Desktop install was not found".to_string()),
        };
    };

    let version = factory_app_version(&install_dir);
    let bundled_droid_path =
        bundled_droid_path(&install_dir).map(|path| path.display().to_string());
    let issue = if bundled_droid_path.is_some() {
        None
    } else {
        Some(
            "Factory Desktop was found, but its bundled Droid executable was not found".to_string(),
        )
    };

    FactoryDesktopReadiness {
        installed: true,
        install_dir: Some(install_dir.display().to_string()),
        version,
        bundled_droid_path,
        issue,
    }
}

fn factory_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(path) = env_path("OPENGATEWAY_FACTORY_DESKTOP_DIR") {
        roots.push(path);
    }

    if cfg!(windows) {
        if let Some(local_app_data) = env_path("LOCALAPPDATA") {
            roots.push(local_app_data.join("Factory"));
        }
    }

    if running_in_wsl() {
        roots.extend(wsl_factory_roots());
    }

    roots.sort();
    roots.dedup();
    roots
}

fn wsl_factory_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(local_app_data) = env_path("OPENGATEWAY_WINDOWS_LOCALAPPDATA") {
        roots.push(local_app_data.join("Factory"));
    }

    if let Some(factory_home) = env_path("OPENGATEWAY_FACTORY_HOME") {
        if let Some(root) = factory_root_from_factory_home(&factory_home) {
            roots.push(root);
        }
    }

    roots.extend(wsl_user_factory_roots(Path::new("/mnt/c/Users")));
    roots.sort();
    roots.dedup();
    roots
}

fn factory_root_from_factory_home(factory_home: &Path) -> Option<PathBuf> {
    if factory_home.file_name()?.to_string_lossy() != ".factory" {
        return None;
    }

    Some(
        factory_home
            .parent()?
            .join("AppData")
            .join("Local")
            .join("Factory"),
    )
}

fn wsl_user_factory_roots(users_root: &Path) -> Vec<PathBuf> {
    let mut roots = fs::read_dir(users_root)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path().join("AppData").join("Local").join("Factory"))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    roots.sort();
    roots
}

fn running_in_wsl() -> bool {
    env::var_os("WSL_DISTRO_NAME")
        .map(|value| !value.is_empty())
        .unwrap_or(false)
        || fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|value| value.to_ascii_lowercase().contains("microsoft"))
            .unwrap_or(false)
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name).and_then(|value| {
        if value.is_empty() {
            None
        } else {
            Some(PathBuf::from(value))
        }
    })
}

fn factory_app_version(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy();
    let version = name.strip_prefix("app-")?;
    parse_version_parts(version)?;
    Some(version.to_string())
}

fn compare_factory_app_dirs(left: &Path, right: &Path) -> Ordering {
    let left_version = factory_app_version(left).unwrap_or_default();
    let right_version = factory_app_version(right).unwrap_or_default();
    compare_version_like(&left_version, &right_version)
}

fn compare_version_like(left: &str, right: &str) -> Ordering {
    let left_parts = parse_version_parts(left).unwrap_or_default();
    let right_parts = parse_version_parts(right).unwrap_or_default();
    left_parts.cmp(&right_parts).then_with(|| left.cmp(right))
}

fn parse_version_parts(raw: &str) -> Option<Vec<u64>> {
    raw.split('.')
        .map(|part| {
            if part.is_empty() {
                None
            } else {
                part.parse::<u64>().ok()
            }
        })
        .collect()
}

fn bundled_droid_path(install_dir: &Path) -> Option<PathBuf> {
    let bin_dir = install_dir.join("resources").join("bin");
    droid_exe_names()
        .iter()
        .map(|name| bin_dir.join(name))
        .find(|path| path.is_file())
}

fn droid_exe_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["droid.exe"]
    } else {
        &["droid", "droid.exe"]
    }
}

#[cfg(test)]
fn droid_exe_name() -> &'static str {
    droid_exe_names()[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn picks_latest_factory_app_with_bundled_droid() {
        let root = temp_root("factory-desktop-latest");
        let older = root.join("app-0.116.0").join("resources").join("bin");
        let newer = root.join("app-0.116.1").join("resources").join("bin");
        fs::create_dir_all(&older).unwrap();
        fs::create_dir_all(&newer).unwrap();
        fs::write(newer.join(droid_exe_name()), "").unwrap();

        let readiness = probe_factory_desktop_from_candidates(vec![root.clone()]);

        assert!(readiness.installed);
        assert_eq!(readiness.version.as_deref(), Some("0.116.1"));
        assert!(readiness
            .bundled_droid_path
            .unwrap()
            .contains("app-0.116.1"));
        assert_eq!(readiness.issue, None);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn picks_windows_bundled_droid_from_wsl_visible_install() {
        let root = temp_root("factory-desktop-windows-droid");
        let bin_dir = root.join("app-0.116.1").join("resources").join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        fs::write(bin_dir.join("droid.exe"), "").unwrap();

        let readiness = probe_factory_desktop_from_candidates(vec![root.clone()]);

        assert!(readiness.installed);
        assert!(readiness.bundled_droid_path.unwrap().ends_with("droid.exe"));
        assert_eq!(readiness.issue, None);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn derives_wsl_factory_root_from_factory_home() {
        let root = factory_root_from_factory_home(Path::new("/mnt/c/Users/alice/.factory"));

        assert_eq!(
            root.as_deref(),
            Some(Path::new("/mnt/c/Users/alice/AppData/Local/Factory"))
        );
    }

    #[test]
    fn discovers_wsl_user_factory_roots() {
        let root = temp_root("factory-desktop-wsl-users");
        let factory = root
            .join("alice")
            .join("AppData")
            .join("Local")
            .join("Factory");
        fs::create_dir_all(&factory).unwrap();

        let roots = wsl_user_factory_roots(&root);

        assert_eq!(roots, vec![factory]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reports_missing_factory_desktop() {
        let readiness = probe_factory_desktop_from_candidates(Vec::new());

        assert!(!readiness.installed);
        assert_eq!(readiness.install_dir, None);
        assert_eq!(
            readiness.issue.as_deref(),
            Some("Factory Desktop install was not found")
        );
    }

    #[test]
    fn reports_missing_bundled_droid() {
        let root = temp_root("factory-desktop-no-droid");
        fs::create_dir_all(root.join("app-0.116.1")).unwrap();

        let readiness = probe_factory_desktop_from_candidates(vec![root.clone()]);

        assert!(readiness.installed);
        assert_eq!(readiness.version.as_deref(), Some("0.116.1"));
        assert_eq!(readiness.bundled_droid_path, None);
        assert_eq!(
            readiness.issue.as_deref(),
            Some("Factory Desktop was found, but its bundled Droid executable was not found")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ignores_invalid_factory_app_versions() {
        let root = temp_root("factory-desktop-invalid-versions");
        let valid = root.join("app-1.2.3").join("resources").join("bin");
        fs::create_dir_all(&valid).unwrap();
        fs::write(valid.join(droid_exe_name()), "").unwrap();

        for name in [
            "app-beta",
            "app-999.x",
            "app-1.2-preview",
            "app-",
            "app-1..2",
        ] {
            fs::create_dir_all(root.join(name).join("resources").join("bin")).unwrap();
            fs::write(
                root.join(name)
                    .join("resources")
                    .join("bin")
                    .join(droid_exe_name()),
                "",
            )
            .unwrap();
        }

        let readiness = probe_factory_desktop_from_candidates(vec![root.clone()]);

        assert!(readiness.installed);
        assert_eq!(readiness.version.as_deref(), Some("1.2.3"));
        assert!(readiness.bundled_droid_path.unwrap().contains("app-1.2.3"));
        fs::remove_dir_all(root).unwrap();
    }

    fn temp_root(name: &str) -> PathBuf {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        env::temp_dir().join(format!("opengateway-{name}-{millis}"))
    }
}
