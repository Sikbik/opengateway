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

    app_dirs.sort_by(compare_factory_app_dirs);
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
    let bundled_droid = install_dir.join("resources").join("bin").join(droid_exe_name());
    let bundled_droid_path = bundled_droid.exists().then(|| bundled_droid.display().to_string());
    let issue = if bundled_droid_path.is_some() {
        None
    } else {
        Some("Factory Desktop was found, but its bundled Droid executable was not found".to_string())
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

    roots
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
    name.strip_prefix("app-").map(|version| version.to_string())
}

fn compare_factory_app_dirs(left: &PathBuf, right: &PathBuf) -> Ordering {
    let left_version = factory_app_version(left).unwrap_or_default();
    let right_version = factory_app_version(right).unwrap_or_default();
    compare_version_like(&left_version, &right_version)
}

fn compare_version_like(left: &str, right: &str) -> Ordering {
    let left_parts = version_parts(left);
    let right_parts = version_parts(right);
    left_parts.cmp(&right_parts).then_with(|| left.cmp(right))
}

fn version_parts(raw: &str) -> Vec<u64> {
    raw.split('.')
        .map(|part| part.parse::<u64>().unwrap_or(0))
        .collect()
}

fn droid_exe_name() -> &'static str {
    if cfg!(windows) {
        "droid.exe"
    } else {
        "droid"
    }
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
        assert!(readiness.bundled_droid_path.unwrap().contains("app-0.116.1"));
        assert_eq!(readiness.issue, None);
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

    fn temp_root(name: &str) -> PathBuf {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        env::temp_dir().join(format!("opengateway-{name}-{millis}"))
    }
}
