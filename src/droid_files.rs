use anyhow::{anyhow, Context, Result};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) const WORKSPACE_DROIDS_RELATIVE: &str = ".factory/droids";

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DroidRecord {
    name: String,
    path: String,
    scope: &'static str,
    model: Option<String>,
    kind: &'static str,
    issues: Vec<String>,
}

pub(crate) fn merge_droids(
    mut workspace: Vec<DroidRecord>,
    mut machine: Vec<DroidRecord>,
) -> Vec<DroidRecord> {
    workspace.sort_by(|left, right| left.name.cmp(&right.name));
    machine.sort_by(|left, right| left.name.cmp(&right.name));
    workspace.extend(machine);
    workspace
}

pub(crate) fn read_droids(dir: &Path, scope: &'static str) -> Vec<DroidRecord> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };

    let mut droids = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("md") {
            continue;
        }
        if let Ok(record) = parse_droid_file(&path, scope) {
            droids.push(record);
        }
    }
    droids
}

pub(crate) fn set_droid_model(
    path: &Path,
    model: &str,
    machine_droids_dir: &Path,
    workspace_droids_dir: Option<&Path>,
) -> Result<DroidRecord> {
    let (canonical, scope) =
        resolve_allowed_droid_path(path, machine_droids_dir, workspace_droids_dir)?;
    let original = fs::read_to_string(&canonical)
        .with_context(|| format!("failed to read {}", canonical.display()))?;
    let normalized_model = normalize_droid_model(model);
    let updated = update_front_matter_model(&original, &normalized_model)?;
    fs::write(&canonical, updated)
        .with_context(|| format!("failed to write {}", canonical.display()))?;
    parse_droid_file(&canonical, scope)
}

fn normalize_droid_model(model: &str) -> String {
    let trimmed = model.trim();
    if trimmed.is_empty() || trimmed == "inherit" || trimmed.starts_with("custom:") {
        trimmed.to_string()
    } else {
        format!("custom:{trimmed}")
    }
}

fn parse_droid_file(path: &Path, scope: &'static str) -> Result<DroidRecord> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let front_matter = extract_front_matter(&raw)?;
    let name = front_matter
        .lines()
        .find_map(|line| {
            line.strip_prefix("name:")
                .map(|value| value.trim().to_string())
        })
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("unknown")
                .to_string()
        });
    let model = front_matter.lines().find_map(|line| {
        line.strip_prefix("model:")
            .map(|value| value.trim().to_string())
    });

    let kind = match model.as_deref() {
        Some(value) if value.starts_with("custom:") => "custom",
        Some("inherit") => "inherit",
        Some(_) => "builtin",
        None => "missing",
    };

    let mut issues = Vec::new();
    if kind == "builtin" {
        issues.push("Pinned to a non-custom model.".to_string());
    }
    if kind == "missing" {
        issues.push("No model declared in front matter.".to_string());
    }

    Ok(DroidRecord {
        name,
        path: path.display().to_string(),
        scope,
        model,
        kind,
        issues,
    })
}

fn resolve_allowed_droid_path(
    path: &Path,
    machine_droids_dir: &Path,
    workspace_droids_dir: Option<&Path>,
) -> Result<(PathBuf, &'static str)> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", path.display()))?;

    if let Ok(machine_dir) = machine_droids_dir.canonicalize() {
        if canonical.starts_with(machine_dir) {
            return Ok((canonical, "machine"));
        }
    }

    if let Some(workspace_droids_dir) = workspace_droids_dir {
        if let Ok(workspace_dir) = workspace_droids_dir.canonicalize() {
            if canonical.starts_with(workspace_dir) {
                return Ok((canonical, "workspace"));
            }
        }
    }

    Err(anyhow!(
        "refusing to edit a file outside the allowed droid directories"
    ))
}

fn extract_front_matter(contents: &str) -> Result<&str> {
    let without_open = contents
        .strip_prefix("---\n")
        .ok_or_else(|| anyhow!("file does not start with front matter"))?;
    let end = without_open
        .find("\n---\n")
        .ok_or_else(|| anyhow!("front matter closing marker not found"))?;
    Ok(&without_open[..end])
}

fn update_front_matter_model(contents: &str, model: &str) -> Result<String> {
    let without_open = contents
        .strip_prefix("---\n")
        .ok_or_else(|| anyhow!("file does not start with front matter"))?;
    let end = without_open
        .find("\n---\n")
        .ok_or_else(|| anyhow!("front matter closing marker not found"))?;
    let front_matter = &without_open[..end];
    let body = &without_open[end + 5..];

    let mut found = false;
    let mut next_front_matter = Vec::new();
    for line in front_matter.lines() {
        if line.trim_start().starts_with("model:") {
            next_front_matter.push(format!("model: {model}"));
            found = true;
        } else {
            next_front_matter.push(line.to_string());
        }
    }

    if !found {
        next_front_matter.push(format!("model: {model}"));
    }

    Ok(format!(
        "---\n{}\n---\n{}",
        next_front_matter.join("\n"),
        body
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new() -> Self {
            let id = NEXT_DIR.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!(
                "opengateway-droid-files-{}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn parses_custom_droid_front_matter() {
        let temp = TestDir::new();
        let path = temp.path().join("worker.md");
        fs::write(
            &path,
            "---\nname: Worker\nmodel: custom:gpt-5.4(xhigh)\n---\nRun the task.\n",
        )
        .unwrap();

        let record = parse_droid_file(&path, "machine").unwrap();

        assert_eq!(record.name, "Worker");
        assert_eq!(record.scope, "machine");
        assert_eq!(record.model.as_deref(), Some("custom:gpt-5.4(xhigh)"));
        assert_eq!(record.kind, "custom");
        assert!(record.issues.is_empty());
    }

    #[test]
    fn set_droid_model_normalizes_raw_model_id() {
        let temp = TestDir::new();
        let machine_dir = temp.path().join("machine");
        fs::create_dir_all(&machine_dir).unwrap();
        let path = machine_dir.join("worker.md");
        fs::write(&path, "---\nname: Worker\n---\nRun the task.\n").unwrap();

        let record = set_droid_model(&path, "gpt-5-codex", &machine_dir, None).unwrap();

        assert_eq!(record.model.as_deref(), Some("custom:gpt-5-codex"));
        assert_eq!(record.kind, "custom");
        let updated = fs::read_to_string(&path).unwrap();
        assert!(updated.contains("\nmodel: custom:gpt-5-codex\n"));
    }

    #[test]
    fn set_droid_model_rejects_paths_outside_allowed_dirs() {
        let temp = TestDir::new();
        let machine_dir = temp.path().join("machine");
        let outside_dir = temp.path().join("outside");
        fs::create_dir_all(&machine_dir).unwrap();
        fs::create_dir_all(&outside_dir).unwrap();
        let path = outside_dir.join("worker.md");
        fs::write(&path, "---\nname: Worker\n---\nRun the task.\n").unwrap();

        let error = set_droid_model(&path, "gpt-5-codex", &machine_dir, None).unwrap_err();

        assert!(error
            .to_string()
            .contains("refusing to edit a file outside the allowed droid directories"));
    }
}
