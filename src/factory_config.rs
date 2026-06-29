use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const FACTORY_PREFERRED_MODEL: &str = "gpt-5.5";
const FACTORY_PREFERRED_REASONING_EFFORT: &str = "xhigh";
const FACTORY_DEFAULT_MAX_OUTPUT_TOKENS: u64 = 16_384;
const DEFAULT_OPENAI_MODEL_CATALOG: [(&str, &str); 14] = [
    ("gpt-5.4", "GPT-5.4"),
    ("gpt-5.3-codex", "GPT-5.3 Codex"),
    ("gpt-5.3-codex-spark", "GPT-5.3 Codex Spark"),
    ("gpt-5.2-codex", "GPT-5.2 Codex"),
    ("gpt-5.1-codex-max", "GPT-5.1 Codex Max"),
    ("gpt-5.1-codex", "GPT-5.1 Codex"),
    ("gpt-5.1-codex-mini", "GPT-5.1 Codex Mini"),
    ("gpt-5-codex", "GPT-5 Codex"),
    ("gpt-5-codex-mini", "GPT-5 Codex Mini"),
    ("gpt-5.2", "GPT-5.2"),
    ("gpt-5.1", "GPT-5.1"),
    ("gpt-5", "GPT-5"),
    ("gpt-5.5", "GPT-5.5"),
    ("gpt-5.5-fast", "GPT-5.5 Fast"),
];
const DEPRECATED_REASONING_ALIAS_MODEL_IDS: [&str; 14] = [
    "gpt-5.4(low)",
    "gpt-5.4(medium)",
    "gpt-5.4(high)",
    "gpt-5.4(xhigh)",
    "gpt-5.3-codex(high)",
    "gpt-5.3-codex(xhigh)",
    "gpt-5.2-codex(high)",
    "gpt-5.2-codex(xhigh)",
    "gpt-5.1-codex-max(high)",
    "gpt-5.1-codex-max(xhigh)",
    "gpt-5.2(high)",
    "gpt-5.2(xhigh)",
    "gpt-5.1(high)",
    "gpt-5(high)",
];

#[derive(Debug)]
pub struct FactorySyncResult {
    pub legacy_added: usize,
    pub legacy_updated: usize,
    pub legacy_backup: Option<PathBuf>,
    pub settings_added: usize,
    pub settings_updated: usize,
    pub settings_backup: Option<PathBuf>,
    pub defaults_updated: bool,
}

pub fn sync_factory_files(
    config_path: &Path,
    settings_path: &Path,
    base_url: &str,
    api_key: &str,
    model_ids: &[String],
) -> Result<FactorySyncResult> {
    let (legacy_added, legacy_updated, legacy_backup) =
        merge_factory_config(config_path, base_url, api_key, model_ids)?;
    let (settings_added, settings_updated, settings_backup, defaults_updated) =
        merge_factory_settings(settings_path, base_url, api_key, model_ids)?;

    Ok(FactorySyncResult {
        legacy_added,
        legacy_updated,
        legacy_backup,
        settings_added,
        settings_updated,
        settings_backup,
        defaults_updated,
    })
}

pub fn factory_settings_needs_sync(
    settings_path: &Path,
    base_url: &str,
    api_key: &str,
    model_ids: &[String],
) -> Result<bool> {
    let existing = fs::read_to_string(settings_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .unwrap_or_else(|| json!({}));
    let (_, added, updated, defaults_updated) =
        merge_factory_settings_document(existing, base_url, api_key, model_ids)?;
    Ok(added > 0 || updated > 0 || defaults_updated)
}

pub fn resolve_model_ids(explicit_models: &str) -> Vec<String> {
    let explicit_models = explicit_models.trim();
    if !explicit_models.is_empty() {
        return parse_model_list(explicit_models);
    }

    let env_models = std::env::var("OPENGATEWAY_MODELS").unwrap_or_default();
    if !env_models.trim().is_empty() {
        return parse_model_list(&env_models);
    }

    DEFAULT_OPENAI_MODEL_CATALOG
        .iter()
        .map(|(model, _)| (*model).to_string())
        .collect()
}

pub fn build_factory_config(base_url: &str, api_key: &str, model_ids: &[String]) -> Value {
    let models = model_ids
        .iter()
        .map(|model_id| {
            json!({
              "model_display_name": model_display_name(model_id),
              "model": model_id,
              "base_url": format!("{base_url}/v1"),
              "api_key": api_key,
              "provider": "openai"
            })
        })
        .collect::<Vec<_>>();

    json!({ "custom_models": models })
}

pub fn merge_factory_config(
    output_path: &Path,
    base_url: &str,
    api_key: &str,
    model_ids: &[String],
) -> Result<(usize, usize, Option<PathBuf>)> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let incoming_models = build_factory_config(base_url, api_key, model_ids)
        .get("custom_models")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut existing = json!({});
    let mut backup: Option<PathBuf> = None;
    if output_path.exists() {
        let original_name = output_path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_else(|| "config.json".to_string());
        let backup_path =
            output_path.with_file_name(format!("{original_name}.bak-{}", epoch_seconds()));
        fs::copy(output_path, &backup_path).with_context(|| {
            format!(
                "failed to create backup {} from {}",
                backup_path.display(),
                output_path.display()
            )
        })?;
        backup = Some(backup_path);

        existing = fs::read_to_string(output_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .unwrap_or_else(|| json!({}));
    }

    if !existing.is_object() {
        existing = json!({});
    }

    let object = existing
        .as_object_mut()
        .ok_or_else(|| anyhow!("internal error: expected JSON object"))?;

    let current_models = object
        .entry("custom_models")
        .or_insert_with(|| Value::Array(Vec::new()));
    if !current_models.is_array() {
        *current_models = Value::Array(Vec::new());
    }
    let current_models = current_models
        .as_array_mut()
        .ok_or_else(|| anyhow!("internal error: custom_models should be array"))?;

    let desired_models = model_ids.iter().map(String::as_str).collect::<HashSet<_>>();
    let removed = prune_deprecated_reasoning_aliases_from_legacy_config(
        current_models,
        &desired_models,
        base_url,
    );

    let mut index_by_model: HashMap<String, usize> = HashMap::new();
    for (index, model) in current_models.iter().enumerate() {
        if let Some(name) = model.get("model").and_then(Value::as_str) {
            index_by_model.insert(name.to_string(), index);
        }
    }

    let mut added = 0;
    let mut updated = removed;
    for model in incoming_models {
        let Some(model_name) = model.get("model").and_then(Value::as_str) else {
            continue;
        };
        if let Some(index) = index_by_model.get(model_name).copied() {
            if current_models[index] != model {
                current_models[index] = model;
                updated += 1;
            }
        } else {
            index_by_model.insert(model_name.to_string(), current_models.len());
            current_models.push(model);
            added += 1;
        }
    }

    let rendered =
        serde_json::to_string_pretty(&existing).context("failed to encode merged config")?;
    fs::write(output_path, format!("{rendered}\n"))
        .with_context(|| format!("failed to write {}", output_path.display()))?;

    Ok((added, updated, backup))
}

pub fn merge_factory_settings(
    output_path: &Path,
    base_url: &str,
    api_key: &str,
    model_ids: &[String],
) -> Result<(usize, usize, Option<PathBuf>, bool)> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let (existing, backup) = read_json_with_backup(output_path)?;
    let (merged, added, updated, defaults_updated) =
        merge_factory_settings_document(existing, base_url, api_key, model_ids)?;

    let rendered =
        serde_json::to_string_pretty(&merged).context("failed to encode merged settings")?;
    fs::write(output_path, format!("{rendered}\n"))
        .with_context(|| format!("failed to write {}", output_path.display()))?;

    Ok((added, updated, backup, defaults_updated))
}

fn parse_model_list(raw: &str) -> Vec<String> {
    let mut models = Vec::new();
    let mut seen = HashSet::new();

    for entry in raw.split(',') {
        let model = entry.trim();
        if model.is_empty() {
            continue;
        }
        if seen.insert(model.to_string()) {
            models.push(model.to_string());
        }
    }

    if models.is_empty() {
        DEFAULT_OPENAI_MODEL_CATALOG
            .iter()
            .map(|(model, _)| (*model).to_string())
            .collect()
    } else {
        models
    }
}

fn model_display_name(model_id: &str) -> String {
    DEFAULT_OPENAI_MODEL_CATALOG
        .iter()
        .find(|(candidate, _)| *candidate == model_id)
        .map(|(_, display_name)| (*display_name).to_string())
        .unwrap_or_else(|| model_id.to_string())
}

fn build_factory_settings_model(
    model_id: &str,
    base_url: &str,
    api_key: &str,
    index: usize,
) -> Value {
    let display_name = model_display_name(model_id);
    json!({
        "model": model_id,
        "id": factory_custom_model_id(&display_name, index),
        "index": index,
        "baseUrl": format!("{base_url}/v1"),
        "apiKey": api_key,
        "displayName": display_name,
        "maxOutputTokens": FACTORY_DEFAULT_MAX_OUTPUT_TOKENS,
        "noImageSupport": false,
        "provider": "openai"
    })
}

fn factory_custom_model_id(display_name: &str, index: usize) -> String {
    format!("custom:{}-{index}", display_name.replace(' ', "-"))
}

fn merge_factory_settings_document(
    mut existing: Value,
    base_url: &str,
    api_key: &str,
    model_ids: &[String],
) -> Result<(Value, usize, usize, bool)> {
    if !existing.is_object() {
        existing = json!({});
    }

    let object = existing
        .as_object_mut()
        .ok_or_else(|| anyhow!("internal error: expected settings JSON object"))?;

    let managed_ids_before;
    let preferred_model_id;
    let added;
    let updated;
    {
        let current_models = object
            .entry("customModels")
            .or_insert_with(|| Value::Array(Vec::new()));
        if !current_models.is_array() {
            *current_models = Value::Array(Vec::new());
        }
        let current_models = current_models
            .as_array_mut()
            .ok_or_else(|| anyhow!("internal error: customModels should be array"))?;

        managed_ids_before = collect_managed_factory_model_ids(current_models, base_url, api_key);
        let merge_result =
            merge_factory_settings_models(current_models, base_url, api_key, model_ids);
        added = merge_result.0;
        updated = merge_result.1;
        preferred_model_id = merge_result.2;
    }

    let defaults_updated = update_factory_settings_defaults(
        object,
        &managed_ids_before,
        preferred_model_id.as_deref(),
    );

    Ok((existing, added, updated, defaults_updated))
}

fn collect_managed_factory_model_ids(
    current_models: &[Value],
    base_url: &str,
    api_key: &str,
) -> HashSet<String> {
    let expected_base_url = format!("{base_url}/v1");

    current_models
        .iter()
        .filter_map(Value::as_object)
        .filter(|model| {
            model.get("provider").and_then(Value::as_str) == Some("openai")
                && model.get("baseUrl").and_then(Value::as_str) == Some(expected_base_url.as_str())
                && (model.get("apiKey").and_then(Value::as_str) == Some(api_key)
                    || model
                        .get("model")
                        .and_then(Value::as_str)
                        .map(is_deprecated_reasoning_alias_model_id)
                        .unwrap_or(false))
        })
        .filter_map(|model| model.get("id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn merge_factory_settings_models(
    current_models: &mut Vec<Value>,
    base_url: &str,
    api_key: &str,
    model_ids: &[String],
) -> (usize, usize, Option<String>) {
    let desired_models = model_ids.iter().map(String::as_str).collect::<HashSet<_>>();
    let removed = prune_deprecated_reasoning_aliases_from_factory_settings(
        current_models,
        &desired_models,
        base_url,
    );

    let mut index_by_model: HashMap<String, usize> = HashMap::new();
    for (index, model) in current_models.iter().enumerate() {
        if let Some(name) = model.get("model").and_then(Value::as_str) {
            index_by_model.insert(name.to_string(), index);
        }
    }

    let mut added = 0;
    let mut updated = removed;
    let mut preferred_model_id = None;

    for (desired_index, model_id) in model_ids.iter().enumerate() {
        let factory_index = factory_settings_model_index(model_id, desired_index);
        if let Some(index) = index_by_model.get(model_id).copied() {
            let replacement =
                build_factory_settings_model(model_id, base_url, api_key, factory_index);
            if model_id == FACTORY_PREFERRED_MODEL {
                preferred_model_id = replacement
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
            if current_models[index] != replacement {
                current_models[index] = replacement;
                updated += 1;
            }
        } else {
            let model = build_factory_settings_model(model_id, base_url, api_key, factory_index);
            if model_id == FACTORY_PREFERRED_MODEL {
                preferred_model_id = model.get("id").and_then(Value::as_str).map(str::to_string);
            }
            index_by_model.insert(model_id.to_string(), current_models.len());
            current_models.push(model);
            added += 1;
        }
    }

    (added, updated, preferred_model_id)
}

fn prune_deprecated_reasoning_aliases_from_legacy_config(
    current_models: &mut Vec<Value>,
    desired_models: &HashSet<&str>,
    base_url: &str,
) -> usize {
    let expected_base_url = format!("{base_url}/v1");
    let before = current_models.len();
    current_models.retain(|model| {
        !model
            .as_object()
            .map(|model| {
                model.get("provider").and_then(Value::as_str) == Some("openai")
                    && model.get("base_url").and_then(Value::as_str)
                        == Some(expected_base_url.as_str())
                    && model
                        .get("model")
                        .and_then(Value::as_str)
                        .map(|model_id| {
                            is_deprecated_reasoning_alias_model_id(model_id)
                                && !desired_models.contains(model_id)
                        })
                        .unwrap_or(false)
            })
            .unwrap_or(false)
    });
    before - current_models.len()
}

fn prune_deprecated_reasoning_aliases_from_factory_settings(
    current_models: &mut Vec<Value>,
    desired_models: &HashSet<&str>,
    base_url: &str,
) -> usize {
    let expected_base_url = format!("{base_url}/v1");
    let before = current_models.len();
    current_models.retain(|model| {
        !model
            .as_object()
            .map(|model| {
                model.get("provider").and_then(Value::as_str) == Some("openai")
                    && model.get("baseUrl").and_then(Value::as_str)
                        == Some(expected_base_url.as_str())
                    && model
                        .get("model")
                        .and_then(Value::as_str)
                        .map(|model_id| {
                            is_deprecated_reasoning_alias_model_id(model_id)
                                && !desired_models.contains(model_id)
                        })
                        .unwrap_or(false)
            })
            .unwrap_or(false)
    });
    before - current_models.len()
}

fn is_deprecated_reasoning_alias_model_id(model_id: &str) -> bool {
    DEPRECATED_REASONING_ALIAS_MODEL_IDS.contains(&model_id)
}

fn factory_settings_model_index(model_id: &str, fallback_index: usize) -> usize {
    match model_id {
        "gpt-5.4" => 0,
        "gpt-5.4(low)" => 1,
        "gpt-5.4(medium)" => 2,
        "gpt-5.4(high)" => 3,
        "gpt-5.4(xhigh)" => 4,
        "gpt-5.3-codex" => 5,
        "gpt-5.3-codex-spark" => 6,
        "gpt-5.3-codex(high)" => 7,
        "gpt-5.3-codex(xhigh)" => 8,
        "gpt-5.2-codex" => 9,
        "gpt-5.2-codex(high)" => 10,
        "gpt-5.2-codex(xhigh)" => 11,
        "gpt-5.1-codex-max" => 12,
        "gpt-5.1-codex-max(high)" => 13,
        "gpt-5.1-codex-max(xhigh)" => 14,
        "gpt-5.1-codex" => 15,
        "gpt-5.1-codex-mini" => 16,
        "gpt-5-codex" => 17,
        "gpt-5-codex-mini" => 18,
        "gpt-5.2" => 19,
        "gpt-5.2(high)" => 20,
        "gpt-5.2(xhigh)" => 21,
        "gpt-5.1" => 22,
        "gpt-5.1(high)" => 23,
        "gpt-5" => 24,
        "gpt-5(high)" => 25,
        "gpt-5.5" => 26,
        "gpt-5.5-fast" => 27,
        _ => fallback_index,
    }
}

fn update_factory_settings_defaults(
    settings: &mut serde_json::Map<String, Value>,
    managed_ids_before: &HashSet<String>,
    preferred_model_id: Option<&str>,
) -> bool {
    let Some(preferred_model_id) = preferred_model_id else {
        return false;
    };

    let mut updated = false;

    let session_defaults = ensure_object_entry(settings, "sessionDefaultSettings");
    if should_update_factory_default_model(
        session_defaults.get("model").and_then(Value::as_str),
        managed_ids_before,
        preferred_model_id,
    ) {
        session_defaults.insert(
            "model".to_string(),
            Value::String(preferred_model_id.to_string()),
        );
        updated = true;
    }
    if session_defaults
        .get("model")
        .and_then(Value::as_str)
        .map(|value| value == preferred_model_id)
        .unwrap_or(false)
        && session_defaults
            .get("reasoningEffort")
            .and_then(Value::as_str)
            != Some(FACTORY_PREFERRED_REASONING_EFFORT)
    {
        session_defaults.insert(
            "reasoningEffort".to_string(),
            Value::String(FACTORY_PREFERRED_REASONING_EFFORT.to_string()),
        );
        updated = true;
    }

    let mission_defaults = ensure_object_entry(settings, "missionModelSettings");
    for (model_key, effort_key) in [
        ("orchestratorModel", "orchestratorReasoningEffort"),
        ("workerModel", "workerReasoningEffort"),
        ("validationWorkerModel", "validationWorkerReasoningEffort"),
    ] {
        if should_update_factory_default_model(
            mission_defaults.get(model_key).and_then(Value::as_str),
            managed_ids_before,
            preferred_model_id,
        ) {
            mission_defaults.insert(
                model_key.to_string(),
                Value::String(preferred_model_id.to_string()),
            );
            updated = true;
        }

        if mission_defaults
            .get(model_key)
            .and_then(Value::as_str)
            .map(|value| value == preferred_model_id)
            .unwrap_or(false)
            && mission_defaults.get(effort_key).and_then(Value::as_str)
                != Some(FACTORY_PREFERRED_REASONING_EFFORT)
        {
            mission_defaults.insert(
                effort_key.to_string(),
                Value::String(FACTORY_PREFERRED_REASONING_EFFORT.to_string()),
            );
            updated = true;
        }
    }

    updated
}

fn should_update_factory_default_model(
    current_model: Option<&str>,
    managed_ids_before: &HashSet<String>,
    preferred_model_id: &str,
) -> bool {
    match current_model {
        None => true,
        Some(value) if value == preferred_model_id => false,
        Some(value) => managed_ids_before.contains(value),
    }
}

fn ensure_object_entry<'a>(
    object: &'a mut serde_json::Map<String, Value>,
    key: &str,
) -> &'a mut serde_json::Map<String, Value> {
    let value = object
        .entry(key.to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !value.is_object() {
        *value = Value::Object(serde_json::Map::new());
    }
    value
        .as_object_mut()
        .expect("object entry should be an object after normalization")
}

fn read_json_with_backup(path: &Path) -> Result<(Value, Option<PathBuf>)> {
    let mut existing = json!({});
    let mut backup = None;

    if path.exists() {
        let original_name = path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_else(|| "config.json".to_string());
        let backup_path = path.with_file_name(format!("{original_name}.bak-{}", epoch_seconds()));
        fs::copy(path, &backup_path).with_context(|| {
            format!(
                "failed to create backup {} from {}",
                backup_path.display(),
                path.display()
            )
        })?;
        backup = Some(backup_path);

        existing = fs::read_to_string(path)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .unwrap_or_else(|| json!({}));
    }

    Ok((existing, backup))
}

fn epoch_seconds() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_factory_custom_model_ids_from_display_names() {
        assert_eq!(
            factory_custom_model_id("GPT-5.4 (XHigh)", 24),
            "custom:GPT-5.4-(XHigh)-24"
        );
    }

    #[test]
    fn default_catalog_includes_gpt_5_5() {
        let models = resolve_model_ids("");

        assert!(models.contains(&"gpt-5.5".to_string()));
        assert_eq!(model_display_name("gpt-5.5"), "GPT-5.5");
    }

    #[test]
    fn default_catalog_includes_gpt_5_5_fast_selectable_model() {
        let models = resolve_model_ids("");

        assert!(models.contains(&"gpt-5.5-fast".to_string()));
        assert_eq!(model_display_name("gpt-5.5-fast"), "GPT-5.5 Fast");
    }

    #[test]
    fn default_catalog_uses_factory_reasoning_selector_instead_of_model_aliases() {
        let models = resolve_model_ids("");

        assert!(models.iter().all(|model| !has_reasoning_alias(model)));
    }

    #[test]
    fn default_catalog_does_not_shift_existing_factory_model_ids() {
        let models = resolve_model_ids("");
        let (settings, _, _, _) = merge_factory_settings_document(
            json!({}),
            "http://127.0.0.1:42069",
            "opengateway-local",
            &models,
        )
        .expect("settings merge should succeed");

        assert_eq!(
            factory_model_id_for_model(&settings, "gpt-5.4").as_deref(),
            Some("custom:GPT-5.4-0")
        );
        assert_eq!(
            factory_model_id_for_model(&settings, "gpt-5.5").as_deref(),
            Some("custom:GPT-5.5-26")
        );
        assert_eq!(
            factory_model_id_for_model(&settings, "gpt-5.5-fast").as_deref(),
            Some("custom:GPT-5.5-Fast-27")
        );
    }

    #[test]
    fn repairs_shifted_managed_model_ids_and_defaults() {
        let existing = json!({
            "customModels": [
                {
                    "model": "gpt-5.5",
                    "id": "custom:GPT-5.5-0",
                    "index": 0,
                    "baseUrl": "http://127.0.0.1:42069/v1",
                    "apiKey": "secret",
                    "displayName": "GPT-5.5",
                    "provider": "openai"
                },
                {
                    "model": "gpt-5.4",
                    "id": "custom:GPT-5.4-5",
                    "index": 5,
                    "baseUrl": "http://127.0.0.1:42069/v1",
                    "apiKey": "secret",
                    "displayName": "GPT-5.4",
                    "provider": "openai"
                }
            ],
            "sessionDefaultSettings": {
                "model": "custom:GPT-5.5-0",
                "reasoningEffort": "xhigh"
            }
        });

        let models = resolve_model_ids("");
        let (settings, _, _, defaults_updated) =
            merge_factory_settings_document(existing, "http://127.0.0.1:42069", "secret", &models)
                .expect("settings merge should succeed");

        assert!(defaults_updated);
        assert_eq!(
            factory_model_id_for_model(&settings, "gpt-5.4").as_deref(),
            Some("custom:GPT-5.4-0")
        );
        assert_eq!(
            factory_model_id_for_model(&settings, "gpt-5.5").as_deref(),
            Some("custom:GPT-5.5-26")
        );
        assert_eq!(
            settings
                .get("sessionDefaultSettings")
                .and_then(Value::as_object)
                .and_then(|settings| settings.get("model"))
                .and_then(Value::as_str),
            Some("custom:GPT-5.5-26")
        );
    }

    #[test]
    fn prunes_stale_managed_reasoning_aliases_from_settings() {
        let existing = json!({
            "customModels": [
                {
                    "model": "gpt-5.4(xhigh)",
                    "id": "custom:GPT-5.4-(XHigh)-4",
                    "index": 4,
                    "baseUrl": "http://127.0.0.1:42069/v1",
                    "apiKey": "stale-key",
                    "displayName": "GPT-5.4 (XHigh)",
                    "provider": "openai"
                },
                {
                    "model": "gpt-5.5",
                    "id": "custom:GPT-5.5-26",
                    "index": 26,
                    "baseUrl": "http://127.0.0.1:42069/v1",
                    "apiKey": "secret",
                    "displayName": "GPT-5.5",
                    "provider": "openai"
                },
                {
                    "model": "local-test-model",
                    "id": "custom:Local-Test-99",
                    "index": 99,
                    "baseUrl": "http://127.0.0.1:42069/v1",
                    "apiKey": "secret",
                    "displayName": "Local Test",
                    "provider": "openai"
                }
            ]
        });

        let models = resolve_model_ids("");
        let (settings, _, updated, _) =
            merge_factory_settings_document(existing, "http://127.0.0.1:42069", "secret", &models)
                .expect("settings merge should succeed");
        let configured_models = factory_models(&settings);

        assert!(updated > 0);
        assert!(!configured_models.contains(&"gpt-5.4(xhigh)".to_string()));
        assert!(configured_models.contains(&"gpt-5.5".to_string()));
        assert!(configured_models.contains(&"local-test-model".to_string()));
    }

    #[test]
    fn prunes_stale_managed_reasoning_aliases_from_legacy_config() {
        let root = std::env::temp_dir().join(format!(
            "opengateway-factory-prune-legacy-{}-{}",
            epoch_seconds(),
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("temp root should be created");
        let config_path = root.join("config.json");
        fs::write(
            &config_path,
            serde_json::to_string_pretty(&json!({
                "custom_models": [
                    {
                        "model_display_name": "GPT-5.4 (High)",
                        "model": "gpt-5.4(high)",
                        "base_url": "http://127.0.0.1:42069/v1",
                        "api_key": "stale-key",
                        "provider": "openai"
                    },
                    {
                        "model_display_name": "Local Test",
                        "model": "local-test-model",
                        "base_url": "http://127.0.0.1:42069/v1",
                        "api_key": "secret",
                        "provider": "openai"
                    }
                ]
            }))
            .expect("legacy config JSON should encode"),
        )
        .expect("legacy config should be written");

        let models = resolve_model_ids("");
        let (_, updated, _) =
            merge_factory_config(&config_path, "http://127.0.0.1:42069", "secret", &models)
                .expect("legacy config merge should succeed");
        let config = fs::read_to_string(&config_path).expect("legacy config should be readable");
        let configured_models = legacy_config_models(
            &serde_json::from_str::<Value>(&config).expect("legacy config should be JSON"),
        );

        assert!(updated > 0);
        assert!(!configured_models.contains(&"gpt-5.4(high)".to_string()));
        assert!(configured_models.contains(&"local-test-model".to_string()));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn merges_factory_settings_and_upgrades_managed_defaults() {
        let existing = json!({
            "customModels": [
                {
                    "model": "gpt-5.3-codex(xhigh)",
                    "id": "custom:GPT-5.3-Codex-(XHigh)-3",
                    "index": 3,
                    "baseUrl": "http://127.0.0.1:42069/v1",
                    "apiKey": "secret",
                    "displayName": "GPT-5.3 Codex (XHigh)",
                    "noImageSupport": false,
                    "provider": "openai"
                }
            ],
            "sessionDefaultSettings": {
                "model": "custom:GPT-5.3-Codex-(XHigh)-3",
                "reasoningEffort": "xhigh"
            },
            "missionModelSettings": {
                "orchestratorModel": "custom:GPT-5.3-Codex-(XHigh)-3",
                "orchestratorReasoningEffort": "none",
                "workerModel": "custom:GPT-5.3-Codex-(XHigh)-3",
                "workerReasoningEffort": "none",
                "validationWorkerModel": "custom:GPT-5.3-Codex-(XHigh)-3",
                "validationWorkerReasoningEffort": "none"
            }
        });

        let model_ids = vec!["gpt-5.3-codex(xhigh)".to_string(), "gpt-5.5".to_string()];

        let (merged, added, updated, defaults_updated) = merge_factory_settings_document(
            existing,
            "http://127.0.0.1:42069",
            "secret",
            &model_ids,
        )
        .expect("settings merge should succeed");

        assert_eq!(added, 1);
        assert_eq!(updated, 1);
        assert!(defaults_updated);

        let custom_models = merged
            .get("customModels")
            .and_then(Value::as_array)
            .expect("customModels should be an array");
        let preferred_model = custom_models
            .iter()
            .find(|entry| entry.get("model").and_then(Value::as_str) == Some("gpt-5.5"))
            .expect("gpt-5.5 should be present");
        let preferred_model_id = preferred_model
            .get("id")
            .and_then(Value::as_str)
            .expect("preferred custom model should have an id");

        assert_eq!(
            preferred_model
                .get("maxOutputTokens")
                .and_then(Value::as_u64),
            Some(16_384)
        );
        assert_eq!(
            merged
                .get("sessionDefaultSettings")
                .and_then(Value::as_object)
                .and_then(|settings| settings.get("model"))
                .and_then(Value::as_str),
            Some(preferred_model_id)
        );
        assert_eq!(
            merged
                .get("missionModelSettings")
                .and_then(Value::as_object)
                .and_then(|settings| settings.get("workerReasoningEffort"))
                .and_then(Value::as_str),
            Some("xhigh")
        );
    }

    #[test]
    fn factory_settings_merge_is_idempotent_when_managed_models_match() {
        let models = resolve_model_ids("");
        let (settings, _, _, _) = merge_factory_settings_document(
            json!({}),
            "http://127.0.0.1:42069",
            "opengateway-local",
            &models,
        )
        .expect("settings merge should succeed");

        let (_, added, updated, defaults_updated) = merge_factory_settings_document(
            settings,
            "http://127.0.0.1:42069",
            "opengateway-local",
            &models,
        )
        .expect("settings merge should succeed");

        assert_eq!(added, 0);
        assert_eq!(updated, 0);
        assert!(!defaults_updated);
    }

    #[test]
    fn creates_factory_files_for_first_run_setup() {
        let root =
            std::env::temp_dir().join(format!("opengateway-factory-first-run-{}", epoch_seconds()));
        let _ = fs::remove_dir_all(&root);

        let config_path = root.join("config.json");
        let settings_path = root.join("settings.json");
        let models = vec!["gpt-5.5".to_string()];

        let (config_added, config_updated, config_backup) = merge_factory_config(
            &config_path,
            "http://127.0.0.1:42069",
            "opengateway-local",
            &models,
        )
        .expect("legacy config merge should succeed");
        let (settings_added, settings_updated, settings_backup, defaults_updated) =
            merge_factory_settings(
                &settings_path,
                "http://127.0.0.1:42069",
                "opengateway-local",
                &models,
            )
            .expect("settings merge should succeed");

        assert_eq!((config_added, config_updated), (1, 0));
        assert_eq!((settings_added, settings_updated), (1, 0));
        assert!(config_backup.is_none());
        assert!(settings_backup.is_none());
        assert!(defaults_updated);

        let config = fs::read_to_string(&config_path).expect("legacy config should be written");
        let settings = fs::read_to_string(&settings_path).expect("settings should be written");
        assert!(config.contains("\"custom_models\""));
        assert!(settings.contains("\"customModels\""));
        assert!(settings.contains("\"sessionDefaultSettings\""));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn factory_settings_sync_check_detects_stale_api_key() {
        let root = std::env::temp_dir().join(format!(
            "opengateway-factory-sync-check-{}-{}",
            epoch_seconds(),
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);

        let config_path = root.join("config.json");
        let settings_path = root.join("settings.json");
        let models = vec!["gpt-5.5".to_string()];

        sync_factory_files(
            &config_path,
            &settings_path,
            "http://127.0.0.1:42069",
            "old-local-key",
            &models,
        )
        .expect("factory sync should succeed");

        assert!(!factory_settings_needs_sync(
            &settings_path,
            "http://127.0.0.1:42069",
            "old-local-key",
            &models
        )
        .expect("settings sync check should succeed"));
        assert!(factory_settings_needs_sync(
            &settings_path,
            "http://127.0.0.1:42069",
            "rotated-local-key",
            &models
        )
        .expect("settings sync check should succeed"));

        sync_factory_files(
            &config_path,
            &settings_path,
            "http://127.0.0.1:42069",
            "rotated-local-key",
            &models,
        )
        .expect("factory sync should succeed");

        assert!(!factory_settings_needs_sync(
            &settings_path,
            "http://127.0.0.1:42069",
            "rotated-local-key",
            &models
        )
        .expect("settings sync check should succeed"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn sync_factory_files_updates_legacy_config_and_settings() {
        let root = std::env::temp_dir().join(format!(
            "opengateway-factory-sync-files-{}",
            epoch_seconds()
        ));
        let _ = fs::remove_dir_all(&root);

        let config_path = root.join("config.json");
        let settings_path = root.join("settings.json");
        let models = vec!["gpt-5.5".to_string()];

        let result = sync_factory_files(
            &config_path,
            &settings_path,
            "http://127.0.0.1:42069",
            "opengateway-local",
            &models,
        )
        .expect("factory sync should succeed");

        assert_eq!((result.legacy_added, result.legacy_updated), (1, 0));
        assert_eq!((result.settings_added, result.settings_updated), (1, 0));
        assert!(result.legacy_backup.is_none());
        assert!(result.settings_backup.is_none());
        assert!(result.defaults_updated);

        let config = fs::read_to_string(&config_path).expect("legacy config should be written");
        let settings = fs::read_to_string(&settings_path).expect("settings should be written");
        assert!(config.contains("\"custom_models\""));
        assert!(settings.contains("\"customModels\""));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn first_run_defaults_to_gpt_5_5_with_xhigh_reasoning() {
        let models = resolve_model_ids("");
        let (settings, _, _, defaults_updated) = merge_factory_settings_document(
            json!({}),
            "http://127.0.0.1:42069",
            "opengateway-local",
            &models,
        )
        .expect("settings merge should succeed");
        let custom_models = settings
            .get("customModels")
            .and_then(Value::as_array)
            .expect("customModels should be an array");
        let preferred_model = custom_models
            .iter()
            .find(|entry| entry.get("model").and_then(Value::as_str) == Some("gpt-5.5"))
            .expect("gpt-5.5 should be present");
        let preferred_model_id = preferred_model
            .get("id")
            .and_then(Value::as_str)
            .expect("preferred custom model should have an id");

        assert!(defaults_updated);
        assert_eq!(
            settings
                .get("sessionDefaultSettings")
                .and_then(Value::as_object)
                .and_then(|settings| settings.get("model"))
                .and_then(Value::as_str),
            Some(preferred_model_id)
        );
        assert_eq!(
            settings
                .get("sessionDefaultSettings")
                .and_then(Value::as_object)
                .and_then(|settings| settings.get("reasoningEffort"))
                .and_then(Value::as_str),
            Some("xhigh")
        );
    }

    fn factory_model_id_for_model(settings: &Value, model_id: &str) -> Option<String> {
        settings
            .get("customModels")
            .and_then(Value::as_array)?
            .iter()
            .find(|entry| entry.get("model").and_then(Value::as_str) == Some(model_id))?
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    fn factory_models(settings: &Value) -> Vec<String> {
        settings
            .get("customModels")
            .and_then(Value::as_array)
            .map(|models| {
                models
                    .iter()
                    .filter_map(|entry| entry.get("model").and_then(Value::as_str))
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn legacy_config_models(config: &Value) -> Vec<String> {
        config
            .get("custom_models")
            .and_then(Value::as_array)
            .map(|models| {
                models
                    .iter()
                    .filter_map(|entry| entry.get("model").and_then(Value::as_str))
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn has_reasoning_alias(model_id: &str) -> bool {
        ["(low)", "(medium)", "(high)", "(xhigh)"]
            .iter()
            .any(|suffix| model_id.ends_with(suffix))
    }
}
