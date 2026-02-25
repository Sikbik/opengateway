use anyhow::{Context, Result};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const STORE_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct NewCredential {
    pub refresh_token: String,
    pub access_token: String,
    pub expires_at_ms: i64,
    pub account_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthCredential {
    pub provider: String,
    pub refresh_token: String,
    pub access_token: String,
    pub expires_at_ms: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuthStoreFile {
    version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_account: Option<String>,
    #[serde(default)]
    accounts: BTreeMap<String, OAuthCredential>,
}

impl Default for AuthStoreFile {
    fn default() -> Self {
        Self {
            version: STORE_VERSION,
            active_account: None,
            accounts: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AuthStore {
    path: PathBuf,
}

impl AuthStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn upsert_openai(&self, input: NewCredential) -> Result<String> {
        self.with_locked_store(|store| {
            let now = now_millis();
            let key = account_key(input.account_id.as_deref());

            let created_at_ms = store
                .accounts
                .get(&key)
                .map(|existing| existing.created_at_ms)
                .unwrap_or(now);

            let credential = OAuthCredential {
                provider: "openai".to_string(),
                refresh_token: input.refresh_token,
                access_token: input.access_token,
                expires_at_ms: input.expires_at_ms,
                account_id: input.account_id,
                created_at_ms,
                updated_at_ms: now,
            };

            store.accounts.insert(key.clone(), credential);
            store.active_account = Some(key.clone());
            Ok(key)
        })
    }

    pub fn count_openai_accounts(&self) -> Result<usize> {
        self.with_locked_store(|store| {
            let count = store
                .accounts
                .values()
                .filter(|item| item.provider.eq_ignore_ascii_case("openai"))
                .count();
            Ok(count)
        })
    }

    pub fn active_openai(&self) -> Result<Option<OAuthCredential>> {
        self.with_locked_store(|store| {
            if let Some(active_key) = &store.active_account {
                if let Some(active) = store.accounts.get(active_key) {
                    if active.provider.eq_ignore_ascii_case("openai") {
                        return Ok(Some(active.clone()));
                    }
                }
            }

            let fallback = store
                .accounts
                .values()
                .find(|item| item.provider.eq_ignore_ascii_case("openai"))
                .cloned();
            Ok(fallback)
        })
    }

    fn with_locked_store<T>(&self, op: impl FnOnce(&mut AuthStoreFile) -> Result<T>) -> Result<T> {
        let parent = self
            .path
            .parent()
            .context("auth store path has no parent directory")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create auth dir {}", parent.display()))?;

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.path)
            .with_context(|| format!("failed to open auth store {}", self.path.display()))?;

        #[cfg(unix)]
        {
            let permissions = fs::Permissions::from_mode(0o600);
            file.set_permissions(permissions)
                .with_context(|| format!("failed to set permissions on {}", self.path.display()))?;
        }

        file.lock_exclusive()
            .with_context(|| format!("failed to lock auth store {}", self.path.display()))?;

        let mut store = load_store(&mut file)
            .with_context(|| format!("failed to read auth store {}", self.path.display()))?;
        let outcome = op(&mut store);

        let persist_result = match outcome {
            Ok(value) => {
                save_store(&mut file, &store).with_context(|| {
                    format!("failed to persist auth store {}", self.path.display())
                })?;
                Ok(value)
            }
            Err(err) => Err(err),
        };

        let _ = file.unlock();
        persist_result
    }
}

fn load_store(file: &mut std::fs::File) -> Result<AuthStoreFile> {
    file.seek(SeekFrom::Start(0))?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)?;
    if raw.trim().is_empty() {
        return Ok(AuthStoreFile::default());
    }

    let mut parsed: AuthStoreFile =
        serde_json::from_str(&raw).context("invalid auth store JSON")?;
    if parsed.version == 0 {
        parsed.version = STORE_VERSION;
    }
    Ok(parsed)
}

fn save_store(file: &mut std::fs::File, store: &AuthStoreFile) -> Result<()> {
    let mut normalized = store.clone();
    normalized.version = STORE_VERSION;

    let payload =
        serde_json::to_string_pretty(&normalized).context("failed to encode auth store")?;
    file.seek(SeekFrom::Start(0))?;
    file.set_len(0)?;
    file.write_all(payload.as_bytes())?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn account_key(account_id: Option<&str>) -> String {
    match account_id {
        Some(value) if !value.trim().is_empty() => format!("openai:{}", value.trim()),
        _ => "openai:default".to_string(),
    }
}

fn now_millis() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_millis() as i64
}
