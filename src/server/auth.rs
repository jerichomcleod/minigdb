//! Authentication configuration for minigdb server.
//!
//! Config is stored in `<data_root>/server.toml`.

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// One entry in the `[[users]]` section of `server.toml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UserEntry {
    pub name: String,
    /// SHA-256 hash stored as `"sha256:<hex>"`.
    pub password_hash: String,
    /// Graph names this user may access. `["*"]` means all graphs.
    pub graphs: Vec<String>,
}

impl UserEntry {
    /// Returns `true` if this user is allowed to access `graph`.
    ///
    /// A wildcard entry `"*"` in `graphs` grants access to all graphs.
    pub fn can_access(&self, graph: &str) -> bool {
        self.graphs.iter().any(|g| g == "*" || g == graph)
    }
}

/// The `[server]` section of `server.toml`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerSection {
    #[serde(default = "default_true")]
    pub auth_required: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ServerSection {
    fn default() -> Self {
        Self { auth_required: true }
    }
}

/// Top-level `server.toml` structure.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ServerConfig {
    #[serde(default)]
    pub server: ServerSection,
    #[serde(default)]
    pub users: Vec<UserEntry>,
}

impl ServerConfig {
    /// Load from `<data_root>/server.toml`, returning `Default` on missing/parse error.
    pub fn load(data_root: &Path) -> Self {
        let path = data_root.join("server.toml");
        let Ok(content) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        toml::from_str(&content).unwrap_or_default()
    }

    /// Write to `<data_root>/server.toml`.
    pub fn save(&self, data_root: &Path) -> io::Result<()> {
        let path = data_root.join("server.toml");
        let content = toml::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        std::fs::write(path, content)
    }

    /// Find a user entry by name (case-sensitive).
    pub fn find_user(&self, name: &str) -> Option<&UserEntry> {
        self.users.iter().find(|u| u.name == name)
    }

    /// Find a user entry by name for mutation (e.g. password change).
    pub fn find_user_mut(&mut self, name: &str) -> Option<&mut UserEntry> {
        self.users.iter_mut().find(|u| u.name == name)
    }
}

/// Hash a plaintext password to `"sha256:<hex>"`.
pub fn hash_password(pw: &str) -> String {
    let hash = Sha256::digest(pw.as_bytes());
    format!("sha256:{:x}", hash)
}

/// Verify a plaintext password against a stored hash.
pub fn verify_password(pw: &str, stored: &str) -> bool {
    hash_password(pw) == stored
}
