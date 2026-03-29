//! Authentication configuration for minigdb server.
//!
//! Config is stored in `<data_root>/server.toml`.

use std::io;
use std::path::Path;

use rand::RngCore;
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

/// Constant-time byte-slice equality to prevent timing attacks.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Hash a plaintext password to `"sha256s:<salt_hex>:<hash_hex>"` using a
/// random 16-byte salt.  The `sha256s` prefix distinguishes salted hashes from
/// legacy unsalted `sha256:` hashes.
pub fn hash_password(pw: &str) -> String {
    let mut salt = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut salt);
    let salt_hex: String = salt.iter().map(|b| format!("{b:02x}")).collect();
    let mut hasher = Sha256::new();
    hasher.update(&salt);
    hasher.update(pw.as_bytes());
    let hash = hasher.finalize();
    let hash_hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    format!("sha256s:{salt_hex}:{hash_hex}")
}

/// Verify a plaintext password against a stored hash.
///
/// Supports both the legacy unsalted `"sha256:<hex>"` format and the new
/// salted `"sha256s:<salt_hex>:<hash_hex>"` format.  Comparison is performed
/// in constant time to prevent timing-based enumeration of valid passwords.
pub fn verify_password(pw: &str, stored: &str) -> bool {
    if let Some(rest) = stored.strip_prefix("sha256s:") {
        // New salted format: sha256s:<salt_hex>:<hash_hex>
        let mut parts = rest.splitn(2, ':');
        let (Some(salt_hex), Some(hash_hex)) = (parts.next(), parts.next()) else {
            return false;
        };
        // Decode salt from hex.
        if salt_hex.len() % 2 != 0 {
            return false;
        }
        let salt: Vec<u8> = (0..salt_hex.len())
            .step_by(2)
            .filter_map(|i| u8::from_str_radix(&salt_hex[i..i + 2], 16).ok())
            .collect();
        if salt.len() != salt_hex.len() / 2 {
            return false;
        }
        // Compute sha256(salt || password).
        let mut hasher = Sha256::new();
        hasher.update(&salt);
        hasher.update(pw.as_bytes());
        let computed = hasher.finalize();
        let computed_hex: String = computed.iter().map(|b| format!("{b:02x}")).collect();
        ct_eq(computed_hex.as_bytes(), hash_hex.as_bytes())
    } else if let Some(hash_hex) = stored.strip_prefix("sha256:") {
        // Legacy unsalted format: sha256:<hex>
        let legacy = Sha256::digest(pw.as_bytes());
        let legacy_hex: String = legacy.iter().map(|b| format!("{b:02x}")).collect();
        ct_eq(legacy_hex.as_bytes(), hash_hex.as_bytes())
    } else {
        false
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Password hashing (MED-1) ──────────────────────────────────────────────

    /// Two calls to `hash_password` for the same password must produce different
    /// hashes because each uses a fresh random salt.
    #[test]
    fn hash_password_unique_salt_per_call() {
        let h1 = hash_password("secret");
        let h2 = hash_password("secret");
        assert_ne!(h1, h2, "same password must produce different hashes due to random salt");
        assert!(h1.starts_with("sha256s:"), "hash must use the new salted prefix");
    }

    /// `verify_password` must accept the correct plaintext against a freshly
    /// generated salted hash.
    #[test]
    fn verify_password_correct_password_accepted() {
        let hash = hash_password("correct-horse");
        assert!(verify_password("correct-horse", &hash), "correct password must be accepted");
    }

    /// `verify_password` must reject an incorrect plaintext.
    #[test]
    fn verify_password_wrong_password_rejected() {
        let hash = hash_password("correct-horse");
        assert!(!verify_password("battery-staple", &hash), "wrong password must be rejected");
    }

    /// `verify_password` must still work with the legacy unsalted `sha256:` format
    /// so that existing user records remain valid after the upgrade.
    #[test]
    fn verify_password_legacy_unsalted_format_accepted() {
        use sha2::{Digest, Sha256};
        let pw = "legacy-password";
        let hex: String = Sha256::digest(pw.as_bytes()).iter()
            .map(|b| format!("{b:02x}")).collect();
        let legacy_hash = format!("sha256:{hex}");
        assert!(verify_password(pw, &legacy_hash), "legacy unsalted hash must still verify");
    }

    /// `verify_password` must return `false` for an unrecognised hash format
    /// rather than panicking or returning `true`.
    #[test]
    fn verify_password_unknown_format_rejected() {
        assert!(!verify_password("any", "md5:abc123"), "unknown hash format must be rejected");
        assert!(!verify_password("any", ""), "empty stored hash must be rejected");
    }
}
