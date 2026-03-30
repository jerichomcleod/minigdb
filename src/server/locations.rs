//! Persistent configuration for additional graph-root directories.
//!
//! Stored as `<data_root>/locations.toml`.  The default root
//! (`<data_root>/graphs/`) is always present and is never written to this
//! file; only user-added extra roots are persisted here.
//!
//! # File format
//! ```toml
//! [[location]]
//! path = "/Users/me/projects/myproject/graphs"
//!
//! [[location]]
//! path = "/mnt/shared/graphs"
//! ```

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ── Types ─────────────────────────────────────────────────────────────────────

/// A single extra graph-root directory entry.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct LocationEntry {
    /// Absolute (or home-relative) path to a directory that will be scanned
    /// for graph subdirectories alongside the default root.
    pub path: PathBuf,
}

/// Top-level `locations.toml` structure.
///
/// Wraps a list of [`LocationEntry`] values under the `[[location]]` TOML key.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct LocationsConfig {
    /// Extra graph-root directories.  Serialised as `[[location]]` array.
    #[serde(default, rename = "location")]
    pub locations: Vec<LocationEntry>,
}

impl LocationsConfig {
    /// Load from `<data_root>/locations.toml`, returning an empty config on
    /// missing file or parse error.
    pub fn load(data_root: &Path) -> Self {
        let path = data_root.join("locations.toml");
        let Ok(content) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        toml::from_str(&content).unwrap_or_default()
    }

    /// Write to `<data_root>/locations.toml`.
    pub fn save(&self, data_root: &Path) -> io::Result<()> {
        let path = data_root.join("locations.toml");
        let content = toml::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        std::fs::write(path, content)
    }

    /// Return the list of extra root paths.
    pub fn paths(&self) -> Vec<PathBuf> {
        self.locations.iter().map(|l| l.path.clone()).collect()
    }

    /// Add `path` to the list.  Returns `false` (no-op) if already present.
    pub fn add(&mut self, path: PathBuf) -> bool {
        if self.locations.iter().any(|l| l.path == path) {
            return false;
        }
        self.locations.push(LocationEntry { path });
        true
    }

    /// Remove `path` from the list.  Returns `false` if it was not present.
    pub fn remove(&mut self, path: &Path) -> bool {
        let before = self.locations.len();
        self.locations.retain(|l| l.path != path);
        self.locations.len() < before
    }
}
