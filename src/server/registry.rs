//! Multi-graph registry for the minigdb server.
//!
//! Graphs are opened lazily on first access and kept open for the lifetime of
//! the server process.  Each graph gets its own `tokio::sync::Mutex` so that
//! per-connection transactions can hold the lock for a whole BEGIN … COMMIT
//! span without blocking other graphs.
//!
//! # Multiple graph roots
//!
//! The registry supports more than one root directory.  The *primary root*
//! (`<data_root>/graphs/`) is always present.  Extra roots can be added at
//! runtime via [`GraphRegistry::add_location`] (persisted to
//! `<data_root>/locations.toml`) or supplied as session-only paths at startup.
//!
//! When looking up a graph by name the registry searches roots in order:
//! primary first, then extra roots in the order they were added.  The first
//! root that contains a matching subdirectory wins.  If no existing directory
//! is found the graph is created under the primary root.
//!
//! Name conflicts (same directory name appearing in two roots) are resolved by
//! first-root-wins; a warning is printed when a duplicate is detected in
//! [`GraphRegistry::list`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::types::DbError;
use crate::Graph;
use super::locations::LocationsConfig;

/// The shared mutable state for one named graph.
pub struct GraphState {
    pub graph: Graph,
    pub txn_id: u64,
    /// Set to `true` by [`GraphRegistry::drop_graph`] before the directory is
    /// removed.  Any connection that still holds an `Arc` to this state should
    /// check this flag before executing queries.
    pub dropped: bool,
}

/// Central registry of all open graphs, supporting multiple root directories.
///
/// Wrapped in `Arc` so it can be shared across Tokio tasks cheaply.
pub struct GraphRegistry {
    /// The default root: `<data_root>/graphs/`.  New graphs are always created
    /// here.  Never removed from the roots list.
    primary_root: PathBuf,
    /// Server data root — used to persist changes to `locations.toml`.
    data_root: PathBuf,
    /// User-added extra root directories.  Includes both roots loaded from
    /// `locations.toml` at startup and session-only roots supplied via
    /// `--graphs-dir`.  Access is guarded so roots can be mutated at runtime.
    extra_roots: Mutex<Vec<PathBuf>>,
    /// Currently-open graphs, keyed by graph name.
    open: Mutex<HashMap<String, Arc<Mutex<GraphState>>>>,
}

impl GraphRegistry {
    /// Create a new registry.
    ///
    /// - `data_root` — the server data directory (e.g. `~/.local/share/minigdb`).
    ///   The primary graph root is `<data_root>/graphs/`.
    /// - `session_roots` — additional roots added for this server session only
    ///   (e.g. from `--graphs-dir` on the CLI).  Not persisted to disk.
    ///
    /// Persistent extra roots are loaded automatically from
    /// `<data_root>/locations.toml`.
    pub fn new(data_root: PathBuf, session_roots: Vec<PathBuf>) -> Arc<Self> {
        let primary_root = data_root.join("graphs");
        let persisted = LocationsConfig::load(&data_root).paths();
        // Persisted roots come first, then session-only roots.
        // Deduplicate while preserving order.
        let mut seen = std::collections::HashSet::new();
        seen.insert(primary_root.clone());
        let mut extra: Vec<PathBuf> = Vec::new();
        for p in persisted.into_iter().chain(session_roots) {
            if seen.insert(p.clone()) {
                extra.push(p);
            }
        }
        Arc::new(Self {
            primary_root,
            data_root,
            extra_roots: Mutex::new(extra),
            open: Mutex::new(HashMap::new()),
        })
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Return a snapshot of all roots: primary first, then extra roots.
    async fn all_roots(&self) -> Vec<PathBuf> {
        let extra = self.extra_roots.lock().await;
        std::iter::once(self.primary_root.clone())
            .chain(extra.iter().cloned())
            .collect()
    }

    /// Search `roots` for a subdirectory named `name` and return the first
    /// match.  Returns `None` if no root contains such a directory.
    fn find_path_in_roots(roots: &[PathBuf], name: &str) -> Option<PathBuf> {
        for root in roots {
            let candidate = root.join(name);
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
        None
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Return the mutex for `name`, opening the graph from disk if needed.
    ///
    /// System graph names (starting with `_`) bypass user-facing name
    /// validation so that internal graphs such as `_meta` can always be opened.
    ///
    /// The outer registry mutex is **not** held while `Graph::open()` runs.
    /// That call can take seconds on large graphs and would otherwise block all
    /// concurrent registry operations.  Instead we use a double-check pattern:
    ///
    /// 1. Check with lock held — return the existing entry if present.
    /// 2. Release the lock.
    /// 3. Open the graph without any lock.
    /// 4. Re-acquire the lock.
    /// 5. Check again — another task may have opened the same graph while we
    ///    were working (race winner wins; we discard our copy via `Drop`).
    /// 6. Insert and return our new entry if still absent.
    pub async fn get_or_open(&self, name: &str) -> Result<Arc<Mutex<GraphState>>, DbError> {
        // Step 1: fast path — graph already open.
        {
            let map = self.open.lock().await;
            if let Some(g) = map.get(name) {
                return Ok(Arc::clone(g));
            }
        }

        // Validate name before touching the filesystem.
        if !name.starts_with('_') {
            validate_graph_name(name)?;
        }

        // Step 3: determine the path and open without holding the registry lock.
        let roots = self.all_roots().await;
        let path = match Self::find_path_in_roots(&roots, name) {
            Some(p) => p,
            None => {
                // Not found in any root — create in the primary root.
                let p = self.primary_root.join(name);
                std::fs::create_dir_all(&p).map_err(DbError::Storage)?;
                p
            }
        };

        let graph = Graph::open(&path).map_err(|e| {
            let msg = e.to_string();
            if msg.contains("LOCK") || msg.contains("lock file") {
                crate::types::DbError::Query(format!(
                    "graph '{}' is locked by another process — stop the REPL or any \
                     other minigdb instance that has this graph open before using the server",
                    name
                ))
            } else {
                e
            }
        })?;
        let new_state = Arc::new(Mutex::new(GraphState { graph, txn_id: 0, dropped: false }));

        // Step 4–6: re-acquire lock, insert only if still absent.
        let mut map = self.open.lock().await;
        if let Some(existing) = map.get(name) {
            return Ok(Arc::clone(existing));
        }
        map.insert(name.to_string(), Arc::clone(&new_state));
        Ok(new_state)
    }

    /// Create a new named graph in the primary root (also opens it in the
    /// registry).
    pub async fn create(&self, name: &str) -> Result<(), DbError> {
        self.get_or_open(name).await?;
        Ok(())
    }

    /// Drop (delete) a named graph.  Removes it from the open map and deletes
    /// the graph directory on disk (searching all roots).
    ///
    /// **Warning**: if any connection currently holds the graph mutex the drop
    /// will still succeed.  Open handles will see errors or stale data.
    /// Callers should ensure no active transactions exist on the graph before
    /// dropping it.
    pub async fn drop_graph(&self, name: &str) -> Result<(), DbError> {
        // Find the path across all roots before locking the open map.
        let roots = self.all_roots().await;
        let path = Self::find_path_in_roots(&roots, name)
            .unwrap_or_else(|| self.primary_root.join(name));

        let mut map = self.open.lock().await;
        if let Some(arc) = map.get(name) {
            let mut state = arc.lock().await;
            state.dropped = true;
        }
        map.remove(name);
        if path.exists() {
            std::fs::remove_dir_all(&path).map_err(DbError::Storage)?;
        }
        Ok(())
    }

    /// List all user-visible graph names found across all roots.
    ///
    /// System graphs (names starting with `_`) are excluded.  When the same
    /// name appears in more than one root, the first root's copy is used and a
    /// warning is printed.
    pub async fn list(&self) -> Vec<String> {
        let roots = self.all_roots().await;
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut names: Vec<String> = Vec::new();

        for root in &roots {
            let Ok(entries) = std::fs::read_dir(root) else { continue };
            for entry in entries.flatten() {
                let Ok(ft) = entry.file_type() else { continue };
                if !ft.is_dir() { continue }
                let Ok(name) = entry.file_name().into_string() else { continue };
                if name.starts_with('_') { continue }
                // Only include directories that contain a RocksDB database
                // (identified by the presence of the "CURRENT" file that
                // RocksDB always creates).  This prevents non-graph subdirs
                // from appearing in the graph list when the user registers a
                // project folder as a location.
                if !entry.path().join("CURRENT").is_file() { continue }
                if !seen.insert(name.clone()) {
                    eprintln!(
                        "Warning: graph '{}' exists in multiple roots; \
                         using the first occurrence",
                        name
                    );
                    continue;
                }
                names.push(name);
            }
        }
        names.sort();
        names
    }

    /// Add a new root directory to the registry and persist it to
    /// `locations.toml`.
    ///
    /// Returns an error if `path` does not exist or is already registered.
    /// Session-only roots (added via `--graphs-dir`) can be re-added here to
    /// make them permanent.
    pub async fn add_location(&self, path: PathBuf) -> Result<(), DbError> {
        if !path.is_dir() {
            return Err(DbError::Query(format!(
                "location '{}' does not exist or is not a directory",
                path.display()
            )));
        }
        {
            let mut extra = self.extra_roots.lock().await;
            if self.primary_root == path || extra.contains(&path) {
                return Err(DbError::Query(format!(
                    "location '{}' is already registered",
                    path.display()
                )));
            }
            extra.push(path.clone());
        }
        // Persist.
        let mut cfg = LocationsConfig::load(&self.data_root);
        cfg.add(path);
        cfg.save(&self.data_root).map_err(DbError::Storage)?;
        Ok(())
    }

    /// Remove a root directory from the registry and update `locations.toml`.
    ///
    /// The primary root cannot be removed.  Returns an error if `path` is not
    /// a registered extra root.
    pub async fn remove_location(&self, path: &std::path::Path) -> Result<(), DbError> {
        if self.primary_root == path {
            return Err(DbError::Query(
                "cannot remove the primary graph root".to_string(),
            ));
        }
        {
            let mut extra = self.extra_roots.lock().await;
            let before = extra.len();
            extra.retain(|p| p != path);
            if extra.len() == before {
                return Err(DbError::Query(format!(
                    "location '{}' is not registered",
                    path.display()
                )));
            }
        }
        // Persist.
        let mut cfg = LocationsConfig::load(&self.data_root);
        cfg.remove(path);
        cfg.save(&self.data_root).map_err(DbError::Storage)?;
        Ok(())
    }

    /// List all registered root directories.
    ///
    /// Returns `(path, is_primary)` pairs in search order.
    pub async fn list_locations(&self) -> Vec<(PathBuf, bool)> {
        let extra = self.extra_roots.lock().await;
        let mut result = vec![(self.primary_root.clone(), true)];
        result.extend(extra.iter().map(|p| (p.clone(), false)));
        result
    }

    /// Flush RocksDB memtables for every currently-open graph.
    /// Called on graceful shutdown.
    pub async fn checkpoint_all(&self) {
        let map = self.open.lock().await;
        for (name, state) in map.iter() {
            let guard = state.lock().await;
            if guard.dropped {
                continue;
            }
            if let Err(e) = guard.graph.store.flush() {
                eprintln!("Warning: checkpoint of graph '{name}' failed: {e}");
            }
        }
    }
}

/// The reserved name for the system metadata graph.
///
/// This graph stores saved views and other internal metadata.
/// It is hidden from user-facing graph lists and cannot be created or dropped
/// by users.  System code may access it directly via [`GraphRegistry::get_or_open`].
pub const META_GRAPH: &str = "_meta";

/// Validate a graph name: alphanumeric + `_` + `-`, non-empty, ≤ 64 chars,
/// and must not start with `_` (that prefix is reserved for system graphs).
pub(crate) fn validate_graph_name(name: &str) -> Result<(), DbError> {
    if name.is_empty()
        || name.len() > 64
        || name.starts_with('_')
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        Err(DbError::Query(format!("invalid graph name '{name}'")))
    } else {
        Ok(())
    }
}
