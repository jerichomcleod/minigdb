//! Multi-graph registry for the minigdb server.
//!
//! Graphs are opened lazily on first access and kept open for the lifetime
//! of the server process.  Each graph gets its own `tokio::sync::Mutex` so
//! that per-connection transactions can hold the lock for a whole
//! BEGIN … COMMIT span without blocking other graphs.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::types::DbError;
use crate::Graph;

/// The shared mutable state for one named graph.
pub struct GraphState {
    pub graph: Graph,
    pub txn_id: u64,
    /// Set to `true` by [`GraphRegistry::drop_graph`] before the directory is
    /// removed.  Any connection that still holds an `Arc` to this state should
    /// check this flag before executing queries.
    pub dropped: bool,
}

/// Central registry of all open graphs.
///
/// Wrapped in `Arc` so it can be shared across Tokio tasks cheaply.
pub struct GraphRegistry {
    graphs_dir: PathBuf,
    open: Mutex<HashMap<String, Arc<Mutex<GraphState>>>>,
}

impl GraphRegistry {
    /// Create a new registry rooted at `graphs_dir`.
    /// Returns an `Arc` so it can be cheaply cloned into Tokio tasks.
    pub fn new(graphs_dir: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            graphs_dir,
            open: Mutex::new(HashMap::new()),
        })
    }

    /// Return the mutex for `name`, opening the graph from disk if needed.
    ///
    /// System graph names (starting with `_`) bypass user-facing name validation
    /// so that internal graphs such as `_meta` can always be opened.
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
    pub async fn get_or_open(
        &self,
        name: &str,
    ) -> Result<Arc<Mutex<GraphState>>, DbError> {
        // Step 1: fast path — graph already open.
        {
            let map = self.open.lock().await;
            if let Some(g) = map.get(name) {
                return Ok(Arc::clone(g));
            }
        }
        // Lock released here.

        // Validate name before touching the filesystem.
        if !name.starts_with('_') {
            validate_graph_name(name)?;
        }

        // Step 3: open without holding the registry lock.
        let path = self.graphs_dir.join(name);
        std::fs::create_dir_all(&path).map_err(DbError::Storage)?;
        let graph = Graph::open(&path).map_err(|e| {
            let msg = e.to_string();
            // RocksDB lock failures contain "LOCK" or "lock file" in the message.
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
            // Another task won the race — drop our freshly-opened graph and
            // return the one that's already in the map.
            return Ok(Arc::clone(existing));
        }
        map.insert(name.to_string(), Arc::clone(&new_state));
        Ok(new_state)
    }

    /// Create a new named graph (also opens it in the registry).
    pub async fn create(&self, name: &str) -> Result<(), DbError> {
        self.get_or_open(name).await?;
        Ok(())
    }

    /// Drop (delete) a named graph.  Removes it from the open map and deletes
    /// the directory on disk.
    ///
    /// **Warning**: if any connection currently holds the graph mutex the drop
    /// will still succeed (the directory removal happens immediately).  Open
    /// handles will see errors or stale data.  Callers should ensure no active
    /// transactions exist on the graph before dropping it.
    pub async fn drop_graph(&self, name: &str) -> Result<(), DbError> {
        let mut map = self.open.lock().await;
        // Mark the state as dropped *before* removing from the map so that any
        // connection currently holding an Arc<Mutex<GraphState>> can detect the
        // sentinel and return a clear error instead of getting raw I/O failures.
        if let Some(arc) = map.get(name) {
            let mut state = arc.lock().await;
            state.dropped = true;
        }
        // Remove from the registry — if other Arcs exist the Graph stays alive
        // until they release it, but the `dropped` flag will tell them to bail.
        map.remove(name);
        let path = self.graphs_dir.join(name);
        if path.exists() {
            std::fs::remove_dir_all(&path).map_err(DbError::Storage)?;
        }
        Ok(())
    }

    /// List all user-visible graph names found in `graphs_dir`.
    ///
    /// System graphs (names starting with `_`) are excluded from the listing.
    pub async fn list(&self) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(&self.graphs_dir) else {
            return Vec::new();
        };
        let mut names: Vec<String> = entries
            .filter_map(|e| {
                let e = e.ok()?;
                if e.file_type().ok()?.is_dir() {
                    let name = e.file_name().into_string().ok()?;
                    // Hide system graphs from user-facing listings.
                    if name.starts_with('_') { None } else { Some(name) }
                } else {
                    None
                }
            })
            .collect();
        names.sort();
        names
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
fn validate_graph_name(name: &str) -> Result<(), DbError> {
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
