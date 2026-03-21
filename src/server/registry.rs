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

/// `(graph, next_txn_id)` — the shared mutable state for one named graph.
pub type GraphState = (Graph, u64);

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
    pub async fn get_or_open(
        &self,
        name: &str,
    ) -> Result<Arc<Mutex<GraphState>>, DbError> {
        let mut map = self.open.lock().await;
        if let Some(g) = map.get(name) {
            return Ok(Arc::clone(g));
        }
        // User-facing graphs are validated; system graphs bypass this check.
        if !name.starts_with('_') {
            validate_graph_name(name)?;
        }
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
        let state = Arc::new(Mutex::new((graph, 0u64)));
        map.insert(name.to_string(), Arc::clone(&state));
        Ok(state)
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
        // Remove and drop the Arc — if other Arcs exist (e.g. a connection
        // mid-transaction) the Graph stays alive until they release it.
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
            let (graph, _) = &*guard;
            if let Err(e) = graph.store.flush() {
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
