//! [`Graph`] — the in-memory/RocksDB property graph and its full read/write API.
//!
//! All graph mutations go through the `apply_*` family of methods, which either
//! write directly to RocksDB (auto-commit mode) or buffer an [`Operation`] for
//! deferred commit (explicit-transaction mode).  Secondary column families
//! (`label_idx`, `adj_out`, `adj_in`, `prop_idx`, `edge_label_idx`) are
//! maintained atomically in the same [`rocksdb::WriteBatch`] as the primary
//! record, so the graph is always consistent even after a crash.

pub(crate) mod ops;
pub mod constraints;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{atomic::{AtomicU64, Ordering}, Arc};

use serde::{Deserialize, Serialize};

use crate::storage::rocks_store::RocksStore;
use crate::transaction::operation::Operation;
use crate::types::{ulid_new, DbError, Edge, EdgeId, Node, NodeId, Value};

// ── Property cache type alias ─────────────────────────────────────────────────

/// Concurrent LRU cache for deserialized `Node` objects (PERF-6).
///
/// `moka::sync::Cache` is thread-safe without external locking, making it safe
/// to read from Rayon parallel projection threads (PERF-12).  The cache stores
/// `Arc<Node>` so cloning a cached value is cheap (ref-count bump only).
///
/// Capacity: 100 K nodes — enough to keep hub nodes warm across queries on
/// typical graphs.  Entries are invalidated at `COMMIT` time (not at write-
/// buffer time) to preserve correct read-your-own-writes semantics.
type NodeCache = moka::sync::Cache<NodeId, Arc<Node>>;

// ── In-memory adjacency ───────────────────────────────────────────────────────

/// A single entry in the in-memory adjacency list.
///
/// `label_id` is an interned index into `Graph::adj_id_to_label`; using a
/// `u32` instead of a heap-allocated `String` reduces per-entry overhead from
/// ~32 bytes to 4 bytes — critical when storing 40 M+ edges × 2 directions.
#[derive(Clone)]
pub struct AdjEntry {
    pub edge_id: EdgeId,
    /// The node on the other end of this edge (to_node for adj_out,
    /// from_node for adj_in).
    pub neighbor: NodeId,
    /// Interned edge label ID; resolve via `Graph::adj_label_str`.
    pub label_id: u32,
}

/// A single tracked change to the in-memory adjacency list during a transaction.
///
/// Accumulated in `Graph::txn_adj_deltas` while a transaction is active.
/// On `ROLLBACK`, the list is reversed and each delta is inverted to restore
/// the pre-transaction adjacency state in O(ops_in_txn) rather than
/// re-scanning all edges from RocksDB (which is O(all_edges)).
#[derive(Clone)]
pub(crate) enum AdjDelta {
    AddOut  { from: NodeId, entry: AdjEntry },
    AddIn   { to:   NodeId, entry: AdjEntry },
    RemoveOut { from: NodeId, entry: AdjEntry },
    RemoveIn  { to:   NodeId, entry: AdjEntry },
}

/// A declared property index: index all nodes with `label` on the value of `property`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PropertyIndexDef {
    pub label: String,
    pub property: String,
}

// ── Temp-dir cleanup helper ──────────────────────────────────────────────────

/// Deletes the directory it wraps when dropped.
/// Used by `Graph::new()` so ephemeral (test) graphs clean up after themselves.
struct CleanupDir(PathBuf);

impl Drop for CleanupDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ── Graph ────────────────────────────────────────────────────────────────────

/// The property graph, backed by RocksDB.
///
/// All reads and writes go through `store`.  `index_defs` is maintained in
/// memory and persisted to the `meta` column family.  Property index data
/// lives in the `prop_idx` column family (maintained atomically in each
/// mutation WriteBatch).
///
/// **Field drop order matters**: `store` must be declared *before* `_owned_dir`
/// so that RocksDB is closed before the temp directory is deleted.
/// Pre-transaction snapshot used for rollback.
///
/// Captured at `BEGIN`.  All writes during the transaction go directly to
/// RocksDB so reads always see the latest state ("read your own writes").
/// On `ROLLBACK`, the graph's RocksDB data is wiped and rebuilt from this
/// snapshot.  On `COMMIT`, the snapshot is simply discarded.
struct TransactionSnapshot {
    nodes: Vec<crate::types::Node>,
    edges: Vec<crate::types::Edge>,
    index_defs: Vec<PropertyIndexDef>,
    constraint_defs: Vec<constraints::ConstraintDef>,
    prop_index_counts: HashMap<(String, String), usize>,
}

pub struct Graph {
    /// RocksDB-backed store — the single source of truth.
    /// MUST be declared first so it drops before `_owned_dir`.
    pub(crate) store: RocksStore,
    /// Declared property index definitions (persisted to meta CF).
    pub(crate) index_defs: Vec<PropertyIndexDef>,
    /// In-memory cache of property index entry counts.
    ///
    /// Maintained alongside every `put_prop_entry` / `delete_prop_entry` call so
    /// that `SHOW INDEXES` is a pure in-memory scan rather than N RocksDB reads.
    /// Also eliminates the synchronous read inside `adjust_prop_count_batch` on
    /// every node insert/update/delete.  Loaded from the meta CF on `open()`.
    pub(crate) prop_index_counts: HashMap<(String, String), usize>,
    /// Declared constraints (persisted to meta CF under key `constraint_defs`).
    ///
    /// Each [`ConstraintDef`](constraints::ConstraintDef) records a `(kind, label,
    /// property)` triple.  Enforcement is performed inside every `insert_node` and
    /// `set_node_property` call via [`ops::check_constraints`].  The list is loaded
    /// from RocksDB on `open()` and updated atomically (save to meta CF) on every
    /// `add_constraint` / `remove_constraint` call.
    pub(crate) constraint_defs: Vec<constraints::ConstraintDef>,
    /// Marker for an active explicit transaction.
    /// `None` = auto-commit mode; `Some(_)` = inside BEGIN…COMMIT.
    /// Writes always go to RocksDB immediately; this field only tracks the
    /// transaction boundary so that `commit` / `rollback` can be validated.
    txn_pending_ops: Option<Vec<Operation>>,
    /// Pre-BEGIN snapshot, used to restore state on ROLLBACK.
    txn_snapshot: Option<Box<TransactionSnapshot>>,
    /// Transient mapping from user-supplied CSV `:ID` strings to their assigned
    /// [`NodeId`]s.  Populated by `LOAD CSV NODES`; consumed by `LOAD CSV EDGES`.
    /// Not persisted to disk.  Cleared by [`Graph::clear_csv_id_map`].
    pub csv_id_map: HashMap<String, NodeId>,

    // ── Property cache (PERF-6) ───────────────────────────────────────────────
    //
    // Concurrent LRU cache for deserialized Node objects.  Avoids repeated
    // bincode::deserialize calls for hot hub nodes accessed across many queries.
    // Invalidated at COMMIT time to preserve read-your-own-writes correctness.
    node_cache: NodeCache,

    // ── In-memory adjacency (ARCH-1) ──────────────────────────────────────────
    //
    // Adjacency lists keep (edge_id, neighbor, label_id) in RAM so that graph
    // traversal is O(degree) RAM access rather than O(degree) RocksDB reads.
    // Edge labels are interned: each unique label string is stored once in
    // `adj_id_to_label`; each AdjEntry carries a 4-byte u32 label_id.
    //
    // Write-through: every `insert_edge`/`delete_edge` in ops.rs updates these
    // maps atomically after the RocksDB write.
    //
    // Rollback: during a transaction, every adj change is appended to
    // `txn_adj_deltas`.  On ROLLBACK the delta list is reversed and inverted
    // (O(ops)), avoiding the O(all_edges) re-scan that a full rebuild would
    // require.  `suppress_adj` is set during the RocksDB re-insert phase so
    // that write-through is suppressed while adj is already correct.
    pub(crate) adj_out: HashMap<NodeId, Vec<AdjEntry>>,
    pub(crate) adj_in:  HashMap<NodeId, Vec<AdjEntry>>,
    adj_label_to_id: HashMap<String, u32>,
    adj_id_to_label: Vec<String>,
    /// Accumulated adj changes for the current transaction.
    /// `None` when no transaction is active.
    pub(crate) txn_adj_deltas: Option<Vec<AdjDelta>>,
    /// Set to `true` during the RocksDB re-insert phase of ROLLBACK so that
    /// adj write-through (and delta recording) is suppressed entirely.
    pub(crate) suppress_adj: bool,

    /// If `Some`, this Graph owns a temp directory deleted on drop.
    /// Declared LAST so it drops after `store` closes RocksDB.
    _owned_dir: Option<CleanupDir>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Create an ephemeral graph backed by a unique temp-dir RocksDB.
    /// The directory is removed when the Graph is dropped.
    /// Used by tests and any code that needs a transient graph.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join(format!("minigdb_{}_{}", std::process::id(), id));
        let store = RocksStore::open(&dir).expect("failed to open temp RocksDB");
        // Mark edge_label_idx as already built (empty DB; nothing to rebuild).
        let _ = store.put_meta(crate::storage::rocks_store::META_EDGE_LABEL_IDX_BUILT, b"1");
        Self {
            store,
            index_defs: Vec::new(),
            prop_index_counts: HashMap::new(),
            constraint_defs: Vec::new(),
            txn_pending_ops: None,
            txn_snapshot: None,
            csv_id_map: HashMap::new(),
            node_cache: moka::sync::Cache::new(100_000),
            adj_out: HashMap::new(),
            adj_in:  HashMap::new(),
            adj_label_to_id: HashMap::new(),
            adj_id_to_label: Vec::new(),
            txn_adj_deltas: None,
            suppress_adj: false,
            _owned_dir: Some(CleanupDir(dir)),
        }
    }

    /// Open (or create) a persistent graph at `path`.
    /// Called by `StorageManager::open`.
    pub fn open(path: &Path) -> Result<Self, DbError> {
        let store = RocksStore::open(path)?;
        let index_defs = ops::load_index_defs(&store)?;
        let constraint_defs = ops::load_constraints(&store)?;
        let mut graph = Self {
            store,
            index_defs,
            prop_index_counts: HashMap::new(),
            constraint_defs,
            txn_pending_ops: None,
            txn_snapshot: None,
            csv_id_map: HashMap::new(),
            node_cache: moka::sync::Cache::new(100_000),
            adj_out: HashMap::new(),
            adj_in:  HashMap::new(),
            adj_label_to_id: HashMap::new(),
            adj_id_to_label: Vec::new(),
            txn_adj_deltas: None,
            suppress_adj: false,
            _owned_dir: None,
        };
        // Migration R6: populate edge_label_idx CF for existing databases that
        // pre-date this index. The meta key is written once after the rebuild.
        if graph.store.get_meta(crate::storage::rocks_store::META_EDGE_LABEL_IDX_BUILT)?.is_none() {
            ops::rebuild_edge_label_idx(&mut graph)?;
        }
        // Build in-memory adjacency list from RocksDB adj_out CF.
        // For large graphs this takes a few seconds; all subsequent traversal
        // is ns-per-hop instead of μs-per-hop.
        graph.build_adj()?;
        // Load prop index counts into memory so SHOW INDEXES is a pure in-memory scan.
        let idx_keys: Vec<(String, String)> = graph.index_defs.iter()
            .map(|d| (d.label.clone(), d.property.clone()))
            .collect();
        for (label, property) in idx_keys {
            let count = graph.store.get_prop_count(&label, &property);
            graph.prop_index_counts.insert((label, property), count);
        }
        Ok(graph)
    }

    /// Allocate a fresh NodeId (ULID).
    pub fn alloc_node_id(&mut self) -> NodeId {
        NodeId(ulid_new())
    }

    /// Allocate a fresh EdgeId (ULID).
    pub fn alloc_edge_id(&mut self) -> EdgeId {
        EdgeId(ulid_new())
    }

    // ── Read API ─────────────────────────────────────────────────────────────

    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        // Fast path: serve from the concurrent node cache (PERF-6).
        // The cache stores Arc<Node> so cloning is a ref-count bump only.
        if let Some(cached) = self.node_cache.get(&id) {
            return Some((*cached).clone());
        }
        // Slow path: load from RocksDB and populate the cache.
        let node = self.store
            .get_node_raw(id.0)
            .ok()
            .flatten()
            .and_then(|data| bincode::deserialize::<Node>(&data).ok())?;
        self.node_cache.insert(id, Arc::new(node.clone()));
        Some(node)
    }

    /// Invalidate the cache entry for `id`.
    ///
    /// Called on COMMIT (not at write-buffer time) so that reads within an
    /// active transaction always see the latest written value without stale
    /// cached data interfering.
    pub(crate) fn node_cache_invalidate(&self, id: NodeId) {
        self.node_cache.invalidate(&id);
    }

    /// Invalidate the entire node cache — used on ROLLBACK.
    pub(crate) fn node_cache_invalidate_all(&self) {
        self.node_cache.invalidate_all();
    }

    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.store
            .get_edge_raw(id.0)
            .ok()
            .flatten()
            .and_then(|data| bincode::deserialize(&data).ok())
    }

    pub fn node_count(&self) -> usize {
        self.store.node_count().unwrap_or(0) as usize
    }

    /// Return the number of nodes carrying `label` — O(1) meta CF lookup.
    ///
    /// Used by the query planner (PERF-10) to order disconnected MATCH patterns
    /// by ascending cardinality.  Returns 0 for labels that existed before the
    /// counter was introduced; the counter is lazily initialized on the next
    /// write to that label.
    pub fn count_nodes_with_label(&self, label: &str) -> usize {
        self.store.get_label_count(label)
    }

    pub fn edge_count(&self) -> usize {
        self.store.edge_count().unwrap_or(0) as usize
    }

    pub fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        self.store
            .scan_label(label)
            .unwrap_or_default()
            .into_iter()
            .map(NodeId)
            .collect()
    }

    /// Like `nodes_by_label` but stops after `limit` IDs — avoids a full scan
    /// when LIMIT is known and no ORDER BY / WHERE is present.
    pub fn nodes_by_label_limit(&self, label: &str, limit: usize) -> Vec<NodeId> {
        self.store
            .scan_label_limit(label, limit)
            .unwrap_or_default()
            .into_iter()
            .map(NodeId)
            .collect()
    }

    /// Return the first `limit` node IDs from the store without a full scan.
    pub fn first_n_node_ids(&self, limit: usize) -> Vec<NodeId> {
        self.store
            .all_node_ids_limit(limit)
            .unwrap_or_default()
            .into_iter()
            .map(NodeId)
            .collect()
    }

    /// All edges with a given label (O(matches) via `edge_label_idx` CF).
    pub fn edges_by_label(&self, label: &str) -> Vec<EdgeId> {
        self.store
            .scan_edge_label(label)
            .unwrap_or_default()
            .into_iter()
            .map(EdgeId)
            .collect()
    }

    pub fn outgoing_edges(&self, node_id: NodeId) -> Vec<EdgeId> {
        if let Some(entries) = self.adj_out.get(&node_id) {
            entries.iter().map(|e| e.edge_id).collect()
        } else {
            self.store
                .scan_adj_out(node_id.0)
                .unwrap_or_default()
                .into_iter()
                .map(|(eid, _, _)| EdgeId(eid))
                .collect()
        }
    }

    pub fn incoming_edges(&self, node_id: NodeId) -> Vec<EdgeId> {
        if let Some(entries) = self.adj_in.get(&node_id) {
            entries.iter().map(|e| e.edge_id).collect()
        } else {
            self.store
                .scan_adj_in(node_id.0)
                .unwrap_or_default()
                .into_iter()
                .map(|(eid, _, _)| EdgeId(eid))
                .collect()
        }
    }

    // ── In-memory adjacency API (ARCH-1) ──────────────────────────────────────

    /// Return the in-memory out-adjacency list for `node_id`.
    ///
    /// Each `AdjEntry` carries `(edge_id, to_node, label_id)` — no RocksDB
    /// lookup required.  Use `adj_label_str(entry.label_id)` to recover the
    /// label string.  Returns an empty slice for nodes with no outgoing edges.
    pub fn neighbors_out_mem(&self, node_id: NodeId) -> &[AdjEntry] {
        self.adj_out.get(&node_id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Return the in-memory in-adjacency list for `node_id`.
    ///
    /// Each `AdjEntry` carries `(edge_id, from_node, label_id)`.
    pub fn neighbors_in_mem(&self, node_id: NodeId) -> &[AdjEntry] {
        self.adj_in.get(&node_id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Resolve an interned label ID to its string.
    ///
    /// Panics on out-of-range IDs (indicates a bug in write-through maintenance).
    #[inline]
    pub fn adj_label_str(&self, label_id: u32) -> &str {
        &self.adj_id_to_label[label_id as usize]
    }

    /// Intern `label` and return its stable `u32` ID.
    ///
    /// If `label` has been seen before, returns the existing ID.  Otherwise
    /// assigns the next sequential ID and stores the string.  Called during
    /// `insert_edge` write-through and during `build_adj`.
    pub(crate) fn intern_adj_label(&mut self, label: &str) -> u32 {
        if let Some(&id) = self.adj_label_to_id.get(label) {
            return id;
        }
        let id = self.adj_id_to_label.len() as u32;
        self.adj_id_to_label.push(label.to_owned());
        self.adj_label_to_id.insert(label.to_owned(), id);
        id
    }

    /// Add an entry to the out-adjacency list for `from`.  Called by
    /// `ops::insert_edge` immediately after the RocksDB write.
    pub(crate) fn adj_add_out(&mut self, from: NodeId, entry: AdjEntry) {
        if self.suppress_adj { return; }
        if let Some(deltas) = &mut self.txn_adj_deltas {
            deltas.push(AdjDelta::AddOut { from, entry: entry.clone() });
        }
        self.adj_out.entry(from).or_default().push(entry);
    }

    /// Add an entry to the in-adjacency list for `to`.
    pub(crate) fn adj_add_in(&mut self, to: NodeId, entry: AdjEntry) {
        if self.suppress_adj { return; }
        if let Some(deltas) = &mut self.txn_adj_deltas {
            deltas.push(AdjDelta::AddIn { to, entry: entry.clone() });
        }
        self.adj_in.entry(to).or_default().push(entry);
    }

    /// Remove all entries with `edge_id` from the out-adjacency list of `from`.
    /// Captures removed entries into the transaction delta when active.
    pub(crate) fn adj_remove_out(&mut self, from: NodeId, edge_id: EdgeId) {
        if self.suppress_adj { return; }
        if let Some(v) = self.adj_out.get_mut(&from) {
            if self.txn_adj_deltas.is_some() {
                // Capture entries before removing so rollback can restore them.
                let mut removed = Vec::new();
                v.retain(|e| {
                    if e.edge_id == edge_id { removed.push(e.clone()); false } else { true }
                });
                if let Some(deltas) = &mut self.txn_adj_deltas {
                    for entry in removed {
                        deltas.push(AdjDelta::RemoveOut { from, entry });
                    }
                }
            } else {
                v.retain(|e| e.edge_id != edge_id);
            }
        }
    }

    /// Remove all entries with `edge_id` from the in-adjacency list of `to`.
    /// Captures removed entries into the transaction delta when active.
    pub(crate) fn adj_remove_in(&mut self, to: NodeId, edge_id: EdgeId) {
        if self.suppress_adj { return; }
        if let Some(v) = self.adj_in.get_mut(&to) {
            if self.txn_adj_deltas.is_some() {
                let mut removed = Vec::new();
                v.retain(|e| {
                    if e.edge_id == edge_id { removed.push(e.clone()); false } else { true }
                });
                if let Some(deltas) = &mut self.txn_adj_deltas {
                    for entry in removed {
                        deltas.push(AdjDelta::RemoveIn { to, entry });
                    }
                }
            } else {
                v.retain(|e| e.edge_id != edge_id);
            }
        }
    }

    /// Populate `adj_out` and `adj_in` from the RocksDB `adj_out` CF.
    ///
    /// Called once in `Graph::open()`.  For large graphs (40 M+ edges) this
    /// takes a few seconds on first open after ARCH-1 is deployed; all
    /// subsequent traversal is ns-per-hop rather than μs-per-hop.
    ///
    /// For new (empty) graphs this is a no-op.
    pub(crate) fn build_adj(&mut self) -> Result<(), DbError> {
        let all_entries = self.store.iter_all_adj_out()?;
        if all_entries.is_empty() {
            return Ok(());
        }

        // Pre-size: use average out-degree to reduce Vec reallocation.
        let approx_node_count = self.store.node_count().unwrap_or(1).max(1) as usize;
        let avg_degree = (all_entries.len() / approx_node_count).max(1);

        eprintln!(
            "minigdb: loading adjacency index ({} edges)...",
            all_entries.len()
        );

        for (from, edge_id, to, label) in all_entries {
            let label_id = self.intern_adj_label(&label);
            let from_id = NodeId(from);
            let to_id   = NodeId(to);
            let eid     = EdgeId(edge_id);

            self.adj_out
                .entry(from_id)
                .or_insert_with(|| Vec::with_capacity(avg_degree))
                .push(AdjEntry { edge_id: eid, neighbor: to_id, label_id });
            self.adj_in
                .entry(to_id)
                .or_insert_with(|| Vec::with_capacity(avg_degree))
                .push(AdjEntry { edge_id: eid, neighbor: from_id, label_id });
        }

        eprintln!("minigdb: adjacency index ready.");
        Ok(())
    }

    pub fn all_nodes(&self) -> Vec<Node> {
        self.store
            .all_node_ids()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|id| {
                self.store
                    .get_node_raw(id)
                    .ok()
                    .flatten()
                    .and_then(|data| bincode::deserialize(&data).ok())
            })
            .collect()
    }

    pub fn all_edges(&self) -> Vec<Edge> {
        self.store
            .all_edge_ids()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|id| {
                self.store
                    .get_edge_raw(id)
                    .ok()
                    .flatten()
                    .and_then(|data| bincode::deserialize(&data).ok())
            })
            .collect()
    }

    // ── Property index API ───────────────────────────────────────────────────

    /// Declare a property index for (label, property).  Returns false if already exists.
    /// Populates the `prop_idx` CF by scanning all nodes with `label`.
    pub fn create_property_index(&mut self, label: &str, property: &str) -> bool {
        let def = PropertyIndexDef { label: label.to_string(), property: property.to_string() };
        if self.index_defs.contains(&def) {
            return false;
        }
        self.index_defs.push(def);
        let _ = ops::save_index_defs(&self.store, &self.index_defs);

        // Populate prop_idx CF for existing nodes.
        let node_ids = self.store.scan_label(label).unwrap_or_default();
        let mut batch = crate::storage::rocks_store::RocksStore::batch();
        let mut count = 0usize;
        for nid in node_ids {
            if let Some(node) = self.get_node(NodeId(nid)) {
                if let Some(val) = node.properties.get(property) {
                    if let Some(encoded) = crate::types::value_index_key(val) {
                        self.store.put_prop_entry(&mut batch, label, property, &encoded, nid);
                        count += 1;
                    }
                }
            }
        }
        self.store.set_prop_count_batch(&mut batch, label, property, count);
        let _ = self.store.write(batch);
        self.prop_index_counts.insert((label.to_string(), property.to_string()), count);
        true
    }

    /// Remove a property index for (label, property).  Returns false if not found.
    pub fn drop_property_index(&mut self, label: &str, property: &str) -> bool {
        let def = PropertyIndexDef { label: label.to_string(), property: property.to_string() };
        let before = self.index_defs.len();
        self.index_defs.retain(|d| d != &def);
        if self.index_defs.len() == before {
            return false;
        }
        let _ = ops::save_index_defs(&self.store, &self.index_defs);
        let _ = self.store.delete_prop_range(label, property);
        let _ = self.store.delete_prop_count(label, property);
        self.prop_index_counts.remove(&(label.to_string(), property.to_string()));
        true
    }

    /// List all declared indexes with their entry counts (O(1) in-memory lookup).
    pub fn list_property_indexes(&self) -> Vec<(String, String, usize)> {
        self.index_defs
            .iter()
            .map(|def| {
                let count = self.prop_index_counts
                    .get(&(def.label.clone(), def.property.clone()))
                    .copied()
                    .unwrap_or(0);
                (def.label.clone(), def.property.clone(), count)
            })
            .collect()
    }

    /// Look up NodeIds by (label, property, value) using the `prop_idx` CF.
    /// Returns None if no index exists for (label, property).
    pub fn lookup_by_property(
        &self,
        label: &str,
        property: &str,
        value: &Value,
    ) -> Option<Vec<NodeId>> {
        if !self.has_property_index(label, property) {
            return None;
        }
        let encoded = crate::types::value_index_key(value)?;
        Some(
            self.store
                .scan_prop(label, property, &encoded)
                .unwrap_or_default()
                .into_iter()
                .map(NodeId)
                .collect(),
        )
    }

    /// Lookup nodes whose `(label, property)` value falls within a range.
    ///
    /// `lo` / `hi` are optional `(value, inclusive)` bounds.  Values must be
    /// `Int` or `Float` for the ordering to be meaningful; `String` values
    /// compare lexicographically, which is also correct.
    ///
    /// Returns `None` if no index exists for `(label, property)`.
    pub fn lookup_by_property_range(
        &self,
        label: &str,
        property: &str,
        lo: Option<(&Value, bool)>,
        hi: Option<(&Value, bool)>,
    ) -> Option<Vec<NodeId>> {
        if !self.has_property_index(label, property) {
            return None;
        }
        let lo_enc = lo.and_then(|(v, incl)| crate::types::value_index_key(v).map(|k| (k, incl)));
        let hi_enc = hi.and_then(|(v, incl)| crate::types::value_index_key(v).map(|k| (k, incl)));

        Some(
            self.store
                .scan_prop_range(
                    label,
                    property,
                    lo_enc.as_ref().map(|(k, incl)| (k.as_str(), *incl)),
                    hi_enc.as_ref().map(|(k, incl)| (k.as_str(), *incl)),
                )
                .unwrap_or_default()
                .into_iter()
                .map(NodeId)
                .collect(),
        )
    }

    /// Look up nodes whose `(label, property)` string value starts with `prefix`.
    ///
    /// Uses the `prop_idx` CF prefix scan — O(log n + matches).
    /// Returns `None` if no index exists for `(label, property)`.
    pub fn lookup_by_property_prefix(
        &self,
        label: &str,
        property: &str,
        prefix: &str,
    ) -> Option<Vec<NodeId>> {
        if !self.has_property_index(label, property) {
            return None;
        }
        Some(
            self.store
                .scan_prop_prefix(label, property, prefix)
                .unwrap_or_default()
                .into_iter()
                .map(NodeId)
                .collect(),
        )
    }

    /// Returns true if a property index exists for (label, property).
    pub fn has_property_index(&self, label: &str, property: &str) -> bool {
        self.index_defs.iter().any(|d| d.label == label && d.property == property)
    }

    // ── Constraint API ───────────────────────────────────────────────────────
    //
    // Two constraint kinds are supported:
    //   * `ConstraintKind::Unique`  — no two nodes with `label` may share the
    //     same value for `property`.  Checking uses the `prop_idx` CF when an
    //     explicit property index has been created; otherwise it falls back to a
    //     linear scan of all nodes carrying the label.
    //   * `ConstraintKind::Type(vk)` — every write to `property` on a node with
    //     `label` must produce a value of the matching `ValueKind` (Integer, Float,
    //     String, or Boolean).
    //
    // Constraints are enforced at both INSERT time (`build_insert_ops` in the
    // executor) and SET time (`set_node_property` in `ops.rs`).  Enforcement
    // during SET passes `self_id = Some(node.id)` so a node is not flagged as a
    // duplicate of itself when its own unique property is unchanged.

    /// Declare a constraint.  Returns `false` if the identical constraint already exists.
    pub fn add_constraint(&mut self, def: constraints::ConstraintDef) -> Result<bool, DbError> {
        if self.constraint_defs.contains(&def) {
            return Ok(false);
        }
        self.constraint_defs.push(def);
        ops::save_constraints(&self.store, &self.constraint_defs)?;
        Ok(true)
    }

    /// Remove a constraint.  Returns `false` if no matching constraint was found.
    pub fn remove_constraint(
        &mut self,
        label: &str,
        property: &str,
        kind: &constraints::ConstraintKind,
    ) -> bool {
        let before = self.constraint_defs.len();
        self.constraint_defs
            .retain(|d| !(d.label == label && d.property == property && &d.kind == kind));
        if self.constraint_defs.len() < before {
            let _ = ops::save_constraints(&self.store, &self.constraint_defs);
            true
        } else {
            false
        }
    }

    /// Return all declared constraints.
    pub fn list_constraints(&self) -> &[constraints::ConstraintDef] {
        &self.constraint_defs
    }

    /// Validate `labels`+`properties` against all declared constraints.
    ///
    /// `self_id`: if `Some(id)`, a node with that ID is allowed to already hold
    /// the same unique value (i.e. it is exempt from the uniqueness check).
    /// Pass `None` for new inserts.
    pub fn check_node_constraints(
        &self,
        labels: &[String],
        properties: &crate::types::Properties,
        self_id: Option<u128>,
    ) -> Result<(), DbError> {
        ops::check_constraints_pub(self, labels, properties, self_id)
    }

    // ── Fast clear ────────────────────────────────────────────────────────────

    /// Delete every node, edge, and index entry in O(1) via RocksDB range tombstones.
    ///
    /// Equivalent to `MATCH (n) DETACH DELETE n` but constant-time regardless of
    /// graph size.  Index *definitions* (created with `CREATE INDEX`) are preserved;
    /// their backing data is wiped and will be repopulated as new data is inserted.
    ///
    /// Returns an error if a transaction is currently active.
    pub fn clear(&mut self) -> Result<(), DbError> {
        if self.txn_pending_ops.is_some() {
            return Err(DbError::Query(
                "cannot TRUNCATE inside a transaction".into(),
            ));
        }
        self.store.clear_all()?;
        self.csv_id_map.clear();
        // Re-stamp the edge-label-index sentinel (empty DB; nothing to rebuild).
        let _ = self.store.put_meta(
            crate::storage::rocks_store::META_EDGE_LABEL_IDX_BUILT,
            b"1",
        );
        Ok(())
    }

    // ── Transaction API ───────────────────────────────────────────────────────

    /// Begin an explicit transaction.  Returns an error if already in one.
    ///
    /// A snapshot of the current graph state is saved so that `rollback_transaction`
    /// can restore it exactly.  All writes during the transaction go directly to
    /// RocksDB, making them immediately visible to reads within the same transaction.
    pub fn begin_transaction(&mut self) -> Result<(), DbError> {
        if self.txn_pending_ops.is_some() {
            return Err(DbError::Query("transaction already active".into()));
        }
        self.txn_snapshot = Some(Box::new(TransactionSnapshot {
            nodes: self.all_nodes(),
            edges: self.all_edges(),
            index_defs: self.index_defs.clone(),
            constraint_defs: self.constraint_defs.clone(),
            prop_index_counts: self.prop_index_counts.clone(),
        }));
        self.txn_pending_ops = Some(Vec::new());
        self.txn_adj_deltas = Some(Vec::new());
        Ok(())
    }

    /// Commit the current transaction: discard the rollback snapshot.
    /// Returns an error if no transaction is active.
    pub fn commit_transaction(&mut self) -> Result<(), DbError> {
        self.txn_pending_ops
            .take()
            .ok_or_else(|| DbError::Query("no active transaction".into()))?;
        self.txn_snapshot = None;
        self.txn_adj_deltas = None;
        // Invalidate node cache at commit time so subsequent reads see the
        // committed state.  We invalidate everything rather than tracking
        // exactly which nodes were mutated during the transaction.
        self.node_cache_invalidate_all();
        Ok(())
    }

    /// Roll back the current transaction: restore the pre-BEGIN snapshot.
    /// Returns an error if no transaction is active.
    pub fn rollback_transaction(&mut self) -> Result<(), DbError> {
        if self.txn_pending_ops.take().is_none() {
            return Err(DbError::Query("no active transaction".into()));
        }
        // Step 1: reverse adj deltas to restore pre-transaction adjacency state.
        // This is O(ops_in_txn) rather than O(all_edges).
        if let Some(deltas) = self.txn_adj_deltas.take() {
            for delta in deltas.into_iter().rev() {
                match delta {
                    AdjDelta::AddOut { from, entry } => {
                        if let Some(v) = self.adj_out.get_mut(&from) {
                            v.retain(|e| e.edge_id != entry.edge_id);
                        }
                    }
                    AdjDelta::AddIn { to, entry } => {
                        if let Some(v) = self.adj_in.get_mut(&to) {
                            v.retain(|e| e.edge_id != entry.edge_id);
                        }
                    }
                    AdjDelta::RemoveOut { from, entry } => {
                        self.adj_out.entry(from).or_default().push(entry);
                    }
                    AdjDelta::RemoveIn { to, entry } => {
                        self.adj_in.entry(to).or_default().push(entry);
                    }
                }
            }
        }

        if let Some(snap) = self.txn_snapshot.take() {
            // Step 2: restore RocksDB to the pre-transaction state.
            self.store.clear_all()?;
            // Restore in-memory defs first so constraint/index checks during
            // re-insert use the pre-transaction definitions.
            self.index_defs = snap.index_defs;
            self.constraint_defs = snap.constraint_defs;
            self.prop_index_counts = snap.prop_index_counts;
            // Suppress adj write-through: adj is already restored via delta
            // reversal above; the re-insert phase must not double-apply changes.
            self.suppress_adj = true;
            for node in snap.nodes {
                ops::insert_node(self, node)?;
            }
            for edge in snap.edges {
                ops::insert_edge(self, edge)?;
            }
            self.suppress_adj = false;
            // Re-stamp the edge-label-index sentinel (clear_all removed it).
            let _ = self.store.put_meta(
                crate::storage::rocks_store::META_EDGE_LABEL_IDX_BUILT,
                b"1",
            );
            // Flush the node cache: all cached entries refer to the
            // transaction-era data that has just been discarded.
            self.node_cache_invalidate_all();
        }
        Ok(())
    }

    /// Returns true if an explicit transaction is currently active.
    pub fn is_in_transaction(&self) -> bool {
        self.txn_pending_ops.is_some()
    }

    // ── Write API (apply_* methods called by ops and WAL replay) ─────────────

    pub(crate) fn apply_insert_node(&mut self, node: Node) {
        let _ = ops::insert_node(self, node);
    }

    pub(crate) fn apply_insert_edge(&mut self, edge: Edge) {
        let _ = ops::insert_edge(self, edge);
    }

    pub(crate) fn apply_set_node_property(
        &mut self,
        node_id: NodeId,
        key: String,
        value: Value,
    ) -> Result<(), DbError> {
        ops::set_node_property(self, node_id, key, value)
    }

    pub(crate) fn apply_set_edge_property(
        &mut self,
        edge_id: EdgeId,
        key: String,
        value: Value,
    ) -> Result<(), DbError> {
        ops::set_edge_property(self, edge_id, key, value)
    }

    pub(crate) fn apply_remove_node_property(
        &mut self,
        node_id: NodeId,
        key: &str,
    ) -> Result<(), DbError> {
        ops::remove_node_property(self, node_id, key)
    }

    pub(crate) fn apply_remove_edge_property(
        &mut self,
        edge_id: EdgeId,
        key: &str,
    ) -> Result<(), DbError> {
        ops::remove_edge_property(self, edge_id, key)
    }

    pub(crate) fn apply_add_label(
        &mut self,
        node_id: NodeId,
        label: String,
    ) -> Result<(), DbError> {
        ops::add_node_label(self, node_id, label)
    }

    pub(crate) fn apply_remove_label(
        &mut self,
        node_id: NodeId,
        label: &str,
    ) -> Result<(), DbError> {
        ops::remove_node_label(self, node_id, label)
    }

    pub(crate) fn apply_delete_node(&mut self, node_id: NodeId) -> Result<(), DbError> {
        ops::delete_node(self, node_id)
    }

    pub(crate) fn apply_delete_node_detach(&mut self, node_id: NodeId) -> Result<(), DbError> {
        ops::delete_node_detach(self, node_id)
    }

    pub(crate) fn apply_delete_edge(&mut self, edge_id: EdgeId) -> Result<(), DbError> {
        ops::delete_edge(self, edge_id)
    }

    pub(crate) fn apply_create_index(&mut self, label: &str, property: &str) {
        self.create_property_index(label, property);
    }

    pub(crate) fn apply_drop_index(&mut self, label: &str, property: &str) {
        self.drop_property_index(label, property);
    }

    /// No-op: RocksDB indexes are always consistent.
    pub fn rebuild_index(&mut self) {}

}

// ── Clone ─────────────────────────────────────────────────────────────────────

/// Clone creates a new ephemeral Graph and copies all data from the source.
/// This is O(nodes + edges) — intended for transaction snapshots, not hot paths.
impl Clone for Graph {
    fn clone(&self) -> Self {
        let mut g = Graph::new();
        // Copy nodes.
        for node in self.all_nodes() {
            let _ = ops::insert_node(&mut g, node);
        }
        // Copy edges.
        for edge in self.all_edges() {
            let _ = ops::insert_edge(&mut g, edge);
        }
        // Copy index defs and counts (cloned graph always starts outside a transaction).
        g.index_defs = self.index_defs.clone();
        g.prop_index_counts = self.prop_index_counts.clone();
        g
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Node, Properties};

    fn make_node(id: NodeId, labels: &[&str], props: Properties) -> Node {
        Node::new(id, labels.iter().map(|s| s.to_string()).collect(), props)
    }

    #[test]
    fn insert_and_get_node() {
        let mut g = Graph::new();
        let id = g.alloc_node_id();
        let node = make_node(id, &["Person"], Properties::new());
        g.apply_insert_node(node.clone());
        assert_eq!(g.get_node(id), Some(node));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn label_index_populated() {
        let mut g = Graph::new();
        let id = g.alloc_node_id();
        g.apply_insert_node(make_node(id, &["Person", "Employee"], Properties::new()));
        assert!(g.nodes_by_label("Person").contains(&id));
        assert!(g.nodes_by_label("Employee").contains(&id));
        assert!(g.nodes_by_label("Robot").is_empty());
    }

    #[test]
    fn delete_node_clears_index() {
        let mut g = Graph::new();
        let id = g.alloc_node_id();
        g.apply_insert_node(make_node(id, &["Person"], Properties::new()));
        g.apply_delete_node(id).unwrap();
        assert!(g.get_node(id).is_none());
        assert!(g.nodes_by_label("Person").is_empty());
    }

    #[test]
    fn delete_node_with_edges_fails() {
        let mut g = Graph::new();
        let a = g.alloc_node_id();
        let b = g.alloc_node_id();
        g.apply_insert_node(make_node(a, &[], Properties::new()));
        g.apply_insert_node(make_node(b, &[], Properties::new()));
        let eid = g.alloc_edge_id();
        g.apply_insert_edge(crate::types::Edge::new(
            eid, "KNOWS".into(), a, b, Properties::new(), true,
        ));
        assert!(matches!(g.apply_delete_node(a), Err(DbError::NodeHasEdges(_))));
    }

    #[test]
    fn detach_delete_removes_edges() {
        let mut g = Graph::new();
        let a = g.alloc_node_id();
        let b = g.alloc_node_id();
        g.apply_insert_node(make_node(a, &[], Properties::new()));
        g.apply_insert_node(make_node(b, &[], Properties::new()));
        let eid = g.alloc_edge_id();
        g.apply_insert_edge(crate::types::Edge::new(
            eid, "KNOWS".into(), a, b, Properties::new(), true,
        ));
        g.apply_delete_node_detach(a).unwrap();
        assert!(g.get_node(a).is_none());
        assert!(g.get_edge(eid).is_none());
        assert!(g.get_node(b).is_some());
    }

    #[test]
    fn property_index_created_and_used() {
        let mut g = Graph::new();
        let id = g.alloc_node_id();
        let mut props = Properties::new();
        props.insert("name".to_string(), Value::String("Alice".to_string()));
        g.apply_insert_node(make_node(id, &["Person"], props));

        g.create_property_index("Person", "name");
        let result = g.lookup_by_property("Person", "name", &Value::String("Alice".to_string()));
        assert_eq!(result, Some(vec![id]));
    }

    #[test]
    fn property_index_maintained_on_insert() {
        let mut g = Graph::new();
        g.create_property_index("Person", "name");

        let id = g.alloc_node_id();
        let mut props = Properties::new();
        props.insert("name".to_string(), Value::String("Bob".to_string()));
        g.apply_insert_node(make_node(id, &["Person"], props));

        let result = g.lookup_by_property("Person", "name", &Value::String("Bob".to_string()));
        assert_eq!(result, Some(vec![id]));
    }

    #[test]
    fn property_index_miss_returns_empty() {
        let mut g = Graph::new();
        g.create_property_index("Person", "name");
        let result = g.lookup_by_property("Person", "name", &Value::String("Nobody".to_string()));
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn no_index_returns_none() {
        let g = Graph::new();
        assert!(g.lookup_by_property("Person", "name", &Value::String("x".to_string())).is_none());
    }

    #[test]
    fn edges_by_label_uses_index() {
        let mut g = Graph::new();
        let a = g.alloc_node_id();
        let b = g.alloc_node_id();
        let c = g.alloc_node_id();
        g.apply_insert_node(make_node(a, &[], Properties::new()));
        g.apply_insert_node(make_node(b, &[], Properties::new()));
        g.apply_insert_node(make_node(c, &[], Properties::new()));
        let e1 = g.alloc_edge_id();
        let e2 = g.alloc_edge_id();
        let e3 = g.alloc_edge_id();
        g.apply_insert_edge(crate::types::Edge::new(e1, "KNOWS".into(), a, b, Properties::new(), true));
        g.apply_insert_edge(crate::types::Edge::new(e2, "LIKES".into(), a, c, Properties::new(), true));
        g.apply_insert_edge(crate::types::Edge::new(e3, "KNOWS".into(), b, c, Properties::new(), true));

        let mut knows = g.edges_by_label("KNOWS");
        knows.sort();
        let mut expected = vec![e1, e3];
        expected.sort();
        assert_eq!(knows, expected);

        assert_eq!(g.edges_by_label("LIKES"), vec![e2]);
        assert!(g.edges_by_label("HATES").is_empty());
    }

    #[test]
    fn edges_by_label_delete_removes_from_index() {
        let mut g = Graph::new();
        let a = g.alloc_node_id();
        let b = g.alloc_node_id();
        g.apply_insert_node(make_node(a, &[], Properties::new()));
        g.apply_insert_node(make_node(b, &[], Properties::new()));
        let eid = g.alloc_edge_id();
        g.apply_insert_edge(crate::types::Edge::new(eid, "KNOWS".into(), a, b, Properties::new(), true));
        assert_eq!(g.edges_by_label("KNOWS").len(), 1);
        g.apply_delete_node_detach(a).unwrap();
        assert!(g.edges_by_label("KNOWS").is_empty());
    }

    #[test]
    fn transaction_commit_applies_ops() {
        let mut g = Graph::new();
        g.begin_transaction().unwrap();
        assert!(g.is_in_transaction());
        let id = g.alloc_node_id();
        g.apply_insert_node(make_node(id, &["Person"], Properties::new()));
        // Write is immediately visible within the transaction (read-your-own-writes).
        assert_eq!(g.node_count(), 1);
        assert!(g.get_node(id).is_some());
        g.commit_transaction().unwrap();
        assert!(!g.is_in_transaction());
        assert_eq!(g.node_count(), 1);
        assert!(g.get_node(id).is_some());
    }

    #[test]
    fn transaction_rollback_discards_ops() {
        let mut g = Graph::new();
        g.begin_transaction().unwrap();
        let id = g.alloc_node_id();
        g.apply_insert_node(make_node(id, &["Person"], Properties::new()));
        g.rollback_transaction().unwrap();
        assert!(!g.is_in_transaction());
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn double_begin_returns_error() {
        let mut g = Graph::new();
        g.begin_transaction().unwrap();
        assert!(g.begin_transaction().is_err());
        g.rollback_transaction().unwrap();
    }

    #[test]
    fn commit_without_begin_returns_error() {
        let mut g = Graph::new();
        assert!(g.commit_transaction().is_err());
    }

    #[test]
    fn rollback_without_begin_returns_error() {
        let mut g = Graph::new();
        assert!(g.rollback_transaction().is_err());
    }

    // ── Constraint API ────────────────────────────────────────────────────────

    #[test]
    fn constraint_add_and_list() {
        let mut g = Graph::new();
        let def = constraints::ConstraintDef {
            kind: constraints::ConstraintKind::Unique,
            label: "Person".into(),
            property: "email".into(),
        };
        let added = g.add_constraint(def.clone()).unwrap();
        assert!(added, "first add should return true");
        let list = g.list_constraints();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0], def);
    }

    #[test]
    fn constraint_duplicate_add_returns_false() {
        let mut g = Graph::new();
        let def = constraints::ConstraintDef {
            kind: constraints::ConstraintKind::Unique,
            label: "Person".into(),
            property: "email".into(),
        };
        g.add_constraint(def.clone()).unwrap();
        let second = g.add_constraint(def).unwrap();
        assert!(!second, "duplicate add should return false");
        assert_eq!(g.list_constraints().len(), 1, "only one constraint stored");
    }

    #[test]
    fn constraint_remove_existing_returns_true() {
        let mut g = Graph::new();
        let def = constraints::ConstraintDef {
            kind: constraints::ConstraintKind::Unique,
            label: "Person".into(),
            property: "email".into(),
        };
        g.add_constraint(def).unwrap();
        let removed = g.remove_constraint("Person", "email", &constraints::ConstraintKind::Unique);
        assert!(removed);
        assert!(g.list_constraints().is_empty());
    }

    #[test]
    fn constraint_remove_nonexistent_returns_false() {
        let mut g = Graph::new();
        let removed = g.remove_constraint("Person", "email", &constraints::ConstraintKind::Unique);
        assert!(!removed);
    }

    #[test]
    fn constraint_multiple_independent_constraints() {
        let mut g = Graph::new();
        g.add_constraint(constraints::ConstraintDef {
            kind: constraints::ConstraintKind::Unique,
            label: "Person".into(),
            property: "email".into(),
        }).unwrap();
        g.add_constraint(constraints::ConstraintDef {
            kind: constraints::ConstraintKind::Type(constraints::ValueKind::Integer),
            label: "Person".into(),
            property: "age".into(),
        }).unwrap();
        g.add_constraint(constraints::ConstraintDef {
            kind: constraints::ConstraintKind::Unique,
            label: "Company".into(),
            property: "tin".into(),
        }).unwrap();
        assert_eq!(g.list_constraints().len(), 3);
        // Removing one does not affect the others.
        g.remove_constraint("Person", "email", &constraints::ConstraintKind::Unique);
        assert_eq!(g.list_constraints().len(), 2);
    }

    #[test]
    fn check_node_constraints_passes_when_no_constraints() {
        let g = Graph::new();
        let mut props = Properties::new();
        props.insert("name".into(), Value::String("Alice".into()));
        // Should not error — no constraints declared.
        g.check_node_constraints(&["Person".to_string()], &props, None).unwrap();
    }

    #[test]
    fn check_node_constraints_type_violation() {
        let mut g = Graph::new();
        g.add_constraint(constraints::ConstraintDef {
            kind: constraints::ConstraintKind::Type(constraints::ValueKind::Integer),
            label: "Person".into(),
            property: "age".into(),
        }).unwrap();
        let mut props = Properties::new();
        props.insert("age".into(), Value::String("thirty".into()));
        let err = g.check_node_constraints(&["Person".to_string()], &props, None);
        assert!(err.is_err());
        assert!(format!("{}", err.unwrap_err()).contains("Constraint violation"));
    }

    #[test]
    fn check_node_constraints_skips_missing_property() {
        // A constraint on :Person(age) should not fire if the node has no `age` property.
        let mut g = Graph::new();
        g.add_constraint(constraints::ConstraintDef {
            kind: constraints::ConstraintKind::Type(constraints::ValueKind::Integer),
            label: "Person".into(),
            property: "age".into(),
        }).unwrap();
        let props = Properties::new(); // no `age` key
        g.check_node_constraints(&["Person".to_string()], &props, None).unwrap();
    }

    #[test]
    fn check_node_constraints_skips_wrong_label() {
        // A constraint on :Person should not fire for a :Robot node.
        let mut g = Graph::new();
        g.add_constraint(constraints::ConstraintDef {
            kind: constraints::ConstraintKind::Type(constraints::ValueKind::Integer),
            label: "Person".into(),
            property: "age".into(),
        }).unwrap();
        let mut props = Properties::new();
        props.insert("age".into(), Value::String("not-a-number".into()));
        // Label is Robot, not Person — should pass.
        g.check_node_constraints(&["Robot".to_string()], &props, None).unwrap();
    }
}
