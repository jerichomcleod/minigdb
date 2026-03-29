//! [`Graph`] — the in-memory property graph with RocksDB write-through.
//!
//! ## Loading model (ARCH-2)
//!
//! All nodes, label indexes, adjacency maps, and property indexes live in RAM.
//! RocksDB is a write-through persistence layer — every mutation is applied
//! in-memory first, then written to RocksDB atomically.
//!
//! **Edge data** is loaded *lazily*: `open()` builds the adjacency maps and
//! edge-label index from compact RocksDB column families (fast at startup), but
//! does not deserialise full `Edge` objects.  An edge is loaded into
//! `graph.edges` on the first mutation or explicit read that requires it.
//! `load_all_edges()` / `all_edges()` force a full load when the complete set
//! is needed (e.g. `begin_transaction()`, graph algorithms via `CALL`).
//!
//! The `edges_fully_loaded` flag tracks whether every edge has been loaded into
//! the in-memory cache; code that needs the full set must check this flag
//! (or call `load_all_edges()`) before assuming `graph.edges` is complete.

pub(crate) mod ops;
pub mod constraints;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::storage::rocks_store::RocksStore;
use crate::transaction::operation::Operation;
use crate::types::{ulid_new, DbError, Edge, EdgeId, Node, NodeId, Value};

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

/// Which graph element a property index targets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum IndexTarget {
    #[default]
    Node,
    Edge,
}

/// A declared property index: index all nodes (or edges) with `label` on the value of `property`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PropertyIndexDef {
    pub label: String,
    pub property: String,
    #[serde(default)]
    pub target: IndexTarget,
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
/// Pre-transaction snapshot used for rollback (ARCH-2).
///
/// Captured at `BEGIN` by cloning the in-memory maps.  On `ROLLBACK`, the
/// in-memory state is restored from this snapshot in O(1) pointer swaps, then
/// RocksDB is cleared and rewritten from the restored maps.
struct TransactionSnapshot {
    nodes:               HashMap<NodeId, Node>,
    edges:               HashMap<EdgeId, Edge>,
    edges_fully_loaded:  bool,
    label_index:         HashMap<String, HashSet<NodeId>>,
    edge_label_index:    HashMap<String, HashSet<EdgeId>>,
    prop_index:          HashMap<(String, String), BTreeMap<String, HashSet<NodeId>>>,
    edge_prop_index:     HashMap<(String, String), BTreeMap<String, HashSet<EdgeId>>>,
    index_defs:          Vec<PropertyIndexDef>,
    constraint_defs:     Vec<constraints::ConstraintDef>,
}

pub struct Graph {
    /// RocksDB-backed store — write-through persistence only (ARCH-2).
    /// Reads during query execution never touch RocksDB; all data lives in RAM.
    /// MUST be declared first so it drops before `_owned_dir`.
    pub(crate) store: RocksStore,

    // ── Primary in-memory stores (ARCH-2) ─────────────────────────────────────
    //
    // Source of truth for all reads after open().  RocksDB is written
    // atomically on every mutation for durability; startup loads these maps
    // from RocksDB once.

    /// All nodes, keyed by NodeId.
    pub(crate) nodes: HashMap<NodeId, Node>,
    /// All edges, keyed by EdgeId.
    pub(crate) edges: HashMap<EdgeId, Edge>,

    // ── Secondary in-memory indexes ───────────────────────────────────────────

    /// Maps label → set of NodeIds with that label.
    pub(crate) label_index: HashMap<String, HashSet<NodeId>>,
    /// Maps edge label → set of EdgeIds with that label.
    pub(crate) edge_label_index: HashMap<String, HashSet<EdgeId>>,

    // ── Property indexes ──────────────────────────────────────────────────────
    //
    // Populated only for labels/properties with a `CREATE INDEX` definition.
    // BTreeMap on encoded value enables equality, range, and prefix lookups in
    // O(log n + matches) with zero RocksDB involvement.

    /// Node property index: (label, property) → encoded_value → NodeId set.
    pub(crate) prop_index: HashMap<(String, String), BTreeMap<String, HashSet<NodeId>>>,
    /// Edge property index: (label, property) → encoded_value → EdgeId set.
    pub(crate) edge_prop_index: HashMap<(String, String), BTreeMap<String, HashSet<EdgeId>>>,

    // ── Metadata ──────────────────────────────────────────────────────────────

    /// Declared property index definitions (persisted to meta CF).
    pub(crate) index_defs: Vec<PropertyIndexDef>,
    /// Declared constraints (persisted to meta CF under key `constraint_defs`).
    pub(crate) constraint_defs: Vec<constraints::ConstraintDef>,
    /// Marker for an active explicit transaction.
    txn_pending_ops: Option<Vec<Operation>>,
    /// Pre-BEGIN snapshot, used to restore state on ROLLBACK.
    txn_snapshot: Option<Box<TransactionSnapshot>>,
    /// `true` once every edge in the database has been loaded into `self.edges`.
    ///
    /// Set to `false` by `open()` (lazy loading) and `true` by `load_all_edges()`.
    /// Guards the `all_edges()` fast path and ensures `begin_transaction()`
    /// captures a complete snapshot so `rollback_transaction()` can restore it.
    pub(crate) edges_fully_loaded: bool,

    /// Transient mapping from user-supplied CSV `:ID` strings to their assigned
    /// [`NodeId`]s.  Populated by `LOAD CSV NODES`; consumed by `LOAD CSV EDGES`.
    pub csv_id_map: HashMap<String, NodeId>,

    // ── In-memory adjacency ───────────────────────────────────────────────────
    //
    // Adjacency lists keep (edge_id, neighbor, label_id) in RAM so that graph
    // traversal is O(degree) RAM access rather than O(degree) RocksDB reads.
    // Edge labels are interned: each unique label string is stored once in
    // `adj_id_to_label`; each AdjEntry carries a 4-byte u32 label_id.
    //
    // Write-through: every `insert_edge`/`delete_edge` in ops.rs updates these
    // maps atomically after the RocksDB write.
    //
    // Rollback: adj is rebuilt from in-memory edges via `build_adj()` after
    // the in-memory snapshot is restored.  `suppress_adj` suppresses
    // write-through during the RocksDB-rewrite phase of rollback.
    pub(crate) adj_out: HashMap<NodeId, Vec<AdjEntry>>,
    pub(crate) adj_in:  HashMap<NodeId, Vec<AdjEntry>>,
    adj_label_to_id: HashMap<String, u32>,
    adj_id_to_label: Vec<String>,
    /// Set to `true` during the RocksDB re-insert phase of ROLLBACK so that
    /// adj write-through is suppressed while adj is being rebuilt.
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
            nodes: HashMap::new(),
            edges: HashMap::new(),
            label_index: HashMap::new(),
            edge_label_index: HashMap::new(),
            prop_index: HashMap::new(),
            edge_prop_index: HashMap::new(),
            index_defs: Vec::new(),
            constraint_defs: Vec::new(),
            txn_pending_ops: None,
            txn_snapshot: None,
            csv_id_map: HashMap::new(),
            adj_out: HashMap::new(),
            adj_in:  HashMap::new(),
            adj_label_to_id: HashMap::new(),
            adj_id_to_label: Vec::new(),

            edges_fully_loaded: true,   // fresh graph has nothing to lazy-load
            suppress_adj: false,
            _owned_dir: Some(CleanupDir(dir)),
        }
    }

    /// Open (or create) a persistent graph at `path`.
    ///
    /// Performs a full load of all nodes, edges, and indexes from RocksDB into
    /// RAM (ARCH-2).  After `open()` returns, all reads are pure in-memory.
    pub fn open(path: &Path) -> Result<Self, DbError> {
        let _t0 = std::time::Instant::now();
        let store = RocksStore::open(path)?;
        eprintln!("[open] RocksStore::open        {:>8.3}s", _t0.elapsed().as_secs_f64());

        let _t1 = std::time::Instant::now();
        let index_defs = ops::load_index_defs(&store)?;
        let constraint_defs = ops::load_constraints(&store)?;
        let mut graph = Self {
            store,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            label_index: HashMap::new(),
            edge_label_index: HashMap::new(),
            prop_index: HashMap::new(),
            edge_prop_index: HashMap::new(),
            index_defs,
            constraint_defs,
            txn_pending_ops: None,
            txn_snapshot: None,
            csv_id_map: HashMap::new(),
            adj_out: HashMap::new(),
            adj_in:  HashMap::new(),
            adj_label_to_id: HashMap::new(),
            adj_id_to_label: Vec::new(),

            edges_fully_loaded: false,  // edges loaded lazily; see load_all_edges()
            suppress_adj: false,
            _owned_dir: None,
        };

        eprintln!("[open] load_index/constraint_defs {:>6.3}s", _t1.elapsed().as_secs_f64());

        // ── Pre-allocate node maps using stored counts ────────────────────────
        let node_hint = graph.store.node_count().unwrap_or(0) as usize;
        let edge_hint = graph.store.edge_count().unwrap_or(0) as usize;
        eprintln!("[open] counts: {} nodes, {} edges", node_hint, edge_hint);
        graph.nodes   = HashMap::with_capacity(node_hint);
        graph.adj_out = HashMap::with_capacity(node_hint);
        graph.adj_in  = HashMap::with_capacity(node_hint);

        // ── Seed declared prop_index / edge_prop_index keys ───────────────────
        for def in &graph.index_defs {
            match def.target {
                IndexTarget::Node =>
                    { graph.prop_index.entry((def.label.clone(), def.property.clone())).or_default(); }
                IndexTarget::Edge =>
                    { graph.edge_prop_index.entry((def.label.clone(), def.property.clone())).or_default(); }
            }
        }

        // ── Pass 1+2: stream nodes → label_index + prop_index in one pass ─────
        let _tp1 = std::time::Instant::now();
        {
            let store = &graph.store;
            let nodes = &mut graph.nodes;
            let li    = &mut graph.label_index;
            let pi    = &mut graph.prop_index;
            // Avoid cloning the label string on every entry when the key already
            // exists — li.entry(label.clone()) always allocates.  For N nodes
            // with L labels each this saves N*L String allocations.
            // Skip the property inner-loop entirely when no prop indexes are
            // declared (the common case); avoids N*L*P clone+lookup iterations.
            let has_prop_index = !pi.is_empty();
            store.for_each_node_raw(|data| {
                if let Ok(node) = bincode::deserialize::<Node>(data) {
                    for label in &node.labels {
                        if let Some(set) = li.get_mut(label.as_str()) {
                            set.insert(node.id);
                        } else {
                            let mut set = HashSet::new();
                            set.insert(node.id);
                            li.insert(label.clone(), set);
                        }
                        if has_prop_index {
                            for (prop, val) in &node.properties {
                                let k = (label.clone(), prop.clone());
                                if let Some(btree) = pi.get_mut(&k) {
                                    if let Some(enc) = crate::types::value_index_key(val) {
                                        btree.entry(enc).or_default().insert(node.id);
                                    }
                                }
                            }
                        }
                    }
                    nodes.insert(node.id, node);
                }
                Ok(())
            })?;
        }

        eprintln!("[open] pass1+2 nodes           {:>8.3}s", _tp1.elapsed().as_secs_f64());

        // ── Pass 3: build adj from compact adj CFs (no full Edge deserialize) ──
        //
        // adj_out CF entries are ~60 bytes each vs ~200-500 bytes for a full
        // serialised Edge.  For large graphs this is the critical startup path.
        // The directed/undirected split is already encoded: undirected edges
        // appear in BOTH directions in each adj CF, so we can build adj_out and
        // adj_in by scanning the corresponding CF directly.
        let avg_deg = (edge_hint / node_hint.max(1)).max(1);

        let _tp3a = std::time::Instant::now();
        // Process adj_out and adj_in inline via field-level borrows.
        //
        // The previous approach collected all E entries into two
        // Vec<(u128,u128,u128,String)> (each pre-allocated with Vec::with_capacity(E)),
        // then iterated them again to intern labels.  For 40M edges that means
        // ~6 GiB of transient Vec capacity plus 80M label.to_owned() allocations —
        // all discarded immediately after processing.
        //
        // Rust allows simultaneous borrows of disjoint struct fields, so we can
        // borrow `store` immutably while holding `&mut adj_out/adj_in/adj_label_*`.
        // Label interning is inlined: only unique labels (typically ≪ edge count)
        // ever allocate a String.
        {
            let store   = &graph.store;
            let adj_out = &mut graph.adj_out;
            let alt     = &mut graph.adj_label_to_id;
            let ali     = &mut graph.adj_id_to_label;
            store.for_each_adj_out(|from, edge_id, to, label| {
                let lid = if let Some(&id) = alt.get(label) {
                    id
                } else {
                    let id = ali.len() as u32;
                    ali.push(label.to_owned());
                    alt.insert(label.to_owned(), id);
                    id
                };
                adj_out
                    .entry(NodeId(from))
                    .or_insert_with(|| Vec::with_capacity(avg_deg))
                    .push(AdjEntry { edge_id: EdgeId(edge_id), neighbor: NodeId(to), label_id: lid });
                Ok(())
            })?;
        }
        eprintln!("[open] pass3a adj_out          {:>8.3}s", _tp3a.elapsed().as_secs_f64());
        let _tp3b = std::time::Instant::now();
        {
            let store  = &graph.store;
            let adj_in = &mut graph.adj_in;
            let alt    = &mut graph.adj_label_to_id;
            let ali    = &mut graph.adj_id_to_label;
            store.for_each_adj_in(|to, edge_id, from, label| {
                let lid = if let Some(&id) = alt.get(label) {
                    id
                } else {
                    let id = ali.len() as u32;
                    ali.push(label.to_owned());
                    alt.insert(label.to_owned(), id);
                    id
                };
                adj_in
                    .entry(NodeId(to))
                    .or_insert_with(|| Vec::with_capacity(avg_deg))
                    .push(AdjEntry { edge_id: EdgeId(edge_id), neighbor: NodeId(from), label_id: lid });
                Ok(())
            })?;
        }

        eprintln!("[open] pass3b adj_in           {:>8.3}s", _tp3b.elapsed().as_secs_f64());

        // ── Pass 4: build edge_label_index from compact edge_label_idx CF ──────
        let _tp4 = std::time::Instant::now();
        {
            let store = &graph.store;
            let eli   = &mut graph.edge_label_index;
            // Avoid label.to_owned() on every iteration (E times) when the
            // label key already exists.  Only unique labels allocate a String.
            store.for_each_edge_label_idx(|label, edge_id| {
                if let Some(set) = eli.get_mut(label) {
                    set.insert(EdgeId(edge_id));
                } else {
                    let mut set = HashSet::new();
                    set.insert(EdgeId(edge_id));
                    eli.insert(label.to_owned(), set);
                }
                Ok(())
            })?;
        }

        eprintln!("[open] pass4 edge_label_index  {:>8.3}s", _tp4.elapsed().as_secs_f64());

        // graph.edges is intentionally left EMPTY on startup.  Edges are
        // fetched lazily from RocksDB by get_edge() as queries access them,
        // and are written into graph.edges by every write-through mutation.
        // This avoids the full edges-CF scan (O(E) I/O + deserialization)
        // that made open() slow for large graphs.

        // ── Edge property index rebuild (only when declared) ──────────────────
        //
        // Edge prop indexes are rare on large graphs.  When they exist, fetch
        // only the edges that belong to the indexed labels.
        let _tp5 = std::time::Instant::now();
        let edge_prop_labels: Vec<(String, String)> = graph.index_defs.iter()
            .filter(|d| d.target == IndexTarget::Edge)
            .map(|d| (d.label.clone(), d.property.clone()))
            .collect();
        if !edge_prop_labels.is_empty() {
            for label in graph.edge_label_index.keys().cloned().collect::<Vec<_>>() {
                let indexed_props: Vec<String> = edge_prop_labels.iter()
                    .filter(|(l, _)| l == &label)
                    .map(|(_, p)| p.clone())
                    .collect();
                if indexed_props.is_empty() { continue; }
                let ids: Vec<EdgeId> = graph.edge_label_index
                    .get(&label).map(|s| s.iter().cloned().collect()).unwrap_or_default();
                for eid in ids {
                    if let Ok(Some(data)) = graph.store.get_edge_raw(eid.0) {
                        if let Ok(edge) = bincode::deserialize::<Edge>(&data) {
                            for prop in &indexed_props {
                                if let Some(val) = edge.properties.get(prop) {
                                    let k = (label.clone(), prop.clone());
                                    if let Some(btree) = graph.edge_prop_index.get_mut(&k) {
                                        if let Some(enc) = crate::types::value_index_key(val) {
                                            btree.entry(enc).or_default().insert(eid);
                                        }
                                    }
                                }
                            }
                            graph.edges.insert(eid, edge);
                        }
                    }
                }
            }
        }

        eprintln!("[open] edge_prop_index rebuild  {:>7.3}s", _tp5.elapsed().as_secs_f64());

        // Migration R6: one-time population of edge_label_idx CF for databases
        // that predate that index.  For new databases the CF is already current.
        let _tp6 = std::time::Instant::now();
        if graph.store.get_meta(crate::storage::rocks_store::META_EDGE_LABEL_IDX_BUILT)?.is_none() {
            // edge_label_index is currently empty (CF was empty on pre-R6 DB).
            // Scan the full edges CF once to rebuild it.
            let mut label_ids: Vec<(String, EdgeId)> = Vec::new();
            graph.store.for_each_edge_raw(|data| {
                if let Ok(edge) = bincode::deserialize::<Edge>(data) {
                    label_ids.push((edge.label, edge.id));
                }
                Ok(())
            })?;
            let mut batch = RocksStore::batch();
            for (label, eid) in label_ids {
                graph.store.put_edge_label_entry(&mut batch, &label, eid.0);
                graph.edge_label_index.entry(label).or_default().insert(eid);
            }
            graph.store.put_meta_batch(
                &mut batch,
                crate::storage::rocks_store::META_EDGE_LABEL_IDX_BUILT,
                b"1",
            );
            graph.store.write(batch)?;
        }

        eprintln!("[open] R6 migration            {:>8.3}s", _tp6.elapsed().as_secs_f64());
        eprintln!("[open] total                   {:>8.3}s", _t0.elapsed().as_secs_f64());
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

    /// Look up a node by ID — O(1) in-memory HashMap lookup (ARCH-2).
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        self.nodes.get(&id).cloned()
    }

    /// Look up an edge by ID.
    ///
    /// Checks the in-memory cache first (populated by write-through mutations
    /// and edge property index rebuilds on open).  Falls back to a RocksDB
    /// point-read for edges that have not yet been accessed this session.
    /// RocksDB's block cache makes repeated reads of the same edge fast.
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        if let Some(e) = self.edges.get(&id) {
            return Some(e.clone());
        }
        // Lazy load from RocksDB.
        if let Ok(Some(data)) = self.store.get_edge_raw(id.0) {
            if let Ok(edge) = bincode::deserialize::<Edge>(&data) {
                return Some(edge);
            }
        }
        None
    }

    /// Number of nodes — O(1) (ARCH-2).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of nodes with `label` — O(1) in-memory set size (ARCH-2).
    pub fn count_nodes_with_label(&self, label: &str) -> usize {
        self.label_index.get(label).map_or(0, |s| s.len())
    }

    /// Number of edges — O(1) meta lookup (edges not preloaded on open).
    pub fn edge_count(&self) -> usize {
        self.store.edge_count().unwrap_or(0) as usize
    }

    /// All NodeIds with `label` — O(1) HashSet clone (ARCH-2).
    pub fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        self.label_index
            .get(label)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Up to `limit` NodeIds with `label` (ARCH-2).
    pub fn nodes_by_label_limit(&self, label: &str, limit: usize) -> Vec<NodeId> {
        self.label_index
            .get(label)
            .map(|s| s.iter().cloned().take(limit).collect())
            .unwrap_or_default()
    }

    /// First `limit` node IDs — O(limit) in-memory (ARCH-2).
    pub fn first_n_node_ids(&self, limit: usize) -> Vec<NodeId> {
        self.nodes.keys().cloned().take(limit).collect()
    }

    /// All EdgeIds with `label` — O(1) HashSet clone (ARCH-2).
    pub fn edges_by_label(&self, label: &str) -> Vec<EdgeId> {
        self.edge_label_index
            .get(label)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// All outgoing edges for `node_id` — O(degree) in-memory adj list (ARCH-2).
    pub fn outgoing_edges(&self, node_id: NodeId) -> Vec<EdgeId> {
        self.adj_out
            .get(&node_id)
            .map(|v| v.iter().map(|e| e.edge_id).collect())
            .unwrap_or_default()
    }

    /// All incoming edges for `node_id` — O(degree) in-memory adj list (ARCH-2).
    pub fn incoming_edges(&self, node_id: NodeId) -> Vec<EdgeId> {
        self.adj_in
            .get(&node_id)
            .map(|v| v.iter().map(|e| e.edge_id).collect())
            .unwrap_or_default()
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

    /// Add an entry to the out-adjacency list for `from`.
    pub(crate) fn adj_add_out(&mut self, from: NodeId, entry: AdjEntry) {
        if self.suppress_adj { return; }
        self.adj_out.entry(from).or_default().push(entry);
    }

    /// Add an entry to the in-adjacency list for `to`.
    pub(crate) fn adj_add_in(&mut self, to: NodeId, entry: AdjEntry) {
        if self.suppress_adj { return; }
        self.adj_in.entry(to).or_default().push(entry);
    }

    /// Remove all entries with `edge_id` from the out-adjacency list of `from`.
    pub(crate) fn adj_remove_out(&mut self, from: NodeId, edge_id: EdgeId) {
        if self.suppress_adj { return; }
        if let Some(v) = self.adj_out.get_mut(&from) {
            v.retain(|e| e.edge_id != edge_id);
        }
    }

    /// Remove all entries with `edge_id` from the in-adjacency list of `to`.
    pub(crate) fn adj_remove_in(&mut self, to: NodeId, edge_id: EdgeId) {
        if self.suppress_adj { return; }
        if let Some(v) = self.adj_in.get_mut(&to) {
            v.retain(|e| e.edge_id != edge_id);
        }
    }

    /// Load every edge from RocksDB into `self.edges` if not already done.
    ///
    /// Called by `begin_transaction()` so the snapshot captured at BEGIN is
    /// complete — a prerequisite for `rollback_transaction()` to restore the
    /// correct state via `write_all_to_rocksdb`.
    pub(crate) fn load_all_edges(&mut self) -> Result<(), DbError> {
        if self.edges_fully_loaded {
            return Ok(());
        }
        self.store.for_each_edge_raw(|data| {
            if let Ok(edge) = bincode::deserialize::<Edge>(data) {
                self.edges.entry(edge.id).or_insert(edge);
            }
            Ok(())
        })?;
        self.edges_fully_loaded = true;
        Ok(())
    }

    /// Build `adj_out` and `adj_in` from the in-memory `edges` map.
    ///
    /// Only called during rollback (after snapshot restore), where edges are
    /// guaranteed to be fully loaded.  Steady-state adjacency is write-through.
    pub(crate) fn build_adj(&mut self) -> Result<(), DbError> {
        let n = self.edges.len();
        let node_count = self.nodes.len().max(1);
        let avg_degree = (n / node_count).max(1);

        // Collect to avoid borrow-checker conflict between &self.edges and
        // &mut self (needed for intern_adj_label).
        let edge_data: Vec<(NodeId, NodeId, EdgeId, String, bool)> = self
            .edges
            .values()
            .map(|e| (e.from_node, e.to_node, e.id, e.label.clone(), e.directed))
            .collect();

        for (from, to, eid, label, directed) in edge_data {
            let label_id = self.intern_adj_label(&label);
            self.adj_out
                .entry(from)
                .or_insert_with(|| Vec::with_capacity(avg_degree))
                .push(AdjEntry { edge_id: eid, neighbor: to, label_id });
            self.adj_in
                .entry(to)
                .or_insert_with(|| Vec::with_capacity(avg_degree))
                .push(AdjEntry { edge_id: eid, neighbor: from, label_id });
            if !directed {
                let lid2 = label_id;
                self.adj_out
                    .entry(to)
                    .or_insert_with(|| Vec::with_capacity(avg_degree))
                    .push(AdjEntry { edge_id: eid, neighbor: from, label_id: lid2 });
                self.adj_in
                    .entry(from)
                    .or_insert_with(|| Vec::with_capacity(avg_degree))
                    .push(AdjEntry { edge_id: eid, neighbor: to, label_id: lid2 });
            }
        }
        Ok(())
    }

    /// All nodes — O(N) clone from in-memory map (ARCH-2).
    pub fn all_nodes(&self) -> Vec<Node> {
        self.nodes.values().cloned().collect()
    }

    /// All edges.
    ///
    /// When every edge is in the in-memory cache (`edges_fully_loaded`) the
    /// result is served from RAM.  Otherwise falls back to a full RocksDB scan
    /// so that `MATCH ()-[e]->()` patterns with no label filter are correct
    /// even on a freshly-opened database where the cache is partial.
    pub fn all_edges(&self) -> Vec<Edge> {
        if self.edges_fully_loaded {
            return self.edges.values().cloned().collect();
        }
        let mut out = Vec::new();
        let _ = self.store.for_each_edge_raw(|data| {
            if let Ok(e) = bincode::deserialize::<Edge>(data) {
                out.push(e);
            }
            Ok(())
        });
        out
    }

    // ── Property index API ───────────────────────────────────────────────────

    /// Declare a node property index for (label, property).  Returns false if already exists.
    /// Backfills from the in-memory `nodes` map — no RocksDB scan needed (ARCH-2).
    pub fn create_property_index(&mut self, label: &str, property: &str) -> bool {
        let def = PropertyIndexDef {
            label: label.to_string(),
            property: property.to_string(),
            target: IndexTarget::Node,
        };
        if self.index_defs.contains(&def) {
            return false;
        }
        self.index_defs.push(def);
        let _ = ops::save_index_defs(&self.store, &self.index_defs);

        // Backfill in-memory prop_index and prop_idx CF from in-memory nodes.
        let key = (label.to_string(), property.to_string());
        let btree = self.prop_index.entry(key.clone()).or_default();
        let mut batch = crate::storage::rocks_store::RocksStore::batch();
        let candidate_ids: Vec<NodeId> = self.label_index
            .get(label)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        for nid in candidate_ids {
            if let Some(node) = self.nodes.get(&nid) {
                if let Some(val) = node.properties.get(property) {
                    if let Some(encoded) = crate::types::value_index_key(val) {
                        self.store.put_prop_entry(&mut batch, label, property, &encoded, nid.0);
                        btree.entry(encoded).or_default().insert(nid);
                    }
                }
            }
        }
        let _ = self.store.write(batch);
        true
    }

    /// Remove a node property index for (label, property).  Returns false if not found.
    pub fn drop_property_index(&mut self, label: &str, property: &str) -> bool {
        let def = PropertyIndexDef {
            label: label.to_string(),
            property: property.to_string(),
            target: IndexTarget::Node,
        };
        let before = self.index_defs.len();
        self.index_defs.retain(|d| d != &def);
        if self.index_defs.len() == before {
            return false;
        }
        let _ = ops::save_index_defs(&self.store, &self.index_defs);
        let _ = self.store.delete_prop_range(label, property);
        self.prop_index.remove(&(label.to_string(), property.to_string()));
        true
    }

    /// List all declared indexes with their entry counts (ARCH-2 — from in-memory maps).
    pub fn list_property_indexes(&self) -> Vec<(String, String, usize, String)> {
        self.index_defs
            .iter()
            .map(|def| {
                let key = (def.label.clone(), def.property.clone());
                let count = match def.target {
                    IndexTarget::Node => self.prop_index.get(&key)
                        .map_or(0, |bt| bt.values().map(|s| s.len()).sum()),
                    IndexTarget::Edge => self.edge_prop_index.get(&key)
                        .map_or(0, |bt| bt.values().map(|s| s.len()).sum()),
                };
                let target_str = match def.target {
                    IndexTarget::Node => "node".to_string(),
                    IndexTarget::Edge => "edge".to_string(),
                };
                (def.label.clone(), def.property.clone(), count, target_str)
            })
            .collect()
    }

    /// Look up NodeIds by (label, property, value) — pure in-memory BTreeMap (ARCH-2).
    /// Returns None if no index exists for (label, property).
    pub fn lookup_by_property(
        &self,
        label: &str,
        property: &str,
        value: &Value,
    ) -> Option<Vec<NodeId>> {
        let encoded = crate::types::value_index_key(value)?;
        let btree = self.prop_index.get(&(label.to_string(), property.to_string()))?;
        Some(btree.get(&encoded).map_or(Vec::new(), |s| s.iter().cloned().collect()))
    }

    /// Range lookup on node property index — BTreeMap::range (ARCH-2).
    /// Returns `None` if no index exists for `(label, property)`.
    pub fn lookup_by_property_range(
        &self,
        label: &str,
        property: &str,
        lo: Option<(&Value, bool)>,
        hi: Option<(&Value, bool)>,
    ) -> Option<Vec<NodeId>> {
        let btree = self.prop_index.get(&(label.to_string(), property.to_string()))?;
        let lo_enc = lo.and_then(|(v, incl)| crate::types::value_index_key(v).map(|k| (k, incl)));
        let hi_enc = hi.and_then(|(v, incl)| crate::types::value_index_key(v).map(|k| (k, incl)));
        let lo_bound = match &lo_enc {
            None => Bound::Unbounded,
            Some((k, true))  => Bound::Included(k.clone()),
            Some((k, false)) => Bound::Excluded(k.clone()),
        };
        let hi_bound = match &hi_enc {
            None => Bound::Unbounded,
            Some((k, true))  => Bound::Included(k.clone()),
            Some((k, false)) => Bound::Excluded(k.clone()),
        };
        let mut ids = Vec::new();
        for (_k, set) in btree.range((lo_bound, hi_bound)) {
            ids.extend(set.iter().cloned());
        }
        Some(ids)
    }

    /// Prefix lookup on node property string index — BTreeMap scan (ARCH-2).
    /// Returns `None` if no index exists for `(label, property)`.
    pub fn lookup_by_property_prefix(
        &self,
        label: &str,
        property: &str,
        prefix: &str,
    ) -> Option<Vec<NodeId>> {
        let btree = self.prop_index.get(&(label.to_string(), property.to_string()))?;
        // String values in the prop index are encoded as "S:<original_value>".
        let encoded_prefix = format!("S:{}", prefix);
        let mut ids = Vec::new();
        for (key, set) in btree.range(encoded_prefix.clone()..) {
            if !key.starts_with(&encoded_prefix) { break; }
            ids.extend(set.iter().cloned());
        }
        Some(ids)
    }

    /// Returns true if a node property index exists for (label, property).
    pub fn has_property_index(&self, label: &str, property: &str) -> bool {
        self.prop_index.contains_key(&(label.to_string(), property.to_string()))
    }

    // ── Edge property index API ──────────────────────────────────────────────

    /// Declare an edge property index for (label, property).  Returns false if already exists.
    pub fn create_edge_property_index(&mut self, label: &str, property: &str) -> bool {
        let def = PropertyIndexDef {
            label: label.to_string(),
            property: property.to_string(),
            target: IndexTarget::Edge,
        };
        if self.index_defs.contains(&def) {
            return false;
        }
        self.index_defs.push(def);
        let _ = ops::save_index_defs(&self.store, &self.index_defs);

        let key = (label.to_string(), property.to_string());
        let btree = self.edge_prop_index.entry(key.clone()).or_default();
        let mut batch = crate::storage::rocks_store::RocksStore::batch();
        let candidate_ids: Vec<EdgeId> = self.edge_label_index
            .get(label)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        for eid in candidate_ids {
            if let Some(edge) = self.edges.get(&eid) {
                if let Some(val) = edge.properties.get(property) {
                    if let Some(encoded) = crate::types::value_index_key(val) {
                        self.store.put_edge_prop_entry(&mut batch, label, property, &encoded, eid.0);
                        btree.entry(encoded).or_default().insert(eid);
                    }
                }
            }
        }
        let _ = self.store.write(batch);
        true
    }

    /// Remove an edge property index.  Returns false if not found.
    pub fn drop_edge_property_index(&mut self, label: &str, property: &str) -> bool {
        let def = PropertyIndexDef {
            label: label.to_string(),
            property: property.to_string(),
            target: IndexTarget::Edge,
        };
        let before = self.index_defs.len();
        self.index_defs.retain(|d| d != &def);
        if self.index_defs.len() == before {
            return false;
        }
        let _ = ops::save_index_defs(&self.store, &self.index_defs);
        let _ = self.store.delete_edge_prop_range(label, property);
        self.edge_prop_index.remove(&(label.to_string(), property.to_string()));
        true
    }

    /// Returns true if an edge property index exists for (label, property).
    pub fn has_edge_property_index(&self, label: &str, property: &str) -> bool {
        self.edge_prop_index.contains_key(&(label.to_string(), property.to_string()))
    }

    /// Look up EdgeIds by (label, property, value) — pure in-memory BTreeMap.
    pub fn lookup_by_edge_property(
        &self,
        label: &str,
        property: &str,
        value: &Value,
    ) -> Option<Vec<EdgeId>> {
        let encoded = crate::types::value_index_key(value)?;
        let btree = self.edge_prop_index.get(&(label.to_string(), property.to_string()))?;
        Some(btree.get(&encoded).map_or(Vec::new(), |s| s.iter().cloned().collect()))
    }

    /// Range lookup on edge property index.
    pub fn lookup_by_edge_property_range(
        &self,
        label: &str,
        property: &str,
        lo: Option<(&Value, bool)>,
        hi: Option<(&Value, bool)>,
    ) -> Option<Vec<EdgeId>> {
        let btree = self.edge_prop_index.get(&(label.to_string(), property.to_string()))?;
        let lo_enc = lo.and_then(|(v, incl)| crate::types::value_index_key(v).map(|k| (k, incl)));
        let hi_enc = hi.and_then(|(v, incl)| crate::types::value_index_key(v).map(|k| (k, incl)));
        let lo_bound = match &lo_enc {
            None => Bound::Unbounded,
            Some((k, true))  => Bound::Included(k.clone()),
            Some((k, false)) => Bound::Excluded(k.clone()),
        };
        let hi_bound = match &hi_enc {
            None => Bound::Unbounded,
            Some((k, true))  => Bound::Included(k.clone()),
            Some((k, false)) => Bound::Excluded(k.clone()),
        };
        let mut ids = Vec::new();
        for (_k, set) in btree.range((lo_bound, hi_bound)) {
            ids.extend(set.iter().cloned());
        }
        Some(ids)
    }

    /// Prefix lookup on edge property string index.
    pub fn lookup_by_edge_property_prefix(
        &self,
        label: &str,
        property: &str,
        prefix: &str,
    ) -> Option<Vec<EdgeId>> {
        let btree = self.edge_prop_index.get(&(label.to_string(), property.to_string()))?;
        let encoded_prefix = format!("S:{}", prefix);
        let mut ids = Vec::new();
        for (key, set) in btree.range(encoded_prefix.clone()..) {
            if !key.starts_with(&encoded_prefix) { break; }
            ids.extend(set.iter().cloned());
        }
        Some(ids)
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
        // Clear all in-memory structures.
        self.nodes.clear();
        self.edges.clear();
        self.label_index.clear();
        self.edge_label_index.clear();
        // Clear prop_index data but preserve declared index keys (structures remain).
        for bt in self.prop_index.values_mut() { bt.clear(); }
        for bt in self.edge_prop_index.values_mut() { bt.clear(); }
        self.adj_out.clear();
        self.adj_in.clear();
        self.csv_id_map.clear();
        // Re-stamp the edge-label-index sentinel.
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
        // Ensure the snapshot captures the complete edge set.  Without this,
        // rollback_transaction would restore an empty edges map, and the
        // subsequent clear_all + write_all_to_rocksdb would erase all edges.
        self.load_all_edges()?;
        self.txn_snapshot = Some(Box::new(TransactionSnapshot {
            nodes:               self.nodes.clone(),
            edges:               self.edges.clone(),
            edges_fully_loaded:  self.edges_fully_loaded,
            label_index:         self.label_index.clone(),
            edge_label_index:    self.edge_label_index.clone(),
            prop_index:          self.prop_index.clone(),
            edge_prop_index:     self.edge_prop_index.clone(),
            index_defs:          self.index_defs.clone(),
            constraint_defs:     self.constraint_defs.clone(),
        }));
        self.txn_pending_ops = Some(Vec::new());
        Ok(())
    }

    /// Commit the current transaction: discard the rollback snapshot.
    /// Returns an error if no transaction is active.
    pub fn commit_transaction(&mut self) -> Result<(), DbError> {
        self.txn_pending_ops
            .take()
            .ok_or_else(|| DbError::Query("no active transaction".into()))?;
        self.txn_snapshot = None;
        Ok(())
    }

    /// Roll back the current transaction: restore the pre-BEGIN snapshot.
    /// Returns an error if no transaction is active.
    pub fn rollback_transaction(&mut self) -> Result<(), DbError> {
        if self.txn_pending_ops.take().is_none() {
            return Err(DbError::Query("no active transaction".into()));
        }

        if let Some(snap) = self.txn_snapshot.take() {
            // Step 1: restore all in-memory structures from snapshot (O(1) swaps).
            self.nodes               = snap.nodes;
            self.edges               = snap.edges;
            self.edges_fully_loaded  = snap.edges_fully_loaded;
            self.label_index         = snap.label_index;
            self.edge_label_index    = snap.edge_label_index;
            self.prop_index          = snap.prop_index;
            self.edge_prop_index     = snap.edge_prop_index;
            self.index_defs          = snap.index_defs;
            self.constraint_defs     = snap.constraint_defs;

            // Step 2: restore adjacency from in-memory edges.
            self.adj_out.clear();
            self.adj_in.clear();
            self.adj_label_to_id.clear();
            self.adj_id_to_label.clear();
            self.build_adj()?;

            // Step 3: restore RocksDB to match in-memory state.
            // Bypass insert_node/insert_edge to avoid re-updating in-memory structures.
            self.store.clear_all()?;
            ops::write_all_to_rocksdb(self)?;
            // Re-stamp the edge-label-index sentinel.
            let _ = self.store.put_meta(
                crate::storage::rocks_store::META_EDGE_LABEL_IDX_BUILT,
                b"1",
            );
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
        // During WAL replay the index_defs record may already include a target.
        // If the def is for an edge label (and not a node label), create the edge index.
        let is_node_label = self.label_index.contains_key(label);
        let is_edge_label = self.edge_label_index.contains_key(label);
        if is_edge_label && !is_node_label {
            self.create_edge_property_index(label, property);
        } else {
            self.create_property_index(label, property);
        }
    }

    pub(crate) fn apply_drop_index(&mut self, label: &str, property: &str) {
        if !self.drop_property_index(label, property) {
            self.drop_edge_property_index(label, property);
        }
    }

    /// No-op: RocksDB indexes are always consistent.
    pub fn rebuild_index(&mut self) {}

}

// ── Clone ─────────────────────────────────────────────────────────────────────

/// Clone creates a new ephemeral Graph and copies all data from the source.
/// Used by the Python API; not used for transaction snapshots (those use
/// `TransactionSnapshot` field clones directly).
impl Clone for Graph {
    fn clone(&self) -> Self {
        let mut g = Graph::new();
        // Copy all in-memory maps.
        g.nodes            = self.nodes.clone();
        g.label_index      = self.label_index.clone();
        g.edge_label_index = self.edge_label_index.clone();
        g.prop_index       = self.prop_index.clone();
        g.edge_prop_index  = self.edge_prop_index.clone();
        g.index_defs       = self.index_defs.clone();
        g.constraint_defs  = self.constraint_defs.clone();

        // Edges may be partially cached; populate g.edges from source.
        // If self already has the full set, clone it directly; otherwise scan.
        if self.edges_fully_loaded {
            g.edges = self.edges.clone();
        } else {
            g.edges = self.edges.clone(); // start with cached
            let _ = self.store.for_each_edge_raw(|data| {
                if let Ok(edge) = bincode::deserialize::<Edge>(data) {
                    g.edges.entry(edge.id).or_insert(edge);
                }
                Ok(())
            });
        }
        g.edges_fully_loaded = true;

        // Write everything to the clone's RocksDB (ephemeral temp dir).
        let _ = ops::write_all_to_rocksdb(&mut g);
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

    // ── ARCH-2 in-memory structure tests ──────────────────────────────────────

    #[test]
    fn arch2_in_memory_maps_populated_on_insert() {
        let mut g = Graph::new();
        let id = g.alloc_node_id();
        let mut props = Properties::new();
        props.insert("name".into(), Value::String("Alice".into()));
        g.apply_insert_node(make_node(id, &["Person"], props));

        // Verify in-memory maps are directly populated.
        assert!(g.nodes.contains_key(&id));
        assert!(g.label_index.get("Person").unwrap().contains(&id));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn arch2_edge_label_index_maintained() {
        let mut g = Graph::new();
        let a = g.alloc_node_id();
        let b = g.alloc_node_id();
        g.apply_insert_node(make_node(a, &[], Properties::new()));
        g.apply_insert_node(make_node(b, &[], Properties::new()));
        let eid = g.alloc_edge_id();
        g.apply_insert_edge(crate::types::Edge::new(eid, "KNOWS".into(), a, b, Properties::new(), true));

        assert!(g.edges.contains_key(&eid));
        assert!(g.edge_label_index.get("KNOWS").unwrap().contains(&eid));

        // Delete edge — should remove from both maps.
        g.apply_delete_edge(eid).unwrap();
        assert!(!g.edges.contains_key(&eid));
        assert!(g.edge_label_index.get("KNOWS").map_or(true, |s| s.is_empty()));
    }

    #[test]
    fn arch2_prop_index_btree_lookup() {
        let mut g = Graph::new();
        g.create_property_index("Person", "age");

        let id1 = g.alloc_node_id();
        let id2 = g.alloc_node_id();
        let mut p1 = Properties::new();
        p1.insert("age".into(), Value::Int(30));
        let mut p2 = Properties::new();
        p2.insert("age".into(), Value::Int(40));
        g.apply_insert_node(make_node(id1, &["Person"], p1));
        g.apply_insert_node(make_node(id2, &["Person"], p2));

        // Equality lookup.
        let hits = g.lookup_by_property("Person", "age", &Value::Int(30)).unwrap();
        assert_eq!(hits, vec![id1]);

        // Range lookup: age > 29 AND age < 41 should return both.
        let range_hits = g.lookup_by_property_range(
            "Person", "age",
            Some((&Value::Int(29), false)),
            Some((&Value::Int(41), false)),
        ).unwrap();
        assert_eq!(range_hits.len(), 2);
    }

    #[test]
    fn arch2_rollback_restores_in_memory_state() {
        let mut g = Graph::new();
        let id = g.alloc_node_id();
        g.apply_insert_node(make_node(id, &["Person"], Properties::new()));

        g.begin_transaction().unwrap();
        let id2 = g.alloc_node_id();
        g.apply_insert_node(make_node(id2, &["Person"], Properties::new()));
        assert_eq!(g.node_count(), 2);

        g.rollback_transaction().unwrap();
        // Both in-memory maps and RocksDB should be back to 1 node.
        assert_eq!(g.node_count(), 1);
        assert!(g.nodes.contains_key(&id));
        assert!(!g.nodes.contains_key(&id2));
        assert!(g.label_index.get("Person").unwrap().contains(&id));
        assert!(!g.label_index.get("Person").unwrap().contains(&id2));
    }

    // ── Edge property index tests ─────────────────────────────────────────────

    #[test]
    fn edge_prop_index_created_and_queried() {
        let mut g = Graph::new();
        let a = g.alloc_node_id();
        let b = g.alloc_node_id();
        g.apply_insert_node(make_node(a, &[], Properties::new()));
        g.apply_insert_node(make_node(b, &[], Properties::new()));

        let eid = g.alloc_edge_id();
        let mut ep = Properties::new();
        ep.insert("since".into(), Value::Int(2020));
        g.apply_insert_edge(crate::types::Edge::new(eid, "KNOWS".into(), a, b, ep, true));

        g.create_edge_property_index("KNOWS", "since");
        let hits = g.lookup_by_edge_property("KNOWS", "since", &Value::Int(2020)).unwrap();
        assert_eq!(hits, vec![eid]);

        // No hit for a different value.
        let no_hits = g.lookup_by_edge_property("KNOWS", "since", &Value::Int(2021)).unwrap();
        assert!(no_hits.is_empty());
    }

    #[test]
    fn edge_prop_index_maintained_on_set_and_delete() {
        let mut g = Graph::new();
        g.create_edge_property_index("KNOWS", "weight");

        let a = g.alloc_node_id();
        let b = g.alloc_node_id();
        g.apply_insert_node(make_node(a, &[], Properties::new()));
        g.apply_insert_node(make_node(b, &[], Properties::new()));

        let eid = g.alloc_edge_id();
        let mut ep = Properties::new();
        ep.insert("weight".into(), Value::Int(5));
        g.apply_insert_edge(crate::types::Edge::new(eid, "KNOWS".into(), a, b, ep, true));

        assert!(g.lookup_by_edge_property("KNOWS", "weight", &Value::Int(5)).unwrap().contains(&eid));

        // Update property.
        g.apply_set_edge_property(eid, "weight".into(), Value::Int(10)).unwrap();
        assert!(g.lookup_by_edge_property("KNOWS", "weight", &Value::Int(5)).unwrap().is_empty());
        assert!(g.lookup_by_edge_property("KNOWS", "weight", &Value::Int(10)).unwrap().contains(&eid));

        // Delete edge.
        g.apply_delete_edge(eid).unwrap();
        assert!(g.lookup_by_edge_property("KNOWS", "weight", &Value::Int(10)).unwrap().is_empty());
    }

    #[test]
    fn edge_prop_index_range_query() {
        let mut g = Graph::new();
        g.create_edge_property_index("TRANSFER", "amount");

        let a = g.alloc_node_id();
        let b = g.alloc_node_id();
        g.apply_insert_node(make_node(a, &[], Properties::new()));
        g.apply_insert_node(make_node(b, &[], Properties::new()));

        for amount in [100i64, 200, 300, 400, 500] {
            let eid = g.alloc_edge_id();
            let mut ep = Properties::new();
            ep.insert("amount".into(), Value::Int(amount));
            g.apply_insert_edge(crate::types::Edge::new(eid, "TRANSFER".into(), a, b, ep, true));
        }

        // Range: 200 <= amount <= 400 → 3 hits
        let hits = g.lookup_by_edge_property_range(
            "TRANSFER", "amount",
            Some((&Value::Int(200), true)),
            Some((&Value::Int(400), true)),
        ).unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn edge_prop_index_drop_clears_data() {
        let mut g = Graph::new();
        let a = g.alloc_node_id();
        let b = g.alloc_node_id();
        g.apply_insert_node(make_node(a, &[], Properties::new()));
        g.apply_insert_node(make_node(b, &[], Properties::new()));

        let eid = g.alloc_edge_id();
        let mut ep = Properties::new();
        ep.insert("since".into(), Value::Int(2020));
        g.apply_insert_edge(crate::types::Edge::new(eid, "KNOWS".into(), a, b, ep, true));
        g.create_edge_property_index("KNOWS", "since");

        g.drop_edge_property_index("KNOWS", "since");
        assert!(!g.has_edge_property_index("KNOWS", "since"));
        assert!(g.lookup_by_edge_property("KNOWS", "since", &Value::Int(2020)).is_none());
    }

    // ── Lazy edge cache invariants (CRIT-1, CRIT-2) ───────────────────────────

    /// ROLLBACK must restore edges that existed before BEGIN even when the graph
    /// was opened from disk (CRIT-1).  The pre-ARCH-2 bug caused `rollback` to
    /// call `clear_all` with an empty edge snapshot, deleting all edges from
    /// RocksDB.
    #[test]
    fn rollback_preserves_persisted_edges() {
        let dir = tempfile::tempdir().unwrap().into_path();
        // Phase 1: open, insert an edge, close.
        {
            let mut g = Graph::open(&dir).unwrap();
            let a = g.alloc_node_id();
            let b = g.alloc_node_id();
            g.apply_insert_node(make_node(a, &["P"], Properties::new()));
            g.apply_insert_node(make_node(b, &["P"], Properties::new()));
            let eid = g.alloc_edge_id();
            g.apply_insert_edge(crate::types::Edge::new(eid, "REL".into(), a, b, Properties::new(), true));
            g.store.flush().unwrap();
        }
        // Phase 2: reopen, begin a transaction, add another edge, then ROLLBACK.
        {
            let mut g = Graph::open(&dir).unwrap();
            let c = g.alloc_node_id();
            let d = g.alloc_node_id();
            g.begin_transaction().unwrap();
            g.apply_insert_node(make_node(c, &["P"], Properties::new()));
            g.apply_insert_node(make_node(d, &["P"], Properties::new()));
            let eid2 = g.alloc_edge_id();
            g.apply_insert_edge(crate::types::Edge::new(eid2, "REL".into(), c, d, Properties::new(), true));
            g.rollback_transaction().unwrap();
            // The pre-existing edge from phase 1 must still be in RocksDB.
            assert_eq!(g.store.edge_count().unwrap(), 1,
                "pre-existing edge must survive rollback");
        }
    }

    /// `all_edges()` must return all edges from RocksDB, not just those cached
    /// in `graph.edges` (CRIT-2).  After a lazy open the cache is empty, so a
    /// naive `!is_empty()` fast-path would return nothing.
    #[test]
    fn all_edges_returns_complete_set_after_open() {
        let dir = tempfile::tempdir().unwrap().into_path();
        let mut edge_ids = Vec::new();
        // Phase 1: write several edges.
        {
            let mut g = Graph::open(&dir).unwrap();
            let a = g.alloc_node_id();
            let b = g.alloc_node_id();
            let c = g.alloc_node_id();
            g.apply_insert_node(make_node(a, &[], Properties::new()));
            g.apply_insert_node(make_node(b, &[], Properties::new()));
            g.apply_insert_node(make_node(c, &[], Properties::new()));
            for (from, to) in [(a, b), (b, c), (a, c)] {
                let eid = g.alloc_edge_id();
                g.apply_insert_edge(crate::types::Edge::new(eid, "E".into(), from, to, Properties::new(), true));
                edge_ids.push(eid);
            }
            g.store.flush().unwrap();
        }
        // Phase 2: reopen and call all_edges() — must return all 3.
        {
            let mut g = Graph::open(&dir).unwrap();
            let edges = g.all_edges();
            assert_eq!(edges.len(), 3, "all_edges() must return all persisted edges");
            let returned_ids: std::collections::HashSet<_> = edges.iter().map(|e| e.id).collect();
            for eid in &edge_ids {
                assert!(returned_ids.contains(eid), "edge {eid:?} must be in all_edges() result");
            }
        }
    }
}
