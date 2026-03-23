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

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::storage::rocks_store::RocksStore;
use crate::transaction::operation::Operation;
use crate::types::{ulid_new, DbError, Edge, EdgeId, Node, NodeId, Value};

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
}

pub struct Graph {
    /// RocksDB-backed store — the single source of truth.
    /// MUST be declared first so it drops before `_owned_dir`.
    pub(crate) store: RocksStore,
    /// Declared property index definitions (persisted to meta CF).
    pub(crate) index_defs: Vec<PropertyIndexDef>,
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
    pub csv_id_map: std::collections::HashMap<String, NodeId>,
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
            constraint_defs: Vec::new(),
            txn_pending_ops: None,
            txn_snapshot: None,
            csv_id_map: std::collections::HashMap::new(),
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
            constraint_defs,
            txn_pending_ops: None,
            txn_snapshot: None,
            csv_id_map: std::collections::HashMap::new(),
            _owned_dir: None,
        };
        // Migration R6: populate edge_label_idx CF for existing databases that
        // pre-date this index. The meta key is written once after the rebuild.
        if graph.store.get_meta(crate::storage::rocks_store::META_EDGE_LABEL_IDX_BUILT)?.is_none() {
            ops::rebuild_edge_label_idx(&mut graph)?;
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
        self.store
            .get_node_raw(id.0)
            .ok()
            .flatten()
            .and_then(|data| bincode::deserialize(&data).ok())
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
        self.store
            .scan_adj_out(node_id.0)
            .unwrap_or_default()
            .into_iter()
            .map(|(eid, _, _)| EdgeId(eid))
            .collect()
    }

    pub fn incoming_edges(&self, node_id: NodeId) -> Vec<EdgeId> {
        self.store
            .scan_adj_in(node_id.0)
            .unwrap_or_default()
            .into_iter()
            .map(|(eid, _, _)| EdgeId(eid))
            .collect()
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
        for nid in node_ids {
            if let Ok(Some(data)) = self.store.get_node_raw(nid) {
                if let Ok(node) = bincode::deserialize::<Node>(&data) {
                    if let Some(val) = node.properties.get(property) {
                        if let Some(encoded) = crate::types::value_index_key(val) {
                            self.store.put_prop_entry(&mut batch, label, property, &encoded, nid);
                        }
                    }
                }
            }
        }
        let _ = self.store.write(batch);
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
        true
    }

    /// List all declared indexes with their entry counts (from the prop_idx CF).
    pub fn list_property_indexes(&self) -> Vec<(String, String, usize)> {
        self.index_defs
            .iter()
            .map(|def| {
                let count = self.store.count_prop_entries(&def.label, &def.property);
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
            // Wipe all data written during the transaction.
            self.store.clear_all()?;
            // Restore in-memory defs first so constraint/index checks during
            // re-insert use the pre-transaction definitions.
            self.index_defs = snap.index_defs;
            self.constraint_defs = snap.constraint_defs;
            // Re-insert all pre-transaction nodes and edges.
            for node in snap.nodes {
                ops::insert_node(self, node)?;
            }
            for edge in snap.edges {
                ops::insert_edge(self, edge)?;
            }
            // Re-stamp the edge-label-index sentinel (clear_all removed it).
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
        // Copy index defs (cloned graph always starts outside a transaction).
        g.index_defs = self.index_defs.clone();
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
