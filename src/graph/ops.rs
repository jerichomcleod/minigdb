//! Mutation helpers — write-through to RocksDB (ARCH-2).
//!
//! Every mutation updates the in-memory structures first, then stages the same
//! change in a `WriteBatch` for atomic RocksDB persistence.  The in-memory
//! state is authoritative for all reads; RocksDB is the durability layer only.

use std::collections::HashSet;

use crate::storage::rocks_store::{RocksStore, META_EDGE_COUNT, META_NODE_COUNT};
use crate::types::{value_index_key, DbError, Edge, EdgeId, Node, NodeId, Value};

use super::Graph;

const META_INDEX_DEFS: &[u8] = b"index_defs";
const META_CONSTRAINT_DEFS: &[u8] = b"constraint_defs";

// ── Index-def persistence ─────────────────────────────────────────────────────

/// Pre-ARCH-2 format: only `label` + `property`, no `target` field.
/// Used as a migration fallback when the new format fails to deserialize.
#[derive(serde::Deserialize)]
struct LegacyIndexDef {
    label: String,
    property: String,
}

pub(crate) fn load_index_defs(
    store: &RocksStore,
) -> Result<Vec<super::PropertyIndexDef>, DbError> {
    match store.get_meta(META_INDEX_DEFS)? {
        None => Ok(Vec::new()),
        Some(bytes) => {
            // Try new format (with `target` field) first.
            if let Ok(defs) = bincode::deserialize::<Vec<super::PropertyIndexDef>>(&bytes) {
                return Ok(defs);
            }
            // Fall back to pre-ARCH-2 format (no `target` field) and migrate.
            let legacy: Vec<LegacyIndexDef> = bincode::deserialize(&bytes)
                .map_err(|e| DbError::Serialization(e.to_string()))?;
            let migrated = legacy
                .into_iter()
                .map(|d| super::PropertyIndexDef {
                    label: d.label,
                    property: d.property,
                    target: super::IndexTarget::Node,
                })
                .collect();
            Ok(migrated)
        }
    }
}

pub(crate) fn save_index_defs(
    store: &RocksStore,
    defs: &[super::PropertyIndexDef],
) -> Result<(), DbError> {
    let bytes = bincode::serialize(defs).map_err(|e| DbError::Serialization(e.to_string()))?;
    store.put_meta(META_INDEX_DEFS, &bytes)
}

// ── Constraint persistence ────────────────────────────────────────────────────

pub(crate) fn load_constraints(
    store: &RocksStore,
) -> Result<Vec<super::constraints::ConstraintDef>, DbError> {
    match store.get_meta(META_CONSTRAINT_DEFS)? {
        None => Ok(Vec::new()),
        Some(bytes) => {
            bincode::deserialize(&bytes).map_err(|e| DbError::Serialization(e.to_string()))
        }
    }
}

pub(crate) fn save_constraints(
    store: &RocksStore,
    defs: &[super::constraints::ConstraintDef],
) -> Result<(), DbError> {
    let bytes = bincode::serialize(defs).map_err(|e| DbError::Serialization(e.to_string()))?;
    store.put_meta(META_CONSTRAINT_DEFS, &bytes)
}

// ── Serialization helpers ────────────────────────────────────────────────────

fn encode_node(node: &Node) -> Result<Vec<u8>, DbError> {
    bincode::serialize(node).map_err(|e| DbError::Serialization(e.to_string()))
}

fn decode_node(data: &[u8]) -> Result<Node, DbError> {
    bincode::deserialize(data).map_err(|e| DbError::Serialization(e.to_string()))
}

fn encode_edge(edge: &Edge) -> Result<Vec<u8>, DbError> {
    bincode::serialize(edge).map_err(|e| DbError::Serialization(e.to_string()))
}

fn decode_edge(data: &[u8]) -> Result<Edge, DbError> {
    bincode::deserialize(data).map_err(|e| DbError::Serialization(e.to_string()))
}

// ── prop_idx CF + in-memory prop_index maintenance helpers (ARCH-2) ──────────

/// Stage a prop_idx CF addition for every (label, property) of `node` that has
/// a declared index, and update the in-memory `prop_index` BTreeMap.
fn batch_prop_add(graph: &mut Graph, batch: &mut rocksdb::WriteBatch, node: &Node) {
    for label in &node.labels {
        for (prop, val) in &node.properties {
            let key = (label.clone(), prop.clone());
            if graph.prop_index.contains_key(&key) {
                if let Some(encoded) = value_index_key(val) {
                    graph.store.put_prop_entry(batch, label, prop, &encoded, node.id.0);
                    graph.prop_index.entry(key).or_default()
                        .entry(encoded).or_default().insert(node.id);
                }
            }
        }
    }
}

/// Stage removal of a single (label, property, old_value, node_id) from the
/// prop_idx CF and from the in-memory `prop_index` BTreeMap.
fn batch_prop_remove_val(
    graph: &mut Graph,
    batch: &mut rocksdb::WriteBatch,
    label: &str,
    prop: &str,
    old_val: &Value,
    node_id: u128,
) {
    let key = (label.to_string(), prop.to_string());
    if graph.prop_index.contains_key(&key) {
        if let Some(encoded) = value_index_key(old_val) {
            graph.store.delete_prop_entry(batch, label, prop, &encoded, node_id);
            if let Some(btree) = graph.prop_index.get_mut(&key) {
                if let Some(set) = btree.get_mut(&encoded) {
                    set.remove(&NodeId(node_id));
                    if set.is_empty() { btree.remove(&encoded); }
                }
            }
        }
    }
}

/// Stage removal of all prop_idx entries for `node` (for delete operations).
fn batch_prop_remove_node(graph: &mut Graph, batch: &mut rocksdb::WriteBatch, node: &Node) {
    for label in &node.labels {
        for (prop, val) in &node.properties {
            batch_prop_remove_val(graph, batch, label, prop, val, node.id.0);
        }
    }
}

// ── edge_prop_idx CF + in-memory edge_prop_index maintenance helpers ──────────

/// Stage an edge_prop_idx CF addition for every declared edge property index,
/// and update the in-memory `edge_prop_index` BTreeMap.
fn batch_edge_prop_add(graph: &mut Graph, batch: &mut rocksdb::WriteBatch, edge: &Edge) {
    for (prop, val) in &edge.properties {
        let key = (edge.label.clone(), prop.clone());
        if graph.edge_prop_index.contains_key(&key) {
            if let Some(encoded) = value_index_key(val) {
                graph.store.put_edge_prop_entry(batch, &edge.label, prop, &encoded, edge.id.0);
                graph.edge_prop_index.entry(key).or_default()
                    .entry(encoded).or_default().insert(edge.id);
            }
        }
    }
}

/// Stage removal of a single edge property entry.
fn batch_edge_prop_remove_val(
    graph: &mut Graph,
    batch: &mut rocksdb::WriteBatch,
    label: &str,
    prop: &str,
    old_val: &Value,
    edge_id: u128,
) {
    let key = (label.to_string(), prop.to_string());
    if graph.edge_prop_index.contains_key(&key) {
        if let Some(encoded) = value_index_key(old_val) {
            graph.store.delete_edge_prop_entry(batch, label, prop, &encoded, edge_id);
            if let Some(btree) = graph.edge_prop_index.get_mut(&key) {
                if let Some(set) = btree.get_mut(&encoded) {
                    set.remove(&EdgeId(edge_id));
                    if set.is_empty() { btree.remove(&encoded); }
                }
            }
        }
    }
}

/// Stage removal of all edge_prop_idx entries for `edge`.
fn batch_edge_prop_remove_edge(graph: &mut Graph, batch: &mut rocksdb::WriteBatch, edge: &Edge) {
    for (prop, val) in &edge.properties {
        batch_edge_prop_remove_val(graph, batch, &edge.label, prop, val, edge.id.0);
    }
}

// ── Mutation helpers ─────────────────────────────────────────────────────────

/// Linear scan over in-memory nodes: return IDs of all nodes with `label`
/// whose `property` == `val`.  Used as fallback for unique-constraint check.
fn scan_label_for_unique(
    graph: &Graph,
    label: &str,
    property: &str,
    val: &Value,
) -> Result<Vec<u128>, DbError> {
    let ids = graph.label_index.get(label).cloned().unwrap_or_default();
    let matches = ids.into_iter()
        .filter(|id| {
            graph.nodes.get(id)
                .and_then(|n| n.properties.get(property))
                .map(|v| v == val)
                .unwrap_or(false)
        })
        .map(|id| id.0)
        .collect();
    Ok(matches)
}

fn check_constraints(
    graph: &Graph,
    labels: &[String],
    properties: &crate::types::Properties,
    self_id: Option<u128>,
) -> Result<(), DbError> {
    use super::constraints::{ConstraintKind, ValueKind};
    for def in &graph.constraint_defs {
        if !labels.iter().any(|l| l == &def.label) {
            continue;
        }
        let Some(val) = properties.get(&def.property) else { continue };
        match &def.kind {
            ConstraintKind::Unique => {
                // Try the in-memory property index first (O(1) lookup).
                let encoded = value_index_key(val);
                let hits: Vec<u128> = if let Some(ref enc) = encoded {
                    let key = (def.label.clone(), def.property.clone());
                    if let Some(btree) = graph.prop_index.get(&key) {
                        btree.get(enc)
                            .map_or(Vec::new(), |s| s.iter().map(|id| id.0).collect())
                    } else {
                        // No property index — fall back to full label scan.
                        scan_label_for_unique(graph, &def.label, &def.property, val)?
                    }
                } else {
                    // Value has no index encoding (e.g. null/list) — linear scan.
                    scan_label_for_unique(graph, &def.label, &def.property, val)?
                };
                let conflict = hits.into_iter().any(|id| Some(id) != self_id);
                if conflict {
                    return Err(DbError::ConstraintViolation(format!(
                        ":{}({}) must be unique, but value {} already exists",
                        def.label, def.property, val
                    )));
                }
            }
            ConstraintKind::Type(vk) => {
                let ok = match vk {
                    ValueKind::Integer => matches!(val, Value::Int(_)),
                    ValueKind::Float   => matches!(val, Value::Float(_)),
                    ValueKind::String  => matches!(val, Value::String(_)),
                    ValueKind::Boolean => matches!(val, Value::Bool(_)),
                };
                if !ok {
                    return Err(DbError::ConstraintViolation(format!(
                        ":{}({}) must be {}, but got {}",
                        def.label, def.property, vk, val
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Public delegate so `Graph::check_node_constraints` can call the private helper.
pub(crate) fn check_constraints_pub(
    graph: &Graph,
    labels: &[String],
    properties: &crate::types::Properties,
    self_id: Option<u128>,
) -> Result<(), DbError> {
    check_constraints(graph, labels, properties, self_id)
}

/// Insert a node.  Replay-safe: no-op if the ID already exists.
pub(crate) fn insert_node(graph: &mut Graph, node: Node) -> Result<(), DbError> {
    if graph.nodes.contains_key(&node.id) {
        return Ok(());
    }
    check_constraints(graph, &node.labels, &node.properties, None)?;
    let data = encode_node(&node)?;
    let new_count = (graph.nodes.len() as u64) + 1;
    let mut batch = RocksStore::batch();

    graph.store.put_node_batch(&mut batch, node.id.0, &data);
    for label in &node.labels {
        graph.store.put_label_entry(&mut batch, label, node.id.0);
        graph.store.adjust_label_count_batch(&mut batch, label, 1);
    }
    // Stage prop_idx CF writes (in-memory prop_index updated below after write).
    batch_prop_add(graph, &mut batch, &node);
    graph.store.put_meta_batch(&mut batch, META_NODE_COUNT, &new_count.to_le_bytes());

    graph.store.write(batch)?;

    // Write-through: update in-memory structures.
    for label in &node.labels {
        graph.label_index.entry(label.clone()).or_default().insert(node.id);
    }
    graph.nodes.insert(node.id, node);
    Ok(())
}

/// Insert an edge.  Replay-safe: no-op if the ID already exists.
pub(crate) fn insert_edge(graph: &mut Graph, edge: Edge) -> Result<(), DbError> {
    if graph.edges.contains_key(&edge.id) {
        return Ok(());
    }
    let data = encode_edge(&edge)?;
    let new_count = (graph.edges.len() as u64) + 1;
    let mut batch = RocksStore::batch();

    graph.store.put_edge_batch(&mut batch, edge.id.0, &data);
    graph.store.put_edge_label_entry(&mut batch, &edge.label, edge.id.0);
    graph.store.put_adj_out(&mut batch, edge.from_node.0, edge.id.0, edge.to_node.0, &edge.label);
    graph.store.put_adj_in(&mut batch, edge.to_node.0, edge.id.0, edge.from_node.0, &edge.label);
    if !edge.directed {
        graph.store.put_adj_out(
            &mut batch, edge.to_node.0, edge.id.0, edge.from_node.0, &edge.label,
        );
        graph.store.put_adj_in(
            &mut batch, edge.from_node.0, edge.id.0, edge.to_node.0, &edge.label,
        );
    }
    // Stage edge prop index writes.
    batch_edge_prop_add(graph, &mut batch, &edge);
    graph.store.put_meta_batch(&mut batch, META_EDGE_COUNT, &new_count.to_le_bytes());

    graph.store.write(batch)?;

    // Write-through: update in-memory adjacency and edge maps.
    let label_id = graph.intern_adj_label(&edge.label);
    let from = edge.from_node;
    let to   = edge.to_node;
    let eid  = edge.id;
    let directed = edge.directed;
    graph.adj_add_out(from, crate::graph::AdjEntry { edge_id: eid, neighbor: to,   label_id });
    graph.adj_add_in (to,   crate::graph::AdjEntry { edge_id: eid, neighbor: from, label_id });
    if !directed {
        graph.adj_add_out(to,   crate::graph::AdjEntry { edge_id: eid, neighbor: from, label_id });
        graph.adj_add_in (from, crate::graph::AdjEntry { edge_id: eid, neighbor: to,   label_id });
    }
    graph.edge_label_index.entry(edge.label.clone()).or_default().insert(eid);
    graph.edges.insert(eid, edge);
    Ok(())
}

/// Set a property on a node (read from in-memory, write-through to RocksDB).
pub(crate) fn set_node_property(
    graph: &mut Graph,
    node_id: NodeId,
    key: String,
    value: Value,
) -> Result<(), DbError> {
    let mut node = graph.nodes.get(&node_id).cloned()
        .ok_or(DbError::NodeNotFound(node_id))?;

    let old_val = node.properties.get(&key).cloned();
    node.properties.insert(key.clone(), value.clone());
    check_constraints(graph, &node.labels, &node.properties, Some(node_id.0))?;
    node.properties.insert(key.clone(), value);
    let new_data = encode_node(&node)?;

    let mut batch = RocksStore::batch();
    graph.store.put_node_batch(&mut batch, node_id.0, &new_data);

    // Update prop_idx CF and in-memory prop_index: remove old, add new.
    for label in &node.labels {
        if let Some(ref old) = old_val {
            batch_prop_remove_val(graph, &mut batch, label, &key, old, node_id.0);
        }
        let idx_key = (label.clone(), key.clone());
        if graph.prop_index.contains_key(&idx_key) {
            if let Some(encoded) = value_index_key(node.properties.get(&key).unwrap()) {
                graph.store.put_prop_entry(&mut batch, label, &key, &encoded, node_id.0);
                graph.prop_index.entry(idx_key).or_default()
                    .entry(encoded).or_default().insert(node_id);
            }
        }
    }

    graph.store.write(batch)?;
    // Update in-memory node.
    graph.nodes.insert(node_id, node);
    Ok(())
}

/// Set a property on an edge (read from in-memory, write-through to RocksDB).
pub(crate) fn set_edge_property(
    graph: &mut Graph,
    edge_id: EdgeId,
    key: String,
    value: Value,
) -> Result<(), DbError> {
    let mut edge = graph.edges.get(&edge_id).cloned()
        .ok_or(DbError::EdgeNotFound(edge_id))?;

    let old_val = edge.properties.get(&key).cloned();
    if let Some(ref old) = old_val {
        let mut batch = RocksStore::batch();
        batch_edge_prop_remove_val(graph, &mut batch, &edge.label, &key, old, edge_id.0);
        graph.store.write(batch)?;
    }

    edge.properties.insert(key.clone(), value.clone());

    let mut batch = RocksStore::batch();
    graph.store.put_edge_batch(&mut batch, edge_id.0, &encode_edge(&edge)?);
    // Add new edge prop index entry.
    let idx_key = (edge.label.clone(), key.clone());
    if graph.edge_prop_index.contains_key(&idx_key) {
        if let Some(encoded) = value_index_key(&value) {
            graph.store.put_edge_prop_entry(&mut batch, &edge.label, &key, &encoded, edge_id.0);
            graph.edge_prop_index.entry(idx_key).or_default()
                .entry(encoded).or_default().insert(edge_id);
        }
    }
    graph.store.write(batch)?;
    graph.edges.insert(edge_id, edge);
    Ok(())
}

/// Remove a property from a node.
pub(crate) fn remove_node_property(
    graph: &mut Graph,
    node_id: NodeId,
    key: &str,
) -> Result<(), DbError> {
    let mut node = graph.nodes.get(&node_id).cloned()
        .ok_or(DbError::NodeNotFound(node_id))?;

    let old_val = node.properties.remove(key);
    let new_data = encode_node(&node)?;

    let mut batch = RocksStore::batch();
    graph.store.put_node_batch(&mut batch, node_id.0, &new_data);
    if let Some(ref old) = old_val {
        for label in &node.labels {
            batch_prop_remove_val(graph, &mut batch, label, key, old, node_id.0);
        }
    }
    graph.store.write(batch)?;
    graph.nodes.insert(node_id, node);
    Ok(())
}

/// Remove a property from an edge.
pub(crate) fn remove_edge_property(
    graph: &mut Graph,
    edge_id: EdgeId,
    key: &str,
) -> Result<(), DbError> {
    let mut edge = graph.edges.get(&edge_id).cloned()
        .ok_or(DbError::EdgeNotFound(edge_id))?;

    let old_val = edge.properties.remove(key);
    let new_data = encode_edge(&edge)?;

    let mut batch = RocksStore::batch();
    graph.store.put_edge_batch(&mut batch, edge_id.0, &new_data);
    if let Some(ref old) = old_val {
        batch_edge_prop_remove_val(graph, &mut batch, &edge.label, key, old, edge_id.0);
    }
    graph.store.write(batch)?;
    graph.edges.insert(edge_id, edge);
    Ok(())
}

/// Add a label to a node.
pub(crate) fn add_node_label(
    graph: &mut Graph,
    node_id: NodeId,
    label: String,
) -> Result<(), DbError> {
    let mut node = graph.nodes.get(&node_id).cloned()
        .ok_or(DbError::NodeNotFound(node_id))?;
    if node.labels.contains(&label) {
        return Ok(());
    }
    node.labels.push(label.clone());
    let new_data = encode_node(&node)?;

    let mut batch = RocksStore::batch();
    graph.store.put_node_batch(&mut batch, node_id.0, &new_data);
    graph.store.put_label_entry(&mut batch, &label, node_id.0);
    graph.store.adjust_label_count_batch(&mut batch, &label, 1);

    // Maintain prop_idx for the new label.
    for (prop, val) in &node.properties {
        let key = (label.clone(), prop.clone());
        if graph.prop_index.contains_key(&key) {
            if let Some(encoded) = value_index_key(val) {
                graph.store.put_prop_entry(&mut batch, &label, prop, &encoded, node_id.0);
                graph.prop_index.entry(key).or_default()
                    .entry(encoded).or_default().insert(node_id);
            }
        }
    }

    graph.store.write(batch)?;
    graph.label_index.entry(label).or_default().insert(node_id);
    graph.nodes.insert(node_id, node);
    Ok(())
}

/// Remove a label from a node.
pub(crate) fn remove_node_label(
    graph: &mut Graph,
    node_id: NodeId,
    label: &str,
) -> Result<(), DbError> {
    let mut node = graph.nodes.get(&node_id).cloned()
        .ok_or(DbError::NodeNotFound(node_id))?;

    let mut batch = RocksStore::batch();

    // Remove prop_idx entries for this label before modifying the node.
    for (prop, val) in &node.properties {
        batch_prop_remove_val(graph, &mut batch, label, prop, val, node_id.0);
    }

    node.labels.retain(|l| l != label);
    let new_data = encode_node(&node)?;
    graph.store.put_node_batch(&mut batch, node_id.0, &new_data);
    graph.store.delete_label_entry(&mut batch, label, node_id.0);
    graph.store.adjust_label_count_batch(&mut batch, label, -1);
    graph.store.write(batch)?;
    if let Some(set) = graph.label_index.get_mut(label) {
        set.remove(&node_id);
    }
    graph.nodes.insert(node_id, node);
    Ok(())
}

/// Delete a node.  Fails if it has any incident edges (use `delete_node_detach`).
pub(crate) fn delete_node(graph: &mut Graph, node_id: NodeId) -> Result<(), DbError> {
    // Use in-memory adjacency to check for incident edges.
    let has_edges = graph.adj_out.get(&node_id).map_or(false, |v| !v.is_empty())
        || graph.adj_in.get(&node_id).map_or(false, |v| !v.is_empty());
    if has_edges {
        return Err(DbError::NodeHasEdges(node_id));
    }

    let node = graph.nodes.get(&node_id).cloned()
        .ok_or(DbError::NodeNotFound(node_id))?;

    let new_count = (graph.nodes.len() as u64).saturating_sub(1);
    let mut batch = RocksStore::batch();
    graph.store.delete_node_batch(&mut batch, node_id.0);
    for lbl in &node.labels {
        graph.store.delete_label_entry(&mut batch, lbl, node_id.0);
        graph.store.adjust_label_count_batch(&mut batch, lbl, -1);
    }
    batch_prop_remove_node(graph, &mut batch, &node);
    graph.store.put_meta_batch(&mut batch, META_NODE_COUNT, &new_count.to_le_bytes());
    graph.store.write(batch)?;
    // Update in-memory structures.
    for lbl in &node.labels {
        if let Some(set) = graph.label_index.get_mut(lbl) { set.remove(&node_id); }
    }
    graph.nodes.remove(&node_id);
    Ok(())
}

/// Delete a node and all its incident edges (DETACH DELETE semantics).
pub(crate) fn delete_node_detach(graph: &mut Graph, node_id: NodeId) -> Result<(), DbError> {
    // Collect unique incident edge IDs from in-memory adjacency.
    let mut edge_ids: HashSet<EdgeId> = HashSet::new();
    if let Some(entries) = graph.adj_out.get(&node_id) {
        for e in entries { edge_ids.insert(e.edge_id); }
    }
    if let Some(entries) = graph.adj_in.get(&node_id) {
        for e in entries { edge_ids.insert(e.edge_id); }
    }
    let edge_ids: Vec<EdgeId> = edge_ids.into_iter().collect();
    for eid in edge_ids {
        delete_edge(graph, eid)?;
    }

    // Delete the node itself.
    if let Some(node) = graph.nodes.get(&node_id).cloned() {
        let new_count = (graph.nodes.len() as u64).saturating_sub(1);
        let mut batch = RocksStore::batch();
        graph.store.delete_node_batch(&mut batch, node_id.0);
        for lbl in &node.labels {
            graph.store.delete_label_entry(&mut batch, lbl, node_id.0);
            graph.store.adjust_label_count_batch(&mut batch, lbl, -1);
        }
        batch_prop_remove_node(graph, &mut batch, &node);
        graph.store.put_meta_batch(&mut batch, META_NODE_COUNT, &new_count.to_le_bytes());
        graph.store.write(batch)?;
        for lbl in &node.labels {
            if let Some(set) = graph.label_index.get_mut(lbl) { set.remove(&node_id); }
        }
        graph.nodes.remove(&node_id);
    }
    Ok(())
}

// ── Migration and bulk-write helpers ─────────────────────────────────────────

/// Populate the `edge_label_idx` CF from in-memory edges (ARCH-2 migration).
/// Called by `Graph::open()` on first open after Phase R6 is introduced.
pub(crate) fn rebuild_edge_label_idx_from_memory(graph: &mut Graph) -> Result<(), DbError> {
    let mut batch = RocksStore::batch();
    for (eid, edge) in &graph.edges {
        graph.store.put_edge_label_entry(&mut batch, &edge.label, eid.0);
    }
    graph.store.put_meta_batch(
        &mut batch,
        crate::storage::rocks_store::META_EDGE_LABEL_IDX_BUILT,
        b"1",
    );
    graph.store.write(batch)
}

/// Write all in-memory nodes and edges to RocksDB in bulk.
///
/// Used by `rollback_transaction` (after clear_all) and `Clone for Graph` to
/// synchronise the store with the current in-memory state without going through
/// `insert_node`/`insert_edge` (which would double-update in-memory structures).
pub(crate) fn write_all_to_rocksdb(graph: &mut Graph) -> Result<(), DbError> {
    // Write nodes.
    {
        let nodes: Vec<_> = graph.nodes.values().cloned().collect();
        let mut batch = RocksStore::batch();
        for node in &nodes {
            let data = encode_node(node)?;
            graph.store.put_node_batch(&mut batch, node.id.0, &data);
            for label in &node.labels {
                graph.store.put_label_entry(&mut batch, label, node.id.0);
                graph.store.adjust_label_count_batch(&mut batch, label, 1);
            }
            // Rebuild prop_idx CF entries.
            for label in &node.labels {
                for (prop, val) in &node.properties {
                    let key = (label.clone(), prop.clone());
                    if graph.prop_index.contains_key(&key) {
                        if let Some(encoded) = value_index_key(val) {
                            graph.store.put_prop_entry(&mut batch, label, prop, &encoded, node.id.0);
                        }
                    }
                }
            }
        }
        graph.store.put_meta_batch(
            &mut batch, META_NODE_COUNT, &(nodes.len() as u64).to_le_bytes(),
        );
        graph.store.write(batch)?;
    }

    // Write edges.
    {
        let edges: Vec<_> = graph.edges.values().cloned().collect();
        let mut batch = RocksStore::batch();
        for edge in &edges {
            let data = encode_edge(edge)?;
            graph.store.put_edge_batch(&mut batch, edge.id.0, &data);
            graph.store.put_edge_label_entry(&mut batch, &edge.label, edge.id.0);
            graph.store.put_adj_out(&mut batch, edge.from_node.0, edge.id.0, edge.to_node.0, &edge.label);
            graph.store.put_adj_in(&mut batch, edge.to_node.0, edge.id.0, edge.from_node.0, &edge.label);
            if !edge.directed {
                graph.store.put_adj_out(&mut batch, edge.to_node.0, edge.id.0, edge.from_node.0, &edge.label);
                graph.store.put_adj_in(&mut batch, edge.from_node.0, edge.id.0, edge.to_node.0, &edge.label);
            }
            // Rebuild edge_prop_idx CF entries.
            for (prop, val) in &edge.properties {
                let key = (edge.label.clone(), prop.clone());
                if graph.edge_prop_index.contains_key(&key) {
                    if let Some(encoded) = value_index_key(val) {
                        graph.store.put_edge_prop_entry(&mut batch, &edge.label, prop, &encoded, edge.id.0);
                    }
                }
            }
        }
        graph.store.put_meta_batch(
            &mut batch, META_EDGE_COUNT, &(edges.len() as u64).to_le_bytes(),
        );
        graph.store.put_meta_batch(
            &mut batch,
            crate::storage::rocks_store::META_EDGE_LABEL_IDX_BUILT,
            b"1",
        );
        graph.store.write(batch)?;
    }

    // Persist index and constraint defs.
    save_index_defs(&graph.store, &graph.index_defs.clone())?;
    save_constraints(&graph.store, &graph.constraint_defs.clone())?;

    Ok(())
}

/// Delete an edge by ID.
pub(crate) fn delete_edge(graph: &mut Graph, edge_id: EdgeId) -> Result<(), DbError> {
    let edge = graph.edges.get(&edge_id).cloned()
        .ok_or(DbError::EdgeNotFound(edge_id))?;

    let new_count = (graph.edges.len() as u64).saturating_sub(1);
    let mut batch = RocksStore::batch();
    graph.store.delete_edge_batch(&mut batch, edge_id.0);
    graph.store.delete_edge_label_entry(&mut batch, &edge.label, edge_id.0);
    graph.store.delete_adj_out_batch(&mut batch, edge.from_node.0, edge_id.0);
    graph.store.delete_adj_in_batch(&mut batch, edge.to_node.0, edge_id.0);
    if !edge.directed {
        graph.store.delete_adj_out_batch(&mut batch, edge.to_node.0, edge_id.0);
        graph.store.delete_adj_in_batch(&mut batch, edge.from_node.0, edge_id.0);
    }
    batch_edge_prop_remove_edge(graph, &mut batch, &edge);
    graph.store.put_meta_batch(&mut batch, META_EDGE_COUNT, &new_count.to_le_bytes());
    graph.store.write(batch)?;

    // Write-through: remove from in-memory adjacency and edge maps.
    graph.adj_remove_out(edge.from_node, edge_id);
    graph.adj_remove_in (edge.to_node,   edge_id);
    if !edge.directed {
        graph.adj_remove_out(edge.to_node,   edge_id);
        graph.adj_remove_in (edge.from_node, edge_id);
    }
    if let Some(set) = graph.edge_label_index.get_mut(&edge.label) {
        set.remove(&edge_id);
    }
    graph.edges.remove(&edge_id);

    Ok(())
}
