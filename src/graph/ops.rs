//! Mutation helpers backed by RocksDB.
//!
//! Every write uses a `WriteBatch` so that the nodes/edges CF and all related
//! secondary CFs (adj_out, adj_in, label_idx, prop_idx, counters) are updated atomically.

use std::collections::HashSet;

use crate::storage::rocks_store::{RocksStore, META_EDGE_COUNT, META_NODE_COUNT};
use crate::types::{value_index_key, DbError, Edge, EdgeId, Node, NodeId, Value};

use super::Graph;

const META_INDEX_DEFS: &[u8] = b"index_defs";
const META_CONSTRAINT_DEFS: &[u8] = b"constraint_defs";

// ── Index-def persistence ─────────────────────────────────────────────────────

pub(crate) fn load_index_defs(
    store: &RocksStore,
) -> Result<Vec<super::PropertyIndexDef>, DbError> {
    match store.get_meta(META_INDEX_DEFS)? {
        None => Ok(Vec::new()),
        Some(bytes) => {
            bincode::deserialize(&bytes).map_err(|e| DbError::Serialization(e.to_string()))
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

// ── prop_idx CF maintenance helpers ──────────────────────────────────────────

/// Queue a prop_idx addition in `batch` for every (label, property) of `node`
/// that has a declared index.
fn batch_prop_add(graph: &Graph, batch: &mut rocksdb::WriteBatch, node: &Node) {
    for label in &node.labels {
        for (prop, val) in &node.properties {
            if graph.has_property_index(label, prop) {
                if let Some(encoded) = value_index_key(val) {
                    graph.store.put_prop_entry(batch, label, prop, &encoded, node.id.0);
                }
            }
        }
    }
}

/// Queue removal of a single (label, property, old_value, node_id) entry.
fn batch_prop_remove_val(
    graph: &Graph,
    batch: &mut rocksdb::WriteBatch,
    label: &str,
    prop: &str,
    old_val: &Value,
    node_id: u128,
) {
    if graph.has_property_index(label, prop) {
        if let Some(encoded) = value_index_key(old_val) {
            graph.store.delete_prop_entry(batch, label, prop, &encoded, node_id);
        }
    }
}

/// Queue removal of all prop_idx entries for `node` (for delete operations).
fn batch_prop_remove_node(graph: &Graph, batch: &mut rocksdb::WriteBatch, node: &Node) {
    for label in &node.labels {
        for (prop, val) in &node.properties {
            batch_prop_remove_val(graph, batch, label, prop, val, node.id.0);
        }
    }
}

// ── Mutation helpers ─────────────────────────────────────────────────────────

/// Check all declared constraints for a node being inserted or updated.
/// `self_id`: when `Some`, skip this node ID from unique-constraint hits (for SET).
/// Linear scan: return IDs of all nodes with `label` whose `property` == `val`.
fn scan_label_for_unique(
    graph: &Graph,
    label: &str,
    property: &str,
    val: &Value,
) -> Result<Vec<u128>, DbError> {
    let ids = graph.store.scan_label(label)?;
    let matches = ids.into_iter().filter(|&id| {
        graph.get_node(NodeId(id))
            .and_then(|n| n.properties.get(property).cloned())
            .map(|v| &v == val)
            .unwrap_or(false)
    }).collect();
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
                // Try the property index first (O(1) if an index exists), otherwise
                // fall back to a linear scan over all nodes with this label.
                let encoded = value_index_key(val);
                let hits: Vec<u128> = if let Some(ref enc) = encoded {
                    let idx_hits = graph.store.scan_prop(&def.label, &def.property, enc)?;
                    if !idx_hits.is_empty() {
                        idx_hits
                    } else {
                        // Index returned nothing — either the index doesn't exist
                        // or there are genuinely no matches.  To be safe, fall
                        // back to a linear scan over all labeled nodes.
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
    if graph.store.get_node_raw(node.id.0)?.is_some() {
        return Ok(());
    }
    check_constraints(graph, &node.labels, &node.properties, None)?;
    let data = encode_node(&node)?;
    let mut batch = RocksStore::batch();

    graph.store.put_node_batch(&mut batch, node.id.0, &data);
    for label in &node.labels {
        graph.store.put_label_entry(&mut batch, label, node.id.0);
    }
    // Maintain prop_idx CF for declared indexes.
    batch_prop_add(graph, &mut batch, &node);
    // Increment node counter atomically in the same batch.
    let count = graph.store.node_count()?;
    graph.store.put_meta_batch(&mut batch, META_NODE_COUNT, &(count + 1).to_le_bytes());

    graph.store.write(batch)?;
    Ok(())
}

/// Insert an edge.  Replay-safe: no-op if the ID already exists.
pub(crate) fn insert_edge(graph: &mut Graph, edge: Edge) -> Result<(), DbError> {
    if graph.store.get_edge_raw(edge.id.0)?.is_some() {
        return Ok(());
    }
    let data = encode_edge(&edge)?;
    let mut batch = RocksStore::batch();

    graph.store.put_edge_batch(&mut batch, edge.id.0, &data);
    graph.store.put_edge_label_entry(&mut batch, &edge.label, edge.id.0);
    graph.store.put_adj_out(&mut batch, edge.from_node.0, edge.id.0, edge.to_node.0, &edge.label);
    graph.store.put_adj_in(&mut batch, edge.to_node.0, edge.id.0, edge.from_node.0, &edge.label);
    // Undirected: also index the reverse direction.
    if !edge.directed {
        graph.store.put_adj_out(
            &mut batch, edge.to_node.0, edge.id.0, edge.from_node.0, &edge.label,
        );
        graph.store.put_adj_in(
            &mut batch, edge.from_node.0, edge.id.0, edge.to_node.0, &edge.label,
        );
    }
    let count = graph.store.edge_count()?;
    graph.store.put_meta_batch(&mut batch, META_EDGE_COUNT, &(count + 1).to_le_bytes());

    graph.store.write(batch)?;
    Ok(())
}

/// Set a property on a node (read-modify-write, atomic via WriteBatch).
pub(crate) fn set_node_property(
    graph: &mut Graph,
    node_id: NodeId,
    key: String,
    value: Value,
) -> Result<(), DbError> {
    let data = graph
        .store
        .get_node_raw(node_id.0)?
        .ok_or(DbError::NodeNotFound(node_id))?;
    let mut node = decode_node(&data)?;

    let old_val = node.properties.get(&key).cloned();
    node.properties.insert(key.clone(), value.clone());
    // Check constraints with the new value, excluding the node itself from unique hits.
    check_constraints(graph, &node.labels, &node.properties, Some(node_id.0))?;
    node.properties.insert(key.clone(), value);
    let new_data = encode_node(&node)?;

    let mut batch = RocksStore::batch();
    graph.store.put_node_batch(&mut batch, node_id.0, &new_data);

    // Update prop_idx: remove old entry, add new entry.
    for label in &node.labels {
        if let Some(ref old) = old_val {
            batch_prop_remove_val(graph, &mut batch, label, &key, old, node_id.0);
        }
        if graph.has_property_index(label, &key) {
            if let Some(encoded) = value_index_key(node.properties.get(&key).unwrap()) {
                graph.store.put_prop_entry(&mut batch, label, &key, &encoded, node_id.0);
            }
        }
    }

    graph.store.write(batch)?;
    Ok(())
}

/// Set a property on an edge (read-modify-write).
pub(crate) fn set_edge_property(
    graph: &mut Graph,
    edge_id: EdgeId,
    key: String,
    value: Value,
) -> Result<(), DbError> {
    let data = graph
        .store
        .get_edge_raw(edge_id.0)?
        .ok_or(DbError::EdgeNotFound(edge_id))?;
    let mut edge = decode_edge(&data)?;
    edge.properties.insert(key, value);
    let new_data = encode_edge(&edge)?;
    graph.store.put_edge_raw(edge_id.0, &new_data)?;
    Ok(())
}

/// Remove a property from a node.
pub(crate) fn remove_node_property(
    graph: &mut Graph,
    node_id: NodeId,
    key: &str,
) -> Result<(), DbError> {
    let data = graph
        .store
        .get_node_raw(node_id.0)?
        .ok_or(DbError::NodeNotFound(node_id))?;
    let mut node = decode_node(&data)?;

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
    Ok(())
}

/// Remove a property from an edge.
pub(crate) fn remove_edge_property(
    graph: &mut Graph,
    edge_id: EdgeId,
    key: &str,
) -> Result<(), DbError> {
    let data = graph
        .store
        .get_edge_raw(edge_id.0)?
        .ok_or(DbError::EdgeNotFound(edge_id))?;
    let mut edge = decode_edge(&data)?;
    edge.properties.remove(key);
    let new_data = encode_edge(&edge)?;
    graph.store.put_edge_raw(edge_id.0, &new_data)?;
    Ok(())
}

/// Add a label to a node.
pub(crate) fn add_node_label(
    graph: &mut Graph,
    node_id: NodeId,
    label: String,
) -> Result<(), DbError> {
    let data = graph
        .store
        .get_node_raw(node_id.0)?
        .ok_or(DbError::NodeNotFound(node_id))?;
    let mut node = decode_node(&data)?;
    if node.labels.contains(&label) {
        return Ok(());
    }
    node.labels.push(label.clone());
    let new_data = encode_node(&node)?;

    let mut batch = RocksStore::batch();
    graph.store.put_node_batch(&mut batch, node_id.0, &new_data);
    graph.store.put_label_entry(&mut batch, &label, node_id.0);

    // Maintain prop_idx for the new label.
    for (prop, val) in &node.properties {
        if graph.has_property_index(&label, prop) {
            if let Some(encoded) = value_index_key(val) {
                graph.store.put_prop_entry(&mut batch, &label, prop, &encoded, node_id.0);
            }
        }
    }

    graph.store.write(batch)?;
    Ok(())
}

/// Remove a label from a node.
pub(crate) fn remove_node_label(
    graph: &mut Graph,
    node_id: NodeId,
    label: &str,
) -> Result<(), DbError> {
    let data = graph
        .store
        .get_node_raw(node_id.0)?
        .ok_or(DbError::NodeNotFound(node_id))?;
    let mut node = decode_node(&data)?;

    let mut batch = RocksStore::batch();

    // Remove prop_idx entries for this label before modifying the node.
    for (prop, val) in &node.properties {
        batch_prop_remove_val(graph, &mut batch, label, prop, val, node_id.0);
    }

    node.labels.retain(|l| l != label);
    let new_data = encode_node(&node)?;
    graph.store.put_node_batch(&mut batch, node_id.0, &new_data);
    graph.store.delete_label_entry(&mut batch, label, node_id.0);
    graph.store.write(batch)?;
    Ok(())
}

/// Delete a node.  Fails if it has any incident edges (use `delete_node_detach`).
pub(crate) fn delete_node(graph: &mut Graph, node_id: NodeId) -> Result<(), DbError> {
    let out = graph.store.scan_adj_out(node_id.0)?;
    let inc = graph.store.scan_adj_in(node_id.0)?;
    if !out.is_empty() || !inc.is_empty() {
        return Err(DbError::NodeHasEdges(node_id));
    }

    let data = graph
        .store
        .get_node_raw(node_id.0)?
        .ok_or(DbError::NodeNotFound(node_id))?;
    let node = decode_node(&data)?;

    let count = graph.store.node_count()?;
    let mut batch = RocksStore::batch();
    graph.store.delete_node_batch(&mut batch, node_id.0);
    for lbl in &node.labels {
        graph.store.delete_label_entry(&mut batch, lbl, node_id.0);
    }
    batch_prop_remove_node(graph, &mut batch, &node);
    graph.store.put_meta_batch(&mut batch, META_NODE_COUNT, &count.saturating_sub(1).to_le_bytes());
    graph.store.write(batch)?;
    Ok(())
}

/// Delete a node and all its incident edges (DETACH DELETE semantics).
pub(crate) fn delete_node_detach(graph: &mut Graph, node_id: NodeId) -> Result<(), DbError> {
    // Collect unique incident edge IDs (dedup handles undirected edges appearing twice).
    let mut edge_ids: HashSet<u128> = HashSet::new();
    for (eid, _, _) in graph.store.scan_adj_out(node_id.0)? {
        edge_ids.insert(eid);
    }
    for (eid, _, _) in graph.store.scan_adj_in(node_id.0)? {
        edge_ids.insert(eid);
    }
    for eid in edge_ids {
        delete_edge(graph, EdgeId(eid))?;
    }

    // Delete the node itself (no edges remain so delete_node won't fail).
    if let Some(data) = graph.store.get_node_raw(node_id.0)? {
        let node = decode_node(&data)?;
        let count = graph.store.node_count()?;
        let mut batch = RocksStore::batch();
        graph.store.delete_node_batch(&mut batch, node_id.0);
        for lbl in &node.labels {
            graph.store.delete_label_entry(&mut batch, lbl, node_id.0);
        }
        batch_prop_remove_node(graph, &mut batch, &node);
        graph.store.put_meta_batch(
            &mut batch,
            META_NODE_COUNT,
            &count.saturating_sub(1).to_le_bytes(),
        );
        graph.store.write(batch)?;
    }
    Ok(())
}

// ── Migration helpers ────────────────────────────────────────────────────────

/// Populate the `edge_label_idx` CF from all existing edges and mark it built.
/// Called by `Graph::open()` on first open after Phase R6 is introduced.
pub(crate) fn rebuild_edge_label_idx(graph: &mut Graph) -> Result<(), DbError> {
    let edge_ids = graph.store.all_edge_ids()?;
    let mut batch = RocksStore::batch();
    for eid in edge_ids {
        if let Some(data) = graph.store.get_edge_raw(eid)? {
            if let Ok(edge) = bincode::deserialize::<crate::types::Edge>(&data) {
                graph.store.put_edge_label_entry(&mut batch, &edge.label, eid);
            }
        }
    }
    graph.store.put_meta_batch(
        &mut batch,
        crate::storage::rocks_store::META_EDGE_LABEL_IDX_BUILT,
        b"1",
    );
    graph.store.write(batch)
}

/// Delete an edge by ID.
pub(crate) fn delete_edge(graph: &mut Graph, edge_id: EdgeId) -> Result<(), DbError> {
    let data = graph
        .store
        .get_edge_raw(edge_id.0)?
        .ok_or(DbError::EdgeNotFound(edge_id))?;
    let edge = decode_edge(&data)?;

    let count = graph.store.edge_count()?;
    let mut batch = RocksStore::batch();
    graph.store.delete_edge_batch(&mut batch, edge_id.0);
    graph.store.delete_edge_label_entry(&mut batch, &edge.label, edge_id.0);
    graph.store.delete_adj_out_batch(&mut batch, edge.from_node.0, edge_id.0);
    graph.store.delete_adj_in_batch(&mut batch, edge.to_node.0, edge_id.0);
    if !edge.directed {
        graph.store.delete_adj_out_batch(&mut batch, edge.to_node.0, edge_id.0);
        graph.store.delete_adj_in_batch(&mut batch, edge.from_node.0, edge_id.0);
    }
    graph.store.put_meta_batch(&mut batch, META_EDGE_COUNT, &count.saturating_sub(1).to_le_bytes());
    graph.store.write(batch)?;
    Ok(())
}
