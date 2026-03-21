//! [`Operation`] — the unit of work for WAL replay and buffered transactions.
//!
//! Every mutation the executor performs is expressed as an `Operation` variant.
//! In auto-commit mode the executor calls `Graph::apply_*` directly; in
//! explicit-transaction mode it stages `Operation` values in a [`Transaction`]
//! buffer and applies them all at commit time.

use serde::{Deserialize, Serialize};

use crate::types::{Edge, EdgeId, Node, NodeId, Value};

/// Identifies whether a property set/remove operation targets a node or an edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyTarget {
    /// A node identified by [`NodeId`].
    Node(NodeId),
    /// An edge identified by [`EdgeId`].
    Edge(EdgeId),
}

/// A single, serializable graph mutation.
///
/// Operations are stored in WAL frames and replayed on recovery, and also
/// buffered in `Vec<Operation>` for explicit-transaction staging.  All
/// variants must be serializable via `bincode`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    /// Insert a new node (idempotent — ignored if the ID already exists).
    CreateNode { node: Node },
    /// Insert a new edge (idempotent — ignored if the ID already exists).
    CreateEdge { edge: Edge },
    /// Set or overwrite a property on a node or edge.
    SetProperty { target: PropertyTarget, key: String, value: Value },
    /// Remove a property key from a node or edge.
    RemoveProperty { target: PropertyTarget, key: String },
    /// Add a label to a node (no-op if the label is already present).
    AddLabel { node_id: NodeId, label: String },
    /// Remove a label from a node.
    RemoveLabel { node_id: NodeId, label: String },
    /// Delete a node (fails if it still has incident edges).
    DeleteNode { node_id: NodeId },
    /// Delete a node and all its incident edges (`DETACH DELETE` semantics).
    DeleteNodeDetach { node_id: NodeId },
    /// Delete an edge by ID.
    DeleteEdge { edge_id: EdgeId },
    /// Declare a property index on `(label, property)`.
    CreateIndex { label: String, property: String },
    /// Remove a property index for `(label, property)`.
    DropIndex { label: String, property: String },
}
