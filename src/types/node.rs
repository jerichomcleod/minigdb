//! [`Node`] — a graph vertex with a stable ULID identity, labels, and properties.

use serde::{Deserialize, Serialize};

use super::{NodeId, Properties};

/// A node (vertex) in the property graph.
///
/// Nodes are identified by a [`NodeId`] (ULID), may carry one or more string
/// labels (e.g. `"Person"`, `"Company"`), and hold an arbitrary key-value
/// [`Properties`] map.
///
/// `PartialEq` is derived for testing; production code should compare by `id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Globally unique, immutable identifier (ULID as `u128`).
    pub id: NodeId,
    /// Type labels — all must be present for label-constraint filters to match.
    pub labels: Vec<String>,
    /// Arbitrary key-value property bag.
    pub properties: Properties,
}

impl Node {
    /// Construct a `Node` from its constituent parts.
    pub fn new(id: NodeId, labels: Vec<String>, properties: Properties) -> Self {
        Self { id, labels, properties }
    }
}
