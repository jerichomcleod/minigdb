//! [`Edge`] — a graph relationship connecting two nodes.

use serde::{Deserialize, Serialize};

use super::{EdgeId, NodeId, Properties};

/// A directed or undirected relationship between two nodes.
///
/// Edges carry a single string `label` (e.g. `"KNOWS"`, `"WORKS_AT"`), link
/// a `from_node` to a `to_node`, and hold an arbitrary key-value
/// [`Properties`] map.
///
/// When `directed = false` the edge is indexed in both adjacency directions so
/// that undirected MATCH patterns (using `--`) find it from either endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    /// Globally unique, immutable identifier (ULID as `u128`).
    pub id: EdgeId,
    /// Relationship type label (e.g. `"KNOWS"`).
    pub label: String,
    /// Source endpoint.
    pub from_node: NodeId,
    /// Target endpoint.
    pub to_node: NodeId,
    /// Arbitrary key-value property bag.
    pub properties: Properties,
    /// `true` = directed (`->`); `false` = undirected (`--`).
    pub directed: bool,
}

impl Edge {
    /// Construct an `Edge` from its constituent parts.
    pub fn new(
        id: EdgeId,
        label: String,
        from_node: NodeId,
        to_node: NodeId,
        properties: Properties,
        directed: bool,
    ) -> Self {
        Self { id, label, from_node, to_node, properties, directed }
    }
}
