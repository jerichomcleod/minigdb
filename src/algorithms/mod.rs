//! Graph algorithm library for minigdb.
//!
//! Algorithms are invoked via the GQL `CALL` statement:
//!
//! ```gql
//! CALL shortestPath(source: "01ABCD...", target: "01FGHI...", weight: "cost")
//! YIELD source, target, cost, path
//! ```
//!
//! Each algorithm accepts a `HashMap<String, Value>` of named parameters,
//! validates them (returning descriptive errors for invalid inputs), and
//! returns `Vec<Row>` — the same row format used throughout the executor.
//!
//! ## Available Algorithms
//!
//! | Name                    | Description                                         |
//! |-------------------------|-----------------------------------------------------|
//! | `bfs` / `dfs`           | Breadth/depth-first traversal from a start node    |
//! | `shortestPath`          | Dijkstra single-source or single-pair paths         |
//! | `wcc`                   | Weakly connected components (Union-Find)            |
//! | `scc`                   | Strongly connected components (Kosaraju two-pass)   |
//! | `pageRank`              | Random-walk rank via power iteration                |
//! | `betweennessCentrality` | Brandes shortest-path betweenness                   |
//! | `closenessCentrality`   | Inverse average distance, with Wasserman-Faust fix  |
//! | `degreeCentrality`      | In/out/total degree counts and normalized scores    |
//! | `triangleCount`         | Local clustering coefficients or global transitivity|
//! | `jaccardSimilarity`     | Neighbourhood overlap similarity (Jaccard/Overlap)  |
//! | `labelPropagation`      | Semi-synchronous label propagation community detection|
//! | `louvain`               | Multi-level modularity-optimizing community detection|
//! | `leiden`                | Louvain with well-connectedness refinement phase    |
//! | `maxFlow`               | Edmonds-Karp max-flow (BFS Ford-Fulkerson)          |
//!
//! ## Architecture
//!
//! Each algorithm sub-module exposes a `run*(graph, params)` function that:
//! 1. Validates and extracts typed parameters using helpers from this module.
//! 2. Calls [`GraphSnapshot::build`] to materialise the adjacency structure in memory.
//! 3. Runs the algorithm against the snapshot.
//! 4. Returns `Vec<Row>` (a `Vec<HashMap<String, Value>>`).
//!
//! The [`GraphSnapshot`] is built once per invocation and reused across
//! multiple passes within the same algorithm.

use std::collections::HashMap;

use crate::graph::Graph;
use crate::types::{ulid_decode, DbError, NodeId, Value};

pub mod bfs;
pub mod centrality;
pub mod community;
pub mod components;
pub mod flow;
pub mod louvain;
pub mod shortest_path;
pub mod similarity;
pub mod triangle;

/// A row returned by an algorithm.  Identical to `crate::query::executor::Row`.
pub type Row = HashMap<String, Value>;

// ── Traversal direction ──────────────────────────────────────────────────────

/// Edge traversal direction used by algorithms that need to control
/// whether they follow outgoing edges, incoming edges, or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Follow only outgoing (forward) edges: `(u) --> (v)`.
    Out,
    /// Follow only incoming (reverse) edges: `(u) <-- (v)`.
    In,
    /// Follow edges in either direction (treat the graph as undirected).
    Any,
}

impl Direction {
    /// Parse a direction string from a GQL parameter value.
    ///
    /// Accepted values (case-insensitive):
    /// - `"out"` or `"outgoing"` → [`Direction::Out`]
    /// - `"in"` or `"incoming"` → [`Direction::In`]
    /// - `"any"` or `"both"` → [`Direction::Any`]
    fn from_str(s: &str) -> Result<Self, DbError> {
        match s.to_ascii_lowercase().as_str() {
            "out" | "outgoing" => Ok(Direction::Out),
            "in" | "incoming" => Ok(Direction::In),
            "any" | "both" => Ok(Direction::Any),
            other => Err(DbError::Query(format!(
                "invalid direction '{other}': expected \"out\", \"in\", or \"any\""
            ))),
        }
    }
}

// ── In-memory adjacency snapshot ─────────────────────────────────────────────

/// An immutable in-memory copy of the graph's adjacency structure, built once
/// per algorithm invocation.
///
/// Materialising the graph into contiguous `Vec` arrays provides cache-friendly
/// random access and avoids repeated RocksDB lookups during multi-pass algorithms.
///
/// ## Memory
///
/// Space is O(N + E) where N = node count and E = edge count.  For large graphs
/// with millions of edges the snapshot may use significant memory; it is dropped
/// at the end of each algorithm call.
///
/// ## Weight Semantics
///
/// If `weight_prop` is `Some("prop")`, edge weights are read from the named
/// integer or float property; missing or non-numeric values default to `1.0`.
/// If `weight_prop` is `None`, all edges have weight `1.0` (unweighted).
/// Negative weights are stored as-is; individual algorithms document whether
/// they require non-negative weights (e.g. Dijkstra).
pub struct GraphSnapshot {
    /// Node IDs in scan order (index `i` → `NodeId`).
    pub node_ids: Vec<NodeId>,
    /// Reverse index: `NodeId` → contiguous index for O(1) lookup.
    pub id_to_idx: HashMap<NodeId, usize>,
    /// `adj_out[i]` = list of `(neighbor_index, edge_weight)` for outgoing edges.
    pub adj_out: Vec<Vec<(usize, f64)>>,
    /// `adj_in[i]` = list of `(neighbor_index, edge_weight)` for incoming edges.
    pub adj_in: Vec<Vec<(usize, f64)>>,
    /// Total node count (`node_ids.len()`).
    pub n: usize,
}

impl GraphSnapshot {
    /// Build a snapshot from a live [`Graph`].
    ///
    /// All nodes are scanned in insertion order and assigned contiguous indices
    /// `0..n`.  All edges are then processed to populate `adj_out` and `adj_in`.
    ///
    /// Undirected edges (`edge.directed == false`) are reflected in **both**
    /// adjacency lists so that traversal in any direction works correctly.
    ///
    /// # Parameters
    ///
    /// - `graph` — the graph to snapshot.
    /// - `weight_prop` — optional edge property name for weights;
    ///   `None` or missing property values default to `1.0`.
    pub fn build(graph: &Graph, weight_prop: Option<&str>) -> Self {
        let nodes = graph.all_nodes();
        let n = nodes.len();
        let node_ids: Vec<NodeId> = nodes.iter().map(|nd| nd.id).collect();
        let id_to_idx: HashMap<NodeId, usize> =
            node_ids.iter().copied().enumerate().map(|(i, id)| (id, i)).collect();

        let mut adj_out = vec![Vec::new(); n];
        let mut adj_in = vec![Vec::new(); n];

        for edge in graph.all_edges() {
            // Skip edges whose endpoints are not in the node index (should not
            // happen in a consistent graph, but be defensive).
            let (Some(&fi), Some(&ti)) = (
                id_to_idx.get(&edge.from_node),
                id_to_idx.get(&edge.to_node),
            ) else {
                continue;
            };

            // Read edge weight from the named property, falling back to 1.0.
            let w = weight_prop
                .and_then(|k| edge.properties.get(k))
                .and_then(|v| match v {
                    Value::Float(f) => Some(*f),
                    Value::Int(i) => Some(*i as f64),
                    _ => None,
                })
                .unwrap_or(1.0);

            adj_out[fi].push((ti, w));
            adj_in[ti].push((fi, w));

            // Undirected edges are symmetric in both adjacency lists.
            if !edge.directed {
                adj_out[ti].push((fi, w));
                adj_in[fi].push((ti, w));
            }
        }

        Self { node_ids, id_to_idx, adj_out, adj_in, n }
    }

    /// Return all neighbors of node `i` in the given direction, as
    /// `(neighbor_index, edge_weight)` pairs.
    ///
    /// For [`Direction::Any`], parallel undirected edges may appear twice —
    /// once from `adj_out` and once from `adj_in`.  Algorithms that need
    /// unique neighbors should deduplicate via [`Self::unique_neighbor_indices`].
    pub fn neighbors(&self, i: usize, dir: Direction) -> Vec<(usize, f64)> {
        match dir {
            Direction::Out => self.adj_out[i].clone(),
            Direction::In => self.adj_in[i].clone(),
            Direction::Any => {
                // Concatenate both adjacency lists; caller deduplicates if needed.
                let mut nbrs = self.adj_out[i].clone();
                nbrs.extend_from_slice(&self.adj_in[i]);
                nbrs
            }
        }
    }

    /// Return unique neighbor *indices* of node `i` in the given direction.
    ///
    /// Parallel edges between the same pair of nodes are collapsed into a
    /// single entry.  Useful for algorithms that treat the graph as a simple
    /// graph (e.g. triangle counting, component labelling).
    pub fn unique_neighbor_indices(&self, i: usize, dir: Direction) -> Vec<usize> {
        let mut seen = std::collections::HashSet::new();
        self.neighbors(i, dir)
            .into_iter()
            .filter_map(|(j, _)| seen.insert(j).then_some(j))
            .collect()
    }
}

// ── Parameter helpers ─────────────────────────────────────────────────────────

/// Require a string-valued GQL `call_arg` parameter.
///
/// Returns an error if the key is absent or its value is not a [`Value::String`].
pub fn require_str<'a>(
    params: &'a HashMap<String, Value>,
    key: &str,
) -> Result<&'a str, DbError> {
    match params.get(key) {
        Some(Value::String(s)) => Ok(s.as_str()),
        Some(other) => Err(DbError::Query(format!(
            "parameter '{key}' must be a string, got {other:?}"
        ))),
        None => Err(DbError::Query(format!("required parameter '{key}' is missing"))),
    }
}

/// Require a node index by decoding the ULID string at `params[key]`.
///
/// Validates that the string is a well-formed ULID and that the corresponding
/// node exists in the snapshot.  Returns the contiguous index into
/// [`GraphSnapshot::node_ids`].
pub fn require_node_idx(
    params: &HashMap<String, Value>,
    key: &str,
    snap: &GraphSnapshot,
) -> Result<usize, DbError> {
    let s = require_str(params, key)?;
    let raw = ulid_decode(s)
        .map_err(|e| DbError::Query(format!("parameter '{key}' is not a valid ULID: {e}")))?;
    snap.id_to_idx.get(&NodeId(raw)).copied().ok_or_else(|| {
        DbError::Query(format!("parameter '{key}': node '{s}' not found in graph"))
    })
}

/// Optionally decode a ULID string at `params[key]` into a node index.
///
/// Returns `Ok(None)` if the key is absent.  Returns an error if the key is
/// present but the value is not a valid ULID or the node does not exist.
pub fn opt_node_idx(
    params: &HashMap<String, Value>,
    key: &str,
    snap: &GraphSnapshot,
) -> Result<Option<usize>, DbError> {
    match params.get(key) {
        None => Ok(None),
        Some(Value::String(s)) => {
            let raw = ulid_decode(s).map_err(|e| {
                DbError::Query(format!("parameter '{key}' is not a valid ULID: {e}"))
            })?;
            let idx = snap.id_to_idx.get(&NodeId(raw)).copied().ok_or_else(|| {
                DbError::Query(format!("parameter '{key}': node '{s}' not found in graph"))
            })?;
            Ok(Some(idx))
        }
        Some(other) => Err(DbError::Query(format!(
            "parameter '{key}' must be a string (ULID), got {other:?}"
        ))),
    }
}

/// Extract an optional `f64` parameter with a fallback default.
///
/// Accepts both [`Value::Float`] and [`Value::Int`] values.  NaN and ±∞ are
/// rejected because they cause undefined behaviour in most numerical algorithms.
pub fn opt_f64(
    params: &HashMap<String, Value>,
    key: &str,
    default: f64,
) -> Result<f64, DbError> {
    match params.get(key) {
        None => Ok(default),
        Some(Value::Float(f)) => {
            if f.is_nan() || f.is_infinite() {
                Err(DbError::Query(format!("parameter '{key}' must be a finite number")))
            } else {
                Ok(*f)
            }
        }
        Some(Value::Int(i)) => Ok(*i as f64),
        Some(other) => Err(DbError::Query(format!(
            "parameter '{key}' must be a number, got {other:?}"
        ))),
    }
}

/// Extract an optional non-negative `usize` parameter with a fallback default.
///
/// Negative integers produce an error; non-integer types produce an error.
pub fn opt_usize(
    params: &HashMap<String, Value>,
    key: &str,
    default: usize,
) -> Result<usize, DbError> {
    match params.get(key) {
        None => Ok(default),
        Some(Value::Int(i)) if *i >= 0 => Ok(*i as usize),
        Some(Value::Int(i)) => Err(DbError::Query(format!(
            "parameter '{key}' must be a non-negative integer, got {i}"
        ))),
        Some(other) => Err(DbError::Query(format!(
            "parameter '{key}' must be an integer, got {other:?}"
        ))),
    }
}

/// Extract an optional `bool` parameter with a fallback default.
pub fn opt_bool(
    params: &HashMap<String, Value>,
    key: &str,
    default: bool,
) -> Result<bool, DbError> {
    match params.get(key) {
        None => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(other) => Err(DbError::Query(format!(
            "parameter '{key}' must be a boolean, got {other:?}"
        ))),
    }
}

/// Extract an optional [`Direction`] parameter with a fallback default.
///
/// The value must be a string; see [`Direction::from_str`] for accepted values.
pub fn opt_direction(
    params: &HashMap<String, Value>,
    key: &str,
    default: Direction,
) -> Result<Direction, DbError> {
    match params.get(key) {
        None => Ok(default),
        Some(Value::String(s)) => Direction::from_str(s),
        Some(other) => Err(DbError::Query(format!(
            "parameter '{key}' must be a string (\"out\", \"in\", or \"any\"), got {other:?}"
        ))),
    }
}

/// Extract an optional string parameter with a fallback default.
pub fn opt_str<'a>(
    params: &'a HashMap<String, Value>,
    key: &str,
    default: &'a str,
) -> Result<&'a str, DbError> {
    match params.get(key) {
        None => Ok(default),
        Some(Value::String(s)) => Ok(s.as_str()),
        Some(other) => Err(DbError::Query(format!(
            "parameter '{key}' must be a string, got {other:?}"
        ))),
    }
}

/// Extract the optional `weight` parameter as a property name.
///
/// Returns `None` when the `"weight"` key is absent (unweighted mode).
/// Returns the property name string when present.
/// Errors if the value is not a string.
pub fn opt_weight_prop<'a>(
    params: &'a HashMap<String, Value>,
) -> Result<Option<&'a str>, DbError> {
    match params.get("weight") {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(other) => Err(DbError::Query(format!(
            "parameter 'weight' must be a string property name, got {other:?}"
        ))),
    }
}

// ── Algorithm dispatch ────────────────────────────────────────────────────────

/// Dispatch a `CALL algorithmName(...)` statement to the appropriate algorithm
/// implementation.
///
/// Called by the query executor whenever it encounters a `CallStatement` in
/// the AST.  The `name` is matched case-sensitively (a few common aliases
/// like `"pagerank"` are also accepted).
///
/// Returns a descriptive error listing all available algorithm names if
/// `name` is not recognised.
pub fn dispatch_call(
    graph: &Graph,
    name: &str,
    params: &HashMap<String, Value>,
) -> Result<Vec<Row>, DbError> {
    match name {
        "bfs" | "dfs" => bfs::run(graph, params),
        "shortestPath" => shortest_path::run(graph, params),
        "wcc" => components::run_wcc(graph, params),
        "scc" => components::run_scc(graph, params),
        "pageRank" | "pagerank" => centrality::run_pagerank(graph, params),
        "betweennessCentrality" => centrality::run_betweenness(graph, params),
        "closenessCentrality" => centrality::run_closeness(graph, params),
        "degreeCentrality" => centrality::run_degree(graph, params),
        "triangleCount" => triangle::run(graph, params),
        "jaccardSimilarity" => similarity::run_jaccard(graph, params),
        "labelPropagation" => community::run_label_propagation(graph, params),
        "louvain" => louvain::run_louvain(graph, params),
        "leiden" => louvain::run_leiden(graph, params),
        "maxFlow" => flow::run_max_flow(graph, params),
        other => Err(DbError::Query(format!(
            "unknown algorithm '{other}'. Available algorithms: \
             bfs, dfs, shortestPath, wcc, scc, pageRank, \
             betweennessCentrality, closenessCentrality, degreeCentrality, \
             triangleCount, jaccardSimilarity, labelPropagation, \
             louvain, leiden, maxFlow"
        ))),
    }
}
