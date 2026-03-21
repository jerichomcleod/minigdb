//! Breadth-first search (BFS) and depth-first search (DFS) graph traversal.
//!
//! Both algorithms start from a single source node and visit reachable nodes
//! level by level (BFS) or branch by branch (DFS).  They share the same `CALL`
//! syntax and differ only in the `algorithm` parameter.
//!
//! ## CALL Syntax
//!
//! ```gql
//! CALL bfs(start: "<ulid>", maxDepth: 5, direction: "out", algorithm: "bfs")
//! YIELD node, depth, predecessor
//! ```
//!
//! ## Parameters
//!
//! | name        | type   | default   | description                                         |
//! |-------------|--------|-----------|-----------------------------------------------------|
//! | `start`     | String | *required*| ULID of the starting node                           |
//! | `maxDepth`  | Int    | unlimited | Maximum hop depth; `0` returns only the start node  |
//! | `direction` | String | `"out"`   | Edge direction: `"out"`, `"in"`, or `"any"`         |
//! | `algorithm` | String | `"bfs"`   | `"bfs"` for breadth-first, `"dfs"` for depth-first |
//!
//! ## Yields
//!
//! | column        | type           | description                                |
//! |---------------|----------------|--------------------------------------------|
//! | `node`        | String (ULID)  | Node ID of the visited node                |
//! | `depth`       | Int            | Hop distance from the start node           |
//! | `predecessor` | String \| null | ULID of the node from which this was reached|
//!
//! ## Complexity
//!
//! Both BFS and DFS run in O(N + E) time and O(N) space for the visited set
//! and traversal queue/stack.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::graph::Graph;
use crate::types::{ulid_encode, DbError, Value};

use super::{opt_direction, opt_str, opt_usize, require_node_idx, Direction, GraphSnapshot, Row};

/// Entry point for `CALL bfs(...)` and `CALL dfs(...)`.
///
/// Validates parameters, builds the in-memory snapshot, then dispatches to
/// [`run_bfs`] or [`run_dfs`] based on the `algorithm` parameter.
///
/// # Errors
///
/// Returns an error if:
/// - `start` is missing or is not a valid ULID.
/// - `start` does not refer to an existing node.
/// - `algorithm` is not `"bfs"` or `"dfs"`.
/// - `direction` is not a recognised direction string.
pub fn run(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    // Validate the algorithm selector first so a wrong value always errors,
    // even on an empty graph.
    let algo = opt_str(params, "algorithm", "bfs")?;
    if !matches!(algo, "bfs" | "dfs") {
        return Err(DbError::Query(format!(
            "unknown 'algorithm' value '{algo}': expected \"bfs\" or \"dfs\""
        )));
    }
    let direction = opt_direction(params, "direction", Direction::Out)?;
    let max_depth = opt_usize(params, "maxDepth", usize::MAX)?;

    // Validate 'start' exists in params before building the (potentially
    // expensive) snapshot so that missing-param errors are fast.
    let _ = super::require_str(params, "start")?;

    let snap = GraphSnapshot::build(graph, None);
    if snap.n == 0 {
        return Ok(vec![]);
    }

    let start = require_node_idx(params, "start", &snap)?;

    match algo {
        "bfs" => run_bfs(&snap, start, max_depth, direction),
        _ => run_dfs(&snap, start, max_depth, direction),
    }
}

/// Iterative breadth-first search from `start`.
///
/// Uses a `VecDeque` as a FIFO queue.  A `HashSet` tracks visited nodes to
/// prevent revisiting in cyclic graphs.  Each node is emitted at most once,
/// at its shortest hop distance from `start`.
///
/// # Returns
///
/// Rows in BFS discovery order (level 0, then level 1, etc.).
fn run_bfs(
    snap: &GraphSnapshot,
    start: usize,
    max_depth: usize,
    dir: Direction,
) -> Result<Vec<Row>, DbError> {
    let mut visited: HashSet<usize> = HashSet::new();
    // Each entry: (node_index, hop_depth, optional_predecessor_index)
    let mut queue: VecDeque<(usize, usize, Option<usize>)> = VecDeque::new();
    let mut rows = Vec::new();

    visited.insert(start);
    queue.push_back((start, 0, None));

    while let Some((curr, depth, pred)) = queue.pop_front() {
        rows.push(make_row(snap, curr, depth, pred));

        // Do not expand nodes beyond the requested depth limit.
        if depth < max_depth {
            for (nbr, _) in snap.neighbors(curr, dir) {
                // `visited.insert` returns false if already present — skip those.
                if visited.insert(nbr) {
                    queue.push_back((nbr, depth + 1, Some(curr)));
                }
            }
        }
    }

    Ok(rows)
}

/// Iterative depth-first search from `start`.
///
/// Uses an explicit stack instead of recursion to avoid stack-overflow on
/// deep graphs.  Neighbours are pushed in reverse order so that the
/// left-to-right adjacency list order is preserved in the DFS visit order.
///
/// # Returns
///
/// Rows in DFS discovery order (pre-order).
fn run_dfs(
    snap: &GraphSnapshot,
    start: usize,
    max_depth: usize,
    dir: Direction,
) -> Result<Vec<Row>, DbError> {
    let mut visited: HashSet<usize> = HashSet::new();
    // Each stack entry: (node_index, hop_depth, optional_predecessor_index)
    let mut stack: Vec<(usize, usize, Option<usize>)> = vec![(start, 0, None)];
    let mut rows = Vec::new();

    while let Some((curr, depth, pred)) = stack.pop() {
        // A node may appear multiple times on the stack (once per predecessor).
        // Skip if already visited.
        if !visited.insert(curr) {
            continue;
        }
        rows.push(make_row(snap, curr, depth, pred));

        if depth < max_depth {
            let nbrs = snap.neighbors(curr, dir);
            // Push in reverse order so the first neighbour ends up on top of
            // the stack and is visited next (preserving adjacency list order).
            for (nbr, _) in nbrs.into_iter().rev() {
                if !visited.contains(&nbr) {
                    stack.push((nbr, depth + 1, Some(curr)));
                }
            }
        }
    }

    Ok(rows)
}

/// Build a single output row for the visited node at the given depth.
fn make_row(snap: &GraphSnapshot, node: usize, depth: usize, pred: Option<usize>) -> Row {
    let mut row = HashMap::new();
    row.insert(
        "node".to_string(),
        Value::String(ulid_encode(snap.node_ids[node].0)),
    );
    row.insert("depth".to_string(), Value::Int(depth as i64));
    row.insert(
        "predecessor".to_string(),
        // The start node has no predecessor; encode that as SQL NULL.
        pred.map(|p| Value::String(ulid_encode(snap.node_ids[p].0)))
            .unwrap_or(Value::Null),
    );
    row
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::types::{ulid_encode, Value};

    /// Insert a node with label "N" and a `name` property.
    fn insert_node(g: &mut Graph, name: &str) -> crate::types::NodeId {
        let mut next = 0u64;
        let id = g.alloc_node_id();
        let node = crate::types::Node {
            id,
            labels: vec!["N".into()],
            properties: [("name".to_string(), Value::String(name.to_string()))]
                .into_iter()
                .collect(),
        };
        g.apply_insert_node(node);
        let _ = next;
        id
    }

    /// Insert a directed edge from `from` to `to` with no properties.
    fn insert_edge(g: &mut Graph, from: crate::types::NodeId, to: crate::types::NodeId) {
        let id = g.alloc_edge_id();
        let edge = crate::types::Edge {
            id,
            from_node: from,
            to_node: to,
            label: "E".into(),
            properties: Default::default(),
            directed: true,
        };
        g.apply_insert_edge(edge);
    }

    /// Build a linear chain: a → b → c → d
    fn linear_graph() -> (Graph, Vec<crate::types::NodeId>) {
        let mut g = Graph::new();
        let ids: Vec<_> = ["a", "b", "c", "d"].iter().map(|n| insert_node(&mut g, n)).collect();
        insert_edge(&mut g, ids[0], ids[1]);
        insert_edge(&mut g, ids[1], ids[2]);
        insert_edge(&mut g, ids[2], ids[3]);
        (g, ids)
    }

    #[test]
    fn bfs_linear_full_depth() {
        let (g, ids) = linear_graph();
        let params: HashMap<String, Value> = [(
            "start".to_string(),
            Value::String(ulid_encode(ids[0].0)),
        )]
        .into_iter()
        .collect();
        let rows = run(&g, &params).unwrap();
        assert_eq!(rows.len(), 4); // visits all 4 nodes
        // BFS order: depths 0, 1, 2, 3
        let depths: Vec<i64> = rows.iter().map(|r| {
            if let Value::Int(d) = r["depth"] { d } else { panic!() }
        }).collect();
        assert_eq!(depths, vec![0, 1, 2, 3]);
    }

    #[test]
    fn bfs_max_depth_one() {
        let (g, ids) = linear_graph();
        let params: HashMap<String, Value> = [
            ("start".to_string(), Value::String(ulid_encode(ids[0].0))),
            ("maxDepth".to_string(), Value::Int(1)),
        ]
        .into_iter()
        .collect();
        let rows = run(&g, &params).unwrap();
        assert_eq!(rows.len(), 2); // start + one hop
    }

    #[test]
    fn bfs_depth_zero_returns_start_only() {
        let (g, ids) = linear_graph();
        let params: HashMap<String, Value> = [
            ("start".to_string(), Value::String(ulid_encode(ids[0].0))),
            ("maxDepth".to_string(), Value::Int(0)),
        ]
        .into_iter()
        .collect();
        let rows = run(&g, &params).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["depth"], Value::Int(0));
        assert_eq!(rows[0]["predecessor"], Value::Null);
    }

    #[test]
    fn dfs_visits_all_nodes() {
        let (g, ids) = linear_graph();
        let params: HashMap<String, Value> = [
            ("start".to_string(), Value::String(ulid_encode(ids[0].0))),
            ("algorithm".to_string(), Value::String("dfs".into())),
        ]
        .into_iter()
        .collect();
        let rows = run(&g, &params).unwrap();
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn bfs_direction_any_undirected() {
        let mut g = Graph::new();
        let a = insert_node(&mut g, "a");
        let b = insert_node(&mut g, "b");
        // undirected edge
        let id = g.alloc_edge_id();
        g.apply_insert_edge(crate::types::Edge {
            id,
            from_node: a,
            to_node: b,
            label: "E".into(),
            properties: Default::default(),
            directed: false,
        });
        // BFS from b going "out" on a directed graph would miss a.
        // With direction "any" on undirected edge we should find a from b.
        let params: HashMap<String, Value> = [
            ("start".to_string(), Value::String(ulid_encode(b.0))),
            ("direction".to_string(), Value::String("any".into())),
        ]
        .into_iter()
        .collect();
        let rows = run(&g, &params).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn bfs_cycle_does_not_loop() {
        // a → b → c → a
        let mut g = Graph::new();
        let a = insert_node(&mut g, "a");
        let b = insert_node(&mut g, "b");
        let c = insert_node(&mut g, "c");
        insert_edge(&mut g, a, b);
        insert_edge(&mut g, b, c);
        insert_edge(&mut g, c, a);
        let params: HashMap<String, Value> = [(
            "start".to_string(),
            Value::String(ulid_encode(a.0)),
        )]
        .into_iter()
        .collect();
        let rows = run(&g, &params).unwrap();
        assert_eq!(rows.len(), 3); // each node visited exactly once
    }

    #[test]
    fn missing_start_param_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> = HashMap::new();
        assert!(run(&g, &params).is_err());
    }

    #[test]
    fn invalid_direction_errors() {
        let (g, ids) = linear_graph();
        let params: HashMap<String, Value> = [
            ("start".to_string(), Value::String(ulid_encode(ids[0].0))),
            ("direction".to_string(), Value::String("sideways".into())),
        ]
        .into_iter()
        .collect();
        assert!(run(&g, &params).is_err());
    }

    #[test]
    fn empty_graph_returns_empty() {
        let g = Graph::new();
        // No nodes — nothing to traverse.
        let params: HashMap<String, Value> = HashMap::new();
        let rows = run(&g, &params).unwrap_or_default();
        assert!(rows.is_empty());
    }
}
