//! Single-source and single-pair shortest paths via Dijkstra's algorithm.
//!
//! Dijkstra's algorithm finds minimum-cost paths in graphs with **non-negative**
//! edge weights.  The implementation uses a binary min-heap for O((N + E) log N)
//! time.  When no weight property is given all edges have weight `1.0` (unweighted
//! BFS-equivalent behaviour).
//!
//! ## CALL Syntax
//!
//! ```gql
//! -- Single-pair shortest path
//! CALL shortestPath(source: "<ulid>", target: "<ulid>", weight: "cost", direction: "out")
//! YIELD source, target, cost, path
//!
//! -- Single-source to all reachable nodes
//! CALL shortestPath(source: "<ulid>", weight: "cost")
//! YIELD source, target, cost, path
//! ```
//!
//! ## Parameters
//!
//! | name        | type   | default    | description                                       |
//! |-------------|--------|------------|---------------------------------------------------|
//! | `source`    | String | *required* | ULID of the source node                           |
//! | `target`    | String | optional   | ULID of target; if omitted, computes to all nodes |
//! | `weight`    | String | none       | Edge property for weight; absent = all edges = 1  |
//! | `direction` | String | `"out"`    | `"out"`, `"in"`, or `"any"`                       |
//! | `maxCost`   | Float  | ∞          | Prune paths exceeding this accumulated cost       |
//!
//! ## Yields
//!
//! | column   | type          | description                              |
//! |----------|---------------|------------------------------------------|
//! | `source` | String (ULID) | The source node ID                       |
//! | `target` | String (ULID) | The reached target node ID               |
//! | `cost`   | Float         | Total path cost from source to target    |
//! | `path`   | List[String]  | Ordered list of node ULIDs on the path   |
//!
//! ## Complexity
//!
//! O((N + E) log N) time with a binary min-heap, O(N + E) space.
//!
//! ## Constraints
//!
//! Negative edge weights are rejected with an error.  For graphs with negative
//! weights consider a Bellman-Ford variant.

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;

use crate::graph::Graph;
use crate::types::{ulid_encode, DbError, Value};

use super::{
    opt_direction, opt_f64, opt_node_idx, opt_weight_prop, require_node_idx, Direction,
    GraphSnapshot, Row,
};

/// Entry point for `CALL shortestPath(...)`.
///
/// Validates parameters and delegates to [`dijkstra`].
///
/// # Errors
///
/// Returns an error if:
/// - `source` is missing or not a valid ULID.
/// - `target` is provided but is not a valid ULID or node.
/// - `maxCost` is negative.
/// - A negative edge weight is encountered during traversal.
pub fn run(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let weight_prop = opt_weight_prop(params)?;
    let snap = GraphSnapshot::build(graph, weight_prop);

    if snap.n == 0 {
        return Ok(vec![]);
    }

    let source = require_node_idx(params, "source", &snap)?;
    let target = opt_node_idx(params, "target", &snap)?;
    let direction = opt_direction(params, "direction", Direction::Out)?;
    let max_cost = opt_f64(params, "maxCost", f64::INFINITY)?;

    if max_cost < 0.0 {
        return Err(DbError::Query("parameter 'maxCost' must be non-negative".into()));
    }

    dijkstra(&snap, source, target, direction, max_cost)
}

/// Dijkstra's single-source shortest-path algorithm.
///
/// Uses a lazy-deletion min-heap: stale (node, distance) pairs are left in
/// the heap and skipped when popped if a shorter distance has already been
/// recorded (`d > dist[u]` guard).
///
/// If `target` is `Some(t)`, traversal stops as soon as `t` is settled,
/// returning a single row.  If `target` is `None`, all reachable nodes are
/// returned (single-source all-targets mode).
///
/// # Algorithm overview
///
/// 1. Initialise `dist[source] = 0`, `dist[v] = ∞` for all other nodes.
/// 2. Push `(0, source)` onto the min-heap.
/// 3. While the heap is non-empty:
///    a. Pop the minimum-distance entry `(d, u)`.
///    b. If `d > dist[u]`, skip (stale entry).
///    c. If `u == target`, stop early.
///    d. For each neighbor `v` of `u` with edge weight `w ≥ 0`:
///       - If `dist[u] + w < dist[v]`, relax and push `(dist[u]+w, v)`.
/// 4. Reconstruct paths via the `prev` predecessor array.
///
/// # Notes on f64 Ordering in BinaryHeap
///
/// `BinaryHeap` requires `Ord`, which `f64` does not implement due to NaN.
/// The workaround is to store distances as their raw IEEE-754 bit pattern
/// (`f64::to_bits()`) — for non-negative finite values, the bit-pattern order
/// matches the numeric order.
fn dijkstra(
    snap: &GraphSnapshot,
    source: usize,
    target: Option<usize>,
    dir: Direction,
    max_cost: f64,
) -> Result<Vec<Row>, DbError> {
    let n = snap.n;
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];
    dist[source] = 0.0;

    // BinaryHeap is a max-heap; Reverse<_> converts it to a min-heap.
    // Distances are stored as raw u64 bit patterns to satisfy Ord.
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((0u64, source)));

    while let Some(Reverse((d_bits, u))) = heap.pop() {
        let d = f64::from_bits(d_bits);

        // Skip stale heap entries (a shorter path was already found for u).
        if d > dist[u] {
            continue;
        }

        // Early termination when a specific target has been settled.
        if target == Some(u) {
            break;
        }

        for (v, w) in snap.neighbors(u, dir) {
            if w < 0.0 {
                return Err(DbError::Query(format!(
                    "shortestPath: negative edge weight {w} encountered. \
                     Dijkstra requires non-negative weights. \
                     Consider using absolute weights or a Bellman-Ford variant."
                )));
            }
            let new_dist = dist[u] + w;
            if new_dist < dist[v] && new_dist <= max_cost {
                dist[v] = new_dist;
                prev[v] = Some(u);
                heap.push(Reverse((new_dist.to_bits(), v)));
            }
        }
    }

    let source_str = Value::String(ulid_encode(snap.node_ids[source].0));

    // Determine which target nodes to emit rows for.
    let targets: Vec<usize> = match target {
        Some(t) => {
            // Single-pair mode: emit only if the target was reached.
            if dist[t].is_finite() { vec![t] } else { vec![] }
        }
        // Single-source mode: emit all reachable nodes except the source itself.
        None => (0..n).filter(|&i| i != source && dist[i].is_finite()).collect(),
    };

    let mut rows = Vec::with_capacity(targets.len());
    for t in targets {
        let path = reconstruct_path(&prev, source, t, snap);
        let mut row = HashMap::new();
        row.insert("source".to_string(), source_str.clone());
        row.insert("target".to_string(), Value::String(ulid_encode(snap.node_ids[t].0)));
        row.insert("cost".to_string(), Value::Float(dist[t]));
        row.insert("path".to_string(), Value::List(path));
        rows.push(row);
    }
    Ok(rows)
}

/// Walk the `prev` predecessor array backwards from `target` to `source` and
/// return the path as an ordered list of ULID strings.
///
/// The result is reversed at the end so the list runs source → ... → target.
fn reconstruct_path(
    prev: &[Option<usize>],
    source: usize,
    target: usize,
    snap: &GraphSnapshot,
) -> Vec<Value> {
    let mut path = Vec::new();
    let mut curr = target;
    loop {
        path.push(Value::String(ulid_encode(snap.node_ids[curr].0)));
        if curr == source {
            break;
        }
        match prev[curr] {
            Some(p) => curr = p,
            None => break, // disconnected path (should not happen for finite dist)
        }
    }
    // Path was built target → source; reverse to get source → target order.
    path.reverse();
    path
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::types::{ulid_encode, Edge, EdgeId, Node, NodeId, Value};

    /// Insert a node with label "N" and a `name` property.
    fn make_node(g: &mut Graph, name: &str) -> NodeId {
        let id = g.alloc_node_id();
        g.apply_insert_node(Node {
            id,
            labels: vec!["N".into()],
            properties: [("name".to_string(), Value::String(name.to_string()))]
                .into_iter()
                .collect(),
        });
        id
    }

    /// Insert a directed edge from `from` to `to` with `"w"` weight property set to `weight`.
    fn make_edge(g: &mut Graph, from: NodeId, to: NodeId, weight: f64) {
        let id = g.alloc_edge_id();
        g.apply_insert_edge(Edge {
            id,
            from_node: from,
            to_node: to,
            label: "E".into(),
            properties: [("w".to_string(), Value::Float(weight))].into_iter().collect(),
            directed: true,
        });
    }

    #[test]
    fn unweighted_direct_path() {
        let mut g = Graph::new();
        let a = make_node(&mut g, "a");
        let b = make_node(&mut g, "b");
        make_edge(&mut g, a, b, 1.0);
        let params: HashMap<String, Value> = [
            ("source".to_string(), Value::String(ulid_encode(a.0))),
            ("target".to_string(), Value::String(ulid_encode(b.0))),
        ]
        .into_iter()
        .collect();
        let rows = run(&g, &params).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["cost"], Value::Float(1.0));
    }

    #[test]
    fn weighted_chooses_cheapest_path() {
        // a --(10)--> b --(1)--> c
        // a --(3)---> c  (direct)
        let mut g = Graph::new();
        let a = make_node(&mut g, "a");
        let b = make_node(&mut g, "b");
        let c = make_node(&mut g, "c");
        make_edge(&mut g, a, b, 10.0);
        make_edge(&mut g, b, c, 1.0);
        make_edge(&mut g, a, c, 3.0);

        let params: HashMap<String, Value> = [
            ("source".to_string(), Value::String(ulid_encode(a.0))),
            ("target".to_string(), Value::String(ulid_encode(c.0))),
            ("weight".to_string(), Value::String("w".into())),
        ]
        .into_iter()
        .collect();
        let rows = run(&g, &params).unwrap();
        assert_eq!(rows[0]["cost"], Value::Float(3.0));
        // path should be [a, c]
        if let Value::List(p) = &rows[0]["path"] {
            assert_eq!(p.len(), 2);
        } else {
            panic!("path not a list");
        }
    }

    #[test]
    fn no_path_returns_empty() {
        let mut g = Graph::new();
        let a = make_node(&mut g, "a");
        let b = make_node(&mut g, "b");
        // No edge between a and b.
        let params: HashMap<String, Value> = [
            ("source".to_string(), Value::String(ulid_encode(a.0))),
            ("target".to_string(), Value::String(ulid_encode(b.0))),
        ]
        .into_iter()
        .collect();
        let rows = run(&g, &params).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn negative_weight_errors() {
        let mut g = Graph::new();
        let a = make_node(&mut g, "a");
        let b = make_node(&mut g, "b");
        make_edge(&mut g, a, b, -5.0);
        let params: HashMap<String, Value> = [
            ("source".to_string(), Value::String(ulid_encode(a.0))),
            ("target".to_string(), Value::String(ulid_encode(b.0))),
            ("weight".to_string(), Value::String("w".into())),
        ]
        .into_iter()
        .collect();
        assert!(run(&g, &params).is_err());
    }

    #[test]
    fn all_targets_when_target_omitted() {
        let mut g = Graph::new();
        let a = make_node(&mut g, "a");
        let b = make_node(&mut g, "b");
        let c = make_node(&mut g, "c");
        make_edge(&mut g, a, b, 1.0);
        make_edge(&mut g, a, c, 2.0);
        let params: HashMap<String, Value> = [(
            "source".to_string(),
            Value::String(ulid_encode(a.0)),
        )]
        .into_iter()
        .collect();
        let rows = run(&g, &params).unwrap();
        assert_eq!(rows.len(), 2); // b and c both reachable
    }

    #[test]
    fn max_cost_prunes_long_paths() {
        let mut g = Graph::new();
        let a = make_node(&mut g, "a");
        let b = make_node(&mut g, "b");
        let c = make_node(&mut g, "c");
        make_edge(&mut g, a, b, 1.0);
        make_edge(&mut g, b, c, 5.0); // total 6 to c
        let params: HashMap<String, Value> = [
            ("source".to_string(), Value::String(ulid_encode(a.0))),
            ("weight".to_string(), Value::String("w".into())),
            ("maxCost".to_string(), Value::Float(3.0)),
        ]
        .into_iter()
        .collect();
        let rows = run(&g, &params).unwrap();
        // Only b is reachable within cost 3
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["cost"], Value::Float(1.0));
    }

    #[test]
    fn empty_graph_returns_empty() {
        let g = Graph::new();
        let params: HashMap<String, Value> = HashMap::new();
        assert!(run(&g, &params).is_ok()); // empty snap → Ok([])
    }
}
