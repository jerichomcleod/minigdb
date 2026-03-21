//! Triangle counting and clustering coefficient.
//!
//! **CALL signature:**
//! ```gql
//! CALL triangleCount(mode: "local") YIELD node, triangles, coefficient
//! CALL triangleCount(mode: "global") YIELD triangles, transitivity
//! ```
//!
//! | param  | type   | default   | description                        |
//! |--------|--------|-----------|------------------------------------|
//! | `mode` | String | `"local"` | `"local"` or `"global"`            |
//!
//! **Local mode yields:** `node` (String), `triangles` (Int),
//!   `coefficient` (Float — local clustering coefficient)
//!
//! **Global mode yields:** one row — `triangles` (Int), `transitivity` (Float)
//!
//! The algorithm treats all edges as undirected and deduplicates parallel edges.
//! Self-loops are ignored.

use std::collections::{HashMap, HashSet};

use crate::graph::Graph;
use crate::types::{ulid_encode, DbError, Value};

use super::{opt_str, Direction, GraphSnapshot, Row};

/// Entry point called by `dispatch_call` for `CALL triangleCount(...)`.
///
/// Delegates to [`run_local`] or [`run_global`] based on the `mode` parameter.
pub fn run(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let mode = opt_str(params, "mode", "local")?;
    match mode {
        "local" => run_local(graph),
        "global" => run_global(graph),
        other => Err(DbError::Query(format!(
            "unknown 'mode' value '{other}': expected \"local\" or \"global\""
        ))),
    }
}

/// Per-node triangle count and local clustering coefficient.
///
/// For each node `u`, counts closed triangles in its neighbourhood by
/// checking every pair of neighbours for a connecting edge — O(N · d²)
/// where `d` is the average degree.
///
/// The local clustering coefficient `C(u)` = triangles(u) / (d·(d-1)/2).
/// For nodes with degree < 2, `C(u) = 0`.
fn run_local(graph: &Graph) -> Result<Vec<Row>, DbError> {
    let snap = GraphSnapshot::build(graph, None);
    let n = snap.n;

    if n == 0 {
        return Ok(vec![]);
    }

    // Build sorted adjacency sets (undirected, no self-loops, deduped).
    let adj: Vec<HashSet<usize>> = (0..n)
        .map(|i| {
            snap.neighbors(i, Direction::Any)
                .into_iter()
                .map(|(j, _)| j)
                .filter(|&j| j != i)
                .collect()
        })
        .collect();

    let mut rows = Vec::with_capacity(n);
    for u in 0..n {
        let nbrs_u = &adj[u];
        let deg = nbrs_u.len();

        // Count triangles: pairs (v, w) in N(u) where v–w edge exists.
        let mut triangles = 0usize;
        let nbrs_vec: Vec<usize> = nbrs_u.iter().copied().collect();
        for i in 0..nbrs_vec.len() {
            for j in (i + 1)..nbrs_vec.len() {
                let v = nbrs_vec[i];
                let w = nbrs_vec[j];
                if adj[v].contains(&w) {
                    triangles += 1;
                }
            }
        }

        // Each triangle is counted once per node.
        let coefficient = if deg < 2 {
            0.0
        } else {
            triangles as f64 / (deg * (deg - 1) / 2) as f64
        };

        let mut row = HashMap::new();
        row.insert(
            "node".to_string(),
            Value::String(ulid_encode(snap.node_ids[u].0)),
        );
        row.insert("triangles".to_string(), Value::Int(triangles as i64));
        row.insert("coefficient".to_string(), Value::Float(coefficient));
        rows.push(row);
    }

    Ok(rows)
}

/// Graph-wide triangle count and transitivity (global clustering coefficient).
///
/// Counts total triangles across the graph by scanning each node's
/// neighbourhood pairs, then divides by 3 to remove the triple-counting
/// (each triangle is counted once per vertex).
///
/// Transitivity = 3 · triangles / open_triplets  (Watts-Strogatz definition).
fn run_global(graph: &Graph) -> Result<Vec<Row>, DbError> {
    let snap = GraphSnapshot::build(graph, None);
    let n = snap.n;

    if n == 0 {
        let mut row = HashMap::new();
        row.insert("triangles".to_string(), Value::Int(0));
        row.insert("transitivity".to_string(), Value::Float(0.0));
        return Ok(vec![row]);
    }

    // Build adjacency sets (undirected, deduped, no self-loops).
    let adj: Vec<HashSet<usize>> = (0..n)
        .map(|i| {
            snap.neighbors(i, Direction::Any)
                .into_iter()
                .map(|(j, _)| j)
                .filter(|&j| j != i)
                .collect()
        })
        .collect();

    // Count global triangles (each triangle counted 3× — once per vertex).
    let mut tri_times3 = 0usize;
    let mut open_triplets = 0usize;

    for u in 0..n {
        let nbrs: Vec<usize> = adj[u].iter().copied().collect();
        let deg = nbrs.len();
        if deg >= 2 {
            open_triplets += deg * (deg - 1) / 2;
        }
        for i in 0..nbrs.len() {
            for j in (i + 1)..nbrs.len() {
                if adj[nbrs[i]].contains(&nbrs[j]) {
                    tri_times3 += 1;
                }
            }
        }
    }

    let total_triangles = tri_times3 / 3;
    let transitivity = if open_triplets == 0 {
        0.0
    } else {
        3.0 * total_triangles as f64 / open_triplets as f64
    };

    let mut row = HashMap::new();
    row.insert("triangles".to_string(), Value::Int(total_triangles as i64));
    row.insert("transitivity".to_string(), Value::Float(transitivity));
    Ok(vec![row])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::types::{Edge, Node, NodeId, Value};

    /// Insert a bare node with label "N" and no properties.
    fn make_node(g: &mut Graph) -> NodeId {
        let id = g.alloc_node_id();
        g.apply_insert_node(Node {
            id,
            labels: vec!["N".into()],
            properties: Default::default(),
        });
        id
    }

    /// Insert a directed edge between `from` and `to`.
    fn make_edge(g: &mut Graph, from: NodeId, to: NodeId) {
        let id = g.alloc_edge_id();
        g.apply_insert_edge(Edge {
            id,
            from_node: from,
            to_node: to,
            label: "E".into(),
            properties: Default::default(),
            directed: true,
        });
    }

    /// Triangle: a — b — c — a
    fn triangle_graph() -> (Graph, NodeId, NodeId, NodeId) {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        make_edge(&mut g, a, b);
        make_edge(&mut g, b, c);
        make_edge(&mut g, c, a);
        (g, a, b, c)
    }

    #[test]
    fn local_triangle_detected() {
        let (g, a, b, c) = triangle_graph();
        let params: HashMap<String, Value> =
            [("mode".to_string(), Value::String("local".into()))].into_iter().collect();
        let rows = run(&g, &params).unwrap();
        // Each node should report 1 triangle.
        for row in &rows {
            assert_eq!(row["triangles"], Value::Int(1));
            // coefficient = 1/1 = 1.0 (2 possible edges among 2 neighbours, 1 present)
            if let Value::Float(c) = row["coefficient"] {
                assert!((c - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn global_triangle_count() {
        let (g, _, _, _) = triangle_graph();
        let params: HashMap<String, Value> =
            [("mode".to_string(), Value::String("global".into()))].into_iter().collect();
        let rows = run(&g, &params).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["triangles"], Value::Int(1));
        if let Value::Float(t) = rows[0]["transitivity"] {
            assert!((t - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn no_triangles_in_path() {
        // a — b — c  (path, no triangle)
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        make_edge(&mut g, a, b);
        make_edge(&mut g, b, c);

        let rows = run(&g, &HashMap::new()).unwrap();
        for row in &rows {
            assert_eq!(row["triangles"], Value::Int(0));
        }
    }

    #[test]
    fn coefficient_zero_for_leaf() {
        // Leaf node (degree 1) always has coefficient 0.
        let (g, a, _, _) = triangle_graph();
        // Add extra isolated leaf.
        let mut g = g;
        let leaf = make_node(&mut g);
        make_edge(&mut g, leaf, a);

        let rows = run(&g, &HashMap::new()).unwrap();
        let leaf_row = rows.iter()
            .find(|r| r["node"] == Value::String(crate::types::ulid_encode(leaf.0)))
            .unwrap();
        if let Value::Float(c) = leaf_row["coefficient"] {
            assert!((c - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_mode_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> =
            [("mode".to_string(), Value::String("both".into()))].into_iter().collect();
        assert!(run(&g, &params).is_err());
    }

    #[test]
    fn empty_graph_local() {
        let g = Graph::new();
        let rows = run(&g, &HashMap::new()).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn empty_graph_global() {
        let g = Graph::new();
        let params: HashMap<String, Value> =
            [("mode".to_string(), Value::String("global".into()))].into_iter().collect();
        let rows = run(&g, &params).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["triangles"], Value::Int(0));
    }
}
