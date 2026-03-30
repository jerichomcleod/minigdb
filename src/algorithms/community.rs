//! Community detection via Label Propagation.
//!
//! **CALL signature:**
//! ```gql
//! CALL labelPropagation(maxIterations: 10, direction: "any")
//! YIELD node, community
//! ```
//!
//! | param           | type   | default  | description                                     |
//! |-----------------|--------|----------|-------------------------------------------------|
//! | `maxIterations` | Int    | 10       | Maximum propagation rounds                      |
//! | `direction`     | String | `"any"`  | `"out"`, `"in"`, or `"any"` for propagation     |
//!
//! **Yields:** `node` (String), `community` (String — representative node ULID)
//!
//! Algorithm:
//! 1. Initialise: each node gets its own index as community label.
//! 2. For up to `maxIterations` rounds, in a fixed traversal order:
//!    each node adopts the most frequent label among its neighbours
//!    (ties broken by minimum label value for determinism).
//! 3. Repeat until no label changes or max iterations reached.
//! 4. The representative of each community is the node whose index
//!    equals the final community label.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::graph::Graph;
use crate::types::{ulid_encode, DbError, Value};

use super::{opt_direction, opt_usize, Direction, GraphSnapshot, Row};

const PAR_N: usize = 256;

pub fn run_label_propagation(
    graph: &Graph,
    params: &HashMap<String, Value>,
) -> Result<Vec<Row>, DbError> {
    let max_iter = opt_usize(params, "maxIterations", 10)?;
    if max_iter == 0 {
        return Err(DbError::Query(
            "parameter 'maxIterations' must be at least 1".into(),
        ));
    }
    let direction = opt_direction(params, "direction", Direction::Any)?;

    let snap = GraphSnapshot::build(graph, None);
    let n = snap.n;

    if n == 0 {
        return Ok(vec![]);
    }

    // Initialise: each node is its own community (label = node index).
    let mut label: Vec<usize> = (0..n).collect();

    for _ in 0..max_iter {
        // Synchronous propagation: read from prev_label, write to new label vec.
        // All per-node updates depend only on prev_label (not each other),
        // so the loop is embarrassingly parallel.
        let prev_label = label.clone();

        let update_node = |i: usize| -> (usize, bool) {
            let nbrs = snap.unique_neighbor_indices(i, direction);
            if nbrs.is_empty() {
                return (prev_label[i], false);
            }
            let mut freq: HashMap<usize, usize> = HashMap::new();
            for &j in &nbrs {
                *freq.entry(prev_label[j]).or_insert(0) += 1;
            }
            let max_freq = *freq.values().max().unwrap();
            let best = freq
                .into_iter()
                .filter(|&(_, f)| f == max_freq)
                .map(|(lbl, _)| lbl)
                .min()
                .unwrap();
            (best, prev_label[i] != best)
        };

        let updates: Vec<(usize, bool)> = if n >= PAR_N {
            (0..n).into_par_iter().map(update_node).collect()
        } else {
            (0..n).map(update_node).collect()
        };

        let changed = updates.iter().any(|&(_, c)| c);
        label = updates.into_iter().map(|(l, _)| l).collect();

        if !changed {
            break;
        }
    }

    // After propagation, compress labels: find the canonical representative of
    // each community (smallest-index node with that label).
    let mut rep: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        rep.entry(label[i]).or_insert(i);
    }

    Ok((0..n)
        .map(|i| {
            let community_rep = rep[&label[i]];
            let mut row = HashMap::new();
            row.insert(
                "node".to_string(),
                Value::String(ulid_encode(snap.node_ids[i].0)),
            );
            row.insert(
                "community".to_string(),
                Value::String(ulid_encode(snap.node_ids[community_rep].0)),
            );
            row
        })
        .collect())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::types::{Edge, Node, NodeId, Value};

    fn make_node(g: &mut Graph) -> NodeId {
        let id = g.alloc_node_id();
        g.apply_insert_node(Node {
            id,
            labels: vec!["N".into()],
            properties: Default::default(),
        });
        id
    }

    fn make_edge(g: &mut Graph, from: NodeId, to: NodeId) {
        let id = g.alloc_edge_id();
        g.apply_insert_edge(Edge {
            id,
            from_node: from,
            to_node: to,
            label: "E".into(),
            properties: Default::default(),
            directed: false,
        });
    }

    #[test]
    fn two_disconnected_cliques_form_two_communities() {
        // Two triangle cliques with NO bridge between them.
        // Label propagation must produce exactly 2 distinct communities.
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        let d = make_node(&mut g);
        let e = make_node(&mut g);
        let f = make_node(&mut g);

        // Clique 1: a-b-c
        make_edge(&mut g, a, b);
        make_edge(&mut g, b, c);
        make_edge(&mut g, a, c);
        // Clique 2: d-e-f  (no connection to clique 1)
        make_edge(&mut g, d, e);
        make_edge(&mut g, e, f);
        make_edge(&mut g, d, f);

        let params: HashMap<String, Value> =
            [("maxIterations".to_string(), Value::Int(20))].into_iter().collect();
        let rows = run_label_propagation(&g, &params).unwrap();
        assert_eq!(rows.len(), 6);

        // Both cliques must collapse to a single community each, and those
        // communities must be distinct.
        let community_of = |id: crate::types::NodeId| -> String {
            rows.iter()
                .find(|r| r["node"] == Value::String(crate::types::ulid_encode(id.0)))
                .map(|r| match &r["community"] {
                    Value::String(s) => s.clone(),
                    other => panic!("expected String, got {other:?}"),
                })
                .unwrap()
        };

        let comm_a = community_of(a);
        let comm_b = community_of(b);
        let comm_c = community_of(c);
        let comm_d = community_of(d);
        let comm_e = community_of(e);
        let comm_f = community_of(f);

        // All nodes in clique 1 have the same community label.
        assert_eq!(comm_a, comm_b);
        assert_eq!(comm_b, comm_c);
        // All nodes in clique 2 have the same community label.
        assert_eq!(comm_d, comm_e);
        assert_eq!(comm_e, comm_f);
        // The two communities are distinct.
        assert_ne!(comm_a, comm_d);
    }

    #[test]
    fn fully_connected_single_community() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        make_edge(&mut g, a, b);
        make_edge(&mut g, b, c);
        make_edge(&mut g, a, c);

        let params = HashMap::new();
        let rows = run_label_propagation(&g, &params).unwrap();
        let comms: Vec<_> = rows.iter().map(|r| r["community"].clone()).collect();
        assert!(comms.iter().all(|c| *c == comms[0]));
    }

    #[test]
    fn isolated_nodes_are_own_communities() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        // No edges.

        let params = HashMap::new();
        let rows = run_label_propagation(&g, &params).unwrap();
        let comms: std::collections::HashSet<String> = rows
            .iter()
            .map(|r| match &r["community"] {
                Value::String(s) => s.clone(),
                other => panic!("expected String, got {other:?}"),
            })
            .collect();
        assert_eq!(comms.len(), 3); // each is its own community
    }

    #[test]
    fn zero_iterations_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> =
            [("maxIterations".to_string(), Value::Int(0))].into_iter().collect();
        assert!(run_label_propagation(&g, &params).is_err());
    }

    #[test]
    fn empty_graph_returns_empty() {
        let g = Graph::new();
        let rows = run_label_propagation(&g, &HashMap::new()).unwrap();
        assert!(rows.is_empty());
    }
}
