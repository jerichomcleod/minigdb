//! Maximum flow via Edmonds-Karp (BFS-augmented Ford-Fulkerson).
//!
//! **CALL signature:**
//! ```gql
//! CALL maxFlow(source: "<ulid>", sink: "<ulid>", capacity: "cap")
//! YIELD source, sink, maxFlow
//! ```
//!
//! | param      | type   | default      | description                                |
//! |------------|--------|--------------|--------------------------------------------|
//! | `source`   | String | *required*   | ULID of the source node                    |
//! | `sink`     | String | *required*   | ULID of the sink node                      |
//! | `capacity` | String | `"capacity"` | Edge property for capacity (Int or Float)  |
//!
//! **Yields:** one row — `source` (String), `sink` (String), `maxFlow` (Float)
//!
//! **Requirements:**
//! - Edge capacities must be non-negative.
//! - Missing capacity property is treated as 0.
//! - Directed edges only; undirected edges contribute capacity in both directions.
//! - Self-loops are ignored.
//!
//! The algorithm runs in O(V · E²) time.

use std::collections::{HashMap, VecDeque};

use crate::graph::Graph;
use crate::types::{ulid_encode, DbError, Value};

use super::{opt_str, require_node_idx, GraphSnapshot, Row};

/// Entry point called by `dispatch_call` for `CALL maxFlow(...)`.
///
/// Builds a residual capacity graph from edge properties, then runs
/// Edmonds-Karp (BFS-augmented Ford-Fulkerson) to compute the maximum
/// flow from `source` to `sink`.
///
/// Returns a single result row: `{ source, sink, maxFlow }`.
pub fn run_max_flow(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let cap_prop = opt_str(params, "capacity", "capacity")?;

    // Build a node-index snapshot (unweighted — we'll read capacities directly
    // from edge properties so that missing properties default to 0, not 1.0).
    let snap = GraphSnapshot::build(graph, None);

    if snap.n == 0 {
        return Ok(vec![]);
    }

    let source = require_node_idx(params, "source", &snap)?;
    let sink = require_node_idx(params, "sink", &snap)?;

    if source == sink {
        return Err(DbError::Query(
            "maxFlow: 'source' and 'sink' must be different nodes".into(),
        ));
    }

    // Build the residual capacity matrix directly from edge data.
    // Missing capacity property → 0.0 (no flow allowed through that edge).
    let n = snap.n;
    let mut cap: HashMap<(usize, usize), f64> = HashMap::new();

    for edge in graph.all_edges() {
        let (Some(&fi), Some(&ti)) = (
            snap.id_to_idx.get(&edge.from_node),
            snap.id_to_idx.get(&edge.to_node),
        ) else {
            continue;
        };
        if fi == ti {
            continue; // skip self-loops
        }
        let w: f64 = edge.properties.get(cap_prop)
            .and_then(|v| match v {
                Value::Float(f) => Some(*f),
                Value::Int(i) => Some(*i as f64),
                _ => None,
            })
            .unwrap_or(0.0); // missing property = 0 capacity

        if w < 0.0 {
            return Err(DbError::Query(format!(
                "maxFlow: negative capacity {w}. Capacities must be non-negative."
            )));
        }
        *cap.entry((fi, ti)).or_insert(0.0) += w;
        cap.entry((ti, fi)).or_insert(0.0); // ensure reverse edge exists
        if !edge.directed {
            *cap.entry((ti, fi)).or_insert(0.0) += w;
            cap.entry((fi, ti)).or_insert(0.0);
        }
    }

    // Edmonds-Karp: repeatedly find augmenting paths via BFS.
    let mut max_flow = 0.0f64;

    loop {
        // BFS to find shortest augmenting path.
        let mut parent = vec![usize::MAX; n];
        parent[source] = source;
        let mut queue = VecDeque::new();
        queue.push_back(source);

        'bfs: while let Some(u) = queue.pop_front() {
            for v in 0..n {
                if parent[v] == usize::MAX && *cap.get(&(u, v)).unwrap_or(&0.0) > 1e-12 {
                    parent[v] = u;
                    if v == sink {
                        break 'bfs;
                    }
                    queue.push_back(v);
                }
            }
        }

        if parent[sink] == usize::MAX {
            break; // no augmenting path found
        }

        // Find bottleneck capacity along the path.
        let mut bottleneck = f64::INFINITY;
        let mut v = sink;
        while v != source {
            let u = parent[v];
            let c = *cap.get(&(u, v)).unwrap_or(&0.0);
            if c < bottleneck {
                bottleneck = c;
            }
            v = u;
        }

        // Update residual capacities.
        let mut v = sink;
        while v != source {
            let u = parent[v];
            *cap.entry((u, v)).or_insert(0.0) -= bottleneck;
            *cap.entry((v, u)).or_insert(0.0) += bottleneck;
            v = u;
        }

        max_flow += bottleneck;
    }

    let mut row = HashMap::new();
    row.insert(
        "source".to_string(),
        Value::String(ulid_encode(snap.node_ids[source].0)),
    );
    row.insert(
        "sink".to_string(),
        Value::String(ulid_encode(snap.node_ids[sink].0)),
    );
    row.insert("maxFlow".to_string(), Value::Float(max_flow));
    Ok(vec![row])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::types::{ulid_encode, Edge, Node, NodeId, Value};

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

    /// Insert a directed edge from `from` to `to` with `capacity` property set to `cap`.
    fn make_cap_edge(g: &mut Graph, from: NodeId, to: NodeId, cap: f64) {
        let id = g.alloc_edge_id();
        g.apply_insert_edge(Edge {
            id,
            from_node: from,
            to_node: to,
            label: "E".into(),
            properties: [("capacity".to_string(), Value::Float(cap))].into_iter().collect(),
            directed: true,
        });
    }

    #[test]
    fn simple_two_edge_path() {
        // s --10--> m --5--> t  → max flow = 5 (bottleneck)
        let mut g = Graph::new();
        let s = make_node(&mut g);
        let m = make_node(&mut g);
        let t = make_node(&mut g);
        make_cap_edge(&mut g, s, m, 10.0);
        make_cap_edge(&mut g, m, t, 5.0);

        let params: HashMap<String, Value> = [
            ("source".to_string(), Value::String(ulid_encode(s.0))),
            ("sink".to_string(), Value::String(ulid_encode(t.0))),
        ]
        .into_iter()
        .collect();
        let rows = run_max_flow(&g, &params).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["maxFlow"], Value::Float(5.0));
    }

    #[test]
    fn two_parallel_paths() {
        // s --3--> a --3--> t
        // s --4--> b --4--> t  → max flow = 7
        let mut g = Graph::new();
        let s = make_node(&mut g);
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let t = make_node(&mut g);
        make_cap_edge(&mut g, s, a, 3.0);
        make_cap_edge(&mut g, a, t, 3.0);
        make_cap_edge(&mut g, s, b, 4.0);
        make_cap_edge(&mut g, b, t, 4.0);

        let params: HashMap<String, Value> = [
            ("source".to_string(), Value::String(ulid_encode(s.0))),
            ("sink".to_string(), Value::String(ulid_encode(t.0))),
        ]
        .into_iter()
        .collect();
        let rows = run_max_flow(&g, &params).unwrap();
        assert_eq!(rows[0]["maxFlow"], Value::Float(7.0));
    }

    #[test]
    fn no_path_returns_zero_flow() {
        let mut g = Graph::new();
        let s = make_node(&mut g);
        let t = make_node(&mut g);
        // No edges.

        let params: HashMap<String, Value> = [
            ("source".to_string(), Value::String(ulid_encode(s.0))),
            ("sink".to_string(), Value::String(ulid_encode(t.0))),
        ]
        .into_iter()
        .collect();
        let rows = run_max_flow(&g, &params).unwrap();
        assert_eq!(rows[0]["maxFlow"], Value::Float(0.0));
    }

    #[test]
    fn source_equals_sink_errors() {
        let mut g = Graph::new();
        let s = make_node(&mut g);

        let params: HashMap<String, Value> = [
            ("source".to_string(), Value::String(ulid_encode(s.0))),
            ("sink".to_string(), Value::String(ulid_encode(s.0))),
        ]
        .into_iter()
        .collect();
        assert!(run_max_flow(&g, &params).is_err());
    }

    #[test]
    fn classic_ford_fulkerson_graph() {
        // Graph:  s --16--> u --12--> t
        //         s --13--> v --20--> t
        //         u --4-->  v
        //         v --9-->  u
        //
        // Total s-out capacity = 16+13 = 29.
        // Total t-in  capacity = 12+20 = 32.
        // Min-cut is {s} with capacity 29.
        // Paths: s→u→t (12), s→v→t (13), s→u→v→t (4)  → max flow = 29.
        let mut g = Graph::new();
        let s = make_node(&mut g);
        let u = make_node(&mut g);
        let v = make_node(&mut g);
        let t = make_node(&mut g);
        make_cap_edge(&mut g, s, u, 16.0);
        make_cap_edge(&mut g, s, v, 13.0);
        make_cap_edge(&mut g, u, t, 12.0);
        make_cap_edge(&mut g, v, t, 20.0);
        make_cap_edge(&mut g, u, v, 4.0);
        make_cap_edge(&mut g, v, u, 9.0);

        let params: HashMap<String, Value> = [
            ("source".to_string(), Value::String(ulid_encode(s.0))),
            ("sink".to_string(), Value::String(ulid_encode(t.0))),
        ]
        .into_iter()
        .collect();
        let rows = run_max_flow(&g, &params).unwrap();
        assert_eq!(rows[0]["maxFlow"], Value::Float(29.0));
    }

    #[test]
    fn missing_capacity_prop_treated_as_zero() {
        // Edges exist but have no "capacity" property → all 0 capacity → flow = 0.
        let mut g = Graph::new();
        let s = make_node(&mut g);
        let t = make_node(&mut g);
        let id = g.alloc_edge_id();
        g.apply_insert_edge(crate::types::Edge {
            id,
            from_node: s,
            to_node: t,
            label: "E".into(),
            properties: Default::default(), // no capacity prop
            directed: true,
        });

        let params: HashMap<String, Value> = [
            ("source".to_string(), Value::String(ulid_encode(s.0))),
            ("sink".to_string(), Value::String(ulid_encode(t.0))),
        ]
        .into_iter()
        .collect();
        let rows = run_max_flow(&g, &params).unwrap();
        assert_eq!(rows[0]["maxFlow"], Value::Float(0.0));
    }
}
