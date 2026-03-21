//! Louvain and Leiden modularity-based community detection.
//!
//! ## Louvain
//! ```gql
//! CALL louvain(resolution: 1.0, maxPasses: 10, maxLevels: 10, weight: "weight")
//! YIELD node, community
//! ```
//!
//! ## Leiden
//! ```gql
//! CALL leiden(resolution: 1.0, maxPasses: 10, maxLevels: 10, theta: 0.01, weight: "weight")
//! YIELD node, community
//! ```
//!
//! | param        | type   | default | description                                        |
//! |--------------|--------|---------|----------------------------------------------------|
//! | `resolution` | Float  | 1.0     | Resolution γ — higher values → smaller communities |
//! | `maxPasses`  | Int    | 10      | Local-move iterations per aggregation level        |
//! | `maxLevels`  | Int    | 10      | Maximum aggregation levels (multi-resolution)      |
//! | `weight`     | String | `""`    | Edge property for weight; empty = unweighted (1.0) |
//! | `theta`      | Float  | 0.01    | Leiden only: refinement merge threshold ∈ [0, 1]   |
//!
//! **Yields:** `node` (String), `community` (String — representative node ULID)
//!
//! Both algorithms optimise the Louvain modularity:
//!   Q = Σ_C [L_C/m − γ·(d_C/2m)²]
//! where L_C = internal edge weight, m = total edge weight, d_C = sum of degrees.
//!
//! Leiden adds a refinement phase after local moving that guarantees every
//! community is internally well-connected (controlled by `theta`).

use std::collections::HashMap;

use crate::graph::Graph;
use crate::types::{ulid_encode, DbError, Value};

use super::{opt_f64, opt_str, opt_usize, GraphSnapshot, Row};

// ── Internal adjacency representation ────────────────────────────────────────

type Adj = Vec<Vec<(usize, f64)>>;

/// Build a symmetrised undirected weighted adjacency list from the raw graph.
/// Directed edges are treated as undirected (both directions added).
/// Self-loops are ignored. Parallel edges are summed.
fn build_undirected_adj(graph: &Graph, snap: &GraphSnapshot, weight_prop: &str) -> (Adj, Vec<f64>, f64) {
    let n = snap.n;
    let mut adj_map: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];

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
        let w: f64 = if weight_prop.is_empty() {
            1.0
        } else {
            edge.properties
                .get(weight_prop)
                .and_then(|v| match v {
                    Value::Float(f) => Some(*f),
                    Value::Int(i) => Some(*i as f64),
                    _ => None,
                })
                .unwrap_or(1.0)
        };
        if w <= 0.0 {
            continue; // skip non-positive weights
        }
        *adj_map[fi].entry(ti).or_insert(0.0) += w;
        *adj_map[ti].entry(fi).or_insert(0.0) += w;
    }

    let adj: Adj = adj_map.into_iter().map(|m| m.into_iter().collect()).collect();
    let degree: Vec<f64> = adj.iter().map(|nbrs| nbrs.iter().map(|&(_, w)| w).sum()).collect();
    let m: f64 = degree.iter().sum::<f64>() / 2.0;
    (adj, degree, m)
}

/// Compact community IDs to 0..k.  Returns (renumbered, k).
fn renumber(community: &[usize]) -> (Vec<usize>, usize) {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut k = 0usize;
    let out: Vec<usize> = community
        .iter()
        .map(|&c| {
            *map.entry(c).or_insert_with(|| {
                let id = k;
                k += 1;
                id
            })
        })
        .collect();
    (out, k)
}

/// Build an aggregated super-node graph.
/// `community[i]` ∈ 0..k maps current super-node i to its new community.
/// New degree of super-node C = sum of degrees of constituent nodes
/// (includes intra-community contributions, preserving total edge weight).
fn aggregate(adj: &Adj, degree: &[f64], community: &[usize], k: usize) -> (Adj, Vec<f64>, f64) {
    let n = adj.len();
    let mut new_adj_map: Vec<HashMap<usize, f64>> = vec![HashMap::new(); k];
    let mut new_degree = vec![0.0f64; k];

    for i in 0..n {
        let ci = community[i];
        new_degree[ci] += degree[i]; // preserves total weight including self-loops
        for &(j, w) in &adj[i] {
            let cj = community[j];
            if ci != cj {
                *new_adj_map[ci].entry(cj).or_insert(0.0) += w;
            }
        }
    }

    let new_adj: Adj = new_adj_map.into_iter().map(|m| m.into_iter().collect()).collect();
    // m is preserved: Σ new_degree / 2 = Σ degree / 2 = m
    let new_m = new_degree.iter().sum::<f64>() / 2.0;
    (new_adj, new_degree, new_m)
}

// ── Louvain local-move phase ──────────────────────────────────────────────────

/// Run the asynchronous Louvain local-move phase.
///
/// Each node greedily moves to the neighbouring community that maximises:
///   ΔQ(i → C) = k_{i,C} − γ · k_i · Σ_C / (2m)
///
/// Returns `true` if any node changed community during the run.
fn local_move(
    adj: &Adj,
    community: &mut Vec<usize>,
    degree: &[f64],
    m: f64,
    resolution: f64,
    max_passes: usize,
) -> bool {
    if m == 0.0 {
        return false; // no edges → nothing to do
    }
    let n = adj.len();
    // comm_total[c] = sum of degrees of all nodes currently in community c.
    // Initially community[i] = i (singleton), so comm_total[i] = degree[i].
    let mut comm_total: Vec<f64> = degree.to_vec();
    let two_m = 2.0 * m;
    let mut any_improved = false;

    for _ in 0..max_passes {
        let mut pass_improved = false;
        for i in 0..n {
            let c_old = community[i];
            let ki = degree[i];

            // Remove i from its current community.
            comm_total[c_old] -= ki;

            // Sum edge weights from i to each neighbouring community.
            let mut k_to: HashMap<usize, f64> = HashMap::new();
            for &(j, w) in &adj[i] {
                *k_to.entry(community[j]).or_insert(0.0) += w;
            }

            // Gain of staying in c_old (now without i).
            let gain_old = k_to.get(&c_old).copied().unwrap_or(0.0)
                - resolution * ki * comm_total[c_old] / two_m;

            let mut best_c = c_old;
            let mut best_gain = gain_old;

            for (&c, &k_ic) in &k_to {
                if c == c_old {
                    continue;
                }
                let gain = k_ic - resolution * ki * comm_total[c] / two_m;
                if gain > best_gain {
                    best_gain = gain;
                    best_c = c;
                }
            }

            community[i] = best_c;
            comm_total[best_c] += ki;
            if best_c != c_old {
                pass_improved = true;
                any_improved = true;
            }
        }
        if !pass_improved {
            break;
        }
    }
    any_improved
}

// ── Leiden refinement phase ───────────────────────────────────────────────────

/// Within each parent community, greedily sub-partition starting from singletons.
///
/// A node i (in parent P) merges into sub-community C (also in P) only when:
///   1. edge weight to C ≥ theta · degree(i)   (well-connectedness guard)
///   2. ΔQ > 0                                  (modularity improvement)
///
/// theta = 0 degenerates to plain Louvain local move within each parent.
fn leiden_refine(
    adj: &Adj,
    parent: &[usize],
    degree: &[f64],
    m: f64,
    resolution: f64,
    theta: f64,
) -> Vec<usize> {
    let n = adj.len();
    // Start from singleton partition (each node in its own sub-community).
    let mut refined: Vec<usize> = (0..n).collect();
    // comm_total tracks sum-of-degrees within each sub-community.
    // Because we only merge within the same parent, comm_total[c] only
    // accumulates nodes from the same parent community.
    let mut comm_total: Vec<f64> = degree.to_vec();
    let two_m = 2.0 * m;

    for i in 0..n {
        let pi = parent[i];
        let ki = degree[i];
        let ci = refined[i];

        // Remove i from its singleton sub-community.
        comm_total[ci] -= ki;

        // Edges from i to neighbours in the same parent community.
        let mut k_to: HashMap<usize, f64> = HashMap::new();
        for &(j, w) in &adj[i] {
            if parent[j] == pi {
                *k_to.entry(refined[j]).or_insert(0.0) += w;
            }
        }

        let mut best_c = ci; // default: stay isolated
        let mut best_gain = 0.0f64;

        for (&c, &k_ic) in &k_to {
            if c == ci {
                continue;
            }
            // Well-connectedness: edge weight to C must be at least theta * ki.
            if k_ic < theta * ki {
                continue;
            }
            let gain = k_ic - resolution * ki * comm_total[c] / two_m;
            if gain > best_gain {
                best_gain = gain;
                best_c = c;
            }
        }

        refined[i] = best_c;
        comm_total[best_c] += ki;
    }

    refined
}

// ── Shared row builder ────────────────────────────────────────────────────────

/// Convert a flat `membership` array back into GQL result rows.
///
/// Each original node `i` maps to its community representative — the node with
/// the lowest original index that shares the same community label.  The
/// representative's ULID is used as the stable `community` identifier so that
/// the column value is human-readable and consistent across runs on the same
/// graph.
///
/// # Returns
/// One row per node: `{ node: <ulid>, community: <rep-ulid> }`.
fn build_rows(snap: &GraphSnapshot, membership: &[usize]) -> Result<Vec<Row>, DbError> {
    let n = snap.n;
    // Representative of each community = lowest original-index node in it.
    let mut rep: HashMap<usize, usize> = HashMap::new();
    for (i, &c) in membership.iter().enumerate() {
        rep.entry(c).or_insert(i);
    }
    Ok((0..n)
        .map(|i| {
            let rep_idx = rep[&membership[i]];
            let mut row = HashMap::new();
            row.insert("node".to_string(), Value::String(ulid_encode(snap.node_ids[i].0)));
            row.insert(
                "community".to_string(),
                Value::String(ulid_encode(snap.node_ids[rep_idx].0)),
            );
            row
        })
        .collect())
}

// ── Louvain entry point ───────────────────────────────────────────────────────

/// Entry point called by `dispatch_call` for `CALL louvain(...)`.
///
/// Runs the multi-level Louvain algorithm on the graph:
/// 1. Build an undirected weighted adjacency list.
/// 2. Perform the *local-move* phase — each node greedily joins the neighbouring
///    community that maximises the modularity gain ΔQ.
/// 3. Aggregate merged communities into super-nodes.
/// 4. Repeat steps 2–3 up to `maxLevels` times, or until no improvement.
///
/// Results are one row per original node with its community label.  The
/// community identifier is the ULID of the lowest-index node in the community.
///
/// # Errors
/// Returns [`DbError::Query`] if any parameter value is out of range.
pub fn run_louvain(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let resolution = opt_f64(params, "resolution", 1.0)?;
    if resolution <= 0.0 {
        return Err(DbError::Query("'resolution' must be positive".into()));
    }
    let max_passes = opt_usize(params, "maxPasses", 10)?;
    if max_passes == 0 {
        return Err(DbError::Query("'maxPasses' must be at least 1".into()));
    }
    let max_levels = opt_usize(params, "maxLevels", 10)?;
    if max_levels == 0 {
        return Err(DbError::Query("'maxLevels' must be at least 1".into()));
    }
    let weight_prop = opt_str(params, "weight", "")?.to_string();

    let snap = GraphSnapshot::build(graph, None);
    let n = snap.n;
    if n == 0 {
        return Ok(vec![]);
    }

    let (mut adj, mut degree, mut m) = build_undirected_adj(graph, &snap, &weight_prop);

    // membership[original_idx] = super-node index in the current level graph.
    let mut membership: Vec<usize> = (0..n).collect();
    let mut cur_n = n;

    for _ in 0..max_levels {
        let mut community: Vec<usize> = (0..cur_n).collect();
        let improved = local_move(&adj, &mut community, &degree, m, resolution, max_passes);
        if !improved {
            break;
        }

        let (renumbered, k) = renumber(&community);
        // Update: original node → new super-node in 0..k.
        for idx in membership.iter_mut() {
            *idx = renumbered[*idx];
        }

        if k >= cur_n {
            break; // no merging occurred (e.g. swaps only)
        }

        let (new_adj, new_degree, new_m) = aggregate(&adj, &degree, &renumbered, k);
        adj = new_adj;
        degree = new_degree;
        m = new_m;
        cur_n = k;
    }

    build_rows(&snap, &membership)
}

// ── Leiden entry point ────────────────────────────────────────────────────────

/// Entry point called by `dispatch_call` for `CALL leiden(...)`.
///
/// Runs the Leiden algorithm — an improved variant of Louvain that guarantees
/// every community is internally well-connected.  Each level consists of two
/// phases:
/// 1. **Local move** — identical to Louvain; greedily move nodes to maximise Q.
/// 2. **Refinement** — within each parent community, sub-partition starting from
///    singletons.  A node only merges into a sub-community if its edge weight
///    to that community exceeds `theta * degree(node)` (the well-connectedness
///    guard).
///
/// Higher `theta` → stricter well-connectedness requirement → more (smaller)
/// refined sub-communities per Louvain community.  `theta = 0` degenerates to
/// plain Louvain local move within each parent.
///
/// # Errors
/// Returns [`DbError::Query`] if any parameter value is out of range.
pub fn run_leiden(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let resolution = opt_f64(params, "resolution", 1.0)?;
    if resolution <= 0.0 {
        return Err(DbError::Query("'resolution' must be positive".into()));
    }
    let max_passes = opt_usize(params, "maxPasses", 10)?;
    if max_passes == 0 {
        return Err(DbError::Query("'maxPasses' must be at least 1".into()));
    }
    let max_levels = opt_usize(params, "maxLevels", 10)?;
    if max_levels == 0 {
        return Err(DbError::Query("'maxLevels' must be at least 1".into()));
    }
    let theta = opt_f64(params, "theta", 0.01)?;
    if !(0.0..=1.0).contains(&theta) {
        return Err(DbError::Query("'theta' must be in [0.0, 1.0]".into()));
    }
    let weight_prop = opt_str(params, "weight", "")?.to_string();

    let snap = GraphSnapshot::build(graph, None);
    let n = snap.n;
    if n == 0 {
        return Ok(vec![]);
    }

    let (mut adj, mut degree, mut m) = build_undirected_adj(graph, &snap, &weight_prop);
    let mut membership: Vec<usize> = (0..n).collect();
    let mut cur_n = n;

    for _ in 0..max_levels {
        // Phase 1: local move.
        let mut community: Vec<usize> = (0..cur_n).collect();
        local_move(&adj, &mut community, &degree, m, resolution, max_passes);

        // Phase 2: refinement within each parent community.
        let refined = leiden_refine(&adj, &community, &degree, m, resolution, theta);
        let (renumbered, k) = renumber(&refined);

        for idx in membership.iter_mut() {
            *idx = renumbered[*idx];
        }

        if k >= cur_n {
            break; // neither local move nor refinement merged anything
        }

        let (new_adj, new_degree, new_m) = aggregate(&adj, &degree, &renumbered, k);
        adj = new_adj;
        degree = new_degree;
        m = new_m;
        cur_n = k;
    }

    build_rows(&snap, &membership)
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

    /// Insert an undirected, unweighted edge between `from` and `to`.
    fn undirected_edge(g: &mut Graph, from: NodeId, to: NodeId) {
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

    /// Insert an undirected edge with a `"weight"` property set to `w`.
    fn weighted_edge(g: &mut Graph, from: NodeId, to: NodeId, w: f64) {
        let id = g.alloc_edge_id();
        g.apply_insert_edge(Edge {
            id,
            from_node: from,
            to_node: to,
            label: "E".into(),
            properties: [("weight".to_string(), Value::Float(w))].into_iter().collect(),
            directed: false,
        });
    }

    /// Two disconnected triangles + their node IDs.
    fn two_cliques() -> (Graph, [NodeId; 6]) {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        let d = make_node(&mut g);
        let e = make_node(&mut g);
        let f = make_node(&mut g);
        undirected_edge(&mut g, a, b);
        undirected_edge(&mut g, b, c);
        undirected_edge(&mut g, a, c);
        undirected_edge(&mut g, d, e);
        undirected_edge(&mut g, e, f);
        undirected_edge(&mut g, d, f);
        (g, [a, b, c, d, e, f])
    }

    /// Look up the community string for a node by ULID.
    fn community_of(rows: &[Row], id: NodeId) -> String {
        let ulid = ulid_encode(id.0);
        rows.iter()
            .find(|r| r["node"] == Value::String(ulid.clone()))
            .map(|r| match &r["community"] {
                Value::String(s) => s.clone(),
                other => panic!("expected String, got {other:?}"),
            })
            .expect("node not found in rows")
    }

    // ── Louvain ──────────────────────────────────────────────────────────────

    #[test]
    fn louvain_two_disconnected_cliques() {
        let (g, ids) = two_cliques();
        let rows = run_louvain(&g, &HashMap::new()).unwrap();
        assert_eq!(rows.len(), 6);
        let ca = community_of(&rows, ids[0]);
        let cb = community_of(&rows, ids[1]);
        let cc = community_of(&rows, ids[2]);
        let cd = community_of(&rows, ids[3]);
        let ce = community_of(&rows, ids[4]);
        let cf = community_of(&rows, ids[5]);
        assert_eq!(ca, cb);
        assert_eq!(cb, cc);
        assert_eq!(cd, ce);
        assert_eq!(ce, cf);
        assert_ne!(ca, cd);
    }

    #[test]
    fn louvain_fully_connected_triangle_one_community() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        undirected_edge(&mut g, a, b);
        undirected_edge(&mut g, b, c);
        undirected_edge(&mut g, a, c);
        let rows = run_louvain(&g, &HashMap::new()).unwrap();
        assert_eq!(rows.len(), 3);
        let comms: Vec<_> = rows.iter().map(|r| r["community"].clone()).collect();
        assert!(comms.iter().all(|c| *c == comms[0]));
    }

    #[test]
    fn louvain_isolated_nodes_each_own_community() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        let rows = run_louvain(&g, &HashMap::new()).unwrap();
        assert_eq!(rows.len(), 3);
        let ca = community_of(&rows, a);
        let cb = community_of(&rows, b);
        let cc = community_of(&rows, c);
        // All distinct.
        assert_ne!(ca, cb);
        assert_ne!(cb, cc);
        assert_ne!(ca, cc);
    }

    #[test]
    fn louvain_empty_graph() {
        let g = Graph::new();
        assert!(run_louvain(&g, &HashMap::new()).unwrap().is_empty());
    }

    #[test]
    fn louvain_weighted_two_cliques_with_bridge() {
        // Two triangles (w=10) connected by a weak bridge (w=0.1).
        // Louvain should still find 2 communities.
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        let d = make_node(&mut g);
        let e = make_node(&mut g);
        let f = make_node(&mut g);
        weighted_edge(&mut g, a, b, 10.0);
        weighted_edge(&mut g, b, c, 10.0);
        weighted_edge(&mut g, a, c, 10.0);
        weighted_edge(&mut g, d, e, 10.0);
        weighted_edge(&mut g, e, f, 10.0);
        weighted_edge(&mut g, d, f, 10.0);
        weighted_edge(&mut g, c, d, 0.1); // weak bridge

        let params: HashMap<String, Value> =
            [("weight".to_string(), Value::String("weight".into()))].into_iter().collect();
        let rows = run_louvain(&g, &params).unwrap();
        assert_eq!(rows.len(), 6);
        let comm_set: std::collections::HashSet<String> = rows
            .iter()
            .map(|r| match &r["community"] {
                Value::String(s) => s.clone(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert!(comm_set.len() >= 2);
    }

    #[test]
    fn louvain_high_resolution_more_communities() {
        // High γ penalises large communities → more communities than low γ.
        let (g, _) = two_cliques();
        let lo: HashMap<String, Value> =
            [("resolution".to_string(), Value::Float(0.1))].into_iter().collect();
        let hi: HashMap<String, Value> =
            [("resolution".to_string(), Value::Float(5.0))].into_iter().collect();
        let count = |params| -> usize {
            let rows: Vec<Row> = run_louvain(&g, params).unwrap();
            rows.iter()
                .map(|r| match &r["community"] {
                    Value::String(s) => s.clone(),
                    _ => panic!(),
                })
                .collect::<std::collections::HashSet<String>>()
                .len()
        };
        assert!(count(&hi) >= count(&lo));
    }

    #[test]
    fn louvain_invalid_resolution_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> =
            [("resolution".to_string(), Value::Float(-1.0))].into_iter().collect();
        assert!(run_louvain(&g, &params).is_err());
    }

    #[test]
    fn louvain_zero_max_passes_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> =
            [("maxPasses".to_string(), Value::Int(0))].into_iter().collect();
        assert!(run_louvain(&g, &params).is_err());
    }

    #[test]
    fn louvain_zero_max_levels_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> =
            [("maxLevels".to_string(), Value::Int(0))].into_iter().collect();
        assert!(run_louvain(&g, &params).is_err());
    }

    // ── Leiden ───────────────────────────────────────────────────────────────

    #[test]
    fn leiden_two_disconnected_cliques() {
        let (g, ids) = two_cliques();
        let rows = run_leiden(&g, &HashMap::new()).unwrap();
        assert_eq!(rows.len(), 6);
        let ca = community_of(&rows, ids[0]);
        let cb = community_of(&rows, ids[1]);
        let cc = community_of(&rows, ids[2]);
        let cd = community_of(&rows, ids[3]);
        let ce = community_of(&rows, ids[4]);
        let cf = community_of(&rows, ids[5]);
        assert_eq!(ca, cb);
        assert_eq!(cb, cc);
        assert_eq!(cd, ce);
        assert_eq!(ce, cf);
        assert_ne!(ca, cd);
    }

    #[test]
    fn leiden_fully_connected_triangle_one_community() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        undirected_edge(&mut g, a, b);
        undirected_edge(&mut g, b, c);
        undirected_edge(&mut g, a, c);
        let rows = run_leiden(&g, &HashMap::new()).unwrap();
        let comms: Vec<_> = rows.iter().map(|r| r["community"].clone()).collect();
        assert!(comms.iter().all(|c| *c == comms[0]));
    }

    #[test]
    fn leiden_empty_graph() {
        let g = Graph::new();
        assert!(run_leiden(&g, &HashMap::new()).unwrap().is_empty());
    }

    #[test]
    fn leiden_theta_out_of_range_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> =
            [("theta".to_string(), Value::Float(1.5))].into_iter().collect();
        assert!(run_leiden(&g, &params).is_err());
    }

    #[test]
    fn leiden_theta_zero_matches_louvain_behaviour() {
        // theta=0 disables well-connectedness filter → same as Louvain on simple cases.
        let (g, ids) = two_cliques();
        let params: HashMap<String, Value> =
            [("theta".to_string(), Value::Float(0.0))].into_iter().collect();
        let rows = run_leiden(&g, &params).unwrap();
        let ca = community_of(&rows, ids[0]);
        let cd = community_of(&rows, ids[3]);
        assert_ne!(ca, cd); // still finds 2 separate communities
    }

    #[test]
    fn leiden_high_resolution_more_communities() {
        let (g, _) = two_cliques();
        let lo: HashMap<String, Value> =
            [("resolution".to_string(), Value::Float(0.1))].into_iter().collect();
        let hi: HashMap<String, Value> =
            [("resolution".to_string(), Value::Float(5.0))].into_iter().collect();
        let count = |params| -> usize {
            let rows: Vec<Row> = run_leiden(&g, params).unwrap();
            rows.iter()
                .map(|r| match &r["community"] {
                    Value::String(s) => s.clone(),
                    _ => panic!(),
                })
                .collect::<std::collections::HashSet<String>>()
                .len()
        };
        assert!(count(&hi) >= count(&lo));
    }

    #[test]
    fn leiden_weighted_two_cliques_with_bridge() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        let d = make_node(&mut g);
        let e = make_node(&mut g);
        let f = make_node(&mut g);
        weighted_edge(&mut g, a, b, 10.0);
        weighted_edge(&mut g, b, c, 10.0);
        weighted_edge(&mut g, a, c, 10.0);
        weighted_edge(&mut g, d, e, 10.0);
        weighted_edge(&mut g, e, f, 10.0);
        weighted_edge(&mut g, d, f, 10.0);
        weighted_edge(&mut g, c, d, 0.1);

        let params: HashMap<String, Value> =
            [("weight".to_string(), Value::String("weight".into()))].into_iter().collect();
        let rows = run_leiden(&g, &params).unwrap();
        let comm_set: std::collections::HashSet<String> = rows
            .iter()
            .map(|r| match &r["community"] {
                Value::String(s) => s.clone(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert!(comm_set.len() >= 2);
    }

    #[test]
    fn leiden_isolated_nodes_each_own_community() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        let rows = run_leiden(&g, &HashMap::new()).unwrap();
        let ca = community_of(&rows, a);
        let cb = community_of(&rows, b);
        let cc = community_of(&rows, c);
        assert_ne!(ca, cb);
        assert_ne!(cb, cc);
        assert_ne!(ca, cc);
    }

    // ── Shared edge cases ─────────────────────────────────────────────────────

    #[test]
    fn single_node_graph() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        for runner in [run_louvain, run_leiden] {
            let rows = runner(&g, &HashMap::new()).unwrap();
            assert_eq!(rows.len(), 1);
            // The single node is its own community representative.
            let ca = community_of(&rows, a);
            assert_eq!(ca, ulid_encode(a.0));
        }
    }

    #[test]
    fn two_node_edge() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        undirected_edge(&mut g, a, b);
        for runner in [run_louvain, run_leiden] {
            let rows = runner(&g, &HashMap::new()).unwrap();
            assert_eq!(rows.len(), 2);
            let ca = community_of(&rows, a);
            let cb = community_of(&rows, b);
            assert_eq!(ca, cb); // merged into one community
        }
    }
}
