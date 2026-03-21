//! Connected component algorithms: Weakly Connected Components (WCC) and
//! Strongly Connected Components (SCC).
//!
//! ## Weakly Connected Components (WCC)
//!
//! Finds groups of nodes that are reachable from each other when edge
//! direction is ignored.  Uses Union-Find (disjoint-set union) with path
//! compression and union by rank for near-linear O(N α(N)) time.
//!
//! ```gql
//! CALL wcc(minSize: 2) YIELD node, component, size
//! ```
//!
//! | param     | type | default | description                           |
//! |-----------|------|---------|---------------------------------------|
//! | `minSize` | Int  | 1       | Exclude components smaller than this  |
//!
//! **Yields:** `node` (String), `component` (String — representative ULID),
//!             `size` (Int — number of nodes in the component)
//!
//! ## Strongly Connected Components (SCC)
//!
//! Finds maximal subgraphs in which every node can reach every other node
//! following directed edges.  Uses Kosaraju's two-pass DFS algorithm.
//!
//! Kosaraju's algorithm:
//! 1. DFS on the **forward** graph; record nodes in their finish order.
//! 2. DFS on the **reverse** graph (follow `adj_in`) in reverse finish order.
//!    Each DFS tree in pass 2 is one SCC.
//!
//! ```gql
//! CALL scc(minSize: 2) YIELD node, component, size
//! ```
//!
//! Same parameters and yields as `wcc`.
//!
//! ## Complexity
//!
//! - WCC: O(N α(N) + E) time, O(N) space.
//! - SCC: O(N + E) time, O(N + E) space (two DFS traversals).

use std::collections::HashMap;

use crate::graph::Graph;
use crate::types::{ulid_encode, DbError, Value};

use super::{opt_usize, Direction, GraphSnapshot, Row};

// ── WCC ───────────────────────────────────────────────────────────────────────

/// Find weakly connected components using Union-Find.
///
/// All edges are treated as undirected regardless of their stored direction.
/// Isolated nodes (no edges) each form their own singleton component.
///
/// # Parameters (from GQL `call_arg`)
///
/// - `minSize` (Int, default 1) — suppress components with fewer nodes than
///   this threshold.  Must be ≥ 1.
///
/// # Returns
///
/// One row per node that belongs to a component of size ≥ `minSize`.
/// The `component` column holds the ULID of the component's representative
/// node (the earliest-scanned node in the component).
///
/// # Complexity
///
/// O(N α(N) + E) where α is the inverse Ackermann function (effectively O(1)
/// per Union-Find operation), O(N) space.
pub fn run_wcc(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let min_size = opt_usize(params, "minSize", 1)?;
    if min_size == 0 {
        return Err(DbError::Query("parameter 'minSize' must be at least 1".into()));
    }

    // Build an undirected snapshot (edges reflected in both adj_out and adj_in).
    let snap = GraphSnapshot::build(graph, None);
    let n = snap.n;

    if n == 0 {
        return Ok(vec![]);
    }

    // ── Union-Find initialisation ─────────────────────────────────────────────

    // `parent[i]` starts as `i` (each node is its own root).
    let mut parent: Vec<usize> = (0..n).collect();
    // `rank[i]` is an upper bound on the height of the tree rooted at i;
    // used to keep trees shallow during union.
    let mut rank: Vec<usize> = vec![0; n];

    // `find` with path compression: flatten the path to the root on every call.
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]); // path compression (recursive)
        }
        parent[x]
    }

    // `union` by rank: attach the shorter tree under the taller one.
    fn union(parent: &mut Vec<usize>, rank: &mut Vec<usize>, x: usize, y: usize) {
        let rx = find(parent, x);
        let ry = find(parent, y);
        if rx == ry {
            return; // already in the same set
        }
        match rank[rx].cmp(&rank[ry]) {
            std::cmp::Ordering::Less => parent[rx] = ry,
            std::cmp::Ordering::Greater => parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                parent[ry] = rx;
                rank[rx] += 1; // rank increases only on equal-rank union
            }
        }
    }

    // ── Union all edges (treat as undirected) ─────────────────────────────────

    for i in 0..n {
        for (j, _) in snap.neighbors(i, Direction::Any) {
            union(&mut parent, &mut rank, i, j);
        }
    }

    // Finalise all roots by running `find` on every node.  After this all
    // `parent[i]` values point directly to the component root.
    for i in 0..n {
        let _ = find(&mut parent, i);
    }

    build_component_rows(&snap, &parent, min_size)
}

// ── SCC (Kosaraju) ────────────────────────────────────────────────────────────

/// Find strongly connected components using Kosaraju's two-pass algorithm.
///
/// A strongly connected component is a maximal set of nodes where every node
/// can reach every other node along directed edges.  In a DAG every node is its
/// own SCC.
///
/// ## Algorithm (Kosaraju)
///
/// **Pass 1** — DFS on the forward graph (`adj_out`).  Nodes are added to a
/// `finish_order` stack in the order they finish (i.e. after all their
/// descendants have been visited).
///
/// **Pass 2** — DFS on the **transposed** (reversed) graph (`adj_in`),
/// starting sources in reverse finish order.  Each DFS tree in pass 2 is
/// exactly one SCC.
///
/// Both passes use an iterative DFS to avoid Rust stack-overflow on deep graphs.
///
/// # Parameters (from GQL `call_arg`)
///
/// - `minSize` (Int, default 1) — suppress SCCs with fewer nodes than this.
///
/// # Complexity
///
/// O(N + E) time (two DFS traversals), O(N + E) space.
pub fn run_scc(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let min_size = opt_usize(params, "minSize", 1)?;
    if min_size == 0 {
        return Err(DbError::Query("parameter 'minSize' must be at least 1".into()));
    }

    let snap = GraphSnapshot::build(graph, None);
    let n = snap.n;

    if n == 0 {
        return Ok(vec![]);
    }

    // ── Pass 1: DFS on forward graph, record finish order ────────────────────

    let mut visited = vec![false; n];
    let mut finish_order: Vec<usize> = Vec::with_capacity(n);

    for start in 0..n {
        if !visited[start] {
            dfs_finish(&snap, start, &mut visited, &mut finish_order, Direction::Out);
        }
    }

    // ── Pass 2: DFS on reverse graph in reverse finish order → SCCs ──────────

    // `component[i]` will be the SCC ID for node i; usize::MAX = unassigned.
    let mut component: Vec<usize> = vec![usize::MAX; n];
    let mut comp_id = 0;

    // Iterating in reverse finish order guarantees that each DFS tree from pass 2
    // visits exactly the nodes of one SCC.
    for &start in finish_order.iter().rev() {
        if component[start] == usize::MAX {
            assign_component(&snap, start, comp_id, &mut component, Direction::In);
            comp_id += 1;
        }
    }

    build_component_rows(&snap, &component, min_size)
}

/// Iterative DFS that records each node's finish time in `finish`.
///
/// Uses an explicit stack of `(node, neighbor_cursor)` pairs to simulate the
/// recursive DFS call stack.  A node is added to `finish` only after all its
/// reachable unvisited descendants have been processed (post-order).
///
/// `dir` controls which adjacency list is followed (forward = `Out`,
/// reverse = `In`).
fn dfs_finish(
    snap: &GraphSnapshot,
    start: usize,
    visited: &mut Vec<bool>,
    finish: &mut Vec<usize>,
    dir: Direction,
) {
    // Stack entry: (current_node, index_into_its_neighbour_list)
    let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
    visited[start] = true;

    // Cache neighbour lists to avoid re-computing them on each stack iteration.
    let mut nbrs_cache: Vec<Vec<usize>> = vec![vec![]; snap.n];
    nbrs_cache[start] = snap.unique_neighbor_indices(start, dir);

    while let Some((node, cursor)) = stack.last_mut() {
        let node = *node;
        if *cursor < nbrs_cache[node].len() {
            let nbr = nbrs_cache[node][*cursor];
            *cursor += 1;
            if !visited[nbr] {
                visited[nbr] = true;
                // Populate neighbour list for the new node before pushing.
                nbrs_cache[nbr] = snap.unique_neighbor_indices(nbr, dir);
                stack.push((nbr, 0));
            }
        } else {
            // All neighbours of `node` have been explored → record finish time.
            finish.push(node);
            stack.pop();
        }
    }
}

/// Iterative DFS on the transposed graph that labels all nodes in the same
/// reachable subtree with `comp_id`.
///
/// Called in pass 2 of Kosaraju.  Uses `Direction::In` to follow edges in
/// reverse (transposed graph).
fn assign_component(
    snap: &GraphSnapshot,
    start: usize,
    comp_id: usize,
    component: &mut Vec<usize>,
    dir: Direction,
) {
    let mut stack = vec![start];
    component[start] = comp_id;
    while let Some(node) = stack.pop() {
        for nbr in snap.unique_neighbor_indices(node, dir) {
            if component[nbr] == usize::MAX {
                component[nbr] = comp_id;
                stack.push(nbr);
            }
        }
    }
}

// ── Shared row builder ────────────────────────────────────────────────────────

/// Convert a flat `component[node] = comp_id` array into output rows.
///
/// The **representative** of each component is the first node (by scan order)
/// assigned to that component ID — its ULID appears in the `component` column.
/// Components with fewer than `min_size` members are omitted entirely.
fn build_component_rows(
    snap: &GraphSnapshot,
    component: &[usize],
    min_size: usize,
) -> Result<Vec<Row>, DbError> {
    let n = snap.n;

    // Count how many nodes belong to each component ID.
    let mut size_map: HashMap<usize, usize> = HashMap::new();
    for &c in component.iter() {
        *size_map.entry(c).or_insert(0) += 1;
    }

    // Map component ID → representative node index (first occurrence in scan order).
    let mut rep_map: HashMap<usize, usize> = HashMap::new();
    for (i, &c) in component.iter().enumerate() {
        rep_map.entry(c).or_insert(i);
    }

    let mut rows = Vec::new();
    for i in 0..n {
        let c = component[i];
        let sz = size_map[&c];

        // Apply the minSize filter.
        if sz < min_size {
            continue;
        }

        let rep_idx = rep_map[&c];
        let mut row = HashMap::new();
        row.insert(
            "node".to_string(),
            Value::String(ulid_encode(snap.node_ids[i].0)),
        );
        row.insert(
            "component".to_string(),
            Value::String(ulid_encode(snap.node_ids[rep_idx].0)),
        );
        row.insert("size".to_string(), Value::Int(sz as i64));
        rows.push(row);
    }
    Ok(rows)
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

    /// Insert a directed edge from `from` to `to` with no properties.
    fn make_directed_edge(g: &mut Graph, from: NodeId, to: NodeId) {
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

    // ── WCC tests ──

    #[test]
    fn wcc_single_component() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        make_directed_edge(&mut g, a, b);
        make_directed_edge(&mut g, b, c);

        let params = HashMap::new();
        let rows = run_wcc(&g, &params).unwrap();
        assert_eq!(rows.len(), 3);
        // All in same component.
        let comps: Vec<&Value> = rows.iter().map(|r| &r["component"]).collect();
        assert!(comps.iter().all(|c| *c == comps[0]));
    }

    #[test]
    fn wcc_two_disconnected_components() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        let d = make_node(&mut g);
        make_directed_edge(&mut g, a, b); // component 1
        make_directed_edge(&mut g, c, d); // component 2

        let params = HashMap::new();
        let rows = run_wcc(&g, &params).unwrap();
        let comp_set: std::collections::HashSet<String> = rows
            .iter()
            .map(|r| match &r["component"] {
                Value::String(s) => s.clone(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(comp_set.len(), 2);
    }

    #[test]
    fn wcc_min_size_filters_small_components() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let lone = make_node(&mut g); // isolated
        make_directed_edge(&mut g, a, b);

        let params: HashMap<String, Value> =
            [("minSize".to_string(), Value::Int(2))].into_iter().collect();
        let rows = run_wcc(&g, &params).unwrap();
        // lone node is in a size-1 component → filtered out
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn wcc_empty_graph() {
        let g = Graph::new();
        let rows = run_wcc(&g, &HashMap::new()).unwrap();
        assert!(rows.is_empty());
    }

    // ── SCC tests ──

    #[test]
    fn scc_cycle_is_single_component() {
        // a → b → c → a  (strongly connected)
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        make_directed_edge(&mut g, a, b);
        make_directed_edge(&mut g, b, c);
        make_directed_edge(&mut g, c, a);

        let params = HashMap::new();
        let rows = run_scc(&g, &params).unwrap();
        assert_eq!(rows.len(), 3);
        let comps: Vec<_> = rows.iter().map(|r| r["component"].clone()).collect();
        assert!(comps.iter().all(|c| *c == comps[0]));
    }

    #[test]
    fn scc_dag_all_singletons() {
        // a → b → c  (no cycles → each node is its own SCC)
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        make_directed_edge(&mut g, a, b);
        make_directed_edge(&mut g, b, c);

        let params = HashMap::new();
        let rows = run_scc(&g, &params).unwrap();
        assert_eq!(rows.len(), 3);
        let comp_set: std::collections::HashSet<String> = rows
            .iter()
            .map(|r| match &r["component"] {
                Value::String(s) => s.clone(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(comp_set.len(), 3); // three different SCCs
    }

    #[test]
    fn scc_min_size_two_filters_singletons() {
        // a → b → c → a  plus isolated node d
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        let _d = make_node(&mut g);
        make_directed_edge(&mut g, a, b);
        make_directed_edge(&mut g, b, c);
        make_directed_edge(&mut g, c, a);

        let params: HashMap<String, Value> =
            [("minSize".to_string(), Value::Int(2))].into_iter().collect();
        let rows = run_scc(&g, &params).unwrap();
        assert_eq!(rows.len(), 3); // only {a,b,c} passes min_size=2
    }

    #[test]
    fn wcc_min_size_zero_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> =
            [("minSize".to_string(), Value::Int(0))].into_iter().collect();
        assert!(run_wcc(&g, &params).is_err());
    }
}
