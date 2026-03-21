//! Node centrality algorithms: PageRank, Betweenness, Closeness, and Degree.
//!
//! Centrality measures quantify the relative importance of nodes in a graph.
//! Each algorithm assigns a numeric `score` to every node.
//!
//! ## PageRank
//!
//! Models a random walk with teleportation.  A walker at node u follows a
//! random outgoing edge with probability `damping`, or jumps to a uniformly
//! random node with probability `1 - damping`.  Scores converge via power
//! iteration.  Dangling nodes (no out-edges) redistribute their mass uniformly.
//!
//! ```gql
//! CALL pageRank(damping: 0.85, iterations: 20, tolerance: 1e-6, weight: "w", normalize: true)
//! YIELD node, score
//! ```
//!
//! | param        | type   | default | description                                   |
//! |--------------|--------|---------|-----------------------------------------------|
//! | `damping`    | Float  | 0.85    | Damping factor d ∈ (0, 1)                     |
//! | `iterations` | Int    | 20      | Maximum power-iteration rounds                |
//! | `tolerance`  | Float  | 1e-6    | Convergence threshold (max rank delta)        |
//! | `weight`     | String | none    | Edge property for transition weights          |
//! | `normalize`  | Bool   | true    | Normalize scores to sum to 1                  |
//!
//! ## Betweenness Centrality (Brandes)
//!
//! Counts the fraction of shortest paths between all pairs (s, t) that pass
//! through each node v.  Uses Brandes' O(NE) accumulation algorithm.
//!
//! ```gql
//! CALL betweennessCentrality(normalized: true, directed: true, weight: "w", sampleSize: 0)
//! YIELD node, score
//! ```
//!
//! | param        | type   | default | description                                          |
//! |--------------|--------|---------|------------------------------------------------------|
//! | `normalized` | Bool   | true    | Divide by (N-1)(N-2) [directed] or /2 [undirected]  |
//! | `directed`   | Bool   | true    | Use edge direction; false = treat as undirected      |
//! | `weight`     | String | none    | Edge property for weighted shortest paths            |
//! | `sampleSize` | Int    | 0       | 0 = exact (all sources); >0 = approximate            |
//!
//! ## Closeness Centrality
//!
//! The reciprocal of the average shortest-path distance from a node to all
//! reachable nodes.  The Wasserman-Faust correction handles disconnected graphs.
//!
//! ```gql
//! CALL closenessCentrality(normalized: true, direction: "out", wfImproved: true)
//! YIELD node, score
//! ```
//!
//! | param        | type   | default  | description                                          |
//! |--------------|--------|----------|------------------------------------------------------|
//! | `normalized` | Bool   | true     | Normalize by (N-1)                                   |
//! | `direction`  | String | `"out"`  | `"out"`, `"in"`, or `"any"`                          |
//! | `wfImproved` | Bool   | true     | Wasserman-Faust correction for disconnected graphs   |
//!
//! ## Degree Centrality
//!
//! Simply counts the number of incident edges for each node.
//!
//! ```gql
//! CALL degreeCentrality(direction: "total", normalized: true)
//! YIELD node, in_degree, out_degree, degree, score
//! ```
//!
//! | param        | type   | default   | description                                  |
//! |--------------|--------|-----------|----------------------------------------------|
//! | `direction`  | String | `"total"` | `"in"`, `"out"`, or `"total"` for score      |
//! | `normalized` | Bool   | true      | Divide by (N-1)                              |

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Reverse;

use crate::graph::Graph;
use crate::types::{ulid_encode, DbError, Value};

use super::{opt_bool, opt_direction, opt_f64, opt_usize, opt_weight_prop, Direction,
            GraphSnapshot, Row};

// ── PageRank ──────────────────────────────────────────────────────────────────

/// Compute PageRank scores via power iteration.
///
/// The PageRank of node v is defined as:
///
/// ```text
/// PR(v) = (1-d)/N + d * [ Σ_{u→v} PR(u) * w(u,v) / out_weight(u) + dangling_mass/N ]
/// ```
///
/// where `d` is the damping factor, `w(u,v)` is the edge weight (or 1 if
/// unweighted), and `dangling_mass` is the total rank of nodes with no
/// outgoing edges (redistributed uniformly to avoid rank sinks).
///
/// # Parameters (from GQL `call_arg`)
///
/// - `damping` (Float, default 0.85) — must be strictly in (0, 1).
/// - `iterations` (Int, default 20) — maximum power-iteration rounds.
/// - `tolerance` (Float, default 1e-6) — stop early if max rank delta < this.
/// - `weight` (String, optional) — edge property for transition weights.
/// - `normalize` (Bool, default true) — normalize final scores to sum to 1.
///
/// # Complexity
///
/// O(iterations × E) time, O(N) space.
pub fn run_pagerank(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let damping = opt_f64(params, "damping", 0.85)?;
    if !(0.0 < damping && damping < 1.0) {
        return Err(DbError::Query(
            "parameter 'damping' must be strictly between 0 and 1".into(),
        ));
    }

    let iterations = opt_usize(params, "iterations", 20)?;
    if iterations == 0 {
        return Err(DbError::Query("parameter 'iterations' must be at least 1".into()));
    }

    let tolerance = opt_f64(params, "tolerance", 1e-6)?;
    if tolerance < 0.0 {
        return Err(DbError::Query("parameter 'tolerance' must be non-negative".into()));
    }

    let normalize = opt_bool(params, "normalize", true)?;
    let weight_prop = opt_weight_prop(params)?;
    let snap = GraphSnapshot::build(graph, weight_prop);
    let n = snap.n;

    if n == 0 {
        return Ok(vec![]);
    }

    // Pre-compute the sum of outgoing edge weights per node.
    // Used to normalise each node's rank contribution to its successors.
    let out_weight_sum: Vec<f64> = (0..n)
        .map(|i| snap.adj_out[i].iter().map(|&(_, w)| w.max(0.0)).sum())
        .collect();

    // Initialise ranks uniformly.
    let mut rank = vec![1.0f64 / n as f64; n];
    // Teleportation mass added to every node per iteration.
    let teleport = (1.0 - damping) / n as f64;

    for _ in 0..iterations {
        let mut new_rank = vec![0.0f64; n];

        // Compute the total rank held by dangling nodes (no out-edges).
        // This mass is redistributed uniformly to all nodes each iteration
        // so that the total rank stays normalised.
        let dangling_sum: f64 = rank
            .iter()
            .enumerate()
            .filter(|&(i, _)| snap.adj_out[i].is_empty())
            .map(|(_, &r)| r)
            .sum::<f64>()
            / n as f64;

        // Each node receives teleportation + its share of dangling mass.
        for v in 0..n {
            new_rank[v] += teleport + damping * dangling_sum;
        }

        // Propagate rank along each outgoing edge, weighted by the edge's
        // fraction of the source node's total outgoing weight.
        for u in 0..n {
            if snap.adj_out[u].is_empty() {
                continue; // dangling nodes already handled above
            }
            let denom = out_weight_sum[u];
            for &(v, w) in &snap.adj_out[u] {
                let contrib = rank[u] * damping * (w.max(0.0) / denom);
                new_rank[v] += contrib;
            }
        }

        // Check convergence: maximum absolute change across all nodes.
        let delta: f64 = rank
            .iter()
            .zip(new_rank.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        rank = new_rank;
        if delta < tolerance {
            break; // converged
        }
    }

    if normalize {
        let total: f64 = rank.iter().sum();
        if total > 0.0 {
            for r in &mut rank {
                *r /= total;
            }
        }
    }

    Ok(rank
        .into_iter()
        .enumerate()
        .map(|(i, score)| {
            let mut row = HashMap::new();
            row.insert(
                "node".to_string(),
                Value::String(ulid_encode(snap.node_ids[i].0)),
            );
            row.insert("score".to_string(), Value::Float(score));
            row
        })
        .collect())
}

// ── Betweenness Centrality (Brandes) ─────────────────────────────────────────

/// Compute betweenness centrality using Brandes' O(NE) algorithm.
///
/// The betweenness centrality of node v is:
///
/// ```text
/// BC(v) = Σ_{s≠v≠t} σ(s,t|v) / σ(s,t)
/// ```
///
/// where σ(s,t) is the number of shortest paths from s to t and σ(s,t|v)
/// is those that pass through v.
///
/// ## Brandes' Algorithm
///
/// For each source node s:
/// 1. Run BFS (unweighted) or Dijkstra (weighted) from s to compute
///    `dist[v]`, `sigma[v]` (number of shortest paths from s to v), and
///    `pred[v]` (predecessors of v on shortest paths).
/// 2. Back-propagate dependency scores from farthest nodes to closest:
///    `δ[v] += (σ[v] / σ[w]) * (1 + δ[w])` for each predecessor v of w.
/// 3. Accumulate `betweenness[w] += δ[w]` for w ≠ s.
///
/// # Parameters (from GQL `call_arg`)
///
/// - `normalized` (Bool, default true) — divide by (N-1)(N-2) for directed
///   graphs or by 2/(N-1)(N-2) for undirected.
/// - `directed` (Bool, default true) — whether to respect edge direction.
/// - `weight` (String, optional) — edge property for weighted shortest paths.
/// - `sampleSize` (Int, default 0) — 0 = exact; > 0 = use this many evenly-
///   spaced source nodes for an approximate result.
///
/// # Complexity
///
/// O(NE) for unweighted, O(NE log N) for weighted graphs, O(N + E) space.
pub fn run_betweenness(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let normalized = opt_bool(params, "normalized", true)?;
    let directed = opt_bool(params, "directed", true)?;
    let sample_size = opt_usize(params, "sampleSize", 0)?;
    let weight_prop = opt_weight_prop(params)?;

    let snap = GraphSnapshot::build(graph, weight_prop);
    let n = snap.n;

    if n == 0 {
        return Ok(vec![]);
    }

    let dir = if directed { Direction::Out } else { Direction::Any };
    let mut betweenness = vec![0.0f64; n];

    // Determine the set of source nodes: all nodes, or a deterministic sample.
    let sources: Vec<usize> = if sample_size == 0 || sample_size >= n {
        (0..n).collect()
    } else {
        // Evenly-spaced indices give a deterministic, reproducible sample.
        let step = n / sample_size;
        (0..sample_size).map(|k| k * step).collect()
    };

    for &s in &sources {
        // BFS (unweighted) or Dijkstra (weighted) from source s.
        // Returns: distance array, shortest-path count array, predecessor lists.
        let (dist, sigma, pred) = if weight_prop.is_none() {
            brandes_bfs(&snap, s, dir)
        } else {
            brandes_dijkstra(&snap, s, dir)
        };

        // Back-propagate the dependency score δ from farthest to closest nodes.
        let mut delta = vec![0.0f64; n];

        // Process nodes in non-increasing distance order (farthest first).
        let mut stack_order: Vec<usize> = (0..n).filter(|&v| dist[v].is_finite()).collect();
        stack_order.sort_unstable_by(|&a, &b| dist[b].partial_cmp(&dist[a]).unwrap());

        for &w in &stack_order {
            for &v in &pred[w] {
                if sigma[w] > 0.0 {
                    // Dependency accumulation: each predecessor v receives a
                    // fraction proportional to its share of paths through w.
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s {
                betweenness[w] += delta[w];
            }
        }
    }

    // Scale approximation up if only a sample of sources was used.
    if sample_size > 0 && sample_size < n {
        let scale = n as f64 / sample_size as f64;
        for b in &mut betweenness {
            *b *= scale;
        }
    }

    // Normalise by the maximum possible betweenness value.
    if normalized && n > 2 {
        // Directed: max = (N-1)(N-2) ordered pairs excluding self
        // Undirected: divide by additional factor of 2 (each pair counted once)
        let norm = if directed {
            1.0 / ((n as f64 - 1.0) * (n as f64 - 2.0))
        } else {
            2.0 / ((n as f64 - 1.0) * (n as f64 - 2.0))
        };
        for b in &mut betweenness {
            *b *= norm;
        }
    }

    Ok(betweenness
        .into_iter()
        .enumerate()
        .map(|(i, score)| {
            let mut row = HashMap::new();
            row.insert(
                "node".to_string(),
                Value::String(ulid_encode(snap.node_ids[i].0)),
            );
            row.insert("score".to_string(), Value::Float(score));
            row
        })
        .collect())
}

/// BFS from source `s` for unweighted betweenness computation.
///
/// Returns `(dist, sigma, pred)` where:
/// - `dist[v]` = hop distance from s to v (∞ if unreachable)
/// - `sigma[v]` = number of shortest paths from s to v
/// - `pred[v]` = predecessor nodes of v on any shortest path from s
fn brandes_bfs(
    snap: &GraphSnapshot,
    s: usize,
    dir: Direction,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<usize>>) {
    let n = snap.n;
    let mut dist = vec![f64::INFINITY; n];
    let mut sigma = vec![0.0f64; n];
    let mut pred: Vec<Vec<usize>> = vec![vec![]; n];

    dist[s] = 0.0;
    sigma[s] = 1.0;
    let mut queue = VecDeque::new();
    queue.push_back(s);

    while let Some(v) = queue.pop_front() {
        for (w, _) in snap.neighbors(v, dir) {
            // First time w is discovered: record its distance.
            if dist[w].is_infinite() {
                dist[w] = dist[v] + 1.0;
                queue.push_back(w);
            }
            // Count paths: any v at distance dist[w]-1 is a valid predecessor.
            if (dist[w] - dist[v] - 1.0).abs() < 1e-9 {
                sigma[w] += sigma[v];
                pred[w].push(v);
            }
        }
    }
    (dist, sigma, pred)
}

/// Dijkstra from source `s` for weighted betweenness computation.
///
/// Uses the same lazy-deletion min-heap as the main shortest-path module.
/// Equal-distance paths are all recorded (multiple predecessors per node).
///
/// Returns `(dist, sigma, pred)` — same semantics as [`brandes_bfs`].
fn brandes_dijkstra(
    snap: &GraphSnapshot,
    s: usize,
    dir: Direction,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<usize>>) {
    let n = snap.n;
    let mut dist = vec![f64::INFINITY; n];
    let mut sigma = vec![0.0f64; n];
    let mut pred: Vec<Vec<usize>> = vec![vec![]; n];

    dist[s] = 0.0;
    sigma[s] = 1.0;

    // Store distances as raw u64 bits for Ord compatibility (non-negative only).
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    heap.push(Reverse((0u64, s)));

    while let Some(Reverse((d_bits, v))) = heap.pop() {
        let d = f64::from_bits(d_bits);
        if d > dist[v] {
            continue; // stale entry
        }
        for (w, weight) in snap.neighbors(v, dir) {
            let w_val = weight.max(0.0); // clamp negative weights to 0
            let new_dist = dist[v] + w_val;

            if new_dist < dist[w] - 1e-12 {
                // Strictly shorter path found: replace predecessor list.
                dist[w] = new_dist;
                sigma[w] = sigma[v];
                pred[w] = vec![v];
                heap.push(Reverse((new_dist.to_bits(), w)));
            } else if (new_dist - dist[w]).abs() < 1e-12 {
                // Equal-length path found: add to predecessor list.
                sigma[w] += sigma[v];
                pred[w].push(v);
            }
        }
    }
    (dist, sigma, pred)
}

// ── Closeness Centrality ──────────────────────────────────────────────────────

/// Compute closeness centrality for all nodes.
///
/// Closeness is the reciprocal of the mean distance from a node to all other
/// reachable nodes:
///
/// ```text
/// C(u) = (reachable - 1) / Σ_{v reachable} dist(u, v)
/// ```
///
/// The **Wasserman-Faust** correction adjusts for disconnected graphs by
/// multiplying by `(reachable / (N-1))²`, which penalises nodes that can
/// only reach a fraction of the graph.
///
/// # Parameters (from GQL `call_arg`)
///
/// - `normalized` (Bool, default true) — normalize by (N-1).
/// - `direction` (String, default `"out"`) — edge direction for BFS.
/// - `wfImproved` (Bool, default true) — apply Wasserman-Faust correction.
///
/// # Complexity
///
/// O(N × (N + E)) time — one BFS per node.  O(N) auxiliary space per BFS.
pub fn run_closeness(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let normalized = opt_bool(params, "normalized", true)?;
    let direction = opt_direction(params, "direction", Direction::Out)?;
    let wf_improved = opt_bool(params, "wfImproved", true)?;

    let snap = GraphSnapshot::build(graph, None);
    let n = snap.n;

    if n == 0 {
        return Ok(vec![]);
    }

    let mut rows = Vec::with_capacity(n);

    for s in 0..n {
        // BFS from s to compute hop distances to all reachable nodes.
        let mut dist = vec![usize::MAX; n];
        dist[s] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(s);

        // Count reachable nodes and their total distance from s.
        let mut reachable = 0usize;
        let mut sum_dist = 0usize;

        while let Some(v) = queue.pop_front() {
            for (w, _) in snap.neighbors(v, direction) {
                if dist[w] == usize::MAX {
                    dist[w] = dist[v] + 1;
                    reachable += 1;
                    sum_dist += dist[w];
                    queue.push_back(w);
                }
            }
        }

        // Compute the closeness score, applying optional corrections.
        let score = if sum_dist == 0 || reachable == 0 {
            // Isolated node (or no outgoing reachable nodes) — score is 0.
            0.0
        } else {
            let raw = reachable as f64 / sum_dist as f64;
            if wf_improved {
                // Wasserman-Faust: multiply by (reachable/(N-1))^2 to
                // penalise nodes that cannot reach most of the graph.
                let wf = (reachable as f64 / (n as f64 - 1.0)).powi(2);
                if normalized { raw * wf } else { raw * wf * (n as f64 - 1.0) }
            } else if normalized {
                raw
            } else {
                // Unnormalized: raw closeness × (N-1) = 1 / mean_dist
                raw * (n as f64 - 1.0)
            }
        };

        let mut row = HashMap::new();
        row.insert(
            "node".to_string(),
            Value::String(ulid_encode(snap.node_ids[s].0)),
        );
        row.insert("score".to_string(), Value::Float(score));
        rows.push(row);
    }

    Ok(rows)
}

// ── Degree Centrality ─────────────────────────────────────────────────────────

/// Compute degree centrality for all nodes.
///
/// Returns in-degree, out-degree, and total degree for each node.  The `score`
/// column reflects whichever direction was requested by the `direction` parameter.
/// Normalization divides the raw degree count by (N - 1) — the maximum possible
/// degree in a simple graph.
///
/// Parallel edges each contribute separately to the degree count (multigraph
/// semantics, consistent with the adjacency snapshot).
///
/// # Parameters (from GQL `call_arg`)
///
/// - `direction` (String, default `"total"`) — which degree to use for `score`:
///   `"in"`, `"out"`, or `"total"`.
/// - `normalized` (Bool, default true) — divide `score` by (N-1).
///
/// # Yields
///
/// `node`, `in_degree`, `out_degree`, `degree` (total), `score`
///
/// # Complexity
///
/// O(N) time and space (reads precomputed adjacency list lengths).
pub fn run_degree(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let normalized = opt_bool(params, "normalized", true)?;

    // Parse the direction string manually here because "total" is not one of
    // the standard Direction enum values.
    let direction_str = match params.get("direction") {
        None => "total",
        Some(Value::String(s)) => s.as_str(),
        Some(other) => {
            return Err(DbError::Query(format!(
                "parameter 'direction' must be a string, got {other:?}"
            )))
        }
    };

    if !matches!(direction_str, "in" | "out" | "total") {
        return Err(DbError::Query(format!(
            "parameter 'direction' must be \"in\", \"out\", or \"total\", got \"{direction_str}\""
        )));
    }

    let snap = GraphSnapshot::build(graph, None);
    let n = snap.n;

    if n == 0 {
        return Ok(vec![]);
    }

    // (N-1) is the maximum degree in a simple graph; guard against N=1.
    let norm_denom = if n > 1 { (n - 1) as f64 } else { 1.0 };

    Ok((0..n)
        .map(|i| {
            let in_deg = snap.adj_in[i].len();
            let out_deg = snap.adj_out[i].len();
            let total_deg = in_deg + out_deg;

            let raw_score = match direction_str {
                "in" => in_deg,
                "out" => out_deg,
                _ => total_deg,
            } as f64;

            let score = if normalized { raw_score / norm_denom } else { raw_score };

            let mut row = HashMap::new();
            row.insert(
                "node".to_string(),
                Value::String(ulid_encode(snap.node_ids[i].0)),
            );
            row.insert("in_degree".to_string(), Value::Int(in_deg as i64));
            row.insert("out_degree".to_string(), Value::Int(out_deg as i64));
            row.insert("degree".to_string(), Value::Int(total_deg as i64));
            row.insert("score".to_string(), Value::Float(score));
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

    // Star graph: a → b, a → c, a → d  (a is hub)
    fn star_graph() -> (Graph, NodeId, Vec<NodeId>) {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let spokes: Vec<NodeId> = (0..3).map(|_| {
            let n = make_node(&mut g);
            make_edge(&mut g, a, n);
            n
        }).collect();
        (g, a, spokes)
    }

    // ── PageRank ──

    #[test]
    fn pagerank_hub_has_highest_score() {
        // Hub a receives all incoming PR from b, c, d → a gets highest in-PR.
        // But in a star (a → spokes), spokes receive from a, so spokes rank higher.
        // Use reverse star: spokes → a.
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        make_edge(&mut g, b, a);
        make_edge(&mut g, c, a);

        let params = HashMap::new();
        let rows = run_pagerank(&g, &params).unwrap();
        let a_score = rows.iter()
            .find(|r| r["node"] == Value::String(crate::types::ulid_encode(a.0)))
            .map(|r| if let Value::Float(f) = r["score"] { f } else { 0.0 })
            .unwrap();
        let b_score = rows.iter()
            .find(|r| r["node"] == Value::String(crate::types::ulid_encode(b.0)))
            .map(|r| if let Value::Float(f) = r["score"] { f } else { 0.0 })
            .unwrap();
        assert!(a_score > b_score);
    }

    #[test]
    fn pagerank_scores_sum_to_one() {
        let (g, _, _) = star_graph();
        let params = HashMap::new();
        let rows = run_pagerank(&g, &params).unwrap();
        let sum: f64 = rows.iter()
            .map(|r| if let Value::Float(f) = r["score"] { f } else { 0.0 })
            .sum();
        assert!((sum - 1.0).abs() < 1e-6, "scores sum to {sum}");
    }

    #[test]
    fn pagerank_invalid_damping_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> =
            [("damping".to_string(), Value::Float(1.5))].into_iter().collect();
        assert!(run_pagerank(&g, &params).is_err());
    }

    #[test]
    fn pagerank_zero_iterations_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> =
            [("iterations".to_string(), Value::Int(0))].into_iter().collect();
        assert!(run_pagerank(&g, &params).is_err());
    }

    // ── Betweenness ──

    #[test]
    fn betweenness_bridge_node_highest() {
        // a — b — c  (b is bridge, highest betweenness)
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        make_edge(&mut g, a, b);
        make_edge(&mut g, b, c);

        let params: HashMap<String, Value> =
            [("directed".to_string(), Value::Bool(false))].into_iter().collect();
        let rows = run_betweenness(&g, &params).unwrap();

        let score_of = |id: NodeId| -> f64 {
            rows.iter()
                .find(|r| r["node"] == Value::String(crate::types::ulid_encode(id.0)))
                .map(|r| if let Value::Float(f) = r["score"] { f } else { 0.0 })
                .unwrap_or(0.0)
        };
        assert!(score_of(b) > score_of(a));
        assert!(score_of(b) > score_of(c));
    }

    #[test]
    fn betweenness_empty_graph() {
        let g = Graph::new();
        let rows = run_betweenness(&g, &HashMap::new()).unwrap();
        assert!(rows.is_empty());
    }

    // ── Closeness ──

    #[test]
    fn closeness_central_node_highest() {
        // a — b — c  (b is most central in undirected view)
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        make_edge(&mut g, a, b);
        make_edge(&mut g, b, c);

        let params: HashMap<String, Value> =
            [("direction".to_string(), Value::String("any".into()))].into_iter().collect();
        let rows = run_closeness(&g, &params).unwrap();

        let b_score = rows.iter()
            .find(|r| r["node"] == Value::String(crate::types::ulid_encode(b.0)))
            .map(|r| if let Value::Float(f) = r["score"] { f } else { 0.0 })
            .unwrap_or(0.0);
        let a_score = rows.iter()
            .find(|r| r["node"] == Value::String(crate::types::ulid_encode(a.0)))
            .map(|r| if let Value::Float(f) = r["score"] { f } else { 0.0 })
            .unwrap_or(0.0);

        assert!(b_score >= a_score);
    }

    // ── Degree ──

    #[test]
    fn degree_counts_correct() {
        let (g, hub, spokes) = star_graph();
        let params = HashMap::new();
        let rows = run_degree(&g, &params).unwrap();

        let hub_row = rows.iter()
            .find(|r| r["node"] == Value::String(crate::types::ulid_encode(hub.0)))
            .unwrap();
        assert_eq!(hub_row["out_degree"], Value::Int(3));
        assert_eq!(hub_row["in_degree"], Value::Int(0));
    }

    #[test]
    fn degree_invalid_direction_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> = [(
            "direction".to_string(),
            Value::String("diagonal".into()),
        )]
        .into_iter()
        .collect();
        assert!(run_degree(&g, &params).is_err());
    }
}
