//! Structural similarity: Jaccard, Overlap, Common Neighbours.
//!
//! **CALL signature:**
//! ```gql
//! CALL jaccardSimilarity(node: "<ulid>", direction: "any", topK: 10)
//! YIELD node1, node2, score
//!
//! // Omit 'node' to compute pairwise similarity for all pairs (expensive for large graphs).
//! CALL jaccardSimilarity(topK: 5) YIELD node1, node2, score
//! ```
//!
//! | param       | type   | default  | description                                               |
//! |-------------|--------|----------|-----------------------------------------------------------|
//! | `node`      | String | none     | Source ULID; if absent, computes all-pairs               |
//! | `direction` | String | `"any"`  | `"out"`, `"in"`, or `"any"` for neighbour sets           |
//! | `topK`      | Int    | 10       | Return only top-K most similar nodes per source          |
//! | `method`    | String | `"jaccard"` | `"jaccard"`, `"overlap"`, or `"commonNeighbors"`      |
//!
//! **Yields:** `node1` (String), `node2` (String), `score` (Float)
//!
//! Jaccard(A, B) = |N(A) ∩ N(B)| / |N(A) ∪ N(B)|
//! Overlap(A, B) = |N(A) ∩ N(B)| / min(|N(A)|, |N(B)|)
//! CommonNeighbors(A, B) = |N(A) ∩ N(B)|  (raw count)

use std::collections::{HashMap, HashSet};

use crate::graph::Graph;
use crate::types::{ulid_encode, DbError, Value};

use super::{opt_direction, opt_node_idx, opt_str, opt_usize, Direction, GraphSnapshot, Row};

/// Entry point called by `dispatch_call` for `CALL jaccardSimilarity(...)`.
///
/// Computes pairwise structural similarity scores between nodes based on
/// their neighbour sets.  When `node` is given, only that node is used as
/// a source; otherwise all-pairs similarity is computed (O(N² · degree)).
///
/// Scores of 0 are suppressed from output.  Results are sorted descending
/// and truncated to `topK` entries per source node.
pub fn run_jaccard(graph: &Graph, params: &HashMap<String, Value>) -> Result<Vec<Row>, DbError> {
    let direction = opt_direction(params, "direction", Direction::Any)?;
    let top_k = opt_usize(params, "topK", 10)?;
    if top_k == 0 {
        return Err(DbError::Query("parameter 'topK' must be at least 1".into()));
    }
    let method = opt_str(params, "method", "jaccard")?;
    if !matches!(method, "jaccard" | "overlap" | "commonNeighbors") {
        return Err(DbError::Query(format!(
            "unknown 'method' value '{method}': expected \
             \"jaccard\", \"overlap\", or \"commonNeighbors\""
        )));
    }

    let snap = GraphSnapshot::build(graph, None);
    let n = snap.n;

    if n == 0 {
        return Ok(vec![]);
    }

    // Pre-compute neighbour sets (deduplicated, excluding self-loops).
    let nbr_sets: Vec<HashSet<usize>> = (0..n)
        .map(|i| {
            snap.neighbors(i, direction)
                .into_iter()
                .map(|(j, _)| j)
                .filter(|&j| j != i)
                .collect()
        })
        .collect();

    let source = opt_node_idx(params, "node", &snap)?;

    let sources: Vec<usize> = match source {
        Some(s) => vec![s],
        None => (0..n).collect(),
    };

    let mut rows: Vec<Row> = Vec::new();

    for &a in &sources {
        // Collect (score, b) for all b ≠ a.
        let mut scored: Vec<(f64, usize)> = (0..n)
            .filter(|&b| b != a)
            .map(|b| {
                let score = compute_score(&nbr_sets[a], &nbr_sets[b], method);
                (score, b)
            })
            .filter(|&(s, _)| s > 0.0)
            .collect();

        // Sort descending, take top_k.
        scored.sort_unstable_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        for (score, b) in scored {
            let mut row = HashMap::new();
            row.insert(
                "node1".to_string(),
                Value::String(ulid_encode(snap.node_ids[a].0)),
            );
            row.insert(
                "node2".to_string(),
                Value::String(ulid_encode(snap.node_ids[b].0)),
            );
            row.insert("score".to_string(), Value::Float(score));
            rows.push(row);
        }
    }

    Ok(rows)
}

/// Compute a pairwise similarity score between neighbour sets `a` and `b`.
///
/// - `"jaccard"`:         |A ∩ B| / |A ∪ B|  — in [0, 1]
/// - `"overlap"`:         |A ∩ B| / min(|A|, |B|) — in [0, 1]
/// - `"commonNeighbors"`: |A ∩ B|  (raw integer count, cast to f64)
///
/// Returns 0.0 for any unrecognised method (guarded by caller validation).
fn compute_score(a: &HashSet<usize>, b: &HashSet<usize>, method: &str) -> f64 {
    let intersection = a.intersection(b).count();
    match method {
        "jaccard" => {
            let union = a.union(b).count();
            if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
        }
        "overlap" => {
            let min_size = a.len().min(b.len());
            if min_size == 0 { 0.0 } else { intersection as f64 / min_size as f64 }
        }
        "commonNeighbors" => intersection as f64,
        _ => 0.0,
    }
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
        g.apply_insert_node(Node { id, labels: vec!["N".into()], properties: Default::default() });
        id
    }

    /// Insert an undirected edge between `from` and `to`.
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
    fn identical_neighbourhoods_score_one() {
        // a and b both connect to {c, d}
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        let d = make_node(&mut g);
        make_edge(&mut g, a, c);
        make_edge(&mut g, a, d);
        make_edge(&mut g, b, c);
        make_edge(&mut g, b, d);

        let params: HashMap<String, Value> = [(
            "node".to_string(),
            Value::String(ulid_encode(a.0)),
        )]
        .into_iter()
        .collect();
        let rows = run_jaccard(&g, &params).unwrap();
        let b_row = rows.iter()
            .find(|r| r["node2"] == Value::String(ulid_encode(b.0)));
        assert!(b_row.is_some(), "b should be in results");
        if let Value::Float(s) = b_row.unwrap()["score"] {
            assert!((s - 1.0).abs() < 1e-9, "Jaccard of identical sets = 1.0, got {s}");
        }
    }

    #[test]
    fn disjoint_neighbourhoods_score_zero() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        let d = make_node(&mut g);
        make_edge(&mut g, a, c); // N(a) = {c}
        make_edge(&mut g, b, d); // N(b) = {d}  — disjoint

        let params: HashMap<String, Value> = [(
            "node".to_string(),
            Value::String(ulid_encode(a.0)),
        )]
        .into_iter()
        .collect();
        let rows = run_jaccard(&g, &params).unwrap();
        // Score 0 rows are filtered out.
        let b_row = rows.iter()
            .find(|r| r["node2"] == Value::String(ulid_encode(b.0)));
        assert!(b_row.is_none());
    }

    #[test]
    fn top_k_limits_results() {
        let mut g = Graph::new();
        let src = make_node(&mut g);
        let shared = make_node(&mut g);
        let others: Vec<_> = (0..5).map(|_| {
            let n = make_node(&mut g);
            make_edge(&mut g, src, shared);
            make_edge(&mut g, n, shared);
            n
        }).collect();

        let params: HashMap<String, Value> = [
            ("node".to_string(), Value::String(ulid_encode(src.0))),
            ("topK".to_string(), Value::Int(2)),
        ]
        .into_iter()
        .collect();
        let rows = run_jaccard(&g, &params).unwrap();
        assert!(rows.len() <= 2);
    }

    #[test]
    fn overlap_method_works() {
        let mut g = Graph::new();
        let a = make_node(&mut g);
        let b = make_node(&mut g);
        let c = make_node(&mut g);
        let d = make_node(&mut g);
        make_edge(&mut g, a, c);
        make_edge(&mut g, a, d);
        make_edge(&mut g, b, c); // N(b) ⊂ N(a) → overlap = 1.0

        let params: HashMap<String, Value> = [
            ("node".to_string(), Value::String(ulid_encode(a.0))),
            ("method".to_string(), Value::String("overlap".into())),
        ]
        .into_iter()
        .collect();
        let rows = run_jaccard(&g, &params).unwrap();
        let b_row = rows.iter()
            .find(|r| r["node2"] == Value::String(ulid_encode(b.0)));
        if let Some(r) = b_row {
            if let Value::Float(s) = r["score"] {
                assert!((s - 1.0).abs() < 1e-9, "overlap = 1.0 when N(b) ⊆ N(a), got {s}");
            }
        }
    }

    #[test]
    fn invalid_method_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> = [(
            "method".to_string(),
            Value::String("cosine".into()),
        )]
        .into_iter()
        .collect();
        assert!(run_jaccard(&g, &params).is_err());
    }

    #[test]
    fn top_k_zero_errors() {
        let g = Graph::new();
        let params: HashMap<String, Value> =
            [("topK".to_string(), Value::Int(0))].into_iter().collect();
        assert!(run_jaccard(&g, &params).is_err());
    }
}
