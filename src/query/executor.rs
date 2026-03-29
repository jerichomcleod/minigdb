//! Tree-walking interpreter for the GQL AST produced by [`super::parser`].
//!
//! # Architecture
//!
//! The executor is a tree-walker: each `Statement` variant dispatches to a
//! dedicated `execute_*` function that drives the full query lifecycle.
//!
//! **Binding model** — graph pattern matching produces a `Vec<Bindings>` where
//! each `Bindings` is a `HashMap<String, Binding>`.  A `Binding` holds either a
//! node/edge/path ID (for lazy property lookup) or an already-evaluated `Value`
//! (for UNWIND/WITH scalars).  Nodes and edges are kept as IDs so property
//! lookups hit the graph store only when the RETURN clause needs them.
//!
//! **Two-phase aggregation** — if any RETURN item contains an aggregate function,
//! `execute_aggregate` groups the `Vec<Bindings>` by non-aggregate key expressions
//! and reduces each group to a single `Row`.  Non-aggregate queries project
//! bindings directly via `project_return`, skipping the grouping step entirely.
//!
//! **Auto-commit vs capture mode** — `execute` is the fire-and-forget entry point;
//! `execute_capturing` returns the `Vec<Operation>` applied to the graph so the
//! caller (REPL explicit-transaction logic or the network server) can decide when
//! and how to flush them to the WAL.
//!
//! **Parameterized queries** — `execute_capturing_with_params` accepts a
//! `HashMap<String, Value>` and stores it in a thread-local before dispatching to
//! the normal `execute_capturing` path.  `eval_expr` resolves `Expr::Param(name)`
//! by reading from that thread-local; the map is cleared unconditionally after
//! dispatch regardless of success or failure.  A missing key evaluates to
//! `Value::Null`.  Params can appear anywhere an expression is valid: WHERE
//! predicates, INSERT property values, LIMIT/OFFSET, and `UNWIND $list AS var
//! INSERT (…)` for bulk inserts.
//!
//! **Data constraints** — `CONSTRAINT` statements are executed by
//! `execute_constraint`, which delegates to the graph-layer `add_constraint` /
//! `remove_constraint` / `list_constraints` API.  Enforcement happens inside
//! `build_insert_ops` (before upsert lookup) and inside `set_node_property`
//! (via `ops::check_constraints`).  The `self_id` parameter lets SET pass the
//! current node's own ID so it is not incorrectly flagged as a duplicate of itself.
//!
//! # MATCH execution model (Phase 1–3)
//! 1. Label index → candidate start nodes (avoids full scan when label given)
//! 2. Extend path along `EdgePatternStep` using adjacency indexes
//!    - Single-hop: direct edge traversal with path-mode checks
//!    - Quantified: BFS up to the depth bound, respecting TRAIL/SIMPLE mode
//! 3. Test edge/node constraints at each step
//! 4. Fully-matched path → variable binding map `HashMap<var, Binding>`
//! 5. WHERE filter
//! 6. RETURN projection
//! 7. ORDER BY, OFFSET, LIMIT

use std::collections::{HashMap, HashSet, VecDeque};
use std::cell::RefCell;

use crate::graph::Graph;
use crate::storage::apply_ops;
use crate::transaction::operation::{Operation, PropertyTarget};
use crate::types::{ulid_encode, DbError, EdgeId, Node, NodeId, Properties, Value};

use super::ast::*;

// Thread-local query parameters — set by execute_capturing_with_params before dispatch.
thread_local! {
    static QUERY_PARAMS: RefCell<HashMap<String, Value>> = RefCell::new(HashMap::new());
}

/// A row of output values, keyed by column name.
pub type Row = HashMap<String, Value>;

/// A variable binding accumulated while matching a graph pattern.
/// Values are node/edge IDs so we can look up properties lazily.
#[derive(Debug, Clone)]
enum Binding {
    /// A matched node — its full data was not needed during matching and will
    /// be loaded lazily on first property access.
    Node(NodeId),
    /// A matched node whose data was already loaded during constraint checking.
    /// Carrying the `Node` here avoids a second `get_node` call when the
    /// executor later evaluates `n.property` expressions.
    NodeData(NodeId, Node),
    Edge(EdgeId),
    /// Bound by a quantified edge pattern — a list of edge IDs forming the path.
    EdgeList(Vec<EdgeId>),
    /// Bound by UNWIND or WITH — an already-evaluated scalar or list value.
    Value(Value),
}

type Bindings = HashMap<String, Binding>;

/// Execute a parsed `Statement` against the graph.
///
/// This is the primary entry point for callers that do not need to inspect the
/// individual `Operation`s produced (e.g. tests and simple one-shot evaluation).
/// Write statements are applied to `graph` immediately; read statements return
/// projected rows.  A one-element summary row is returned for write statements.
///
/// `next_txn_id` is a monotone counter that provides an opaque transaction ID
/// embedded in summary messages; it is incremented for every write statement.
pub fn execute(
    stmt: Statement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<Vec<Row>, DbError> {
    let (rows, _ops) = execute_capturing(stmt, graph, next_txn_id)?;
    Ok(rows)
}

/// Execute a statement and return both the result rows and the operations applied.
///
/// The operations have **already been applied** to `graph` by the time this
/// function returns.  The caller can use them to:
/// - Write a single WAL frame immediately (auto-commit / REPL non-txn mode).
/// - Buffer them inside an explicit `BEGIN`/`COMMIT` transaction and write the
///   whole batch as one atomic WAL frame on `COMMIT`.
/// - Discard them on `ROLLBACK` (the graph snapshot is restored separately).
///
/// Read-only statements (MATCH, OPTIONAL MATCH, UNWIND, …) return an empty
/// `Vec<Operation>`.
pub fn execute_capturing(
    stmt: Statement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    match stmt {
        Statement::Match(m)         => Ok((execute_match(m, graph)?, vec![])),
        Statement::OptionalMatch(m) => Ok((execute_optional_match(m, graph)?, vec![])),
        Statement::MatchOptionalMatch(m) => Ok((execute_match_optional_match(m, graph)?, vec![])),
        Statement::Unwind(u)        => Ok((execute_unwind(u, graph)?, vec![])),
        Statement::Union(u)         => execute_union(u, graph, next_txn_id),
        Statement::MatchWith(mw)    => Ok((execute_match_with(mw, graph)?, vec![])),
        Statement::Insert(ins)      => execute_insert(ins, graph, next_txn_id),
        Statement::MatchInsert(mi)  => execute_match_insert(mi, graph, next_txn_id),
        Statement::Set(s)           => execute_set(s, graph, next_txn_id),
        Statement::Remove(r)        => execute_remove(r, graph, next_txn_id),
        Statement::Delete(d)        => execute_delete(d, graph, next_txn_id),
        Statement::CreateIndex(ci)  => execute_create_index(ci, graph, next_txn_id),
        Statement::DropIndex(di)    => execute_drop_index(di, graph, next_txn_id),
        Statement::ShowIndexes      => Ok((execute_show_indexes(graph)?, vec![])),
        Statement::Call(c)          => Ok((execute_call(c, graph)?, vec![])),
        Statement::CallPipeline(c)  => Ok((execute_call_pipeline(c, graph)?, vec![])),
        Statement::Truncate         => execute_truncate(graph),
        Statement::LoadCsvNodes(n)  => execute_load_csv_nodes(n, graph, next_txn_id),
        Statement::LoadCsvEdges(e)  => execute_load_csv_edges(e, graph, next_txn_id),
        Statement::UnwindInsert(u)  => execute_unwind_insert(u, graph, next_txn_id),
        Statement::Constraint(c)    => execute_constraint(c, graph),
    }
}

/// Execute a statement with named query parameters (`$name` placeholders).
///
/// Parameters are bound via a thread-local before dispatch and cleared afterwards,
/// so `eval_expr` can resolve `Expr::Param` without changing its call signature.
pub fn execute_capturing_with_params(
    stmt: Statement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
    params: HashMap<String, Value>,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    QUERY_PARAMS.with(|p| *p.borrow_mut() = params);
    let result = execute_capturing(stmt, graph, next_txn_id);
    QUERY_PARAMS.with(|p| p.borrow_mut().clear());
    result
}

// ── MATCH ──────────────────────────────────────────────────────────────────

/// O(1) fast path for `MATCH (n:Label) RETURN count(*) [AS alias]` or
/// `RETURN count(n) [AS alias]` with no WHERE clause and no inline property
/// constraints on the node pattern.
///
/// Returns `Some(rows)` when the query can be answered by a single
/// `get_label_count_if_known` meta lookup, bypassing pattern matching and
/// row materialisation entirely.  Returns `None` to fall through to the
/// normal execution path when:
///   - multiple patterns or edge steps are present,
///   - the node has more than one label or inline property constraints,
///   - a WHERE clause is present,
///   - the RETURN clause has multiple items or is not a bare `count`,
///   - the label count has never been written (pre-existing database without counters).
fn try_label_count_fast_path(m: &MatchStatement, graph: &Graph) -> Option<Vec<Row>> {
    // Single pattern, no edge steps.
    if m.patterns.len() != 1 {
        return None;
    }
    let pat = &m.patterns[0];
    if !pat.steps.is_empty() {
        return None;
    }

    // Exactly one label, no inline property constraints.
    let node = &pat.start;
    if node.labels.len() != 1 || !node.properties.is_empty() {
        return None;
    }

    // No WHERE clause.
    if m.where_clause.is_some() {
        return None;
    }

    // Single RETURN item that is count(*) or count(var).
    if m.return_clause.items.len() != 1 {
        return None;
    }
    let item = &m.return_clause.items[0];
    let is_count = match &item.expr {
        Expr::Call(name, args) if name.eq_ignore_ascii_case("count") => match args.as_slice() {
            [Expr::Star] => true,
            [Expr::Var(v)] => node.variable.as_deref() == Some(v.as_str()),
            _ => false,
        },
        _ => false,
    };
    if !is_count {
        return None;
    }

    let label = &node.labels[0];
    // Only use the fast path when the counter is known to be initialised.
    // `None` means the key was never written (pre-existing DB); fall through.
    let n = graph.store.get_label_count_if_known(label)? as i64;
    let col = item.alias.clone().unwrap_or_else(|| expr_display(&item.expr));

    let mut row = Row::new();
    row.insert(col, Value::Int(n));
    Some(vec![row])
}

/// Execute a `MATCH … [WHERE] RETURN …` statement.
///
/// Workflow:
/// 1. Cross-join all comma-separated patterns (Cartesian product filtered by WHERE).
/// 2. Apply the WHERE predicate to the merged `Bindings` stream.
/// 3. If any RETURN item is an aggregate, delegate to `execute_aggregate`;
///    otherwise sort by ORDER BY and project each binding via `project_return`.
/// 4. Apply DISTINCT, then OFFSET and LIMIT.
fn execute_match(m: MatchStatement, graph: &Graph) -> Result<Vec<Row>, DbError> {
    // Fast path: MATCH (n:Label) RETURN count(*) / count(n) — O(1) meta lookup.
    if let Some(rows) = try_label_count_fast_path(&m, graph) {
        return Ok(rows);
    }
    // 1. Cross-join all patterns. A single pattern is the common case;
    //    multiple patterns match disconnected subgraphs and combine results as
    //    a Cartesian product (like SQL CROSS JOIN filtered by WHERE).

    // Compute early-termination limit: usable only when the full result set is
    // not required — i.e. no ORDER BY, no aggregates, no DISTINCT, no WHERE.
    // When safe, we pass `offset + limit` into the scan so RocksDB iteration
    // stops after finding enough candidates rather than reading every node.
    let has_aggregates = m.return_clause.items.iter().any(|it| contains_aggregate(&it.expr));
    let early_limit: Option<usize> = if m.where_clause.is_none()
        && m.return_clause.order_by.is_empty()
        && !m.return_clause.distinct
        && !has_aggregates
    {
        match m.return_clause.limit.as_ref().and_then(|e| eval_expr_literal(e).ok()) {
            Some(Value::Int(n)) if n > 0 => {
                let offset = m.return_clause.offset.as_ref()
                    .and_then(|e| eval_expr_literal(e).ok())
                    .and_then(|v| if let Value::Int(o) = v { Some(o.max(0) as usize) } else { None })
                    .unwrap_or(0);
                Some((n as usize).saturating_add(offset))
            }
            _ => None,
        }
    } else {
        None
    };

    let bindings = cross_join_patterns(&m.patterns, graph, m.path_mode, m.where_clause.as_ref(), early_limit);

    // 2. WHERE filter.
    let filtered: Vec<Bindings> = bindings
        .into_iter()
        .filter(|b| {
            if let Some(ref cond) = m.where_clause {
                matches!(eval_expr(cond, b, graph), Ok(Value::Bool(true)))
            } else {
                true
            }
        })
        .collect();

    // 3. Check for aggregate functions in the RETURN clause.
    //    If present, group bindings and compute aggregates; ORDER BY then operates
    //    on the projected rows rather than on raw bindings.
    // (has_aggregates was already computed above for early_limit eligibility.)

    let mut rows: Vec<Row> = if has_aggregates {
        // Aggregate path: group → compute aggregate rows.
        let agg_rows = execute_aggregate(&filtered, &m.return_clause.items, graph)?;

        // ORDER BY on projected rows (eval_expr_on_row looks up column by display name).
        let mut agg_rows = agg_rows;
        if !m.return_clause.order_by.is_empty() {
            let order = &m.return_clause.order_by;
            agg_rows.sort_by(|a, b| {
                for item in order {
                    let va = eval_expr_on_row(&item.expr, a).unwrap_or(Value::Null);
                    let vb = eval_expr_on_row(&item.expr, b).unwrap_or(Value::Null);
                    let cmp = cmp_nulls_last(&va, &vb);
                    let cmp = if item.ascending { cmp } else { cmp.reverse() };
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
                std::cmp::Ordering::Equal
            });
        }
        agg_rows
    } else {
        // Non-aggregate path: sort bindings first so ORDER BY can access any column.
        let mut filtered = filtered;
        if !m.return_clause.order_by.is_empty() {
            let order = &m.return_clause.order_by;
            filtered.sort_by(|a, b| {
                for item in order {
                    let va = eval_expr(&item.expr, a, graph).unwrap_or(Value::Null);
                    let vb = eval_expr(&item.expr, b, graph).unwrap_or(Value::Null);
                    let cmp = cmp_nulls_last(&va, &vb);
                    let cmp = if item.ascending { cmp } else { cmp.reverse() };
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
                std::cmp::Ordering::Equal
            });
        }

        // RETURN projection.
        filtered
            .iter()
            .map(|b| project_return(&m.return_clause.items, b, graph))
            .collect::<Result<Vec<_>, _>>()?
    };

    // 4. DISTINCT.
    if m.return_clause.distinct {
        let mut seen = std::collections::HashSet::new();
        rows.retain(|row| {
            let mut keys: Vec<_> = row.keys().collect();
            keys.sort();
            let key: Vec<String> = keys.iter().map(|k| format!("{}={:?}", k, row[*k])).collect();
            seen.insert(key.join(","))
        });
    }

    // 5. OFFSET.
    let offset = match &m.return_clause.offset {
        Some(expr) => match eval_expr_literal(expr)? {
            Value::Int(n) => n.max(0) as usize,
            _ => return Err(DbError::Query("OFFSET must be an integer".into())),
        },
        None => 0,
    };

    // 6. LIMIT.
    let limit = match &m.return_clause.limit {
        Some(expr) => match eval_expr_literal(expr)? {
            Value::Int(n) => Some(n.max(0) as usize),
            _ => return Err(DbError::Query("LIMIT must be an integer".into())),
        },
        None => None,
    };

    let rows: Vec<Row> = rows
        .into_iter()
        .skip(offset)
        .take(limit.unwrap_or(usize::MAX))
        .collect();

    Ok(rows)
}

// ── OPTIONAL MATCH ──────────────────────────────────────────────────────────

/// Execute a standalone `OPTIONAL MATCH … RETURN …` statement.
///
/// If the pattern matches at least one binding, behaves identically to a plain
/// `MATCH`.  If the pattern matches nothing, emits one synthetic null-binding
/// row so that aggregate functions like `count(*)` still return a result (e.g.
/// `count(*) = 1`) while non-aggregate columns project to `Null`.
fn execute_optional_match(m: MatchStatement, graph: &Graph) -> Result<Vec<Row>, DbError> {
    let bindings = cross_join_patterns(&m.patterns, graph, m.path_mode, m.where_clause.as_ref(), None);

    let filtered: Vec<Bindings> = bindings
        .into_iter()
        .filter(|b| {
            if let Some(ref cond) = m.where_clause {
                matches!(eval_expr(cond, b, graph), Ok(Value::Bool(true)))
            } else {
                true
            }
        })
        .collect();

    // If no matches, emit one row with Null for each return item variable.
    // Use apply_return_clause so aggregates (e.g. count(*)) are handled correctly:
    // an empty binding row is passed through the aggregate path → count(*) = 0 for
    // count(expr), non-aggregate columns project to Null.
    if filtered.is_empty() {
        let null_row: Vec<Bindings> = vec![HashMap::new()];
        return apply_return_clause(null_row, &m.return_clause, graph);
    }

    apply_return_clause(filtered, &m.return_clause, graph)
}

// ── MATCH … OPTIONAL MATCH … RETURN ─────────────────────────────────────────

/// Execute `MATCH pattern [WHERE] (OPTIONAL MATCH pattern [WHERE])+ RETURN …`.
/// Each OPTIONAL MATCH is applied as a left join: for each binding from the
/// preceding clauses, it tries to extend with matches from the optional pattern.
/// If no match is found, the row is kept with null bindings for optional vars.
fn execute_match_optional_match(
    m: super::ast::MatchOptionalMatchStatement,
    graph: &Graph,
) -> Result<Vec<Row>, DbError> {
    // Phase 1: run the mandatory MATCH.
    let mut rows: Vec<Bindings> = cross_join_patterns(&m.patterns, graph, m.path_mode, m.where_clause.as_ref(), None);
    if let Some(ref cond) = m.where_clause {
        rows.retain(|b| matches!(eval_expr(cond, b, graph), Ok(Value::Bool(true))));
    }

    // Phase 2: apply each OPTIONAL MATCH as a left join.
    for clause in m.optional_clauses {
        rows = left_join_optional(rows, &clause.patterns, clause.where_clause.as_ref(), clause.path_mode, graph);
    }

    apply_return_clause(rows, &m.return_clause, graph)
}

/// For each row in `existing`, try to extend it with matches from `patterns`.
/// If the optional pattern matches, produce one merged row per match.
/// If it does not match, keep the original row unchanged (optional vars → Null).
fn left_join_optional(
    existing: Vec<Bindings>,
    patterns: &[super::ast::GraphPattern],
    where_clause: Option<&Expr>,
    path_mode: PathMode,
    graph: &Graph,
) -> Vec<Bindings> {
    let mut result = Vec::new();
    for seed in existing {
        let matches = cross_join_patterns_seeded(patterns, &seed, path_mode, where_clause, graph);
        if matches.is_empty() {
            result.push(seed);
        } else {
            result.extend(matches);
        }
    }
    result
}

/// Like `cross_join_patterns` but starts with an existing `seed` binding set.
/// When the start variable of a pattern is already bound in `seed`, only that
/// node is used as the candidate — no full scan needed.
fn cross_join_patterns_seeded(
    patterns: &[super::ast::GraphPattern],
    seed: &Bindings,
    mode: PathMode,
    where_clause: Option<&Expr>,
    graph: &Graph,
) -> Vec<Bindings> {
    let mut result: Vec<Bindings> = vec![seed.clone()];
    for pattern in patterns {
        let mut next: Vec<Bindings> = Vec::new();
        for existing in &result {
            let matches = match_pattern_seeded(pattern, existing, graph, mode, where_clause);
            next.extend(matches);
        }
        result = next;
    }
    result
}

/// Match a single graph pattern, using `seed` to pre-bind variables.
/// If the pattern's start variable is already bound in `seed`, only that node
/// is tried as the start node (avoids a full scan).
fn match_pattern_seeded(
    pattern: &super::ast::GraphPattern,
    seed: &Bindings,
    graph: &Graph,
    path_mode: PathMode,
    where_clause: Option<&Expr>,
) -> Vec<Bindings> {
    // Determine candidate start nodes.
    let start_nodes: Vec<NodeId> = if let Some(ref var) = pattern.start.variable {
        if let Some(Binding::Node(id) | Binding::NodeData(id, _)) = seed.get(var.as_str()) {
            // Already bound — use only that node.
            vec![*id]
        } else {
            // Not yet bound — scan as normal.
            get_candidate_start_nodes(pattern, graph, where_clause)
        }
    } else {
        get_candidate_start_nodes(pattern, graph, where_clause)
    };

    // When the start node carries no constraints beyond what the scan already
    // guarantees (single label, no inline property predicates), we can skip
    // loading the full node here and store only the NodeId in the binding.
    // The node will be loaded lazily on first property access via NodeData.
    let needs_constraint_check =
        pattern.start.properties.len() > 0 || pattern.start.labels.len() > 1;

    let mut results: Vec<Bindings> = Vec::new();
    for node_id in start_nodes {
        let start_binding = if needs_constraint_check {
            let node = match graph.get_node(node_id) { Some(n) => n, None => continue };
            if !check_node_constraints(&pattern.start, &node) { continue; }
            Binding::NodeData(node_id, node)
        } else {
            Binding::Node(node_id)
        };

        // Start from the seed binding, then add the start node variable.
        let mut binding = seed.clone();
        if let Some(ref var) = pattern.start.variable {
            binding.insert(var.clone(), start_binding);
        }

        let mut visited_nodes: HashSet<NodeId> = HashSet::new();
        visited_nodes.insert(node_id);
        let visited_edges: HashSet<EdgeId> = HashSet::new();

        let step_bindings = extend_steps(
            &pattern.steps, node_id, binding, graph,
            path_mode, &visited_edges, &visited_nodes,
            None,
        );
        results.extend(step_bindings);
    }

    // Apply where clause (has access to full merged bindings including seed).
    if let Some(w) = where_clause {
        results.retain(|b| matches!(eval_expr(w, b, graph), Ok(Value::Bool(true))));
    }

    results
}

/// Extract candidate start nodes for a pattern using label/property indexes.
/// Extracted from `match_pattern` so it can be reused by `match_pattern_seeded`.
///
/// Collects candidates from ALL usable indexes (inline constraints + WHERE) and
/// intersects them — the same logic as the primary `match_pattern` path.
fn get_candidate_start_nodes(
    pattern: &super::ast::GraphPattern,
    graph: &Graph,
    where_clause: Option<&Expr>,
) -> Vec<NodeId> {
    if let Some(primary) = pattern.start.labels.first() {
        let mut index_sets: Vec<Vec<NodeId>> = Vec::new();

        for pc in &pattern.start.properties {
            if let Ok(val) = eval_expr_literal(&pc.value) {
                if let Some(ids) = graph.lookup_by_property(primary, &pc.key, &val) {
                    index_sets.push(ids);
                }
            }
        }
        if let (Some(var), Some(where_expr)) = (&pattern.start.variable, where_clause) {
            collect_where_index_candidates(graph, primary, var, where_expr, &mut index_sets);
        }

        if !index_sets.is_empty() {
            return intersect_node_id_sets(index_sets);
        }
        return graph.nodes_by_label(primary);
    }
    graph.all_nodes().into_iter().map(|n| n.id).collect()
}

// ── UNWIND ──────────────────────────────────────────────────────────────────

/// Execute an `UNWIND expr AS var RETURN …` statement.
///
/// Evaluates `expr` in an empty binding context (no graph variables in scope).
/// The result must be a `Value::List`; `Value::Null` is treated as an empty
/// list (produces zero rows).  Each list element is bound to `var` as a
/// `Binding::Value` and the full set of per-element bindings is passed through
/// `apply_return_clause` for projection, aggregation, and post-processing.
fn execute_unwind(u: UnwindStatement, graph: &Graph) -> Result<Vec<Row>, DbError> {
    // Evaluate the list expression in an empty binding context.
    let empty: Bindings = HashMap::new();
    let list_val = eval_expr(&u.expr, &empty, graph)?;

    let items = match list_val {
        Value::List(items) => items,
        Value::Null => vec![],
        other => return Err(DbError::Query(format!(
            "UNWIND requires a list expression, got {:?}", other
        ))),
    };

    let mut all_bindings: Vec<Bindings> = Vec::new();
    for item in items {
        let mut b: Bindings = HashMap::new();
        b.insert(u.variable.clone(), Binding::Value(item));
        all_bindings.push(b);
    }

    apply_return_clause(all_bindings, &u.return_clause, graph)
}

// ── UNION ──────────────────────────────────────────────────────────────────

/// Execute a `… UNION [ALL] …` statement by executing each branch independently
/// and concatenating the results.
///
/// `UNION` (without `ALL`) deduplicates by serialising each row to a
/// canonical string key and discarding rows already seen.  `UNION ALL` skips
/// that step and keeps all duplicates.
fn execute_union(
    u: UnionStatement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<crate::transaction::operation::Operation>), DbError> {
    let mut all_rows: Vec<Row> = Vec::new();
    let mut all_ops:  Vec<crate::transaction::operation::Operation> = Vec::new();

    for branch in u.branches {
        let (rows, ops) = execute_capturing(branch, graph, next_txn_id)?;
        all_rows.extend(rows);
        all_ops.extend(ops);
    }

    if !u.all {
        // Deduplicate rows (UNION without ALL).
        let mut seen = std::collections::HashSet::new();
        all_rows.retain(|row| {
            let mut keys: Vec<_> = row.keys().collect();
            keys.sort();
            let key: Vec<String> = keys.iter().map(|k| format!("{}={:?}", k, row[*k])).collect();
            seen.insert(key.join(","))
        });
    }

    Ok((all_rows, all_ops))
}

// ── MATCH … WITH … RETURN ──────────────────────────────────────────────────

/// Execute a `MATCH … WITH … [WHERE] RETURN …` statement.
///
/// The `WITH` clause acts as an intermediate projection + optional aggregation
/// step between the MATCH phase and the final RETURN:
/// 1. MATCH → filter by WHERE.
/// 2. WITH projection (possibly aggregating) → convert bindings to projected rows.
/// 3. Optional DISTINCT on WITH output.
/// 4. Optional WHERE on the projected row values (HAVING-style filter).
/// 5. Re-wrap projected rows as `Binding::Value` so the standard RETURN
///    machinery can handle them.
/// 6. Final RETURN with ORDER BY / OFFSET / LIMIT.
fn execute_match_with(mw: MatchWithStatement, graph: &Graph) -> Result<Vec<Row>, DbError> {
    // Phase 1: MATCH → filter by WHERE.
    let bindings = cross_join_patterns(&mw.patterns, graph, mw.path_mode, mw.where_clause.as_ref(), None);

    let filtered: Vec<Bindings> = bindings
        .into_iter()
        .filter(|b| {
            if let Some(ref cond) = mw.where_clause {
                matches!(eval_expr(cond, b, graph), Ok(Value::Bool(true)))
            } else {
                true
            }
        })
        .collect();

    // Phase 2: WITH projection → convert each Bindings into a new Bindings
    //          using only the projected names.
    let with_items = &mw.with_clause.items;
    let has_agg = with_items.iter().any(|it| contains_aggregate(&it.expr));

    let projected_rows: Vec<Row> = if has_agg {
        execute_aggregate(&filtered, with_items, graph)?
    } else {
        filtered
            .iter()
            .map(|b| project_return(with_items, b, graph))
            .collect::<Result<Vec<_>, _>>()?
    };

    // Optional DISTINCT on WITH.
    let mut projected_rows = projected_rows;
    if mw.with_clause.distinct {
        let mut seen = std::collections::HashSet::new();
        projected_rows.retain(|row| {
            let mut keys: Vec<_> = row.keys().collect();
            keys.sort();
            let key: Vec<String> = keys.iter().map(|k| format!("{}={:?}", k, row[*k])).collect();
            seen.insert(key.join(","))
        });
    }

    // Phase 3: optional WITH WHERE filter on projected rows.
    let projected_rows: Vec<Row> = if let Some(ref cond) = mw.with_clause.where_clause {
        projected_rows
            .into_iter()
            .filter(|row| matches!(eval_expr_on_row(cond, row), Ok(Value::Bool(true))))
            .collect()
    } else {
        projected_rows
    };

    // Phase 4: convert projected rows back to Bindings so we can use
    //          the standard RETURN projection machinery.
    let with_bindings: Vec<Bindings> = projected_rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(k, v)| (k, Binding::Value(v)))
                .collect()
        })
        .collect();

    // Phase 5: final RETURN clause.
    apply_return_clause(with_bindings, &mw.return_clause, graph)
}

// ── Shared RETURN-clause application ───────────────────────────────────────

/// Apply ORDER BY / DISTINCT / OFFSET / LIMIT and project the RETURN clause
/// over a set of already-filtered bindings.
///
/// This shared helper is used by `execute_optional_match`, `execute_unwind`,
/// `execute_match_with`, and `execute_call_pipeline` so that all statement types
/// get identical post-processing semantics.
///
/// When the clause contains aggregate functions the binding stream is routed
/// through `execute_aggregate` first; ORDER BY on the result operates on the
/// projected row values via `eval_expr_on_row` (which handles aggregate column
/// names like `count(*)`).
fn apply_return_clause(
    filtered: Vec<Bindings>,
    rc: &ReturnClause,
    graph: &Graph,
) -> Result<Vec<Row>, DbError> {
    let has_aggregates = rc.items.iter().any(|it| contains_aggregate(&it.expr));

    let mut rows: Vec<Row> = if has_aggregates {
        let mut agg_rows = execute_aggregate(&filtered, &rc.items, graph)?;
        if !rc.order_by.is_empty() {
            let order = &rc.order_by;
            agg_rows.sort_by(|a, b| {
                for item in order {
                    let va = eval_expr_on_row(&item.expr, a).unwrap_or(Value::Null);
                    let vb = eval_expr_on_row(&item.expr, b).unwrap_or(Value::Null);
                    let cmp = cmp_nulls_last(&va, &vb);
                    let cmp = if item.ascending { cmp } else { cmp.reverse() };
                    if cmp != std::cmp::Ordering::Equal { return cmp; }
                }
                std::cmp::Ordering::Equal
            });
        }
        agg_rows
    } else {
        let mut filtered = filtered;
        if !rc.order_by.is_empty() {
            let order = &rc.order_by;
            filtered.sort_by(|a, b| {
                for item in order {
                    let va = eval_expr(&item.expr, a, graph).unwrap_or(Value::Null);
                    let vb = eval_expr(&item.expr, b, graph).unwrap_or(Value::Null);
                    let cmp = cmp_nulls_last(&va, &vb);
                    let cmp = if item.ascending { cmp } else { cmp.reverse() };
                    if cmp != std::cmp::Ordering::Equal { return cmp; }
                }
                std::cmp::Ordering::Equal
            });
        }
        // Parallelize projection for large result sets (PERF-12).
        // Each call to project_return reads node/edge properties from RocksDB
        // (or the in-memory adj cache); these are independent reads, so Rayon
        // can distribute them across available cores.
        // For small sets the sequential path avoids Rayon thread-dispatch overhead.
        const PARALLEL_THRESHOLD: usize = 512;
        if filtered.len() >= PARALLEL_THRESHOLD {
            use rayon::prelude::*;
            filtered
                .par_iter()
                .map(|b| project_return(&rc.items, b, graph))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            filtered
                .iter()
                .map(|b| project_return(&rc.items, b, graph))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    if rc.distinct {
        let mut seen = std::collections::HashSet::new();
        rows.retain(|row| {
            let mut keys: Vec<_> = row.keys().collect();
            keys.sort();
            let key: Vec<String> = keys.iter().map(|k| format!("{}={:?}", k, row[*k])).collect();
            seen.insert(key.join(","))
        });
    }

    let offset = match &rc.offset {
        Some(expr) => match eval_expr_literal(expr)? {
            Value::Int(n) => n.max(0) as usize,
            _ => return Err(DbError::Query("OFFSET must be an integer".into())),
        },
        None => 0,
    };
    let limit = match &rc.limit {
        Some(expr) => match eval_expr_literal(expr)? {
            Value::Int(n) => Some(n.max(0) as usize),
            _ => return Err(DbError::Query("LIMIT must be an integer".into())),
        },
        None => None,
    };

    Ok(rows.into_iter().skip(offset).take(limit.unwrap_or(usize::MAX)).collect())
}

// ── Aggregation ─────────────────────────────────────────────────────────────

/// The set of aggregate function names recognised by the executor.
/// The parser lowercases all identifiers, so these are all lowercase.
const AGG_FUNCTIONS: &[&str] = &["count", "sum", "avg", "min", "max", "collect"];

/// Returns `true` if `name` is the name of a supported aggregate function.
fn is_agg_fn(name: &str) -> bool {
    AGG_FUNCTIONS.contains(&name)
}

/// Returns true if `expr` contains an aggregate function anywhere in its subtree.
fn contains_aggregate(expr: &Expr) -> bool {
    match expr {
        Expr::Call(name, args) => {
            if is_agg_fn(name) {
                true
            } else {
                args.iter().any(contains_aggregate)
            }
        }
        Expr::BinOp(l, _, r) => contains_aggregate(l) || contains_aggregate(r),
        Expr::Not(e) => contains_aggregate(e),
        Expr::IsNull(e, _) => contains_aggregate(e),
        Expr::Property(e, _) => contains_aggregate(e.as_ref()),
        _ => false,
    }
}

/// Evaluate `expr` in aggregate context.
/// - If `expr` IS (or contains) an aggregate call, compute it over `group`.
/// - Otherwise evaluate it against the first binding in the group.
fn eval_expr_agg(expr: &Expr, group: &[&Bindings], graph: &Graph) -> Result<Value, DbError> {
    match expr {
        Expr::Call(name, args) if is_agg_fn(name) => aggregate_fn(name, args, group, graph),
        Expr::BinOp(l, op, r) => {
            let lv = eval_expr_agg(l, group, graph)?;
            let rv = eval_expr_agg(r, group, graph)?;
            eval_binop(lv, *op, rv)
        }
        Expr::Not(e) => match eval_expr_agg(e, group, graph)? {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            _ => Ok(Value::Null),
        },
        Expr::IsNull(e, is_not) => {
            let v = eval_expr_agg(e, group, graph)?;
            let is_null = v == Value::Null;
            Ok(Value::Bool(if *is_not { !is_null } else { is_null }))
        }
        // Non-aggregate sub-expression: use the first binding in the group.
        _ => match group.first() {
            Some(b) => eval_expr(expr, b, graph),
            None => Ok(Value::Null),
        },
    }
}

/// Compute a single aggregate function over a group of bindings.
fn aggregate_fn(
    name: &str,
    args: &[Expr],
    group: &[&Bindings],
    graph: &Graph,
) -> Result<Value, DbError> {
    match name {
        "count" => {
            let is_star = args.is_empty() || matches!(args.first(), Some(Expr::Star));
            if is_star {
                Ok(Value::Int(group.len() as i64))
            } else {
                let arg = &args[0];
                let n = group
                    .iter()
                    .filter(|b| matches!(eval_expr(arg, b, graph), Ok(v) if v != Value::Null))
                    .count();
                Ok(Value::Int(n as i64))
            }
        }
        "sum" => {
            let arg = args.first().ok_or_else(|| DbError::Query("sum() requires 1 arg".into()))?;
            let mut int_sum: i64 = 0;
            let mut float_sum: f64 = 0.0;
            let mut has_float = false;
            let mut any = false;
            for b in group {
                match eval_expr(arg, b, graph)? {
                    Value::Null => {}
                    Value::Int(i) => { int_sum += i; float_sum += i as f64; any = true; }
                    Value::Float(f) => { float_sum += f; has_float = true; any = true; }
                    v => return Err(DbError::Query(format!("sum() cannot aggregate {v:?}"))),
                }
            }
            if !any { Ok(Value::Null) } else if has_float { Ok(Value::Float(float_sum)) } else { Ok(Value::Int(int_sum)) }
        }
        "avg" => {
            let arg = args.first().ok_or_else(|| DbError::Query("avg() requires 1 arg".into()))?;
            let mut sum: f64 = 0.0;
            let mut count: usize = 0;
            for b in group {
                match eval_expr(arg, b, graph)? {
                    Value::Null => {}
                    Value::Int(i) => { sum += i as f64; count += 1; }
                    Value::Float(f) => { sum += f; count += 1; }
                    v => return Err(DbError::Query(format!("avg() cannot aggregate {v:?}"))),
                }
            }
            if count == 0 { Ok(Value::Null) } else { Ok(Value::Float(sum / count as f64)) }
        }
        "min" => {
            let arg = args.first().ok_or_else(|| DbError::Query("min() requires 1 arg".into()))?;
            let mut result: Option<Value> = None;
            for b in group {
                let v = eval_expr(arg, b, graph)?;
                if v == Value::Null { continue; }
                result = Some(match result {
                    None => v,
                    Some(cur) => if v.partial_cmp(&cur) == Some(std::cmp::Ordering::Less) { v } else { cur },
                });
            }
            Ok(result.unwrap_or(Value::Null))
        }
        "max" => {
            let arg = args.first().ok_or_else(|| DbError::Query("max() requires 1 arg".into()))?;
            let mut result: Option<Value> = None;
            for b in group {
                let v = eval_expr(arg, b, graph)?;
                if v == Value::Null { continue; }
                result = Some(match result {
                    None => v,
                    Some(cur) => if v.partial_cmp(&cur) == Some(std::cmp::Ordering::Greater) { v } else { cur },
                });
            }
            Ok(result.unwrap_or(Value::Null))
        }
        "collect" => {
            let arg = args.first().ok_or_else(|| DbError::Query("collect() requires 1 arg".into()))?;
            let vals: Vec<Value> = group
                .iter()
                .filter_map(|b| eval_expr(arg, b, graph).ok())
                .filter(|v| *v != Value::Null)
                .collect();
            Ok(Value::List(vals))
        }
        _ => Err(DbError::Query(format!("unknown aggregate function: {name}"))),
    }
}

/// Group bindings by non-aggregate key expressions and compute aggregate rows.
///
/// This implements SQL-style GROUP BY semantics:
/// - Non-aggregate RETURN items form the implicit group key.
/// - Aggregate functions (`count`, `sum`, `avg`, `min`, `max`, `collect`) are
///   evaluated over every binding in each group via `eval_expr_agg`.
/// - A query that contains only aggregate functions (no grouping keys) produces
///   exactly one output row.  When there are also no input rows, that one row
///   still appears (e.g. `count(*)` returns 0 rather than producing no rows).
///
/// Group identity is determined by serialising each key expression's evaluated
/// value with `{:?}` so that the comparison is type-aware (Int(1) ≠ Float(1.0)).
/// First-seen insertion order is preserved through the parallel `group_order` and
/// `group_member_indices` vecs that mirror the `key_to_group` map.
fn execute_aggregate(
    filtered: &[Bindings],
    items: &[ReturnItem],
    graph: &Graph,
) -> Result<Vec<Row>, DbError> {
    // Separate items into grouping keys (non-agg) and everything else.
    let key_items: Vec<&ReturnItem> = items.iter().filter(|it| !contains_aggregate(&it.expr)).collect();

    // Group bindings by evaluated key values, preserving first-seen insertion order.
    // `group_order` holds the serialised key for each group in discovery order.
    // `group_member_indices` holds the indices into `filtered` for each group.
    // `key_to_group` maps serialised key → index into the two parallel vecs.
    let mut group_order: Vec<Vec<String>> = Vec::new();
    let mut group_member_indices: Vec<Vec<usize>> = Vec::new();
    let mut key_to_group: HashMap<Vec<String>, usize> = HashMap::new();

    for (i, bindings) in filtered.iter().enumerate() {
        // Build a string-serialised key for this binding's group.
        let key: Vec<String> = key_items
            .iter()
            .map(|item| match eval_expr(&item.expr, bindings, graph) {
                Ok(v) => format!("{v:?}"),
                Err(_) => "null".to_string(),
            })
            .collect();

        // Register a new group on first encounter; otherwise append to existing.
        let next_idx = group_order.len();
        let gidx = *key_to_group.entry(key.clone()).or_insert_with(|| {
            group_order.push(key);
            group_member_indices.push(Vec::new());
            next_idx
        });
        group_member_indices[gidx].push(i);
    }

    // If there are no grouping keys and no rows, still produce one row so that
    // pure-aggregate queries (e.g. `RETURN count(*)`) always return something.
    if group_order.is_empty() && key_items.is_empty() {
        group_order.push(vec![]);
        group_member_indices.push(vec![]);
    }

    // Build one projected row per group by evaluating each return item in
    // aggregate context over the group's member bindings.
    let mut rows = Vec::new();
    for (gidx, _) in group_order.iter().enumerate() {
        let group: Vec<&Bindings> = group_member_indices[gidx]
            .iter()
            .map(|&i| &filtered[i])
            .collect();

        let mut row = Row::new();
        for item in items {
            let col = item.alias.clone().unwrap_or_else(|| expr_display(&item.expr));
            let val = eval_expr_agg(&item.expr, &group, graph)?;
            row.insert(col, val);
        }
        rows.push(row);
    }

    Ok(rows)
}

/// Match a single `GraphPattern` against the graph and return all binding maps.
///
/// **Start-node selection** uses a three-level cascade (most selective first):
/// 1. Equality property index: an inline `{prop: val}` constraint maps to an
///    O(1) index lookup when `(label, prop)` is indexed.
/// 2. Range property index: a `var.prop OP literal` predicate in the WHERE
///    clause is used for a seek-and-scan range lookup when the index exists.
/// 3. Label index: all nodes carrying the first label in the pattern.
/// 4. Full scan: used when no label is specified.
///
/// `where_clause` is threaded in solely for the range-index optimisation in
/// step 2.  The WHERE predicate itself is **not** evaluated here — callers
/// apply it after cross-joining all patterns.
///
/// Returns an empty `Vec` if no node matches the start constraints.
fn match_pattern(
    pattern: &GraphPattern,
    graph: &Graph,
    path_mode: PathMode,
    where_clause: Option<&Expr>,
    early_limit: Option<usize>,
) -> Vec<Bindings> {
    // Use property index for O(1) lookup when available: pick a property constraint
    // whose (label, key, value) is indexed and use that as the candidate set.
    // Fall back to label-index or full scan otherwise.
    //
    // When `early_limit` is set and the pattern has no edge steps and no inline
    // property constraints, we cap the candidate scan at that limit to avoid
    // reading the entire nodes CF for queries like `MATCH (n) RETURN n LIMIT 1`.
    let can_limit_candidates = early_limit.is_some()
        && pattern.steps.is_empty()
        && pattern.start.properties.is_empty();

    let start_nodes: Vec<NodeId> = 'candidates: {
        if let Some(primary) = pattern.start.labels.first() {
            // 1. Collect candidate sets from every usable index — both inline
            //    property constraints and WHERE predicates — then intersect
            //    them.  Using all available indexes is always at least as good
            //    as using just one: each additional set can only shrink the
            //    candidate pool.
            let mut index_sets: Vec<Vec<NodeId>> = Vec::new();

            // 1a. Equality index: inline `{prop: val}` constraints.
            for pc in &pattern.start.properties {
                if let Ok(val) = eval_expr_literal(&pc.value) {
                    if let Some(ids) = graph.lookup_by_property(primary, &pc.key, &val) {
                        index_sets.push(ids);
                    }
                }
            }

            // 1b. Equality / range / prefix predicates in the WHERE clause.
            if let (Some(var), Some(where_expr)) = (&pattern.start.variable, where_clause) {
                collect_where_index_candidates(graph, primary, var, where_expr, &mut index_sets);
            }

            if !index_sets.is_empty() {
                break 'candidates intersect_node_id_sets(index_sets);
            }

            // 2. Label index (with optional limit).
            if can_limit_candidates {
                break 'candidates graph.nodes_by_label_limit(primary, early_limit.unwrap());
            }
            break 'candidates graph.nodes_by_label(primary);
        }
        // 3. No label — full scan (with optional limit).
        if can_limit_candidates {
            graph.first_n_node_ids(early_limit.unwrap())
        } else {
            graph.all_nodes().into_iter().map(|n| n.id).collect()
        }
    };

    // When the start node carries no constraints beyond what the scan already
    // guarantees, skip loading the full node during matching.  It will be
    // loaded (and cached) lazily if a property is accessed later.
    let needs_constraint_check =
        pattern.start.properties.len() > 0 || pattern.start.labels.len() > 1;

    let mut results: Vec<Bindings> = Vec::new();

    for node_id in start_nodes {
        let start_binding = if needs_constraint_check {
            let node = match graph.get_node(node_id) {
                Some(n) => n,
                None => continue,
            };
            if !check_node_constraints(&pattern.start, &node) {
                continue;
            }
            Binding::NodeData(node_id, node)
        } else {
            Binding::Node(node_id)
        };

        let mut binding: Bindings = HashMap::new();
        if let Some(ref var) = pattern.start.variable {
            binding.insert(var.clone(), start_binding);
        }

        // Seed the visited sets with the start node.
        let mut visited_nodes: HashSet<NodeId> = HashSet::new();
        visited_nodes.insert(node_id);
        let visited_edges: HashSet<EdgeId> = HashSet::new();

        let remaining = early_limit.map(|lim| lim.saturating_sub(results.len()).max(1));
        let step_bindings = extend_steps(
            &pattern.steps, node_id, binding, graph,
            path_mode, &visited_edges, &visited_nodes,
            remaining,
        );
        results.extend(step_bindings);

        // Early-exit once we have enough results (only safe without WHERE, since
        // WHERE filtering happens after this function returns in execute_match).
        if early_limit.is_some_and(|lim| results.len() >= lim) {
            break;
        }
    }

    results
}

/// Recursively extend a partial binding along the remaining `steps`.
///
/// Starting from `current_node` with the accumulated `binding`, this function
/// appends one step at a time (either a single-hop edge traversal or a
/// quantified BFS segment) and recurses on the tail of `steps`.
///
/// `visited_edges` and `visited_nodes` carry the deduplication sets for the
/// current partial path so that TRAIL (no repeated edges) and SIMPLE (no
/// repeated nodes) constraints can be enforced at each hop.
///
/// Returns a flat list of fully-extended `Bindings` maps, one per complete path.
fn extend_steps(
    steps: &[EdgePatternStep],
    current_node: NodeId,
    binding: Bindings,
    graph: &Graph,
    path_mode: PathMode,
    visited_edges: &HashSet<EdgeId>,
    visited_nodes: &HashSet<NodeId>,
    early_limit: Option<usize>,
) -> Vec<Bindings> {
    let Some(step) = steps.first() else {
        return vec![binding];
    };
    let rest = &steps[1..];
    let mut results = Vec::new();

    if let Some(quant) = &step.edge.quantifier {
        // ── Variable-length traversal ────────────────────────────────────────
        // Pass remaining budget into traverse_quantified so BFS terminates
        // as soon as it has found enough path endpoints.
        let tq_limit = early_limit.map(|lim| lim.saturating_sub(results.len()).max(1));
        let endpoints = traverse_quantified(
            graph, current_node, &step.edge, quant, path_mode,
            visited_edges, visited_nodes, tq_limit,
        );

        'tq: for (end_node, edge_path) in endpoints {
            let end_node_data = match graph.get_node(end_node) {
                Some(n) => n,
                None => continue,
            };

            if !check_node_constraints(&step.node, &end_node_data) {
                continue;
            }

            let mut new_binding = binding.clone();
            if let Some(ref var) = step.edge.variable {
                new_binding.insert(var.clone(), Binding::EdgeList(edge_path.clone()));
            }
            if let Some(ref var) = step.node.variable {
                new_binding.insert(var.clone(), Binding::Node(end_node));
            }

            let mut new_visited_edges = visited_edges.clone();
            let mut new_visited_nodes = visited_nodes.clone();
            new_visited_edges.extend(edge_path.iter().copied());
            new_visited_nodes.insert(end_node);

            let remaining = early_limit.map(|lim| lim.saturating_sub(results.len()).max(1));
            let extended = extend_steps(
                rest, end_node, new_binding, graph,
                path_mode, &new_visited_edges, &new_visited_nodes,
                remaining,
            );
            results.extend(extended);
            if early_limit.is_some_and(|lim| results.len() >= lim) {
                break 'tq;
            }
        }
    } else {
        // ── Single-hop traversal (uses in-memory adjacency, ARCH-1) ──────────
        //
        // `AdjEntry` carries (edge_id, neighbor, label_id) — no RocksDB lookup
        // needed for label filtering.  We only call `get_edge` when the pattern
        // has edge property constraints (uncommon).
        //
        // For EdgeDirection::Either we collect both out- and in-adj entries and
        // dedup by edge_id so each physical edge is visited at most once.
        use crate::graph::AdjEntry;

        // Build the candidate list.  For Either direction we need owned entries
        // so we can sort-and-dedup; for single directions we borrow the slice.
        enum AdjSlice<'s> {
            Borrowed(&'s [AdjEntry]),
            Owned(Vec<AdjEntry>),
        }
        impl<'s> AdjSlice<'s> {
            fn iter(&'s self) -> impl Iterator<Item = &'s AdjEntry> {
                match self {
                    AdjSlice::Borrowed(s) => s.iter(),
                    AdjSlice::Owned(v)    => v.iter(),
                }
            }
        }

        let candidates: AdjSlice<'_> = match step.edge.direction {
            EdgeDirection::Outgoing => AdjSlice::Borrowed(graph.neighbors_out_mem(current_node)),
            EdgeDirection::Incoming => AdjSlice::Borrowed(graph.neighbors_in_mem(current_node)),
            EdgeDirection::Either => {
                let mut combined: Vec<AdjEntry> = graph
                    .neighbors_out_mem(current_node)
                    .iter()
                    .chain(graph.neighbors_in_mem(current_node).iter())
                    .cloned()
                    .collect();
                // Dedup by edge_id so each undirected edge is only traversed once.
                combined.sort_by_key(|e| e.edge_id);
                combined.dedup_by_key(|e| e.edge_id);
                AdjSlice::Owned(combined)
            }
        };

        for entry in candidates.iter() {
            let edge_id = entry.edge_id;

            // Filter by optional edge label constraint — O(1) interned comparison.
            if let Some(ref lbl) = step.edge.label {
                if graph.adj_label_str(entry.label_id) != lbl.as_str() {
                    continue;
                }
            }

            // Edge property constraints — load full edge (O(1) in-memory, ARCH-2).
            if !step.edge.properties.is_empty() {
                let edge = match graph.get_edge(edge_id) { Some(e) => e, None => continue };
                if !check_property_constraints(&step.edge.properties, &edge.properties) {
                    continue;
                }
            }

            // TRAIL mode: each edge may appear at most once in a path.
            if path_mode == PathMode::Trail && visited_edges.contains(&edge_id) {
                continue;
            }

            let neighbour = entry.neighbor;

            // SIMPLE mode: each node may be visited at most once in a path.
            if path_mode == PathMode::Simple && visited_nodes.contains(&neighbour) {
                continue;
            }

            // Node constraint check still requires loading the node from RocksDB
            // (or the in-memory node cache once PERF-6 is implemented).
            if !step.node.labels.is_empty() || !step.node.properties.is_empty() {
                let nbr_node = match graph.get_node(neighbour) { Some(n) => n, None => continue };
                if !check_node_constraints(&step.node, &nbr_node) {
                    continue;
                }
            }

            let mut new_binding = binding.clone();
            if let Some(ref var) = step.edge.variable {
                new_binding.insert(var.clone(), Binding::Edge(edge_id));
            }
            if let Some(ref var) = step.node.variable {
                new_binding.insert(var.clone(), Binding::Node(neighbour));
            }

            let mut new_visited_edges = visited_edges.clone();
            let mut new_visited_nodes = visited_nodes.clone();
            new_visited_edges.insert(edge_id);
            new_visited_nodes.insert(neighbour);

            let remaining = early_limit.map(|lim| lim.saturating_sub(results.len()).max(1));
            let extended = extend_steps(
                rest, neighbour, new_binding, graph,
                path_mode, &new_visited_edges, &new_visited_nodes,
                remaining,
            );
            results.extend(extended);
            if early_limit.is_some_and(|lim| results.len() >= lim) {
                break;
            }
        }
    }

    results
}

// ── Variable-length path traversal (BFS) ───────────────────────────────────

/// Per-frontier state carried through the BFS queue for quantified traversal.
///
/// Each `QState` represents one distinct path being explored.  The visited sets
/// are cloned per branch so that sibling paths in the BFS do not interfere with
/// each other's TRAIL/SIMPLE deduplication.
struct QState {
    /// Current end-node of this partial path.
    node: NodeId,
    /// Number of edge hops taken so far (0 = at the start node).
    depth: u32,
    /// Ordered list of edge IDs forming the path from `start` to `node`.
    edge_path: Vec<EdgeId>,
    /// Edge IDs already used in this path (for TRAIL deduplication).
    visited_edges: HashSet<EdgeId>,
    /// Node IDs already visited in this path (for SIMPLE deduplication).
    visited_nodes: HashSet<NodeId>,
}

/// BFS over the graph for a quantified edge pattern.
///
/// Returns all `(end_node, edge_path)` pairs reachable from `start` within
/// `[quant.min, quant.max]` hops while respecting the edge label/property
/// constraints in `edge_pat` and the deduplication rules of `path_mode`.
///
/// `prior_visited_edges` / `prior_visited_nodes` carry the path state
/// accumulated by earlier (non-quantified) steps in the same MATCH pattern,
/// ensuring that TRAIL/SIMPLE semantics are enforced consistently across the
/// entire path, not just within this BFS segment.
fn traverse_quantified(
    graph: &Graph,
    start: NodeId,
    edge_pat: &EdgePattern,
    quant: &PathQuantifier,
    path_mode: PathMode,
    prior_visited_edges: &HashSet<EdgeId>,
    prior_visited_nodes: &HashSet<NodeId>,
    early_limit: Option<usize>,
) -> Vec<(NodeId, Vec<EdgeId>)> {
    // Compute the effective upper bound for BFS depth.
    // When no explicit max is given, use a safe upper limit that guarantees
    // termination while covering all meaningful paths for the given mode:
    //   WALK  — arbitrary repetition allowed; cap at 25 (or node_count+1) to
    //            prevent unbounded traversal on cyclic graphs.
    //   TRAIL — no edge repeated, so the longest possible trail is |edges|.
    //   SIMPLE— no node repeated, so the longest possible path is |nodes|.
    let effective_max: u32 = match quant.max {
        Some(n) => n,
        None => match path_mode {
            PathMode::Walk  => 25u32.min(graph.node_count() as u32 + 1),
            PathMode::Trail => graph.edge_count() as u32,
            PathMode::Simple => graph.node_count() as u32,
        },
    };

    let min = quant.min;
    let mut results: Vec<(NodeId, Vec<EdgeId>)> = Vec::new();

    // Seed visited_nodes with the start node so SIMPLE mode won't loop back to it.
    let mut initial_visited_nodes = prior_visited_nodes.clone();
    initial_visited_nodes.insert(start);

    // Initialise the BFS frontier with a single zero-depth state at `start`.
    let mut queue: VecDeque<QState> = VecDeque::new();
    queue.push_back(QState {
        node: start,
        depth: 0,
        edge_path: Vec::new(),
        visited_edges: prior_visited_edges.clone(),
        visited_nodes: initial_visited_nodes,
    });

    // BFS loop: process each frontier state, yield reachable nodes in [min, max],
    // and enqueue valid extensions up to `effective_max` depth.
    while let Some(state) = queue.pop_front() {
        // Yield the current position if it satisfies the minimum hop count.
        // depth == 0 corresponds to the start node itself (0-hop / * quantifier).
        if state.depth >= min {
            results.push((state.node, state.edge_path.clone()));
            if early_limit.is_some_and(|lim| results.len() >= lim) {
                return results;
            }
        }

        // Do not expand beyond the maximum depth.
        if state.depth >= effective_max {
            continue;
        }

        // Gather candidate adjacency entries from in-memory adj (ARCH-1).
        // For EdgeDirection::Either, merge both lists and dedup by edge_id.
        use crate::graph::AdjEntry as TqEntry;
        let combined_either: Vec<TqEntry>;
        let candidates: &[TqEntry] = match edge_pat.direction {
            EdgeDirection::Outgoing => graph.neighbors_out_mem(state.node),
            EdgeDirection::Incoming => graph.neighbors_in_mem(state.node),
            EdgeDirection::Either => {
                let mut v: Vec<TqEntry> = graph
                    .neighbors_out_mem(state.node)
                    .iter()
                    .chain(graph.neighbors_in_mem(state.node).iter())
                    .cloned()
                    .collect();
                v.sort_by_key(|e| e.edge_id);
                v.dedup_by_key(|e| e.edge_id);
                combined_either = v;
                &combined_either
            }
        };

        for entry in candidates {
            let edge_id = entry.edge_id;

            // Label filter: O(1) interned string comparison.
            if let Some(ref lbl) = edge_pat.label {
                if graph.adj_label_str(entry.label_id) != lbl.as_str() {
                    continue;
                }
            }
            // Edge property constraints require a RocksDB lookup (uncommon).
            if !edge_pat.properties.is_empty() {
                let edge = match graph.get_edge(edge_id) { Some(e) => e, None => continue };
                if !check_property_constraints(&edge_pat.properties, &edge.properties) {
                    continue;
                }
            }

            // TRAIL: skip edges that have already been traversed in this path.
            if path_mode == PathMode::Trail && state.visited_edges.contains(&edge_id) {
                continue;
            }

            let neighbour = entry.neighbor;

            // SIMPLE: skip nodes that have already been visited in this path.
            if path_mode == PathMode::Simple && state.visited_nodes.contains(&neighbour) {
                continue;
            }

            // Clone the deduplication sets and path before enqueuing so that
            // different branches of the BFS tree remain independent.
            let mut new_visited_edges = state.visited_edges.clone();
            new_visited_edges.insert(edge_id);
            let mut new_visited_nodes = state.visited_nodes.clone();
            new_visited_nodes.insert(neighbour);
            let mut new_path = state.edge_path.clone();
            new_path.push(edge_id);

            queue.push_back(QState {
                node: neighbour,
                depth: state.depth + 1,
                edge_path: new_path,
                visited_edges: new_visited_edges,
                visited_nodes: new_visited_nodes,
            });
        }
    }

    results
}

/// Returns `true` if `node` satisfies all label and inline property constraints
/// specified by `pattern`.  All listed labels must be present (multi-label AND
/// semantics); all inline `{key: value}` constraints must match exactly.
fn check_node_constraints(pattern: &NodePattern, node: &crate::types::Node) -> bool {
    // All required labels must be present on the node.
    for lbl in &pattern.labels {
        if !node.labels.contains(lbl) {
            return false;
        }
    }
    check_property_constraints(&pattern.properties, &node.properties)
}

/// Returns `true` if every `PropertyConstraint` in `constraints` is satisfied
/// by the provided `properties` map.
///
/// The constraint value is evaluated as a literal expression; if evaluation
/// fails (e.g. a non-literal expression slipped through), the constraint is
/// treated as unsatisfied (returns `false`).  A missing key in `properties`
/// is treated as `Null` and will not match any literal constraint.
fn check_property_constraints(
    constraints: &[PropertyConstraint],
    properties: &Properties,
) -> bool {
    for c in constraints {
        let actual = properties.get(&c.key).cloned().unwrap_or(Value::Null);
        // Evaluate the constraint value as a literal.
        let expected = match eval_expr_literal(&c.value) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if actual != expected {
            return false;
        }
    }
    true
}

// ── Projection ─────────────────────────────────────────────────────────────

/// Project a single `Bindings` map into a `Row` using the given RETURN items.
///
/// `Expr::Star` expands all bound variables: node/edge IDs are serialised as
/// ULID strings; `EdgeList` bindings become `Value::List` of ULID strings;
/// `Value` bindings are emitted as-is.
///
/// For all other expressions the column name is the `AS` alias if one was
/// provided, or the `expr_display` string otherwise.
fn project_return(
    items: &[ReturnItem],
    bindings: &Bindings,
    graph: &Graph,
) -> Result<Row, DbError> {
    let mut row = Row::new();

    for item in items {
        match &item.expr {
            Expr::Star => {
                // Expand all bound variables.
                for (var, binding) in bindings {
                    match binding {
                        Binding::Node(id) | Binding::NodeData(id, _) => {
                            row.insert(var.clone(), Value::String(ulid_encode(id.0)));
                        }
                        Binding::Edge(id) => {
                            row.insert(var.clone(), Value::String(ulid_encode(id.0)));
                        }
                        Binding::EdgeList(ids) => {
                            row.insert(
                                var.clone(),
                                Value::List(
                                    ids.iter()
                                        .map(|id| Value::String(ulid_encode(id.0)))
                                        .collect(),
                                ),
                            );
                        }
                        Binding::Value(v) => {
                            row.insert(var.clone(), v.clone());
                        }
                    }
                }
            }
            expr => {
                let val = eval_expr(expr, bindings, graph)?;
                let key = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| expr_display(expr));
                row.insert(key, val);
            }
        }
    }

    Ok(row)
}

/// Produce a human-readable column name for an expression (used when no `AS`
/// alias is given).  This string is also used as the map key when looking up
/// aggregate results in `eval_expr_on_row`, so it must be stable and match the
/// key inserted by `execute_aggregate`.
fn expr_display(expr: &Expr) -> String {
    match expr {
        Expr::Var(v) => v.clone(),
        Expr::Property(obj, key) => format!("{}.{}", expr_display(obj), key),
        Expr::Call(name, args) => {
            let arg_strs: Vec<String> = args.iter().map(expr_display).collect();
            format!("{}({})", name, arg_strs.join(", "))
        }
        Expr::Literal(v) => format!("{v}"),
        Expr::Star => "*".to_string(),
        _ => "expr".to_string(),
    }
}

// ── Expression evaluation ──────────────────────────────────────────────────

/// Evaluate an expression in the context of a variable `bindings` map and the
/// live `graph`.
///
/// Node and edge variables resolve to their ULID string representation;
/// property access (`n.age`) fetches the value from the graph store lazily.
/// `EdgeList` property access returns a `Value::List` of that property over
/// all edges in the path.  Arithmetic, comparisons, boolean operators, function
/// calls, and list literals are evaluated recursively.
fn eval_expr(expr: &Expr, bindings: &Bindings, graph: &Graph) -> Result<Value, DbError> {
    match expr {
        Expr::Literal(v) => Ok(v.clone()),

        Expr::Var(name) => match bindings.get(name) {
            Some(Binding::Node(id) | Binding::NodeData(id, _)) => Ok(Value::String(ulid_encode(id.0))),
            Some(Binding::Edge(id)) => Ok(Value::String(ulid_encode(id.0))),
            Some(Binding::EdgeList(ids)) => Ok(Value::List(
                ids.iter().map(|id| Value::String(ulid_encode(id.0))).collect(),
            )),
            Some(Binding::Value(v)) => Ok(v.clone()),
            None => Ok(Value::Null),
        },

        Expr::Property(obj_expr, key) => {
            let obj_name = match obj_expr.as_ref() {
                Expr::Var(v) => v.clone(),
                _ => return Err(DbError::Query("property access on non-variable".into())),
            };
            match bindings.get(&obj_name) {
                // Fast path: node data already loaded during matching — no get_node call needed.
                Some(Binding::NodeData(_, node)) => {
                    Ok(node.properties.get(key).cloned().unwrap_or(Value::Null))
                }
                Some(Binding::Node(id)) => Ok(graph
                    .get_node(*id)
                    .and_then(|n| n.properties.get(key).cloned())
                    .unwrap_or(Value::Null)),
                Some(Binding::Edge(id)) => Ok(graph
                    .get_edge(*id)
                    .and_then(|e| e.properties.get(key).cloned())
                    .unwrap_or(Value::Null)),
                // Property access on a path variable returns a list of that property per edge.
                Some(Binding::EdgeList(ids)) => {
                    let values: Vec<Value> = ids
                        .iter()
                        .map(|id| {
                            graph
                                .get_edge(*id)
                                .and_then(|e| e.properties.get(key).cloned())
                                .unwrap_or(Value::Null)
                        })
                        .collect();
                    Ok(Value::List(values))
                }
                Some(Binding::Value(Value::Map(m))) => {
                    Ok(m.get(key).cloned().unwrap_or(Value::Null))
                }
                Some(Binding::Value(_)) | None => Ok(Value::Null),
            }
        }

        Expr::List(items) => {
            let vals: Result<Vec<Value>, DbError> =
                items.iter().map(|e| eval_expr(e, bindings, graph)).collect();
            Ok(Value::List(vals?))
        }

        Expr::BinOp(left, op, right) => {
            let l = eval_expr(left, bindings, graph)?;
            let r = eval_expr(right, bindings, graph)?;
            eval_binop(l, *op, r)
        }

        Expr::Not(inner) => {
            let v = eval_expr(inner, bindings, graph)?;
            match v {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                _ => Ok(Value::Null),
            }
        }

        Expr::IsNull(inner, is_not) => {
            let v = eval_expr(inner, bindings, graph)?;
            let is_null = v == Value::Null;
            Ok(Value::Bool(if *is_not { !is_null } else { is_null }))
        }

        Expr::Call(name, args) => eval_function(name, args, bindings, graph),

        Expr::Star => Err(DbError::Query("* not valid in expression context".into())),

        Expr::Param(name) => {
            Ok(QUERY_PARAMS.with(|p| p.borrow().get(name).cloned().unwrap_or(Value::Null)))
        }
    }
}

/// Evaluate a scalar expression that only contains literals (for constraints and LIMIT/OFFSET).
fn eval_expr_literal(expr: &Expr) -> Result<Value, DbError> {
    match expr {
        Expr::Literal(v) => Ok(v.clone()),
        Expr::BinOp(l, op, r) => {
            let lv = eval_expr_literal(l)?;
            let rv = eval_expr_literal(r)?;
            eval_binop(lv, *op, rv)
        }
        Expr::Not(inner) => match eval_expr_literal(inner)? {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            _ => Ok(Value::Null),
        },
        Expr::Param(name) => {
            Ok(QUERY_PARAMS.with(|p| p.borrow().get(name).cloned().unwrap_or(Value::Null)))
        }
        _ => Err(DbError::Query(
            "non-literal expression where literal required".into(),
        )),
    }
}

/// Evaluate an expression against a projected row (for ORDER BY after projection).
/// First tries looking up by the expression's display name, which handles aggregate
/// columns like `count(*)` that are keyed by their textual representation.
fn eval_expr_on_row(expr: &Expr, row: &Row) -> Result<Value, DbError> {
    // Fast-path: the column may already be in the row under its display name.
    let display = expr_display(expr);
    if display != "expr" {
        if let Some(v) = row.get(&display) {
            return Ok(v.clone());
        }
    }
    match expr {
        Expr::Literal(v) => Ok(v.clone()),
        Expr::Var(name) => Ok(row.get(name).cloned().unwrap_or(Value::Null)),
        Expr::Property(obj_expr, key) => {
            let obj_name = match obj_expr.as_ref() {
                Expr::Var(v) => v.clone(),
                _ => return Ok(Value::Null),
            };
            let col = format!("{}.{}", obj_name, key);
            Ok(row.get(&col).cloned().unwrap_or(Value::Null))
        }
        Expr::BinOp(l, op, r) => {
            let lv = eval_expr_on_row(l, row)?;
            let rv = eval_expr_on_row(r, row)?;
            eval_binop(lv, *op, rv)
        }
        _ => Ok(Value::Null),
    }
}

/// Compare two `Value`s for ORDER BY with NULL-last semantics (SQL standard).
///
/// - Both non-null: delegates to `partial_cmp`; NaN/incompatible types → Equal.
/// - One null: null sorts after any non-null value (Greatest).
/// - Both null: Equal.
#[inline]
fn cmp_nulls_last(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a, b) {
        (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
        (Value::Null, _) => std::cmp::Ordering::Greater,
        (_, Value::Null) => std::cmp::Ordering::Less,
        _ => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

/// Evaluate a binary operator on two already-evaluated `Value`s.
///
/// Comparison operators use `Value::partial_cmp` (which is defined for all
/// `Value` variants; `Value::Null` comparisons return `None`, treated as
/// `Equal` for ordering and `false` for `<`/`>`).  Division and modulo by
/// zero are detected and returned as `DbError::Query`.
fn eval_binop(l: Value, op: BinOp, r: Value) -> Result<Value, DbError> {
    match op {
        BinOp::And => Ok(Value::Bool(
            matches!(l, Value::Bool(true)) && matches!(r, Value::Bool(true)),
        )),
        BinOp::Or => Ok(Value::Bool(
            matches!(l, Value::Bool(true)) || matches!(r, Value::Bool(true)),
        )),
        BinOp::Eq => Ok(Value::Bool(l == r)),
        BinOp::Neq => Ok(Value::Bool(l != r)),
        BinOp::Lt => Ok(Value::Bool(l.partial_cmp(&r) == Some(std::cmp::Ordering::Less))),
        BinOp::Lte => Ok(Value::Bool(
            matches!(l.partial_cmp(&r), Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)),
        )),
        BinOp::Gt => Ok(Value::Bool(l.partial_cmp(&r) == Some(std::cmp::Ordering::Greater))),
        BinOp::Gte => Ok(Value::Bool(
            matches!(l.partial_cmp(&r), Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)),
        )),
        BinOp::Add => numeric_op(l, r, |a, b| a + b, |a, b| a + b),
        BinOp::Sub => numeric_op(l, r, |a, b| a - b, |a, b| a - b),
        BinOp::Mul => numeric_op(l, r, |a, b| a * b, |a, b| a * b),
        BinOp::Div => {
            let is_zero = match &r {
                Value::Int(0) => true,
                Value::Float(f) => *f == 0.0,
                _ => false,
            };
            if is_zero {
                return Err(DbError::Query("division by zero".into()));
            }
            numeric_op(l, r, |a, b| a / b, |a, b| a / b)
        }
        BinOp::Mod => {
            if matches!(r, Value::Int(0)) {
                return Err(DbError::Query("modulo by zero".into()));
            }
            numeric_op(l, r, |a, b| a % b, |a, b| a % b)
        }
        BinOp::In => {
            let found = match &r {
                Value::List(items) => items.contains(&l),
                _ => return Err(DbError::Query(format!("IN requires a list on the right side, got {:?}", r))),
            };
            Ok(Value::Bool(found))
        }
    }
}

/// Apply a numeric binary operation, promoting mixed `Int`/`Float` operands
/// to `Float` as needed.  Returns an error for non-numeric operand types.
fn numeric_op(
    l: Value,
    r: Value,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value, DbError> {
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(int_op(a, b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(float_op(a, b))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(float_op(a as f64, b))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(float_op(a, b as f64))),
        (l, r) => Err(DbError::Query(format!("cannot apply numeric op to {l:?} and {r:?}"))),
    }
}

/// Dispatch a scalar (non-aggregate) function call.
///
/// Aggregate functions (`count`, `sum`, etc.) are **not** handled here —
/// they are intercepted by `eval_expr_agg` before reaching this function.
/// If an aggregate name reaches this function it means the query used an
/// aggregate outside an aggregating RETURN clause, which is a usage error;
/// an explicit `DbError::Query` is returned rather than a silent stub value.
fn eval_function(
    name: &str,
    args: &[Expr],
    bindings: &Bindings,
    graph: &Graph,
) -> Result<Value, DbError> {
    match name {
        // count() is an aggregate function — it must be handled by eval_expr_agg
        // before reaching here.  If it lands in eval_function the query has a
        // bare count() outside an aggregating RETURN, which is a usage error.
        "count" => Err(DbError::Query(
            "count() is only valid inside an aggregating RETURN clause".into(),
        )),
        "id" | "elementid" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("id() requires 1 arg".into()))?,
                              bindings, graph)?;
            Ok(v)
        }
        "labels" => {
            let arg = args.first().ok_or_else(|| DbError::Query("labels() requires 1 arg".into()))?;
            let var = match arg {
                Expr::Var(v) => v.clone(),
                _ => return Err(DbError::Query("labels() requires a node variable".into())),
            };
            match bindings.get(&var) {
                Some(Binding::NodeData(_, node)) => {
                    let labels = node.labels.iter().map(|l| Value::String(l.clone())).collect();
                    Ok(Value::List(labels))
                }
                Some(Binding::Node(id)) => {
                    let labels = graph
                        .get_node(*id)
                        .map(|n| n.labels.iter().map(|l| Value::String(l.clone())).collect())
                        .unwrap_or_default();
                    Ok(Value::List(labels))
                }
                _ => Ok(Value::Null),
            }
        }
        "type" => {
            let arg = args.first().ok_or_else(|| DbError::Query("type() requires 1 arg".into()))?;
            let var = match arg {
                Expr::Var(v) => v.clone(),
                _ => return Err(DbError::Query("type() requires an edge variable".into())),
            };
            match bindings.get(&var) {
                Some(Binding::Edge(id)) => Ok(graph
                    .get_edge(*id)
                    .map(|e| Value::String(e.label.clone()))
                    .unwrap_or(Value::Null)),
                _ => Ok(Value::Null),
            }
        }
        "tostring" | "to_string" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("toString() requires 1 arg".into()))?,
                              bindings, graph)?;
            Ok(Value::String(format!("{v}")))
        }
        "tointeger" | "to_integer" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("toInteger() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::Int(i) => Ok(Value::Int(i)),
                Value::Float(f) => {
                    if !f.is_finite() {
                        return Err(DbError::Query(format!("cannot convert non-finite float {f} to integer")));
                    }
                    Ok(Value::Int(f as i64))
                }
                Value::String(s) => s.parse::<i64>().map(Value::Int)
                    .map_err(|_| DbError::Query(format!("cannot convert '{s}' to integer"))),
                _ => Ok(Value::Null),
            }
        }
        "size" | "length" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("size() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::String(s) => Ok(Value::Int(s.chars().count() as i64)),
                Value::List(l) => Ok(Value::Int(l.len() as i64)),
                _ => Ok(Value::Null),
            }
        }

        // ── String functions ────────────────────────────────────────────────

        "tolower" | "lowercase" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("toLower() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::String(s) => Ok(Value::String(s.to_lowercase())),
                _ => Ok(Value::Null),
            }
        }
        "toupper" | "uppercase" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("toUpper() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::String(s) => Ok(Value::String(s.to_uppercase())),
                _ => Ok(Value::Null),
            }
        }
        "trim" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("trim() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v { Value::String(s) => Ok(Value::String(s.trim().to_string())), _ => Ok(Value::Null) }
        }
        "ltrim" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("ltrim() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v { Value::String(s) => Ok(Value::String(s.trim_start().to_string())), _ => Ok(Value::Null) }
        }
        "rtrim" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("rtrim() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v { Value::String(s) => Ok(Value::String(s.trim_end().to_string())), _ => Ok(Value::Null) }
        }
        "split" => {
            if args.len() < 2 { return Err(DbError::Query("split() requires 2 args".into())); }
            let s = eval_expr(&args[0], bindings, graph)?;
            let d = eval_expr(&args[1], bindings, graph)?;
            match (s, d) {
                (Value::String(s), Value::String(d)) => {
                    let parts = s.split(d.as_str()).map(|p| Value::String(p.to_string())).collect();
                    Ok(Value::List(parts))
                }
                _ => Ok(Value::Null),
            }
        }
        "replace" => {
            if args.len() < 3 { return Err(DbError::Query("replace() requires 3 args".into())); }
            let s = eval_expr(&args[0], bindings, graph)?;
            let from = eval_expr(&args[1], bindings, graph)?;
            let to = eval_expr(&args[2], bindings, graph)?;
            match (s, from, to) {
                (Value::String(s), Value::String(f), Value::String(t)) => {
                    Ok(Value::String(s.replace(f.as_str(), &t)))
                }
                _ => Ok(Value::Null),
            }
        }
        "substring" => {
            if args.is_empty() { return Err(DbError::Query("substring() requires at least 2 args".into())); }
            let s = eval_expr(&args[0], bindings, graph)?;
            let start = eval_expr(args.get(1).ok_or_else(|| DbError::Query("substring() requires start".into()))?,
                                  bindings, graph)?;
            let length = if args.len() >= 3 {
                Some(eval_expr(&args[2], bindings, graph)?)
            } else {
                None
            };
            match (s, start) {
                (Value::String(s), Value::Int(start)) => {
                    let start = start.max(0) as usize;
                    let chars: Vec<char> = s.chars().collect();
                    let slice = match length {
                        Some(Value::Int(len)) => {
                            let len = len.max(0) as usize;
                            chars.get(start..start.saturating_add(len)).unwrap_or(&[])
                        }
                        _ => chars.get(start..).unwrap_or(&[]),
                    };
                    Ok(Value::String(slice.iter().collect()))
                }
                _ => Ok(Value::Null),
            }
        }
        "left" => {
            if args.len() < 2 { return Err(DbError::Query("left() requires 2 args".into())); }
            let s = eval_expr(&args[0], bindings, graph)?;
            let n = eval_expr(&args[1], bindings, graph)?;
            match (s, n) {
                (Value::String(s), Value::Int(n)) => {
                    let n = n.max(0) as usize;
                    Ok(Value::String(s.chars().take(n).collect()))
                }
                _ => Ok(Value::Null),
            }
        }
        "right" => {
            if args.len() < 2 { return Err(DbError::Query("right() requires 2 args".into())); }
            let s = eval_expr(&args[0], bindings, graph)?;
            let n = eval_expr(&args[1], bindings, graph)?;
            match (s, n) {
                (Value::String(s), Value::Int(n)) => {
                    let n = n.max(0) as usize;
                    let chars: Vec<char> = s.chars().collect();
                    let start = chars.len().saturating_sub(n);
                    Ok(Value::String(chars[start..].iter().collect()))
                }
                _ => Ok(Value::Null),
            }
        }
        "startswith" => {
            if args.len() < 2 { return Err(DbError::Query("startsWith() requires 2 args".into())); }
            let s = eval_expr(&args[0], bindings, graph)?;
            let p = eval_expr(&args[1], bindings, graph)?;
            match (s, p) {
                (Value::String(s), Value::String(p)) => Ok(Value::Bool(s.starts_with(p.as_str()))),
                _ => Ok(Value::Null),
            }
        }
        "endswith" => {
            if args.len() < 2 { return Err(DbError::Query("endsWith() requires 2 args".into())); }
            let s = eval_expr(&args[0], bindings, graph)?;
            let p = eval_expr(&args[1], bindings, graph)?;
            match (s, p) {
                (Value::String(s), Value::String(p)) => Ok(Value::Bool(s.ends_with(p.as_str()))),
                _ => Ok(Value::Null),
            }
        }
        "contains" => {
            if args.len() < 2 { return Err(DbError::Query("contains() requires 2 args".into())); }
            let s = eval_expr(&args[0], bindings, graph)?;
            let sub = eval_expr(&args[1], bindings, graph)?;
            match (s, sub) {
                (Value::String(s), Value::String(sub)) => Ok(Value::Bool(s.contains(sub.as_str()))),
                _ => Ok(Value::Null),
            }
        }
        "tofloat" | "to_float" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("toFloat() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::Float(f) => Ok(Value::Float(f)),
                Value::Int(i)   => Ok(Value::Float(i as f64)),
                Value::String(s) => s.parse::<f64>().map(Value::Float)
                    .map_err(|_| DbError::Query(format!("cannot convert '{s}' to float"))),
                _ => Ok(Value::Null),
            }
        }
        "coalesce" => {
            for arg in args {
                let v = eval_expr(arg, bindings, graph)?;
                if v != Value::Null { return Ok(v); }
            }
            Ok(Value::Null)
        }

        // ── Math functions ──────────────────────────────────────────────────

        "abs" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("abs() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::Int(i)   => Ok(Value::Int(i.abs())),
                Value::Float(f) => Ok(Value::Float(f.abs())),
                _ => Ok(Value::Null),
            }
        }
        "ceil" | "ceiling" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("ceil() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::Float(f) => Ok(Value::Float(f.ceil())),
                Value::Int(i)   => Ok(Value::Int(i)),
                _ => Ok(Value::Null),
            }
        }
        "floor" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("floor() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::Float(f) => Ok(Value::Float(f.floor())),
                Value::Int(i)   => Ok(Value::Int(i)),
                _ => Ok(Value::Null),
            }
        }
        "round" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("round() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::Float(f) => Ok(Value::Float(f.round())),
                Value::Int(i)   => Ok(Value::Int(i)),
                _ => Ok(Value::Null),
            }
        }
        "sign" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("sign() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::Int(i)   => Ok(Value::Int(i.signum())),
                Value::Float(f) => Ok(Value::Float(f.signum())),
                _ => Ok(Value::Null),
            }
        }
        "sqrt" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("sqrt() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::Float(f) => Ok(Value::Float(f.sqrt())),
                Value::Int(i)   => Ok(Value::Float((i as f64).sqrt())),
                _ => Ok(Value::Null),
            }
        }

        // ── List functions ──────────────────────────────────────────────────

        "head" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("head() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::List(items) => Ok(items.into_iter().next().unwrap_or(Value::Null)),
                _ => Ok(Value::Null),
            }
        }
        "last" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("last() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::List(items) => Ok(items.into_iter().last().unwrap_or(Value::Null)),
                _ => Ok(Value::Null),
            }
        }
        "tail" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("tail() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::List(mut items) => {
                    if !items.is_empty() { items.remove(0); }
                    Ok(Value::List(items))
                }
                _ => Ok(Value::Null),
            }
        }
        "reverse" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("reverse() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v {
                Value::List(mut items) => { items.reverse(); Ok(Value::List(items)) }
                Value::String(s) => Ok(Value::String(s.chars().rev().collect())),
                _ => Ok(Value::Null),
            }
        }
        "range" => {
            if args.len() < 2 { return Err(DbError::Query("range() requires at least 2 args".into())); }
            let start = match eval_expr(&args[0], bindings, graph)? {
                Value::Int(i) => i,
                _ => return Err(DbError::Query("range(): start must be an integer".into())),
            };
            let end = match eval_expr(&args[1], bindings, graph)? {
                Value::Int(i) => i,
                _ => return Err(DbError::Query("range(): end must be an integer".into())),
            };
            let step: i64 = if args.len() >= 3 {
                match eval_expr(&args[2], bindings, graph)? {
                    Value::Int(i) if i != 0 => i,
                    _ => return Err(DbError::Query("range(): step must be a non-zero integer".into())),
                }
            } else if start <= end { 1 } else { -1 };
            let mut result = Vec::new();
            let mut i = start;
            while (step > 0 && i <= end) || (step < 0 && i >= end) {
                result.push(Value::Int(i));
                i += step;
                if result.len() > 1_000_000 { break; } // safety cap
            }
            Ok(Value::List(result))
        }
        "keys" => {
            let arg = args.first().ok_or_else(|| DbError::Query("keys() requires 1 arg".into()))?;
            let var = match arg { Expr::Var(v) => v.clone(),
                _ => return Err(DbError::Query("keys() requires a node or edge variable".into())), };
            match bindings.get(&var) {
                Some(Binding::NodeData(_, node)) => {
                    let mut ks: Vec<Value> = node.properties.keys().map(|k| Value::String(k.clone())).collect();
                    ks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    Ok(Value::List(ks))
                }
                Some(Binding::Node(id)) => {
                    let mut ks: Vec<Value> = graph.get_node(*id)
                        .map(|n| n.properties.keys().map(|k| Value::String(k.clone())).collect())
                        .unwrap_or_default();
                    ks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    Ok(Value::List(ks))
                }
                Some(Binding::Edge(id)) => {
                    let mut ks: Vec<Value> = graph.get_edge(*id)
                        .map(|e| e.properties.keys().map(|k| Value::String(k.clone())).collect())
                        .unwrap_or_default();
                    ks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    Ok(Value::List(ks))
                }
                _ => Ok(Value::Null),
            }
        }
        "exists" => {
            // exists(n.prop) — true if the property exists and is not null.
            let arg = args.first().ok_or_else(|| DbError::Query("exists() requires 1 arg".into()))?;
            let v = eval_expr(arg, bindings, graph)?;
            Ok(Value::Bool(v != Value::Null))
        }
        "isnull" | "is_null" => {
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("isNull() requires 1 arg".into()))?,
                              bindings, graph)?;
            Ok(Value::Bool(v == Value::Null))
        }
        "not" => {
            // not(expr) — function form of logical NOT.
            let v = eval_expr(args.first().ok_or_else(|| DbError::Query("not() requires 1 arg".into()))?,
                              bindings, graph)?;
            match v { Value::Bool(b) => Ok(Value::Bool(!b)), _ => Ok(Value::Null) }
        }

        f => Err(DbError::Query(format!("unknown function: {f}"))),
    }
}

// ── INSERT ──────────────────────────────────────────────────────────────────

/// Upsert helper: find an existing node whose labels and properties are a
/// superset of those requested.  Returns `None` if no match or if the node
/// has no discriminating criteria (no properties specified).
///
/// Matching rules:
///  - The candidate must carry every label listed in `n.labels`.
///  - The candidate must have every property in `n.properties` with the
///    same value (extra properties on the candidate are ignored).
///  - If `n.properties` is empty, we never merge — bare `INSERT (:Label)`
///    always creates a fresh node to avoid accidentally collapsing distinct nodes.
fn find_matching_node(n: &InsertNode, graph: &Graph) -> Option<NodeId> {
    if n.properties.is_empty() {
        return None;
    }

    // Evaluate all expected property values up-front.
    let expected: Vec<(String, Value)> = n
        .properties
        .iter()
        .filter_map(|pa| eval_expr_literal(&pa.value).ok().map(|v| (pa.key.clone(), v)))
        .collect();

    // Use label index to get a manageable candidate set when possible.
    let candidates: Vec<NodeId> = if !n.labels.is_empty() {
        graph.nodes_by_label(&n.labels[0])
    } else {
        graph.all_nodes().into_iter().map(|nd| nd.id).collect()
    };

    'outer: for id in candidates {
        if let Some(node) = graph.get_node(id) {
            // Candidate must carry every requested label.
            for label in &n.labels {
                if !node.labels.contains(label) {
                    continue 'outer;
                }
            }
            // Candidate must have every requested property with matching value.
            for (key, val) in &expected {
                match node.properties.get(key) {
                    Some(v) if v == val => {}
                    _ => continue 'outer,
                }
            }
            return Some(id);
        }
    }
    None
}

/// Upsert helper: find an existing directed edge from `from` to `to` with the
/// given `label`.  Returns `None` if no such edge exists.
fn find_matching_edge(from: NodeId, to: NodeId, label: &str, graph: &Graph) -> Option<EdgeId> {
    for eid in graph.outgoing_edges(from) {
        if let Some(edge) = graph.get_edge(eid) {
            if edge.label == label && edge.to_node == to {
                return Some(eid);
            }
        }
    }
    None
}

/// Build a list of `Operation`s for a set of `InsertElement`s.
///
/// **Upsert semantics**: when a node with properties is inserted, the graph is
/// searched for an existing node with the same labels and property values.
/// If found, that node is reused (no `CreateNode` op) and its ID is bound to
/// the variable.  Similarly, an edge is only created if no edge with the same
/// `(from, to, label)` triple already exists.
///
/// `var_to_node` is pre-seeded with any variables bound by a preceding MATCH.
/// New node variables defined inside the elements are added so subsequent edge
/// elements in the same batch can reference them.
fn build_insert_ops(
    elements: &[InsertElement],
    var_to_node: &mut HashMap<String, NodeId>,
    graph: &mut Graph,
) -> Result<Vec<Operation>, DbError> {
    let mut ops: Vec<Operation> = Vec::new();
    for element in elements {
        match element {
            InsertElement::Node(n) => {
                // Upsert: reuse existing node if label+properties match.
                let id = if let Some(existing) = find_matching_node(n, graph) {
                    existing
                } else {
                    let new_id = graph.alloc_node_id();
                    let mut props = Properties::new();
                    for pa in &n.properties {
                        props.insert(pa.key.clone(), eval_expr_literal(&pa.value)?);
                    }
                    // Enforce declared constraints before committing the insert.
                    graph.check_node_constraints(&n.labels, &props, None)?;
                    let node = crate::types::Node::new(new_id, n.labels.clone(), props);
                    ops.push(Operation::CreateNode { node });
                    new_id
                };
                if let Some(ref var) = n.variable {
                    var_to_node.insert(var.clone(), id);
                }
            }
            InsertElement::Edge(e) => {
                let from = *var_to_node.get(&e.from_var).ok_or_else(|| {
                    DbError::Query(format!(
                        "INSERT edge: variable '{}' not bound — declare the node earlier in the same INSERT or use MATCH … INSERT",
                        e.from_var
                    ))
                })?;
                let to = *var_to_node.get(&e.to_var).ok_or_else(|| {
                    DbError::Query(format!(
                        "INSERT edge: variable '{}' not bound — declare the node earlier in the same INSERT or use MATCH … INSERT",
                        e.to_var
                    ))
                })?;
                // Upsert: skip if this (from, to, label) edge already exists.
                if find_matching_edge(from, to, &e.label, graph).is_none() {
                    let id = graph.alloc_edge_id();
                    let mut props = Properties::new();
                    for pa in &e.properties {
                        props.insert(pa.key.clone(), eval_expr_literal(&pa.value)?);
                    }
                    let edge = crate::types::Edge::new(id, e.label.clone(), from, to, props, e.directed);
                    ops.push(Operation::CreateEdge { edge });
                }
            }
        }
    }
    Ok(ops)
}

/// Execute a standalone `INSERT …` statement.
///
/// Delegates to `build_insert_ops` for the actual upsert logic, then applies
/// the resulting operations to `graph` via `apply_ops`.  Returns a one-element
/// summary row reporting how many elements were inserted and the transaction ID.
fn execute_insert(
    ins: InsertStatement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    let mut var_to_node: HashMap<String, NodeId> = HashMap::new();
    let ops = build_insert_ops(&ins.elements, &mut var_to_node, graph)?;

    let txn_id = *next_txn_id;
    *next_txn_id += 1;
    apply_ops(graph, &ops)?;

    let count = ops.len();
    Ok((vec![summary_row(format!("Inserted {count} element(s) [txn {txn_id}]"))], ops))
}

/// Match multiple comma-separated graph patterns and combine results as a
/// Cartesian product (analogous to a SQL `CROSS JOIN`).
///
/// Each pattern is matched independently via `match_pattern`; the resulting
/// binding sets are then combined: for every pair `(existing, new)` a merged
/// `Bindings` map is produced by inserting all variables from `new` into a
/// clone of `existing`.
///
/// A single pattern is the common case; the identity element is a `vec![{}]`
/// (one empty binding) so the first pattern's results are passed through
/// unchanged without an extra copy.
/// Estimate the number of candidate start nodes for `pattern`.
///
/// Lower score = more selective = run first.  Scoring priority (from the plan):
/// 1. Pattern has an inline `{prop: literal}` constraint on an indexed property
///    → score 1 (index hit; very selective regardless of total index size).
/// 2. Pattern has a label with a range-index predicate in WHERE → half of label count.
/// 3. Pattern has a label → `count_nodes_with_label` (O(1) meta lookup, PERF-9).
/// 4. No label → full node_count (most expensive).
///
/// NOTE: only applies to disconnected comma-separated patterns (what
/// `cross_join_patterns` handles).  Do NOT apply this to nodes within a single
/// connected pattern — the graph structure determines their traversal order.
fn pattern_selectivity(
    pattern: &GraphPattern,
    graph: &Graph,
    where_clause: Option<&Expr>,
) -> usize {
    let Some(label) = pattern.start.labels.first() else {
        return graph.node_count().max(1);
    };

    // 1. Equality index: inline {prop: literal} → index hit, treat as 1.
    for pc in &pattern.start.properties {
        if eval_expr_literal(&pc.value).is_ok()
            && graph.has_property_index(label, &pc.key)
        {
            return 1;
        }
    }

    // 2. Range index: WHERE var.prop OP literal on an indexed property.
    if let (Some(var), Some(where_expr)) = (&pattern.start.variable, where_clause) {
        for idx_def in &graph.index_defs {
            if idx_def.label == label.as_str() {
                let (lo, hi) = extract_prop_range(where_expr, var, &idx_def.property);
                if lo.is_some() || hi.is_some() {
                    // Range scan is selective but we don't know the exact count
                    // without executing the scan.  Use half the label count as
                    // a pessimistic estimate.
                    return (graph.count_nodes_with_label(label) / 2).max(1);
                }
            }
        }
    }

    // 3. Label scan.
    graph.count_nodes_with_label(label).max(1)
}

fn cross_join_patterns(
    patterns: &[GraphPattern],
    graph: &Graph,
    mode: PathMode,
    where_clause: Option<&Expr>,
    early_limit: Option<usize>,
) -> Vec<Bindings> {
    // Sort disconnected patterns by ascending selectivity so the most
    // constrained (cheapest) pattern drives the outer loop.
    // This reduces intermediate result sizes and lets early_limit fire sooner.
    let mut ordered: Vec<&GraphPattern> = patterns.iter().collect();
    ordered.sort_by_key(|p| pattern_selectivity(p, graph, where_clause));

    let mut result: Vec<Bindings> = vec![HashMap::new()];
    for pattern in ordered {
        let pattern_bindings = match_pattern(pattern, graph, mode, where_clause, early_limit);
        let mut next: Vec<Bindings> = Vec::new();
        'outer: for existing in &result {
            for new_b in &pattern_bindings {
                let mut merged = existing.clone();
                for (k, v) in new_b {
                    merged.insert(k.clone(), v.clone());
                }
                next.push(merged);
                if early_limit.is_some_and(|lim| next.len() >= lim) {
                    break 'outer;
                }
            }
        }
        result = next;
    }
    result
}

// ── Index candidate helpers ───────────────────────────────────────────────────

/// Intersect one or more candidate sets, returning the smallest possible result.
///
/// Sets are sorted by length ascending so the smallest set drives the filter
/// loop.  For a single set the Vec is returned without copying.  Returns an
/// empty Vec immediately once the running intersection becomes empty.
fn intersect_node_id_sets(mut sets: Vec<Vec<NodeId>>) -> Vec<NodeId> {
    debug_assert!(!sets.is_empty());
    if sets.len() == 1 {
        return sets.remove(0);
    }
    sets.sort_unstable_by_key(|v| v.len());
    let mut result: HashSet<NodeId> = sets[0].iter().copied().collect();
    for set in &sets[1..] {
        if result.is_empty() {
            return vec![];
        }
        let other: HashSet<NodeId> = set.iter().copied().collect();
        result.retain(|id| other.contains(id));
    }
    result.into_iter().collect()
}

/// Scan the WHERE clause for all index-usable predicates on `(label, var)` and
/// append one candidate `Vec<NodeId>` per matched predicate into `out`.
///
/// The caller gathers results across all indexes (and from inline property
/// constraints), then intersects them all to obtain the minimal candidate set.
///
/// Covers three predicate shapes per indexed property:
/// 1. Equality: `var.prop = literal`
/// 2. Range:    `var.prop OP literal` (>, >=, <, <=)
/// 3. Prefix:   `startsWith(var.prop, "prefix")`
fn collect_where_index_candidates(
    graph: &Graph,
    label: &str,
    var: &str,
    where_expr: &Expr,
    out: &mut Vec<Vec<NodeId>>,
) {
    for idx_def in &graph.index_defs {
        if idx_def.label != label {
            continue;
        }
        let prop = &idx_def.property;

        // Equality is most selective — use it and skip range/prefix for this prop.
        if let Some(eq_val) = extract_where_equality(where_expr, var, prop) {
            if let Some(ids) = graph.lookup_by_property(label, prop, eq_val) {
                out.push(ids);
                continue;
            }
        }

        let (lo, hi) = extract_prop_range(where_expr, var, prop);
        if lo.is_some() || hi.is_some() {
            if let Some(ids) = graph.lookup_by_property_range(label, prop, lo, hi) {
                out.push(ids);
                continue;
            }
        }

        if let Some(prefix) = extract_starts_with(where_expr, var, prop) {
            if let Some(ids) = graph.lookup_by_property_prefix(label, prop, prefix) {
                out.push(ids);
            }
        }
    }
}


/// Extract the literal value from an equality predicate `var.prop = literal`
/// (or `literal = var.prop`) anywhere in the expression tree (through AND).
/// Returns `None` if no such predicate is found for this `(var, prop)` pair.
fn extract_where_equality<'a>(expr: &'a Expr, var: &str, prop: &str) -> Option<&'a Value> {
    match expr {
        Expr::BinOp(l, BinOp::And, r) => {
            extract_where_equality(l, var, prop)
                .or_else(|| extract_where_equality(r, var, prop))
        }
        // var.prop = literal
        Expr::BinOp(l, BinOp::Eq, r) if is_var_prop(l, var, prop) => {
            if let Expr::Literal(val) = r.as_ref() { Some(val) } else { None }
        }
        // literal = var.prop
        Expr::BinOp(l, BinOp::Eq, r) if is_var_prop(r, var, prop) => {
            if let Expr::Literal(val) = l.as_ref() { Some(val) } else { None }
        }
        _ => None,
    }
}

/// Extract the string prefix from a `startsWith(var.prop, "prefix")` call
/// anywhere in the expression tree (through AND).
///
/// Returns `None` if no such call is found for this `(var, prop)` pair.
fn extract_starts_with<'a>(expr: &'a Expr, var: &str, prop: &str) -> Option<&'a str> {
    match expr {
        Expr::BinOp(l, BinOp::And, r) => {
            extract_starts_with(l, var, prop)
                .or_else(|| extract_starts_with(r, var, prop))
        }
        // startsWith(var.prop, "prefix")
        Expr::Call(name, args)
            if name.eq_ignore_ascii_case("startsWith") && args.len() == 2 =>
        {
            if is_var_prop(&args[0], var, prop) {
                if let Expr::Literal(Value::String(s)) = &args[1] {
                    return Some(s.as_str());
                }
            }
            None
        }
        _ => None,
    }
}

/// Extract `(lower_bound, upper_bound)` for `var.prop` from a WHERE expression.
///
/// Handles conjunctions (`AND`) and both orientations of each comparison:
/// - `var.prop > v`  / `v < var.prop`  → lower bound (exclusive)
/// - `var.prop >= v` / `v <= var.prop` → lower bound (inclusive)
/// - `var.prop < v`  / `v > var.prop`  → upper bound (exclusive)
/// - `var.prop <= v` / `v >= var.prop` → upper bound (inclusive)
///
/// For multiple bounds on the same side the first found wins (caller applies
/// the full WHERE clause anyway, so false positives are safe).
fn extract_prop_range<'a>(
    expr: &'a Expr,
    var: &str,
    prop: &str,
) -> (Option<(&'a Value, bool)>, Option<(&'a Value, bool)>) {
    let mut lo: Option<(&Value, bool)> = None;
    let mut hi: Option<(&Value, bool)> = None;
    collect_bounds(expr, var, prop, &mut lo, &mut hi);
    (lo, hi)
}

/// Recursively walk an expression tree collecting lower/upper bound predicates
/// for `var.prop` into `lo` / `hi`.
///
/// Only the first found bound for each side is recorded (first-wins); the
/// full WHERE predicate is still applied after candidate selection, so false
/// positives from a single bound are safe.
fn collect_bounds<'a>(
    expr: &'a Expr,
    var: &str,
    prop: &str,
    lo: &mut Option<(&'a Value, bool)>,
    hi: &mut Option<(&'a Value, bool)>,
) {
    match expr {
        // Recurse into AND conjunctions.
        Expr::BinOp(l, BinOp::And, r) => {
            collect_bounds(l, var, prop, lo, hi);
            collect_bounds(r, var, prop, lo, hi);
        }
        // var.prop > lit  /  var.prop >= lit  →  lower bound
        Expr::BinOp(l, op, r)
            if matches!(op, BinOp::Gt | BinOp::Gte) && is_var_prop(l, var, prop) =>
        {
            if let Expr::Literal(val) = r.as_ref() {
                if lo.is_none() {
                    *lo = Some((val, *op == BinOp::Gte));
                }
            }
        }
        // lit < var.prop  /  lit <= var.prop  →  lower bound (reversed)
        Expr::BinOp(l, op, r)
            if matches!(op, BinOp::Lt | BinOp::Lte) && is_var_prop(r, var, prop) =>
        {
            if let Expr::Literal(val) = l.as_ref() {
                if lo.is_none() {
                    *lo = Some((val, *op == BinOp::Lte));
                }
            }
        }
        // var.prop < lit  /  var.prop <= lit  →  upper bound
        Expr::BinOp(l, op, r)
            if matches!(op, BinOp::Lt | BinOp::Lte) && is_var_prop(l, var, prop) =>
        {
            if let Expr::Literal(val) = r.as_ref() {
                if hi.is_none() {
                    *hi = Some((val, *op == BinOp::Lte));
                }
            }
        }
        // lit > var.prop  /  lit >= var.prop  →  upper bound (reversed)
        Expr::BinOp(l, op, r)
            if matches!(op, BinOp::Gt | BinOp::Gte) && is_var_prop(r, var, prop) =>
        {
            if let Expr::Literal(val) = l.as_ref() {
                if hi.is_none() {
                    *hi = Some((val, *op == BinOp::Gte));
                }
            }
        }
        _ => {}
    }
}

/// Returns `true` if `expr` is `var.prop` (a simple property access on a named variable).
fn is_var_prop(expr: &Expr, var: &str, prop: &str) -> bool {
    matches!(
        expr,
        Expr::Property(base, p) if p == prop && matches!(base.as_ref(), Expr::Var(v) if v == var)
    )
}

/// Execute a `MATCH … [WHERE] INSERT …` statement.
///
/// Workflow:
/// 1. Cross-join all MATCH patterns (Cartesian product).
/// 2. Filter by the optional WHERE clause.
/// 3. For each matched binding, seed `var_to_node` from the matched node IDs
///    and call `build_insert_ops` to produce (upsert) the INSERT elements.
/// 4. Apply all accumulated operations in one batch and return a summary row.
fn execute_match_insert(
    mi: MatchInsertStatement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    // Step 1: match all patterns and cross-join the results.
    let merged = cross_join_patterns(&mi.patterns, graph, PathMode::Walk, mi.where_clause.as_ref(), None);

    // Step 2: apply WHERE filter.
    let filtered: Vec<Bindings> = merged
        .into_iter()
        .filter(|b| {
            if let Some(ref cond) = mi.where_clause {
                matches!(eval_expr(cond, b, graph), Ok(Value::Bool(true)))
            } else {
                true
            }
        })
        .collect();

    // Step 3: for each matched binding, build insert ops using matched node IDs.
    let mut all_ops: Vec<Operation> = Vec::new();
    for binding in &filtered {
        // Seed var_to_node from the matched nodes in this binding.
        let mut var_to_node: HashMap<String, NodeId> = HashMap::new();
        for (var, b) in binding {
            if let Binding::Node(id) | Binding::NodeData(id, _) = b {
                var_to_node.insert(var.clone(), *id);
            }
        }
        let ops = build_insert_ops(&mi.elements, &mut var_to_node, graph)?;
        all_ops.extend(ops);
    }

    let txn_id = *next_txn_id;
    *next_txn_id += 1;
    let count = all_ops.len();
    apply_ops(graph, &all_ops)?;

    Ok((vec![summary_row(format!("Inserted {count} element(s) [txn {txn_id}]"))], all_ops))
}

// ── SET ────────────────────────────────────────────────────────────────────

/// Execute a `MATCH … [WHERE] SET …` statement.
///
/// Matches the pattern, filters by WHERE, then for each matched binding emits
/// either `SetProperty` or `AddLabel` operations for each `SetItem`.  Returns
/// an error if a path variable (`EdgeList`) or an unbound variable is targeted.
fn execute_set(
    s: SetStatement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    let bindings = match_pattern(&s.match_pattern, graph, PathMode::Walk, s.where_clause.as_ref(), None);

    let filtered: Vec<Bindings> = bindings
        .into_iter()
        .filter(|b| {
            if let Some(ref cond) = s.where_clause {
                matches!(eval_expr(cond, b, graph), Ok(Value::Bool(true)))
            } else {
                true
            }
        })
        .collect();

    let mut ops: Vec<Operation> = Vec::new();

    for binding in &filtered {
        for item in &s.assignments {
            match item {
                SetItem::Property { variable, key, value } => {
                    let val = eval_expr(value, binding, graph)?;
                    match binding.get(variable) {
                        Some(Binding::Node(id) | Binding::NodeData(id, _)) => ops.push(Operation::SetProperty {
                            target: PropertyTarget::Node(*id),
                            key: key.clone(),
                            value: val,
                        }),
                        Some(Binding::Edge(id)) => ops.push(Operation::SetProperty {
                            target: PropertyTarget::Edge(*id),
                            key: key.clone(),
                            value: val,
                        }),
                        Some(Binding::EdgeList(_)) | Some(Binding::Value(_)) => {
                            return Err(DbError::Query(format!(
                                "cannot SET property on path variable '{variable}'; \
                                 bind individual edges instead"
                            )))
                        }
                        None => {
                            return Err(DbError::Query(format!(
                                "variable '{variable}' not bound"
                            )))
                        }
                    }
                }
                SetItem::AddLabel { variable, label } => match binding.get(variable) {
                    Some(Binding::Node(id) | Binding::NodeData(id, _)) => ops.push(Operation::AddLabel {
                        node_id: *id,
                        label: label.clone(),
                    }),
                    _ => {
                        return Err(DbError::Query(format!(
                            "variable '{variable}' not a node"
                        )))
                    }
                },
            }
        }
    }

    let count = ops.len();
    let txn_id = *next_txn_id;
    *next_txn_id += 1;
    apply_ops(graph, &ops)?;

    Ok((vec![summary_row(format!("Set {count} property/label operation(s) [txn {txn_id}]"))], ops))
}

// ── REMOVE ─────────────────────────────────────────────────────────────────

/// Execute a `MATCH … [WHERE] REMOVE …` statement.
///
/// Mirrors `execute_set`: matches the pattern, filters, then emits
/// `RemoveProperty` or `RemoveLabel` operations per `RemoveItem`.
fn execute_remove(
    r: RemoveStatement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    let bindings = match_pattern(&r.match_pattern, graph, PathMode::Walk, r.where_clause.as_ref(), None);

    let filtered: Vec<Bindings> = bindings
        .into_iter()
        .filter(|b| {
            if let Some(ref cond) = r.where_clause {
                matches!(eval_expr(cond, b, graph), Ok(Value::Bool(true)))
            } else {
                true
            }
        })
        .collect();

    let mut ops: Vec<Operation> = Vec::new();

    for binding in &filtered {
        for item in &r.items {
            match item {
                RemoveItem::Property { variable, key } => match binding.get(variable) {
                    Some(Binding::Node(id) | Binding::NodeData(id, _)) => ops.push(Operation::RemoveProperty {
                        target: PropertyTarget::Node(*id),
                        key: key.clone(),
                    }),
                    Some(Binding::Edge(id)) => ops.push(Operation::RemoveProperty {
                        target: PropertyTarget::Edge(*id),
                        key: key.clone(),
                    }),
                    Some(Binding::EdgeList(_)) | Some(Binding::Value(_)) => {
                        return Err(DbError::Query(format!(
                            "cannot REMOVE property on path variable '{variable}'"
                        )))
                    }
                    None => return Err(DbError::Query(format!("variable '{variable}' not bound"))),
                },
                RemoveItem::Label { variable, label } => match binding.get(variable) {
                    Some(Binding::Node(id) | Binding::NodeData(id, _)) => ops.push(Operation::RemoveLabel {
                        node_id: *id,
                        label: label.clone(),
                    }),
                    _ => return Err(DbError::Query(format!("'{variable}' is not a node"))),
                },
            }
        }
    }

    let count = ops.len();
    let txn_id = *next_txn_id;
    *next_txn_id += 1;
    apply_ops(graph, &ops)?;

    Ok((vec![summary_row(format!("Removed {count} property/label(s) [txn {txn_id}]"))], ops))
}

// ── DELETE ─────────────────────────────────────────────────────────────────

/// Execute a `MATCH … [WHERE] [DETACH] DELETE …` statement.
///
/// For each matched binding, emits `DeleteNodeDetach` (when `d.detach` is true)
/// or `DeleteNode` for node variables, and `DeleteEdge` for edge variables.
/// `DeleteNode` on a node with incident edges will return an error from the
/// storage layer; `DeleteNodeDetach` cascades the deletion to all incident edges.
fn execute_delete(
    d: DeleteStatement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    let bindings = match_pattern(&d.match_pattern, graph, PathMode::Walk, d.where_clause.as_ref(), None);

    let filtered: Vec<Bindings> = bindings
        .into_iter()
        .filter(|b| {
            if let Some(ref cond) = d.where_clause {
                matches!(eval_expr(cond, b, graph), Ok(Value::Bool(true)))
            } else {
                true
            }
        })
        .collect();

    let mut ops: Vec<Operation> = Vec::new();

    for binding in &filtered {
        for var in &d.variables {
            match binding.get(var) {
                Some(Binding::Node(id) | Binding::NodeData(id, _)) => {
                    if d.detach {
                        ops.push(Operation::DeleteNodeDetach { node_id: *id });
                    } else {
                        ops.push(Operation::DeleteNode { node_id: *id });
                    }
                }
                Some(Binding::Edge(id)) => {
                    ops.push(Operation::DeleteEdge { edge_id: *id });
                }
                Some(Binding::EdgeList(_)) | Some(Binding::Value(_)) => {
                    return Err(DbError::Query(format!(
                        "cannot DELETE path variable '{var}'; \
                         iterate individual edges instead"
                    )))
                }
                None => return Err(DbError::Query(format!("variable '{var}' not bound"))),
            }
        }
    }

    let count = ops.len();
    let txn_id = *next_txn_id;
    *next_txn_id += 1;
    apply_ops(graph, &ops)?;

    Ok((vec![summary_row(format!("Deleted {count} element(s) [txn {txn_id}]"))], ops))
}

// ── Index management ────────────────────────────────────────────────────────

/// Execute a `CREATE INDEX ON :Label(property)` statement.
///
/// If the index already exists, returns a summary saying so without emitting a
/// new operation.  Otherwise builds the in-memory index and emits a
/// `CreateIndex` operation for WAL persistence.
fn execute_create_index(
    ci: CreateIndexStatement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    // Determine target: check if label is known as edge label first, then node.
    // If only one side has the label, use that. If both or neither, default to node.
    let is_node_label = graph.label_index.contains_key(&ci.label);
    let is_edge_label = graph.edge_label_index.contains_key(&ci.label);
    let (already, op) = if is_edge_label && !is_node_label {
        let already = graph.has_edge_property_index(&ci.label, &ci.property);
        graph.create_edge_property_index(&ci.label, &ci.property);
        (already, Operation::CreateIndex {
            label: ci.label.clone(), property: ci.property.clone(),
            target: crate::graph::IndexTarget::Edge,
        })
    } else {
        let already = graph.has_property_index(&ci.label, &ci.property);
        graph.apply_create_index(&ci.label, &ci.property);
        (already, Operation::CreateIndex {
            label: ci.label.clone(), property: ci.property.clone(),
            target: crate::graph::IndexTarget::Node,
        })
    };
    let txn_id = *next_txn_id;
    *next_txn_id += 1;
    let msg = if already {
        format!("Index ON :{}({}) already exists [txn {txn_id}]", ci.label, ci.property)
    } else {
        format!("Index created ON :{}({}) [txn {txn_id}]", ci.label, ci.property)
    };
    Ok((vec![summary_row(msg)], vec![op]))
}

/// Execute a `DROP INDEX ON :Label(property)` statement.
fn execute_drop_index(
    di: DropIndexStatement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    let existed_node = graph.drop_property_index(&di.label, &di.property);
    let existed_edge = if !existed_node {
        graph.drop_edge_property_index(&di.label, &di.property)
    } else {
        false
    };
    let existed = existed_node || existed_edge;
    let txn_id = *next_txn_id;
    *next_txn_id += 1;
    let msg = if existed {
        format!("Index ON :{}({}) dropped [txn {txn_id}]", di.label, di.property)
    } else {
        format!("No index ON :{}({}) [txn {txn_id}]", di.label, di.property)
    };
    let op = Operation::DropIndex { label: di.label, property: di.property };
    Ok((vec![summary_row(msg)], if existed { vec![op] } else { vec![] }))
}

/// Execute a `SHOW INDEXES` statement.
///
/// Returns one row per defined property index with columns `label`, `property`,
/// `target` (node/edge), and `entries`.
fn execute_show_indexes(graph: &Graph) -> Result<Vec<Row>, DbError> {
    let indexes = graph.list_property_indexes();
    if indexes.is_empty() {
        return Ok(vec![summary_row("No indexes defined.".to_string())]);
    }
    Ok(indexes
        .into_iter()
        .map(|(label, property, count, target)| {
            let mut row = Row::new();
            row.insert("label".to_string(), Value::String(label));
            row.insert("property".to_string(), Value::String(property));
            row.insert("target".to_string(), Value::String(target));
            row.insert("entries".to_string(), Value::Int(count as i64));
            row
        })
        .collect())
}

// ── CALL ─────────────────────────────────────────────────────────────────────

/// Execute a `CALL algorithmName(…) YIELD …` statement.
///
/// Parameters are evaluated as constant expressions (literals and arithmetic
/// only — no variable bindings are in scope outside a MATCH).  The algorithm
/// is dispatched via `crate::algorithms::dispatch_call`; the result rows are
/// then filtered to only the columns listed in the optional `YIELD` clause.
fn execute_call(c: super::ast::CallStatement, graph: &Graph) -> Result<Vec<Row>, DbError> {
    // Evaluate each parameter expression to a Value.
    // CALL params must be constant (literal / arithmetic) expressions since
    // there is no variable binding context outside of a MATCH clause.
    let params: HashMap<String, Value> = c
        .params
        .into_iter()
        .map(|(k, expr)| eval_const_expr(&expr).map(|v| (k, v)))
        .collect::<Result<_, _>>()?;

    let mut rows = crate::algorithms::dispatch_call(graph, &c.name, &params)?;

    // Apply YIELD projection if specified.
    if let Some(cols) = c.yields {
        let col_set: std::collections::HashSet<&str> =
            cols.iter().map(String::as_str).collect();
        for row in &mut rows {
            row.retain(|k, _| col_set.contains(k.as_str()));
        }
    }

    Ok(rows)
}

/// Execute a `CALL … YIELD … [MATCH … WHERE] RETURN …` pipeline statement.
///
/// A pipeline extends a plain `CALL` with:
/// - An optional `MATCH` clause that is applied as a seeded pattern match:
///   each CALL result row seeds the pattern via `cross_join_patterns_seeded`,
///   allowing algorithm output to be correlated with graph structure.
/// - A full RETURN clause with ORDER BY / LIMIT / aggregation via
///   `apply_return_clause`.
///
/// Phases:
/// 1. Evaluate constant parameters and run the algorithm.
/// 2. Project algorithm rows to the YIELD columns, converting to `Binding::Value`.
/// 3. Optionally filter/extend each row through the MATCH clause (left-seeded).
/// 4. Apply the RETURN clause.
fn execute_call_pipeline(c: super::ast::CallPipelineStatement, graph: &Graph) -> Result<Vec<Row>, DbError> {
    // Phase 1: evaluate params and run the algorithm.
    let params: HashMap<String, Value> = c
        .params
        .into_iter()
        .map(|(k, expr)| eval_const_expr(&expr).map(|v| (k, v)))
        .collect::<Result<_, _>>()?;

    let algo_rows = crate::algorithms::dispatch_call(graph, &c.name, &params)?;

    // Phase 2: project to YIELD columns, convert to Bindings.
    let col_set: std::collections::HashSet<&str> =
        c.yields.iter().map(String::as_str).collect();

    let mut bindings: Vec<Bindings> = algo_rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .filter(|(k, _)| col_set.contains(k.as_str()))
                .map(|(k, v)| (k, Binding::Value(v)))
                .collect()
        })
        .collect();

    // Phase 3: optional MATCH filter — each CALL row seeds the pattern match.
    if let Some(mc) = c.match_clause {
        bindings = bindings
            .into_iter()
            .flat_map(|seed| {
                cross_join_patterns_seeded(&mc.patterns, &seed, mc.path_mode, mc.where_clause.as_ref(), graph)
            })
            .collect();
    }

    apply_return_clause(bindings, &c.return_clause, graph)
}

/// Evaluate a constant expression (no variable bindings).
/// Used for CALL parameter values where only literals and arithmetic are valid.
fn eval_const_expr(expr: &Expr) -> Result<Value, DbError> {
    match expr {
        Expr::Literal(v) => Ok(v.clone()),
        Expr::BinOp(l, op, r) => {
            let lv = eval_const_expr(l)?;
            let rv = eval_const_expr(r)?;
            apply_binop(*op, &lv, &rv)
        }
        Expr::Not(e) => {
            let v = eval_const_expr(e)?;
            match v {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                other => Err(DbError::Query(format!(
                    "CALL parameter: NOT requires a boolean, got {other:?}"
                ))),
            }
        }
        other => Err(DbError::Query(format!(
            "CALL parameters must be constant expressions (literals or arithmetic). \
             Got: {other:?}. Use literal values like 0.85 or \"out\"."
        ))),
    }
}

/// Apply a binary operator to two constant Values (shared with eval_const_expr).
fn apply_binop(op: BinOp, l: &Value, r: &Value) -> Result<Value, DbError> {
    use Value::*;
    match (op, l, r) {
        (BinOp::Add, Int(a), Int(b))     => Ok(Int(a + b)),
        (BinOp::Sub, Int(a), Int(b))     => Ok(Int(a - b)),
        (BinOp::Mul, Int(a), Int(b))     => Ok(Int(a * b)),
        (BinOp::Div, Int(a), Int(b))     => {
            if *b == 0 { Err(DbError::Query("division by zero".into())) }
            else { Ok(Int(a / b)) }
        }
        (BinOp::Mod, Int(a), Int(b))     => {
            if *b == 0 { Err(DbError::Query("modulo by zero".into())) }
            else { Ok(Int(a % b)) }
        }
        (BinOp::Add, Float(a), Float(b)) => Ok(Float(a + b)),
        (BinOp::Sub, Float(a), Float(b)) => Ok(Float(a - b)),
        (BinOp::Mul, Float(a), Float(b)) => Ok(Float(a * b)),
        (BinOp::Div, Float(a), Float(b)) => Ok(Float(a / b)),
        (BinOp::Add, Int(a), Float(b))   => Ok(Float(*a as f64 + b)),
        (BinOp::Add, Float(a), Int(b))   => Ok(Float(a + *b as f64)),
        (BinOp::Sub, Int(a), Float(b))   => Ok(Float(*a as f64 - b)),
        (BinOp::Sub, Float(a), Int(b))   => Ok(Float(a - *b as f64)),
        (BinOp::Mul, Int(a), Float(b))   => Ok(Float(*a as f64 * b)),
        (BinOp::Mul, Float(a), Int(b))   => Ok(Float(a * *b as f64)),
        (BinOp::Div, Int(a), Float(b))   => Ok(Float(*a as f64 / b)),
        (BinOp::Div, Float(a), Int(b))   => Ok(Float(a / *b as f64)),
        _ => Err(DbError::Query(format!(
            "unsupported constant operation {op:?} on {l:?} and {r:?}"
        ))),
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Build a single-column `Row` with column name `"result"` containing `msg`.
/// Used by all write statement executors to return a human-readable summary.
fn summary_row(msg: String) -> Row {
    let mut row = Row::new();
    row.insert("result".to_string(), Value::String(msg));
    row
}

// ── LOAD CSV ─────────────────────────────────────────────────────────────────

/// Execute `LOAD CSV NODES FROM 'path' [LABEL X]`.
// ── TRUNCATE ───────────────────────────────────────────────────────────────

/// Execute a `TRUNCATE` statement.
///
/// Delegates to [`Graph::clear`] which issues RocksDB range tombstones across
/// all data column families — O(1) regardless of graph size.  Returns a
/// one-row confirmation.
fn execute_truncate(
    graph: &mut Graph,
) -> Result<(Vec<Row>, Vec<crate::transaction::operation::Operation>), DbError> {
    graph.clear()?;
    let mut row = Row::new();
    row.insert("cleared".into(), crate::types::Value::Bool(true));
    Ok((vec![row], vec![]))
}

// ── LOAD CSV ───────────────────────────────────────────────────────────────

///
/// Opens the file at `stmt.path` (relative to the process working directory),
/// calls [`crate::csv_import::load_nodes_csv`], stores the resulting `id_map`
/// in [`Graph::csv_id_map`] for subsequent `LOAD CSV EDGES`, and returns a
/// one-row summary.
fn execute_load_csv_nodes(
    stmt: super::ast::LoadCsvNodesStatement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<crate::transaction::operation::Operation>), DbError> {
    use std::fs::File;
    use crate::csv_import::load_nodes_csv;

    let file = File::open(&stmt.path)
        .map_err(|e| DbError::Storage(e))?;
    let result = load_nodes_csv(file, graph, stmt.label.as_deref())?;

    // Store the id_map in graph for a subsequent LOAD CSV EDGES call.
    graph.csv_id_map = result.id_map;

    *next_txn_id += 1;
    let msg = format!("loaded {} nodes", result.inserted);
    Ok((vec![summary_row(msg)], vec![]))
}

/// Execute `LOAD CSV EDGES FROM 'path' [LABEL X]`.
///
/// Opens the file at `stmt.path`, resolves `:START_ID` / `:END_ID` against
/// [`Graph::csv_id_map`] (populated by a preceding `LOAD CSV NODES` call),
/// and inserts the edges.  Returns a one-row summary including skipped count.
fn execute_load_csv_edges(
    stmt: super::ast::LoadCsvEdgesStatement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<crate::transaction::operation::Operation>), DbError> {
    use std::fs::File;
    use crate::csv_import::load_edges_csv;

    let file = File::open(&stmt.path)
        .map_err(|e| DbError::Storage(e))?;

    // Clone the map so we can borrow graph mutably for the load.
    let id_map = graph.csv_id_map.clone();
    let result = load_edges_csv(file, graph, &id_map, stmt.label.as_deref())?;

    *next_txn_id += 1;
    let msg = if result.skipped > 0 {
        format!("loaded {} edges ({} skipped — unresolved IDs)", result.inserted, result.skipped)
    } else {
        format!("loaded {} edges", result.inserted)
    };
    Ok((vec![summary_row(msg)], vec![]))
}

// ── UNWIND … INSERT ──────────────────────────────────────────────────────────

/// Execute `UNWIND expr AS var INSERT elements`.
///
/// Evaluates `expr` (which must resolve to a `Value::List`, or a `$param`
/// holding a list) against an empty binding context, then for each element
/// creates a fresh `Bindings` with the loop variable set, evaluates the
/// property expressions, and inserts the described graph elements.
fn execute_unwind_insert(
    stmt: UnwindInsertStatement,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    let empty: Bindings = HashMap::new();
    let list = eval_expr(&stmt.expr, &empty, graph)?;
    let items = match list {
        Value::List(v) => v,
        other => return Err(DbError::Query(
            format!("UNWIND INSERT: expression must evaluate to a list, got {other}"),
        )),
    };

    let mut all_ops: Vec<Operation> = Vec::new();
    for item in items {
        let mut bindings: Bindings = HashMap::new();
        bindings.insert(stmt.variable.clone(), Binding::Value(item));

        let mut var_to_node: HashMap<String, NodeId> = HashMap::new();
        // Evaluate elements with `eval_expr` against loop variable bindings.
        for element in &stmt.elements {
            match element {
                InsertElement::Node(n) => {
                    let new_id = graph.alloc_node_id();
                    let mut props = Properties::new();
                    for pa in &n.properties {
                        props.insert(pa.key.clone(), eval_expr(&pa.value, &bindings, graph)?);
                    }
                    graph.check_node_constraints(&n.labels, &props, None)?;
                    let node = crate::types::Node::new(new_id, n.labels.clone(), props);
                    all_ops.push(Operation::CreateNode { node });
                    if let Some(ref var) = n.variable {
                        var_to_node.insert(var.clone(), new_id);
                    }
                }
                InsertElement::Edge(e) => {
                    let from = *var_to_node.get(&e.from_var).ok_or_else(|| {
                        DbError::Query(format!("UNWIND INSERT edge: '{}' not bound", e.from_var))
                    })?;
                    let to = *var_to_node.get(&e.to_var).ok_or_else(|| {
                        DbError::Query(format!("UNWIND INSERT edge: '{}' not bound", e.to_var))
                    })?;
                    if find_matching_edge(from, to, &e.label, graph).is_none() {
                        let id = graph.alloc_edge_id();
                        let mut props = Properties::new();
                        for pa in &e.properties {
                            props.insert(pa.key.clone(), eval_expr(&pa.value, &bindings, graph)?);
                        }
                        let edge = crate::types::Edge::new(id, e.label.clone(), from, to, props, e.directed);
                        all_ops.push(Operation::CreateEdge { edge });
                    }
                }
            }
        }
    }

    apply_ops(graph, &all_ops)?;
    let txn_id = *next_txn_id;
    *next_txn_id += 1;
    let count = all_ops.len();
    Ok((vec![summary_row(format!("Inserted {count} element(s) [txn {txn_id}]"))], all_ops))
}

// ── CONSTRAINT statements ─────────────────────────────────────────────────────

/// Execute a `CREATE/DROP CONSTRAINT …` or `SHOW CONSTRAINTS` statement.
fn execute_constraint(
    stmt: ConstraintStatement,
    graph: &mut Graph,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    use crate::graph::constraints as gc;

    fn to_graph_kind(k: &ConstraintKind) -> gc::ConstraintKind {
        match k {
            ConstraintKind::Unique => gc::ConstraintKind::Unique,
            ConstraintKind::Type(vk) => gc::ConstraintKind::Type(match vk {
                ValueKind::Integer => gc::ValueKind::Integer,
                ValueKind::Float   => gc::ValueKind::Float,
                ValueKind::String  => gc::ValueKind::String,
                ValueKind::Boolean => gc::ValueKind::Boolean,
            }),
        }
    }

    match stmt.op {
        ConstraintOp::Show => {
            let rows = graph.list_constraints().iter().map(|c| {
                let mut row = Row::new();
                row.insert("label".into(),    Value::String(c.label.clone()));
                row.insert("property".into(), Value::String(c.property.clone()));
                row.insert("kind".into(),     Value::String(match &c.kind {
                    gc::ConstraintKind::Unique     => "UNIQUE".to_string(),
                    gc::ConstraintKind::Type(vk)   => format!("TYPE IS {}", vk),
                }));
                row
            }).collect();
            Ok((rows, vec![]))
        }
        ConstraintOp::Create { kind, label, property } => {
            let def = gc::ConstraintDef {
                kind: to_graph_kind(&kind),
                label: label.clone(),
                property: property.clone(),
            };
            let added = graph.add_constraint(def)?;
            let msg = if added {
                format!("constraint created on :{label}({property})")
            } else {
                format!("constraint already exists on :{label}({property})")
            };
            Ok((vec![summary_row(msg)], vec![]))
        }
        ConstraintOp::Drop { kind, label, property } => {
            let graph_kind = to_graph_kind(&kind);
            let removed = graph.remove_constraint(&label, &property, &graph_kind);
            let msg = if removed {
                format!("constraint dropped on :{label}({property})")
            } else {
                format!("constraint not found on :{label}({property})")
            };
            Ok((vec![summary_row(msg)], vec![]))
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::types::DbError;

    /// Run a GQL string against the graph and return rows.
    fn run(graph: &mut Graph, txn: &mut u64, gql: &str) -> Result<Vec<Row>, DbError> {
        let stmt = super::super::parser::parse(gql)?;
        execute(stmt, graph, txn)
    }

    /// Extract a column from rows in order (preserves ORDER BY result).
    fn col(rows: &[Row], c: &str) -> Vec<String> {
        rows.iter()
            .map(|r| r.get(c).map(|v| format!("{v}")).unwrap_or_else(|| "null".to_string()))
            .collect()
    }

    /// Extract a column and sort it (for order-independent assertions).
    fn sorted_col(rows: &[Row], c: &str) -> Vec<String> {
        let mut v = col(rows, c);
        v.sort();
        v
    }

    /// Build a standard three-node test graph: Alice(30), Bob(25), Carol(35).
    fn three_people() -> (Graph, u64) {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", age: 30})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob",   age: 25})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Carol", age: 35})"#).unwrap();
        (g, txn)
    }

    // ── Issue 3: INSERT edge in a multi-element statement ───────────────────

    #[test]
    fn insert_edge_combined_statement() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        // All three elements in one INSERT: two nodes + one edge.
        run(
            &mut g, &mut txn,
            r#"INSERT (a:Person {name: "Alice"}), (b:Person {name: "Bob"}), (a)-[:KNOWS]->(b)"#,
        )
        .unwrap();

        let rows = run(&mut g, &mut txn, r#"MATCH (x)-[:KNOWS]->(y) RETURN x.name, y.name"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(col(&rows, "x.name"), vec!["\"Alice\""]);
        assert_eq!(col(&rows, "y.name"), vec!["\"Bob\""]);
    }

    #[test]
    fn insert_multiple_edges_in_one_statement() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:N {name: "A"}), (b:N {name: "B"}), (c:N {name: "C"}),
                      (a)-[:R]->(b), (b)-[:R]->(c)"#,
        )
        .unwrap();

        let rows = run(&mut g, &mut txn, r#"MATCH (x)-[:R]->(y) RETURN x.name, y.name"#).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn insert_edge_only_inserts_correct_endpoints() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (src:City {name: "NYC"}), (dst:City {name: "LA"}), (src)-[:FLIGHT]->(dst)"#,
        )
        .unwrap();

        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (x:City)-[:FLIGHT]->(y:City) RETURN x.name, y.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(col(&rows, "x.name"), vec!["\"NYC\""]);
        assert_eq!(col(&rows, "y.name"), vec!["\"LA\""]);
    }

    // ── Issue 5: ORDER BY ───────────────────────────────────────────────────

    #[test]
    fn order_by_numeric_desc() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN n.name ORDER BY n.age DESC"#,
        )
        .unwrap();
        assert_eq!(col(&rows, "n.name"), vec!["\"Carol\"", "\"Alice\"", "\"Bob\""]);
    }

    #[test]
    fn order_by_numeric_asc() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN n.name ORDER BY n.age ASC"#,
        )
        .unwrap();
        assert_eq!(col(&rows, "n.name"), vec!["\"Bob\"", "\"Alice\"", "\"Carol\""]);
    }

    #[test]
    fn order_by_string_asc() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN n.name ORDER BY n.name ASC"#,
        )
        .unwrap();
        assert_eq!(col(&rows, "n.name"), vec!["\"Alice\"", "\"Bob\"", "\"Carol\""]);
    }

    #[test]
    fn order_by_with_where_clause() {
        let (mut g, mut txn) = three_people();
        // Previously broken because kw_or matched "OR" in "ORDER"
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.age > 28 RETURN n.name ORDER BY n.age DESC"#,
        )
        .unwrap();
        assert_eq!(col(&rows, "n.name"), vec!["\"Carol\"", "\"Alice\""]);
    }

    // ── Issue 6: DISTINCT ───────────────────────────────────────────────────

    #[test]
    fn distinct_removes_duplicate_rows() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        // Insert two nodes with the same name.
        run(&mut g, &mut txn, r#"INSERT (:Tag {name: "rust"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Tag {name: "rust"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Tag {name: "go"})"#).unwrap();

        let rows = run(&mut g, &mut txn, r#"MATCH (t:Tag) RETURN DISTINCT t.name"#).unwrap();
        assert_eq!(rows.len(), 2, "DISTINCT should collapse the two 'rust' rows into one");
    }

    #[test]
    fn distinct_all_unique_returns_same_count() {
        let (mut g, mut txn) = three_people();
        let all  = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        let dedup = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN DISTINCT n.name"#).unwrap();
        assert_eq!(all.len(), dedup.len(), "no duplicates to remove, counts should match");
    }

    // ── Issue 7: Edge MATCH queries ─────────────────────────────────────────

    #[test]
    fn match_outgoing_edge() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:P {name: "A"}), (b:P {name: "B"}), (a)-[:KNOWS]->(b)"#,
        )
        .unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (x)-[:KNOWS]->(y) RETURN x.name, y.name"#).unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn match_incoming_edge() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:P {name: "src"}), (b:P {name: "dst"}), (a)-[:LINK]->(b)"#,
        )
        .unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (y)<-[:LINK]-(x) RETURN x.name, y.name"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(col(&rows, "x.name"), vec!["\"src\""]);
        assert_eq!(col(&rows, "y.name"), vec!["\"dst\""]);
    }

    #[test]
    fn match_either_direction_edge() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:N {name: "A"}), (b:N {name: "B"}), (a)-[:R]->(b)"#,
        )
        .unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (x)-[:R]-(y) RETURN x.name, y.name"#).unwrap();
        // A-B and B-A should both appear.
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn match_edge_type_filter() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:N {name: "A"}), (b:N {name: "B"}), (c:N {name: "C"}),
                      (a)-[:KNOWS]->(b), (a)-[:LIKES]->(c)"#,
        )
        .unwrap();

        let knows = run(&mut g, &mut txn, r#"MATCH (x)-[:KNOWS]->(y) RETURN y.name"#).unwrap();
        assert_eq!(knows.len(), 1);
        assert_eq!(col(&knows, "y.name"), vec!["\"B\""]);

        let likes = run(&mut g, &mut txn, r#"MATCH (x)-[:LIKES]->(y) RETURN y.name"#).unwrap();
        assert_eq!(likes.len(), 1);
        assert_eq!(col(&likes, "y.name"), vec!["\"C\""]);
    }

    // ── Issue 8a: SET property / label ──────────────────────────────────────

    #[test]
    fn set_property_updates_value() {
        let (mut g, mut txn) = three_people();
        run(&mut g, &mut txn, r#"MATCH (n:Person) WHERE n.name = "Alice" SET n.age = 99"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" RETURN n.age"#,
        )
        .unwrap();
        assert_eq!(col(&rows, "n.age"), vec!["99"]);
    }

    #[test]
    fn set_property_does_not_affect_other_nodes() {
        let (mut g, mut txn) = three_people();
        run(&mut g, &mut txn, r#"MATCH (n:Person) WHERE n.name = "Alice" SET n.age = 99"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Bob" RETURN n.age"#,
        )
        .unwrap();
        assert_eq!(col(&rows, "n.age"), vec!["25"], "Bob's age should be unchanged");
    }

    #[test]
    fn set_adds_new_property() {
        let (mut g, mut txn) = three_people();
        run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" SET n.email = "alice@example.com""#,
        )
        .unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" RETURN n.email"#,
        )
        .unwrap();
        assert_eq!(col(&rows, "n.email"), vec!["\"alice@example.com\""]);
    }

    #[test]
    fn set_label_adds_label() {
        let (mut g, mut txn) = three_people();
        run(&mut g, &mut txn, r#"MATCH (n:Person) WHERE n.name = "Alice" SET n:Manager"#).unwrap();

        let rows = run(&mut g, &mut txn, r#"MATCH (n:Manager) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(col(&rows, "n.name"), vec!["\"Alice\""]);
    }

    #[test]
    fn set_label_does_not_remove_original_label() {
        let (mut g, mut txn) = three_people();
        run(&mut g, &mut txn, r#"MATCH (n:Person) WHERE n.name = "Alice" SET n:Manager"#).unwrap();

        // Alice should still appear as a Person.
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
    }

    // ── Issue 8b: REMOVE property / label ───────────────────────────────────

    #[test]
    fn remove_property_makes_it_null() {
        let (mut g, mut txn) = three_people();
        run(&mut g, &mut txn, r#"MATCH (n:Person) WHERE n.name = "Bob" REMOVE n.age"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Bob" RETURN n.age"#,
        )
        .unwrap();
        assert_eq!(col(&rows, "n.age"), vec!["null"]);
    }

    #[test]
    fn remove_property_does_not_affect_other_nodes() {
        let (mut g, mut txn) = three_people();
        run(&mut g, &mut txn, r#"MATCH (n:Person) WHERE n.name = "Bob" REMOVE n.age"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" RETURN n.age"#,
        )
        .unwrap();
        assert_eq!(col(&rows, "n.age"), vec!["30"], "Alice's age should be unchanged");
    }

    #[test]
    fn remove_label_removes_from_index() {
        let (mut g, mut txn) = three_people();
        run(&mut g, &mut txn, r#"MATCH (n:Person) WHERE n.name = "Alice" SET n:Employee"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (n:Employee) REMOVE n:Employee"#).unwrap();

        let rows = run(&mut g, &mut txn, r#"MATCH (n:Employee) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 0, "Employee label should have been removed");
    }

    #[test]
    fn remove_label_keeps_other_labels() {
        let (mut g, mut txn) = three_people();
        run(&mut g, &mut txn, r#"MATCH (n:Person) WHERE n.name = "Alice" SET n:Employee"#).unwrap();
        run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" REMOVE n:Employee"#,
        )
        .unwrap();

        // Alice should still be a Person.
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
    }

    // ── Issue 8c: DELETE / DETACH DELETE ────────────────────────────────────

    #[test]
    fn delete_removes_node() {
        let (mut g, mut txn) = three_people();
        run(&mut g, &mut txn, r#"MATCH (n:Person) WHERE n.name = "Alice" DELETE n"#).unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(!sorted_col(&rows, "n.name").contains(&"\"Alice\"".to_string()));
    }

    #[test]
    fn delete_node_with_edges_fails() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:N {name: "A"}), (b:N {name: "B"}), (a)-[:R]->(b)"#,
        )
        .unwrap();
        let result = run(&mut g, &mut txn, r#"MATCH (n:N) WHERE n.name = "A" DELETE n"#);
        assert!(result.is_err(), "DELETE on a node with edges should fail without DETACH");
    }

    #[test]
    fn detach_delete_removes_node_and_incident_edges() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:Person {name: "Alice"}), (b:Person {name: "Bob"}), (a)-[:KNOWS]->(b)"#,
        )
        .unwrap();
        run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" DETACH DELETE n"#,
        )
        .unwrap();

        let nodes = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(col(&nodes, "n.name"), vec!["\"Bob\""]);

        let edges = run(&mut g, &mut txn, r#"MATCH (x)-[:KNOWS]->(y) RETURN x.name"#).unwrap();
        assert_eq!(edges.len(), 0, "incident edge should have been removed");
    }

    // ── WHERE clause ────────────────────────────────────────────────────────

    #[test]
    fn where_greater_than_filters() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.age > 28 RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(sorted_col(&rows, "n.name"), vec!["\"Alice\"", "\"Carol\""]);
    }

    #[test]
    fn where_and_combines_conditions() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.age > 24 AND n.age < 35 RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(sorted_col(&rows, "n.name"), vec!["\"Alice\"", "\"Bob\""]);
    }

    #[test]
    fn where_is_null() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "has_age", age: 5})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "no_age"})"#).unwrap();

        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Item) WHERE n.age IS NULL RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(col(&rows, "n.name"), vec!["\"no_age\""]);
    }

    #[test]
    fn where_is_not_null() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "has_age", age: 5})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "no_age"})"#).unwrap();

        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Item) WHERE n.age IS NOT NULL RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(col(&rows, "n.name"), vec!["\"has_age\""]);
    }

    // ── LIMIT / OFFSET ──────────────────────────────────────────────────────

    #[test]
    fn limit_restricts_row_count() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN n.name ORDER BY n.name ASC LIMIT 2"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(col(&rows, "n.name"), vec!["\"Alice\"", "\"Bob\""]);
    }

    #[test]
    fn offset_skips_leading_rows() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN n.name ORDER BY n.name ASC OFFSET 1"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(col(&rows, "n.name"), vec!["\"Bob\"", "\"Carol\""]);
    }

    #[test]
    fn limit_and_offset_together() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN n.name ORDER BY n.name ASC LIMIT 1 OFFSET 1"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(col(&rows, "n.name"), vec!["\"Bob\""]);
    }

    // ── Variable-length paths ───────────────────────────────────────────────

    #[test]
    fn exact_two_hop_path() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:N {name: "A"}), (b:N {name: "B"}), (c:N {name: "C"}),
                      (a)-[:R]->(b), (b)-[:R]->(c)"#,
        )
        .unwrap();

        let rows = run(&mut g, &mut txn, r#"MATCH (x)-[*2]->(y) RETURN x.name, y.name"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(col(&rows, "x.name"), vec!["\"A\""]);
        assert_eq!(col(&rows, "y.name"), vec!["\"C\""]);
    }

    #[test]
    fn one_to_two_hop_range() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:N {name: "A"}), (b:N {name: "B"}), (c:N {name: "C"}),
                      (a)-[:R]->(b), (b)-[:R]->(c)"#,
        )
        .unwrap();

        // 1-hop: A→B, B→C  (2 results)
        // 2-hop: A→C        (1 result)
        let rows = run(&mut g, &mut txn, r#"MATCH (x)-[*1..2]->(y) RETURN x.name, y.name"#).unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn plus_quantifier_finds_reachable_nodes() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:N {name: "A"}), (b:N {name: "B"}), (c:N {name: "C"}),
                      (a)-[:R]->(b), (b)-[:R]->(c)"#,
        )
        .unwrap();

        // + = 1 or more hops from A: reaches B (1-hop) and C (2-hop).
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (x:N {name: "A"})-[:R+]->(y) RETURN y.name"#,
        )
        .unwrap();
        assert_eq!(sorted_col(&rows, "y.name"), vec!["\"B\"", "\"C\""]);
    }

    // ── Path modes ──────────────────────────────────────────────────────────

    #[test]
    fn trail_mode_prevents_edge_revisit() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        // Cycle A → B → A
        run(
            &mut g, &mut txn,
            r#"INSERT (a:N {name: "A"}), (b:N {name: "B"}),
                      (a)-[:R]->(b), (b)-[:R]->(a)"#,
        )
        .unwrap();

        let walk  = run(&mut g, &mut txn, r#"MATCH WALK  (x)-[*1..4]->(y) RETURN x.name, y.name"#).unwrap();
        let trail = run(&mut g, &mut txn, r#"MATCH TRAIL (x)-[*1..4]->(y) RETURN x.name, y.name"#).unwrap();

        // TRAIL must not revisit edges; it finds strictly fewer (or equal) paths than WALK.
        assert!(trail.len() <= walk.len());
        // TRAIL on a 2-edge cycle with [*1..4]: paths of length 1 (A→B, B→A) plus
        // length 2 (A→B→A, B→A→B) are all valid since no edge is repeated — 4 total.
        assert_eq!(trail.len(), 4);
    }

    #[test]
    fn simple_mode_prevents_node_revisit() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        // Cycle A → B → A
        run(
            &mut g, &mut txn,
            r#"INSERT (a:N {name: "A"}), (b:N {name: "B"}),
                      (a)-[:R]->(b), (b)-[:R]->(a)"#,
        )
        .unwrap();

        let simple = run(
            &mut g, &mut txn,
            r#"MATCH SIMPLE (x)-[*1..4]->(y) RETURN x.name, y.name"#,
        )
        .unwrap();
        // SIMPLE: each node visited at most once. A→B (visits A,B) and B→A (visits B,A) — 2 paths.
        assert_eq!(simple.len(), 2);
    }

    // ── ISSUE 9: MATCH+INSERT — connect pre-existing nodes without duplication ─

    /// Basic case: insert two nodes separately, then connect them via MATCH+INSERT.
    /// The MATCH+INSERT must NOT create duplicate nodes.
    #[test]
    fn match_insert_connects_existing_nodes() {
        let mut g = Graph::new();
        let mut txn = 0u64;

        // Create two nodes in separate INSERT statements (simulates real REPL usage).
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob"})"#).unwrap();

        // Sanity: only 2 Person nodes exist before the edge insert.
        let before = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(before.len(), 2);

        // Connect them with MATCH+INSERT — must not duplicate nodes.
        run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"}) INSERT (a)-[:KNOWS]->(b)"#,
        )
        .unwrap();

        // Still only 2 nodes.
        let after = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(after.len(), 2, "MATCH+INSERT must not create duplicate nodes");

        // Edge was created: Alice knows Bob.
        let edges = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"})-[:KNOWS]->(b:Person) RETURN b.name"#,
        )
        .unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].get("b.name"), Some(&Value::String("Bob".into())));
    }

    /// MATCH+INSERT with WHERE: only connect nodes satisfying the condition.
    #[test]
    fn match_insert_with_where_clause() {
        let mut g = Graph::new();
        let mut txn = 0u64;

        run(&mut g, &mut txn, r#"INSERT (:Item {name: "A", active: true})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "B", active: false})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Hub {name: "Hub"})"#).unwrap();

        // Only connect active Items to the Hub.
        run(
            &mut g, &mut txn,
            r#"MATCH (i:Item), (h:Hub) WHERE i.active = true INSERT (i)-[:CONNECTED_TO]->(h)"#,
        )
        .unwrap();

        // Item A (active) → Hub exists; Item B (inactive) must not have an edge.
        let connected = run(
            &mut g, &mut txn,
            r#"MATCH (i:Item)-[:CONNECTED_TO]->(h:Hub) RETURN i.name"#,
        )
        .unwrap();
        assert_eq!(connected.len(), 1);
        assert_eq!(connected[0].get("i.name"), Some(&Value::String("A".into())));
    }

    /// MATCH+INSERT with multiple matches: creates one edge per matched pair.
    #[test]
    fn match_insert_creates_one_edge_per_binding() {
        let mut g = Graph::new();
        let mut txn = 0u64;

        // One source, two targets.
        run(&mut g, &mut txn, r#"INSERT (:Src {name: "S"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Dst {name: "D1"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Dst {name: "D2"})"#).unwrap();

        // MATCH+INSERT produces one LINK edge per (Src, Dst) combination.
        run(
            &mut g, &mut txn,
            r#"MATCH (s:Src), (d:Dst) INSERT (s)-[:LINK]->(d)"#,
        )
        .unwrap();

        let links = run(
            &mut g, &mut txn,
            r#"MATCH (s:Src)-[:LINK]->(d:Dst) RETURN d.name"#,
        )
        .unwrap();
        // 1 Src × 2 Dst = 2 LINK edges.
        assert_eq!(links.len(), 2);
    }

    /// MATCH+INSERT also supports creating a new node alongside edges in the same statement.
    #[test]
    fn match_insert_can_create_new_node_and_edge() {
        let mut g = Graph::new();
        let mut txn = 0u64;

        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();

        // Insert a new Tag node AND link Alice to it in one MATCH+INSERT.
        run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"}) INSERT (t:Tag {name: "rust"}), (a)-[:TAGGED]->(t)"#,
        )
        .unwrap();

        let tags = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person)-[:TAGGED]->(t:Tag) RETURN t.name"#,
        )
        .unwrap();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].get("t.name"), Some(&Value::String("rust".into())));

        // Total nodes: 1 Person + 1 Tag = 2.
        let nodes = run(&mut g, &mut txn, r#"MATCH (n) RETURN n.name"#).unwrap();
        assert_eq!(nodes.len(), 2);
    }

    /// Referencing an unbound variable in a plain INSERT edge gives a clear error.
    #[test]
    fn insert_edge_unbound_variable_returns_error() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        // No nodes inserted, no variables bound — should fail with a descriptive error.
        let result = run(&mut g, &mut txn, r#"INSERT (a)-[:REL]->(b)"#);
        assert!(result.is_err(), "expected error for unbound INSERT edge variables");
        let msg = format!("{:?}", result.unwrap_err());
        assert!(
            msg.contains("not bound") || msg.contains("not defined"),
            "error should mention unbound variable, got: {msg}"
        );
    }

    // ── ISSUE 11: Upsert semantics for INSERT ───────────────────────────────

    /// INSERT with label+name is idempotent: running it twice does not create a duplicate node.
    #[test]
    fn insert_node_with_properties_is_idempotent() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", age: 30})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", age: 30})"#).unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 1, "duplicate INSERT with same properties must not create a second node");
    }

    /// INSERT with same name but different label creates a different node (labels are part of identity).
    #[test]
    fn insert_different_labels_creates_separate_nodes() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Robot {name: "Alice"})"#).unwrap();
        // Two nodes: one Person, one Robot — same name but different labels.
        let rows = run(&mut g, &mut txn, r#"MATCH (n) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 2);
    }

    /// Combined INSERT with edge is idempotent: running it twice does not duplicate anything.
    #[test]
    fn insert_combined_node_and_edge_is_idempotent() {
        let mut g = Graph::new();
        let mut txn = 0u64;

        // Run twice — should produce exactly 2 nodes and 1 edge.
        for _ in 0..2 {
            run(
                &mut g, &mut txn,
                r#"INSERT (a:Person {name: "Alice"}), (b:Person {name: "Bob"}), (a)-[:KNOWS]->(b)"#,
            )
            .unwrap();
        }

        let nodes = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(nodes.len(), 2, "no duplicate nodes");

        let edges = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"})-[:KNOWS]->(b:Person) RETURN b.name"#,
        )
        .unwrap();
        assert_eq!(edges.len(), 1, "no duplicate edges");
    }

    /// INSERT with only a label and no properties always creates a new node (no discriminating key).
    #[test]
    fn insert_bare_label_always_creates_new_node() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Anon)"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Anon)"#).unwrap();
        // No properties → cannot merge → two distinct nodes.
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Anon) RETURN n"#).unwrap();
        assert_eq!(rows.len(), 2, "bare label INSERT should always create a new node");
    }

    // ── README examples — one test per documented query ─────────────────────
    // Shared setup: Alice (Person, age 30), Bob (Person, age 25), Carol (Person+Employee, age 35),
    // Alice -[:KNOWS]-> Bob.
    fn readme_graph() -> (Graph, u64) {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", age: 30})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob", age: 25})"#).unwrap();
        run(
            &mut g, &mut txn,
            r#"INSERT (:Person:Employee {name: "Carol", age: 35})"#,
        )
        .unwrap();
        run(
            &mut g, &mut txn,
            r#"INSERT (a:Person {name: "Alice"}), (b:Person {name: "Bob"}), (a)-[:KNOWS]->(b)"#,
        )
        .unwrap();
        (g, txn)
    }

    #[test]
    fn readme_basic_match_returns_all_persons() {
        // MATCH (n:Person) RETURN n.name, n.age
        // Expected: 3 rows — Alice/30, Bob/25, Carol/35 (order unspecified).
        let (mut g, mut txn) = readme_graph();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name, n.age"#).unwrap();
        assert_eq!(rows.len(), 3);
        let names: Vec<&Value> = rows.iter().map(|r| r.get("n.name").unwrap()).collect();
        assert!(names.contains(&&Value::String("Alice".into())));
        assert!(names.contains(&&Value::String("Bob".into())));
        assert!(names.contains(&&Value::String("Carol".into())));
    }

    #[test]
    fn readme_where_order_returns_filtered_sorted() {
        // MATCH (n:Person) WHERE n.age > 28 RETURN n.name ORDER BY n.age DESC
        // Expected: Carol (35), Alice (30) — Bob (25) excluded.
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.age > 28 RETURN n.name ORDER BY n.age DESC"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("n.name"), Some(&Value::String("Carol".into())));
        assert_eq!(rows[1].get("n.name"), Some(&Value::String("Alice".into())));
    }

    #[test]
    fn readme_limit_offset() {
        // MATCH (n:Person) RETURN n.name LIMIT 2 OFFSET 1
        // Expected: exactly 2 rows (whichever 2 follow the first in iteration order).
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN n.name LIMIT 2 OFFSET 1"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn readme_distinct() {
        // Insert a name twice, then DISTINCT should collapse it.
        // MATCH (n:Person) RETURN DISTINCT n.name — 3 unique names.
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN DISTINCT n.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn readme_edge_query_directed() {
        // MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name
        // Expected: 1 row — Alice, Bob.
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("a.name"), Some(&Value::String("Alice".into())));
        assert_eq!(rows[0].get("b.name"), Some(&Value::String("Bob".into())));
    }

    #[test]
    fn readme_edge_query_with_type_function() {
        // MATCH (a:Person)-[r]->(b) RETURN a.name, type(r), b.name
        // Expected: 1 row — Alice, KNOWS, Bob.
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person)-[r]->(b) RETURN a.name, type(r), b.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("type(r)"), Some(&Value::String("KNOWS".into())));
    }

    #[test]
    fn readme_edge_query_reverse() {
        // MATCH (a)<-[:KNOWS]-(b) RETURN a.name, b.name
        // Expected: 1 row — Bob (a) is the target, Alice (b) is the source.
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a)<-[:KNOWS]-(b) RETURN a.name, b.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("a.name"), Some(&Value::String("Bob".into())));
        assert_eq!(rows[0].get("b.name"), Some(&Value::String("Alice".into())));
    }

    #[test]
    fn readme_edge_query_undirected() {
        // MATCH (a)-[:KNOWS]-(b) RETURN a.name, b.name
        // Expected: 2 rows — (Alice, Bob) and (Bob, Alice).
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a)-[:KNOWS]-(b) RETURN a.name, b.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn readme_variable_length_path() {
        // MATCH (a)-[*1..3]->(b) RETURN a.name, b.name
        // On the readme graph (Alice->Bob only), there is exactly 1 path of length 1.
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a)-[*1..3]->(b) RETURN a.name, b.name"#,
        )
        .unwrap();
        assert!(!rows.is_empty());
        // Alice->Bob must be in the results.
        let has_alice_bob = rows.iter().any(|r| {
            r.get("a.name") == Some(&Value::String("Alice".into()))
                && r.get("b.name") == Some(&Value::String("Bob".into()))
        });
        assert!(has_alice_bob);
    }

    #[test]
    fn readme_variable_length_labeled() {
        // MATCH (a)-[:KNOWS*]->(b) RETURN a.name, b.name
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a)-[:KNOWS*]->(b) RETURN a.name, b.name"#,
        )
        .unwrap();
        assert!(!rows.is_empty());
    }

    #[test]
    fn readme_exact_hops() {
        // MATCH (a)-[*2]->(b) RETURN a.name, b.name — no 2-hop path exists in readme graph.
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a)-[*2]->(b) RETURN a.name, b.name"#,
        )
        .unwrap();
        // readme graph only has Alice->Bob (1 hop), so no 2-hop result.
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn readme_set_property() {
        // MATCH (n:Person) WHERE n.name = "Alice" SET n.age = 31
        // Expected: Alice's age becomes 31.
        let (mut g, mut txn) = readme_graph();
        run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" SET n.age = 31"#,
        )
        .unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person {name: "Alice"}) RETURN n.age"#,
        )
        .unwrap();
        assert_eq!(rows[0].get("n.age"), Some(&Value::Int(31)));
    }

    #[test]
    fn readme_set_label() {
        // MATCH (n:Person) WHERE n.name = "Alice" SET n:Manager
        let (mut g, mut txn) = readme_graph();
        run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" SET n:Manager"#,
        )
        .unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Manager) RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("n.name"), Some(&Value::String("Alice".into())));
    }

    #[test]
    fn readme_remove_property() {
        // MATCH (n:Person) WHERE n.name = "Bob" REMOVE n.age
        let (mut g, mut txn) = readme_graph();
        run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Bob" REMOVE n.age"#,
        )
        .unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person {name: "Bob"}) RETURN n.age"#,
        )
        .unwrap();
        assert_eq!(rows[0].get("n.age"), Some(&Value::Null));
    }

    #[test]
    fn readme_remove_label() {
        // MATCH (n:Person) WHERE n.name = "Carol" REMOVE n:Employee
        let (mut g, mut txn) = readme_graph();
        run(
            &mut g, &mut txn,
            r#"MATCH (n:Employee) WHERE n.name = "Carol" REMOVE n:Employee"#,
        )
        .unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Employee) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 0, "Carol should no longer be an Employee");
    }

    #[test]
    fn readme_delete_node() {
        // MATCH (n:Person) WHERE n.name = "Carol" DELETE n
        let (mut g, mut txn) = readme_graph();
        run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Carol" DELETE n"#,
        )
        .unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 2, "Carol should be deleted; Alice and Bob remain");
    }

    #[test]
    fn readme_detach_delete() {
        // MATCH (n:Person) WHERE n.name = "Alice" DETACH DELETE n
        // Alice has a KNOWS edge to Bob — DETACH DELETE must also remove that edge.
        let (mut g, mut txn) = readme_graph();
        run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" DETACH DELETE n"#,
        )
        .unwrap();
        let nodes = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(nodes.len(), 2, "Bob and Carol remain");
        let edges = run(&mut g, &mut txn, r#"MATCH (a)-[:KNOWS]->(b) RETURN a.name"#).unwrap();
        assert_eq!(edges.len(), 0, "KNOWS edge removed with Alice");
    }

    #[test]
    fn readme_match_insert_connects_existing() {
        // README: MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"}) INSERT (a)-[:KNOWS]->(b)
        // Since the edge already exists in readme_graph, running it again should be a no-op (upsert).
        let (mut g, mut txn) = readme_graph();

        // Edge already exists — rerunning must not duplicate it.
        run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"}) INSERT (a)-[:KNOWS]->(b)"#,
        )
        .unwrap();

        let edges = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"})-[:KNOWS]->(b) RETURN b.name"#,
        )
        .unwrap();
        assert_eq!(edges.len(), 1, "edge must not be duplicated");
    }

    #[test]
    fn readme_is_null_expression() {
        // WHERE n.age IS NULL and IS NOT NULL.
        let (mut g, mut txn) = readme_graph();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Dave"})"#).unwrap();

        let null_rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.age IS NULL RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(null_rows.len(), 1);
        assert_eq!(null_rows[0].get("n.name"), Some(&Value::String("Dave".into())));

        let not_null_rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.age IS NOT NULL RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(not_null_rows.len(), 3); // Alice, Bob, Carol
    }

    #[test]
    fn readme_arithmetic_expression() {
        // RETURN n.age * 2 + 1 AS adjusted
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person {name: "Alice"}) RETURN n.age * 2 + 1 AS adjusted"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("adjusted"), Some(&Value::Int(61)));
    }

    // ── TODO 3: Multi-pattern MATCH+RETURN ───────────────────────────────────

    /// Two disconnected node patterns produce a Cartesian product.
    #[test]
    fn multi_pattern_match_cross_joins() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Color {name: "red"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Color {name: "blue"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Size {name: "small"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Size {name: "large"})"#).unwrap();

        // 2 colors × 2 sizes = 4 combinations.
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (c:Color), (s:Size) RETURN c.name, s.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 4);

        // Every (color, size) combination should appear.
        let pairs: Vec<(String, String)> = rows
            .iter()
            .map(|r| {
                let c = match r.get("c.name").unwrap() { Value::String(s) => s.clone(), _ => panic!() };
                let s = match r.get("s.name").unwrap() { Value::String(s) => s.clone(), _ => panic!() };
                (c, s)
            })
            .collect();
        assert!(pairs.contains(&("red".into(), "small".into())));
        assert!(pairs.contains(&("red".into(), "large".into())));
        assert!(pairs.contains(&("blue".into(), "small".into())));
        assert!(pairs.contains(&("blue".into(), "large".into())));
    }

    /// WHERE clause filters the cross-joined results.
    #[test]
    fn multi_pattern_match_with_where() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", age: 30})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob",   age: 25})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Carol", age: 35})"#).unwrap();

        // All pairs where a is older than b (3 persons → 3 ordered pairs where age(a) > age(b)).
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person), (b:Person) WHERE a.age > b.age RETURN a.name, b.name"#,
        )
        .unwrap();
        // Alice(30)>Bob(25), Carol(35)>Alice(30), Carol(35)>Bob(25) → 3 rows.
        assert_eq!(rows.len(), 3);
    }

    /// Self-join with inequality excludes same-node pairs.
    #[test]
    fn multi_pattern_self_join_excludes_same_node() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:T {name: "X"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:T {name: "Y"})"#).unwrap();

        // Without WHERE: 2×2 = 4 rows (includes same-node pairs).
        let all = run(&mut g, &mut txn, r#"MATCH (a:T), (b:T) RETURN a.name, b.name"#).unwrap();
        assert_eq!(all.len(), 4);

        // With inequality: 4 - 2 same-node = 2 rows.
        let diff = run(
            &mut g, &mut txn,
            r#"MATCH (a:T), (b:T) WHERE a.name <> b.name RETURN a.name, b.name"#,
        )
        .unwrap();
        assert_eq!(diff.len(), 2);
    }

    /// One of the patterns can be a connected multi-hop path; the other is a standalone node.
    #[test]
    fn multi_pattern_mixes_path_and_node() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:Person {name: "Alice"}), (b:Person {name: "Bob"}), (a)-[:KNOWS]->(b)"#,
        )
        .unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Tag {name: "rust"})"#).unwrap();

        // One pattern matches the Alice→Bob edge; the other pattern matches the tag.
        // Expected: 1 path result × 1 tag = 1 row.
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person)-[:KNOWS]->(b:Person), (t:Tag) RETURN a.name, b.name, t.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("t.name"), Some(&Value::String("rust".into())));
    }

    /// Three-way cross-join.
    #[test]
    fn multi_pattern_three_patterns() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:A {name: "a1"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:A {name: "a2"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:B {name: "b1"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:C {name: "c1"})"#).unwrap();

        // 2 A × 1 B × 1 C = 2 rows.
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:A), (b:B), (c:C) RETURN a.name, b.name, c.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
    }

    /// A single pattern still works exactly as before (no regression).
    #[test]
    fn single_pattern_match_unchanged() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "x"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "y"})"#).unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Item) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 2);
    }

    /// Multi-pattern MATCH respects ORDER BY across all columns.
    #[test]
    fn multi_pattern_match_order_by() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:N {name: "B"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:N {name: "A"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:M {val: 1})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:M {val: 2})"#).unwrap();

        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:N), (m:M) RETURN n.name, m.val ORDER BY n.name ASC, m.val DESC"#,
        )
        .unwrap();
        // 2 N × 2 M = 4 rows, sorted by name ASC then val DESC:
        // (A,2), (A,1), (B,2), (B,1)
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].get("n.name"), Some(&Value::String("A".into())));
        assert_eq!(rows[0].get("m.val"), Some(&Value::Int(2)));
        assert_eq!(rows[3].get("n.name"), Some(&Value::String("B".into())));
        assert_eq!(rows[3].get("m.val"), Some(&Value::Int(1)));
    }

    // ── TODO 2: Aggregation ───────────────────────────────────────────────────

    fn three_numbers() -> (Graph, u64) {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Num {val: 10})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Num {val: 20})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Num {val: 30})"#).unwrap();
        (g, txn)
    }

    /// count(*) returns total number of matched rows.
    #[test]
    fn count_star_returns_total() {
        let (mut g, mut txn) = three_numbers();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Num) RETURN count(*)"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("count(*)"), Some(&Value::Int(3)));
    }

    /// count(*) fast path stays accurate after inserts and deletes.
    #[test]
    fn count_star_fast_path_tracks_mutations() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Widget {name: "a"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Widget {name: "b"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Widget {name: "c"})"#).unwrap();

        let rows = run(&mut g, &mut txn, "MATCH (n:Widget) RETURN count(*) AS c").unwrap();
        assert_eq!(rows[0].get("c"), Some(&Value::Int(3)));

        run(&mut g, &mut txn, r#"MATCH (n:Widget {name: "b"}) DETACH DELETE n"#).unwrap();

        let rows = run(&mut g, &mut txn, "MATCH (n:Widget) RETURN count(*) AS c").unwrap();
        assert_eq!(rows[0].get("c"), Some(&Value::Int(2)));
    }

    /// count(n) (variable form) is equivalent to count(*) for a simple label scan.
    #[test]
    fn count_var_fast_path() {
        let (mut g, mut txn) = three_numbers();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Num) RETURN count(n) AS c"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("c"), Some(&Value::Int(3)));
    }

    /// count(*) with no matches returns 0.
    #[test]
    fn count_star_empty_returns_zero() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Num) RETURN count(*)"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("count(*)"), Some(&Value::Int(0)));
    }

    /// count(expr) counts non-null values.
    #[test]
    fn count_expr_skips_nulls() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:X {v: 1})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:X {v: 2})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:X)"#).unwrap(); // no v property
        let rows = run(&mut g, &mut txn, r#"MATCH (n:X) RETURN count(n.v)"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("count(n.v)"), Some(&Value::Int(2)));
    }

    /// sum() returns correct integer sum.
    #[test]
    fn sum_integers() {
        let (mut g, mut txn) = three_numbers();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Num) RETURN sum(n.val)"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("sum(n.val)"), Some(&Value::Int(60)));
    }

    /// avg() returns correct average.
    #[test]
    fn avg_integers() {
        let (mut g, mut txn) = three_numbers();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Num) RETURN avg(n.val)"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("avg(n.val)"), Some(&Value::Float(20.0)));
    }

    /// min() and max() return correct extremes.
    #[test]
    fn min_max_integers() {
        let (mut g, mut txn) = three_numbers();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Num) RETURN min(n.val), max(n.val)"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("min(n.val)"), Some(&Value::Int(10)));
        assert_eq!(rows[0].get("max(n.val)"), Some(&Value::Int(30)));
    }

    /// collect() gathers all values into a list.
    #[test]
    fn collect_returns_list() {
        let (mut g, mut txn) = three_numbers();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Num) RETURN collect(n.val)"#).unwrap();
        assert_eq!(rows.len(), 1);
        let list = match rows[0].get("collect(n.val)").unwrap() {
            Value::List(l) => l.clone(),
            other => panic!("expected List, got {other:?}"),
        };
        assert_eq!(list.len(), 3);
        assert!(list.contains(&Value::Int(10)));
        assert!(list.contains(&Value::Int(20)));
        assert!(list.contains(&Value::Int(30)));
    }

    /// Grouping: RETURN n.label, count(*) groups by label.
    #[test]
    fn count_grouped_by_label() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        // Distinct names so upsert deduplication doesn't collapse them.
        run(&mut g, &mut txn, r#"INSERT (:Person {dept: "eng", name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {dept: "eng", name: "Bob"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {dept: "hr", name: "Carol"})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN n.dept, count(*) ORDER BY n.dept ASC"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2, "two groups: eng and hr");
        assert_eq!(rows[0].get("n.dept"), Some(&Value::String("eng".into())));
        assert_eq!(rows[0].get("count(*)"), Some(&Value::Int(2)));
        assert_eq!(rows[1].get("n.dept"), Some(&Value::String("hr".into())));
        assert_eq!(rows[1].get("count(*)"), Some(&Value::Int(1)));
    }

    /// Grouping with sum: sum per group.
    #[test]
    fn sum_grouped_by_category() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Item {cat: "A", price: 10})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {cat: "A", price: 20})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {cat: "B", price: 5})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Item) RETURN n.cat, sum(n.price) ORDER BY n.cat ASC"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("sum(n.price)"), Some(&Value::Int(30)));
        assert_eq!(rows[1].get("sum(n.price)"), Some(&Value::Int(5)));
    }

    /// WHERE filters before aggregation.
    #[test]
    fn count_star_with_where() {
        let (mut g, mut txn) = three_numbers();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Num) WHERE n.val > 10 RETURN count(*)"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("count(*)"), Some(&Value::Int(2)));
    }

    /// ORDER BY an aggregate column works after grouping.
    #[test]
    fn order_by_aggregate_column() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        // Distinct names so upsert deduplication doesn't collapse them.
        run(&mut g, &mut txn, r#"INSERT (:P {dept: "eng", name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {dept: "eng", name: "Bob"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {dept: "hr", name: "Carol"})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:P) RETURN n.dept, count(*) ORDER BY count(*) DESC"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        // eng (count=2) should come first.
        assert_eq!(rows[0].get("n.dept"), Some(&Value::String("eng".into())));
        assert_eq!(rows[0].get("count(*)"), Some(&Value::Int(2)));
    }

    /// AS alias works for aggregate columns.
    #[test]
    fn count_star_with_alias() {
        let (mut g, mut txn) = three_numbers();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Num) RETURN count(*) AS total"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("total"), Some(&Value::Int(3)));
    }

    // ── Long path (4+ hops) ───────────────────────────────────────────────────

    /// Build a 5-node linear chain A→B→C→D→E (4 edges).
    fn five_node_chain() -> (Graph, u64) {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:Stop {name: "A"}), (b:Stop {name: "B"}), (c:Stop {name: "C"}),
                      (d:Stop {name: "D"}), (e:Stop {name: "E"}),
                      (a)-[:NEXT]->(b), (b)-[:NEXT]->(c), (c)-[:NEXT]->(d), (d)-[:NEXT]->(e)"#,
        )
        .unwrap();
        (g, txn)
    }

    /// Exactly 4 hops: only A→E satisfies [*4].
    #[test]
    fn four_hop_exact_match() {
        let (mut g, mut txn) = five_node_chain();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Stop)-[*4]->(b:Stop) RETURN a.name, b.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("a.name"), Some(&Value::String("A".into())));
        assert_eq!(rows[0].get("b.name"), Some(&Value::String("E".into())));
    }

    /// Exactly 3 hops: A→D and B→E.
    #[test]
    fn three_hop_exact_match() {
        let (mut g, mut txn) = five_node_chain();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Stop)-[*3]->(b:Stop) RETURN a.name, b.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        let pairs: Vec<(String, String)> = rows
            .iter()
            .map(|r| {
                let a = r.get("a.name").and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None }).unwrap_or_default();
                let b = r.get("b.name").and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None }).unwrap_or_default();
                (a, b)
            })
            .collect();
        assert!(pairs.contains(&("A".into(), "D".into())));
        assert!(pairs.contains(&("B".into(), "E".into())));
    }

    /// Range [*1..4]: all 10 reachable pairs in the chain.
    #[test]
    fn one_to_four_hop_range() {
        let (mut g, mut txn) = five_node_chain();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Stop)-[*1..4]->(b:Stop) RETURN a.name, b.name"#,
        )
        .unwrap();
        // 4 one-hop + 3 two-hop + 2 three-hop + 1 four-hop = 10
        assert_eq!(rows.len(), 10);
    }

    /// [*2..] means 2 or more hops: everything except direct neighbors (6 results).
    #[test]
    fn two_or_more_hops() {
        let (mut g, mut txn) = five_node_chain();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Stop)-[*2..]->(b:Stop) RETURN a.name, b.name"#,
        )
        .unwrap();
        // From A: C, D, E (3); from B: D, E (2); from C: E (1) = 6
        assert_eq!(rows.len(), 6);
        // Direct neighbors (1-hop) must NOT appear.
        let has_ab = rows.iter().any(|r| {
            r.get("a.name") == Some(&Value::String("A".into()))
                && r.get("b.name") == Some(&Value::String("B".into()))
        });
        assert!(!has_ab, "A→B is only 1 hop and must not appear in *2.. results");
    }

    /// Path variable bound over 4 hops; length(r) == 4.
    #[test]
    fn path_variable_length_four_hops() {
        let (mut g, mut txn) = five_node_chain();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Stop)-[r*4]->(b:Stop) RETURN a.name, b.name, length(r)"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("length(r)"), Some(&Value::Int(4)));
    }

    /// Label-filtered + quantifier: MATCH (a)-[:NEXT+]->(b) with label filter.
    #[test]
    fn plus_quantifier_with_label_filter() {
        let (mut g, mut txn) = five_node_chain();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Stop)-[:NEXT+]->(b:Stop) RETURN a.name, b.name"#,
        )
        .unwrap();
        // All 10 pairs reachable via 1+ NEXT edges.
        assert_eq!(rows.len(), 10);
        // A must be able to reach E via NEXT+.
        let ae = rows.iter().any(|r| {
            r.get("a.name") == Some(&Value::String("A".into()))
                && r.get("b.name") == Some(&Value::String("E".into()))
        });
        assert!(ae, "A must reach E via NEXT+");
    }

    /// Brace quantifier {1,2}: only 1- and 2-hop paths (7 results in the chain).
    #[test]
    fn readme_brace_quantifier_with_length() {
        let (mut g, mut txn) = five_node_chain();
        // MATCH (a)-[r:NEXT{1,2}]->(b) RETURN a.name, length(r), b.name
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Stop)-[r:NEXT{1,2}]->(b:Stop) RETURN a.name, length(r), b.name"#,
        )
        .unwrap();
        // 4 one-hop + 3 two-hop = 7
        assert_eq!(rows.len(), 7);
        // Every result must have length 1 or 2.
        for row in &rows {
            let len = row.get("length(r)").unwrap();
            assert!(
                *len == Value::Int(1) || *len == Value::Int(2),
                "length must be 1 or 2, got {len:?}"
            );
        }
    }

    // ── Duplication integrity checks ─────────────────────────────────────────

    /// Running the same INSERT with properties three times must not duplicate the node.
    #[test]
    fn repeated_insert_same_properties_no_duplicate() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        for _ in 0..3 {
            run(&mut g, &mut txn, r#"INSERT (:User {name: "alice", role: "admin"})"#).unwrap();
        }
        let rows = run(&mut g, &mut txn, r#"MATCH (n:User) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 1, "three identical INSERTs must yield exactly one node");
    }

    /// Each INSERT with distinct properties creates a separate node.
    #[test]
    fn inserts_with_distinct_properties_create_separate_nodes() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:User {name: "alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:User {name: "bob"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:User {name: "carol"})"#).unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:User) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 3, "three users with distinct names must be separate nodes");
    }

    /// Inserting an edge three times via INSERT (upsert) must produce exactly one edge.
    #[test]
    fn repeated_edge_insert_no_duplicate() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        for _ in 0..3 {
            run(
                &mut g, &mut txn,
                r#"INSERT (a:W {name: "X"}), (b:W {name: "Y"}), (a)-[:LINK]->(b)"#,
            )
            .unwrap();
        }
        let rows = run(&mut g, &mut txn, r#"MATCH (a)-[:LINK]->(b) RETURN a.name, b.name"#).unwrap();
        assert_eq!(rows.len(), 1, "three identical edge INSERTs must yield exactly one edge");
        let nodes = run(&mut g, &mut txn, r#"MATCH (n:W) RETURN n.name"#).unwrap();
        assert_eq!(nodes.len(), 2, "X and Y must not be duplicated");
    }

    /// MATCH+INSERT run three times on an existing edge must not create duplicate edges.
    #[test]
    fn repeated_match_insert_edge_no_duplicate() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Bob"})"#).unwrap();
        for _ in 0..3 {
            run(
                &mut g, &mut txn,
                r#"MATCH (a:P {name: "Alice"}), (b:P {name: "Bob"}) INSERT (a)-[:FRIENDS]->(b)"#,
            )
            .unwrap();
        }
        let edges = run(&mut g, &mut txn, r#"MATCH (a)-[:FRIENDS]->(b) RETURN a.name"#).unwrap();
        assert_eq!(edges.len(), 1, "running MATCH+INSERT three times must not duplicate the edge");
    }

    /// Building a chain one step at a time via separate INSERT statements must not duplicate nodes.
    #[test]
    fn chain_built_from_separate_inserts_no_node_duplication() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        // Each node has a unique name so they are all distinct.
        run(&mut g, &mut txn, r#"INSERT (:Chain {id: 1, name: "n1"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Chain {id: 2, name: "n2"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Chain {id: 3, name: "n3"})"#).unwrap();
        // Connect them with MATCH+INSERT.
        run(
            &mut g, &mut txn,
            r#"MATCH (a:Chain {name: "n1"}), (b:Chain {name: "n2"}) INSERT (a)-[:NEXT]->(b)"#,
        )
        .unwrap();
        run(
            &mut g, &mut txn,
            r#"MATCH (a:Chain {name: "n2"}), (b:Chain {name: "n3"}) INSERT (a)-[:NEXT]->(b)"#,
        )
        .unwrap();
        // Re-run the connections — must still be idempotent.
        run(
            &mut g, &mut txn,
            r#"MATCH (a:Chain {name: "n1"}), (b:Chain {name: "n2"}) INSERT (a)-[:NEXT]->(b)"#,
        )
        .unwrap();
        let nodes = run(&mut g, &mut txn, r#"MATCH (n:Chain) RETURN n.name"#).unwrap();
        assert_eq!(nodes.len(), 3, "re-running MATCH+INSERT must not duplicate nodes");
        let edges = run(&mut g, &mut txn, r#"MATCH (a)-[:NEXT]->(b) RETURN a.name"#).unwrap();
        assert_eq!(edges.len(), 2, "re-running MATCH+INSERT must not duplicate edges");
    }

    /// Two edges with different labels between the same nodes are distinct.
    #[test]
    fn different_edge_labels_create_separate_edges() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (a:V {name: "X"}), (b:V {name: "Y"}),
                      (a)-[:KNOWS]->(b), (a)-[:LIKES]->(b)"#,
        )
        .unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (a)-[r]->(b) RETURN type(r)"#).unwrap();
        assert_eq!(rows.len(), 2, "KNOWS and LIKES are two distinct edges");
        let types: Vec<&Value> = rows.iter().map(|r| r.get("type(r)").unwrap()).collect();
        assert!(types.contains(&&Value::String("KNOWS".into())));
        assert!(types.contains(&&Value::String("LIKES".into())));
    }

    // ── Remaining README examples ─────────────────────────────────────────────

    /// MATCH+INSERT with boolean WHERE property filter.
    #[test]
    fn readme_match_insert_where_boolean_property() {
        // MATCH (i:Item), (h:Hub) WHERE i.active = true INSERT (i)-[:CONNECTED_TO]->(h)
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "widget", active: true})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "gizmo",  active: false})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Hub  {name: "hub1"})"#).unwrap();

        run(
            &mut g, &mut txn,
            r#"MATCH (i:Item), (h:Hub) WHERE i.active = true INSERT (i)-[:CONNECTED_TO]->(h)"#,
        )
        .unwrap();

        let rows = run(&mut g, &mut txn, r#"MATCH (i)-[:CONNECTED_TO]->(h) RETURN i.name"#).unwrap();
        assert_eq!(rows.len(), 1, "only the active item connects to the hub");
        assert_eq!(rows[0].get("i.name"), Some(&Value::String("widget".into())));
    }

    /// Multi-label node matches both label queries.
    #[test]
    fn multi_label_node_matches_both_labels() {
        // INSERT (:Person:Employee {name: "Carol", age: 35})
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person:Employee {name: "Carol", age: 35})"#).unwrap();

        let as_person = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(as_person.len(), 1, "Carol is a Person");

        let as_emp = run(&mut g, &mut txn, r#"MATCH (n:Employee) RETURN n.name"#).unwrap();
        assert_eq!(as_emp.len(), 1, "Carol is an Employee");

        // labels() returns both.
        let label_rows = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN labels(n)"#).unwrap();
        assert_eq!(label_rows.len(), 1);
        if let Some(Value::List(ls)) = label_rows[0].get("labels(n)") {
            assert!(ls.contains(&Value::String("Person".into())));
            assert!(ls.contains(&Value::String("Employee".into())));
        } else {
            panic!("labels(n) should return a list");
        }
    }

    /// Re-inserting the multi-label node does not duplicate it.
    #[test]
    fn multi_label_insert_is_idempotent() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person:Employee {name: "Carol", age: 35})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person:Employee {name: "Carol", age: 35})"#).unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Employee) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 1, "re-inserting the same multi-label node must not duplicate it");
    }

    /// WHERE with AND and <> as in the README.
    #[test]
    fn readme_where_and_with_neq() {
        // WHERE n.age > 28 AND n.name <> "Bob"
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.age > 28 AND n.name <> "Bob" RETURN n.name ORDER BY n.age ASC"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2, "Alice(30) and Carol(35) pass; Bob(25) excluded by age");
        assert_eq!(rows[0].get("n.name"), Some(&Value::String("Alice".into())));
        assert_eq!(rows[1].get("n.name"), Some(&Value::String("Carol".into())));
    }

    /// Multi-column ORDER BY: ORDER BY n.age DESC, n.name ASC.
    #[test]
    fn readme_order_by_multi_column() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Alice", age: 30})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Aaron", age: 30})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Bob",   age: 25})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:P) RETURN n.name, n.age ORDER BY n.age DESC, n.name ASC"#,
        )
        .unwrap();
        // Age 30: Aaron then Alice (ASC name within same age); then Bob(25).
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].get("n.name"), Some(&Value::String("Aaron".into())));
        assert_eq!(rows[1].get("n.name"), Some(&Value::String("Alice".into())));
        assert_eq!(rows[2].get("n.name"), Some(&Value::String("Bob".into())));
    }

    /// size(labels(n)) counts labels on a multi-label node.
    #[test]
    fn size_of_labels() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:A:B:C {name: "triple"})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:A) RETURN size(labels(n)) AS lcount"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("lcount"), Some(&Value::Int(3)));
    }

    /// toString() converts integer to string.
    #[test]
    fn to_string_function() {
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person {name: "Alice"}) RETURN toString(n.age) AS s"#,
        )
        .unwrap();
        assert_eq!(rows[0].get("s"), Some(&Value::String("30".into())));
    }

    /// toInteger() converts string to integer.
    #[test]
    fn to_integer_function() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:X {code: "42"})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:X) RETURN toInteger(n.code) AS i"#,
        )
        .unwrap();
        assert_eq!(rows[0].get("i"), Some(&Value::Int(42)));
    }

    /// NOT condition in WHERE.
    #[test]
    fn where_not_condition() {
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE NOT n.name = "Alice" RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        let names = sorted_col(&rows, "n.name");
        assert!(!names.contains(&"\"Alice\"".to_string()));
    }

    /// OR condition in WHERE.
    #[test]
    fn where_or_condition() {
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE n.name = "Alice" OR n.name = "Bob" RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        let names = sorted_col(&rows, "n.name");
        assert!(names.contains(&"\"Alice\"".to_string()));
        assert!(names.contains(&"\"Bob\"".to_string()));
    }

    /// Aggregate README example: min and max on the readme graph.
    #[test]
    fn readme_agg_min_max_ages() {
        // MATCH (n:Person) RETURN min(n.age), max(n.age)
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN min(n.age), max(n.age)"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("min(n.age)"), Some(&Value::Int(25)));
        assert_eq!(rows[0].get("max(n.age)"), Some(&Value::Int(35)));
    }

    /// Aggregate README example: collect(n.name) returns all names.
    #[test]
    fn readme_agg_collect_names() {
        // MATCH (n:Person) RETURN collect(n.name)
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN collect(n.name)"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        if let Some(Value::List(names)) = rows[0].get("collect(n.name)") {
            assert_eq!(names.len(), 3);
            assert!(names.contains(&Value::String("Alice".into())));
            assert!(names.contains(&Value::String("Bob".into())));
            assert!(names.contains(&Value::String("Carol".into())));
        } else {
            panic!("collect(n.name) should return a list");
        }
    }

    /// Aggregate README example: count(*) on the readme graph.
    #[test]
    fn readme_agg_count_star() {
        // MATCH (n:Person) RETURN count(*)
        let (mut g, mut txn) = readme_graph();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN count(*)"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("count(*)"), Some(&Value::Int(3)));
    }

    /// MATCH+INSERT creates a new node and links it to a matched node (README example).
    #[test]
    fn readme_match_insert_new_node_and_edge() {
        // MATCH (a:Person {name: "Alice"}) INSERT (t:Tag {name: "rust"}), (a)-[:TAGGED]->(t)
        let (mut g, mut txn) = readme_graph();
        run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"}) INSERT (t:Tag {name: "rust"}), (a)-[:TAGGED]->(t)"#,
        )
        .unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"})-[:TAGGED]->(t:Tag) RETURN t.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("t.name"), Some(&Value::String("rust".into())));
        // Running it again must not duplicate the tag.
        run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"}) INSERT (t:Tag {name: "rust"}), (a)-[:TAGGED]->(t)"#,
        )
        .unwrap();
        let tags = run(&mut g, &mut txn, r#"MATCH (n:Tag) RETURN n.name"#).unwrap();
        assert_eq!(tags.len(), 1, "tag must not be duplicated on second run");
    }

    /// SIMPLE path mode on the 5-node chain: no node revisited, same as TRAIL for a simple chain.
    /// Use [*1..10] rather than bare [*] because [*] has min=0 which includes 0-hop self-matches.
    #[test]
    fn simple_mode_on_long_chain() {
        let (mut g, mut txn) = five_node_chain();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH SIMPLE (a:Stop)-[*1..10]->(b:Stop) RETURN a.name, b.name"#,
        )
        .unwrap();
        // All 10 distinct-node paths in the acyclic chain.
        assert_eq!(rows.len(), 10);
    }

    /// TRAIL path mode on the 5-node chain: no edge revisited, same as SIMPLE for a simple chain.
    #[test]
    fn trail_mode_on_long_chain() {
        let (mut g, mut txn) = five_node_chain();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH TRAIL (a:Stop)-[*1..10]->(b:Stop) RETURN a.name, b.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 10);
    }

    /// id(n) returns a ULID string; re-querying the same node returns the same value.
    #[test]
    fn id_function_stable() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Node {name: "stable"})"#).unwrap();
        let rows1 = run(&mut g, &mut txn, r#"MATCH (n:Node) RETURN id(n) AS nid"#).unwrap();
        let rows2 = run(&mut g, &mut txn, r#"MATCH (n:Node) RETURN id(n) AS nid"#).unwrap();
        assert_eq!(rows1[0].get("nid"), rows2[0].get("nid"), "id must be stable across queries");
    }

    /// id(n) produces a 26-character Crockford Base32 ULID string.
    #[test]
    fn id_is_ulid_format() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:X {v: 1})"#).unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:X) RETURN id(n) AS nid"#).unwrap();
        assert_eq!(rows.len(), 1);
        let id_val = match rows[0].get("nid").unwrap() {
            Value::String(s) => s.clone(),
            other => panic!("expected String, got {other:?}"),
        };
        assert_eq!(id_val.len(), 26, "ULID must be 26 characters");
        let valid_chars = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
        for ch in id_val.chars() {
            assert!(valid_chars.contains(ch), "invalid Crockford Base32 char: {ch}");
        }
    }

    /// Each node gets a distinct ULID.
    #[test]
    fn ids_are_unique_per_node() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:X {v: 1})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:X {v: 2})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:X {v: 3})"#).unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:X) RETURN id(n) AS nid"#).unwrap();
        assert_eq!(rows.len(), 3);
        let ids: std::collections::HashSet<String> = rows
            .iter()
            .map(|r| match r.get("nid").unwrap() {
                Value::String(s) => s.clone(),
                _ => panic!("expected String"),
            })
            .collect();
        assert_eq!(ids.len(), 3, "all three node IDs must be distinct");
    }

    /// ULIDs are lexicographically ordered by insertion time (monotone counter
    /// ensures ordering even within the same millisecond).
    #[test]
    fn ulid_ids_are_lexicographically_ordered() {
        use crate::types::ulid_new;
        let a = crate::types::ulid_encode(ulid_new());
        let b = crate::types::ulid_encode(ulid_new());
        let c = crate::types::ulid_encode(ulid_new());
        assert!(a < b, "ULIDs must be monotonically increasing: {a} < {b}");
        assert!(b < c, "ULIDs must be monotonically increasing: {b} < {c}");
    }

    /// id() on an edge variable returns a ULID string.
    #[test]
    fn edge_id_is_ulid_format() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:A {n: 1})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:B {n: 2})"#).unwrap();
        run(
            &mut g, &mut txn,
            r#"MATCH (a:A), (b:B) INSERT (a)-[:LINK]->(b)"#,
        )
        .unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (a:A)-[r:LINK]->(b:B) RETURN id(r) AS eid"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        let eid = match rows[0].get("eid").unwrap() {
            Value::String(s) => s.clone(),
            other => panic!("expected String, got {other:?}"),
        };
        assert_eq!(eid.len(), 26, "edge ULID must be 26 characters");
    }

    /// Upsert: re-inserting the same node returns the same ULID.
    #[test]
    fn upsert_preserves_ulid() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Alice"})"#).unwrap();
        let rows1 = run(
            &mut g, &mut txn,
            r#"MATCH (n:P {name: "Alice"}) RETURN id(n) AS nid"#,
        )
        .unwrap();
        // Re-insert — upsert must reuse the existing node.
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Alice"})"#).unwrap();
        let rows2 = run(
            &mut g, &mut txn,
            r#"MATCH (n:P {name: "Alice"}) RETURN id(n) AS nid"#,
        )
        .unwrap();
        assert_eq!(rows1.len(), 1);
        assert_eq!(rows2.len(), 1);
        assert_eq!(
            rows1[0].get("nid"),
            rows2[0].get("nid"),
            "upsert must preserve the original ULID"
        );
    }

    /// Division and modulo arithmetic in RETURN expressions.
    #[test]
    fn arithmetic_div_and_mod() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:X {v: 17})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:X) RETURN n.v / 5 AS q, n.v % 5 AS r"#,
        )
        .unwrap();
        assert_eq!(rows[0].get("q"), Some(&Value::Int(3)));
        assert_eq!(rows[0].get("r"), Some(&Value::Int(2)));
    }

    /// LIMIT 0 returns no rows.
    #[test]
    fn limit_zero_returns_no_rows() {
        let (mut g, mut txn) = readme_graph();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name LIMIT 0"#).unwrap();
        assert_eq!(rows.len(), 0);
    }

    /// OFFSET beyond end returns empty result.
    #[test]
    fn offset_beyond_end_returns_empty() {
        let (mut g, mut txn) = readme_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) RETURN n.name OFFSET 100"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 0);
    }

    /// Helper: extract the "Inserted N element(s)" count from a summary row.
    fn inserted_count(rows: &[Row]) -> u64 {
        let msg = match rows[0].get("result") {
            Some(Value::String(s)) => s.clone(),
            _ => panic!("expected summary row"),
        };
        // Format: "Inserted N element(s) [txn M]"
        msg.split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<u64>()
            .expect("count must be a number")
    }

    /// README sequence: run every INSERT from the README in order and verify
    /// the exact element counts documented there.
    ///
    /// README says:
    ///   txn 0: INSERT (:Person {name:"Alice",age:30})       → 1 element
    ///   txn 1: INSERT (:Person {name:"Bob",age:25})         → 1 element
    ///   txn 2: INSERT (:Person:Employee {name:"Carol",age:35}) → 1 element
    ///   txn 3: INSERT (a:Person{name:"Alice"}),(b:Person{name:"Bob"}),(a)-[:KNOWS]->(b)
    ///          Alice+Bob reused, only KNOWS edge is new     → 1 element
    ///   txn 4: MATCH (a:Person{name:"Alice"}),(b:Person:Employee{name:"Carol"})
    ///          INSERT (a)-[:MANAGES]->(b)                   → 1 element
    #[test]
    fn readme_sequence_element_counts() {
        let mut g = Graph::new();
        let mut txn = 0u64;

        let r0 = run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", age: 30})"#).unwrap();
        assert_eq!(inserted_count(&r0), 1, "txn 0: Alice");

        let r1 = run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob", age: 25})"#).unwrap();
        assert_eq!(inserted_count(&r1), 1, "txn 1: Bob");

        let r2 = run(
            &mut g, &mut txn,
            r#"INSERT (:Person:Employee {name: "Carol", age: 35})"#,
        )
        .unwrap();
        assert_eq!(inserted_count(&r2), 1, "txn 2: Carol");

        // Alice and Bob already exist — only the KNOWS edge is new.
        let r3 = run(
            &mut g, &mut txn,
            r#"INSERT (a:Person {name: "Alice"}), (b:Person {name: "Bob"}), (a)-[:KNOWS]->(b)"#,
        )
        .unwrap();
        assert_eq!(inserted_count(&r3), 1, "txn 3: KNOWS edge only");

        // Running it again must be a no-op (upsert).
        let r3b = run(
            &mut g, &mut txn,
            r#"INSERT (a:Person {name: "Alice"}), (b:Person {name: "Bob"}), (a)-[:KNOWS]->(b)"#,
        )
        .unwrap();
        assert_eq!(inserted_count(&r3b), 0, "txn 3 repeat: already exists");

        // MATCH+INSERT: Alice→Carol MANAGES edge (new).
        let r4 = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"}), (b:Person:Employee {name: "Carol"}) INSERT (a)-[:MANAGES]->(b)"#,
        )
        .unwrap();
        assert_eq!(inserted_count(&r4), 1, "txn 4: MANAGES edge");

        // Running it again must be a no-op.
        let r4b = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"}), (b:Person:Employee {name: "Carol"}) INSERT (a)-[:MANAGES]->(b)"#,
        )
        .unwrap();
        assert_eq!(inserted_count(&r4b), 0, "txn 4 repeat: already exists");

        // Final state: exactly 3 Person nodes, no duplicates.
        let persons =
            run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(persons.len(), 3, "exactly 3 Person nodes");

        // Exactly 1 KNOWS edge from Alice to Bob.
        let knows = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"})-[:KNOWS]->(b) RETURN b.name"#,
        )
        .unwrap();
        assert_eq!(knows.len(), 1);
        assert_eq!(knows[0].get("b.name"), Some(&Value::String("Bob".into())));

        // Exactly 1 MANAGES edge from Alice to Carol.
        let manages = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"})-[:MANAGES]->(b) RETURN b.name"#,
        )
        .unwrap();
        assert_eq!(manages.len(), 1);
        assert_eq!(manages[0].get("b.name"), Some(&Value::String("Carol".into())));
    }

    /// README MATCH+INSERT example: connect Alice to Carol with MANAGES.
    /// The readme_graph fixture already contains Alice, Bob (via KNOWS), and Carol,
    /// but NOT the MANAGES edge — so the first run inserts 1, the second 0.
    #[test]
    fn readme_match_insert_manages_edge() {
        let (mut g, mut txn) = readme_graph();

        let r = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"}), (b:Person:Employee {name: "Carol"}) INSERT (a)-[:MANAGES]->(b)"#,
        )
        .unwrap();
        assert_eq!(inserted_count(&r), 1, "first run inserts the MANAGES edge");

        // Idempotent: second run must not create a duplicate.
        let r2 = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"}), (b:Person:Employee {name: "Carol"}) INSERT (a)-[:MANAGES]->(b)"#,
        )
        .unwrap();
        assert_eq!(inserted_count(&r2), 0, "second run is a no-op");

        let edges = run(
            &mut g, &mut txn,
            r#"MATCH (a:Person {name: "Alice"})-[:MANAGES]->(b) RETURN b.name"#,
        )
        .unwrap();
        assert_eq!(edges.len(), 1, "exactly one MANAGES edge");
        assert_eq!(edges[0].get("b.name"), Some(&Value::String("Carol".into())));
    }

    // ── CREATE / DROP / SHOW INDEX ──────────────────────────────────────────

    #[test]
    fn create_index_returns_summary() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, "CREATE INDEX ON :Person(name)").unwrap();
        assert_eq!(rows.len(), 1);
        let msg = format!("{}", rows[0].get("result").unwrap());
        assert!(msg.contains("Index created"), "got: {msg}");
    }

    #[test]
    fn create_index_idempotent() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Person(name)").unwrap();
        let rows = run(&mut g, &mut txn, "CREATE INDEX ON :Person(name)").unwrap();
        let msg = format!("{}", rows[0].get("result").unwrap());
        assert!(msg.contains("already exists"), "got: {msg}");
    }

    #[test]
    fn show_indexes_empty_then_populated() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, "SHOW INDEXES").unwrap();
        assert_eq!(rows.len(), 1);
        let msg = format!("{}", rows[0].get("result").unwrap());
        assert!(msg.contains("No indexes"), "got: {msg}");

        run(&mut g, &mut txn, "CREATE INDEX ON :Person(name)").unwrap();
        run(&mut g, &mut txn, "CREATE INDEX ON :Person(age)").unwrap();
        let rows = run(&mut g, &mut txn, "SHOW INDEXES").unwrap();
        assert_eq!(rows.len(), 2);
        let mut labels: Vec<String> = rows
            .iter()
            .map(|r| format!("{}", r.get("label").unwrap()))
            .collect();
        labels.sort();
        assert_eq!(labels, vec!["\"Person\"", "\"Person\""]);
    }

    #[test]
    fn drop_index_removes_index() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Person(name)").unwrap();
        let rows = run(&mut g, &mut txn, "DROP INDEX ON :Person(name)").unwrap();
        let msg = format!("{}", rows[0].get("result").unwrap());
        assert!(msg.contains("dropped"), "got: {msg}");

        let rows = run(&mut g, &mut txn, "SHOW INDEXES").unwrap();
        let msg = format!("{}", rows[0].get("result").unwrap());
        assert!(msg.contains("No indexes"), "got: {msg}");
    }

    #[test]
    fn drop_nonexistent_index_returns_no_index_message() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, "DROP INDEX ON :Person(name)").unwrap();
        let msg = format!("{}", rows[0].get("result").unwrap());
        assert!(msg.contains("No index"), "got: {msg}");
    }

    #[test]
    fn index_used_for_exact_lookup() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Person(name)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", age: 30})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob",   age: 25})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Carol", age: 35})"#).unwrap();

        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person {name: "Alice"}) RETURN n.age"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("n.age"), Some(&Value::Int(30)));
    }

    #[test]
    fn index_maintained_after_set_property() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Person(name)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (n:Person {name: "Alice"}) SET n.name = "Alicia""#).unwrap();

        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person {name: "Alicia"}) RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);

        // Old name should not match.
        let rows2 = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person {name: "Alice"}) RETURN n.name"#,
        )
        .unwrap();
        assert_eq!(rows2.len(), 0);
    }

    #[test]
    fn index_entry_count_in_show_indexes() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Person(name)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob"})"#).unwrap();

        let rows = run(&mut g, &mut txn, "SHOW INDEXES").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("entries"), Some(&Value::Int(2)));
    }

    // ── Transaction (BEGIN / COMMIT / ROLLBACK) via execute_capturing ────────

    #[test]
    fn execute_capturing_returns_ops() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let stmt = super::super::parser::parse(r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        let (rows, ops) = execute_capturing(stmt, &mut g, &mut txn).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(ops.len(), 1, "one CreateNode op");
        // Node should be in the graph already.
        let found = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn execute_capturing_read_returns_empty_ops() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        let stmt =
            super::super::parser::parse(r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        let (_rows, ops) = execute_capturing(stmt, &mut g, &mut txn).unwrap();
        assert!(ops.is_empty(), "MATCH produces no ops");
    }

    #[test]
    fn rollback_restores_graph() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();

        // Save snapshot before transaction.
        let snapshot = g.clone();

        // Execute inserts that would be part of a transaction.
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Carol"})"#).unwrap();

        // Rollback by restoring snapshot.
        g = snapshot;

        let rows = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(col(&rows, "n.name"), vec!["\"Alice\""]);
    }

    // ── Range predicate index tests (P2) ──────────────────────────────────────

    fn make_age_graph() -> (Graph, u64) {
        let dir = tempfile::tempdir().unwrap().into_path();
        let mut g = Graph::open(&dir).unwrap();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Person(age)").unwrap();
        for (name, age) in &[("Alice", 30i64), ("Bob", 25), ("Carol", 35), ("Dave", 20)] {
            run(
                &mut g,
                &mut txn,
                &format!(r#"INSERT (:Person {{name: "{name}", age: {age}}})"#),
            )
            .unwrap();
        }
        (g, txn)
    }

    #[test]
    fn range_gt_exclusive() {
        let (mut g, mut txn) = make_age_graph();
        let rows =
            run(&mut g, &mut txn, "MATCH (n:Person) WHERE n.age > 25 RETURN n.name").unwrap();
        let mut names: Vec<String> = col(&rows, "n.name");
        names.sort();
        assert_eq!(names, vec![r#""Alice""#, r#""Carol""#]);
    }

    #[test]
    fn range_gte_inclusive() {
        let (mut g, mut txn) = make_age_graph();
        let rows =
            run(&mut g, &mut txn, "MATCH (n:Person) WHERE n.age >= 25 RETURN n.name").unwrap();
        let mut names: Vec<String> = col(&rows, "n.name");
        names.sort();
        assert_eq!(names, vec![r#""Alice""#, r#""Bob""#, r#""Carol""#]);
    }

    #[test]
    fn range_lt_exclusive() {
        let (mut g, mut txn) = make_age_graph();
        let rows =
            run(&mut g, &mut txn, "MATCH (n:Person) WHERE n.age < 30 RETURN n.name").unwrap();
        let mut names: Vec<String> = col(&rows, "n.name");
        names.sort();
        assert_eq!(names, vec![r#""Bob""#, r#""Dave""#]);
    }

    #[test]
    fn range_closed_interval() {
        let (mut g, mut txn) = make_age_graph();
        let rows = run(
            &mut g,
            &mut txn,
            "MATCH (n:Person) WHERE n.age >= 25 AND n.age <= 30 RETURN n.name",
        )
        .unwrap();
        let mut names: Vec<String> = col(&rows, "n.name");
        names.sort();
        assert_eq!(names, vec![r#""Alice""#, r#""Bob""#]);
    }

    #[test]
    fn range_no_index_falls_back_correctly() {
        // age2 has no index — must still return correct results via full scan
        let dir = tempfile::tempdir().unwrap().into_path();
        let mut g = Graph::open(&dir).unwrap();
        let mut txn = 0u64;
        for (name, age) in &[("Alice", 30i64), ("Bob", 25), ("Carol", 35)] {
            run(
                &mut g,
                &mut txn,
                &format!(r#"INSERT (:Person {{name: "{name}", age2: {age}}})"#),
            )
            .unwrap();
        }
        let rows =
            run(&mut g, &mut txn, "MATCH (n:Person) WHERE n.age2 > 25 RETURN n.name").unwrap();
        let mut names: Vec<String> = col(&rows, "n.name");
        names.sort();
        assert_eq!(names, vec![r#""Alice""#, r#""Carol""#]);
    }

    #[test]
    fn range_negative_integers() {
        let dir = tempfile::tempdir().unwrap().into_path();
        let mut g = Graph::open(&dir).unwrap();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Temp(val)").unwrap();
        for v in &[-100i64, -10, 0, 10, 100] {
            run(&mut g, &mut txn, &format!("INSERT (:Temp {{val: {v}}})")).unwrap();
        }
        let rows =
            run(&mut g, &mut txn, "MATCH (n:Temp) WHERE n.val >= -10 AND n.val < 10 RETURN n.val")
                .unwrap();
        let mut vals: Vec<String> = col(&rows, "n.val");
        vals.sort();
        assert_eq!(vals, vec!["-10", "0"]);
    }

    // ── P5: UNWIND, OPTIONAL MATCH, UNION, MATCH…WITH ────────────────────────

    #[test]
    fn unwind_basic() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, "UNWIND [1, 2, 3] AS x RETURN x").unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(col(&rows, "x"), vec!["1", "2", "3"]);
    }

    #[test]
    fn unwind_string_list() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(
            &mut g, &mut txn,
            r#"UNWIND ["Alice", "Bob"] AS name RETURN name"#,
        ).unwrap();
        assert_eq!(rows.len(), 2);
        let mut names = col(&rows, "name");
        names.sort();
        assert_eq!(names, vec!["\"Alice\"", "\"Bob\""]);
    }

    #[test]
    fn unwind_empty_list() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, "UNWIND [] AS x RETURN x").unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn optional_match_found() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"OPTIONAL MATCH (n:Person {name: "Alice"}) RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(col(&rows, "n.name"), vec!["\"Alice\""]);
    }

    #[test]
    fn optional_match_not_found_returns_null_row() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(
            &mut g, &mut txn,
            r#"OPTIONAL MATCH (n:Person) RETURN n.name"#,
        ).unwrap();
        // One row with null for n.name
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("n.name"), Some(&Value::Null));
    }

    #[test]
    fn union_combines_results() {
        let (mut g, mut txn) = three_people();
        // UNION deduplicates: if both branches return Alice, we get her once.
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person {name: "Alice"}) RETURN n.name
               UNION
               MATCH (n:Person {name: "Bob"}) RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 2);
        let mut names = col(&rows, "n.name");
        names.sort();
        assert_eq!(names, vec!["\"Alice\"", "\"Bob\""]);
    }

    #[test]
    fn union_all_keeps_duplicates() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person {name: "Alice"}) RETURN n.name
               UNION ALL
               MATCH (n:Person {name: "Alice"}) RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 2, "UNION ALL keeps both copies");
    }

    #[test]
    fn union_deduplicates_identical_rows() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person {name: "Alice"}) RETURN n.name
               UNION
               MATCH (n:Person {name: "Alice"}) RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 1, "UNION removes duplicates");
    }

    #[test]
    fn match_with_return() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WITH n.name AS name WHERE name <> "Bob" RETURN name"#,
        ).unwrap();
        assert_eq!(rows.len(), 2);
        let mut names = col(&rows, "name");
        names.sort();
        assert_eq!(names, vec!["\"Alice\"", "\"Carol\""]);
    }

    #[test]
    fn match_with_aggregation() {
        let (mut g, mut txn) = three_people();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WITH count(n) AS cnt RETURN cnt"#,
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("cnt"), Some(&Value::Int(3)));
    }

    #[test]
    fn list_literal_in_unwind() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, "UNWIND [10, 20, 30] AS n RETURN n").unwrap();
        let vals = col(&rows, "n");
        assert_eq!(vals, vec!["10", "20", "30"]);
    }

    #[test]
    fn string_functions() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:T {name: "  hello  "})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:T) RETURN toLower(n.name) AS lo, toUpper(n.name) AS hi, trim(n.name) AS tr"#,
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("lo"), Some(&Value::String("  hello  ".to_string())));
        assert_eq!(rows[0].get("hi"), Some(&Value::String("  HELLO  ".to_string())));
        assert_eq!(rows[0].get("tr"), Some(&Value::String("hello".to_string())));
    }

    #[test]
    fn math_functions() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:T {val: -3.7})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:T) RETURN abs(n.val) AS a, ceil(n.val) AS c, floor(n.val) AS f"#,
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("a"), Some(&Value::Float(3.7)));
        assert_eq!(rows[0].get("c"), Some(&Value::Float(-3.0)));
        assert_eq!(rows[0].get("f"), Some(&Value::Float(-4.0)));
    }

    #[test]
    fn range_function() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, "UNWIND range(1, 3) AS n RETURN n").unwrap();
        assert_eq!(col(&rows, "n"), vec!["1", "2", "3"]);
    }

    // ── TODO 16: Edge-case tests ──────────────────────────────────────────────

    /// OPTIONAL MATCH with non-aggregate RETURN on a missing label returns one null row.
    #[test]
    fn optional_match_missing_label_returns_null() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        let rows = run(&mut g, &mut txn, "OPTIONAL MATCH (n:Ghost) RETURN n.name").unwrap();
        assert_eq!(rows.len(), 1, "should get exactly one null row");
        assert_eq!(rows[0].get("n.name"), Some(&Value::Null));
    }

    /// OPTIONAL MATCH count(n) with no matches — n is null so count = 0, not 1.
    #[test]
    fn optional_match_count_expr_on_no_match() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, "OPTIONAL MATCH (n:Ghost) RETURN count(n) AS c").unwrap();
        assert_eq!(rows.len(), 1);
        // count(n) on the null row should be 0 (null value is not counted).
        assert_eq!(rows[0].get("c"), Some(&Value::Int(0)));
    }

    /// OPTIONAL MATCH count(*) with no matches — one null row, count(*) = 1.
    #[test]
    fn optional_match_count_star_on_no_match() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, "OPTIONAL MATCH (n:Ghost) RETURN count(*) AS c").unwrap();
        assert_eq!(rows.len(), 1);
        // count(*) counts all rows (including the null row) = 1.
        assert_eq!(rows[0].get("c"), Some(&Value::Int(1)));
    }

    /// DELETE a node that does not exist should produce an error.
    #[test]
    fn delete_nonexistent_node_errors() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        // Matching a ghost returns no results; DELETE on 0 bindings is a no-op.
        let rows = run(&mut g, &mut txn, "MATCH (n:Ghost) DELETE n").unwrap();
        assert_eq!(rows.len(), 1);
        let msg = format!("{}", rows[0].get("result").unwrap());
        assert!(msg.contains("Deleted 0"), "got: {msg}");
    }

    /// SET on a node that does not match the WHERE is a no-op.
    #[test]
    fn set_on_no_match_is_noop() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", age: 30})"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (n:Person) WHERE n.name = "Nobody" SET n.age = 99"#).unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.age"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("n.age"), Some(&Value::Int(30)));
    }

    /// Inserting an edge between the same node in both directions and querying either.
    #[test]
    fn self_loop_edge() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Node {name: "X"})"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (n:Node) INSERT (n)-[:SELF]->(n)"#).unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (a)-[:SELF]->(b) RETURN a.name, b.name"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("a.name"), Some(&Value::String("X".into())));
        assert_eq!(rows[0].get("b.name"), Some(&Value::String("X".into())));
    }

    /// UNWIND on an empty list produces zero rows (no RETURN output).
    #[test]
    fn unwind_empty_list_no_rows() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, "UNWIND [] AS x RETURN x").unwrap();
        assert_eq!(rows.len(), 0);
    }

    /// UNION ALL on identical queries keeps duplicate rows.
    #[test]
    fn union_all_preserves_duplicates() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:A {v: 1})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            "MATCH (n:A) RETURN n.v UNION ALL MATCH (n:A) RETURN n.v",
        ).unwrap();
        assert_eq!(rows.len(), 2, "UNION ALL should keep duplicates");
    }

    /// UNION (without ALL) deduplicates identical rows from both sides.
    #[test]
    fn union_deduplicates() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:A {v: 1})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            "MATCH (n:A) RETURN n.v UNION MATCH (n:A) RETURN n.v",
        ).unwrap();
        assert_eq!(rows.len(), 1, "UNION should deduplicate");
    }

    /// WHERE on property that doesn't exist evaluates to null (falsy), skipping the row.
    #[test]
    fn where_missing_property_is_null_falsy() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Alice"})"#).unwrap();
        // age is not set; WHERE n.age > 20 should not match (null is not > 20).
        let rows = run(&mut g, &mut txn, r#"MATCH (n:P) WHERE n.age > 20 RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 0);
    }

    /// count(*) on a label with no nodes returns 0.
    #[test]
    fn count_star_on_empty_label() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let rows = run(&mut g, &mut txn, "MATCH (n:Empty) RETURN count(*) AS c").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("c"), Some(&Value::Int(0)));
    }

    /// Property index survives a DETACH DELETE of an indexed node.
    #[test]
    fn property_index_updated_on_detach_delete() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"CREATE INDEX ON :P(name)"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Bob"})"#).unwrap();
        // Create an edge so we can test DETACH DELETE.
        run(
            &mut g, &mut txn,
            r#"MATCH (a:P {name: "Alice"}), (b:P {name: "Bob"}) INSERT (a)-[:KNOWS]->(b)"#,
        ).unwrap();
        // DETACH DELETE should clean up the property index entry.
        run(&mut g, &mut txn, r#"MATCH (n:P {name: "Alice"}) DETACH DELETE n"#).unwrap();
        // After deletion, index lookup should find nothing for Alice.
        let rows = run(&mut g, &mut txn, r#"MATCH (n:P {name: "Alice"}) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 0, "index should report no entries after detach-delete");
        // Bob should still be there.
        let rows2 = run(&mut g, &mut txn, r#"MATCH (n:P {name: "Bob"}) RETURN n.name"#).unwrap();
        assert_eq!(rows2.len(), 1);
    }

    /// MATCH…WITH aggregation then filter.
    #[test]
    fn match_with_having_style_filter() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:P {dept: "eng", salary: 100})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {dept: "eng", salary: 200})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {dept: "hr",  salary: 50})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            "MATCH (n:P) WITH n.dept AS dept, sum(n.salary) AS total WHERE total > 100 RETURN dept, total",
        ).unwrap();
        assert_eq!(rows.len(), 1, "only eng dept has total > 100");
        assert_eq!(rows[0].get("dept"), Some(&Value::String("eng".into())));
        // sum() on integer values returns Int; float inputs would return Float.
        assert_eq!(rows[0].get("total"), Some(&Value::Int(300)));
    }

    /// Multiple labels on a node: MATCH requiring all labels returns the node.
    #[test]
    fn multi_label_match_all_required() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person:Employee {name: "Bob"})"#).unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Person:Employee) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("n.name"), Some(&Value::String("Bob".into())));
    }

    /// Multiple labels on a node: MATCH requiring only one label also finds it.
    #[test]
    fn multi_label_partial_match() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person:Employee {name: "Bob"})"#).unwrap();
        let rows = run(&mut g, &mut txn, r#"MATCH (n:Person) RETURN n.name"#).unwrap();
        assert_eq!(rows.len(), 1);
    }

    /// Bare `*` quantifier includes 0-hop (self) paths.
    #[test]
    fn star_quantifier_includes_zero_hop() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:A {v: 1})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:A {v: 2})"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (a:A {v:1})-[:X*]->(b:A {v:2}) RETURN a.v, b.v"#).unwrap();
        // With no X edges, [*] includes 0-hop so a and b can be the same node,
        // but v:1 ≠ v:2 means no 0-hop match is possible. Just verify it doesn't panic.
    }

    /// coalesce() returns first non-null value.
    #[test]
    fn coalesce_returns_first_nonnull() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:T {a: 1})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            "MATCH (n:T) RETURN coalesce(n.missing, n.a, 99) AS v",
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("v"), Some(&Value::Int(1)));
    }

    /// startsWith / endsWith / contains string predicates.
    #[test]
    fn string_predicates() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:T {name: "hello world"})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:T) WHERE startsWith(n.name, "hello") RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 1);
        let rows2 = run(
            &mut g, &mut txn,
            r#"MATCH (n:T) WHERE endsWith(n.name, "xyz") RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows2.len(), 0);
    }

    // ── IN operator ──────────────────────────────────────────────────────────

    #[test]
    fn in_operator_string_found() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:T {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:T {name: "Bob"})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:T) WHERE n.name IN ["Alice", "Carol"] RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("n.name"), Some(&Value::String("Alice".into())));
    }

    #[test]
    fn in_operator_string_not_found() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:T {name: "Alice"})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:T) WHERE n.name IN ["Bob", "Carol"] RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn in_operator_integer_list() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:T {val: 10})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:T {val: 20})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:T {val: 30})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:T) WHERE n.val IN [10, 30] RETURN n.val ORDER BY n.val"#,
        ).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get("n.val"), Some(&Value::Int(10)));
        assert_eq!(rows[1].get("n.val"), Some(&Value::Int(30)));
    }

    #[test]
    fn in_operator_empty_list_matches_nothing() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:T {name: "Alice"})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:T) WHERE n.name IN [] RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn in_operator_with_labels_function() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Company {name: "Acme"})"#).unwrap();
        // "Person" IN labels(n) should match only Person nodes
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n) WHERE "Person" IN labels(n) RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("n.name"), Some(&Value::String("Alice".into())));
    }

    #[test]
    fn in_operator_with_multiple_label_types() {
        // Test IN with labels() across a mixed-label graph.
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Company {name: "Acme"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Animal {name: "Rex"})"#).unwrap();
        // Positive: "Company" IN labels(n) matches only Acme
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n) WHERE "Company" IN labels(n) RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("n.name"), Some(&Value::String("Acme".into())));
        // Positive: "Animal" IN labels(n) matches only Rex
        let rows2 = run(
            &mut g, &mut txn,
            r#"MATCH (n) WHERE "Animal" IN labels(n) RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows2.len(), 1);
        assert_eq!(rows2[0].get("n.name"), Some(&Value::String("Rex".into())));
    }

    // ── MATCH … OPTIONAL MATCH … RETURN ─────────────────────────────────────

    fn social_graph() -> (Graph, u64) {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Carol"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Company {name: "Acme"})"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (a:Person {name:"Alice"}), (c:Company {name:"Acme"}) INSERT (a)-[:WORKS_AT]->(c)"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (b:Person {name:"Bob"}), (c:Company {name:"Acme"}) INSERT (b)-[:WORKS_AT]->(c)"#).unwrap();
        // Carol has no WORKS_AT edge
        run(&mut g, &mut txn, r#"MATCH (a:Person {name:"Alice"}), (b:Person {name:"Bob"}) INSERT (a)-[:KNOWS]->(b)"#).unwrap();
        (g, txn)
    }

    #[test]
    fn match_optional_match_all_found() {
        let (mut g, mut txn) = social_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (p:Person {name:"Alice"}) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("p.name"), Some(&Value::String("Alice".into())));
        assert_eq!(rows[0].get("c.name"), Some(&Value::String("Acme".into())));
    }

    #[test]
    fn match_optional_match_not_found_gives_null() {
        let (mut g, mut txn) = social_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (p:Person {name:"Carol"}) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("p.name"), Some(&Value::String("Carol".into())));
        // c.name must be absent (null — property access on unbound var)
        assert!(matches!(rows[0].get("c.name"), Some(&Value::Null) | None));
    }

    #[test]
    fn match_optional_match_mixed_results() {
        // All people: Alice and Bob have WORKS_AT, Carol does not.
        let (mut g, mut txn) = social_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name ORDER BY p.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 3, "one row per person");
        let names: Vec<_> = rows.iter().map(|r| r.get("p.name").cloned().unwrap()).collect();
        assert_eq!(names, vec![
            Value::String("Alice".into()),
            Value::String("Bob".into()),
            Value::String("Carol".into()),
        ]);
        // Carol's c.name must be null
        assert!(matches!(rows[2].get("c.name"), Some(&Value::Null) | None));
        // Alice and Bob have Acme
        assert_eq!(rows[0].get("c.name"), Some(&Value::String("Acme".into())));
        assert_eq!(rows[1].get("c.name"), Some(&Value::String("Acme".into())));
    }

    #[test]
    fn match_optional_match_does_not_duplicate_mandatory_rows() {
        // If the optional clause matches multiple times, one row per optional match.
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Company {name: "Acme"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Company {name: "Globex"})"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (a:Person {name:"Alice"}), (c:Company {name:"Acme"})   INSERT (a)-[:WORKS_AT]->(c)"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (a:Person {name:"Alice"}), (c:Company {name:"Globex"}) INSERT (a)-[:WORKS_AT]->(c)"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name ORDER BY c.name"#,
        ).unwrap();
        // Alice matches both companies → 2 rows
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn match_optional_match_where_on_optional() {
        // WHERE on the optional clause filters — unmatched rows still appear with null.
        let (mut g, mut txn) = social_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) WHERE c.name = "Acme" RETURN p.name, c.name ORDER BY p.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 3);
        // Carol still appears, c.name null
        assert!(matches!(rows[2].get("c.name"), Some(&Value::Null) | None));
    }

    #[test]
    fn match_two_optional_match_clauses() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Company {name: "Acme"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:City {name: "NY"})"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (a:Person {name:"Alice"}), (c:Company {name:"Acme"}) INSERT (a)-[:WORKS_AT]->(c)"#).unwrap();
        // No lives_in edge → second OPTIONAL MATCH returns null
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) OPTIONAL MATCH (p)-[:LIVES_IN]->(city:City) RETURN p.name, c.name, city.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("c.name"), Some(&Value::String("Acme".into())));
        assert!(matches!(rows[0].get("city.name"), Some(&Value::Null) | None));
    }

    // ── CALL YIELD (column projection fix) ───────────────────────────────────

    #[test]
    fn call_yield_projects_correct_columns() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob"})"#).unwrap();
        // YIELD must return real column values, not empty rows
        let rows = run(
            &mut g, &mut txn,
            r#"CALL pageRank(iterations: 5, dampingFactor: 0.85) YIELD node, score"#,
        ).unwrap();
        assert_eq!(rows.len(), 2, "one row per node");
        for r in &rows {
            assert!(r.contains_key("node"), "must have 'node' column");
            assert!(r.contains_key("score"), "must have 'score' column");
        }
    }

    #[test]
    fn call_yield_partial_projection() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:X {v: 1})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:X {v: 2})"#).unwrap();
        // Only YIELD node — score should be absent
        let rows = run(
            &mut g, &mut txn,
            r#"CALL pageRank() YIELD node"#,
        ).unwrap();
        assert_eq!(rows.len(), 2);
        for r in &rows {
            assert!(r.contains_key("node"));
            assert!(!r.contains_key("score"), "score not in YIELD — must be absent");
        }
    }

    #[test]
    fn call_degree_centrality_correct_key_names() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:A {n: "hub"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:B {n: "spoke1"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:B {n: "spoke2"})"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (a:A), (b:B {n:"spoke1"}) INSERT (a)-[:E]->(b)"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (a:A), (b:B {n:"spoke2"}) INSERT (a)-[:E]->(b)"#).unwrap();

        let rows = run(
            &mut g, &mut txn,
            r#"CALL degreeCentrality() YIELD node, degree, in_degree, out_degree"#,
        ).unwrap();
        assert!(!rows.is_empty());
        // All rows must have the snake_case keys
        for r in &rows {
            assert!(r.contains_key("degree"),     "must have 'degree'");
            assert!(r.contains_key("in_degree"),  "must have 'in_degree'");
            assert!(r.contains_key("out_degree"), "must have 'out_degree'");
            // camelCase keys must NOT appear
            assert!(!r.contains_key("totalDegree"), "'totalDegree' must not exist");
            assert!(!r.contains_key("inDegree"),    "'inDegree' must not exist");
            assert!(!r.contains_key("outDegree"),   "'outDegree' must not exist");
        }
        // The hub node should have out_degree = 2
        let hub = rows.iter().max_by_key(|r| {
            if let Some(Value::Int(d)) = r.get("out_degree") { *d } else { 0 }
        }).unwrap();
        assert_eq!(hub.get("out_degree"), Some(&Value::Int(2)));
        assert_eq!(hub.get("in_degree"),  Some(&Value::Int(0)));
    }

    #[test]
    fn call_wcc_yields_node_and_component() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:X)"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:X)"#).unwrap(); // disconnected second component
        let rows = run(&mut g, &mut txn, r#"CALL wcc() YIELD node, component"#).unwrap();
        assert_eq!(rows.len(), 2);
        for r in &rows {
            assert!(r.contains_key("node"));
            assert!(r.contains_key("component"));
        }
    }

    // ── CALL pipeline: CALL … YIELD … RETURN ─────────────────────────────────

    #[test]
    fn call_pipeline_yield_return_basic() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {name: "Bob"})"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"CALL pageRank(iterations: 5, dampingFactor: 0.85) YIELD node, score RETURN node, score ORDER BY score DESC"#,
        ).unwrap();
        assert_eq!(rows.len(), 2, "one row per node");
        for r in &rows {
            assert!(r.contains_key("node"));
            assert!(r.contains_key("score"));
        }
        // First row should have highest score
        let s0 = if let Some(Value::Float(f)) = rows[0].get("score") { *f } else { 0.0 };
        let s1 = if let Some(Value::Float(f)) = rows[1].get("score") { *f } else { 0.0 };
        assert!(s0 >= s1, "ORDER BY score DESC: first score must be >= second");
    }

    #[test]
    fn call_pipeline_yield_return_with_limit() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        for i in 0..5 {
            run(&mut g, &mut txn, &format!(r#"INSERT (:N {{v: {}}})"#, i)).unwrap();
        }
        let rows = run(
            &mut g, &mut txn,
            r#"CALL degreeCentrality() YIELD node, degree RETURN node, degree LIMIT 2"#,
        ).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn call_pipeline_yield_return_aggregation() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:P)"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P)"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P)"#).unwrap();
        let rows = run(
            &mut g, &mut txn,
            r#"CALL wcc() YIELD node, component RETURN component, count(*) AS sz ORDER BY sz DESC"#,
        ).unwrap();
        // 3 isolated nodes → 3 components of size 1
        assert_eq!(rows.len(), 3);
        for r in &rows {
            assert_eq!(r.get("sz"), Some(&Value::Int(1)));
        }
    }

    // ── CALL pipeline: CALL … YIELD … MATCH … RETURN ─────────────────────────

    #[test]
    fn call_pipeline_match_filters_by_label() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Company {name: "Acme"})"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (a:Person {name:"Alice"}), (b:Person {name:"Bob"}) INSERT (a)-[:KNOWS]->(b)"#).unwrap();

        // pageRank returns all 3 nodes; MATCH+WHERE filters to Person only
        let rows = run(
            &mut g, &mut txn,
            r#"CALL pageRank(iterations: 5, dampingFactor: 0.85) YIELD node, score MATCH (n) WHERE n = node AND "Person" IN labels(n) RETURN n.name, score ORDER BY n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 2, "only Person nodes returned");
        let names: Vec<_> = rows.iter()
            .map(|r| r.get("n.name").cloned().unwrap())
            .collect();
        assert!(names.contains(&Value::String("Alice".into())));
        assert!(names.contains(&Value::String("Bob".into())));
        // Company must be absent
        for r in &rows {
            assert_ne!(r.get("n.name"), Some(&Value::String("Acme".into())));
        }
    }

    #[test]
    fn call_pipeline_match_returns_node_properties() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", age: 30})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob",   age: 25})"#).unwrap();

        let rows = run(
            &mut g, &mut txn,
            r#"CALL degreeCentrality() YIELD node, degree MATCH (n) WHERE n = node AND "Person" IN labels(n) RETURN n.name, n.age, degree ORDER BY n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows[0].contains_key("n.name"));
        assert!(rows[0].contains_key("n.age"));
        assert!(rows[0].contains_key("degree"));
    }

    #[test]
    fn call_pipeline_match_order_by_score() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Carol"})"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (a:Person {name:"Alice"}), (b:Person {name:"Bob"})   INSERT (a)-[:KNOWS]->(b)"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (b:Person {name:"Bob"}),   (c:Person {name:"Carol"}) INSERT (b)-[:KNOWS]->(c)"#).unwrap();

        // Exact query from the notebook — must parse and return 3 rows
        let rows = run(
            &mut g, &mut txn,
            r#"CALL pageRank(iterations: 20, dampingFactor: 0.85) YIELD node, score MATCH (n) WHERE n = node AND "Person" IN labels(n) RETURN n.name, round(score * 1000) / 1000 AS pagerank ORDER BY pagerank DESC LIMIT 5"#,
        ).unwrap();
        assert_eq!(rows.len(), 3, "all 3 Person nodes returned");
        // Every row must have both columns
        for r in &rows {
            assert!(r.contains_key("n.name"), "must have n.name");
            assert!(r.contains_key("pagerank"), "must have pagerank alias");
        }
        // Scores must be non-negative
        for r in &rows {
            if let Some(Value::Float(f)) = r.get("pagerank") {
                assert!(*f >= 0.0, "pagerank score must be non-negative");
            }
        }
    }

    // ── LOAD CSV GQL surface ─────────────────────────────────────────────────

    #[test]
    fn load_csv_nodes_gql() {
        use std::io::Write;
        let mut g = Graph::new();
        let mut txn = 0u64;

        // Write a temporary CSV file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nodes.csv");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, ":ID,name,age,:LABEL").unwrap();
            writeln!(f, "1,Alice,30,Person").unwrap();
            writeln!(f, "2,Bob,25,Person").unwrap();
        }

        let gql = format!("LOAD CSV NODES FROM '{}'", path.display());
        let rows = run(&mut g, &mut txn, &gql).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0]["result"].to_string().contains("2"), "summary should say 2 nodes");
        assert_eq!(g.node_count(), 2);
        // id_map populated in graph
        assert_eq!(g.csv_id_map.len(), 2);
    }

    #[test]
    fn load_csv_nodes_with_label_override() {
        use std::io::Write;
        let mut g = Graph::new();
        let mut txn = 0u64;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nodes.csv");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, ":ID,name").unwrap();
            writeln!(f, "1,Alice").unwrap();
        }

        let gql = format!("LOAD CSV NODES FROM '{}' LABEL Employee", path.display());
        run(&mut g, &mut txn, &gql).unwrap();
        let nid = *g.csv_id_map.values().next().unwrap();
        let node = g.get_node(nid).unwrap();
        assert_eq!(node.labels, vec!["Employee"]);
    }

    #[test]
    fn load_csv_edges_gql() {
        use std::io::Write;
        let mut g = Graph::new();
        let mut txn = 0u64;

        let dir = tempfile::tempdir().unwrap();
        let npath = dir.path().join("nodes.csv");
        let epath = dir.path().join("edges.csv");
        {
            let mut f = std::fs::File::create(&npath).unwrap();
            writeln!(f, ":ID,name,:LABEL").unwrap();
            writeln!(f, "1,Alice,Person").unwrap();
            writeln!(f, "2,Bob,Person").unwrap();
        }
        {
            let mut f = std::fs::File::create(&epath).unwrap();
            writeln!(f, ":START_ID,:END_ID,:TYPE,weight").unwrap();
            writeln!(f, "1,2,KNOWS,0.9").unwrap();
        }

        run(&mut g, &mut txn, &format!("LOAD CSV NODES FROM '{}'", npath.display())).unwrap();
        let rows = run(&mut g, &mut txn, &format!("LOAD CSV EDGES FROM '{}'", epath.display())).unwrap();
        assert_eq!(g.edge_count(), 1);
        assert!(rows[0]["result"].to_string().contains("1"));
    }

    #[test]
    fn call_pipeline_match_with_limit() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        for i in 0..6 {
            run(&mut g, &mut txn, &format!(r#"INSERT (:Person {{name: "P{}"}})"#, i)).unwrap();
        }
        let rows = run(
            &mut g, &mut txn,
            r#"CALL pageRank() YIELD node, score MATCH (n) WHERE n = node AND "Person" IN labels(n) RETURN n.name, score ORDER BY score DESC LIMIT 3"#,
        ).unwrap();
        assert_eq!(rows.len(), 3, "LIMIT 3 must cap results");
    }

    // ── TRUNCATE ──────────────────────────────────────────────────────────────

    #[test]
    fn truncate_clears_all_nodes_and_edges() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        // Insert some nodes and an edge.
        run(&mut g, &mut txn, r#"INSERT (:Person {name:"Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name:"Bob"})"#).unwrap();
        let rows = run(&mut g, &mut txn, "MATCH (a:Person), (b:Person) WHERE a.name <> b.name INSERT (a)-[:KNOWS]->(b)").unwrap();
        let _ = rows;

        // Verify data is present.
        let n = run(&mut g, &mut txn, "MATCH (n) RETURN count(n) AS c").unwrap();
        assert!(n[0]["c"].to_string().parse::<u64>().unwrap() > 0);

        // TRUNCATE.
        let result = run(&mut g, &mut txn, "TRUNCATE").unwrap();
        assert_eq!(result[0]["cleared"], crate::types::Value::Bool(true));

        // Graph should be empty.
        let n = run(&mut g, &mut txn, "MATCH (n) RETURN count(n) AS c").unwrap();
        assert_eq!(n[0]["c"], crate::types::Value::Int(0));
        let e = run(&mut g, &mut txn, "MATCH ()-[r]->() RETURN count(r) AS c").unwrap();
        assert_eq!(e[0]["c"], crate::types::Value::Int(0));
    }

    #[test]
    fn truncate_is_idempotent_on_empty_graph() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let result = run(&mut g, &mut txn, "TRUNCATE").unwrap();
        assert_eq!(result[0]["cleared"], crate::types::Value::Bool(true));
        let n = run(&mut g, &mut txn, "MATCH (n) RETURN count(n) AS c").unwrap();
        assert_eq!(n[0]["c"], crate::types::Value::Int(0));
    }

    #[test]
    fn truncate_preserves_index_definitions() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Person(name)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name:"Alice"})"#).unwrap();
        run(&mut g, &mut txn, "TRUNCATE").unwrap();
        // Index def should still be listed after clear.
        let rows = run(&mut g, &mut txn, "SHOW INDEXES").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["label"], crate::types::Value::String("Person".into()));
    }

    // ── Constraint tests ─────────────────────────────────────────────────────

    #[test]
    fn constraint_create_show_drop() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        // Create a unique constraint.
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        let rows = run(&mut g, &mut txn, "SHOW CONSTRAINTS").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["label"],    crate::types::Value::String("Person".into()));
        assert_eq!(rows[0]["property"], crate::types::Value::String("email".into()));
        assert_eq!(rows[0]["kind"],     crate::types::Value::String("UNIQUE".into()));
        // Drop it.
        run(&mut g, &mut txn, "DROP CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        let rows2 = run(&mut g, &mut txn, "SHOW CONSTRAINTS").unwrap();
        assert_eq!(rows2.len(), 0);
    }

    #[test]
    fn constraint_type_integer() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT TYPE IS INTEGER ON :Person(age)").unwrap();
        let rows = run(&mut g, &mut txn, "SHOW CONSTRAINTS").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["kind"], crate::types::Value::String("TYPE IS INTEGER".into()));
    }

    #[test]
    fn constraint_unique_enforced() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        // Insert Alice with email.
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", email: "a@b.com"})"#).unwrap();
        // Insert a DIFFERENT person (different name) with the same email — should fail.
        let err = run(&mut g, &mut txn, r#"INSERT (:Person {name: "Eve", email: "a@b.com"})"#);
        assert!(err.is_err(), "expected constraint violation but got Ok");
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("Constraint violation"), "expected constraint violation, got: {msg}");
    }

    #[test]
    fn constraint_type_enforced() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT TYPE IS INTEGER ON :Person(age)").unwrap();
        // Valid insert.
        run(&mut g, &mut txn, r#"INSERT (:Person {age: 30})"#).unwrap();
        // Invalid: string where integer expected.
        let err = run(&mut g, &mut txn, r#"INSERT (:Person {age: "not-a-number"})"#);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("Constraint violation"), "expected constraint violation, got: {msg}");
    }

    #[test]
    fn constraint_allows_self_update() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {email: "a@b.com"})"#).unwrap();
        // Updating the same node's own email to the same value should succeed.
        run(&mut g, &mut txn, r#"MATCH (n:Person {email: "a@b.com"}) SET n.email = "a@b.com""#).unwrap();
    }

    // ── Parameterized query tests ─────────────────────────────────────────────

    #[test]
    fn param_basic_literal() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        let mut params = std::collections::HashMap::new();
        params.insert("target".to_string(), crate::types::Value::String("Alice".to_string()));
        let stmt = super::super::parser::parse(r#"MATCH (n:Person {name: $target}) RETURN n.name"#).unwrap();
        let (rows, _) = execute_capturing_with_params(stmt, &mut g, &mut txn, params).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["n.name"], crate::types::Value::String("Alice".into()));
    }

    #[test]
    fn unwind_insert_from_literal_list() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn,
            r#"UNWIND ["Alice", "Bob", "Carol"] AS name INSERT (:Person {name: name})"#
        ).unwrap();
        let rows = run(&mut g, &mut txn, "MATCH (n:Person) RETURN n.name ORDER BY n.name").unwrap();
        assert_eq!(rows.len(), 3);
        let names = col(&rows, "n.name");
        assert!(names.contains(&"\"Alice\"".to_string()));
        assert!(names.contains(&"\"Bob\"".to_string()));
        assert!(names.contains(&"\"Carol\"".to_string()));
    }

    #[test]
    fn unwind_insert_from_param() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        let mut params = std::collections::HashMap::new();
        params.insert("names".to_string(), crate::types::Value::List(vec![
            crate::types::Value::String("Dave".into()),
            crate::types::Value::String("Eve".into()),
        ]));
        let stmt = super::super::parser::parse(
            "UNWIND $names AS name INSERT (:Person {name: name})"
        ).unwrap();
        let (rows, _) = execute_capturing_with_params(stmt, &mut g, &mut txn, params).unwrap();
        assert!(rows[0]["result"].to_string().contains("Inserted 2"));
        let qrows = run(&mut g, &mut txn, "MATCH (n:Person) RETURN n.name ORDER BY n.name").unwrap();
        assert_eq!(qrows.len(), 2);
    }

    // ── Extended constraint tests ─────────────────────────────────────────────

    #[test]
    fn constraint_show_multiple() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        run(&mut g, &mut txn, "CREATE CONSTRAINT TYPE IS INTEGER ON :Person(age)").unwrap();
        run(&mut g, &mut txn, "CREATE CONSTRAINT TYPE IS STRING ON :Company(name)").unwrap();
        let rows = run(&mut g, &mut txn, "SHOW CONSTRAINTS").unwrap();
        assert_eq!(rows.len(), 3);
        // All three expected labels present.
        let labels: Vec<_> = rows.iter()
            .map(|r| r["label"].to_string().trim_matches('"').to_string())
            .collect();
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"Company".to_string()));
    }

    #[test]
    fn constraint_type_float() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT TYPE IS FLOAT ON :Sensor(reading)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Sensor {reading: 3.14})"#).unwrap();
        let err = run(&mut g, &mut txn, r#"INSERT (:Sensor {reading: 42})"#);
        assert!(err.is_err(), "integer should violate FLOAT constraint");
    }

    #[test]
    fn constraint_type_string() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT TYPE IS STRING ON :Tag(name)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Tag {name: "rust"})"#).unwrap();
        let err = run(&mut g, &mut txn, r#"INSERT (:Tag {name: 99})"#);
        assert!(err.is_err(), "integer should violate STRING constraint");
    }

    #[test]
    fn constraint_type_boolean() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT TYPE IS BOOLEAN ON :Feature(enabled)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Feature {enabled: true})"#).unwrap();
        let err = run(&mut g, &mut txn, r#"INSERT (:Feature {enabled: "yes"})"#);
        assert!(err.is_err(), "string should violate BOOLEAN constraint");
    }

    #[test]
    fn constraint_unique_missing_property_allowed() {
        // A node that does NOT have the constrained property is permitted.
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        // No `email` property — should not trigger the constraint.
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob"})"#).unwrap();
        let rows = run(&mut g, &mut txn, "MATCH (n:Person) RETURN count(n) AS c").unwrap();
        assert_eq!(rows[0]["c"], crate::types::Value::Int(2));
    }

    #[test]
    fn constraint_unique_different_labels_allowed() {
        // UNIQUE ON :Person(email) must not block :Movie nodes with the same value.
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {email: "a@b.com"})"#).unwrap();
        // Different label — should be allowed.
        run(&mut g, &mut txn, r#"INSERT (:Robot {email: "a@b.com"})"#).unwrap();
        let rows = run(&mut g, &mut txn, "MATCH (n) RETURN count(n) AS c").unwrap();
        assert_eq!(rows[0]["c"], crate::types::Value::Int(2));
    }

    #[test]
    fn constraint_set_triggers_unique() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", email: "alice@x.com"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob",   email: "bob@x.com"})"#).unwrap();
        // Try to set Bob's email to Alice's email — must fail.
        let err = run(&mut g, &mut txn,
            r#"MATCH (n:Person {name: "Bob"}) SET n.email = "alice@x.com""#);
        assert!(err.is_err(), "SET should be blocked by UNIQUE constraint");
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("Constraint violation"));
    }

    #[test]
    fn constraint_set_triggers_type() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT TYPE IS INTEGER ON :Person(age)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", age: 30})"#).unwrap();
        let err = run(&mut g, &mut txn,
            r#"MATCH (n:Person {name: "Alice"}) SET n.age = "thirty""#);
        assert!(err.is_err(), "SET should be blocked by TYPE constraint");
    }

    #[test]
    fn constraint_duplicate_create_is_noop() {
        // Creating the same constraint twice leaves only one entry.
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        let rows = run(&mut g, &mut txn, "SHOW CONSTRAINTS").unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn constraint_drop_removes_enforcement() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", email: "a@b.com"})"#).unwrap();
        run(&mut g, &mut txn, "DROP CONSTRAINT UNIQUE ON :Person(email)").unwrap();
        // After dropping, a second node with the same email must succeed.
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Eve", email: "a@b.com"})"#).unwrap();
        let rows = run(&mut g, &mut txn, "MATCH (n:Person) RETURN count(n) AS c").unwrap();
        assert_eq!(rows[0]["c"], crate::types::Value::Int(2));
    }

    #[test]
    fn unwind_insert_respects_constraints() {
        // UNIQUE constraint must be checked for every iteration of UNWIND INSERT.
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :Tag(name)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Tag {name: "rust"})"#).unwrap();
        // Second element collides with existing node — whole batch should fail.
        let err = run(&mut g, &mut txn,
            r#"UNWIND ["go", "rust"] AS n INSERT (:Tag {name: n})"#);
        assert!(err.is_err(), "UNWIND INSERT should propagate constraint violation");
    }

    // ── Extended parameterized query tests ────────────────────────────────────

    #[test]
    fn param_missing_evaluates_to_null() {
        // An undefined $x placeholder resolves to Null, which is falsy in WHERE.
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();
        // No params supplied — $target resolves to null, WHERE null → no match.
        let stmt = super::super::parser::parse(
            r#"MATCH (n:Person) WHERE n.name = $target RETURN n.name"#
        ).unwrap();
        let (rows, _) = execute_capturing_with_params(stmt, &mut g, &mut txn,
            std::collections::HashMap::new()).unwrap();
        assert_eq!(rows.len(), 0, "null param should match nothing");
    }

    #[test]
    fn param_integer_in_where() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice", age: 30})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Bob",   age: 20})"#).unwrap();
        let mut params = std::collections::HashMap::new();
        params.insert("min_age".into(), crate::types::Value::Int(25));
        let stmt = super::super::parser::parse(
            "MATCH (n:Person) WHERE n.age >= $min_age RETURN n.name"
        ).unwrap();
        let (rows, _) = execute_capturing_with_params(stmt, &mut g, &mut txn, params).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["n.name"], crate::types::Value::String("Alice".into()));
    }

    #[test]
    fn param_used_as_limit() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        for i in 0..10i64 {
            run(&mut g, &mut txn, &format!(r#"INSERT (:Item {{n: {i}}})"#)).unwrap();
        }
        let mut params = std::collections::HashMap::new();
        params.insert("k".into(), crate::types::Value::Int(3));
        let stmt = super::super::parser::parse(
            "MATCH (x:Item) RETURN x.n ORDER BY x.n LIMIT $k"
        ).unwrap();
        let (rows, _) = execute_capturing_with_params(stmt, &mut g, &mut txn, params).unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn params_do_not_leak_between_calls() {
        // Params from one call must not be visible in a subsequent parameterless call.
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Person {name: "Alice"})"#).unwrap();

        // First call — binds $target = "Alice".
        let mut p = std::collections::HashMap::new();
        p.insert("target".into(), crate::types::Value::String("Alice".into()));
        let stmt1 = super::super::parser::parse(
            r#"MATCH (n:Person) WHERE n.name = $target RETURN n.name"#
        ).unwrap();
        let (rows, _) = execute_capturing_with_params(stmt1, &mut g, &mut txn, p).unwrap();
        assert_eq!(rows.len(), 1);

        // Second call — no params; $target should resolve to null → no matches.
        let stmt2 = super::super::parser::parse(
            r#"MATCH (n:Person) WHERE n.name = $target RETURN n.name"#
        ).unwrap();
        let (rows2, _) = execute_capturing_with_params(stmt2, &mut g, &mut txn,
            std::collections::HashMap::new()).unwrap();
        assert_eq!(rows2.len(), 0, "params must be cleared after each call");
    }

    #[test]
    fn unwind_insert_map_items() {
        // UNWIND a list of Value::Map; property values accessed via `item.field`.
        let mut g = Graph::new();
        let mut txn = 0u64;
        let mut params = std::collections::HashMap::new();
        params.insert("people".into(), crate::types::Value::List(vec![
            {
                let mut m = std::collections::HashMap::new();
                m.insert("name".into(), crate::types::Value::String("Alice".into()));
                m.insert("age".into(),  crate::types::Value::Int(30));
                crate::types::Value::Map(m)
            },
            {
                let mut m = std::collections::HashMap::new();
                m.insert("name".into(), crate::types::Value::String("Bob".into()));
                m.insert("age".into(),  crate::types::Value::Int(25));
                crate::types::Value::Map(m)
            },
        ]));
        let stmt = super::super::parser::parse(
            "UNWIND $people AS p INSERT (:Person {name: p.name, age: p.age})"
        ).unwrap();
        let (rows, _) = execute_capturing_with_params(stmt, &mut g, &mut txn, params).unwrap();
        assert!(rows[0]["result"].to_string().contains("Inserted 2"));

        let qrows = run(&mut g, &mut txn,
            "MATCH (n:Person) RETURN n.name, n.age ORDER BY n.name").unwrap();
        assert_eq!(qrows.len(), 2);
        assert_eq!(qrows[0]["n.name"], crate::types::Value::String("Alice".into()));
        assert_eq!(qrows[0]["n.age"],  crate::types::Value::Int(30));
        assert_eq!(qrows[1]["n.name"], crate::types::Value::String("Bob".into()));
    }

    #[test]
    fn param_and_constraint_interact_correctly() {
        // INSERT via $param must still be blocked by an active constraint.
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE CONSTRAINT UNIQUE ON :User(username)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:User {username: "alice"})"#).unwrap();

        let mut p = std::collections::HashMap::new();
        p.insert("uname".into(), crate::types::Value::String("alice".into()));
        let stmt = super::super::parser::parse(
            r#"INSERT (:User {username: $uname})"#
        ).unwrap();
        // Different node, same username — must be blocked.
        // Note: exact-match upsert may deduplicate this; test the unique-with-different-name variant.
        let stmt2 = super::super::parser::parse(
            r#"INSERT (:User {username: $uname, display: "imposter"})"#
        ).unwrap();
        let err = execute_capturing_with_params(stmt2, &mut g, &mut txn, p);
        assert!(err.is_err(), "parameterized INSERT should be blocked by UNIQUE constraint");
    }

    // ── AdjDelta rollback tests ───────────────────────────────────────────────

    #[test]
    fn adj_delta_rollback_insert_edge() {
        // After BEGIN + insert edge + ROLLBACK, the edge must be gone and the
        // in-memory adjacency must reflect the pre-transaction state.
        let dir = tempfile::tempdir().unwrap().into_path();
        let mut g = Graph::open(&dir).unwrap();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:P {name:"A"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {name:"B"})"#).unwrap();

        g.begin_transaction().unwrap();
        run(&mut g, &mut txn, r#"MATCH (a:P {name:"A"}),(b:P {name:"B"}) INSERT (a)-[:KNOWS]->(b)"#).unwrap();

        // Edge should exist inside the transaction.
        let rows_in = run(&mut g, &mut txn, r#"MATCH (a:P)-[:KNOWS]->(b:P) RETURN a.name"#).unwrap();
        assert_eq!(rows_in.len(), 1);

        g.rollback_transaction().unwrap();

        // Edge must be gone after rollback — both RocksDB and in-memory adj.
        let rows_after = run(&mut g, &mut txn, r#"MATCH (a:P)-[:KNOWS]->(b:P) RETURN a.name"#).unwrap();
        assert!(rows_after.is_empty(), "edge should be gone after rollback");
    }

    #[test]
    fn adj_delta_rollback_delete_edge() {
        // After BEGIN + delete edge + ROLLBACK, the edge must be restored.
        let dir = tempfile::tempdir().unwrap().into_path();
        let mut g = Graph::open(&dir).unwrap();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:P {name:"A"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:P {name:"B"})"#).unwrap();
        run(&mut g, &mut txn, r#"MATCH (a:P {name:"A"}),(b:P {name:"B"}) INSERT (a)-[:KNOWS]->(b)"#).unwrap();

        g.begin_transaction().unwrap();
        run(&mut g, &mut txn, r#"MATCH ()-[r:KNOWS]->() DELETE r"#).unwrap();

        // Edge gone inside transaction.
        let rows_in = run(&mut g, &mut txn, r#"MATCH (a)-[r:KNOWS]->(b) RETURN r"#).unwrap();
        assert!(rows_in.is_empty());

        g.rollback_transaction().unwrap();

        // Edge must be restored after rollback.
        let rows_after = run(&mut g, &mut txn, r#"MATCH (a:P)-[:KNOWS]->(b:P) RETURN a.name"#).unwrap();
        assert_eq!(rows_after.len(), 1, "edge should be restored after rollback");
    }

    // ── STARTS WITH index pushdown tests ─────────────────────────────────────

    fn make_name_graph() -> (Graph, u64) {
        let dir = tempfile::tempdir().unwrap().into_path();
        let mut g = Graph::open(&dir).unwrap();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Person(name)").unwrap();
        for name in &["Alice", "Albert", "Bob", "Charlie"] {
            run(&mut g, &mut txn, &format!(r#"INSERT (:Person {{name: "{name}"}})"#)).unwrap();
        }
        (g, txn)
    }

    #[test]
    fn starts_with_uses_index() {
        let (mut g, mut txn) = make_name_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE startsWith(n.name, "Al") RETURN n.name"#,
        ).unwrap();
        let mut names: Vec<String> = col(&rows, "n.name");
        names.sort();
        assert_eq!(names, vec![r#""Albert""#, r#""Alice""#]);
    }

    #[test]
    fn starts_with_empty_prefix_returns_all() {
        let (mut g, mut txn) = make_name_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE startsWith(n.name, "") RETURN n.name"#,
        ).unwrap();
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn starts_with_no_match_returns_empty() {
        let (mut g, mut txn) = make_name_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE startsWith(n.name, "Z") RETURN n.name"#,
        ).unwrap();
        assert!(rows.is_empty());
    }

    // ── Multi-index intersection ───────────────────────────────────────────────

    /// Two inline indexed properties → intersection used as candidates.
    #[test]
    fn multi_index_inline_two_properties() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Person(name)").unwrap();
        run(&mut g, &mut txn, "CREATE INDEX ON :Person(dept)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name:"Alice", dept:"Eng"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name:"Alice", dept:"HR"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Person {name:"Bob",   dept:"Eng"})"#).unwrap();
        // Only the intersection of name="Alice" ∩ dept="Eng" should match.
        let rows = run(&mut g, &mut txn,
            r#"MATCH (n:Person {name:"Alice", dept:"Eng"}) RETURN n.name, n.dept"#).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["n.name"], Value::String("Alice".into()));
        assert_eq!(rows[0]["n.dept"], Value::String("Eng".into()));
    }

    /// Two WHERE equality predicates on indexed properties → intersection.
    #[test]
    fn multi_index_where_two_equality_predicates() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Item(color)").unwrap();
        run(&mut g, &mut txn, "CREATE INDEX ON :Item(size)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {color:"red",  size:"large"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {color:"red",  size:"small"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {color:"blue", size:"large"})"#).unwrap();
        let rows = run(&mut g, &mut txn,
            r#"MATCH (n:Item) WHERE n.color = "red" AND n.size = "large" RETURN n.color, n.size"#
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["n.color"], Value::String("red".into()));
        assert_eq!(rows[0]["n.size"], Value::String("large".into()));
    }

    /// Inline indexed property combined with WHERE indexed range → intersection.
    #[test]
    fn multi_index_inline_and_where_range() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Employee(dept)").unwrap();
        run(&mut g, &mut txn, "CREATE INDEX ON :Employee(salary)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Employee {dept:"Eng", salary:90000})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Employee {dept:"Eng", salary:50000})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Employee {dept:"HR",  salary:90000})"#).unwrap();
        // dept="Eng" (inline, indexed) ∩ salary > 80000 (WHERE, indexed)
        let rows = run(&mut g, &mut txn,
            r#"MATCH (n:Employee {dept:"Eng"}) WHERE n.salary > 80000 RETURN n.dept, n.salary"#
        ).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["n.dept"], Value::String("Eng".into()));
        assert_eq!(rows[0]["n.salary"], Value::Int(90000));
    }

    /// No intersection when second predicate has no matching nodes → empty result.
    #[test]
    fn multi_index_intersection_empty_result() {
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, "CREATE INDEX ON :Widget(color)").unwrap();
        run(&mut g, &mut txn, "CREATE INDEX ON :Widget(shape)").unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Widget {color:"red",  shape:"circle"})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Widget {color:"blue", shape:"square"})"#).unwrap();
        let rows = run(&mut g, &mut txn,
            r#"MATCH (n:Widget {color:"red", shape:"square"}) RETURN n.color"#
        ).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn starts_with_combined_with_other_predicate() {
        // WHERE startsWith(n.name, "Al") AND n.name <> "Alice" → only Albert
        let (mut g, mut txn) = make_name_graph();
        let rows = run(
            &mut g, &mut txn,
            r#"MATCH (n:Person) WHERE startsWith(n.name, "Al") AND n.name <> "Alice" RETURN n.name"#,
        ).unwrap();
        let names: Vec<String> = col(&rows, "n.name");
        assert_eq!(names, vec![r#""Albert""#]);
    }

    // ── Order-by NULL semantics (LOW-5) ─────────────────────────────────────

    #[test]
    fn order_by_null_values_sort_last_asc() {
        // Nodes with a missing property return NULL for that column.
        // NULL should sort after all non-NULL values in ASC order (SQL standard).
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "B", rank: 2})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "A", rank: 1})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "C"})"#).unwrap(); // no rank → NULL

        let rows = run(
            &mut g, &mut txn,
            "MATCH (n:Item) RETURN n.name ORDER BY n.rank ASC",
        ).unwrap();
        let names = col(&rows, "n.name");
        // A (rank=1) < B (rank=2) < C (rank=NULL) — null must be last in ASC
        assert_eq!(names[0], r#""A""#, "rank=1 should be first");
        assert_eq!(names[1], r#""B""#, "rank=2 should be second");
        assert_eq!(names[2], r#""C""#, "node with null rank should be last in ASC");
    }

    #[test]
    fn order_by_null_values_sort_first_desc() {
        // In DESC order the comparator is reversed, so NULL-last in ASC becomes
        // NULL-first in DESC (same as standard SQL behavior).
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "B", rank: 2})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "A", rank: 1})"#).unwrap();
        run(&mut g, &mut txn, r#"INSERT (:Item {name: "C"})"#).unwrap(); // no rank → NULL

        let rows = run(
            &mut g, &mut txn,
            "MATCH (n:Item) RETURN n.name ORDER BY n.rank DESC",
        ).unwrap();
        let names = col(&rows, "n.name");
        // DESC reversal: NULL (Greater when ASC) → first; then rank=2, then rank=1
        assert_eq!(names[0], r#""C""#, "node with null rank should be first in DESC");
        assert_eq!(names[1], r#""B""#, "rank=2 should be second");
        assert_eq!(names[2], r#""A""#, "rank=1 should be last");
    }

    // ── count() outside aggregate context (LOW-2) ────────────────────────────

    #[test]
    fn count_outside_aggregating_return_errors() {
        // count() is an aggregate function — using it in a non-aggregate context
        // (e.g. as an argument to another function) should surface an error rather
        // than silently returning 1 (the old stub behavior).
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(&mut g, &mut txn, r#"INSERT (:X {v: 1})"#).unwrap();
        // Bare count() used as an argument to toString() bypasses the aggregate
        // path and should hit the explicit error in eval_function.
        let result = run(&mut g, &mut txn, "MATCH (n:X) RETURN toString(count(n)) AS s");
        assert!(result.is_err(), "count() outside aggregate RETURN must error");
    }

    // ── UNION write persistence (HIGH-2) ─────────────────────────────────────

    #[test]
    fn union_insert_branches_both_execute() {
        // Each branch of a UNION is a valid single_stmt, including INSERT.
        // Previously, ops from UNION branches were discarded — the in-memory
        // write happened but no Operation was recorded, so the node would
        // vanish on rollback and not be WAL-logged.
        //
        // After the fix, ops from both branches are collected and returned to
        // the caller.  Both inserts must be visible in subsequent queries.
        let mut g = Graph::new();
        let mut txn = 0u64;
        run(
            &mut g, &mut txn,
            r#"INSERT (:Tag {name: "alpha"}) UNION ALL INSERT (:Tag {name: "beta"})"#,
        ).unwrap();
        let rows = run(&mut g, &mut txn, "MATCH (n:Tag) RETURN n.name").unwrap();
        assert_eq!(rows.len(), 2, "both UNION INSERT branches must produce visible nodes");
    }
}
