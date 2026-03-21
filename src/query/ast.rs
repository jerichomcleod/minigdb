//! AST node types for the minigdb GQL subset.
//!
//! The parser (`parser.rs`) converts a raw GQL string into a [`Statement`],
//! which the executor (`executor.rs`) then evaluates against a [`Graph`].
//!
//! ## Statement taxonomy
//!
//! | Variant | GQL syntax |
//! |---|---|
//! | `Match` | `MATCH … RETURN` |
//! | `OptionalMatch` | `OPTIONAL MATCH … RETURN` |
//! | `MatchOptionalMatch` | `MATCH … OPTIONAL MATCH … RETURN` |
//! | `MatchWith` | `MATCH … WITH … RETURN` |
//! | `Union` | `stmt UNION [ALL] stmt` |
//! | `Unwind` | `UNWIND expr AS var RETURN` |
//! | `Insert` | `INSERT (…)` |
//! | `MatchInsert` | `MATCH … INSERT (…)` |
//! | `UnwindInsert` | `UNWIND expr AS var INSERT (…)` — bulk insert from list or `$param` |
//! | `Set` | `MATCH … SET n.prop = expr` |
//! | `Remove` | `MATCH … REMOVE n.prop` |
//! | `Delete` | `MATCH … [DETACH] DELETE n` |
//! | `Call` | `CALL algo(…) YIELD …` |
//! | `CallPipeline` | `CALL algo(…) YIELD … [MATCH …] RETURN` |
//! | `CreateIndex` | `CREATE INDEX ON :Label(prop)` |
//! | `DropIndex` | `DROP INDEX ON :Label(prop)` |
//! | `ShowIndexes` | `SHOW INDEXES` |
//! | `Constraint` | `CREATE/DROP CONSTRAINT … ON :Label(prop)` / `SHOW CONSTRAINTS` |
//!
//! ## Parameterized queries
//!
//! GQL strings may contain `$name` placeholders at any expression position.
//! Parameters are supplied at call time via
//! [`query_capturing_with_params`](super::query_capturing_with_params) and resolved by
//! the executor through a thread-local map set immediately before dispatch.
//!
//! ```gql
//! MATCH (n:Person) WHERE n.email = $email RETURN n.name
//! UNWIND $batch AS row INSERT (:Order {id: row.id, total: row.total})
//! ```
//!
//! A missing parameter evaluates to [`Value::Null`](crate::types::Value::Null).

use crate::types::Value;

// ── Top-level statement ────────────────────────────────────────────────────

/// The root of every parsed GQL statement.
///
/// Constructed by [`super::parser::parse`] and consumed by
/// [`super::executor::execute`].
#[derive(Debug, Clone)]
pub enum Statement {
    Match(MatchStatement),
    OptionalMatch(MatchStatement),
    MatchOptionalMatch(MatchOptionalMatchStatement),
    Unwind(UnwindStatement),
    Union(UnionStatement),
    MatchWith(MatchWithStatement),
    Insert(InsertStatement),
    MatchInsert(MatchInsertStatement),
    Set(SetStatement),
    Remove(RemoveStatement),
    Delete(DeleteStatement),
    CreateIndex(CreateIndexStatement),
    DropIndex(DropIndexStatement),
    /// `SHOW INDEXES` — no associated data.
    ShowIndexes,
    Call(CallStatement),
    CallPipeline(CallPipelineStatement),
    /// `LOAD CSV NODES FROM 'path' [LABEL X]`
    Truncate,
    LoadCsvNodes(LoadCsvNodesStatement),
    /// `LOAD CSV EDGES FROM 'path' [LABEL X]`
    LoadCsvEdges(LoadCsvEdgesStatement),
    /// `UNWIND expr AS var INSERT elements`
    UnwindInsert(UnwindInsertStatement),
    /// `CREATE/DROP CONSTRAINT … ON :Label(prop)` / `SHOW CONSTRAINTS`
    Constraint(ConstraintStatement),
}

// ── CALL pipeline ─────────────────────────────────────────────────────────

/// `CALL algo(...) YIELD cols [MATCH pattern [WHERE]] RETURN ...`
///
/// Pipes algorithm output rows into an optional MATCH filter and then into a
/// RETURN clause, allowing algorithm results to be correlated with graph
/// patterns (e.g. look up node properties for each scored node).
#[derive(Debug, Clone)]
pub struct CallPipelineStatement {
    /// Algorithm name (e.g. `"pageRank"`, `"shortestPath"`).
    pub name: String,
    /// Named parameter key-value pairs.
    pub params: Vec<(String, Expr)>,
    /// YIELD column names — required (non-optional) in a pipeline.
    pub yields: Vec<String>,
    /// Optional `MATCH pattern [WHERE cond]` to filter/extend CALL results.
    pub match_clause: Option<CallPipelineMatch>,
    pub return_clause: ReturnClause,
}

/// The optional `MATCH pattern [WHERE cond]` block inside a [`CallPipelineStatement`].
///
/// When present, each CALL output row is used as a seed to extend the graph
/// match — variables yielded by CALL can be referenced in the pattern.
#[derive(Debug, Clone)]
pub struct CallPipelineMatch {
    pub patterns: Vec<GraphPattern>,
    pub where_clause: Option<Expr>,
    pub path_mode: PathMode,
}

// ── CALL (standalone) ─────────────────────────────────────────────────────

/// `CALL algorithmName(key: expr, ...) YIELD col1, col2`
///
/// Invokes a built-in graph algorithm and returns its result rows.
/// All arguments must be named (`source: "id"`, not positional).
#[derive(Debug, Clone)]
pub struct CallStatement {
    /// Algorithm name (e.g. `"shortestPath"`, `"pageRank"`).
    pub name: String,
    /// Named parameter pairs — all values must be constant expressions.
    pub params: Vec<(String, Expr)>,
    /// YIELD projection — if `None`, all algorithm columns are returned.
    pub yields: Option<Vec<String>>,
}

// ── UNWIND ────────────────────────────────────────────────────────────────

/// `UNWIND expr AS var RETURN ...`
///
/// Evaluates `expr` to a `List` and produces one set of bindings per element,
/// binding each element to `variable`.  Analogous to SQL's `UNNEST`.
#[derive(Debug, Clone)]
pub struct UnwindStatement {
    /// Expression that must evaluate to a [`Value::List`].
    pub expr: Expr,
    /// Variable name bound to each list element.
    pub variable: String,
    pub return_clause: ReturnClause,
}

// ── UNION ─────────────────────────────────────────────────────────────────

/// `stmt UNION [ALL] stmt ...`
///
/// Executes each branch independently against the same graph and concatenates
/// the results.  Without `ALL`, duplicate rows are removed (SQL `UNION`).
/// With `ALL`, duplicates are kept (SQL `UNION ALL`).
#[derive(Debug, Clone)]
pub struct UnionStatement {
    /// Two or more statement branches.
    pub branches: Vec<Statement>,
    /// `true` = `UNION ALL` (keep duplicates); `false` = `UNION` (deduplicate).
    pub all: bool,
}

// ── WITH (pipeline projection) ────────────────────────────────────────────

/// `WITH [DISTINCT] items [WHERE cond]`
///
/// Projects MATCH bindings into a new named scope, optionally filtering.
/// Analogous to a SQL subquery that renames columns before the outer query.
#[derive(Debug, Clone)]
pub struct WithClause {
    /// If `true`, deduplicate on the projected columns before continuing.
    pub distinct: bool,
    pub items: Vec<ReturnItem>,
    /// Optional filter applied after projection.
    pub where_clause: Option<Expr>,
}

/// `MATCH pattern [WHERE] WITH projections [WHERE] RETURN ...`
///
/// The WITH clause acts as a pipeline checkpoint: it projects MATCH bindings
/// into a new scope, optionally filters them, and feeds the result into RETURN.
#[derive(Debug, Clone)]
pub struct MatchWithStatement {
    pub patterns: Vec<GraphPattern>,
    pub where_clause: Option<Expr>,
    pub path_mode: PathMode,
    pub with_clause: WithClause,
    pub return_clause: ReturnClause,
}

// ── MATCH … OPTIONAL MATCH … RETURN ──────────────────────────────────────

/// A single `OPTIONAL MATCH pattern [WHERE cond]` clause in a chained statement.
///
/// Each optional clause performs a LEFT JOIN on the rows accumulated so far:
/// rows that fail to match produce `null` bindings for the new variables
/// rather than being eliminated.
#[derive(Debug, Clone)]
pub struct OptionalMatchClause {
    pub patterns: Vec<GraphPattern>,
    pub where_clause: Option<Expr>,
    pub path_mode: PathMode,
}

/// `MATCH pattern [WHERE] (OPTIONAL MATCH pattern [WHERE])+ RETURN …`
///
/// Equivalent to a chain of SQL LEFT JOINs: each OPTIONAL MATCH extends the
/// current binding set without discarding non-matching rows.
#[derive(Debug, Clone)]
pub struct MatchOptionalMatchStatement {
    /// The mandatory initial MATCH patterns.
    pub patterns: Vec<GraphPattern>,
    pub where_clause: Option<Expr>,
    pub path_mode: PathMode,
    /// One or more OPTIONAL MATCH clauses applied in order.
    pub optional_clauses: Vec<OptionalMatchClause>,
    pub return_clause: ReturnClause,
}

// ── Index management ──────────────────────────────────────────────────────

/// `CREATE INDEX ON :Label(property)`
#[derive(Debug, Clone)]
pub struct CreateIndexStatement {
    pub label: String,
    pub property: String,
}

/// `DROP INDEX ON :Label(property)`
#[derive(Debug, Clone)]
pub struct DropIndexStatement {
    pub label: String,
    pub property: String,
}

// ── Path mode & quantifier ────────────────────────────────────────────────

/// Controls path deduplication semantics for variable-length MATCH traversals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathMode {
    /// Default (no keyword): edges and nodes may be revisited.
    #[default]
    Walk,
    /// `TRAIL`: no repeated *edges* within a single path.
    Trail,
    /// `SIMPLE`: no repeated *nodes* within a single path (implies no repeated edges).
    Simple,
}

/// Hop-count constraint on a variable-length edge pattern (`[r*1..3]`, `[r+]`, …).
#[derive(Debug, Clone)]
pub struct PathQuantifier {
    /// Minimum number of hops (inclusive).
    pub min: u32,
    /// Maximum number of hops (inclusive).  `None` = unbounded;
    /// capped at runtime based on graph size and path mode to prevent runaway traversals.
    pub max: Option<u32>,
}

// ── MATCH ─────────────────────────────────────────────────────────────────

/// `MATCH [path_mode] pattern [WHERE cond] RETURN ...`
///
/// When `patterns` contains more than one entry the patterns are
/// cross-joined (Cartesian product), allowing disconnected subgraphs to be
/// matched in one statement.
#[derive(Debug, Clone)]
pub struct MatchStatement {
    /// One or more graph patterns; multiple patterns are cross-joined.
    pub patterns: Vec<GraphPattern>,
    pub where_clause: Option<Expr>,
    pub return_clause: ReturnClause,
    pub path_mode: PathMode,
}

/// A linear graph pattern: a start node followed by zero or more
/// edge–node steps (e.g. `(a:Person)-[:KNOWS]->(b:Person)`).
#[derive(Debug, Clone)]
pub struct GraphPattern {
    pub start: NodePattern,
    pub steps: Vec<EdgePatternStep>,
}

/// A node constraint in a graph pattern (e.g. `(n:Person {name: "Alice"})`).
#[derive(Debug, Clone)]
pub struct NodePattern {
    /// Optional variable binding (e.g. `n` in `(n:Person)`).
    pub variable: Option<String>,
    /// Label filters — all must be present on a matching node.
    /// Empty = match any node.
    pub labels: Vec<String>,
    /// Inline property equality constraints.
    pub properties: Vec<PropertyConstraint>,
}

/// One step in a graph pattern: an edge constraint plus the node it leads to.
#[derive(Debug, Clone)]
pub struct EdgePatternStep {
    pub edge: EdgePattern,
    pub node: NodePattern,
}

/// An edge constraint in a graph pattern (e.g. `[r:KNOWS*1..3]`).
#[derive(Debug, Clone)]
pub struct EdgePattern {
    /// Optional variable binding (e.g. `r` in `[r:KNOWS]`).
    pub variable: Option<String>,
    /// Optional label filter (e.g. `"KNOWS"`).
    pub label: Option<String>,
    pub properties: Vec<PropertyConstraint>,
    pub direction: EdgeDirection,
    /// Present on variable-length patterns (`*`, `+`, `{m,n}`).
    pub quantifier: Option<PathQuantifier>,
}

/// Traversal direction of an edge pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    /// `-->`: edge points away from the left node.
    Outgoing,
    /// `<--`: edge points toward the left node.
    Incoming,
    /// `--`: match in either direction (undirected or directed).
    Either,
}

/// An inline property equality check inside a node or edge pattern.
#[derive(Debug, Clone)]
pub struct PropertyConstraint {
    pub key: String,
    pub value: Expr,
}

// ── RETURN ────────────────────────────────────────────────────────────────

/// The `RETURN [DISTINCT] items [ORDER BY …] [LIMIT n] [OFFSET n]` clause.
#[derive(Debug, Clone)]
pub struct ReturnClause {
    /// If `true`, duplicate rows are eliminated before output.
    pub distinct: bool,
    /// The expressions (and optional aliases) to project.
    pub items: Vec<ReturnItem>,
    /// `ORDER BY` sort keys; empty = no sorting.
    pub order_by: Vec<OrderByItem>,
    /// Maximum number of rows to return.
    pub limit: Option<Expr>,
    /// Number of leading rows to skip.
    pub offset: Option<Expr>,
}

/// A single projected expression in a RETURN or WITH clause.
#[derive(Debug, Clone)]
pub struct ReturnItem {
    pub expr: Expr,
    /// Optional `AS alias` renaming.
    pub alias: Option<String>,
}

/// A single sort key in an `ORDER BY` clause.
#[derive(Debug, Clone)]
pub struct OrderByItem {
    pub expr: Expr,
    /// `true` = ascending (default); `false` = descending (`DESC`).
    pub ascending: bool,
}

// ── MATCH + INSERT ────────────────────────────────────────────────────────

/// `MATCH (a:X), (b:Y) [WHERE cond] INSERT (a)-[:REL]->(b)`
///
/// Matches existing nodes/edges and inserts new graph elements using the
/// matched variable bindings as endpoints.  Only the *new* elements are
/// inserted; matched nodes are not duplicated.
#[derive(Debug, Clone)]
pub struct MatchInsertStatement {
    /// One or more disconnected patterns; matched as a cross-product.
    pub patterns: Vec<GraphPattern>,
    pub where_clause: Option<Expr>,
    pub elements: Vec<InsertElement>,
}

// ── INSERT ────────────────────────────────────────────────────────────────

/// `INSERT element, ...`
#[derive(Debug, Clone)]
pub struct InsertStatement {
    pub elements: Vec<InsertElement>,
}

/// A node or edge to be inserted.
#[derive(Debug, Clone)]
pub enum InsertElement {
    Node(InsertNode),
    Edge(InsertEdge),
}

/// A node to be inserted (e.g. `(:Person {name: "Alice"})`).
#[derive(Debug, Clone)]
pub struct InsertNode {
    /// Optional variable name — used to reference this node in a subsequent
    /// edge definition within the same INSERT statement.
    pub variable: Option<String>,
    pub labels: Vec<String>,
    pub properties: Vec<PropertyAssignment>,
}

/// An edge to be inserted (e.g. `(a)-[:KNOWS {since: 2020}]->(b)`).
#[derive(Debug, Clone)]
pub struct InsertEdge {
    /// Variable name referencing the source node.
    pub from_var: String,
    /// Variable name referencing the target node.
    pub to_var: String,
    pub label: String,
    pub properties: Vec<PropertyAssignment>,
    /// `true` = directed (`->`); `false` = undirected (`--`).
    pub directed: bool,
}

/// A `key: expr` property assignment in an INSERT or SET clause.
#[derive(Debug, Clone)]
pub struct PropertyAssignment {
    pub key: String,
    pub value: Expr,
}

// ── SET ───────────────────────────────────────────────────────────────────

/// `MATCH pattern [WHERE cond] SET item, ...`
#[derive(Debug, Clone)]
pub struct SetStatement {
    pub match_pattern: GraphPattern,
    pub where_clause: Option<Expr>,
    pub assignments: Vec<SetItem>,
}

/// A single SET operation.
#[derive(Debug, Clone)]
pub enum SetItem {
    /// `n.prop = expr` — set or overwrite a property.
    Property { variable: String, key: String, value: Expr },
    /// `n:Label` — add a label to the node.
    AddLabel { variable: String, label: String },
}

// ── REMOVE ────────────────────────────────────────────────────────────────

/// `MATCH pattern [WHERE cond] REMOVE item, ...`
#[derive(Debug, Clone)]
pub struct RemoveStatement {
    pub match_pattern: GraphPattern,
    pub where_clause: Option<Expr>,
    pub items: Vec<RemoveItem>,
}

/// A single REMOVE operation.
#[derive(Debug, Clone)]
pub enum RemoveItem {
    /// `n.prop` — remove a property key from the node or edge.
    Property { variable: String, key: String },
    /// `n:Label` — remove a label from the node.
    Label { variable: String, label: String },
}

// ── DELETE ────────────────────────────────────────────────────────────────

/// `MATCH pattern [WHERE cond] [DETACH] DELETE var, ...`
#[derive(Debug, Clone)]
pub struct DeleteStatement {
    pub match_pattern: GraphPattern,
    pub where_clause: Option<Expr>,
    /// Variables to delete (must resolve to nodes or edges at runtime).
    pub variables: Vec<String>,
    /// If `true`, incident edges are removed before the node (`DETACH DELETE`).
    pub detach: bool,
}

// ── LOAD CSV ──────────────────────────────────────────────────────────────────

/// `LOAD CSV NODES FROM 'path' [LABEL LabelName]`
///
/// Bulk-inserts nodes from a CSV file.  The CSV must use the `:ID` / `:LABEL`
/// column conventions (see [`crate::csv_import`]).
#[derive(Debug, Clone)]
pub struct LoadCsvNodesStatement {
    /// Path to the CSV file (resolved relative to the working directory).
    pub path: String,
    /// Default label applied when the CSV row has no `:LABEL` column or an
    /// empty cell.  `None` = no label.
    pub label: Option<String>,
}

/// `LOAD CSV EDGES FROM 'path' [LABEL EdgeLabel]`
///
/// Bulk-inserts edges from a CSV file.  Uses the `csv_id_map` stored in the
/// graph by a preceding `LOAD CSV NODES` call to resolve `:START_ID` / `:END_ID`.
#[derive(Debug, Clone)]
pub struct LoadCsvEdgesStatement {
    /// Path to the CSV file (resolved relative to the working directory).
    pub path: String,
    /// Default label applied when the CSV row has no `:TYPE` column or an
    /// empty cell.  `None` = `"RELATED"`.
    pub label: Option<String>,
}

// ── Expressions ───────────────────────────────────────────────────────────

/// An expression that can appear in WHERE, RETURN, SET, ORDER BY, etc.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A compile-time constant value.
    Literal(Value),
    /// A variable reference: `n`, `r`, etc.
    Var(String),
    /// Property access: `n.name`, `r.weight`.
    Property(Box<Expr>, String),
    /// Binary infix operation: `a + b`, `a AND b`, `a IN list`.
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    /// Logical negation: `NOT expr`.
    Not(Box<Expr>),
    /// `expr IS [NOT] NULL` — second field is `true` for `IS NOT NULL`.
    IsNull(Box<Expr>, bool),
    /// Function call: `count(n)`, `labels(n)`, `toLower(s)`, etc.
    Call(String, Vec<Expr>),
    /// List literal: `[1, 2, 3]`.
    List(Vec<Expr>),
    /// Wildcard `*` in `RETURN *`.
    Star,
    /// A named query parameter `$name` supplied at call time.
    Param(String),
}

// ── UNWIND … INSERT ──────────────────────────────────────────────────────────

/// `UNWIND expr AS var INSERT elements`
///
/// Evaluates `expr` to a `List` and inserts the given elements once per
/// element, with `variable` bound to the current element.  Useful for
/// bulk-inserting nodes from a literal list or a `$param`.
#[derive(Debug, Clone)]
pub struct UnwindInsertStatement {
    /// Expression that must evaluate to a [`Value::List`].
    pub expr: Expr,
    /// Variable bound to each list element inside `elements`.
    pub variable: String,
    pub elements: Vec<InsertElement>,
}

// ── CONSTRAINT statements ─────────────────────────────────────────────────────

/// `CREATE CONSTRAINT … ON :Label(prop)` / `DROP CONSTRAINT … ON :Label(prop)` / `SHOW CONSTRAINTS`
#[derive(Debug, Clone)]
pub struct ConstraintStatement {
    pub op: ConstraintOp,
}

/// The three constraint operations.
#[derive(Debug, Clone)]
pub enum ConstraintOp {
    /// `CREATE CONSTRAINT UNIQUE ON :Label(property)`
    Create {
        kind: ConstraintKind,
        label: String,
        property: String,
    },
    /// `DROP CONSTRAINT UNIQUE ON :Label(property)`
    Drop {
        kind: ConstraintKind,
        label: String,
        property: String,
    },
    /// `SHOW CONSTRAINTS`
    Show,
}

/// Mirror of [`crate::graph::constraints::ConstraintKind`] — duplicated here
/// so the AST has no dependency on the graph layer.
#[derive(Debug, Clone)]
pub enum ConstraintKind {
    Unique,
    Type(ValueKind),
}

/// Mirror of [`crate::graph::constraints::ValueKind`].
#[derive(Debug, Clone)]
pub enum ValueKind {
    Integer,
    Float,
    String,
    Boolean,
}

/// Binary operator — used in [`Expr::BinOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    And,
    Or,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    /// `x IN list` — true if `x` equals any element of the list.
    In,
}
