//! GQL query pipeline: parse → AST → execute.
//!
//! ## Entry points
//!
//! | Function | Use case |
//! |---|---|
//! | [`query`] | Parse and execute a GQL string; return result rows |
//! | [`query_capturing`] | Same, but also return the [`Operation`]s applied — for WAL persistence and explicit transactions |
//!
//! ## Pipeline
//!
//! ```text
//! &str  ──►  parse_or_cache()  ──►  Statement (AST)  ──►  executor::execute()  ──►  Vec<Row>
//! ```
//!
//! ## Parse cache
//!
//! Parsed ASTs are cached in a process-wide `Mutex<HashMap>` keyed on the GQL
//! string.  This avoids re-running the pest grammar on every call — especially
//! important for hot loops that execute the same parameterised query thousands
//! of times.  The cache holds at most [`PARSE_CACHE_CAP`] entries; when full it
//! is cleared entirely (simple eviction — a more sophisticated LRU is not worth
//! the extra dependency for this use case).
//!
//! The parser uses a PEG grammar (`gql.pest`) via the `pest` crate.
//! The executor is a recursive tree-walker over the AST.

pub(crate) mod ast;
pub(crate) mod executor;
pub(crate) mod parser;

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::graph::Graph;
use crate::transaction::operation::Operation;
use crate::types::{DbError, Value};

pub use executor::Row;

// ── Parse cache ───────────────────────────────────────────────────────────────

/// Maximum number of distinct GQL strings held in the parse cache.
///
/// When this limit is reached the entire cache is cleared (not LRU-evicted).
/// 512 covers virtually all real-world query patterns without unbounded growth.
const PARSE_CACHE_CAP: usize = 512;

static PARSE_CACHE: OnceLock<Mutex<HashMap<String, ast::Statement>>> = OnceLock::new();

/// Parse `input`, returning a cached clone when available.
///
/// Only successful parses are cached.  Failed parses propagate the error
/// without touching the cache.
fn parse_or_cache(input: &str) -> Result<ast::Statement, DbError> {
    let cache = PARSE_CACHE.get_or_init(|| Mutex::new(HashMap::with_capacity(64)));

    // Check cache first (brief lock, no parse work inside).
    {
        let guard = cache.lock().unwrap();
        if let Some(stmt) = guard.get(input) {
            return Ok(stmt.clone());
        }
    }

    // Cache miss — parse, then insert.
    let stmt = parser::parse(input)?;
    {
        let mut guard = cache.lock().unwrap();
        if guard.len() >= PARSE_CACHE_CAP {
            guard.clear();
        }
        guard.insert(input.to_string(), stmt.clone());
    }
    Ok(stmt)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse and execute a GQL string against `graph`.
///
/// `next_txn_id` is a monotonic counter used to assign unique IDs to implicit
/// transactions; the caller must persist it between calls.
///
/// # Errors
///
/// Returns [`DbError::Parse`] if the GQL is syntactically invalid, or
/// [`DbError::Query`] / [`DbError::Storage`] for runtime and I/O failures.
pub fn query(input: &str, graph: &mut Graph, next_txn_id: &mut u64) -> Result<Vec<Row>, DbError> {
    let stmt = parse_or_cache(input)?;
    executor::execute(stmt, graph, next_txn_id)
}

/// Like [`query`], but also returns the [`Operation`]s applied to the graph.
///
/// Used by the REPL and server to write WAL frames after each auto-commit
/// statement and to buffer ops for explicit `BEGIN`/`COMMIT` transactions.
pub fn query_capturing(
    input: &str,
    graph: &mut Graph,
    next_txn_id: &mut u64,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    let stmt = parse_or_cache(input)?;
    executor::execute_capturing(stmt, graph, next_txn_id)
}

/// Like [`query_capturing`], but binds named `$param` placeholders.
///
/// Parameters are supplied as a `HashMap<String, Value>` where each key
/// matches the `$name` placeholder used in the GQL string (without the `$`).
///
/// ```gql
/// UNWIND $names AS name INSERT (:Person {name: name})
/// ```
/// ```rust,no_run
/// # use std::collections::HashMap;
/// # use minigdb::{query_capturing_with_params, Value};
/// # let (_, mut graph) = minigdb::StorageManager::open(std::path::Path::new("/tmp/x")).unwrap();
/// # let mut txn = 0u64;
/// let mut p = HashMap::new();
/// p.insert("names".into(), Value::List(vec![Value::String("Alice".into())]));
/// query_capturing_with_params("UNWIND $names AS name INSERT (:Person {name: name})", &mut graph, &mut txn, p).unwrap();
/// ```
pub fn query_capturing_with_params(
    input: &str,
    graph: &mut Graph,
    next_txn_id: &mut u64,
    params: HashMap<String, Value>,
) -> Result<(Vec<Row>, Vec<Operation>), DbError> {
    let stmt = parse_or_cache(input)?;
    executor::execute_capturing_with_params(stmt, graph, next_txn_id, params)
}
