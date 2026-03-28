//! PyO3 bindings that expose minigdb as a native Python extension module named
//! `minigdb` (built with `maturin --features python`).
//!
//! # Python API surface
//!
//! ## Local access
//! ```python
//! import minigdb
//!
//! # Open (or create) a named graph in the platform data directory.
//! db = minigdb.open("mygraph")
//!
//! # Open a graph at an explicit path — useful for testing.
//! db = minigdb.open_at("/tmp/test_graph")
//!
//! # Execute a GQL statement; returns a list[dict].
//! rows = db.query("MATCH (n:Person) RETURN n.name, n.age")
//!
//! # DataFrame variants (polars / pandas must be installed separately).
//! df = db.query_df("MATCH (n:Person) RETURN n.name")      # polars
//! df = db.query_pandas("MATCH (n:Person) RETURN n.name")  # pandas
//!
//! # Explicit transactions.
//! db.begin()
//! db.query("INSERT (:Person {name:'Alice'})")
//! db.commit()   # or db.rollback()
//!
//! # Context-manager support guarantees close() even on exception.
//! with minigdb.open("mygraph") as db:
//!     rows = db.query("MATCH (n) RETURN n")
//! ```
//!
//! ## Remote access
//! ```python
//! with minigdb.connect("localhost:7474") as db:
//!     rows = db.query("MATCH (n:Person) RETURN n.name")
//! ```
//!
//! # Error mapping
//! - [`DbError::Storage`] (wraps `std::io::Error`) → `OSError`
//! - All other [`DbError`] variants → `RuntimeError`
//!
//! # Feature gate
//! This module is compiled only when the `python` Cargo feature is enabled.
//! The crate must be built as `cdylib` for Python to import it.

use pyo3::exceptions::{PyOSError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::{Graph, StorageManager, Value};

// ── Value → Python ────────────────────────────────────────────────────────────

/// Recursively convert a minigdb [`Value`] to its natural Python equivalent.
///
/// | Rust `Value`      | Python type |
/// |-------------------|-------------|
/// | `Null`            | `None`      |
/// | `Bool(b)`         | `bool`      |
/// | `Int(i)`          | `int`       |
/// | `Float(f)`        | `float`     |
/// | `String(s)`       | `str`       |
/// | `List(items)`     | `list`      |
fn value_to_py(py: Python<'_>, v: &Value) -> PyObject {
    match v {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_py(py),
        Value::Int(i) => i.into_py(py),
        Value::Float(f) => f.into_py(py),
        Value::String(s) => s.into_py(py),
        Value::List(items) => items
            .iter()
            .map(|x| value_to_py(py, x))
            .collect::<Vec<_>>()
            .into_py(py),
        Value::Map(m) => {
            let d = PyDict::new_bound(py);
            for (k, val) in m {
                d.set_item(k, value_to_py(py, val)).unwrap();
            }
            d.into_py(py)
        }
    }
}

/// Convert a Python object to a minigdb [`Value`].
///
/// Supports `None`, `bool`, `int`, `float`, `str`, `list`, and `dict`.
/// Other types are coerced to their string representation.
fn py_to_value(py: Python<'_>, obj: &PyObject) -> Value {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString};
    let bound = obj.bind(py);
    if bound.is_none() {
        return Value::Null;
    }
    if let Ok(b) = bound.downcast::<PyBool>() {
        return Value::Bool(b.is_true());
    }
    if let Ok(i) = bound.downcast::<PyInt>() {
        if let Ok(n) = i.extract::<i64>() {
            return Value::Int(n);
        }
    }
    if let Ok(f) = bound.downcast::<PyFloat>() {
        return Value::Float(f.value());
    }
    if let Ok(s) = bound.downcast::<PyString>() {
        return Value::String(s.extract::<String>().unwrap_or_default());
    }
    if let Ok(lst) = bound.downcast::<PyList>() {
        let items: Vec<Value> = lst.iter()
            .map(|x| py_to_value(py, &x.into_py(py)))
            .collect();
        return Value::List(items);
    }
    if let Ok(d) = bound.downcast::<PyDict>() {
        let mut map = std::collections::HashMap::new();
        for (k, v) in d.iter() {
            let key = k.str().and_then(|s| s.extract::<String>()).unwrap_or_default();
            map.insert(key, py_to_value(py, &v.into_py(py)));
        }
        return Value::Map(map);
    }
    // Fallback: string representation.
    Value::String(obj.bind(py).str().and_then(|s| s.extract::<String>()).unwrap_or_default())
}

// ── DbError → PyErr ───────────────────────────────────────────────────────────

/// Map a [`crate::types::DbError`] to the appropriate Python exception type.
///
/// `DbError::Storage` carries a `std::io::Error` and maps to `OSError` so that
/// Python callers can use `except OSError` for I/O failures.  Every other error
/// variant maps to `RuntimeError`.
fn db_err_to_py(e: crate::types::DbError) -> PyErr {
    use crate::types::DbError::*;
    match e {
        Storage(io) => PyOSError::new_err(io.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

// ── Columnar conversion ───────────────────────────────────────────────────────

/// Convert row-oriented query results into a columnar Python dict `{col: [v, …]}`.
///
/// Column order is determined by first-seen order across all rows.
/// If a row is missing a key (heterogeneous results), `None` is inserted.
/// This format is accepted directly by both `polars.DataFrame()` and `pandas.DataFrame()`.
fn rows_to_columnar(
    py: Python<'_>,
    rows: &[HashMap<String, Value>],
) -> PyResult<PyObject> {
    // Collect column names in first-seen order (stable across identical queries).
    let mut col_order: Vec<String> = Vec::new();
    let mut col_seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for row in rows {
        for k in row.keys() {
            if col_seen.insert(k.as_str()) {
                col_order.push(k.clone());
            }
        }
    }

    let d = PyDict::new_bound(py);
    for col in &col_order {
        let vals: Vec<PyObject> = rows
            .iter()
            .map(|row| {
                // Missing keys in sparse rows become Python None.
                row.get(col.as_str())
                    .map_or_else(|| py.None(), |v| value_to_py(py, v))
            })
            .collect();
        d.set_item(col, vals)?;
    }
    Ok(d.into_py(py))
}

// ── MiniGdb class ─────────────────────────────────────────────────────────────

/// A handle to a named graph database opened in the current process.
///
/// Wraps a [`Graph`] and its [`StorageManager`] so that Python callers never
/// touch the underlying Rust types directly.  Both fields are `Option` so that
/// [`MiniGdb::close`] can take ownership and release the RocksDB lock without
/// consuming `self` (PyO3 requires `&mut self`).
///
/// Create instances via the module-level [`open`] or [`open_at`] functions —
/// do not construct this class directly from Python.
#[pyclass]
pub struct MiniGdb {
    /// Live graph instance; `None` after `close()` has been called.
    graph: Option<Graph>,
    /// Paired storage manager used for checkpointing; `None` after `close()`.
    storage: Option<StorageManager>,
    /// Monotonically increasing counter used to assign IDs to WAL frames/ops.
    next_txn_id: u64,
}

#[pymethods]
impl MiniGdb {
    /// Execute a GQL statement and return a list of result dicts.
    ///
    /// Python signature: `query(gql: str) -> list[dict]`
    ///
    /// Each dict maps column names (as returned by `RETURN`) to Python values.
    /// Outside a transaction each statement is auto-committed to the WAL.
    /// Inside a `begin()`…`commit()` block the ops are buffered in memory and
    /// written atomically only when `commit()` is called.
    ///
    /// Raises `RuntimeError` if the database has been closed or if the GQL is
    /// invalid.  Raises `OSError` on I/O failure.
    fn query(&mut self, py: Python<'_>, gql: &str) -> PyResult<Vec<PyObject>> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        // Release the GIL while executing the query so the Jupyter kernel
        // event loop (and VS Code's cell-status polling) stays responsive.
        let (rows, _ops) = py.allow_threads(|| {
            crate::query_capturing(gql, graph, &mut self.next_txn_id)
        }).map_err(db_err_to_py)?;
        // Graph manages transaction buffering internally (R3).

        // Convert rows (HashMap<String, Value>) to Python dicts.
        let py_rows = rows
            .into_iter()
            .map(|row| {
                let d = PyDict::new_bound(py);
                for (k, v) in &row {
                    d.set_item(k, value_to_py(py, v)).unwrap();
                }
                d.into_py(py)
            })
            .collect();

        Ok(py_rows)
    }

    /// Execute a GQL statement with named `$param` substitutions.
    ///
    /// Python signature: `query_with_params(gql: str, params: dict) -> list[dict]`
    ///
    /// Pass a dict mapping parameter names (without the leading `$`) to values:
    ///
    /// ```python
    /// rows = db.query_with_params(
    ///     "MATCH (n:Person {name: $name}) RETURN n.age",
    ///     {"name": "Alice"}
    /// )
    /// ```
    ///
    /// The values may be any Python primitive (`None`, `bool`, `int`, `float`,
    /// `str`), a `list`, or a nested `dict`.  Lists are especially useful with
    /// `UNWIND $list AS item INSERT …`.
    fn query_with_params(
        &mut self,
        py: Python<'_>,
        gql: &str,
        params: &pyo3::Bound<'_, PyDict>,
    ) -> PyResult<Vec<PyObject>> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;

        // Convert Python dict → HashMap<String, Value>.
        let rust_params: HashMap<String, Value> = params
            .iter()
            .filter_map(|(k, v)| {
                let key = k.str().ok()?.extract::<String>().ok()?;
                Some((key, py_to_value(py, &v.into_py(py))))
            })
            .collect();

        let (rows, _ops) = crate::query_capturing_with_params(gql, graph, &mut self.next_txn_id, rust_params)
            .map_err(db_err_to_py)?;

        let py_rows = rows
            .into_iter()
            .map(|row| {
                let d = PyDict::new_bound(py);
                for (k, v) in &row {
                    d.set_item(k, value_to_py(py, v)).unwrap();
                }
                d.into_py(py)
            })
            .collect();
        Ok(py_rows)
    }

    /// Execute a GQL query and return a **polars DataFrame**.
    ///
    /// Python signature: `query_df(gql: str) -> polars.DataFrame`
    ///
    /// Each column in the result becomes a Series in the DataFrame.
    /// Requires `polars` to be installed (`pip install polars`).
    ///
    /// Raises `RuntimeError` if polars is not installed, the database is
    /// closed, or the GQL is invalid.
    ///
    /// ```python
    /// df = db.query_df("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age")
    /// print(df)
    /// ```
    fn query_df(&mut self, py: Python<'_>, gql: &str) -> PyResult<PyObject> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let (rows, _ops) =
            crate::query_capturing(gql, graph, &mut self.next_txn_id)
                .map_err(db_err_to_py)?;
        let cols = rows_to_columnar(py, &rows)?;
        // Dynamically import polars — it is an optional runtime dependency.
        let pl = py.import_bound("polars").map_err(|_| {
            PyRuntimeError::new_err("polars is not installed. Run: pip install polars")
        })?;
        let df = pl.getattr("DataFrame")?.call1((cols,))?;
        Ok(df.into_py(py))
    }

    /// Execute a GQL query and return a **pandas DataFrame**.
    ///
    /// Python signature: `query_pandas(gql: str) -> pandas.DataFrame`
    ///
    /// Each column in the result becomes a Series in the DataFrame.
    /// Requires `pandas` to be installed (`pip install pandas`).
    ///
    /// Raises `RuntimeError` if pandas is not installed, the database is
    /// closed, or the GQL is invalid.
    ///
    /// ```python
    /// df = db.query_pandas("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age")
    /// print(df)
    /// ```
    fn query_pandas(&mut self, py: Python<'_>, gql: &str) -> PyResult<PyObject> {
        let graph = self.graph.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?;
        let (rows, _ops) =
            crate::query_capturing(gql, graph, &mut self.next_txn_id)
                .map_err(db_err_to_py)?;
        let cols = rows_to_columnar(py, &rows)?;
        // Dynamically import pandas — it is an optional runtime dependency.
        let pd = py.import_bound("pandas").map_err(|_| {
            PyRuntimeError::new_err("pandas is not installed. Run: pip install pandas")
        })?;
        let df = pd.getattr("DataFrame")?.call1((cols,))?;
        Ok(df.into_py(py))
    }

    /// Begin an explicit transaction.
    ///
    /// Python signature: `begin() -> None`
    ///
    /// After `begin()`, subsequent `query()` calls are buffered in memory.
    /// The transaction remains open until `commit()` or `rollback()` is called.
    /// Raises `RuntimeError` if already inside a transaction or if the database
    /// is closed.
    fn begin(&mut self) -> PyResult<()> {
        self.graph.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?
            .begin_transaction().map_err(db_err_to_py)
    }

    /// Commit the current transaction.
    ///
    /// Python signature: `commit() -> None`
    ///
    /// Applies all buffered operations to RocksDB as a single atomic write
    /// batch, then clears the transaction state.  Raises `RuntimeError` if no
    /// transaction is open or if the database is closed.
    fn commit(&mut self) -> PyResult<()> {
        self.graph.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?
            .commit_transaction().map_err(db_err_to_py)
    }

    /// Roll back the current transaction.
    ///
    /// Python signature: `rollback() -> None`
    ///
    /// Discards all buffered operations without writing to RocksDB (O(1) cost
    /// — the in-memory pending list is simply cleared).  Raises `RuntimeError`
    /// if no transaction is open or if the database is closed.
    fn rollback(&mut self) -> PyResult<()> {
        self.graph.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?
            .rollback_transaction().map_err(db_err_to_py)
    }

    // ── Fast clear ────────────────────────────────────────────────────────────

    /// Delete every node and edge in O(1) via RocksDB range tombstones.
    ///
    /// Python signature: `clear() -> None`
    ///
    /// Equivalent to running `MATCH (n) DETACH DELETE n` but constant-time
    /// regardless of graph size.  Index definitions (created with
    /// `CREATE INDEX`) are preserved.  Raises `RuntimeError` if the database
    /// is closed or if a transaction is currently active.
    fn clear(&mut self) -> PyResult<()> {
        self.graph.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))?
            .clear()
            .map_err(db_err_to_py)
    }

    // ── CSV bulk-load methods ─────────────────────────────────────────────────

    /// Load nodes from a CSV file into the graph.
    ///
    /// Python signature: `load_csv_nodes(path: str, label: str | None = None) -> dict[str, str]`
    ///
    /// Returns a mapping of user-supplied `:ID` column values to their
    /// assigned ULID strings.  Pass this dict to [`load_csv_edges`] as `id_map`
    /// so the edge loader can resolve `:START_ID` / `:END_ID` references.
    ///
    /// CSV format — header row must contain some of:
    /// - `:ID` — user-assigned ID (stored as `_csv_id` property).
    /// - `:LABEL` — per-row node label (overrides `label` parameter when non-empty).
    /// - Any other non-`:`-prefixed column → node property (type-inferred).
    ///
    /// Raises `RuntimeError` on malformed CSV or if the database is closed.
    #[pyo3(signature = (path, label=None))]
    fn load_csv_nodes(
        &mut self,
        path: &str,
        label: Option<&str>,
    ) -> PyResult<HashMap<String, String>> {
        use std::fs::File;
        use crate::csv_import::load_nodes_csv;

        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("database is closed"))?;

        let file = File::open(path)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        let result = load_nodes_csv(file, graph, label).map_err(db_err_to_py)?;

        // Persist id_map in graph for subsequent load_csv_edges(id_map=None).
        graph.csv_id_map = result.id_map.clone();

        // Return id_map as {user_id: ulid_string}.
        let out = crate::csv_import::id_map_to_strings(&result.id_map);
        Ok(out)
    }

    /// Load edges from a CSV file into the graph.
    ///
    /// Python signature:
    ///   `load_csv_edges(path: str, id_map: dict[str, str] | None = None, label: str | None = None) -> dict`
    ///
    /// `id_map` must map user `:ID` strings (from a prior `load_csv_nodes` call)
    /// to ULID strings.  If `None`, the id_map previously stored by
    /// `load_csv_nodes` is used automatically.
    ///
    /// Returns `{"inserted": int, "skipped": int}`.
    ///
    /// CSV format — header row must contain some of:
    /// - `:START_ID` — user ID of the source node.
    /// - `:END_ID` — user ID of the target node.
    /// - `:TYPE` — per-row edge label (overrides `label` parameter when non-empty).
    /// - Any other non-`:`-prefixed column → edge property (type-inferred).
    ///
    /// Raises `RuntimeError` on malformed CSV or if the database is closed.
    #[pyo3(signature = (path, id_map=None, label=None))]
    fn load_csv_edges(
        &mut self,
        py: Python<'_>,
        path: &str,
        id_map: Option<HashMap<String, String>>,
        label: Option<&str>,
    ) -> PyResult<PyObject> {
        use std::fs::File;
        use crate::csv_import::{load_edges_csv, id_map_from_strings};
        use pyo3::types::PyDict;

        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("database is closed"))?;

        let file = File::open(path)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

        // Resolve the id_map: use the passed one (decoded from ULID strings)
        // or fall back to the map stored by a preceding load_csv_nodes call.
        let resolved = if let Some(m) = id_map.as_ref() {
            id_map_from_strings(m)
        } else {
            graph.csv_id_map.clone()
        };

        let result = load_edges_csv(file, graph, &resolved, label).map_err(db_err_to_py)?;

        let d = PyDict::new_bound(py);
        d.set_item("inserted", result.inserted)?;
        d.set_item("skipped",  result.skipped)?;
        Ok(d.into_py(py))
    }

    /// Load nodes and edges from two CSV files in one call.
    ///
    /// Python signature:
    ///   `load_csv(nodes_path: str, edges_path: str, node_label: str | None = None,
    ///             edge_label: str | None = None) -> dict`
    ///
    /// Equivalent to calling `load_csv_nodes(nodes_path, label=node_label)` and
    /// then `load_csv_edges(edges_path, label=edge_label)` with the returned
    /// id_map.  Returns `{"nodes_inserted": N, "edges_inserted": M, "skipped": K}`.
    #[pyo3(signature = (nodes_path, edges_path, node_label=None, edge_label=None))]
    fn load_csv(
        &mut self,
        py: Python<'_>,
        nodes_path: &str,
        edges_path: &str,
        node_label: Option<&str>,
        edge_label: Option<&str>,
    ) -> PyResult<PyObject> {
        use std::fs::File;
        use crate::csv_import::{load_nodes_csv, load_edges_csv};
        use pyo3::types::PyDict;

        let graph = self.graph.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("database is closed"))?;

        let nf = File::open(nodes_path)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        let nr = load_nodes_csv(nf, graph, node_label).map_err(db_err_to_py)?;
        let id_map = nr.id_map.clone();
        // Also update the in-graph map for completeness.
        graph.csv_id_map = id_map.clone();

        let ef = File::open(edges_path)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        let er = load_edges_csv(ef, graph, &id_map, edge_label).map_err(db_err_to_py)?;

        let d = PyDict::new_bound(py);
        d.set_item("nodes_inserted", nr.inserted)?;
        d.set_item("edges_inserted", er.inserted)?;
        d.set_item("skipped",        er.skipped)?;
        Ok(d.into_py(py))
    }

    /// Flush all pending data to disk and release all resources.
    ///
    /// Python signature: `close() -> None`
    ///
    /// Steps performed:
    /// 1. If a transaction is open, roll it back silently.
    /// 2. Write a RocksDB checkpoint (compacts WAL into the SST files).
    /// 3. Drop the [`Graph`] handle, which releases the RocksDB exclusive lock.
    ///
    /// After `close()` returns, the same path can be opened again in the same
    /// process.  Calling `close()` on an already-closed instance is a no-op.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(mut graph) = self.graph.take() {
            // Implicit rollback: discard any uncommitted buffered ops.
            if graph.is_in_transaction() {
                let _ = graph.rollback_transaction();
            }
            if let Some(storage) = self.storage.take() {
                py.allow_threads(|| storage.checkpoint(&graph))
                    .map_err(db_err_to_py)?;
            }
            // `graph` drops here → RocksDB handle released → lock freed.
        }
        Ok(())
    }

    /// Support `with minigdb.open(...) as db:` syntax.
    ///
    /// Returns `self` so the bound variable refers to the same object.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Called automatically at the end of a `with` block.
    ///
    /// Calls `close()` regardless of whether an exception occurred.
    /// Returns `False` so that any active exception is **not** suppressed.
    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: PyObject,
        _exc_val: PyObject,
        _exc_tb: PyObject,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false) // do not suppress exceptions
    }
}

// ── Module-level open() ───────────────────────────────────────────────────────

/// Resolve the platform-appropriate data directory for a named graph.
///
/// Returns `<data_dir>/minigdb/graphs/<name>` where `data_dir` is:
/// - Linux/macOS: `~/.local/share` / `~/Library/Application Support`
/// - Windows:     `%APPDATA%`
///
/// Validates the name using the same rules as the REPL:
/// non-empty, ≤ 64 chars, only `[A-Za-z0-9_-]`.
fn graph_dir(name: &str) -> PyResult<PathBuf> {
    // Validate name: non-empty, alphanumeric + _ + -, max 64 chars.
    if name.is_empty() || name.len() > 64 {
        return Err(PyRuntimeError::new_err(
            "Graph name must be 1–64 characters",
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return Err(PyRuntimeError::new_err(
            "Graph name must contain only alphanumeric characters, '_', or '-'",
        ));
    }

    let base = dirs::data_dir().ok_or_else(|| {
        PyOSError::new_err("Cannot determine platform data directory")
    })?;
    Ok(base.join("minigdb").join("graphs").join(name))
}

/// Open (or create) a named graph database stored in the platform data directory.
///
/// Python signature: `open(name: str) -> MiniGdb`
///
/// The on-disk location follows platform conventions:
/// - Linux:   `~/.local/share/minigdb/graphs/<name>/`
/// - macOS:   `~/Library/Application Support/minigdb/graphs/<name>/`
/// - Windows: `%APPDATA%\\minigdb\\graphs\\<name>\\`
///
/// The directory is created if it does not yet exist.  For an explicit path
/// (e.g. in tests) use [`open_at`] instead.
///
/// Raises `OSError` on I/O failure and `RuntimeError` on invalid names or
/// storage engine errors.
#[pyfunction]
fn open(py: Python<'_>, name: &str) -> PyResult<MiniGdb> {
    let dir = graph_dir(name)?;
    std::fs::create_dir_all(&dir)
        .map_err(|e| PyOSError::new_err(e.to_string()))?;

    let (storage, graph) = py.allow_threads(|| StorageManager::open(&dir))
        .map_err(db_err_to_py)?;

    Ok(MiniGdb { graph: Some(graph), storage: Some(storage), next_txn_id: 0 })
}

/// Open a graph database at an explicit filesystem path.
///
/// Python signature: `open_at(path: str) -> MiniGdb`
///
/// Intended for unit tests and scripts that need reproducible paths rather than
/// the platform data directory.  The directory is created if it does not exist.
///
/// Raises `OSError` on I/O failure and `RuntimeError` on storage engine errors.
#[pyfunction]
fn open_at(py: Python<'_>, path: &str) -> PyResult<MiniGdb> {
    let dir = std::path::Path::new(path);
    std::fs::create_dir_all(dir)
        .map_err(|e| PyOSError::new_err(e.to_string()))?;

    let (storage, graph) = py.allow_threads(|| StorageManager::open(dir))
        .map_err(db_err_to_py)?;

    Ok(MiniGdb { graph: Some(graph), storage: Some(storage), next_txn_id: 0 })
}

// ── Network client ────────────────────────────────────────────────────────────

/// Convert a `serde_json::Value` (received from the server) to a Python object.
///
/// JSON numbers are mapped to `int` when they fit in `i64`, otherwise `float`.
/// JSON objects become Python `dict`; JSON arrays become `list`.
fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyObject {
    match v {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.into_py(py),
        serde_json::Value::Number(n) => {
            // Prefer integer representation to avoid floating-point noise.
            if let Some(i) = n.as_i64() {
                i.into_py(py)
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_py(py)
            }
        }
        serde_json::Value::String(s) => s.into_py(py),
        serde_json::Value::Array(a) => a
            .iter()
            .map(|x| json_to_py(py, x))
            .collect::<Vec<_>>()
            .into_py(py),
        serde_json::Value::Object(o) => {
            let d = PyDict::new_bound(py);
            for (k, v) in o {
                d.set_item(k, json_to_py(py, v)).unwrap();
            }
            d.into_py(py)
        }
    }
}

/// Build a columnar `{col: [values…]}` Python dict from JSON rows for
/// DataFrame construction.
///
/// Mirrors [`rows_to_columnar`] but operates on `serde_json::Value` rows
/// received from the network rather than on in-process [`Value`] rows.
/// Missing keys in sparse rows become `None`.
fn json_rows_to_columnar(
    py: Python<'_>,
    rows: &[std::collections::HashMap<String, serde_json::Value>],
) -> PyResult<PyObject> {
    let mut col_order: Vec<String> = Vec::new();
    let mut col_seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for row in rows {
        for k in row.keys() {
            if col_seen.insert(k.as_str()) {
                col_order.push(k.clone());
            }
        }
    }
    let d = PyDict::new_bound(py);
    for col in &col_order {
        let vals: Vec<PyObject> = rows
            .iter()
            .map(|row| {
                row.get(col.as_str())
                    .map_or_else(|| py.None(), |v| json_to_py(py, v))
            })
            .collect();
        d.set_item(col, vals)?;
    }
    Ok(d.into_py(py))
}

/// A handle to a remote minigdb server.  Created by [`connect`].
///
/// The connection is a synchronous TCP stream using the newline-delimited JSON
/// protocol (Protocol v2).  The same `query()`, `query_df()`, and
/// `query_pandas()` interface as [`MiniGdb`] (local) is provided, so
/// application code can swap the two without changes.
///
/// The `reader` and `writer` are two ends of the same `TcpStream`; the writer
/// is a `try_clone()` of the reader's underlying stream.  Keeping them
/// separate avoids the need for locking on the common case of a single-threaded
/// Python caller.
#[pyclass]
pub struct MiniGdbClient {
    /// Buffered reader wrapping the TCP stream — used to read response lines.
    reader: std::io::BufReader<std::net::TcpStream>,
    /// Cloned TCP stream handle used for writing request lines.
    writer: std::net::TcpStream,
    /// Monotonically increasing request ID, echoed in server responses for
    /// correlation (currently unused by the client but required by the protocol).
    next_id: u64,
}

impl MiniGdbClient {
    /// Serialize `gql` as a JSON request, send it to the server, read exactly
    /// one response line, and return the parsed rows.
    ///
    /// The protocol is newline-delimited JSON:
    /// - Request:  `{"id":<u64>,"query":"<gql>"}\n`
    /// - Response: `{"id":<u64>,"rows":[{...}],"elapsed_ms":<f64>}\n`
    ///             or `{"id":<u64>,"error":"<message>"}\n`
    ///
    /// Server-side errors are surfaced as `RuntimeError`; network I/O errors
    /// as `OSError`.
    fn send_query(
        &mut self,
        gql: &str,
    ) -> PyResult<Vec<std::collections::HashMap<String, serde_json::Value>>> {
        use std::io::{BufRead, Write};

        let id = self.next_id;
        self.next_id += 1;

        // Serialize and send the request line.
        let req = serde_json::json!({"id": id, "query": gql});
        let mut line = serde_json::to_string(&req).unwrap();
        line.push('\n');
        self.writer
            .write_all(line.as_bytes())
            .map_err(|e| PyOSError::new_err(e.to_string()))?;

        // Read one response line.
        let mut resp_line = String::new();
        self.reader
            .read_line(&mut resp_line)
            .map_err(|e| PyOSError::new_err(e.to_string()))?;

        // Parse the JSON response.
        let resp: serde_json::Value = serde_json::from_str(resp_line.trim())
            .map_err(|e| PyRuntimeError::new_err(format!("server sent invalid JSON: {e}")))?;

        // Surface server-side errors as Python RuntimeError.
        if let Some(err) = resp.get("error").and_then(|e| e.as_str()) {
            return Err(PyRuntimeError::new_err(err.to_string()));
        }

        // Extract the rows array; treat a missing/non-array field as empty.
        let rows = resp
            .get("rows")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|row| {
                        row.as_object().map(|obj| {
                            obj.iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect::<std::collections::HashMap<_, _>>()
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(rows)
    }
}

#[pymethods]
impl MiniGdbClient {
    /// Execute a GQL statement against the remote server and return a list of result dicts.
    ///
    /// Python signature: `query(gql: str) -> list[dict]`
    ///
    /// Raises `OSError` on network failure and `RuntimeError` on server-side errors.
    fn query(&mut self, py: Python<'_>, gql: &str) -> PyResult<Vec<PyObject>> {
        let rows = self.send_query(gql)?;
        let py_rows = rows
            .iter()
            .map(|row| {
                let d = PyDict::new_bound(py);
                for (k, v) in row {
                    d.set_item(k, json_to_py(py, v)).unwrap();
                }
                d.into_py(py)
            })
            .collect();
        Ok(py_rows)
    }

    /// Execute a GQL query against the remote server and return a **polars DataFrame**.
    ///
    /// Python signature: `query_df(gql: str) -> polars.DataFrame`
    ///
    /// Requires `polars` to be installed (`pip install polars`).
    fn query_df(&mut self, py: Python<'_>, gql: &str) -> PyResult<PyObject> {
        let rows = self.send_query(gql)?;
        let cols = json_rows_to_columnar(py, &rows)?;
        let pl = py.import_bound("polars").map_err(|_| {
            PyRuntimeError::new_err("polars is not installed. Run: pip install polars")
        })?;
        let df = pl.getattr("DataFrame")?.call1((cols,))?;
        Ok(df.into_py(py))
    }

    /// Execute a GQL query against the remote server and return a **pandas DataFrame**.
    ///
    /// Python signature: `query_pandas(gql: str) -> pandas.DataFrame`
    ///
    /// Requires `pandas` to be installed (`pip install pandas`).
    fn query_pandas(&mut self, py: Python<'_>, gql: &str) -> PyResult<PyObject> {
        let rows = self.send_query(gql)?;
        let cols = json_rows_to_columnar(py, &rows)?;
        let pd = py.import_bound("pandas").map_err(|_| {
            PyRuntimeError::new_err("pandas is not installed. Run: pip install pandas")
        })?;
        let df = pd.getattr("DataFrame")?.call1((cols,))?;
        Ok(df.into_py(py))
    }

    /// Send `BEGIN` to start an explicit transaction on the server.
    ///
    /// Python signature: `begin() -> None`
    ///
    /// The server acquires a per-graph mutex and holds it until `commit()`,
    /// `rollback()`, or the connection drops.  This serializes concurrent
    /// writers — do not hold transactions open longer than necessary.
    fn begin(&mut self) -> PyResult<()> {
        self.send_query("BEGIN").map(|_| ())
    }

    /// Send `COMMIT` to commit the current transaction on the server.
    ///
    /// Python signature: `commit() -> None`
    fn commit(&mut self) -> PyResult<()> {
        self.send_query("COMMIT").map(|_| ())
    }

    /// Send `ROLLBACK` to discard the current transaction on the server.
    ///
    /// Python signature: `rollback() -> None`
    fn rollback(&mut self) -> PyResult<()> {
        self.send_query("ROLLBACK").map(|_| ())
    }

    /// Close the TCP connection by issuing a `Shutdown::Both` on the socket.
    ///
    /// Python signature: `close() -> None`
    ///
    /// Raises `OSError` if the shutdown syscall fails (e.g. already closed).
    fn close(&mut self) -> PyResult<()> {
        self.writer
            .shutdown(std::net::Shutdown::Both)
            .map_err(|e| PyOSError::new_err(e.to_string()))
    }

    /// Support `with minigdb.connect(...) as db:` syntax.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Close the connection at the end of a `with` block.
    ///
    /// Errors from `close()` are silently swallowed here (the socket may
    /// already be half-closed).  Returns `False` so active exceptions are not
    /// suppressed.
    fn __exit__(
        &mut self,
        _exc_type: PyObject,
        _exc_val: PyObject,
        _exc_tb: PyObject,
    ) -> PyResult<bool> {
        let _ = self.close();
        Ok(false)
    }
}

// ── Module-level connect() ────────────────────────────────────────────────────

/// Connect to a running minigdb server and return a [`MiniGdbClient`].
///
/// Python signature: `connect(addr: str) -> MiniGdbClient`
///
/// `addr` is a `"host:port"` string, e.g. `"127.0.0.1:7474"` or
/// `"localhost:7474"`.
///
/// The TCP connection is established synchronously.  The server sends a `hello`
/// message on connect; the client must authenticate if the server requires it
/// (see Protocol v2 documentation).  Authentication is not yet exposed via this
/// Python API — start the server with `--no-auth` for unauthenticated access.
///
/// Raises `OSError` if the TCP connection cannot be established.
///
/// ```python
/// with minigdb.connect("localhost:7474") as db:
///     rows = db.query("MATCH (n:Person) RETURN n.name")
/// ```
#[pyfunction]
fn connect(addr: &str) -> PyResult<MiniGdbClient> {
    use std::io::BufRead;

    let stream = std::net::TcpStream::connect(addr)
        .map_err(|e| PyOSError::new_err(format!("cannot connect to {addr}: {e}")))?;
    // Clone the stream so we have separate handles for reading and writing,
    // avoiding the need for a Mutex on the common single-threaded Python path.
    let writer = stream
        .try_clone()
        .map_err(|e| PyOSError::new_err(e.to_string()))?;
    let mut reader = std::io::BufReader::new(stream);
    // Protocol v2: the server always sends a hello frame immediately on
    // connection.  Consume it before the caller issues any queries, otherwise
    // every response would be shifted by one (the first query reads the hello,
    // the second reads the first query's response, etc.).
    let mut hello = String::new();
    reader
        .read_line(&mut hello)
        .map_err(|e| PyOSError::new_err(format!("reading hello from {addr}: {e}")))?;
    Ok(MiniGdbClient { reader, writer, next_id: 0 })
}

// ── PyModule ──────────────────────────────────────────────────────────────────

/// minigdb — embedded property-graph database with a GQL query language.
///
/// # Quick start
///
/// ```python
/// import minigdb
///
/// # Open (or create) a named graph in the platform data directory.
/// with minigdb.open("mygraph") as db:
///     db.query('INSERT (:Person {name: "Alice", age: 30})')
///     rows = db.query("MATCH (n:Person) RETURN n.name, n.age")
///     # rows → [{"n.name": "Alice", "n.age": 30}]
///
/// # DataFrame output (polars or pandas must be installed separately).
/// with minigdb.open("mygraph") as db:
///     df = db.query_df("MATCH (n:Person) RETURN n.name, n.age")
///
/// # Connect to a running server.
/// with minigdb.connect("localhost:7474") as db:
///     rows = db.query("MATCH (n:Person) RETURN n.name")
/// ```
///
/// # Exports
///
/// | Name | Kind | Description |
/// |------|------|-------------|
/// | `open(name)` | function | Open/create a named local graph (platform data dir) |
/// | `open_at(path)` | function | Open/create a graph at an explicit filesystem path |
/// | `connect(addr)` | function | Connect to a running minigdb TCP server |
/// | `MiniGdb` | class | Handle for a local graph opened in-process |
/// | `MiniGdbClient` | class | Handle for a remote graph accessed over TCP |
///
/// # Error handling
///
/// | Condition | Python exception |
/// |-----------|-----------------|
/// | Parse or query error | `RuntimeError` |
/// | I/O / storage error | `OSError` |
#[pymodule]
fn minigdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MiniGdb>()?;
    m.add_class::<MiniGdbClient>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(open_at, m)?)?;
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    Ok(())
}
