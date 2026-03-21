# Python API

[← Back to README](../README.md) · [Query Language](query-language.md) · [Algorithms](algorithms.md) · [Server](server.md)

minigdb can be used as an embedded Python library. Pre-built wheels are available for Linux, macOS, and Windows — no Rust toolchain required.

---

## Installation

```bash
pip install minigdb
```

With optional DataFrame support:

```bash
pip install minigdb[polars]     # polars DataFrame output
pip install minigdb[pandas]     # pandas DataFrame output
pip install minigdb[dataframes] # both
```

**Supported platforms (pre-built wheels):**

| Platform | Architecture |
|----------|-------------|
| Linux (glibc 2.28+) | x86_64, aarch64 |
| macOS 11+ | Universal (Intel + Apple Silicon) |
| Windows | x86_64 |

For other platforms (Alpine Linux, FreeBSD, etc.), install from source — requires a Rust toolchain, cmake, and a C++ compiler:

```bash
pip install minigdb --no-binary minigdb
```

**Building from source for development** (see [CONTRIBUTING.md](../CONTRIBUTING.md)):

```bash
pip install maturin
maturin develop --features python   # builds and installs into the active venv
```

---

## Local usage

```python
import minigdb

# Open a named graph (stored in the platform data directory, same as the REPL)
db = minigdb.open("myproject")

rows = db.query('MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age')
# rows: [{"n.name": "Alice", "n.age": 30}, ...]

db.close()   # writes a checkpoint

# Context manager — close/checkpoint called automatically
with minigdb.open("myproject") as db:
    db.query('INSERT (:Person {name: "Alice", age: 30})')
    rows = db.query('MATCH (n:Person) RETURN n.name')
```

---

## Explicit transactions

```python
with minigdb.open("myproject") as db:
    db.begin()
    db.query('INSERT (:Person {name: "Dave", age: 40})')
    db.query('INSERT (:Person {name: "Eve",  age: 22})')
    db.commit()      # single atomic WAL frame
    # db.rollback()  # discard all changes since begin()
```

Each `query()` outside a transaction is auto-committed immediately. Inside `begin()`…`commit()`, operations are buffered and written as one frame on `commit()`.

---

## DataFrame output

```python
df = db.query_df("MATCH (n:Person) RETURN n.name, n.age")     # polars DataFrame
df = db.query_pandas("MATCH (n:Person) RETURN n.name, n.age") # pandas DataFrame
```

polars and pandas are optional runtime dependencies — install as needed:

```bash
pip install minigdb[polars]
pip install minigdb[pandas]
```

---

## Network client

```python
import minigdb

# Connect to a running server (start the server with --no-auth for open access)
db = minigdb.connect("localhost:7474")

rows = db.query("MATCH (n:Person) RETURN n.name")
df   = db.query_df("MATCH (n:Person) RETURN n.name, n.age")   # polars DataFrame
```

The same `query()`, `query_df()`, `query_pandas()`, `begin()`, `commit()`, and `rollback()` methods are available on the network client.

> **Note:** Authentication credentials are not yet accepted by the Python `connect()` call. Start the server with `--no-auth` for unauthenticated access, or use the raw TCP protocol if you need per-user auth from Python.

---

## Parameterized queries

`query_with_params(gql, params)` substitutes `$name` placeholders with values from the `params` dict. Any Python value that maps to a GQL type is accepted, including lists and dicts.

```python
# Scalar placeholder
rows = db.query_with_params(
    "MATCH (n:Person) WHERE n.name = $name RETURN n.age",
    {"name": "Alice"}
)

# UNWIND $list INSERT — insert a batch in one round-trip
db.query_with_params(
    "UNWIND $tags AS t INSERT (:Tag {name: t})",
    {"tags": ["python", "rust", "graph"]}
)

# UNWIND with dict items — access fields via dot notation
db.query_with_params(
    "UNWIND $people AS p INSERT (:Person {name: p.name, age: p.age})",
    {"people": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
)

# Numeric param as LIMIT
rows = db.query_with_params(
    "MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT $k",
    {"k": 5}
)
```

---

## CSV bulk import

```python
# Load nodes — returns a dict mapping :ID values to assigned ULIDs
id_map = db.load_csv_nodes("nodes.csv", label="Person")

# Load edges using the id_map from the node load
result = db.load_csv_edges("edges.csv", id_map=id_map, label="KNOWS")
print(result)  # {"inserted": 42, "skipped": 0}

# Load both in one call
result = db.load_csv("nodes.csv", "edges.csv", node_label="Person", edge_label="KNOWS")
print(result)  # {"nodes_inserted": 10, "edges_inserted": 42, "skipped": 0}
```

**Node CSV format** — header row:
- `:ID` — user-assigned ID (stored as `_csv_id` property; used to resolve edge endpoints)
- `:LABEL` — per-row node label (overrides the `label` parameter when non-empty)
- Any other column → node property (type-inferred)

**Edge CSV format** — header row:
- `:START_ID` — user ID of the source node
- `:END_ID` — user ID of the target node
- `:TYPE` — per-row edge label (overrides the `label` parameter when non-empty)
- Any other column → edge property (type-inferred)

---

## Value mapping

| GQL type | Python type |
|----------|-------------|
| `null` | `None` |
| `bool` | `bool` |
| `integer` | `int` |
| `float` | `float` |
| `string` | `str` |
| `map` (dict, used for `$param` list items) | `dict` |
| list (e.g. from `collect()`) | `list` |

---

## Error handling

| Condition | Python exception |
|-----------|-----------------|
| Parse or query error | `RuntimeError` |
| I/O / storage error | `OSError` |

---

## Data location

Named graphs opened via Python share the same storage layout as the REPL:

| Platform | Path |
|----------|------|
| Linux    | `~/.local/share/minigdb/graphs/<name>/` |
| macOS    | `~/Library/Application Support/minigdb/graphs/<name>/` |
| Windows  | `%AppData%\minigdb\graphs\<name>\` |
