# minigdb

[![Crates.io](https://img.shields.io/crates/v/minigdb)](https://crates.io/crates/minigdb)
[![docs.rs](https://img.shields.io/docsrs/minigdb)](https://docs.rs/minigdb)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](LICENSE)

An embedded property-graph database built in Rust on RocksDB. Property graph model (nodes and edges with labels and key-value properties), ACID-compliant RocksDB-backed storage, and a GQL (ISO/IEC 39075) query language subset — usable as a REPL, a network server, or an embedded Rust/Python library.

**Feature highlights:**
- GQL queries: `MATCH`, `INSERT`, `SET`, `DELETE`, variable-length paths, `OPTIONAL MATCH`, `UNION`, aggregates, `WITH`, `UNWIND`
- 15 built-in graph algorithms: PageRank, shortest path, community detection (Louvain/Leiden), centrality measures, max-flow, and more
- Property indexes, uniqueness and type constraints, parameterized queries, CSV bulk import
- Multi-graph support, explicit transactions
- TCP server with authentication, per-connection transaction isolation, and an embedded web GUI
- Python bindings via PyO3 with polars/pandas DataFrame output

---

## Quick Start

Requires Rust (install via [rustup](https://rustup.rs)).

```bash
# Build and launch the interactive REPL
cargo run

# Or build first, then run the binary directly
cargo build
./target/debug/minigdb

# Run the test suite
cargo test
```

Data is stored in the platform user-data directory:

| Platform | Path |
|----------|------|
| Linux    | `~/.local/share/minigdb/` |
| macOS    | `~/Library/Application Support/minigdb/` |
| Windows  | `%AppData%\minigdb\` |

The REPL prints the active data directory on startup. Each named graph gets its own `graphs/<name>/` subfolder.

---

## REPL

The REPL auto-detects complete statements, so semicolons are optional. Multi-line input is supported; the prompt shows `->` until the statement is complete (or `txn->` inside a transaction). Input history is saved to `.minigdb_history` in the current directory.

### Meta-commands

| Command | Effect |
|---------|--------|
| `:graphs` | List all graphs, marking the active one |
| `:create <name>` | Create a new named graph and switch to it |
| `:use <name>` | Checkpoint the current graph and switch to an existing one |
| `:drop <name>` | Permanently delete a named graph |
| `:checkpoint` | Force a snapshot and truncate the WAL immediately |
| `:quit` or `:q` | Exit (writes a final checkpoint) |
| `Ctrl-C` | Cancel the current input line |
| `Ctrl-D` | Exit |

Graph management commands are not allowed inside a `BEGIN` transaction.

---

## Example queries

```
INSERT (:Person {name: "Alice", age: 30})
INSERT (:Person {name: "Bob",   age: 25})
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
INSERT (a)-[:KNOWS]->(b)

MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name
CALL pageRank() YIELD node, score RETURN node, score ORDER BY score DESC
```

---

## Documentation

| Topic | Description |
|-------|-------------|
| [Query Language](docs/query-language.md) | Full GQL reference: INSERT, MATCH, WHERE, aggregates, path quantifiers, indexes, constraints, transactions |
| [Graph Algorithms](docs/algorithms.md) | 15 algorithms invoked via `CALL … YIELD`: PageRank, shortest path, centrality, community detection, max-flow |
| [Network Server](docs/server.md) | TCP server, authentication, web GUI, wire protocol, HTTP API |
| [Python API](docs/python.md) | Installation, local/remote usage, DataFrame output, parameterized queries, CSV import |

---

## Storage

Each named graph is stored in its own `graphs/<name>/` subdirectory, backed by [RocksDB](https://rocksdb.org) using eight column families:

| Column Family | Contents |
|---------------|----------|
| `nodes` | Node records (ID → bincode-serialized Node) |
| `edges` | Edge records (ID → bincode-serialized Edge) |
| `adj_out` | Outgoing adjacency index |
| `adj_in` | Incoming adjacency index |
| `label_idx` | Label → node ID set |
| `edge_label_idx` | Edge label → edge ID set |
| `prop_idx` | Property index (label + key + encoded value → node ID set) |
| `meta` | Graph metadata (index definitions, constraint definitions) |

RocksDB provides atomic writes via `WriteBatch`, crash recovery via SST compaction, and supports graphs that exceed available RAM through its internal block cache.

---

## Architecture

```
src/
├── main.rs           # REPL (rustyline, graph management commands, server entry points)
├── lib.rs            # Public API and re-exports
├── gql.pest          # PEG grammar (pest, read at compile time)
├── python.rs         # Python bindings (PyO3, feature = "python")
├── csv_import.rs     # CSV bulk import
├── types/            # Value, NodeId (ULID), EdgeId (ULID), Node, Edge, DbError
├── graph/            # Graph struct + RocksDB-backed mutation helpers (ops.rs, constraints.rs)
├── storage/          # RocksStore (8 CFs), StorageManager, apply_ops replay helper
├── transaction/      # Operation enum (WAL replay units)
├── algorithms/       # 15 graph algorithm modules dispatched via CALL
├── query/            # AST, pest parser, tree-walking executor
└── server/
    ├── mod.rs        # Async TCP server, auth handshake, per-connection state
    ├── auth.rs       # ServerConfig, UserEntry, SHA-256 password hashing
    ├── registry.rs   # GraphRegistry — lazy-open, Arc<Mutex> per graph
    ├── protocol.rs   # Request/Response/ClientMessage/ServerMessage types
    └── http.rs       # Axum HTTP server + embedded GUI (feature = "gui")
```

### Cargo features

| Feature | Included in default | Description |
|---------|---------------------|-------------|
| `repl` | yes | Interactive REPL (rustyline, readline history) |
| `server` | yes | TCP server (tokio, SHA-256 auth) |
| `gui` | yes | Embedded HTTP GUI (axum); requires `server` |
| `python` | no | PyO3 Python bindings; build with `maturin develop --features python` |

---

## License

minigdb is distributed under the [GNU Lesser General Public License v3.0](LICENSE).
