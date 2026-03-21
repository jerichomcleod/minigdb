//! # minigdb
//!
//! A property-graph database engine backed by RocksDB.
//!
//! ## Module layout
//!
//! | Module | Role |
//! |---|---|
//! | [`types`] | Core data types: [`Value`], [`Node`], [`Edge`], IDs, errors |
//! | [`graph`] | [`Graph`] struct — all read/write graph operations |
//! | [`storage`] | [`StorageManager`] — persistent open/close/checkpoint |
//! | [`transaction`] | [`Operation`] enum used for WAL replay and buffered transactions |
//! | [`query`] | GQL parser, AST, and tree-walking executor |
//! | [`algorithms`] | Graph algorithms invokable via `CALL` GQL syntax |
//! | [`server`] | Async TCP + HTTP server with auth (feature `server`) |
//! | [`python`] | PyO3 Python bindings (feature `python`) |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use std::path::Path;
//! use minigdb::{query, StorageManager};
//!
//! let mut txn_id = 0u64;
//! let (_sm, mut graph) = StorageManager::open(Path::new("my_graph")).unwrap();
//! query("INSERT (:Person {name: \"Alice\"})", &mut graph, &mut txn_id).unwrap();
//! let rows = query("MATCH (n:Person) RETURN n.name", &mut graph, &mut txn_id).unwrap();
//! println!("{:?}", rows);
//! ```

pub mod algorithms;
pub mod csv_import;
pub mod graph;
pub mod query;
pub mod storage;
pub mod transaction;
pub mod types;

// Top-level re-exports covering the most common public API surface.
pub use graph::Graph;
pub use query::{query, query_capturing, query_capturing_with_params, Row};
pub use storage::StorageManager;
pub use types::{DbError, Edge, EdgeId, Node, NodeId, Properties, Value};

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "python")]
pub mod python;
