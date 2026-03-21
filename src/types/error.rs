//! [`DbError`] — the unified error type for all database operations.

use thiserror::Error;

use super::{EdgeId, NodeId};

/// All errors that can arise from graph operations, queries, or storage I/O.
///
/// Uses [`thiserror`] so every variant automatically implements
/// `std::error::Error` with a human-readable `Display` message.
#[derive(Debug, Error)]
pub enum DbError {
    /// A [`NodeId`] referenced in a query or mutation does not exist.
    #[error("Node {0} not found")]
    NodeNotFound(NodeId),

    /// An [`EdgeId`] referenced in a query or mutation does not exist.
    #[error("Edge {0} not found")]
    EdgeNotFound(EdgeId),

    /// Attempt to delete a node that still has incident edges.
    /// Use `DETACH DELETE` to remove the node and its edges together.
    #[error("Node {0} has existing edges; use DETACH DELETE to remove with edges")]
    NodeHasEdges(NodeId),

    /// An OS-level I/O error from RocksDB or the file system.
    /// Automatically constructed from [`std::io::Error`] via `#[from]`.
    #[error("Storage error: {0}")]
    Storage(#[from] std::io::Error),

    /// `bincode` serialization or deserialization failure.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// A WAL frame failed its CRC check — the file is corrupted.
    #[error("WAL frame corrupted (CRC mismatch or bad magic)")]
    CorruptFrame,

    /// Attempted to use a transaction that has already been committed or
    /// rolled back.
    #[error("Transaction already committed or rolled back")]
    TransactionFinished,

    /// The GQL parser rejected the input string.
    #[error("Parse error: {0}")]
    Parse(String),

    /// The query executor encountered a semantic or runtime error
    /// (e.g. type mismatch, undefined variable, unsupported operation).
    #[error("Query error: {0}")]
    Query(String),

    /// A raw RocksDB error not mapped to a more specific variant.
    #[error("RocksDB error: {0}")]
    RocksDb(String),

    /// A constraint declared with `CREATE CONSTRAINT` was violated at write time.
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
}
