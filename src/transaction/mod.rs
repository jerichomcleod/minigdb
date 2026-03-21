//! Explicit write transaction — stages [`Operation`]s in memory and applies
//! them atomically on commit (or discards them on rollback).
//!
//! In normal (auto-commit) mode the executor bypasses this struct and writes
//! directly via `Graph::apply_*` methods.  The `Transaction` type is used by
//! the REPL and server when the user issues explicit `BEGIN`/`COMMIT`/`ROLLBACK`.

pub(crate) mod operation;

pub use operation::{Operation, PropertyTarget};

use crate::types::DbError;

/// An in-progress write transaction.
///
/// Operations are accumulated in `ops` and applied atomically to the graph on
/// [`commit`](Self::take_ops).  Because nothing is written to RocksDB until
/// commit time, rollback is a zero-cost drop of the in-memory buffer.
pub struct Transaction {
    /// Buffered mutations, applied in order on commit.
    pub(crate) ops: Vec<Operation>,
    /// Opaque transaction identifier, used for WAL framing.
    pub(crate) txn_id: u64,
    /// Set to `true` after `take_ops` or `rollback` to prevent double-commit.
    pub(crate) finished: bool,
}

impl Transaction {
    /// Stage a mutation.
    ///
    /// # Errors
    ///
    /// Returns [`DbError::TransactionFinished`] if the transaction has already
    /// been committed or rolled back.
    pub fn stage(&mut self, op: Operation) -> Result<(), DbError> {
        if self.finished {
            return Err(DbError::TransactionFinished);
        }
        self.ops.push(op);
        Ok(())
    }

    /// Discard all staged operations without applying them.
    pub fn rollback(&mut self) {
        self.ops.clear();
        self.finished = true;
    }

    /// Returns the transaction's opaque identifier.
    pub fn txn_id(&self) -> u64 {
        self.txn_id
    }
}
