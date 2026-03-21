//! Persistence layer — [`StorageManager`] wraps a RocksDB-backed [`Graph`]
//! and provides open/checkpoint/commit helpers.
//!
//! ## Design note
//!
//! In Phase R2+ all durability is delegated to RocksDB.  The original WAL and
//! chunk-file infrastructure was removed in Phase R4.  [`StorageManager`] is
//! now a thin wrapper that owns the graph directory path and exposes a
//! `checkpoint` method (flush memtables to SST).  `commit_wal_only` is kept
//! for API compatibility with the REPL and Python bindings but is a no-op.

pub(crate) mod rocks_store;

use std::path::{Path, PathBuf};

use crate::graph::Graph;
use crate::transaction::operation::Operation;
use crate::types::DbError;

/// Manages persistence for a single named graph directory.
///
/// The heavy lifting (durability, crash recovery, column families) is handled
/// by RocksDB inside the [`Graph`] that is returned alongside this struct from
/// [`StorageManager::open`].  `StorageManager` provides the open/checkpoint API
/// used by the REPL, server, and Python bindings.
pub struct StorageManager {
    /// Retained so callers can know which directory backs this graph.
    _dir: PathBuf,
}

impl StorageManager {
    /// Open (or create) a persistent graph at `dir`.
    ///
    /// Returns a `(StorageManager, Graph)` pair.  The `Graph` holds the live
    /// RocksDB handle; the `StorageManager` holds the directory path and
    /// provides persistence helpers.
    pub fn open(dir: &Path) -> Result<(Self, Graph), DbError> {
        let graph = Graph::open(dir)?;
        Ok((Self { _dir: dir.to_path_buf() }, graph))
    }

    /// Flush RocksDB memtables to SST files so recent writes survive a crash
    /// without relying solely on the RocksDB WAL.
    pub fn checkpoint(&self, graph: &Graph) -> Result<(), DbError> {
        graph.store.flush()
    }

    /// No-op in Phase R2+: writes go directly to RocksDB on each mutation.
    ///
    /// Kept for API compatibility with the REPL and Python bindings, which
    /// call this after every auto-commit statement.
    pub fn commit_wal_only(
        &self,
        _txn_id: u64,
        _ops: &[Operation],
        _graph: &Graph,
    ) -> Result<(), DbError> {
        Ok(())
    }
}

// ── Shared operation replay ───────────────────────────────────────────────────

/// Apply a slice of [`Operation`]s to the graph in order.
///
/// Used by [`Graph::commit_transaction`] (explicit-transaction commit) and,
/// historically, by WAL replay on startup.  Each variant delegates to the
/// corresponding `Graph::apply_*` method.
pub(crate) fn apply_ops(graph: &mut Graph, ops: &[Operation]) -> Result<(), DbError> {
    use crate::transaction::operation::{Operation::*, PropertyTarget};

    for op in ops {
        match op {
            CreateNode { node } => {
                graph.apply_insert_node(node.clone());
            }

            CreateEdge { edge } => {
                graph.apply_insert_edge(edge.clone());
            }

            SetProperty { target, key, value } => match target {
                PropertyTarget::Node(id) => {
                    graph.apply_set_node_property(*id, key.clone(), value.clone())?;
                }
                PropertyTarget::Edge(id) => {
                    graph.apply_set_edge_property(*id, key.clone(), value.clone())?;
                }
            },

            RemoveProperty { target, key } => match target {
                PropertyTarget::Node(id) => {
                    graph.apply_remove_node_property(*id, key)?;
                }
                PropertyTarget::Edge(id) => {
                    graph.apply_remove_edge_property(*id, key)?;
                }
            },

            AddLabel { node_id, label } => {
                graph.apply_add_label(*node_id, label.clone())?;
            }

            RemoveLabel { node_id, label } => {
                graph.apply_remove_label(*node_id, label)?;
            }

            DeleteNode { node_id } => {
                graph.apply_delete_node(*node_id)?;
            }

            DeleteNodeDetach { node_id } => {
                graph.apply_delete_node_detach(*node_id)?;
            }

            DeleteEdge { edge_id } => {
                graph.apply_delete_edge(*edge_id)?;
            }

            CreateIndex { label, property } => {
                graph.apply_create_index(label, property);
            }

            DropIndex { label, property } => {
                graph.apply_drop_index(label, property);
            }
        }
    }

    Ok(())
}
