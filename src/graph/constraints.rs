//! Constraint definitions: uniqueness and type enforcement for graph nodes.
//!
//! # GQL syntax
//!
//! ```gql
//! CREATE CONSTRAINT UNIQUE ON :Person(email)
//! CREATE CONSTRAINT TYPE IS INTEGER ON :Person(age)
//! CREATE CONSTRAINT TYPE IS FLOAT   ON :Measurement(value)
//! CREATE CONSTRAINT TYPE IS STRING  ON :Document(title)
//! CREATE CONSTRAINT TYPE IS BOOLEAN ON :Feature(enabled)
//! DROP   CONSTRAINT UNIQUE ON :Person(email)
//! SHOW   CONSTRAINTS
//! ```
//!
//! # Storage
//!
//! Constraints are persisted as a bincode-serialized `Vec<ConstraintDef>` in the
//! `meta` column family under key `constraint_defs`.  They are loaded into
//! `Graph::constraint_defs` on `open()` and updated atomically on every
//! `add_constraint` / `remove_constraint` call.
//!
//! # Enforcement lifecycle
//!
//! 1. **INSERT** — `build_insert_ops` in the executor calls
//!    [`Graph::check_node_constraints`](crate::graph::Graph::check_node_constraints)
//!    for each node before building the `Operation::CreateNode`.  Nodes that pass
//!    the upsert dedup check (exact-match existing node) bypass constraint checking
//!    entirely because no new data is written.
//! 2. **SET** — `ops::set_node_property` calls the private `check_constraints` with
//!    `self_id = Some(node_id)` so the node being updated is excluded from the
//!    uniqueness check (a node may SET its own property to the same value it already
//!    holds without triggering a violation).
//! 3. **UNWIND … INSERT** — `execute_unwind_insert` calls `check_node_constraints`
//!    for every iteration before queuing a `CreateNode` operation.
//!
//! # Uniqueness check fallback
//!
//! The fast path uses the `prop_idx` column family (O(log n) lookup).  This index
//! is only populated when the user has also run `CREATE INDEX ON :Label(prop)`.
//! When the index returns nothing, the check falls back to a linear scan of all
//! nodes carrying the label to guarantee correctness even without an explicit index.

use serde::{Deserialize, Serialize};

/// The declared type a property value must conform to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValueKind {
    Integer,
    Float,
    String,
    Boolean,
}

impl std::fmt::Display for ValueKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueKind::Integer => write!(f, "INTEGER"),
            ValueKind::Float   => write!(f, "FLOAT"),
            ValueKind::String  => write!(f, "STRING"),
            ValueKind::Boolean => write!(f, "BOOLEAN"),
        }
    }
}

/// The kind of constraint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// No two nodes with this label may share the same value for this property.
    Unique,
    /// Every value written to this property must match the declared type.
    Type(ValueKind),
}

/// A single declared constraint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintDef {
    pub kind:     ConstraintKind,
    pub label:    String,
    pub property: String,
}
