//! Core data types shared across all modules.
//!
//! | Item | Description |
//! |---|---|
//! | [`Value`] | Scalar property value (Null, Bool, Int, Float, String, List) |
//! | [`Properties`] | `HashMap<String, Value>` property bag |
//! | [`Node`] | Graph node: ID + labels + properties |
//! | [`Edge`] | Graph edge: ID + label + endpoints + properties + directed flag |
//! | [`NodeId`] / [`EdgeId`] | ULID-based opaque identifiers |
//! | [`DbError`] | Unified error type for all database operations |
//! | [`value_index_key`] | Order-preserving index key encoder for the prop_idx CF |

pub mod error;
pub mod id;
pub mod node;
pub mod edge;
pub mod value;

pub use error::DbError;
pub use id::{ulid_decode, ulid_encode, ulid_new, EdgeId, NodeId};
pub use node::Node;
pub use edge::Edge;
pub use value::{value_index_key, Properties, Value};
