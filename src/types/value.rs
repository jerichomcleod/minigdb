//! [`Value`] — the scalar property type, plus the [`Properties`] type alias
//! and the [`value_index_key`] order-preserving encoder.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A property value stored on a node or edge.
///
/// Implements `PartialOrd` but **not** `Ord` because `f64` has `NaN`, which
/// does not compare equal to itself.  `ORDER BY` uses
/// `.partial_cmp().unwrap_or(Equal)` to produce a total order at runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// SQL-style null / missing value.
    Null,
    /// Boolean.
    Bool(bool),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit IEEE-754 floating-point.
    Float(f64),
    /// UTF-8 string.
    String(String),
    /// Ordered heterogeneous list (e.g. result of `collect()`).
    List(Vec<Value>),
    /// String-keyed map — used for parameter rows in UNWIND and `$param` passing.
    /// Not storable as a node/edge property; lives only in query evaluation context.
    Map(HashMap<String, Value>),
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;
        match (self, other) {
            (Value::Null,      Value::Null)      => Some(Equal),
            (Value::Bool(a),   Value::Bool(b))   => a.partial_cmp(b),
            (Value::Int(a),    Value::Int(b))     => a.partial_cmp(b),
            (Value::Float(a),  Value::Float(b))   => a.partial_cmp(b),
            (Value::String(a), Value::String(b)) => a.partial_cmp(b),
            (Value::List(a),   Value::List(b))   => a.partial_cmp(b),
            // Map and cross-type comparisons are unordered.
            _ => None,
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::String(s) => write!(f, "{s:?}"),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Value::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// Key-value property bag used on nodes and edges.
pub type Properties = HashMap<String, Value>;

/// Encode a `Value` as a canonical, **lexicographically order-preserving** string
/// suitable for use as a RocksDB property-index key segment.
///
/// Returns `None` for `List` (not indexable) and `Float(NaN)` (not orderable).
///
/// Encoding guarantees:
/// - `Int`   — `"I:"` + 16-char hex of `(i as u64) ^ 0x8000_0000_0000_0000`.
///   The XOR flips the sign bit, mapping `i64::MIN` → `0x0000…` and
///   `i64::MAX` → `0xffff…`, so byte-wise comparison matches numeric order.
/// - `Float` — `"F:"` + 16-char hex of IEEE-754 order-preserving transform.
///   Positive floats: flip sign bit. Negative floats: flip all bits.
/// - `Bool`  — `"B:0"` / `"B:1"` (false < true lexicographically).
/// - `String` — `"S:"` + raw string (no transformation; used for equality only).
/// - `Null`  — `"N:"`.
pub fn value_index_key(v: &Value) -> Option<String> {
    match v {
        Value::Null => Some("N:".to_string()),
        Value::Bool(b) => Some(if *b { "B:1".to_string() } else { "B:0".to_string() }),
        Value::Int(i) => {
            let ordered = (*i as u64) ^ 0x8000_0000_0000_0000u64;
            Some(format!("I:{ordered:016x}"))
        }
        Value::Float(f) => {
            if f.is_nan() {
                return None; // NaN is not orderable; exclude from index
            }
            let bits = f.to_bits();
            let ordered = if bits >> 63 == 0 {
                bits ^ 0x8000_0000_0000_0000u64 // positive: flip sign bit
            } else {
                !bits // negative: flip all bits
            };
            Some(format!("F:{ordered:016x}"))
        }
        Value::String(s) => Some(format!("S:{s}")),
        Value::List(_) | Value::Map(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(v: Value) -> String {
        value_index_key(&v).unwrap()
    }

    #[test]
    fn int_encoding_is_order_preserving() {
        let vals = [i64::MIN, -1000, -1, 0, 1, 1000, i64::MAX];
        let encoded: Vec<String> = vals.iter().map(|&i| key(Value::Int(i))).collect();
        for w in encoded.windows(2) {
            assert!(w[0] < w[1], "expected {} < {}", w[0], w[1]);
        }
    }

    #[test]
    fn float_encoding_is_order_preserving() {
        let vals: Vec<f64> = vec![f64::NEG_INFINITY, -1e10, -1.0, -0.0, 0.0, 1.0, 1e10, f64::INFINITY];
        let encoded: Vec<String> = vals.iter().map(|&f| key(Value::Float(f))).collect();
        for w in encoded.windows(2) {
            assert!(w[0] <= w[1], "expected {} <= {}", w[0], w[1]);
        }
    }

    #[test]
    fn nan_float_is_not_indexable() {
        assert!(value_index_key(&Value::Float(f64::NAN)).is_none());
    }

    #[test]
    fn list_is_not_indexable() {
        assert!(value_index_key(&Value::List(vec![])).is_none());
    }

    #[test]
    fn type_prefixes_separate_types() {
        // Int(0) and Float(0.0) and String("0") should all produce different keys.
        let ki = key(Value::Int(0));
        let kf = key(Value::Float(0.0));
        let ks = key(Value::String("0".into()));
        assert_ne!(ki, kf);
        assert_ne!(ki, ks);
        assert_ne!(kf, ks);
    }

    #[test]
    fn bool_order() {
        assert!(key(Value::Bool(false)) < key(Value::Bool(true)));
    }
}
