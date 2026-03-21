//! CSV bulk import for nodes and edges.
//!
//! This module provides the shared core used by all three load surfaces:
//! the GQL `LOAD CSV` statement (REPL), the HTTP `/api/upload` endpoints,
//! and the Python `load_csv_nodes` / `load_csv_edges` methods.
//!
//! # CSV format — nodes
//!
//! ```csv
//! :ID,name,age,:LABEL
//! 1,Alice,30,Person
//! 2,Bob,25,Person
//! ```
//!
//! - `:ID` — user-assigned identifier used to resolve edge endpoints.
//!   Stored as the `_csv_id` property on the node (allows edge loads to
//!   match nodes by `_csv_id` without requiring a separate id_map lookup).
//!   May be omitted; nodes without `:ID` are inserted with no `_csv_id`.
//! - `:LABEL` — per-row label; overrides `default_label` when non-empty.
//! - All other non-`:`-prefixed columns → node properties (type-inferred).
//!
//! # CSV format — edges
//!
//! ```csv
//! :START_ID,:END_ID,:TYPE,weight
//! 1,2,KNOWS,0.9
//! 2,3,KNOWS,0.5
//! ```
//!
//! - `:START_ID` / `:END_ID` — must match `:ID` values from the node import.
//! - `:TYPE` — per-row edge label; overrides `default_label` when non-empty.
//! - Edges whose `:START_ID` or `:END_ID` are not in `id_map` are skipped.
//!
//! # Type inference
//!
//! Applied cell-by-cell (not column-by-column) for simplicity:
//! 1. Empty cell → `Value::Null` (property is omitted).
//! 2. `"true"` / `"false"` (case-insensitive) → `Value::Bool`.
//! 3. Valid `i64` → `Value::Int`.
//! 4. Valid `f64` → `Value::Float`.
//! 5. Anything else → `Value::String`.

use std::collections::HashMap;
use std::io::Read;

use crate::graph::Graph;
use crate::types::{ulid_encode, DbError, Edge, Node, NodeId, Properties, Value};

// ── Result types ──────────────────────────────────────────────────────────────

/// Result of a [`load_nodes_csv`] call.
#[derive(Debug, Default)]
pub struct NodesLoaded {
    /// Number of nodes successfully inserted.
    pub inserted: usize,
    /// Maps each user-supplied `:ID` column value to its assigned [`NodeId`].
    ///
    /// Pass this map (or its serialised form) to [`load_edges_csv`] so that
    /// `:START_ID` / `:END_ID` values in the edge CSV resolve correctly.
    pub id_map: HashMap<String, NodeId>,
}

/// Result of a [`load_edges_csv`] call.
#[derive(Debug, Default)]
pub struct EdgesLoaded {
    /// Number of edges successfully inserted.
    pub inserted: usize,
    /// Rows skipped because `:START_ID` or `:END_ID` was not in `id_map`.
    pub skipped: usize,
}

// ── Value inference ───────────────────────────────────────────────────────────

/// Infer a minigdb [`Value`] from a raw CSV field string.
///
/// The inference order is: empty → Null; true/false → Bool; integer → Int;
/// float → Float; otherwise → String.  This is applied cell-by-cell so mixed
/// columns (e.g. `"30"`, `""`, `"senior"`) produce the most specific type
/// for each individual cell.
pub fn infer_value(s: &str) -> Value {
    if s.is_empty() {
        return Value::Null;
    }
    let lower = s.to_ascii_lowercase();
    if lower == "true" {
        return Value::Bool(true);
    }
    if lower == "false" {
        return Value::Bool(false);
    }
    if let Ok(i) = s.parse::<i64>() {
        return Value::Int(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return Value::Float(f);
    }
    Value::String(s.to_string())
}

// ── Node loader ───────────────────────────────────────────────────────────────

/// Load nodes from a CSV reader into the graph.
///
/// Opens the given `reader`, parses it as RFC 4180 CSV, and inserts one node
/// per data row.  The `:ID` column (if present) is stored as the `_csv_id`
/// property on each node and also collected into the returned `id_map` so that
/// a subsequent [`load_edges_csv`] call can resolve `:START_ID` / `:END_ID`.
///
/// # Arguments
/// - `reader` — any `Read` source (file, `Cursor<Vec<u8>>`, etc.).
/// - `graph` — the live graph; mutations go through `apply_insert_node`.
/// - `default_label` — label to use when the row has no `:LABEL` column or the
///   column value is empty.  Pass `None` to insert unlabelled nodes.
///
/// # Errors
/// Returns [`DbError::Parse`] on malformed CSV (bad encoding, mis-quoted fields).
/// Individual insert failures are silently ignored (consistent with the existing
/// `apply_insert_node` contract which drops write errors in auto-commit mode).
pub fn load_nodes_csv<R: Read>(
    reader: R,
    graph: &mut Graph,
    default_label: Option<&str>,
) -> Result<NodesLoaded, DbError> {
    let mut rdr = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(reader);

    let headers = rdr
        .headers()
        .map_err(|e| DbError::Parse(format!("CSV header error: {e}")))?
        .clone();

    // Find the column indices for special columns.
    let id_col    = headers.iter().position(|h| h == ":ID");
    let label_col = headers.iter().position(|h| h == ":LABEL");

    let mut result = NodesLoaded::default();

    for record in rdr.records() {
        let rec = record.map_err(|e| DbError::Parse(format!("CSV row error: {e}")))?;

        // ── Determine the node label ────────────────────────────────────────
        let label: String = label_col
            .and_then(|i| rec.get(i))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| default_label.unwrap_or("").to_string());

        // ── Collect the :ID value ───────────────────────────────────────────
        let csv_id: Option<String> = id_col
            .and_then(|i| rec.get(i))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        // ── Build the property map ──────────────────────────────────────────
        let mut props = Properties::new();

        // Store :ID as _csv_id so edge loads can scan for it if needed.
        if let Some(ref cid) = csv_id {
            props.insert("_csv_id".to_string(), Value::String(cid.clone()));
        }

        for (i, header) in headers.iter().enumerate() {
            // Skip all special (colon-prefixed) columns.
            if header.starts_with(':') {
                continue;
            }
            let cell = rec.get(i).unwrap_or("").trim();
            let v = infer_value(cell);
            if v != Value::Null {
                props.insert(header.to_string(), v);
            }
        }

        // ── Insert the node ─────────────────────────────────────────────────
        let node_id = graph.alloc_node_id();
        let labels  = if label.is_empty() { vec![] } else { vec![label] };
        graph.apply_insert_node(Node::new(node_id, labels, props));

        // ── Record the id_map entry ─────────────────────────────────────────
        if let Some(cid) = csv_id {
            result.id_map.insert(cid, node_id);
        }
        result.inserted += 1;
    }

    Ok(result)
}

// ── Edge loader ───────────────────────────────────────────────────────────────

/// Load edges from a CSV reader into the graph.
///
/// Resolves `:START_ID` and `:END_ID` column values against `id_map`
/// (the map produced by [`load_nodes_csv`]).  Rows whose source or target
/// cannot be resolved are silently skipped and counted in `result.skipped`.
///
/// # Arguments
/// - `reader` — any `Read` source.
/// - `graph` — the live graph; mutations go through `apply_insert_edge`.
/// - `id_map` — maps user `:ID` strings to minigdb [`NodeId`]s.
/// - `default_label` — label to use when the row has no `:TYPE` column or the
///   column value is empty.  Defaults to `"RELATED"` when `None`.
///
/// # Errors
/// Returns [`DbError::Parse`] on malformed CSV.
pub fn load_edges_csv<R: Read>(
    reader: R,
    graph: &mut Graph,
    id_map: &HashMap<String, NodeId>,
    default_label: Option<&str>,
) -> Result<EdgesLoaded, DbError> {
    let mut rdr = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(reader);

    let headers = rdr
        .headers()
        .map_err(|e| DbError::Parse(format!("CSV header error: {e}")))?
        .clone();

    let start_col = headers.iter().position(|h| h == ":START_ID");
    let end_col   = headers.iter().position(|h| h == ":END_ID");
    let type_col  = headers.iter().position(|h| h == ":TYPE");

    let mut result = EdgesLoaded::default();

    for record in rdr.records() {
        let rec = record.map_err(|e| DbError::Parse(format!("CSV row error: {e}")))?;

        // ── Resolve endpoints ───────────────────────────────────────────────
        let src_str = start_col.and_then(|i| rec.get(i)).unwrap_or("").trim();
        let dst_str = end_col.and_then(|i| rec.get(i)).unwrap_or("").trim();

        let (from_node, to_node) = match (id_map.get(src_str), id_map.get(dst_str)) {
            (Some(&f), Some(&t)) => (f, t),
            _ => {
                result.skipped += 1;
                continue;
            }
        };

        // ── Determine the edge label ────────────────────────────────────────
        let label: String = type_col
            .and_then(|i| rec.get(i))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| default_label.unwrap_or("RELATED").to_string());

        // ── Build the property map ──────────────────────────────────────────
        let mut props = Properties::new();

        for (i, header) in headers.iter().enumerate() {
            if header.starts_with(':') {
                continue;
            }
            let cell = rec.get(i).unwrap_or("").trim();
            let v = infer_value(cell);
            if v != Value::Null {
                props.insert(header.to_string(), v);
            }
        }

        // ── Insert the edge ─────────────────────────────────────────────────
        let edge_id = graph.alloc_edge_id();
        graph.apply_insert_edge(Edge::new(edge_id, label, from_node, to_node, props, true));
        result.inserted += 1;
    }

    Ok(result)
}

// ── id_map serialisation helpers (for HTTP surface) ──────────────────────────

/// Serialise a `NodeId → ULID-string` map into a plain `String → String` map
/// suitable for JSON transmission (e.g. in HTTP upload responses).
pub fn id_map_to_strings(id_map: &HashMap<String, NodeId>) -> HashMap<String, String> {
    id_map
        .iter()
        .map(|(k, v)| (k.clone(), ulid_encode(v.0)))
        .collect()
}

/// Deserialise a `String → ULID-string` map (received from an HTTP client)
/// back into a `String → NodeId` map.
///
/// Entries with invalid ULID strings are silently dropped.
pub fn id_map_from_strings(raw: &HashMap<String, String>) -> HashMap<String, NodeId> {
    raw.iter()
        .filter_map(|(k, v)| {
            crate::types::ulid_decode(v)
                .ok()
                .map(|raw| (k.clone(), NodeId(raw)))
        })
        .collect()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    // Helper: CSV bytes from a string literal.
    fn bytes(s: &str) -> &[u8] {
        s.as_bytes()
    }

    // ── infer_value ───────────────────────────────────────────────────────────

    #[test]
    fn infer_empty_is_null() {
        assert_eq!(infer_value(""), Value::Null);
    }

    #[test]
    fn infer_bool() {
        assert_eq!(infer_value("true"),  Value::Bool(true));
        assert_eq!(infer_value("True"),  Value::Bool(true));
        assert_eq!(infer_value("FALSE"), Value::Bool(false));
    }

    #[test]
    fn infer_int() {
        assert_eq!(infer_value("42"),  Value::Int(42));
        assert_eq!(infer_value("-7"),  Value::Int(-7));
    }

    #[test]
    fn infer_float() {
        assert_eq!(infer_value("3.14"), Value::Float(3.14));
    }

    #[test]
    fn infer_string() {
        assert_eq!(infer_value("hello"), Value::String("hello".to_string()));
    }

    // ── load_nodes_csv ────────────────────────────────────────────────────────

    #[test]
    fn load_nodes_basic() {
        let mut g = Graph::new();
        let csv = ":ID,name,age,:LABEL\n1,Alice,30,Person\n2,Bob,25,Person\n";
        let r = load_nodes_csv(bytes(csv), &mut g, None).unwrap();

        assert_eq!(r.inserted, 2);
        assert_eq!(r.id_map.len(), 2);
        assert!(r.id_map.contains_key("1"));
        assert!(r.id_map.contains_key("2"));
        assert_eq!(g.node_count(), 2);

        let nid = r.id_map["1"];
        let node = g.get_node(nid).unwrap();
        assert_eq!(node.labels, vec!["Person"]);
        assert_eq!(node.properties["name"], Value::String("Alice".to_string()));
        assert_eq!(node.properties["age"],  Value::Int(30));
        // :ID stored as _csv_id
        assert_eq!(node.properties["_csv_id"], Value::String("1".to_string()));
    }

    #[test]
    fn load_nodes_default_label() {
        let mut g = Graph::new();
        let csv = ":ID,name\n1,Alice\n";
        let r = load_nodes_csv(bytes(csv), &mut g, Some("Employee")).unwrap();
        let node = g.get_node(r.id_map["1"]).unwrap();
        assert_eq!(node.labels, vec!["Employee"]);
    }

    #[test]
    fn load_nodes_no_id_column() {
        let mut g = Graph::new();
        let csv = "name,age\nAlice,30\nBob,25\n";
        let r = load_nodes_csv(bytes(csv), &mut g, Some("Person")).unwrap();
        assert_eq!(r.inserted, 2);
        assert_eq!(r.id_map.len(), 0); // no :ID column → empty map
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn load_nodes_bool_and_float_props() {
        let mut g = Graph::new();
        let csv = ":ID,active,score\n1,true,9.5\n";
        let r = load_nodes_csv(bytes(csv), &mut g, Some("P")).unwrap();
        let node = g.get_node(r.id_map["1"]).unwrap();
        assert_eq!(node.properties["active"], Value::Bool(true));
        assert_eq!(node.properties["score"],  Value::Float(9.5));
    }

    // ── load_edges_csv ────────────────────────────────────────────────────────

    #[test]
    fn load_edges_basic() {
        let mut g = Graph::new();
        // Load nodes first.
        let node_csv = ":ID,name\n1,Alice\n2,Bob\n";
        let nr = load_nodes_csv(bytes(node_csv), &mut g, Some("Person")).unwrap();

        let edge_csv = ":START_ID,:END_ID,:TYPE,weight\n1,2,KNOWS,0.9\n";
        let er = load_edges_csv(bytes(edge_csv), &mut g, &nr.id_map, None).unwrap();

        assert_eq!(er.inserted, 1);
        assert_eq!(er.skipped,  0);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn load_edges_skips_unresolved() {
        let mut g = Graph::new();
        let node_csv = ":ID,name\n1,Alice\n";
        let nr = load_nodes_csv(bytes(node_csv), &mut g, Some("P")).unwrap();

        let edge_csv = ":START_ID,:END_ID,:TYPE\n1,99,KNOWS\n"; // 99 not in map
        let er = load_edges_csv(bytes(edge_csv), &mut g, &nr.id_map, None).unwrap();
        assert_eq!(er.inserted, 0);
        assert_eq!(er.skipped,  1);
    }

    #[test]
    fn load_edges_default_label() {
        let mut g = Graph::new();
        let node_csv = ":ID,name\n1,Alice\n2,Bob\n";
        let nr = load_nodes_csv(bytes(node_csv), &mut g, Some("P")).unwrap();

        let edge_csv = ":START_ID,:END_ID\n1,2\n"; // no :TYPE column
        let er = load_edges_csv(bytes(edge_csv), &mut g, &nr.id_map, Some("LINKED")).unwrap();
        assert_eq!(er.inserted, 1);
        let eid = g.outgoing_edges(nr.id_map["1"])[0];
        let edge = g.get_edge(eid).unwrap();
        assert_eq!(edge.label, "LINKED");
    }

    #[test]
    fn load_edges_with_props() {
        let mut g = Graph::new();
        let node_csv = ":ID,name\n1,A\n2,B\n";
        let nr = load_nodes_csv(bytes(node_csv), &mut g, Some("P")).unwrap();
        let edge_csv = ":START_ID,:END_ID,:TYPE,weight,since\n1,2,KNOWS,0.9,2020\n";
        let er = load_edges_csv(bytes(edge_csv), &mut g, &nr.id_map, None).unwrap();
        assert_eq!(er.inserted, 1);
        let eid = g.outgoing_edges(nr.id_map["1"])[0];
        let edge = g.get_edge(eid).unwrap();
        assert_eq!(edge.properties["weight"], Value::Float(0.9));
        assert_eq!(edge.properties["since"],  Value::Int(2020));
    }

    // ── id_map serialisation ──────────────────────────────────────────────────

    #[test]
    fn id_map_round_trip() {
        let mut g = Graph::new();
        let csv = ":ID,name\nA,Alice\n";
        let r = load_nodes_csv(bytes(csv), &mut g, Some("P")).unwrap();
        let strings = id_map_to_strings(&r.id_map);
        let back = id_map_from_strings(&strings);
        assert_eq!(back["A"], r.id_map["A"]);
    }
}
