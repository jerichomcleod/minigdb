//! Wire protocol types for the minigdb TCP server.
//!
//! **Protocol v2** flow:
//! 1. Server sends [`ServerMessage::Hello`] on connect (includes `auth_required`).
//! 2. If auth is required, client sends [`ClientMessage::Auth`]; server replies
//!    [`ServerMessage::AuthOk`] or [`ServerMessage::AuthFail`].
//! 3. Client sends newline-delimited JSON [`Request`] objects; server replies
//!    with [`Response`] objects.
//! 4. Client may send [`ClientMessage::Admin`] for out-of-band admin commands.
//!
//! All messages are newline-delimited JSON (`serde_json`).

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::Value;

// ── Wire types ────────────────────────────────────────────────────────────────

/// A GQL query request sent by the client.
///
/// The optional `graph` field switches the connection's active graph for
/// this request (and all subsequent requests that omit the field).
#[derive(Debug, Deserialize)]
pub struct Request {
    /// Opaque client ID, echoed unchanged in the response.
    pub id: u64,
    /// The GQL statement to execute.
    pub query: String,
    /// Optional graph to query (switches current graph if provided).
    #[serde(default)]
    pub graph: Option<String>,
}

/// A server response for one GQL request.
#[derive(Debug, Serialize)]
pub struct Response {
    /// Echoed from the corresponding request.
    pub id: u64,
    /// Result rows — present on success, absent on error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows: Option<Vec<HashMap<String, JsonValue>>>,
    /// Error message — present on error, absent on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Wall-clock query execution time in milliseconds.
    pub elapsed_ms: f64,
}

impl Response {
    pub fn ok(id: u64, rows: Vec<HashMap<String, JsonValue>>, elapsed: Duration) -> Self {
        Self {
            id,
            rows: Some(rows),
            error: None,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        }
    }

    pub fn err(id: u64, msg: String, elapsed: Duration) -> Self {
        Self {
            id,
            rows: None,
            error: Some(msg),
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        }
    }
}

// ── Protocol v2 message types ─────────────────────────────────────────────────

/// Messages sent by the client outside the normal `Request` flow.
///
/// Detected by the presence of a `"type"` field.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Authentication credentials.
    Auth { user: String, password: String },
    /// Admin command.
    Admin {
        cmd: String,
        /// Graph name argument (used by `create` and `drop`).
        #[serde(default)]
        name: Option<String>,
        /// Filesystem path argument (used by `add_location` and
        /// `remove_location`).
        #[serde(default)]
        path: Option<String>,
    },
}

/// Messages sent by the server outside the normal `Response` flow.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Sent immediately on connection.
    Hello {
        version: &'static str,
        auth_required: bool,
    },
    /// Auth succeeded.
    AuthOk { user: String },
    /// Auth failed.
    AuthFail { error: String },
    /// Admin command succeeded; `data` carries optional payload fields.
    AdminOk {
        #[serde(flatten)]
        data: JsonValue,
    },
    /// Admin command failed.
    AdminFail { error: String },
}

// ── Value serialization ───────────────────────────────────────────────────────

/// Convert a `crate::Value` to `serde_json::Value` for wire transmission.
///
/// `Float(NaN)` and `Float(±Inf)` become `null` (JSON has no special float
/// literals; null is the least surprising substitute).
pub fn value_to_json(v: &Value) -> JsonValue {
    match v {
        Value::Null => JsonValue::Null,
        Value::Bool(b) => JsonValue::Bool(*b),
        Value::Int(i) => JsonValue::Number((*i).into()),
        Value::Float(f) => {
            if f.is_nan() || f.is_infinite() {
                JsonValue::Null
            } else {
                serde_json::Number::from_f64(*f)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null)
            }
        }
        Value::String(s) => JsonValue::String(s.clone()),
        Value::List(items) => JsonValue::Array(items.iter().map(value_to_json).collect()),
        Value::Map(m) => {
            let obj: serde_json::Map<String, JsonValue> = m
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect();
            JsonValue::Object(obj)
        }
    }
}

/// Convert a result row (`HashMap<String, Value>`) to its JSON representation.
pub fn row_to_json(row: &HashMap<String, Value>) -> HashMap<String, JsonValue> {
    row.iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect()
}
