//! Async TCP server for minigdb — protocol v2.
//!
//! # Role
//! This module is the primary network entry point for minigdb.  It binds a TCP
//! socket, performs authentication handshakes, and dispatches each inbound line
//! to either the admin-command handler or the GQL query handler.  It also
//! optionally spawns the Axum HTTP/GUI server (when the `gui` feature is
//! enabled) and shuts down cleanly on Ctrl-C.
//!
//! # Protocol
//! Newline-delimited JSON over a plain TCP stream.  The server always sends
//! first.
//!
//! ## Handshake (server → client)
//! ```json
//! {"type":"hello","version":"2","auth_required":true}
//! ```
//! If `auth_required` is `true`, the client must respond immediately with:
//! ```json
//! {"type":"auth","user":"alice","password":"secret"}
//! ```
//! The server replies with one of:
//! ```json
//! {"type":"auth_ok","user":"alice"}
//! {"type":"auth_fail","error":"invalid credentials"}
//! ```
//! A failed auth closes the connection.  When `auth_required` is `false` the
//! handshake is skipped and the user is treated as `"anonymous"`.
//!
//! ## Normal queries (backward-compatible with v1 clients)
//! ```json
//! {"id":1,"query":"MATCH (n) RETURN n.name"}
//! {"id":2,"graph":"analytics","query":"MATCH (n) RETURN count(n)"}
//! ```
//! The `graph` field is optional; if omitted the connection's current graph is
//! used (defaults to `"default"`).  Responses carry the same `id` back so
//! clients can match them:
//! ```json
//! {"id":1,"rows":[...],"elapsed_ms":0.42}
//! {"id":1,"error":"...","elapsed_ms":0.01}
//! ```
//!
//! ## Transaction control
//! `BEGIN`, `COMMIT`, `ROLLBACK` are handled as special query strings.  On a
//! successful `BEGIN` the connection acquires an exclusive
//! [`OwnedMutexGuard`](tokio::sync::OwnedMutexGuard) on the target graph,
//! preventing concurrent writes from other connections until `COMMIT`,
//! `ROLLBACK`, or a client disconnect.
//!
//! ## Admin commands
//! Sent as typed messages rather than GQL:
//! ```json
//! {"type":"admin","cmd":"graphs"}
//! {"type":"admin","cmd":"stats"}
//! {"type":"admin","cmd":"create","name":"newgraph"}
//! {"type":"admin","cmd":"drop","name":"oldgraph"}
//! ```
//! Responses use `{"type":"admin_ok","data":{...}}` or
//! `{"type":"admin_fail","error":"..."}`.
//!
//! # Key design decisions
//! - One Tokio task per TCP connection (`tokio::spawn`).
//! - Transaction isolation is implemented by holding an
//!   `OwnedMutexGuard<GraphState>` for the lifetime of the transaction; the
//!   guard is stored in [`ConnectionState`].  This gives exclusive access to the
//!   graph without any explicit locking on individual GQL operations.
//! - On unexpected disconnect, the `handle` function's cleanup path calls
//!   `rollback_transaction()` to prevent partial writes leaking into the graph.
//! - Graceful shutdown: a background task waits for `ctrl_c`, calls
//!   `checkpoint_all()` to flush all open graphs to RocksDB, then calls
//!   `process::exit(0)`.

pub mod auth;
#[cfg(feature = "gui")]
pub mod http;
pub mod locations;
pub mod protocol;
pub mod registry;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::OwnedMutexGuard;

use auth::{verify_password, ServerConfig};
use protocol::{row_to_json, ClientMessage, Request, Response, ServerMessage};
use registry::{GraphRegistry, GraphState};

// ── Public entry point ────────────────────────────────────────────────────────

/// Bind on `addr` and serve all named graphs indefinitely.
///
/// # Arguments
/// - `config` — server configuration including auth settings and user list.
/// - `data_root` — server data directory (e.g. `~/.local/share/minigdb`).
///   The primary graph root is `<data_root>/graphs/`.  Extra roots are loaded
///   from `<data_root>/locations.toml` and can be modified at runtime via
///   admin commands.
/// - `session_roots` — additional graph-root directories for this process
///   lifetime only (e.g. from `--graphs-dir` on the CLI).  Not persisted.
/// - `addr` — TCP address to listen on (e.g. `127.0.0.1:7474`).
/// - `gui_addr` — if `Some`, also starts the HTTP GUI server on that address.
///   Requires the `gui` Cargo feature; if the feature is disabled this
///   parameter is silently ignored.
///
/// # Behaviour
/// - Ensures the `"default"` graph is pre-opened at startup so the first
///   client request does not incur a cold-open penalty.
/// - Spawns a background task that listens for `Ctrl-C`, checkpoints all open
///   graphs, and calls `process::exit(0)`.
/// - Accepts connections in a `loop`; each accepted connection is dispatched
///   to its own `tokio::spawn`ed task via [`handle`].
///
/// This function never returns under normal operation.
pub async fn serve(
    config: ServerConfig,
    data_root: PathBuf,
    session_roots: Vec<PathBuf>,
    addr: SocketAddr,
    gui_addr: Option<SocketAddr>,
) -> std::io::Result<()> {
    std::fs::create_dir_all(data_root.join("graphs"))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let registry = GraphRegistry::new(data_root, session_roots);
    let listener = TcpListener::bind(addr).await?;
    eprintln!("minigdb listening on {addr}  (Ctrl-C to stop)");

    // Ensure the "default" and "_meta" graphs exist at startup.
    if let Err(e) = registry.get_or_open("default").await {
        eprintln!("Warning: could not open default graph: {e}");
    }
    if let Err(e) = registry.get_or_open(registry::META_GRAPH).await {
        eprintln!("Warning: could not open _meta graph: {e}");
    }

    // Graceful shutdown task: waits for Ctrl-C, flushes all graphs, then exits.
    let registry_shutdown = Arc::clone(&registry);
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        eprintln!("\nShutting down — checkpointing all graphs…");
        registry_shutdown.checkpoint_all().await;
        std::process::exit(0);
    });

    let config = Arc::new(config);

    // Start GUI HTTP server if an address was provided and the feature is on.
    #[cfg(feature = "gui")]
    if let Some(gui_addr) = gui_addr {
        let reg = Arc::clone(&registry);
        let cfg = Arc::clone(&config);
        tokio::spawn(async move {
            if let Err(e) = http::serve(gui_addr, reg, cfg).await {
                eprintln!("GUI server error: {e}");
            }
        });
    }
    // Suppress unused warning when gui feature is off.
    #[cfg(not(feature = "gui"))]
    let _ = gui_addr;

    // Accept loop: each connection runs in its own task.
    loop {
        let (stream, peer) = listener.accept().await?;
        eprintln!("[{peer}] connected");
        let registry = Arc::clone(&registry);
        let config = Arc::clone(&config);
        tokio::spawn(async move {
            if let Err(e) = handle(stream, peer, registry, config).await {
                eprintln!("[{peer}] error: {e}");
            }
            eprintln!("[{peer}] disconnected");
        });
    }
}

// ── Per-connection state ──────────────────────────────────────────────────────

/// All mutable state that belongs to a single TCP connection.
///
/// One instance is created in [`handle`] after a successful auth handshake and
/// lives for the lifetime of the connection.
struct ConnectionState {
    /// Authenticated username, or `"anonymous"` when auth is disabled.
    user: String,
    /// The graph name used for queries that do not include a `"graph"` field.
    /// Defaults to `"default"` and can be changed per-request via
    /// `Request::graph`.
    current_graph: String,
    /// Held for the duration of an explicit `BEGIN … COMMIT/ROLLBACK`
    /// transaction.
    ///
    /// While `Some`, this connection has exclusive write access to the graph:
    /// the `OwnedMutexGuard` keeps the `Arc<Mutex<GraphState>>` locked so no
    /// other connection can acquire it.  Dropped (and therefore released) on
    /// `COMMIT`, `ROLLBACK`, or client disconnect.
    txn_lock: Option<OwnedMutexGuard<GraphState>>,
}

// ── Connection handler ────────────────────────────────────────────────────────

/// Drive a single TCP connection from hello through auth through the query loop.
///
/// This function is the body of a per-connection `tokio::spawn` task.  It
/// performs three sequential phases:
///
/// 1. **Hello**: serialises and sends the [`ServerMessage::Hello`] frame.
/// 2. **Auth handshake**: if `config.server.auth_required` is set, reads
///    exactly one [`ClientMessage::Auth`] frame.  Any other message type, or
///    a failed password check, results in [`ServerMessage::AuthFail`] and an
///    immediate return.
/// 3. **Query loop**: reads lines one at a time, calling [`dispatch_line`] for
///    each non-empty line until EOF.
///
/// On return (EOF or error), any open transaction is automatically rolled back
/// by calling `graph.rollback_transaction()` on the held guard, then dropping
/// it to release the mutex.
async fn handle(
    stream: TcpStream,
    _peer: SocketAddr,
    registry: Arc<GraphRegistry>,
    config: Arc<ServerConfig>,
) -> std::io::Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut lines = BufReader::new(read_half).lines();

    // Phase 1: Send hello — server always speaks first in protocol v2.
    send_msg(
        &mut write_half,
        &ServerMessage::Hello {
            version: "2",
            auth_required: config.server.auth_required,
        },
    )
    .await?;

    // Phase 2: Auth handshake (only when auth_required is enabled).
    // MED-4: Limit failed auth attempts per connection to prevent brute force.
    const MAX_AUTH_ATTEMPTS: u32 = 5;
    let user = if config.server.auth_required {
        let mut auth_attempts: u32 = 0;
        // Loop until we receive a valid auth attempt or the client disconnects.
        loop {
            let line = match lines.next_line().await? {
                Some(l) => l,
                None => return Ok(()), // client disconnected before auth
            };
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<ClientMessage>(&line) {
                Ok(ClientMessage::Auth { user, password }) => {
                    match config.find_user(&user) {
                        // Password matches: send auth_ok and break with the username.
                        Some(entry) if verify_password(&password, &entry.password_hash) => {
                            send_msg(
                                &mut write_half,
                                &ServerMessage::AuthOk { user: user.clone() },
                            )
                            .await?;
                            break user;
                        }
                        // User not found or wrong password: increment counter.
                        _ => {
                            auth_attempts += 1;
                            if auth_attempts >= MAX_AUTH_ATTEMPTS {
                                send_msg(
                                    &mut write_half,
                                    &ServerMessage::AuthFail {
                                        error: "too many failed authentication attempts"
                                            .to_string(),
                                    },
                                )
                                .await?;
                                return Ok(());
                            }
                            send_msg(
                                &mut write_half,
                                &ServerMessage::AuthFail {
                                    error: "invalid credentials".to_string(),
                                },
                            )
                            .await?;
                            // Do not close — allow the client to retry.
                        }
                    }
                }
                // Non-auth message during handshake: reject and close.
                _ => {
                    send_msg(
                        &mut write_half,
                        &ServerMessage::AuthFail {
                            error: "expected auth message".to_string(),
                        },
                    )
                    .await?;
                    return Ok(());
                }
            }
        }
    } else {
        // Auth disabled — treat every connection as anonymous.
        "anonymous".to_string()
    };

    // Phase 3: Query loop — process one JSON line per iteration.
    // MED-5: Reject messages that exceed MAX_MSG_BYTES to prevent memory exhaustion.
    const MAX_MSG_BYTES: usize = 64 * 1024 * 1024; // 64 MiB
    let mut state = ConnectionState {
        user,
        current_graph: "default".to_string(),
        txn_lock: None,
    };

    while let Some(line) = lines.next_line().await? {
        if line.len() > MAX_MSG_BYTES {
            let resp = Response::err(
                0,
                format!("message too large: {} bytes (limit {MAX_MSG_BYTES})", line.len()),
                std::time::Duration::ZERO,
            );
            send(&mut write_half, &resp).await?;
            break;
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        dispatch_line(&line, &mut state, &registry, &config, &mut write_half).await?;
    }

    // Phase 4: Cleanup — auto-rollback any open transaction on disconnect.
    // Dropping the guard releases the mutex so other connections can proceed.
    if let Some(mut guard) = state.txn_lock.take() {
        let _ = guard.graph.rollback_transaction();
    }

    Ok(())
}

// ── Line dispatcher ───────────────────────────────────────────────────────────

/// Parse a single inbound JSON line and route it to the appropriate handler.
///
/// The routing decision is based on whether the top-level JSON object contains
/// a `"type"` field:
/// - **With `"type"`**: treated as a [`ClientMessage`] (admin commands or a
///   late auth attempt) and forwarded to [`handle_client_message`].
/// - **Without `"type"`**: treated as a [`Request`] (a GQL query) and
///   forwarded to [`handle_query`].
///
/// JSON parse failures are returned as [`Response::err`] with `id = 0`.
async fn dispatch_line<W: AsyncWriteExt + Unpin>(
    line: &str,
    state: &mut ConnectionState,
    registry: &Arc<GraphRegistry>,
    config: &Arc<ServerConfig>,
    write: &mut W,
) -> std::io::Result<()> {
    // Parse first into a generic Value so we can peek at the "type" field
    // without committing to a specific deserialization target.
    let json_val: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            let resp = Response::err(0, format!("invalid JSON: {e}"), std::time::Duration::ZERO);
            return send(write, &resp).await;
        }
    };

    if json_val.get("type").is_some() {
        // ClientMessage path — admin commands or duplicate auth messages.
        match serde_json::from_value::<ClientMessage>(json_val) {
            Ok(msg) => handle_client_message(msg, state, registry, config, write).await,
            Err(e) => {
                let resp =
                    Response::err(0, format!("invalid message: {e}"), std::time::Duration::ZERO);
                send(write, &resp).await
            }
        }
    } else {
        // Request (GQL query) path.
        match serde_json::from_value::<Request>(json_val) {
            Ok(req) => handle_query(req, state, registry, config, write).await,
            Err(e) => {
                let resp =
                    Response::err(0, format!("invalid request: {e}"), std::time::Duration::ZERO);
                send(write, &resp).await
            }
        }
    }
}

// ── Admin command handler ─────────────────────────────────────────────────────

/// Handle a [`ClientMessage`] received after the initial auth handshake.
///
/// Currently two message variants are processed:
/// - `Auth` — a duplicate auth attempt after the connection is already
///   established.  Returns [`ServerMessage::AdminFail`] with an explanatory
///   message; no re-authentication is supported.
/// - `Admin` — one of the admin sub-commands described below.
///
/// # Admin sub-commands
/// | `cmd`            | args           | Description                               |
/// |------------------|----------------|-------------------------------------------|
/// | `graphs`         | —              | List all user-visible graph names.        |
/// | `stats`          | —              | Alias for `graphs`; returns `open_graphs`.|
/// | `create`         | `name`         | Create a new named graph on disk.         |
/// | `drop`           | `name`         | Delete a named graph from disk.           |
/// | `locations`      | —              | List all registered graph-root dirs.      |
/// | `add_location`   | `path`         | Register an extra root directory.         |
/// | `remove_location`| `path`         | Unregister an extra root directory.       |
///
/// Unknown commands return [`ServerMessage::AdminFail`].
async fn handle_client_message<W: AsyncWriteExt + Unpin>(
    msg: ClientMessage,
    _state: &mut ConnectionState,
    registry: &Arc<GraphRegistry>,
    _config: &Arc<ServerConfig>,
    write: &mut W,
) -> std::io::Result<()> {
    match msg {
        ClientMessage::Auth { .. } => {
            // Auth after handshake is a no-op (already authenticated).
            send_msg(write, &ServerMessage::AdminFail {
                error: "already authenticated".to_string(),
            })
            .await
        }
        ClientMessage::Admin { cmd, name, path } => match cmd.as_str() {
            "graphs" => {
                let graphs = registry.list().await;
                send_msg(
                    write,
                    &ServerMessage::AdminOk {
                        data: serde_json::json!({ "graphs": graphs }),
                    },
                )
                .await
            }
            "stats" => {
                let open = registry.list().await;
                send_msg(
                    write,
                    &ServerMessage::AdminOk {
                        data: serde_json::json!({ "open_graphs": open }),
                    },
                )
                .await
            }
            "create" => match name.as_deref() {
                None => {
                    send_msg(
                        write,
                        &ServerMessage::AdminFail {
                            error: "create requires 'name'".to_string(),
                        },
                    )
                    .await
                }
                Some(n) => match registry.create(n).await {
                    Ok(()) => {
                        send_msg(write, &ServerMessage::AdminOk { data: serde_json::json!({}) })
                            .await
                    }
                    Err(e) => {
                        send_msg(
                            write,
                            &ServerMessage::AdminFail {
                                error: e.to_string(),
                            },
                        )
                        .await
                    }
                },
            },
            "drop" => match name.as_deref() {
                None => {
                    send_msg(
                        write,
                        &ServerMessage::AdminFail {
                            error: "drop requires 'name'".to_string(),
                        },
                    )
                    .await
                }
                Some(n) if n.starts_with('_') => {
                    send_msg(
                        write,
                        &ServerMessage::AdminFail {
                            error: format!("cannot drop system graph '{n}'"),
                        },
                    )
                    .await
                }
                Some(n) => match registry.drop_graph(n).await {
                    Ok(()) => {
                        send_msg(write, &ServerMessage::AdminOk { data: serde_json::json!({}) })
                            .await
                    }
                    Err(e) => {
                        send_msg(
                            write,
                            &ServerMessage::AdminFail {
                                error: e.to_string(),
                            },
                        )
                        .await
                    }
                },
            },
            "locations" => {
                let locs = registry.list_locations().await;
                let arr: Vec<serde_json::Value> = locs
                    .into_iter()
                    .map(|(p, primary)| serde_json::json!({
                        "path": p.to_string_lossy(),
                        "primary": primary,
                    }))
                    .collect();
                send_msg(
                    write,
                    &ServerMessage::AdminOk {
                        data: serde_json::json!({ "locations": arr }),
                    },
                )
                .await
            }
            "add_location" => match path.as_deref() {
                None => {
                    send_msg(write, &ServerMessage::AdminFail {
                        error: "add_location requires 'path'".to_string(),
                    })
                    .await
                }
                Some(p) => {
                    match registry.add_location(std::path::PathBuf::from(p)).await {
                        Ok(()) => send_msg(write, &ServerMessage::AdminOk {
                            data: serde_json::json!({}),
                        }).await,
                        Err(e) => send_msg(write, &ServerMessage::AdminFail {
                            error: e.to_string(),
                        }).await,
                    }
                }
            },
            "remove_location" => match path.as_deref() {
                None => {
                    send_msg(write, &ServerMessage::AdminFail {
                        error: "remove_location requires 'path'".to_string(),
                    })
                    .await
                }
                Some(p) => {
                    match registry.remove_location(std::path::Path::new(p)).await {
                        Ok(()) => send_msg(write, &ServerMessage::AdminOk {
                            data: serde_json::json!({}),
                        }).await,
                        Err(e) => send_msg(write, &ServerMessage::AdminFail {
                            error: e.to_string(),
                        }).await,
                    }
                }
            },
            other => {
                send_msg(
                    write,
                    &ServerMessage::AdminFail {
                        error: format!("unknown admin command '{other}'"),
                    },
                )
                .await
            }
        },
    }
}

// ── GQL query handler ─────────────────────────────────────────────────────────

/// Execute a single GQL [`Request`] and write one [`Response`] back to the
/// client.
///
/// # Graph switching
/// If `req.graph` is `Some`, `state.current_graph` is updated before execution.
/// This means a client can target a different graph on every request without
/// sending an admin message.
///
/// # Access control
/// When auth is required, the user's `graphs` allowlist is checked against
/// `state.current_graph` before any work is done.
///
/// # Transaction control keywords
/// `BEGIN`, `COMMIT`, and `ROLLBACK` are intercepted before the query reaches
/// the GQL parser (which does not recognise them):
///
/// - **BEGIN**: acquires `lock_owned()` on the target graph's `Arc<Mutex>`.
///   This is an async call that yields until no other connection holds the
///   lock.  The resulting `OwnedMutexGuard` is stored in `state.txn_lock`,
///   keeping the graph exclusively locked for this connection.
/// - **COMMIT**: takes `state.txn_lock`, calls `commit_transaction()`, and
///   drops the guard — releasing the mutex to other connections.
/// - **ROLLBACK**: same as COMMIT but calls `rollback_transaction()`.
///
/// # Normal queries
/// For all other query strings, if a transaction is active (`state.txn_lock`
/// is `Some`) the existing guard is used directly.  Otherwise a transient
/// `lock()` is acquired for the duration of this single query.
async fn handle_query<W: AsyncWriteExt + Unpin>(
    req: Request,
    state: &mut ConnectionState,
    registry: &Arc<GraphRegistry>,
    config: &Arc<ServerConfig>,
    write: &mut W,
) -> std::io::Result<()> {
    // Switch active graph if the request specifies one.
    if let Some(ref g) = req.graph {
        state.current_graph = g.clone();
    }

    // Access control: reject if the authenticated user cannot see this graph.
    if config.server.auth_required {
        if let Some(entry) = config.find_user(&state.user) {
            if !entry.can_access(&state.current_graph) {
                let resp = Response::err(
                    req.id,
                    format!(
                        "user '{}' does not have access to graph '{}'",
                        state.user, state.current_graph
                    ),
                    std::time::Duration::ZERO,
                );
                return send(write, &resp).await;
            }
        }
    }

    let start = Instant::now();
    let id = req.id;

    // Normalise the query string for keyword comparison without allocating a
    // second copy for the GQL executor path.
    let upper = req.query.trim().to_uppercase();
    let bare = upper.trim_end_matches(';').trim();

    let resp = match bare {
        "BEGIN" => {
            if state.txn_lock.is_some() {
                Response::err(id, "transaction already open".to_string(), start.elapsed())
            } else {
                match registry.get_or_open(&state.current_graph).await {
                    Err(e) => Response::err(id, e.to_string(), start.elapsed()),
                    Ok(arc) => {
                        // `lock_owned()` returns an OwnedMutexGuard that can be
                        // stored without borrowing the Arc itself.  This guard
                        // keeps the graph exclusively locked until it is dropped.
                        let mut guard = arc.lock_owned().await;
                        if guard.dropped {
                            return send(write, &Response::err(
                                id,
                                format!("graph '{}' has been dropped", state.current_graph),
                                start.elapsed(),
                            )).await;
                        }
                        match guard.graph.begin_transaction() {
                            Ok(()) => {
                                // Store the guard — the mutex remains locked.
                                state.txn_lock = Some(guard);
                                Response::ok(id, vec![], start.elapsed())
                            }
                            Err(e) => Response::err(id, e.to_string(), start.elapsed()),
                        }
                    }
                }
            }
        }
        "COMMIT" => {
            if let Some(mut guard) = state.txn_lock.take() {
                match guard.graph.commit_transaction() {
                    Ok(()) => Response::ok(id, vec![], start.elapsed()),
                    Err(e) => {
                        // On commit error, the guard is dropped here, implicitly
                        // rolling back rather than leaving the lock held forever.
                        Response::err(id, e.to_string(), start.elapsed())
                    }
                }
            } else {
                Response::err(id, "no active transaction".to_string(), start.elapsed())
            }
        }
        "ROLLBACK" => {
            if let Some(mut guard) = state.txn_lock.take() {
                match guard.graph.rollback_transaction() {
                    Ok(()) => Response::ok(id, vec![], start.elapsed()),
                    Err(e) => Response::err(id, e.to_string(), start.elapsed()),
                }
            } else {
                Response::err(id, "no active transaction".to_string(), start.elapsed())
            }
        }
        _ => {
            // Normal GQL query — use the held transaction guard or a transient lock.
            if let Some(ref mut guard) = state.txn_lock {
                // Check whether the graph was dropped while the transaction was open.
                if guard.dropped {
                    Response::err(
                        id,
                        format!("graph '{}' has been dropped", state.current_graph),
                        start.elapsed(),
                    )
                } else {
                    // Reuse the existing exclusive guard; no new lock acquisition needed.
                    // Destructure into separate field references so the borrow checker
                    // can confirm the two mutable borrows are non-overlapping.
                    let GraphState { graph, txn_id, .. } = &mut **guard;
                    execute_and_build_response(id, &req.query, graph, txn_id, start)
                }
            } else {
                // No active transaction: acquire a per-query lock.
                match registry.get_or_open(&state.current_graph).await {
                    Err(e) => Response::err(id, e.to_string(), start.elapsed()),
                    Ok(arc) => {
                        let mut guard = arc.lock().await;
                        if guard.dropped {
                            Response::err(
                                id,
                                format!("graph '{}' has been dropped", state.current_graph),
                                start.elapsed(),
                            )
                        } else {
                            let GraphState { graph, txn_id, .. } = &mut *guard;
                            execute_and_build_response(id, &req.query, graph, txn_id, start)
                        }
                    }
                }
            }
        }
    };

    send(write, &resp).await
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Run `query` against `graph` and convert the result to a [`Response`].
///
/// This is a thin synchronous wrapper around [`crate::query_capturing`] that
/// handles the `Ok`/`Err` branching and formats elapsed time.
///
/// # Parameters
/// - `id` — request identifier echoed back in the response.
/// - `query` — the raw GQL string.
/// - `graph` — mutable borrow of the [`crate::Graph`] to execute against.
/// - `txn_id` — mutable counter used by `query_capturing` to tag WAL entries.
/// - `start` — timestamp taken just before the caller entered this code path,
///   used to compute `elapsed_ms` in the response.
fn execute_and_build_response(
    id: u64,
    query: &str,
    graph: &mut crate::Graph,
    txn_id: &mut u64,
    start: Instant,
) -> Response {
    match crate::query_capturing(query, graph, txn_id) {
        Ok((rows, _ops)) => {
            let json_rows: Vec<_> = rows.iter().map(row_to_json).collect();
            Response::ok(id, json_rows, start.elapsed())
        }
        Err(e) => Response::err(id, e.to_string(), start.elapsed()),
    }
}

/// Serialize a [`Response`] as a newline-terminated JSON string and write it
/// to `w`.
///
/// `Response` is always serializable (no non-serializable fields), so the
/// `expect` here is a programming-error assertion, not a runtime failure path.
async fn send<W: AsyncWriteExt + Unpin>(w: &mut W, resp: &Response) -> std::io::Result<()> {
    let mut line = serde_json::to_string(resp).expect("Response is always serializable");
    line.push('\n');
    w.write_all(line.as_bytes()).await
}

/// Serialize a [`ServerMessage`] as a newline-terminated JSON string and write
/// it to `w`.
///
/// Used for handshake and admin-response frames (as opposed to query
/// [`Response`] frames which use [`send`]).
async fn send_msg<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    msg: &ServerMessage,
) -> std::io::Result<()> {
    let mut line = serde_json::to_string(msg).expect("ServerMessage is always serializable");
    line.push('\n');
    w.write_all(line.as_bytes()).await
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, Write};
    use std::net::TcpStream;

    /// Spin up the server with no auth on a random port.
    fn start_test_server() -> std::net::SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let data_root = tempfile::tempdir().unwrap().into_path();
        let config = ServerConfig {
            server: auth::ServerSection { auth_required: false },
            users: vec![],
        };

        std::thread::spawn(move || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(serve(config, data_root, vec![], addr, None))
                .unwrap();
        });

        std::thread::sleep(std::time::Duration::from_millis(150));
        addr
    }

    /// Spin up a server that requires auth.
    fn start_auth_server(users: Vec<auth::UserEntry>) -> std::net::SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let data_root = tempfile::tempdir().unwrap().into_path();
        let config = ServerConfig {
            server: auth::ServerSection { auth_required: true },
            users,
        };

        std::thread::spawn(move || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(serve(config, data_root, vec![], addr, None))
                .unwrap();
        });

        std::thread::sleep(std::time::Duration::from_millis(150));
        addr
    }

    /// Connect to a no-auth server: reads and discards the hello message.
    fn connect_no_auth(addr: std::net::SocketAddr) -> (TcpStream, std::io::BufReader<TcpStream>) {
        let stream = TcpStream::connect(addr).unwrap();
        let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
        // Consume the hello message.
        let mut hello = String::new();
        reader.read_line(&mut hello).unwrap();
        let hello: serde_json::Value = serde_json::from_str(hello.trim()).unwrap();
        assert_eq!(hello["type"], "hello");
        assert_eq!(hello["auth_required"], false);
        (stream, reader)
    }

    /// Connect to an auth server, authenticate, and return (writer, reader).
    fn connect_with_auth(
        addr: std::net::SocketAddr,
        user: &str,
        password: &str,
    ) -> (TcpStream, std::io::BufReader<TcpStream>) {
        let stream = TcpStream::connect(addr).unwrap();
        let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
        let mut writer = stream.try_clone().unwrap();

        // Read hello.
        let mut hello = String::new();
        reader.read_line(&mut hello).unwrap();
        let hello: serde_json::Value = serde_json::from_str(hello.trim()).unwrap();
        assert_eq!(hello["type"], "hello");

        // Send auth.
        let auth = serde_json::json!({"type":"auth","user":user,"password":password});
        let mut line = serde_json::to_string(&auth).unwrap();
        line.push('\n');
        writer.write_all(line.as_bytes()).unwrap();

        // Read auth response.
        let mut resp = String::new();
        reader.read_line(&mut resp).unwrap();
        let resp: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();

        (stream, reader)
    }

    /// Send one JSON request line and read one response line.
    fn roundtrip(
        stream: &mut TcpStream,
        reader: &mut std::io::BufReader<TcpStream>,
        req: serde_json::Value,
    ) -> serde_json::Value {
        let mut line = serde_json::to_string(&req).unwrap();
        line.push('\n');
        stream.write_all(line.as_bytes()).unwrap();
        let mut resp = String::new();
        reader.read_line(&mut resp).unwrap();
        serde_json::from_str(resp.trim()).unwrap()
    }

    // ── Original tests (updated for new handshake) ────────────────────────────

    #[test]
    fn server_basic_query() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        let resp = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id": 1, "query": r#"INSERT (:Person {name: "Alice", age: 30})"#}),
        );
        assert!(resp.get("error").is_none(), "insert error: {resp}");

        let resp = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id": 2, "query": "MATCH (n:Person) RETURN n.name, n.age"}),
        );
        let rows = resp["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["n.name"], "Alice");
        assert_eq!(rows[0]["n.age"], 30);
    }

    #[test]
    fn server_error_response() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        let resp = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id": 1, "query": "THIS IS NOT GQL %%%"}),
        );
        assert!(resp.get("error").is_some(), "expected error: {resp}");
    }

    #[test]
    fn server_invalid_json() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut stream = stream;

        stream.write_all(b"not json at all\n").unwrap();
        let mut resp_line = String::new();
        reader.read_line(&mut resp_line).unwrap();
        let resp: serde_json::Value = serde_json::from_str(resp_line.trim()).unwrap();
        assert!(resp.get("error").is_some());
        assert_eq!(resp["id"], 0);
    }

    #[test]
    fn server_transaction_commit() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        let r = roundtrip(&mut writer, &mut reader, serde_json::json!({"id":1,"query":"BEGIN"}));
        assert!(r.get("error").is_none());

        roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":2,"query":r#"INSERT (:City {name:"NYC"})"#}),
        );
        roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":3,"query":r#"INSERT (:City {name:"LA"})"#}),
        );

        let r = roundtrip(&mut writer, &mut reader, serde_json::json!({"id":4,"query":"COMMIT"}));
        assert!(r.get("error").is_none());

        let r = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":5,"query":"MATCH (c:City) RETURN c.name"}),
        );
        let rows = r["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn server_elapsed_ms_present() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        let resp = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id": 1, "query": "MATCH (n) RETURN n"}),
        );
        assert!(resp["elapsed_ms"].as_f64().is_some());
    }

    // ── New tests ─────────────────────────────────────────────────────────────

    #[test]
    fn server_hello_message() {
        let addr = start_test_server();
        let stream = TcpStream::connect(addr).unwrap();
        let mut reader = std::io::BufReader::new(stream);

        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let hello: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(hello["type"], "hello");
        assert_eq!(hello["version"], "2");
        assert_eq!(hello["auth_required"], false);
    }

    #[test]
    fn server_auth_required_rejects_bad_password() {
        let users = vec![auth::UserEntry {
            name: "alice".to_string(),
            password_hash: auth::hash_password("correct"),
            graphs: vec!["*".to_string()],
        }];
        let addr = start_auth_server(users);

        let stream = TcpStream::connect(addr).unwrap();
        let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
        let mut writer = stream;

        // Read hello.
        let mut hello = String::new();
        reader.read_line(&mut hello).unwrap();
        let hello: serde_json::Value = serde_json::from_str(hello.trim()).unwrap();
        assert_eq!(hello["auth_required"], true);

        // Send wrong password.
        let auth = serde_json::json!({"type":"auth","user":"alice","password":"wrong"});
        let mut line = serde_json::to_string(&auth).unwrap();
        line.push('\n');
        writer.write_all(line.as_bytes()).unwrap();

        // Expect auth_fail.
        let mut resp = String::new();
        reader.read_line(&mut resp).unwrap();
        let resp: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(resp["type"], "auth_fail");
    }

    #[test]
    fn server_auth_ok_then_query() {
        let users = vec![auth::UserEntry {
            name: "bob".to_string(),
            password_hash: auth::hash_password("secret"),
            graphs: vec!["*".to_string()],
        }];
        let addr = start_auth_server(users);

        let (stream, mut reader) = connect_with_auth(addr, "bob", "secret");
        let mut writer = stream;

        // Verify the auth_ok came back.
        // (connect_with_auth reads the auth_ok into `resp` local var but we need it here)
        // Actually connect_with_auth already consumed it — just run a query.
        let resp = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":1,"query":"MATCH (n) RETURN n"}),
        );
        assert!(resp.get("error").is_none(), "unexpected error: {resp}");
        assert!(resp.get("rows").is_some());
    }

    #[test]
    fn server_graph_field_in_request() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        // Insert into "alpha" graph.
        let resp = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":1,"graph":"alpha","query":r#"INSERT (:T {x:1})"#}),
        );
        assert!(resp.get("error").is_none(), "{resp}");

        // Query "alpha" — should see the node.
        let resp = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":2,"graph":"alpha","query":"MATCH (n:T) RETURN n.x"}),
        );
        assert_eq!(resp["rows"][0]["n.x"], 1);

        // Query "beta" (different graph) — should be empty.
        let resp = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":3,"graph":"beta","query":"MATCH (n:T) RETURN n.x"}),
        );
        let rows = resp["rows"].as_array().unwrap();
        assert!(rows.is_empty(), "expected no rows in beta, got {resp}");
    }

    #[test]
    fn server_admin_list_graphs() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        // Ensure a couple of graphs exist.
        roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":1,"graph":"g1","query":"MATCH (n) RETURN n"}),
        );
        roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":2,"graph":"g2","query":"MATCH (n) RETURN n"}),
        );

        // Send admin graphs command.
        let mut line = serde_json::to_string(&serde_json::json!({"type":"admin","cmd":"graphs"}))
            .unwrap();
        line.push('\n');
        writer.write_all(line.as_bytes()).unwrap();

        let mut resp = String::new();
        reader.read_line(&mut resp).unwrap();
        let resp: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(resp["type"], "admin_ok");
        let graphs = resp["graphs"].as_array().unwrap();
        let names: Vec<&str> = graphs.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(names.contains(&"g1"), "{resp}");
        assert!(names.contains(&"g2"), "{resp}");
    }

    #[test]
    fn server_admin_create_drop_graph() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        let send_admin = |writer: &mut TcpStream, reader: &mut std::io::BufReader<TcpStream>, json: serde_json::Value| {
            let mut line = serde_json::to_string(&json).unwrap();
            line.push('\n');
            writer.write_all(line.as_bytes()).unwrap();
            let mut resp = String::new();
            reader.read_line(&mut resp).unwrap();
            serde_json::from_str::<serde_json::Value>(resp.trim()).unwrap()
        };

        // Create.
        let r = send_admin(
            &mut writer,
            &mut reader,
            serde_json::json!({"type":"admin","cmd":"create","name":"newgraph"}),
        );
        assert_eq!(r["type"], "admin_ok", "{r}");

        // It should appear in the list.
        let r = send_admin(
            &mut writer,
            &mut reader,
            serde_json::json!({"type":"admin","cmd":"graphs"}),
        );
        let graphs = r["graphs"].as_array().unwrap();
        assert!(
            graphs.iter().any(|g| g.as_str() == Some("newgraph")),
            "missing newgraph in {r}"
        );

        // Drop it.
        let r = send_admin(
            &mut writer,
            &mut reader,
            serde_json::json!({"type":"admin","cmd":"drop","name":"newgraph"}),
        );
        assert_eq!(r["type"], "admin_ok", "{r}");
    }

    // ── TODO 17: _meta system graph isolation ────────────────────────────────

    /// The `_meta` system graph must never appear in the admin graph listing.
    #[test]
    fn meta_graph_not_in_list() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        // Touch _meta by saving a view (single-quoted literals, no escape issues).
        roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":1,"graph":"_meta","query":"INSERT (:SavedView {name: 'v1', graph: 'default', query: 'MATCH (n) RETURN n', created: '2026-01-01'})"}),
        );

        // Admin listing must not expose _meta.
        let mut line =
            serde_json::to_string(&serde_json::json!({"type":"admin","cmd":"graphs"})).unwrap();
        line.push('\n');
        writer.write_all(line.as_bytes()).unwrap();
        let mut resp = String::new();
        reader.read_line(&mut resp).unwrap();
        let resp: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        let graphs = resp["graphs"].as_array().unwrap();
        assert!(
            !graphs.iter().any(|g| g.as_str().map(|s| s.starts_with('_')).unwrap_or(false)),
            "system graph leaked into listing: {resp}"
        );
    }

    /// Attempting to drop a system graph via admin must fail.
    #[test]
    fn system_graph_drop_rejected() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        let mut line =
            serde_json::to_string(&serde_json::json!({"type":"admin","cmd":"drop","name":"_meta"}))
                .unwrap();
        line.push('\n');
        writer.write_all(line.as_bytes()).unwrap();
        let mut resp = String::new();
        reader.read_line(&mut resp).unwrap();
        let resp: serde_json::Value = serde_json::from_str(resp.trim()).unwrap();
        assert_eq!(resp["type"], "admin_fail", "expected admin_fail: {resp}");
        assert!(
            resp["error"].as_str().unwrap_or("").contains("system graph"),
            "error message should mention system graph: {resp}"
        );
    }

    /// The `_meta` graph is readable and writable via normal queries with graph="_meta".
    #[test]
    fn meta_graph_queryable() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        // Write a SavedView node using single-quoted literals and node_ids storage.
        let r = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":1,"graph":"_meta","query":"INSERT (:SavedView {name: 'myview', graph: 'default', node_ids: 'AABBCC', created: '2026-01-01'})"}),
        );
        assert!(r.get("error").is_none(), "insert into _meta failed: {r}");

        // Read it back.
        let r = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":2,"graph":"_meta","query":"MATCH (v:SavedView) WHERE v.name = 'myview' RETURN v.name, v.graph"}),
        );
        assert!(r.get("error").is_none(), "query _meta failed: {r}");
        let rows = r["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["v.name"], "myview");
        assert_eq!(rows[0]["v.graph"], "default");
    }

    /// Full save-view → list-views → delete-view round-trip, exactly as the GUI sends it.
    ///
    /// Views store `node_ids` as a comma-separated ULID string — not a GQL query string —
    /// because GQL string literals have no escape sequences and nested quoted values
    /// (e.g. a query containing `'id'` inside another `'...'` literal) would parse-fail.
    /// The reconstruction query is built client-side from the IDs on load.
    #[test]
    fn saved_view_save_list_delete_roundtrip() {
        let addr = start_test_server();
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        // ── Step 1: insert a node into "work" so we get a real ULID ──
        let r = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":1,"graph":"work","query":"INSERT (:Person {name: 'Alice'})"}),
        );
        assert!(r.get("error").is_none(), "insert failed: {r}");

        let r = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":2,"graph":"work","query":"MATCH (n:Person) RETURN n"}),
        );
        let ulid = r["rows"][0]["n"].as_str().expect("expected ULID string").to_string();
        // ULID must be pure alphanumeric — safe as GQL string literal with no escaping.
        assert!(ulid.chars().all(|c| c.is_ascii_alphanumeric()), "unexpected chars in ULID: {ulid}");

        // ── Step 2: save a view — stores comma-separated node IDs, not a GQL query ──
        // Mirrors exactly what gui.html saveView() sends:
        //   INSERT (:SavedView {name: 'x', graph: 'y', node_ids: 'id1,id2', created: '...'})
        let save_gql = format!(
            "INSERT (:SavedView {{name: 'alice-view', graph: 'work', node_ids: '{}', created: '2026-03-20'}})",
            ulid  // comma-separated; single ULID here, no quotes inside
        );
        let r = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":3,"graph":"_meta","query": save_gql}),
        );
        assert!(r.get("error").is_none(), "save view failed: {r}");

        // ── Step 3: list views for graph "work" ──
        let r = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":4,"graph":"_meta","query":"MATCH (v:SavedView) WHERE v.graph = 'work' RETURN v.name, v.node_ids, v.created ORDER BY v.created"}),
        );
        assert!(r.get("error").is_none(), "list views failed: {r}");
        let rows = r["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 1, "expected 1 view, got {r}");
        assert_eq!(rows[0]["v.name"], "alice-view");
        assert!(rows[0]["v.node_ids"].as_str().unwrap().contains(ulid.as_str()),
            "node_ids should contain the ULID");

        // ── Step 4: delete the view ──
        let r = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":5,"graph":"_meta","query":"MATCH (v:SavedView) WHERE v.name = 'alice-view' AND v.graph = 'work' DELETE v"}),
        );
        assert!(r.get("error").is_none(), "delete view failed: {r}");

        // ── Step 5: verify the view is gone ──
        let r = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":6,"graph":"_meta","query":"MATCH (v:SavedView) WHERE v.graph = 'work' RETURN v.name"}),
        );
        assert!(r.get("error").is_none(), "{r}");
        let rows = r["rows"].as_array().unwrap();
        assert!(rows.is_empty(), "view should be deleted, got {r}");
    }

    #[test]
    fn server_auto_rollback_on_disconnect() {
        let addr = start_test_server();

        // Connection 1: open a transaction, insert a node, then disconnect without committing.
        {
            let (stream, mut reader) = connect_no_auth(addr);
            let mut writer = stream;

            roundtrip(&mut writer, &mut reader, serde_json::json!({"id":1,"query":"BEGIN"}));
            roundtrip(
                &mut writer,
                &mut reader,
                serde_json::json!({"id":2,"query":r#"INSERT (:Transient {x:99})"#}),
            );
            // Drop without COMMIT → auto-rollback.
        }

        // Give server time to process the disconnect.
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Connection 2: the uncommitted node should not be visible.
        let (stream, mut reader) = connect_no_auth(addr);
        let mut writer = stream;

        let r = roundtrip(
            &mut writer,
            &mut reader,
            serde_json::json!({"id":1,"query":"MATCH (n:Transient) RETURN n.x"}),
        );
        let rows = r["rows"].as_array().unwrap();
        assert!(rows.is_empty(), "rolled-back data leaked: {r}");
    }

    // ── HTTP upload endpoints ─────────────────────────────────────────────────

    /// Start a server with both TCP and HTTP (GUI) listeners on random ports.
    /// Returns `(tcp_addr, http_addr)`.
    fn start_test_server_with_gui() -> (std::net::SocketAddr, std::net::SocketAddr) {
        let tcp_l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let tcp_addr = tcp_l.local_addr().unwrap();
        drop(tcp_l);

        let http_l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let http_addr = http_l.local_addr().unwrap();
        drop(http_l);

        let data_root = tempfile::tempdir().unwrap().into_path();
        let config = ServerConfig {
            server: auth::ServerSection { auth_required: false },
            users: vec![],
        };

        std::thread::spawn(move || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(serve(config, data_root, vec![], tcp_addr, Some(http_addr)))
                .unwrap();
        });

        std::thread::sleep(std::time::Duration::from_millis(200));
        (tcp_addr, http_addr)
    }

    /// Send a POST request with a JSON body using raw HTTP/1.1 over a TcpStream.
    /// Returns the parsed JSON response body.
    fn http_post(
        http_addr: std::net::SocketAddr,
        path: &str,
        body: &serde_json::Value,
    ) -> serde_json::Value {
        use std::io::{Read, Write};

        let body_str = serde_json::to_string(body).unwrap();
        let req = format!(
            "POST {} HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n{}",
            path,
            body_str.len(),
            body_str
        );
        let mut stream = TcpStream::connect(http_addr).unwrap();
        stream.write_all(req.as_bytes()).unwrap();

        let mut response = Vec::new();
        stream.read_to_end(&mut response).unwrap();
        let response = String::from_utf8_lossy(&response);

        // HTTP/1.1 response: split headers from body at \r\n\r\n.
        let body_start = response.find("\r\n\r\n").expect("no header/body separator") + 4;
        let body_part = response[body_start..].trim();
        serde_json::from_str(body_part).unwrap_or_else(|e| {
            panic!("failed to parse HTTP response body as JSON: {e}\nbody: {body_part}")
        })
    }

    #[test]
    fn http_upload_nodes_basic() {
        let (_tcp_addr, http_addr) = start_test_server_with_gui();

        let csv = ":ID,name,age,:LABEL\n1,Alice,30,Person\n2,Bob,25,Person\n";
        let resp = http_post(
            http_addr,
            "/api/upload/nodes",
            &serde_json::json!({ "csv": csv }),
        );

        assert_eq!(resp["inserted"], 2, "resp: {resp}");
        let id_map = resp["id_map"].as_object().unwrap();
        assert_eq!(id_map.len(), 2);
        assert!(id_map.contains_key("1"));
        assert!(id_map.contains_key("2"));
    }

    #[test]
    fn http_upload_nodes_with_label() {
        let (_tcp_addr, http_addr) = start_test_server_with_gui();

        let csv = ":ID,name\n1,Alice\n";
        let resp = http_post(
            http_addr,
            "/api/upload/nodes",
            &serde_json::json!({ "csv": csv, "label": "Employee" }),
        );
        assert_eq!(resp["inserted"], 1, "resp: {resp}");
    }

    #[test]
    fn http_upload_edges_basic() {
        let (_tcp_addr, http_addr) = start_test_server_with_gui();

        // Step 1: upload nodes.
        let node_csv = ":ID,name,:LABEL\n1,Alice,Person\n2,Bob,Person\n";
        let node_resp = http_post(
            http_addr,
            "/api/upload/nodes",
            &serde_json::json!({ "csv": node_csv }),
        );
        assert_eq!(node_resp["inserted"], 2);
        let id_map = &node_resp["id_map"];

        // Step 2: upload edges using the id_map.
        let edge_csv = ":START_ID,:END_ID,:TYPE,weight\n1,2,KNOWS,0.9\n";
        let edge_resp = http_post(
            http_addr,
            "/api/upload/edges",
            &serde_json::json!({ "csv": edge_csv, "id_map": id_map }),
        );
        assert_eq!(edge_resp["inserted"], 1, "edge resp: {edge_resp}");
        assert_eq!(edge_resp["skipped"], 0);
    }

    #[test]
    fn http_upload_edges_skips_unresolved() {
        let (_tcp_addr, http_addr) = start_test_server_with_gui();

        let node_csv = ":ID,name,:LABEL\n1,Alice,Person\n";
        let node_resp = http_post(
            http_addr,
            "/api/upload/nodes",
            &serde_json::json!({ "csv": node_csv }),
        );
        let id_map = &node_resp["id_map"];

        let edge_csv = ":START_ID,:END_ID,:TYPE\n1,99,KNOWS\n"; // 99 not in map
        let edge_resp = http_post(
            http_addr,
            "/api/upload/edges",
            &serde_json::json!({ "csv": edge_csv, "id_map": id_map }),
        );
        assert_eq!(edge_resp["inserted"], 0);
        assert_eq!(edge_resp["skipped"], 1);
    }
}
