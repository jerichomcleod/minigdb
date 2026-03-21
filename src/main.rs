//! Interactive REPL and CLI entry point for minigdb.
//!
//! # CLI overview
//!
//! ```text
//! minigdb                        # Start the interactive REPL
//! minigdb serve [options]        # Start the TCP + HTTP server
//! minigdb adduser <name>         # Add a server user (prompts for password)
//! minigdb passwd  <name>         # Change a server user's password
//! minigdb users                  # List all server users
//! ```
//!
//! All CLI subcommands are gated on the `server` Cargo feature.  The REPL
//! itself is gated on the `repl` feature (enabled by default).
//!
//! # REPL quick reference
//!
//! The REPL reads GQL from stdin using rustyline (readline-compatible).
//! Multi-line input is supported: the prompt changes from `>` to `->` while
//! accumulating a statement that has not yet been detected as complete.
//!
//! ## Graph management meta-commands (always single-line, prefix `:`)
//!
//! | Command               | Effect                                         |
//! |-----------------------|------------------------------------------------|
//! | `:graphs`             | List all named graphs; mark the active one     |
//! | `:create <name>`      | Create a new graph and switch to it            |
//! | `:use <name>`         | Switch to an existing graph                    |
//! | `:drop <name>`        | Permanently delete a graph (must not be active)|
//! | `:clear`              | Delete all nodes and edges in O(1)             |
//! | `:checkpoint`         | Flush WAL to RocksDB SST files immediately     |
//! | `:quit` / `:exit` / `:q` | Exit the REPL cleanly                       |
//!
//! ## Transaction control (GQL keywords)
//!
//! | Command    | Effect                                                  |
//! |------------|---------------------------------------------------------|
//! | `BEGIN`    | Open an explicit transaction; prompt changes to `txn>`  |
//! | `COMMIT`   | Write all buffered ops to RocksDB atomically            |
//! | `ROLLBACK` | Discard all buffered ops (O(1))                         |
//!
//! Graph management meta-commands are **rejected** while a transaction is open
//! to prevent accidentally switching graphs mid-transaction.
//!
//! # Named graph storage layout
//!
//! All graphs live under `<data_root>/graphs/`:
//! - Linux:   `~/.local/share/minigdb/graphs/`
//! - macOS:   `~/Library/Application Support/minigdb/graphs/`
//! - Windows: `%APPDATA%\minigdb\graphs\`
//!
//! Each graph occupies its own subdirectory, which RocksDB uses as its data
//! directory.  Switching graphs closes the current RocksDB instance (via
//! [`StorageManager`] drop) and opens another.
//!
//! # Compile-time feature flags
//!
//! | Feature  | Adds                                    |
//! |----------|-----------------------------------------|
//! | `repl`   | rustyline REPL, all helper functions    |
//! | `server` | `serve`, `adduser`, `passwd`, `users`  |
//! | `gui`    | Axum HTTP server + web GUI              |

#[cfg(feature = "repl")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use minigdb::query_capturing;
    use rustyline::{error::ReadlineError, DefaultEditor};

    env_logger::init();

    // ── Subcommand dispatch ───────────────────────────────────────────────────
    //
    // When compiled with the `server` feature, check argv[1] before dropping
    // into the REPL.  Each subcommand handler returns directly, so the REPL
    // initialisation below is never reached for server commands.
    #[cfg(feature = "server")]
    {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match args.first().map(|s| s.as_str()) {
            Some("serve")   => return serve_main(&args[1..]),
            Some("adduser") => return adduser_main(&args[1..]),
            Some("passwd")  => return passwd_main(&args[1..]),
            Some("users")   => return users_main(),
            _ => {}
        }
    }

    // ── Data directory resolution ─────────────────────────────────────────────
    //
    // Resolve the user data directory:
    //   Linux:   ~/.local/share/minigdb/
    //   macOS:   ~/Library/Application Support/minigdb/
    //   Windows: %AppData%\minigdb\
    // Falls back to the current working directory if the platform dir is unavailable.
    let data_root = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("minigdb");

    let graphs_dir = data_root.join("graphs");
    std::fs::create_dir_all(&graphs_dir)?;

    // Open (or create) the default graph on startup.
    let mut current_graph_name = String::from("default");
    let (mut storage, mut graph) = open_graph(&graphs_dir, &current_graph_name)
        .map_err(|e| format!("Failed to open default graph: {e}"))?;

    // Monotonically increasing counter for WAL frame / operation IDs.
    let mut next_txn_id: u64 = 0;

    println!("minigdb v{}", env!("CARGO_PKG_VERSION"));
    println!("Data: {}", data_root.display());
    println!("Type GQL queries (end with ';'), or :quit / :exit to exit.");
    println!("Transaction commands: BEGIN, COMMIT, ROLLBACK");
    println!("Graph commands: :graphs  :create <name>  :use <name>  :drop <name>  :clear");
    println!();

    // ── Rustyline setup ───────────────────────────────────────────────────────

    let mut rl = DefaultEditor::new()?;

    // History is persisted across sessions in the data root directory.
    let history_path = data_root.join(".minigdb_history");
    let history_path = history_path.to_str().unwrap_or(".minigdb_history");
    let _ = rl.load_history(history_path);

    // `pending` accumulates lines for multi-line input until the statement is complete.
    let mut pending = String::new();

    // ── Main REPL loop ────────────────────────────────────────────────────────
    loop {
        // The prompt encodes both the transaction state and the current graph name.
        //   `graph>`     — normal, at the start of a new statement
        //   `graph->`    — normal, mid-statement (continuation)
        //   `txn(graph)>`  — inside BEGIN…COMMIT, start of statement
        //   `txn(graph)->`  — inside BEGIN…COMMIT, continuation
        let prompt = match (graph.is_in_transaction(), pending.is_empty()) {
            (true, true)   => format!("txn({})> ", current_graph_name),
            (true, false)  => format!("txn({})-> ", current_graph_name),
            (false, true)  => format!("{}> ", current_graph_name),
            (false, false) => format!("{}-> ", current_graph_name),
        };

        let readline = rl.readline(&prompt);
        match readline {
            Ok(line) => {
                let trimmed = line.trim();

                // ── Exit commands ─────────────────────────────────────────────
                if trimmed.eq_ignore_ascii_case(":quit")
                    || trimmed.eq_ignore_ascii_case(":q")
                    || trimmed.eq_ignore_ascii_case(":exit")
                    || trimmed.eq_ignore_ascii_case("exit")
                    || trimmed.eq_ignore_ascii_case("quit")
                {
                    println!("Goodbye.");
                    break;
                }

                // ── Manual checkpoint ─────────────────────────────────────────
                if trimmed.eq_ignore_ascii_case(":checkpoint") {
                    match storage.checkpoint(&graph) {
                        Ok(_) => println!("Checkpoint written."),
                        Err(e) => eprintln!("Error: {e}"),
                    }
                    rl.add_history_entry(trimmed)?;
                    continue;
                }

                // ── Graph management meta-commands ────────────────────────────
                //
                // Any line starting with `:` (other than the special cases
                // handled above) is a graph management command.  These are
                // rejected inside an active transaction to prevent accidentally
                // switching graphs mid-transaction.
                let is_graph_cmd = trimmed.starts_with(':')
                    && !trimmed.eq_ignore_ascii_case(":quit")
                    && !trimmed.eq_ignore_ascii_case(":q")
                    && !trimmed.eq_ignore_ascii_case(":exit")
                    && !trimmed.eq_ignore_ascii_case("exit")
                    && !trimmed.eq_ignore_ascii_case("quit")
                    && !trimmed.eq_ignore_ascii_case(":checkpoint");

                if is_graph_cmd {
                    let cmd = trimmed.trim_end_matches(';').trim();
                    if graph.is_in_transaction() {
                        eprintln!("Error: graph management commands are not allowed inside a transaction (COMMIT or ROLLBACK first)");
                        rl.add_history_entry(trimmed)?;
                        continue;
                    }

                    if cmd.eq_ignore_ascii_case(":graphs") {
                        // List all subdirectories of graphs_dir, alphabetically sorted.
                        let names = list_graph_names(&graphs_dir);
                        if names.is_empty() {
                            println!("(no graphs)");
                        } else {
                            for name in &names {
                                if name == &current_graph_name {
                                    println!("  {} (active)", name);
                                } else {
                                    println!("  {}", name);
                                }
                            }
                        }
                    } else if let Some(rest) = cmd.strip_prefix(":create ").or_else(|| cmd.strip_prefix(":CREATE ")) {
                        let name = rest.trim();
                        if !validate_graph_name(name) {
                            eprintln!("Error: invalid graph name '{}' (use alphanumeric, '_', '-', max 64 chars)", name);
                        } else {
                            // Checkpoint current graph before switching so no
                            // data is lost if the new graph open fails.
                            if let Err(e) = storage.checkpoint(&graph) {
                                eprintln!("Warning: could not checkpoint current graph: {e}");
                            }
                            match open_graph(&graphs_dir, name) {
                                Ok((new_storage, new_graph)) => {
                                    storage = new_storage;
                                    graph = new_graph;
                                    current_graph_name = name.to_string();
                                    println!("Created and switched to graph '{}'.", name);
                                }
                                Err(e) => eprintln!("Error creating graph '{}': {}", name, e),
                            }
                        }
                    } else if let Some(rest) = cmd.strip_prefix(":use ").or_else(|| cmd.strip_prefix(":USE ")) {
                        let name = rest.trim();
                        if !validate_graph_name(name) {
                            eprintln!("Error: invalid graph name '{}'", name);
                        } else {
                            let graph_path = graphs_dir.join(name);
                            if !graph_path.exists() {
                                eprintln!("Error: graph '{}' does not exist (use :create to create it)", name);
                            } else {
                                // Checkpoint current graph before switching.
                                if let Err(e) = storage.checkpoint(&graph) {
                                    eprintln!("Warning: could not checkpoint current graph: {e}");
                                }
                                match open_graph(&graphs_dir, name) {
                                    Ok((new_storage, new_graph)) => {
                                        storage = new_storage;
                                        graph = new_graph;
                                        current_graph_name = name.to_string();
                                        println!("Switched to graph '{}'.", name);
                                    }
                                    Err(e) => eprintln!("Error opening graph '{}': {}", name, e),
                                }
                            }
                        }
                    } else if let Some(rest) = cmd.strip_prefix(":drop ").or_else(|| cmd.strip_prefix(":DROP ")) {
                        let name = rest.trim();
                        if !validate_graph_name(name) {
                            eprintln!("Error: invalid graph name '{}'", name);
                        } else if name == current_graph_name {
                            // Dropping the active graph would leave the REPL without
                            // a graph; require the user to switch away first.
                            eprintln!("Error: cannot drop the active graph (switch to another graph first)");
                        } else {
                            let graph_path = graphs_dir.join(name);
                            if !graph_path.exists() {
                                eprintln!("Error: graph '{}' does not exist", name);
                            } else {
                                match std::fs::remove_dir_all(&graph_path) {
                                    Ok(_) => println!("Dropped graph '{}'.", name),
                                    Err(e) => eprintln!("Error dropping graph '{}': {}", name, e),
                                }
                            }
                        }
                    } else if cmd.eq_ignore_ascii_case(":clear") {
                        if graph.is_in_transaction() {
                            eprintln!("Error: cannot :clear inside a transaction.");
                        } else {
                            match graph.clear() {
                                Ok(_)  => println!("Graph cleared."),
                                Err(e) => eprintln!("Error: {e}"),
                            }
                        }
                    } else {
                        eprintln!("Unknown command: '{}'. Type :graphs, :create <name>, :use <name>, :drop <name>, :clear.", cmd);
                    }

                    rl.add_history_entry(trimmed)?;
                    continue;
                }
                // ─────────────────────────────────────────────────────────────

                // ── Transaction control ───────────────────────────────────────
                //
                // BEGIN / COMMIT / ROLLBACK are handled here rather than being
                // passed to the GQL executor so the REPL can update its prompt
                // state immediately and avoid writing these control keywords to
                // the WAL as GQL statements.
                let upper = trimmed.to_uppercase();
                let bare = upper.trim_end_matches(';').trim();

                if bare == "BEGIN" {
                    match graph.begin_transaction() {
                        Ok(_) => println!("Transaction started."),
                        Err(e) => eprintln!("Error: {e}"),
                    }
                    rl.add_history_entry(trimmed)?;
                    continue;
                }

                if bare == "COMMIT" {
                    match graph.commit_transaction() {
                        Ok(_) => println!("Transaction committed."),
                        Err(e) => eprintln!("Error: {e}"),
                    }
                    rl.add_history_entry(trimmed)?;
                    continue;
                }

                if bare == "ROLLBACK" {
                    match graph.rollback_transaction() {
                        Ok(_) => println!("Transaction rolled back."),
                        Err(e) => eprintln!("Error: {e}"),
                    }
                    rl.add_history_entry(trimmed)?;
                    continue;
                }
                // ─────────────────────────────────────────────────────────────

                // Add non-empty lines to rustyline history for up-arrow recall.
                if !trimmed.is_empty() {
                    rl.add_history_entry(trimmed)?;
                }

                // Accumulate lines into `pending` until the statement is complete.
                pending.push_str(trimmed);
                pending.push(' ');

                // A statement is ready to execute when it ends with `;` (explicit
                // terminator) or when `is_complete_statement` heuristically detects
                // that the accumulated text forms a full GQL statement.  The
                // trailing-comma guard prevents premature execution of multi-value
                // INSERT lists that span lines.
                let ready = pending.trim().ends_with(';')
                    || (!pending.trim().is_empty()
                        && !pending.trim().ends_with(',')
                        && is_complete_statement(pending.trim()));

                if ready {
                    let input = pending.trim().to_string();
                    pending.clear();

                    match query_capturing(&input, &mut graph, &mut next_txn_id) {
                        Ok((rows, _ops)) => {
                            // Graph handles buffering/auto-commit internally (R3).

                            if rows.is_empty() {
                                println!("(no results)");
                            } else if rows.len() == 1
                                && rows[0].len() == 1
                                && rows[0].contains_key("result")
                            {
                                // Write-operation summary rows have a single "result"
                                // key containing a human-readable string (e.g.
                                // "Inserted 1 node(s)").  Print these as plain lines
                                // rather than table-formatted output.
                                if let Some(minigdb::Value::String(msg)) = rows[0].get("result") {
                                    println!("{msg}");
                                }
                            } else {
                                print_rows(&rows);
                            }
                        }
                        Err(e) => eprintln!("Error: {e}"),
                    }
                }
            }

            // Ctrl-C clears the pending buffer without exiting, mirroring
            // standard shell behaviour.
            Err(ReadlineError::Interrupted) => {
                pending.clear();
                println!("(interrupted)");
            }

            // Ctrl-D (EOF) exits cleanly.
            Err(ReadlineError::Eof) => {
                println!("Goodbye.");
                break;
            }

            Err(e) => {
                eprintln!("Readline error: {e}");
                break;
            }
        }
    }

    // Persist command history across sessions.
    let _ = rl.save_history(history_path);

    // Final checkpoint on clean exit: flush all WAL entries to RocksDB SST files.
    if let Err(e) = storage.checkpoint(&graph) {
        eprintln!("Warning: could not write final checkpoint: {e}");
    }

    Ok(())
}

// ── Graph helpers ─────────────────────────────────────────────────────────────

/// Open (or create) a named graph rooted at `graphs_dir/<name>/`.
///
/// Creates the directory if it does not exist, then delegates to
/// [`minigdb::StorageManager::open`] which initialises RocksDB and replays
/// any outstanding WAL entries.
///
/// Returns a `(StorageManager, Graph)` pair.  The `StorageManager` owns the
/// checkpoint functionality; the `Graph` owns the in-memory state and the
/// RocksDB handle.  Both must be kept alive for the graph to remain usable.
#[cfg(feature = "repl")]
fn open_graph(
    graphs_dir: &std::path::Path,
    name: &str,
) -> Result<(minigdb::StorageManager, minigdb::Graph), minigdb::DbError> {
    let graph_path = graphs_dir.join(name);
    std::fs::create_dir_all(&graph_path)
        .map_err(minigdb::DbError::Storage)?;
    minigdb::StorageManager::open(&graph_path)
}

/// List all graph names (subdirectories of `graphs_dir`), sorted alphabetically.
///
/// Plain files in `graphs_dir` are silently ignored.  Returns an empty `Vec`
/// if `graphs_dir` cannot be read (e.g. does not exist yet).
#[cfg(any(feature = "repl", test))]
fn list_graph_names(graphs_dir: &std::path::Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(graphs_dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| {
            let e = e.ok()?;
            // Only include actual directories — filter out regular files.
            // Names starting with '_' are reserved system graphs; exclude them.
            if e.file_type().ok()?.is_dir() {
                let name = e.file_name().into_string().ok()?;
                if name.starts_with('_') { None } else { Some(name) }
            } else {
                None
            }
        })
        .collect();
    names.sort();
    names
}

/// Return `true` if `name` is a valid graph (or username) identifier.
///
/// Rules (identical for both graphs and server users):
/// - Non-empty
/// - At most 64 characters
/// - Does **not** start with `_` (reserved for system graphs such as `_meta`)
/// - Every character is ASCII alphanumeric, `_`, or `-`
///
/// This deliberately excludes `.`, `/`, spaces, and other characters that could
/// be interpreted as path components, preventing directory traversal attacks
/// when the name is joined to `graphs_dir`.
#[cfg(any(feature = "repl", test))]
fn validate_graph_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('_')   // '_' prefix reserved for system graphs
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

// ── Statement completeness detection ─────────────────────────────────────────

/// Heuristically determine whether the accumulated input `s` forms a complete
/// GQL statement that can be sent to the executor.
///
/// This is used in the REPL's multi-line input loop: when the user does not
/// end a line with `;`, the loop calls this function to decide whether to
/// execute immediately or wait for more input.
///
/// ## Detection rules (evaluated in order)
///
/// 1. **Colon meta-commands** (`:`…) — always single-line; return `true`.
/// 2. **`RETURN`** anywhere in the statement — covers `MATCH…RETURN`,
///    `UNWIND…RETURN`, `OPTIONAL MATCH…RETURN`, `WITH…RETURN`, etc.
/// 3. **Standalone write statements** — `INSERT`, `DELETE`, `DETACH` are
///    complete by themselves (no trailing clause needed).
/// 4. **`UNWIND`** — complete only if `RETURN` is also present (already
///    covered by rule 2, but kept explicit for clarity).
/// 5. **`OPTIONAL MATCH`** — same as `UNWIND`.
/// 6. **`SHOW`** — always complete (e.g. `SHOW INDEXES`).
/// 7. **`CREATE INDEX` / `DROP INDEX`** — complete when the closing `)` of
///    the property specifier has been typed.
/// 8. **`CALL`** — complete when the closing `)` of the argument list is present.
/// 9. **`MATCH`** — complete when a mutating clause (`SET`, `REMOVE`, `DELETE`,
///    `DETACH`, `INSERT`) appears after the pattern.
///
/// Returns `false` for anything that does not match, causing the REPL to
/// prompt for more input on the next iteration.
fn is_complete_statement(s: &str) -> bool {
    // Graph management meta-commands are always single-line.
    if s.starts_with(':') {
        return true;
    }

    let upper = s.to_uppercase();

    // MATCH...RETURN is always complete at RETURN.
    if upper.contains("RETURN") {
        return true;
    }

    // Standalone write statements.
    if upper.starts_with("INSERT") || upper.starts_with("DELETE") || upper.starts_with("DETACH") {
        return true;
    }

    // UNWIND ... RETURN or UNWIND ... INSERT.
    if upper.starts_with("UNWIND") {
        return upper.contains("RETURN") || upper.contains("INSERT");
    }

    // OPTIONAL MATCH ... RETURN.
    if upper.starts_with("OPTIONAL") {
        return upper.contains("RETURN");
    }

    // Index management.
    if upper.starts_with("SHOW") {
        return true;
    }
    if upper.starts_with("CREATE INDEX") || upper.starts_with("DROP INDEX") {
        // Complete once we see the closing paren: CREATE INDEX ON :Label(prop)
        return upper.contains(')');
    }

    // Constraint management.
    if upper.starts_with("CREATE CONSTRAINT") || upper.starts_with("DROP CONSTRAINT") {
        // Complete once the closing paren is present: ... ON :Label(prop)
        return upper.contains(')');
    }
    if upper.starts_with("SHOW CONSTRAINTS") {
        return true;
    }

    // TRUNCATE: single keyword, always complete.
    if upper.starts_with("TRUNCATE") {
        return true;
    }

    // LOAD CSV statement: always single-line.
    if upper.starts_with("LOAD") {
        return true;
    }

    // CALL statement: complete once the closing paren of the argument list is present.
    if upper.starts_with("CALL") {
        return upper.contains(')');
    }

    // SET, REMOVE, DELETE, and INSERT can all follow MATCH.
    if upper.starts_with("MATCH") {
        return upper.contains(" SET ") || upper.contains(" REMOVE ")
            || upper.contains(" DELETE ") || upper.contains(" DETACH ")
            || upper.contains(" INSERT ");
    }

    false
}

// ── Result display ────────────────────────────────────────────────────────────

/// Print a slice of result rows as an ASCII table to stdout.
///
/// Columns are sorted alphabetically by name.  Column widths are computed as
/// the maximum of the header width and the widest formatted value in that
/// column.  An empty slice is a no-op.
fn print_rows(rows: &[minigdb::Row]) {
    if rows.is_empty() {
        return;
    }

    // Sort column names for deterministic output ordering.
    let mut cols: Vec<String> = rows[0].keys().cloned().collect();
    cols.sort();

    // Compute the display width of each column.
    let col_widths: Vec<usize> = cols
        .iter()
        .map(|c| {
            let header_w = c.len();
            let max_val_w = rows
                .iter()
                .map(|r| r.get(c).map(|v| format!("{v}").len()).unwrap_or(4))
                .max()
                .unwrap_or(4);
            header_w.max(max_val_w)
        })
        .collect();

    print_separator(&col_widths);
    print!("| ");
    for (i, col) in cols.iter().enumerate() {
        print!("{:width$} | ", col, width = col_widths[i]);
    }
    println!();
    print_separator(&col_widths);

    for row in rows {
        print!("| ");
        for (i, col) in cols.iter().enumerate() {
            let val = row
                .get(col)
                .map(|v| format!("{v}"))
                .unwrap_or_else(|| "null".to_string());
            print!("{:width$} | ", val, width = col_widths[i]);
        }
        println!();
    }

    print_separator(&col_widths);
    println!("{} row(s)", rows.len());
}

/// Print a `+---+---+` row separator matching the given column widths.
fn print_separator(widths: &[usize]) {
    print!("+");
    for w in widths {
        // Each cell contributes `width + 2` dashes (one space padding each side).
        print!("{}", "-".repeat(w + 2));
        print!("+");
    }
    println!();
}

// ── Server subcommand handlers ────────────────────────────────────────────────

/// Start the TCP (and optionally HTTP) server.
///
/// Usage: `minigdb serve [--host <addr>] [--port <port>] [--no-auth]
///                       [--gui-port <port>] [--no-gui]`
///
/// Defaults:
/// - host: `127.0.0.1`
/// - port: `7474` (TCP JSON protocol)
/// - gui-port: `7475` (HTTP + web GUI, disabled with `--no-gui`)
///
/// Blocks the calling thread in a Tokio `multi_thread` runtime until the
/// process receives SIGINT (Ctrl-C), at which point the server checkpoints
/// all open graphs before exiting.
#[cfg(all(feature = "repl", feature = "server"))]
fn serve_main(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut host = String::from("127.0.0.1");
    let mut port: u16 = 7474;
    let mut no_auth = false;
    // GUI is enabled by default on port 7475; pass `--no-gui` to disable.
    let mut gui_port: Option<u16> = Some(7475);

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => {
                i += 1;
                host = args.get(i).cloned().ok_or("--host requires a value")?;
            }
            "--port" | "-p" => {
                i += 1;
                port = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or("--port requires a numeric value")?;
            }
            "--no-auth" => no_auth = true,
            "--no-gui" => gui_port = None,
            "--gui-port" => {
                i += 1;
                gui_port = Some(
                    args.get(i)
                        .and_then(|s| s.parse().ok())
                        .ok_or("--gui-port requires a numeric value")?,
                );
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: minigdb serve [--host <addr>] [--port <port>] [--no-auth] [--gui-port <port>] [--no-gui]\n\
                     \n\
                     Options:\n\
                       --host           Bind address        (default: 127.0.0.1)\n\
                       --port, -p       TCP port            (default: 7474)\n\
                       --no-auth        Disable authentication\n\
                       --gui-port       HTTP GUI port       (default: 7475)\n\
                       --no-gui         Disable the web GUI\n\
                     \n\
                     User management:\n\
                       minigdb adduser <name>   Add a new user\n\
                       minigdb passwd <name>    Change a user's password\n\
                       minigdb users            List all users\n\
                     \n\
                     Protocol v2: newline-delimited JSON over TCP.\n\
                     Server sends hello on connect, then client may auth, then queries.\n\
                     Request:  {{\"id\":1,\"query\":\"MATCH (n) RETURN n\"}}\n\
                     Response: {{\"id\":1,\"rows\":[...],\"elapsed_ms\":0.3}}"
                );
                return Ok(());
            }
            other => {
                return Err(format!("Unknown argument: {other}  (try --help)").into());
            }
        }
        i += 1;
    }

    let data_root = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("minigdb");
    let graphs_dir = data_root.join("graphs");
    std::fs::create_dir_all(&graphs_dir)?;

    let mut config = minigdb::server::auth::ServerConfig::load(&data_root);
    // `--no-auth` overrides whatever is set in server.toml at runtime.
    if no_auth {
        config.server.auth_required = false;
    }

    let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
    // `gui_addr` is `None` when `--no-gui` was passed, disabling the HTTP server.
    let gui_addr: Option<std::net::SocketAddr> = gui_port
        .map(|p| format!("{host}:{p}").parse())
        .transpose()?;

    // Build a multi-threaded Tokio runtime and block on the async server future.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(minigdb::server::serve(config, graphs_dir, addr, gui_addr))?;

    Ok(())
}

/// Add a new user to `server.toml`.
///
/// Usage: `minigdb adduser <name>`
///
/// Prompts interactively for a password (twice for confirmation) and for a
/// comma-separated list of allowed graph names (`*` means all graphs).
/// Username validation reuses `validate_graph_name` — same character rules apply.
#[cfg(all(feature = "repl", feature = "server"))]
fn adduser_main(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let name = args.first().ok_or("Usage: minigdb adduser <name>")?;
    // Reuse graph name validation: same character rules for usernames.
    if !validate_graph_name(name) {
        return Err(format!("Invalid username '{name}' (alphanumeric, '_', '-', max 64 chars)").into());
    }

    let data_root = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("minigdb");
    std::fs::create_dir_all(&data_root)?;

    let mut config = minigdb::server::auth::ServerConfig::load(&data_root);
    if config.find_user(name).is_some() {
        return Err(format!("User '{name}' already exists (use 'passwd' to change password)").into());
    }

    // rpassword hides the typed password from the terminal.
    let password = rpassword::prompt_password(format!("Password for '{name}': "))?;
    let confirm  = rpassword::prompt_password("Confirm password: ")?;
    if password != confirm {
        return Err("Passwords do not match.".into());
    }

    print!("Allowed graphs (comma-separated, or * for all) [*]: ");
    std::io::Write::flush(&mut std::io::stdout())?;
    let mut graphs_input = String::new();
    std::io::BufRead::read_line(&mut std::io::stdin().lock(), &mut graphs_input)?;
    let graphs_input = graphs_input.trim();
    // Empty input or a bare `*` both mean "all graphs".
    let graphs: Vec<String> = if graphs_input.is_empty() || graphs_input == "*" {
        vec!["*".to_string()]
    } else {
        graphs_input.split(',').map(|s| s.trim().to_string()).collect()
    };

    config.users.push(minigdb::server::auth::UserEntry {
        name: name.clone(),
        password_hash: minigdb::server::auth::hash_password(&password),
        graphs,
    });
    config.save(&data_root)?;
    println!("User '{name}' added.");
    Ok(())
}

/// Change a user's password in `server.toml`.
///
/// Usage: `minigdb passwd <name>`
///
/// Prompts interactively for a new password (twice for confirmation).
/// The password is stored as a SHA-256 hash via [`minigdb::server::auth::hash_password`].
#[cfg(all(feature = "repl", feature = "server"))]
fn passwd_main(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let name = args.first().ok_or("Usage: minigdb passwd <name>")?;

    let data_root = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("minigdb");

    let mut config = minigdb::server::auth::ServerConfig::load(&data_root);
    if config.find_user(name).is_none() {
        return Err(format!("User '{name}' does not exist.").into());
    }

    let password = rpassword::prompt_password(format!("New password for '{name}': "))?;
    let confirm  = rpassword::prompt_password("Confirm new password: ")?;
    if password != confirm {
        return Err("Passwords do not match.".into());
    }

    if let Some(entry) = config.find_user_mut(name) {
        entry.password_hash = minigdb::server::auth::hash_password(&password);
    }
    config.save(&data_root)?;
    println!("Password updated for '{name}'.");
    Ok(())
}

/// List all users defined in `server.toml`.
///
/// Usage: `minigdb users`
///
/// Prints a two-column table: username and allowed graphs.  If no users are
/// configured, prints a hint to run `minigdb adduser`.
#[cfg(all(feature = "repl", feature = "server"))]
fn users_main() -> Result<(), Box<dyn std::error::Error>> {
    let data_root = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("minigdb");

    let config = minigdb::server::auth::ServerConfig::load(&data_root);
    if config.users.is_empty() {
        println!("No users configured.");
        println!("Run 'minigdb adduser <name>' to add one.");
    } else {
        println!("{:<24} {}", "User", "Graphs");
        println!("{}", "-".repeat(48));
        for u in &config.users {
            println!("{:<24} {}", u.name, u.graphs.join(", "));
        }
    }
    Ok(())
}

// ── Stub for builds without the `repl` feature ───────────────────────────────

/// Fallback `main` for builds that do not include the `repl` feature.
///
/// Exits immediately with a non-zero status and an explanatory message.
#[cfg(not(feature = "repl"))]
fn main() {
    eprintln!("Built without 'repl' feature.");
    std::process::exit(1);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // ── validate_graph_name ──────────────────────────────────────────────────

    #[test]
    fn validate_graph_name_accepts_simple_names() {
        assert!(validate_graph_name("default"));
        assert!(validate_graph_name("myproject"));
        assert!(validate_graph_name("my-project"));
        assert!(validate_graph_name("my_project"));
        assert!(validate_graph_name("Graph1"));
        assert!(validate_graph_name("a"));
    }

    #[test]
    fn validate_graph_name_rejects_empty() {
        assert!(!validate_graph_name(""));
    }

    #[test]
    fn validate_graph_name_rejects_too_long() {
        let long = "a".repeat(65);
        assert!(!validate_graph_name(&long));
        let exact = "a".repeat(64);
        assert!(validate_graph_name(&exact));
    }

    #[test]
    fn validate_graph_name_rejects_bad_chars() {
        assert!(!validate_graph_name("my graph")); // space
        assert!(!validate_graph_name("my/graph")); // slash
        assert!(!validate_graph_name("my.graph")); // dot
        assert!(!validate_graph_name("my:graph")); // colon
        assert!(!validate_graph_name("../evil")); // path traversal
    }

    // ── list_graph_names ────────────────────────────────────────────────────

    #[test]
    fn list_graph_names_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let names = list_graph_names(dir.path());
        assert!(names.is_empty());
    }

    #[test]
    fn list_graph_names_lists_subdirs_sorted() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("zebra")).unwrap();
        fs::create_dir(dir.path().join("alpha")).unwrap();
        fs::create_dir(dir.path().join("beta")).unwrap();
        // A file should not appear in the list.
        fs::write(dir.path().join("notadir.txt"), b"").unwrap();

        let names = list_graph_names(dir.path());
        assert_eq!(names, vec!["alpha", "beta", "zebra"]);
    }

    #[test]
    fn list_graph_names_ignores_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("myfile"), b"").unwrap();
        let names = list_graph_names(dir.path());
        assert!(names.is_empty());
    }

    // ── validate_graph_name: system graph prefix ────────────────────────────

    #[test]
    fn validate_graph_name_rejects_underscore_prefix() {
        // Names starting with '_' are reserved for system graphs.
        assert!(!validate_graph_name("_meta"));
        assert!(!validate_graph_name("_internal"));
        assert!(!validate_graph_name("_"));
        // Mid-name underscores are still allowed.
        assert!(validate_graph_name("my_graph"));
        assert!(validate_graph_name("a_b_c"));
    }

    // ── list_graph_names: system graph filtering ─────────────────────────────

    #[test]
    fn list_graph_names_filters_system_graphs() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("default")).unwrap();
        fs::create_dir(dir.path().join("analytics")).unwrap();
        // System graph — must be hidden from user-facing listing.
        fs::create_dir(dir.path().join("_meta")).unwrap();
        fs::create_dir(dir.path().join("_internal")).unwrap();

        let names = list_graph_names(dir.path());
        assert_eq!(names, vec!["analytics", "default"]);
        assert!(!names.iter().any(|n| n.starts_with('_')));
    }

    // ── is_complete_statement (colon meta-commands) ─────────────────────────

    #[test]
    fn is_complete_for_colon_commands() {
        assert!(is_complete_statement(":graphs"));
        assert!(is_complete_statement(":create myproject"));
        assert!(is_complete_statement(":use default"));
        assert!(is_complete_statement(":drop analytics"));
        assert!(is_complete_statement(":checkpoint"));
        assert!(is_complete_statement(":quit"));
    }

    #[test]
    fn is_complete_for_load_csv() {
        assert!(is_complete_statement("LOAD CSV NODES FROM 'people.csv'"));
        assert!(is_complete_statement("LOAD CSV NODES FROM 'people.csv' LABEL Person"));
        assert!(is_complete_statement("LOAD CSV EDGES FROM 'knows.csv' LABEL KNOWS"));
        // Incomplete / unrelated statements should still be false.
        assert!(!is_complete_statement("MATCH (n:Person)"));
    }
}
