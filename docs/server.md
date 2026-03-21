# Network Server

[← Back to README](../README.md) · [Query Language](query-language.md) · [Algorithms](algorithms.md) · [Python API](python.md)

minigdb includes a multi-graph TCP server with authentication, per-connection transaction isolation, and an embedded web GUI.

---

## Starting the server

```bash
# Start with auth enabled (reads <data_dir>/server.toml)
cargo run -- serve

# Start without auth (open access — for development/testing)
cargo run -- serve --no-auth

# Custom host/port
cargo run -- serve --host 0.0.0.0 --port 7474 --gui-port 8080

# Flags:
#   --host <addr>       bind address for TCP API    (default: 127.0.0.1)
#   --port <port>       TCP API port                (default: 7474)
#   --gui-port <port>   HTTP web GUI port           (default: 7475)
#   --no-auth           disable authentication
#   --no-gui            disable the web GUI
```

The server checkpoints all open graphs and exits cleanly on `Ctrl-C`.

---

## User management

```bash
cargo run -- adduser alice    # add a user (prompts for password + graph access)
cargo run -- passwd alice     # change a password
cargo run -- users            # list all users
```

Users are stored in `<data_dir>/server.toml`:

```toml
[server]
auth_required = true

[[users]]
name = "alice"
password_hash = "sha256:5e884898..."
graphs = ["*"]          # "*" = all graphs; or ["myproject", "analytics"]
```

---

## Web GUI

Open **http://localhost:7475** in a browser. The GUI is a dark single-page app powered by [Cytoscape.js](https://js.cytoscape.org).

**Editor panel:**
- GQL editor with keyboard shortcuts: `Ctrl+Enter` (run query), `Ctrl+Shift+Enter` (visualize)
- Query history navigation: `Shift+↑` / `Shift+↓` (stores last 30 queries across sessions)
- `BEGIN` / `COMMIT` / `ROLLBACK` buttons for explicit transaction control
- `↓ CSV` button to export the current result set

**Results table:**
- Click any column header to sort ascending / descending
- Node and edge IDs that appear in result cells are underlined and clickable — clicking selects and centers that element in the graph view
- Full cell value shown on hover for truncated values

**Graph visualization panel:**
- Multiple layout algorithms: Force-directed (fcose), Force classic, Circular, Concentric, Hierarchical (dagre), Multipartite, Grid
- Color nodes by label or any property value, with an auto-generated legend
- Filter nodes by label or property (`Filter nodes…` input — non-matching nodes and their edges fade out)
- Drag a node to pin it at a fixed position (shown with an orange dashed border); right-click to unpin; `⊘ Unpin All` releases all pins
- Click a **node** to show its labels, ID, and all properties in the info panel below the graph
- Click an **edge** to show its label, endpoint IDs, and properties in the info panel
- Hover over any node or edge for a floating tooltip summary
- `⌗ Labels` button to toggle all node/edge label rendering
- `⛶` button for fullscreen graph mode
- `↔ Replace` / `➕ Add` toggle — Add mode merges new visualization results into the existing view instead of replacing it
- `Del` key removes selected elements from the view (does not delete from the database)
- Live node/edge count displayed in the graph panel
- **`⊕ Expand`** button — fetches one hop of neighbours for all selected nodes (or all visible nodes if none selected) and merges them into the current view; limited to 500 visible nodes

**Graph management:**
- Switch between named graphs from the dropdown in the header
- Create or drop graphs directly from the GUI

**Authentication:**
- Login form shown automatically if the server has auth enabled
- Session token stored in `sessionStorage`

---

## Wire protocol (v2)

Protocol is **newline-delimited JSON** over a plain TCP stream.

**1. Server sends hello on connect:**
```json
{"type":"hello","version":"2","auth_required":true}
```

**2. Client authenticates (if `auth_required`):**
```json
{"type":"auth","user":"alice","password":"secret"}
```
Server replies:
```json
{"type":"auth_ok","user":"alice"}
```

**3. GQL queries:**
```json
{"id":1,"query":"INSERT (:Person {name:\"Alice\"})"}
{"id":2,"query":"MATCH (n:Person) RETURN n.name"}
{"id":3,"graph":"analytics","query":"MATCH (n) RETURN count(n)"}
```
Response:
```json
{"id":2,"rows":[{"n.name":"Alice"}],"elapsed_ms":0.3}
```

**4. Transactions:**
```json
{"id":4,"query":"BEGIN"}
{"id":5,"query":"INSERT (:City {name:\"NYC\"})"}
{"id":6,"query":"COMMIT"}
```

Each connection holds an exclusive per-graph lock for the duration of a transaction. Disconnecting without `COMMIT` automatically rolls back.

**5. Admin commands:**
```json
{"type":"admin","cmd":"graphs"}
→ {"type":"admin_ok","graphs":["default","analytics","myproject"]}

{"type":"admin","cmd":"create","name":"newgraph"}
→ {"type":"admin_ok"}

{"type":"admin","cmd":"drop","name":"oldgraph"}
→ {"type":"admin_ok"}

{"type":"admin","cmd":"stats"}
→ {"type":"admin_ok","open_graphs":["default"]}
```

---

## Quick test with netcat

```bash
cargo run -- serve --no-auth &

echo '{"id":1,"query":"INSERT (:Person {name:\"Alice\"})"}' | nc localhost 7474
echo '{"id":2,"query":"MATCH (n:Person) RETURN n.name"}' | nc localhost 7474
```

---

## HTTP API

The web GUI server also exposes a REST API (used internally by the GUI):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/info` | `{"auth_required": bool}` |
| `POST` | `/api/auth` | `{user, password}` → `{token}` |
| `GET` | `/api/graphs` | List graph names |
| `POST` | `/api/graphs` | Create a graph: `{name}` |
| `DELETE` | `/api/graphs/:name` | Drop a graph |
| `POST` | `/api/query` | Run GQL: `{graph?, query}` → `{rows, elapsed_ms}` |
| `POST` | `/api/viz` | Visualize: `{graph?, query}` → `{rows, nodes, edges}` — automatically resolves ULID strings in results to full node/edge records |
| `POST` | `/api/upload/nodes` | CSV node import: `{csv, graph?, label?}` → `{inserted, id_map}` |
| `POST` | `/api/upload/edges` | CSV edge import: `{csv, id_map, graph?, label?}` → `{inserted, skipped}` |
