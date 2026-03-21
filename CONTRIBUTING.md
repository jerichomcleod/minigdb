# Contributing to minigdb

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Rust | stable (≥ 1.75) | [rustup.rs](https://rustup.rs) |
| cmake | ≥ 3.14 | `brew install cmake` / `apt install cmake` / [cmake.org](https://cmake.org) |
| C++ compiler | any modern | Xcode CLT / `build-essential` / MSVC |
| Python | ≥ 3.8 | [python.org](https://python.org) |
| maturin | ≥ 1.7 | `pip install maturin` |

cmake and a C++ compiler are required because RocksDB is compiled from source and linked statically into the library. This happens automatically on `cargo build` / `maturin develop` — you don't invoke cmake manually.

---

## Building

### Rust binary / REPL

```bash
cargo build                  # debug build
cargo build --release        # optimised build

cargo run                    # launch the REPL (debug)
cargo run --release          # launch the REPL (optimised)
```

### Run tests

```bash
cargo test                   # all 331 tests
cargo test <filter>          # e.g. cargo test transaction
cargo test -- --nocapture    # show println! output
```

Tests are hermetic — each test opens a `tempfile::TempDir`-backed RocksDB and cleans up on drop. No shared state between tests.

### Python bindings (development)

```bash
# Build and install into the active virtual environment
pip install maturin
maturin develop --features python

# Verify
python -c "import minigdb; print('ok')"
```

`maturin develop` compiles a debug build and installs it as `minigdb` in the active venv. After any Rust source change, re-run `maturin develop` to pick it up.

For a release-optimised local build:

```bash
maturin develop --release --features python
```

### Build a wheel locally (without publishing)

```bash
maturin build --release --features python --out dist/
# → dist/minigdb-0.x.y-cp311-cp311-<platform>.whl

pip install dist/minigdb-*.whl
```

---

## Project structure

```
src/
├── main.rs           # REPL, server entry points, graph management commands
├── lib.rs            # Public API re-exports
├── gql.pest          # PEG grammar (pest reads this at compile time from src/)
├── python.rs         # Python bindings (feature = "python")
├── types/            # Value, NodeId/EdgeId (ULID), Node, Edge, DbError
├── graph/            # Graph struct, RocksDB-backed ops (ops.rs)
├── storage/          # RocksStore (8 column families), StorageManager
├── transaction/      # Operation enum (replay units)
├── algorithms/       # 15 graph algorithm modules dispatched via CALL
├── query/            # AST (ast.rs), pest parser (parser.rs), executor (executor.rs)
└── server/
    ├── mod.rs        # Tokio TCP server, per-connection state, auth handshake
    ├── auth.rs       # ServerConfig, UserEntry, SHA-256 password hashing
    ├── registry.rs   # GraphRegistry — lazy-open, Arc<Mutex> per graph
    ├── protocol.rs   # Wire types: Request, Response, ClientMessage, ServerMessage
    └── http.rs       # Axum HTTP server + embedded GUI HTML (feature = "gui")
```

### Cargo features

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `repl` | yes | Interactive REPL (rustyline + readline history) |
| `server` | yes | TCP server (tokio + SHA-256 auth) |
| `gui` | yes | Embedded HTTP GUI (axum); requires `server` |
| `python` | no | PyO3 Python bindings; build with `maturin develop --features python` |

To build only the core library without REPL or server:

```bash
cargo build --no-default-features
```

---

## Making changes

### Grammar changes

The PEG grammar lives in `src/gql.pest`. pest reads it at compile time using a
path relative to `src/`, so the file must stay at that location.

After editing the grammar, rebuild and run the parser tests:

```bash
cargo test parser
```

### Adding a graph algorithm

1. Add `src/algorithms/<name>.rs` implementing the algorithm against `GraphSnapshot`
2. Register it in `src/algorithms/mod.rs` inside `dispatch_call()`
3. Add at least one integration test in `tests/`

### Modifying the Python API

Python bindings are in `src/python.rs`. After any change, run:

```bash
maturin develop --features python
python -c "import minigdb; <smoke test>"
```

---

## Releasing a new version

### 1. Update version numbers

Both files must be kept in sync — maturin uses `pyproject.toml` for the Python
package version and `Cargo.toml` for the crate version.

```
pyproject.toml  →  [project] version = "0.x.y"
Cargo.toml      →  [package] version = "0.x.y"
```

### 2. Run the full test suite

```bash
cargo test
```

All tests must pass before tagging.

### 3. Tag and push

```bash
git tag v0.x.y
git push origin v0.x.y
```

Pushing a `v*` tag triggers the release CI workflow (`.github/workflows/release.yml`).

### 4. What CI does automatically

The workflow runs in parallel across four targets:

| Target | Runner | Notes |
|--------|--------|-------|
| Linux x86_64 | ubuntu-latest | Built inside manylinux_2_28 container |
| Linux aarch64 | ubuntu-latest | Cross-compiled via QEMU |
| macOS universal2 | macos-latest | Single fat wheel: Intel + Apple Silicon |
| Windows x86_64 | windows-latest | MSVC toolchain |

For each target, the workflow:
1. Compiles RocksDB from source (C++, statically linked)
2. Compiles the Rust library with `--features python`
3. Packages the result as a `.whl` with maturin

After all wheels build successfully, CI also builds a source distribution (`sdist`) and uploads everything to PyPI via OIDC trusted publishing.

### 5. Verify the release

```bash
# In a fresh venv
pip install minigdb==0.x.y
python -c "import minigdb; db = minigdb.open('test'); db.query('INSERT (:T {x: 1})'); print(db.query('MATCH (n:T) RETURN n.x')); db.close()"
```

---

## PyPI trusted publishing setup (one-time)

This only needs to be done once, before the first release.

**On PyPI:**
1. Log in to [pypi.org](https://pypi.org) and go to Account Settings → Publishing
2. Click **Add a new publisher**
3. Fill in:
   - PyPI project name: `minigdb`
   - GitHub owner: `jerichomcleod`
   - Repository name: `minigdb`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
4. Save

**On GitHub:**
1. Go to the repository Settings → Environments → **New environment**
2. Name it `pypi`
3. Optionally add a required reviewer as a manual approval gate before publish

**First publish:**

The first upload must be done manually to register the project name on PyPI (trusted
publishing only works for projects that already exist):

```bash
maturin build --release --features python --out dist/
maturin upload dist/*
# prompts for PyPI username + password / API token
```

After the first manual publish, all subsequent releases are fully automated by
pushing a version tag.

---

## Troubleshooting

### RocksDB build fails locally

```
error: failed to run custom build command for `rocksdb-sys`
```

Make sure cmake and a C++ compiler are installed:

```bash
# macOS
xcode-select --install
brew install cmake

# Ubuntu / Debian
sudo apt install cmake build-essential

# Windows
# Install Visual Studio Build Tools with "Desktop development with C++" workload
# Install cmake from https://cmake.org/download/
```

### `maturin develop` can't find Python

Activate your virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin
maturin develop --features python
```

### Wheel installs but `import minigdb` fails on Linux

The pre-built wheels target `manylinux_2_28` (glibc 2.28+). If you are on an
older system (e.g. Ubuntu 18.04 with glibc 2.27), install from source:

```bash
pip install minigdb --no-binary minigdb
```

This requires Rust, cmake, and a C++ compiler on the host machine.
