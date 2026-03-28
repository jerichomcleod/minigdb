"""
Python-side integration tests for the minigdb PyO3 API.

These tests exercise the public Python binding exposed by ``src/python.rs``
and compiled via maturin.  They cover:
  - Basic insert and MATCH / WHERE / ORDER BY / LIMIT / OFFSET queries
  - Edge traversal
  - Property mutations (SET, DETACH DELETE)
  - Value type round-tripping (bool, int, float, str, null, list)
  - Aggregation functions (count, avg, min, max, collect)
  - Explicit transaction semantics (begin / commit / rollback)
  - Context-manager lifecycle (checkpoint on ``__exit__``, implicit rollback)
  - Error handling (parse errors, invalid paths, bad graph names)
  - Persistence across open/close cycles (snapshot + WAL replay)
  - Graph name validation enforced by ``minigdb.open()``
  - STARTS WITH index pushdown (property index prefix scan)
  - Adjacency-delta rollback (edge insert/delete inside a transaction)

Prerequisites:
    maturin develop --features python

Run:
    pytest tests/test_api.py -v
"""

import pytest
import minigdb


# ── fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    """Yield a fresh, empty graph opened in a temporary directory.

    The handle is closed (triggering checkpoint) after each test, and the
    temporary directory is automatically removed by pytest.

    Args:
        tmp_path: pytest built-in fixture providing a per-test ``pathlib.Path``.

    Yields:
        An open ``minigdb.MiniGdb`` handle backed by ``tmp_path``.
    """
    handle = minigdb.open_at(str(tmp_path))
    yield handle
    handle.close()


# ── basic insert / query ──────────────────────────────────────────────────────


def test_empty_graph_returns_no_rows(db):
    """MATCH against an empty graph should return an empty list."""
    rows = db.query('MATCH (n:Person) RETURN n.name')
    assert rows == []


def test_insert_and_match_single_node(db):
    """A single inserted node should be retrievable with correct property values."""
    db.query('INSERT (:Person {name: "Alice", age: 30})')
    rows = db.query('MATCH (n:Person) RETURN n.name, n.age')
    assert len(rows) == 1
    assert rows[0]['n.name'] == 'Alice'
    assert rows[0]['n.age'] == 30


def test_insert_multiple_nodes(db):
    """Multiple nodes should all be returned, alphabetically ordered on request."""
    db.query('INSERT (:Person {name: "Alice"})')
    db.query('INSERT (:Person {name: "Bob"})')
    db.query('INSERT (:Person {name: "Carol"})')
    rows = db.query('MATCH (n:Person) RETURN n.name ORDER BY n.name')
    assert [r['n.name'] for r in rows] == ['Alice', 'Bob', 'Carol']


def test_where_filter(db):
    """WHERE predicate should exclude nodes that do not satisfy the condition."""
    db.query('INSERT (:Person {name: "Alice", age: 30})')
    db.query('INSERT (:Person {name: "Bob",   age: 20})')
    rows = db.query('MATCH (n:Person) WHERE n.age > 25 RETURN n.name')
    assert len(rows) == 1
    assert rows[0]['n.name'] == 'Alice'


def test_limit_and_offset(db):
    """LIMIT and OFFSET should paginate a sorted result set correctly."""
    for name in ['A', 'B', 'C', 'D']:
        db.query(f'INSERT (:X {{name: "{name}"}})')

    # First page: two items starting from the beginning.
    rows = db.query('MATCH (n:X) RETURN n.name ORDER BY n.name LIMIT 2')
    assert [r['n.name'] for r in rows] == ['A', 'B']

    # Second page: two items starting after the first two.
    rows = db.query('MATCH (n:X) RETURN n.name ORDER BY n.name LIMIT 2 OFFSET 2')
    assert [r['n.name'] for r in rows] == ['C', 'D']


# ── edge traversal ────────────────────────────────────────────────────────────


def test_insert_and_traverse_edge(db):
    """An inserted directed edge should be traversable in a MATCH pattern."""
    db.query('INSERT (:Person {name: "Alice"})')
    db.query('INSERT (:Person {name: "Bob"})')
    # MATCH+INSERT pattern: locate both endpoints then create the edge.
    db.query('MATCH (a:Person {name:"Alice"}), (b:Person {name:"Bob"}) INSERT (a)-[:KNOWS]->(b)')
    rows = db.query('MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name')
    assert len(rows) == 1
    assert rows[0]['a.name'] == 'Alice'
    assert rows[0]['b.name'] == 'Bob'


# ── mutations ─────────────────────────────────────────────────────────────────


def test_set_property(db):
    """SET should overwrite an existing property with the new value."""
    db.query('INSERT (:Person {name: "Alice", age: 30})')
    db.query('MATCH (n:Person {name:"Alice"}) SET n.age = 31')
    rows = db.query('MATCH (n:Person) RETURN n.age')
    assert rows[0]['n.age'] == 31


def test_delete_node(db):
    """DETACH DELETE should remove the target node and leave others intact."""
    db.query('INSERT (:Person {name: "Alice"})')
    db.query('INSERT (:Person {name: "Bob"})')
    db.query('MATCH (n:Person {name:"Bob"}) DETACH DELETE n')
    rows = db.query('MATCH (n:Person) RETURN n.name')
    assert len(rows) == 1
    assert rows[0]['n.name'] == 'Alice'


# ── value type mapping ────────────────────────────────────────────────────────


def test_value_types(db):
    """All five primitive GQL value types should round-trip to correct Python types.

    Mapping:
        GQL bool  → Python bool
        GQL int   → Python int
        GQL float → Python float
        GQL str   → Python str
        GQL null  → Python None
    """
    db.query('INSERT (:T {b: true, i: 42, f: 3.14, s: "hello", n: null})')
    rows = db.query('MATCH (x:T) RETURN x.b, x.i, x.f, x.s, x.n')
    assert len(rows) == 1
    r = rows[0]
    assert r['x.b'] is True
    assert r['x.i'] == 42
    assert isinstance(r['x.f'], float)
    # Use a tolerance for the float comparison to allow for IEEE 754 rounding.
    assert abs(r['x.f'] - 3.14) < 1e-9
    assert r['x.s'] == 'hello'
    assert r['x.n'] is None


def test_list_value(db):
    """collect() aggregate should map to a Python list of the collected values."""
    db.query('INSERT (:Person {name: "Alice"})')
    db.query('INSERT (:Person {name: "Bob"})')
    rows = db.query('MATCH (n:Person) RETURN collect(n.name) AS names ORDER BY names')
    # collect() returns a GQL List value, which the binding converts to a Python list.
    names = rows[0]['names']
    assert isinstance(names, list)
    # Use set comparison since collect order is not guaranteed.
    assert set(names) == {'Alice', 'Bob'}


# ── aggregation ───────────────────────────────────────────────────────────────


def test_count_star(db):
    """count(*) should return the total number of matched rows."""
    for name in ['Alice', 'Bob', 'Carol']:
        db.query(f'INSERT (:Person {{name: "{name}"}})')
    rows = db.query('MATCH (n:Person) RETURN count(*)')
    assert rows[0]['count(*)'] == 3


def test_count_star_empty(db):
    """count(*) on an empty match should return a single row with value 0 (SQL semantics)."""
    rows = db.query('MATCH (n:Person) RETURN count(*)')
    assert rows[0]['count(*)'] == 0


def test_avg_min_max(db):
    """avg / min / max should compute the correct statistics over a numeric property."""
    db.query('INSERT (:N {v: 10})')
    db.query('INSERT (:N {v: 20})')
    db.query('INSERT (:N {v: 30})')
    rows = db.query('MATCH (n:N) RETURN avg(n.v), min(n.v), max(n.v)')
    # avg returns a float; use pytest.approx to handle floating-point representation.
    assert rows[0]['avg(n.v)'] == pytest.approx(20.0)
    assert rows[0]['min(n.v)'] == 10
    assert rows[0]['max(n.v)'] == 30


# ── explicit transactions ─────────────────────────────────────────────────────


def test_commit_makes_changes_visible(db):
    """Inserts inside a committed transaction should be visible on subsequent queries."""
    db.begin()
    db.query('INSERT (:Person {name: "Alice"})')
    db.query('INSERT (:Person {name: "Bob"})')
    db.commit()
    rows = db.query('MATCH (n:Person) RETURN n.name ORDER BY n.name')
    assert [r['n.name'] for r in rows] == ['Alice', 'Bob']


def test_rollback_discards_changes(db):
    """Inserts inside a rolled-back transaction must not appear after rollback."""
    # Commit Alice outside the transaction so she persists regardless.
    db.query('INSERT (:Person {name: "Alice"})')
    db.begin()
    db.query('INSERT (:Person {name: "Ghost"})')
    db.rollback()  # Restores the pre-BEGIN snapshot; Ghost is discarded.
    rows = db.query('MATCH (n:Person) RETURN n.name')
    assert len(rows) == 1
    assert rows[0]['n.name'] == 'Alice'


def test_nested_begin_raises(db):
    """Calling begin() while a transaction is already open must raise RuntimeError."""
    db.begin()
    with pytest.raises(RuntimeError, match="Transaction already open"):
        db.begin()
    db.rollback()


def test_commit_without_begin_raises(db):
    """commit() with no open transaction must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="No open transaction"):
        db.commit()


def test_rollback_without_begin_raises(db):
    """rollback() with no open transaction must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="No open transaction"):
        db.rollback()


# ── context manager ───────────────────────────────────────────────────────────


def test_context_manager_closes_on_exit(tmp_path):
    """The ``with`` block must checkpoint and close the graph on normal exit.

    Data written inside the block should survive a full close-and-reopen cycle
    because ``__exit__`` triggers a checkpoint flush to disk.
    """
    with minigdb.open_at(str(tmp_path)) as db:
        db.query('INSERT (:Person {name: "Alice"})')

    # Re-open the same path and verify the checkpoint was written.
    with minigdb.open_at(str(tmp_path)) as db:
        rows = db.query('MATCH (n:Person) RETURN n.name')

    assert rows[0]['n.name'] == 'Alice'


def test_context_manager_rollback_on_open_txn(tmp_path):
    """An open transaction must be implicitly rolled back when the context exits.

    Data committed before ``begin()`` should persist; data inserted inside the
    open transaction (never committed) must be discarded.
    """
    with minigdb.open_at(str(tmp_path)) as db:
        db.query('INSERT (:Person {name: "Alice"})')
        db.begin()
        db.query('INSERT (:Person {name: "Ghost"})')
        # Exit without calling commit() — __exit__ should trigger rollback.

    # Reopen and confirm only Alice (the pre-transaction insert) is present.
    with minigdb.open_at(str(tmp_path)) as db:
        rows = db.query('MATCH (n:Person) RETURN n.name')

    assert len(rows) == 1
    assert rows[0]['n.name'] == 'Alice'


# ── error handling ────────────────────────────────────────────────────────────


def test_parse_error_raises_runtime_error(db):
    """A syntactically invalid GQL string must raise RuntimeError."""
    with pytest.raises(RuntimeError):
        db.query('NOT VALID GQL %%%')


def test_open_at_invalid_path_raises(tmp_path):
    """open_at() must raise OSError when the path cannot be created as a directory.

    A file placed at the target location makes ``create_dir_all`` fail, which
    the binding surfaces as ``DbError::Storage`` → ``PyOSError``.
    """
    # Create a regular file that occupies the path where a directory is expected.
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file")
    with pytest.raises(OSError):
        minigdb.open_at(str(blocker / "subdir"))


# ── persistence across open/close ────────────────────────────────────────────


def test_data_survives_reopen(tmp_path):
    """Data written and checkpointed must be present after a full close-and-reopen."""
    with minigdb.open_at(str(tmp_path)) as db:
        db.query('INSERT (:Person {name: "Alice", age: 30})')

    with minigdb.open_at(str(tmp_path)) as db:
        rows = db.query('MATCH (n:Person) RETURN n.name, n.age')

    assert rows[0]['n.name'] == 'Alice'
    assert rows[0]['n.age'] == 30


def test_wal_replay_after_reopen(tmp_path):
    """Write several auto-committed queries, reopen, verify WAL was replayed.

    Each ``query()`` call outside a transaction auto-commits via
    ``commit_wal_only()``.  On reopen the WAL frames are replayed in order,
    reconstructing the full graph state.
    """
    with minigdb.open_at(str(tmp_path)) as db:
        for name in ['A', 'B', 'C']:
            db.query(f'INSERT (:X {{name: "{name}"}})')

    with minigdb.open_at(str(tmp_path)) as db:
        rows = db.query('MATCH (n:X) RETURN n.name ORDER BY n.name')

    assert [r['n.name'] for r in rows] == ['A', 'B', 'C']


# ── open() name validation ────────────────────────────────────────────────────


def test_open_empty_name_raises():
    """open() must reject an empty string as a graph name."""
    with pytest.raises(RuntimeError, match="1–64"):
        minigdb.open('')


def test_open_bad_chars_raises():
    """open() must reject names containing characters outside [a-zA-Z0-9_-]."""
    with pytest.raises(RuntimeError, match="alphanumeric"):
        minigdb.open('bad name!')


def test_open_too_long_raises():
    """open() must reject names longer than 64 characters."""
    with pytest.raises(RuntimeError, match="1–64"):
        minigdb.open('x' * 65)


# ── STARTS WITH index pushdown ────────────────────────────────────────────────


def test_starts_with_uses_index(db):
    """startsWith() in WHERE should use the property index and return only matching nodes."""
    db.query('CREATE INDEX ON :Product(sku)')
    db.query('INSERT (:Product {sku: "BOOK-001", title: "Rust Programming"})')
    db.query('INSERT (:Product {sku: "BOOK-002", title: "Graph Databases"})')
    db.query('INSERT (:Product {sku: "ELEC-100", title: "Keyboard"})')

    rows = db.query('MATCH (p:Product) WHERE startsWith(p.sku, "BOOK") RETURN p.title ORDER BY p.title')
    assert len(rows) == 2
    assert rows[0]['p.title'] == 'Graph Databases'
    assert rows[1]['p.title'] == 'Rust Programming'


def test_starts_with_no_index_fallback(db):
    """startsWith() without an index should still return the correct results via full scan."""
    db.query('INSERT (:Product {sku: "BOOK-001", title: "Rust Programming"})')
    db.query('INSERT (:Product {sku: "BOOK-002", title: "Graph Databases"})')
    db.query('INSERT (:Product {sku: "ELEC-100", title: "Keyboard"})')

    rows = db.query('MATCH (p:Product) WHERE startsWith(p.sku, "BOOK") RETURN p.title ORDER BY p.title')
    assert len(rows) == 2
    assert {r['p.title'] for r in rows} == {'Rust Programming', 'Graph Databases'}


def test_starts_with_empty_prefix_returns_all_strings(db):
    """startsWith(prop, "") should match every node that has the property set."""
    db.query('CREATE INDEX ON :Item(code)')
    db.query('INSERT (:Item {code: "A1"})')
    db.query('INSERT (:Item {code: "B2"})')
    db.query('INSERT (:Item {code: "C3"})')

    rows = db.query('MATCH (i:Item) WHERE startsWith(i.code, "") RETURN i.code ORDER BY i.code')
    assert [r['i.code'] for r in rows] == ['A1', 'B2', 'C3']


def test_starts_with_no_match_returns_empty(db):
    """startsWith() with a prefix that matches nothing should return an empty result."""
    db.query('CREATE INDEX ON :Item(code)')
    db.query('INSERT (:Item {code: "A1"})')
    db.query('INSERT (:Item {code: "A2"})')

    rows = db.query('MATCH (i:Item) WHERE startsWith(i.code, "Z") RETURN i.code')
    assert rows == []


def test_starts_with_combined_with_and(db):
    """startsWith() combined with another AND predicate should apply both filters."""
    db.query('CREATE INDEX ON :Person(name)')
    db.query('INSERT (:Person {name: "Alice", age: 30})')
    db.query('INSERT (:Person {name: "Alan",  age: 20})')
    db.query('INSERT (:Person {name: "Bob",   age: 25})')

    rows = db.query(
        'MATCH (n:Person) WHERE startsWith(n.name, "Al") AND n.age > 25 RETURN n.name'
    )
    assert len(rows) == 1
    assert rows[0]['n.name'] == 'Alice'


# ── adjacency-delta rollback ──────────────────────────────────────────────────


def test_rollback_edge_insert(db):
    """An edge inserted inside a transaction must disappear after rollback."""
    db.query('INSERT (:Person {name: "Alice"})')
    db.query('INSERT (:Person {name: "Bob"})')

    db.begin()
    db.query('MATCH (a:Person {name:"Alice"}), (b:Person {name:"Bob"}) INSERT (a)-[:KNOWS]->(b)')
    # Confirm the edge is visible inside the transaction.
    rows = db.query('MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name')
    assert len(rows) == 1

    db.rollback()

    # After rollback the edge must be gone.
    rows = db.query('MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name')
    assert rows == []


def test_rollback_edge_delete(db):
    """An edge deleted inside a transaction must be restored after rollback."""
    db.query('INSERT (:Person {name: "Alice"})')
    db.query('INSERT (:Person {name: "Bob"})')
    db.query('MATCH (a:Person {name:"Alice"}), (b:Person {name:"Bob"}) INSERT (a)-[:KNOWS]->(b)')

    db.begin()
    db.query('MATCH (a:Person {name:"Alice"})-[r:KNOWS]->(b:Person {name:"Bob"}) DELETE r')
    rows = db.query('MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name')
    assert rows == []  # gone inside the transaction

    db.rollback()

    # After rollback the edge must be restored.
    rows = db.query('MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name')
    assert len(rows) == 1
    assert rows[0]['a.name'] == 'Alice'
    assert rows[0]['b.name'] == 'Bob'
