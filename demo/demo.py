"""
minigdb Python API demonstration.

Mirrors the README sequence end-to-end, exercising the core Python binding:
  - Node and edge insertion via GQL
  - MATCH / WHERE / ORDER BY queries
  - Aggregation functions (count, avg, min, max)
  - Property mutation (SET)
  - Explicit transactions (begin / commit / rollback)
  - DETACH DELETE
  - Variable-length path quantifiers
  - Optional DataFrame output via query_df() and query_pandas()

Prerequisites:
    maturin develop --features python

Run:
    python demo/demo.py
"""

import minigdb


# Name of the demo graph stored under the default data directory.
GRAPH = "demo_readme"


# ── helpers ───────────────────────────────────────────────────────────────────


def section(title):
    """Print a prominently formatted section header to stdout.

    Args:
        title: Human-readable label for the upcoming demo section.
    """
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def show(label, rows):
    """Print a labelled list of result rows to stdout.

    Each row is a dict mapping column name to value.  When the result set is
    empty, a placeholder message is printed instead.

    Args:
        label: Descriptive heading printed above the rows.
        rows: Sequence of row dicts returned by ``db.query()``.
    """
    print(f"\n{label}")
    if not rows:
        print("  (no results)")
    for row in rows:
        print(" ", row)


# ── main demo ─────────────────────────────────────────────────────────────────

# Open (or create) the demo graph.  The context manager guarantees checkpoint
# and close even if an exception is raised mid-demo.
with minigdb.open(GRAPH) as db:

    # -- Node insertion -------------------------------------------------------

    section("Insert nodes")
    db.query('INSERT (:Person {name: "Alice", age: 30})')
    db.query('INSERT (:Person {name: "Bob",   age: 25})')
    db.query('INSERT (:Person {name: "Carol", age: 35})')
    print("  Inserted Alice, Bob, Carol")

    # -- Edge insertion via MATCH+INSERT pattern ------------------------------

    section("Insert edge")
    # MATCH+INSERT locates existing nodes by label/property and connects them.
    db.query('MATCH (a:Person {name:"Alice"}), (b:Person {name:"Bob"}) INSERT (a)-[:KNOWS]->(b)')
    db.query('MATCH (a:Person {name:"Alice"}), (c:Person {name:"Carol"}) INSERT (a)-[:MANAGES]->(c)')
    print("  Inserted KNOWS + MANAGES edges")

    # -- Basic read queries ---------------------------------------------------

    section("Basic MATCH / RETURN")
    rows = db.query('MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age')
    show("All people (ordered by age):", rows)

    section("WHERE filter")
    rows = db.query('MATCH (n:Person) WHERE n.age > 28 RETURN n.name, n.age')
    show("People older than 28:", rows)

    section("Edge traversal")
    rows = db.query(
        'MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name'
    )
    show("KNOWS relationships:", rows)

    # -- Aggregation ----------------------------------------------------------

    section("Aggregation")
    rows = db.query('MATCH (n:Person) RETURN count(*), avg(n.age), min(n.age), max(n.age)')
    show("Aggregate stats:", rows)

    # -- Property mutation ----------------------------------------------------

    section("SET property")
    db.query('MATCH (n:Person {name:"Alice"}) SET n.age = 31')
    rows = db.query('MATCH (n:Person {name:"Alice"}) RETURN n.name, n.age')
    show("Alice after SET:", rows)

    # -- Explicit transaction: atomic multi-statement batch -------------------

    section("Explicit transaction — atomic batch")
    db.begin()
    db.query('INSERT (:Person {name: "Dave", age: 22})')
    db.query('INSERT (:Person {name: "Eve",  age: 27})')
    # Both inserts are written to the WAL as a single frame on commit.
    db.commit()
    rows = db.query('MATCH (n:Person) RETURN n.name ORDER BY n.name')
    show("All people after committed batch:", rows)

    # -- Rollback: changes made inside the transaction are discarded ----------

    section("Rollback demo")
    db.begin()
    db.query('INSERT (:Person {name: "Ghost"})')
    db.rollback()  # Restore the pre-BEGIN graph snapshot; Ghost is gone.
    rows = db.query('MATCH (n:Person {name:"Ghost"}) RETURN n.name')
    show("Ghost after rollback (expect empty):", rows)

    # -- Node deletion --------------------------------------------------------

    section("DELETE node")
    # DETACH DELETE removes the node and all its incident edges atomically.
    db.query('MATCH (n:Person {name:"Dave"}) DETACH DELETE n')
    rows = db.query('MATCH (n:Person) RETURN n.name ORDER BY n.name')
    show("People after deleting Dave:", rows)

    # -- Variable-length path quantifier -------------------------------------

    section("Path quantifier (multi-hop)")
    # `*1..3` means "traverse between 1 and 3 edges along any relationship".
    rows = db.query(
        'MATCH (a:Person {name:"Alice"})-[*1..3]->(x:Person) RETURN a.name, x.name'
    )
    show("Reachable from Alice (1-3 hops):", rows)


# ── optional: DataFrame output via query_df / query_pandas ───────────────────

# Reopen the same graph for DataFrame demonstration.  polars and pandas are
# optional runtime dependencies; each block gracefully skips when absent.
with minigdb.open(GRAPH) as db:

    try:
        section("query_df() → polars DataFrame")
        df = db.query_df("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.name")
        print()
        print(df)
    except RuntimeError as e:
        # polars is not installed — skip without failing the demo.
        print(f"\n  (skipped: {e})")

    try:
        section("query_pandas() → pandas DataFrame")
        df = db.query_pandas("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.name")
        print()
        # index=False keeps the output clean by suppressing the row index column.
        print(df.to_string(index=False))
    except RuntimeError as e:
        # pandas is not installed — skip without failing the demo.
        print(f"\n  (skipped: {e})")


print("\nDemo complete.")
