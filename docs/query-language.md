# GQL Query Language Reference

[← Back to README](../README.md) · [Algorithms](algorithms.md) · [Server](server.md) · [Python API](python.md)

minigdb implements a subset of GQL (ISO/IEC 39075). Semicolons are optional; multi-line input is supported in the REPL.

---

## Insert nodes

INSERT is **idempotent** on label + properties — if a node with the same label and all the same property values already exists, it is reused; no duplicate is created.

```
INSERT (:Person {name: "Alice", age: 30});
INSERT (:Person {name: "Bob", age: 25});
INSERT (:Person:Employee {name: "Carol", age: 35});
```

## Insert edges

Variables declared in the **same** INSERT can be used as edge endpoints. Existing nodes (matched by label + all supplied properties) are reused automatically.

```
INSERT (a:Person {name: "Alice"}), (b:Person {name: "Bob"}), (a)-[:KNOWS]->(b)
```

Alice and Bob already exist, so they are reused. Only the KNOWS edge is new.

## Connect existing nodes with MATCH+INSERT

To add an edge between nodes created in separate statements, match them first. This never duplicates existing nodes.

```
MATCH (a:Person {name: "Alice"}), (b:Person:Employee {name: "Carol"})
INSERT (a)-[:MANAGES]->(b)
```

Multiple comma-separated patterns in MATCH are cross-joined. A WHERE clause is supported:

```
MATCH (i:Item), (h:Hub) WHERE i.active = true
INSERT (i)-[:CONNECTED_TO]->(h)
```

You can also create a new node and link it to a matched node in one statement:

```
MATCH (a:Person {name: "Alice"})
INSERT (t:Tag {name: "rust"}), (a)-[:TAGGED]->(t)
```

> **Note:** Plain `INSERT` always creates new nodes unless label+properties match an existing one. To connect nodes that already exist, use `MATCH … INSERT`.

## Query nodes

```
MATCH (n:Person) RETURN n.name, n.age
MATCH (n:Person) WHERE n.age > 28 RETURN n.name ORDER BY n.age DESC
MATCH (n:Person) RETURN n.name LIMIT 2 OFFSET 1
MATCH (n:Person) RETURN DISTINCT n.name
```

## Query edges (single-hop)

```
MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name
MATCH (a:Person)-[r]->(b) RETURN a.name, type(r), b.name
MATCH (a)<-[:KNOWS]-(b) RETURN a.name, b.name   -- a is the target, b is the source
MATCH (a)-[:KNOWS]-(b) RETURN a.name, b.name     -- either direction
```

## Variable-length paths

```
MATCH (a)-[*1..3]->(b) RETURN a.name, b.name        -- 1 to 3 hops
MATCH (a)-[:KNOWS*]->(b) RETURN a.name, b.name       -- any number of KNOWS hops
MATCH (a)-[:KNOWS+]->(b) RETURN a.name, b.name       -- 1 or more hops
MATCH (a)-[r:KNOWS{1,2}]->(b) RETURN a.name, length(r), b.name
MATCH (a)-[*2]->(b) RETURN a.name, b.name            -- exactly 2 hops
MATCH (a)-[*2..]->(b) RETURN a.name, b.name          -- 2 or more hops
```

A path variable (e.g. `r`) binds to the list of edge IDs traversed. Use `length(r)` to get the hop count.

**Quantifier forms:** `*`, `+`, `*n`, `*m..n`, `*m..`, `{n}`, `{m,n}`, `{m,}`

## Path modes

Path modes control whether nodes or edges may be revisited within a single matched path.

```
MATCH TRAIL  (a)-[*]->(b) RETURN a.name, b.name   -- no repeated edges
MATCH SIMPLE (a)-[*]->(b) RETURN a.name, b.name   -- no repeated nodes
MATCH WALK   (a)-[*]->(b) RETURN a.name, b.name   -- default, no restriction
```

## Optional match

`OPTIONAL MATCH` returns one null row when the pattern has no results, instead of returning nothing. This preserves the outer context in a pipeline.

```
OPTIONAL MATCH (n:Person {name: "Unknown"}) RETURN n.name
```

```
+--------+
| n.name |
+--------+
| null   |
+--------+
1 row(s)
```

## UNWIND

`UNWIND` expands a list expression into individual rows.

```
UNWIND [1, 2, 3] AS x RETURN x * 2 AS doubled
```

Use with `collect()` to post-process aggregated lists:

```
MATCH (n:Person) WITH collect(n.name) AS names UNWIND names AS name RETURN name
```

## UNION

Combine result sets from two statements. `UNION` deduplicates; `UNION ALL` keeps all rows. Both sides must project the same column names.

```
MATCH (n:Person) RETURN n.name
UNION
MATCH (n:Employee) RETURN n.name
```

## MATCH … WITH … RETURN

`WITH` projects intermediate results (with optional aggregation and filtering) before the final `RETURN`. It supports the same expressions and aggregates as `RETURN`.

```
MATCH (n:Person)
WITH n.dept AS dept, count(*) AS cnt
WHERE cnt > 1
RETURN dept, cnt ORDER BY cnt DESC
```

## Modify data

```
MATCH (n:Person) WHERE n.name = "Alice" SET n.age = 31
MATCH (n:Person) WHERE n.name = "Alice" SET n:Manager
MATCH (n:Person) WHERE n.name = "Bob"   REMOVE n.age
MATCH (n:Person) WHERE n.name = "Bob"   REMOVE n:Employee
MATCH (n:Person) WHERE n.name = "Carol" DELETE n
MATCH (n:Person) WHERE n.name = "Carol" DETACH DELETE n   -- also removes incident edges
```

## Aggregate functions

Non-aggregate expressions in RETURN form an implicit GROUP BY key, as in SQL.

```
MATCH (n:Person) RETURN count(*)
MATCH (n:Person) RETURN n.dept, count(*) ORDER BY count(*) DESC
MATCH (n:Person) RETURN n.dept, sum(n.salary), avg(n.salary)
MATCH (n:Person) RETURN min(n.age), max(n.age)
MATCH (n:Person) RETURN collect(n.name)
```

| Function | Description |
|----------|-------------|
| `count(*)` | Count of matched rows |
| `count(expr)` | Count of non-null values |
| `sum(expr)` | Sum of numeric values |
| `avg(expr)` | Average of numeric values |
| `min(expr)` | Minimum value |
| `max(expr)` | Maximum value |
| `collect(expr)` | Gather all non-null values into a list |

## Scalar functions

**Graph functions**

| Function | Description |
|----------|-------------|
| `labels(n)` | List of labels on node `n` |
| `type(r)` | Label (type) of edge `r` |
| `length(path)` | Number of edges in a path variable |
| `id(n)` | ULID string ID of node or edge |
| `size(x)` | Length of a string or list |

**String functions**

| Function | Description |
|----------|-------------|
| `toLower(s)` | Lowercase string |
| `toUpper(s)` | Uppercase string |
| `trim(s)` | Remove leading/trailing whitespace |
| `ltrim(s)` / `rtrim(s)` | Remove leading / trailing whitespace |
| `split(s, delim)` | Split string into list |
| `replace(s, old, new)` | Replace all occurrences of `old` with `new` |
| `substring(s, start, len)` | Extract substring starting at `start` (0-based), up to `len` characters |
| `left(s, n)` | First `n` characters |
| `right(s, n)` | Last `n` characters |
| `startsWith(s, prefix)` | Boolean prefix check |
| `endsWith(s, suffix)` | Boolean suffix check |
| `contains(s, sub)` | Boolean substring check |
| `coalesce(a, b, ...)` | First non-null argument |

**Math functions**

| Function | Description |
|----------|-------------|
| `abs(x)` | Absolute value |
| `ceil(x)` / `ceiling(x)` | Round up to nearest integer |
| `floor(x)` | Round down to nearest integer |
| `round(x)` | Round to nearest integer |
| `sqrt(x)` | Square root |
| `sign(x)` | Sign: -1, 0, or 1 |
| `range(start, end)` | Integer list from `start` to `end` (inclusive) |

**Conversion functions**

| Function | Description |
|----------|-------------|
| `toString(x)` | Convert value to string |
| `toInteger(x)` | Convert value to integer |
| `toFloat(x)` | Convert value to float |

**Boolean / list functions**

| Function | Description |
|----------|-------------|
| `isNull(x)` | True if `x` is null |
| `exists(x)` | True if `x` is not null |
| `not(x)` | Boolean negation |
| `head(list)` | First element of a list |
| `last(list)` | Last element of a list |
| `tail(list)` | All elements except the first |
| `reverse(list)` | Reversed list |
| `keys(n)` | List of property key names for a node |

Column names in results match the expression text exactly. Use `AS` to rename: `RETURN type(r) AS rel_type`.

## Expressions and operators

```
WHERE n.age > 28 AND n.name <> "Bob"
WHERE n.age IS NULL
WHERE n.age IS NOT NULL
RETURN n.age * 2 + 1 AS adjusted
ORDER BY n.age DESC, n.name ASC
```

## Property indexes

Property indexes make point-lookups and range scans O(log n) instead of a linear label-scan.

```
CREATE INDEX ON :Person(name)   -- create an index
SHOW INDEXES                     -- list all indexes with entry counts
DROP INDEX ON :Person(name)      -- remove an index
```

After creating the index, queries use it automatically — no query changes needed:

| Predicate form | Index use |
|----------------|-----------|
| `{name: "Alice"}` or `WHERE n.name = "Alice"` | Point lookup — O(log n) |
| `WHERE n.age > 30` (also `<`, `>=`, `<=`, range) | Range scan — O(log n + matches) |
| `WHERE startsWith(n.sku, "BOOK")` | Prefix scan — O(log n + matches) |

```
CREATE INDEX ON :Product(sku)
MATCH (p:Product) WHERE startsWith(p.sku, "BOOK") RETURN p.title
-- Uses the index: scans only matching entries instead of all :Product nodes
```

Index definitions are persisted to the snapshot, so they survive restarts. The index data is rebuilt from stored nodes on load.

## Data constraints

Constraints enforce data quality at the schema level. Every `INSERT` and `SET` that would violate a constraint is rejected with a `Constraint violation` error.

```
CREATE CONSTRAINT UNIQUE ON :Person(email)
CREATE CONSTRAINT TYPE IS INTEGER ON :Person(age)
CREATE CONSTRAINT TYPE IS FLOAT   ON :Sensor(reading)
CREATE CONSTRAINT TYPE IS STRING  ON :Document(title)
CREATE CONSTRAINT TYPE IS BOOLEAN ON :Feature(enabled)
SHOW CONSTRAINTS
DROP CONSTRAINT UNIQUE ON :Person(email)
```

**`UNIQUE`** — no two nodes with the given label may share the same value for the property. Nodes that lack the property entirely are not affected.

```
CREATE CONSTRAINT UNIQUE ON :User(username)
INSERT (:User {username: "alice"})                       -- ok
INSERT (:User {username: "alice"})                       -- ok (upsert, same node reused)
INSERT (:User {username: "alice", role: "admin"})        -- ERROR: Constraint violation
```

**`TYPE IS <type>`** — every write to the property must produce a value of the declared type. Supported types: `INTEGER`, `FLOAT`, `STRING`, `BOOLEAN`.

```
CREATE CONSTRAINT TYPE IS INTEGER ON :Product(quantity)
INSERT (:Product {quantity: 10})       -- ok
INSERT (:Product {quantity: "ten"})    -- ERROR: Constraint violation
```

`SHOW CONSTRAINTS` returns one row per declared constraint with columns `label`, `property`, and `kind` (e.g. `"UNIQUE"` or `"TYPE IS INTEGER"`).

Constraint definitions are persisted to the `meta` column family and survive restarts.

> **Tip:** For `UNIQUE` constraints, creating a matching property index (`CREATE INDEX ON :User(username)`) makes the uniqueness check O(log n) instead of a linear scan.

## Parameterized queries

GQL strings may contain `$name` placeholders at any expression position. Parameters are supplied at call time via the Rust API or Python bindings, preventing injection issues and enabling efficient bulk operations.

```gql
-- Named placeholder in WHERE
MATCH (n:Person) WHERE n.email = $email RETURN n.name

-- Placeholder in LIMIT
MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT $k

-- UNWIND $list AS var INSERT — bulk insert in one statement
UNWIND $names AS name INSERT (:Person {name: name})
```

**`UNWIND expr AS var INSERT (…)`** evaluates `expr` to a list and inserts the given elements once per element. The loop variable can be a plain scalar or a `Map` (dict in Python), with properties accessed via dot notation:

```gql
UNWIND $people AS p INSERT (:Person {name: p.name, age: p.age})
```

**Rust API:**

```rust
use std::collections::HashMap;
use minigdb::{query_capturing_with_params, Value};

let mut params = HashMap::new();
params.insert("email".into(), Value::String("alice@example.com".into()));
query_capturing_with_params(
    "MATCH (n:Person) WHERE n.email = $email RETURN n.name",
    &mut graph, &mut txn_id, params,
)?;
```

**Python API:** see [Python API → Parameterized queries](python.md#parameterized-queries).

A missing parameter evaluates to `null` (and is falsy in `WHERE`), so omitting a param produces no matches rather than an error.

## Explicit transactions

```
BEGIN
INSERT (:Person {name: "Dave", age: 40})
INSERT (:Person {name: "Eve",  age: 22})
MATCH (a:Person {name: "Dave"}), (b:Person {name: "Eve"})
INSERT (a)-[:KNOWS]->(b)
COMMIT
```

- `BEGIN` — starts a transaction. The REPL prompt changes to `txn>`.
- `COMMIT` — writes all staged operations to the WAL as one atomic frame.
- `ROLLBACK` — discards all changes since `BEGIN` and restores the prior state.

Each statement inside a transaction is applied to the in-memory graph immediately (so a `MATCH` can see nodes inserted earlier in the same transaction), but nothing is written to the WAL until `COMMIT`.
