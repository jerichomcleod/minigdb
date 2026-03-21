# Graph Algorithms

[← Back to README](../README.md) · [Query Language](query-language.md) · [Server](server.md) · [Python API](python.md)

Algorithms are invoked with `CALL algorithmName(...) YIELD col1, col2, ...`. Parameters are passed as named key-value pairs; all have defaults and can be omitted.

```
CALL pageRank() YIELD node, score RETURN node, score ORDER BY score DESC LIMIT 10
CALL shortestPath(source: "01J...", target: "01J...") YIELD cost, path RETURN cost
CALL wcc(minSize: 2) YIELD node, component, size RETURN component, collect(node), size
```

---

## Traversal

| Algorithm | Key Parameters | Yields |
|-----------|----------------|--------|
| **bfs** | `start` (ULID, required), `maxDepth` (default: unlimited), `direction` ("out"/"in"/"any"), `algorithm` ("bfs"/"dfs") | `node`, `depth`, `predecessor` |
| **dfs** | Same as bfs with `algorithm: "dfs"` | `node`, `depth`, `predecessor` |

---

## Shortest paths

| Algorithm | Key Parameters | Yields |
|-----------|----------------|--------|
| **shortestPath** | `source` (ULID, required), `target` (ULID, optional — all nodes if omitted), `weight` (edge property name), `direction`, `maxCost` | `source`, `target`, `cost`, `path` |

Uses Dijkstra's algorithm. Requires non-negative edge weights when `weight` is specified.

---

## Connected components

| Algorithm | Key Parameters | Yields |
|-----------|----------------|--------|
| **wcc** | `minSize` (default: 1) | `node`, `component`, `size` |
| **scc** | `minSize` (default: 1) | `node`, `component`, `size` |

`wcc` finds weakly connected components (ignores edge direction). `scc` finds strongly connected components (Kosaraju's algorithm, respects direction).

---

## Centrality

| Algorithm | Key Parameters | Yields |
|-----------|----------------|--------|
| **pageRank** | `damping` (0.85), `iterations` (20), `weight`, `normalize` (true) | `node`, `score` |
| **betweennessCentrality** | `normalized` (true), `directed` (true), `weight`, `sampleSize` (0) | `node`, `score` |
| **closenessCentrality** | `normalized` (true), `direction` ("out"), `wfImproved` (true) | `node`, `score` |
| **degreeCentrality** | `direction` ("total"), `normalized` (true) | `node`, `inDegree`, `outDegree`, `totalDegree`, `score` |

`betweennessCentrality` uses the Brandes algorithm. `sampleSize > 0` samples that many source nodes for approximation on large graphs. `closenessCentrality` applies the Wasserman-Faust correction for disconnected graphs when `wfImproved` is true.

---

## Triangles

| Algorithm | Key Parameters | Yields |
|-----------|----------------|--------|
| **triangleCount** | `mode` ("local"/"global", default "local") | local: `node`, `triangles`, `coefficient`; global: `triangles`, `transitivity` |

---

## Similarity

| Algorithm | Key Parameters | Yields |
|-----------|----------------|--------|
| **jaccardSimilarity** | `node` (ULID, optional — all pairs if omitted), `direction` ("any"), `topK` (10), `method` ("jaccard"/"overlap"/"commonNeighbors") | `node1`, `node2`, `score` |

---

## Community detection

| Algorithm | Key Parameters | Yields |
|-----------|----------------|--------|
| **labelPropagation** | `maxIterations` (10), `direction` ("any") | `node`, `community` |
| **louvain** | `resolution` (1.0), `maxPasses` (10), `maxLevels` (10), `weight` | `node`, `community` |
| **leiden** | `resolution` (1.0), `maxPasses` (10), `maxLevels` (10), `theta` (0.01), `weight` | `node`, `community` |

`leiden` extends Louvain with a refinement phase that improves well-connectedness of communities. `theta` controls the refinement threshold.

---

## Flow

| Algorithm | Key Parameters | Yields |
|-----------|----------------|--------|
| **maxFlow** | `source` (ULID, required), `sink` (ULID, required), `capacity` (edge property, default "capacity") | `source`, `sink`, `maxFlow` |

Uses the Edmonds-Karp algorithm (BFS-augmented Ford-Fulkerson). Requires non-negative capacity values.
