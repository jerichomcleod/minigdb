//! RocksDB-backed storage for minigdb.
//!
//! # Column family layout
//!
//! Eight column families hold the complete property graph and its indexes:
//!
//! ```text
//! nodes          : id(16)                                        → bincode(Node)
//! edges          : id(16)                                        → bincode(Edge)
//! adj_out        : from(16) | edge_id(16)                        → to(16) | label(UTF-8)
//! adj_in         : to(16)   | edge_id(16)                        → from(16) | label(UTF-8)
//! label_idx      : label | \0 | node_id(16)                      → []
//! edge_label_idx : label | \0 | edge_id(16)                      → []
//! prop_idx       : label | \0 | prop | \0 | val | \0 | node_id(16) → []
//! meta           : arbitrary key                                 → arbitrary bytes
//! ```
//!
//! # Key encoding scheme
//!
//! All 128-bit node and edge IDs (ULIDs) are stored as **big-endian** byte
//! arrays so that RocksDB's default lexicographic key order matches ULID
//! temporal order — allowing time-ordered iteration and prefix scans over
//! adjacency and index CFs without any custom comparator.
//!
//! Variable-length key segments (labels, property names, encoded values) are
//! separated by a NUL byte (`\0`).  Because NUL cannot appear in valid UTF-8
//! label or property strings, this provides unambiguous segment boundaries
//! even when used as a prefix for range scans.
//!
//! Property-index values are encoded by `value_index_key()` in
//! `types/value.rs`.  The encoding uses type-prefixed, order-preserving hex
//! strings (`I:` for integers using XOR sign-bit, `F:` for IEEE-ordered
//! floats, `S:` for strings) so that lexicographic key order matches the
//! natural ordering of each type — enabling range scans over `prop_idx` with
//! no post-processing beyond a string comparison.
//!
//! # Migration
//!
//! The `edge_label_idx` CF was added after the initial schema.  On first open
//! of an existing database the caller must check for the
//! [`META_EDGE_LABEL_IDX_BUILT`] sentinel in the `meta` CF; if absent, it
//! backfills the index from the `edges` CF and then writes the sentinel so
//! subsequent opens skip the migration.

use std::path::Path;

use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamilyDescriptor, Direction, IteratorMode, Options,
    SliceTransform, WriteBatch, DB,
};

use crate::types::DbError;

// ── Column family names ───────────────────────────────────────────────────────

const CF_NODES: &str = "nodes";
const CF_EDGES: &str = "edges";
const CF_ADJ_OUT: &str = "adj_out";
const CF_ADJ_IN: &str = "adj_in";
const CF_LABEL_IDX: &str = "label_idx";
const CF_EDGE_LABEL_IDX: &str = "edge_label_idx";
const CF_PROP_IDX: &str = "prop_idx";
/// Edge property index CF — mirrors prop_idx but keyed by edge label.
const CF_EDGE_PROP_IDX: &str = "edge_prop_idx";
const CF_META: &str = "meta";

/// Ordered list of every column family name used by this schema.
pub(crate) const ALL_CFS: &[&str] = &[
    CF_NODES,
    CF_EDGES,
    CF_ADJ_OUT,
    CF_ADJ_IN,
    CF_LABEL_IDX,
    CF_EDGE_LABEL_IDX,
    CF_PROP_IDX,
    CF_EDGE_PROP_IDX,
    CF_META,
];

// ── Meta keys ────────────────────────────────────────────────────────────────

/// Meta CF key that stores the current node count as a little-endian `u64`.
///
/// Maintained in every write batch that inserts or removes a node so that
/// `node_count()` is an O(1) point lookup rather than a full CF scan.
pub(crate) const META_NODE_COUNT: &[u8] = b"node_count";

/// Meta CF key that stores the current edge count as a little-endian `u64`.
///
/// Maintained in every write batch that inserts or removes an edge so that
/// `edge_count()` is an O(1) point lookup rather than a full CF scan.
pub(crate) const META_EDGE_COUNT: &[u8] = b"edge_count";

/// Prefix for per-index entry-count keys in the `meta` CF.
///
/// The full key is `b"pidx_cnt\0" + label + b"\0" + property`.
/// Maintained alongside every `put_prop_entry` / `delete_prop_entry` so that
/// `SHOW INDEXES` is an O(1) meta lookup rather than a full prop_idx scan.
pub(crate) const META_PIDX_CNT_PREFIX: &[u8] = b"pidx_cnt\0";

/// Prefix for per-label node-count keys in the `meta` CF.
///
/// The full key is `b"label_cnt\0" + label`.  Maintained alongside every
/// `put_label_entry` / `delete_label_entry` so that label cardinality queries
/// are O(1) meta lookups rather than full `label_idx` scans.  Used by the
/// query planner for join-ordering (PERF-10).
///
/// INVARIANT: writers hold the per-graph mutex (server) or are single-threaded
/// (REPL), so the read-modify-write in `adjust_label_count_batch` is safe.
pub(crate) const META_LABEL_CNT_PREFIX: &[u8] = b"label_cnt\0";

/// Written once into the `meta` CF to signal that the `edge_label_idx` CF is
/// fully populated.
///
/// On first open of a database created before `edge_label_idx` was introduced
/// (phase R6), the caller checks for this key.  If absent, it backfills the
/// index from the `edges` CF and then writes this sentinel so that the
/// one-time migration is never repeated.
pub(crate) const META_EDGE_LABEL_IDX_BUILT: &[u8] = b"edge_label_idx_v1";

// ── RocksStore ────────────────────────────────────────────────────────────────

/// RocksDB-backed storage engine for a single graph database directory.
///
/// Each instance owns one open `DB` directory and exposes low-level read,
/// write, and scan operations over all eight column families.  Higher-level
/// graph semantics (node/edge structs, transaction buffering, index
/// maintenance) are handled by `StorageManager` and `Graph` in the layers
/// above.
///
/// All mutations that touch more than one CF are staged into a [`WriteBatch`]
/// by the caller and committed atomically via [`RocksStore::write`].
pub(crate) struct RocksStore {
    /// The underlying RocksDB instance.  Exposed as `pub(crate)` so that
    /// `StorageManager` can pass it to algorithm helpers that need direct
    /// iterator access without additional wrapper overhead.
    pub(crate) db: DB,
}

impl RocksStore {
    /// Open (or create) the RocksDB database at `path`.
    ///
    /// Creates `path` and all eight column families if they do not yet exist.
    /// If the directory already contains a RocksDB database with a subset of
    /// the expected CFs, `create_missing_column_families` causes RocksDB to
    /// add the missing ones automatically.
    ///
    /// Returns `DbError::RocksDb` if RocksDB reports an error during open.
    pub fn open(path: &Path) -> Result<Self, DbError> {
        // Ensure the parent directory exists before RocksDB tries to create
        // its own files inside it.
        std::fs::create_dir_all(path)?;

        // One shared block cache for all CFs — global LRU eviction.
        // Separate caches per CF would prevent hot CFs from borrowing capacity
        // from cold ones.  32 MiB default; sufficient for typical embedded use.
        let shared_cache = Cache::new_lru_cache(32 * 1024 * 1024);

        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        // Reserve half the available cores for query threads; cap compaction
        // background jobs at that value so they don't starve the foreground.
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        db_opts.set_max_background_jobs((cores / 2).max(2) as i32);

        let cf_descs: Vec<ColumnFamilyDescriptor> = ALL_CFS
            .iter()
            .map(|&name| {
                ColumnFamilyDescriptor::new(name, Self::cf_options(name, &shared_cache))
            })
            .collect();

        let db = DB::open_cf_descriptors(&db_opts, path, cf_descs)
            .map_err(|e| DbError::RocksDb(e.to_string()))?;

        Ok(Self { db })
    }

    /// Build per-CF RocksDB options tuned to each CF's access pattern.
    ///
    /// All CFs share a single `Cache` for global LRU eviction.  Point-lookup
    /// CFs (`nodes`, `edges`, `meta`) use whole-key bloom filters.  Fixed-
    /// prefix scan CFs (`adj_out`, `adj_in`) use prefix bloom filters with a
    /// matching `SliceTransform`.  Variable-prefix CFs (label/prop indexes) use
    /// whole-key blooms (no prefix extractor, since label lengths vary).
    ///
    /// NOTE: `set_optimize_filters_for_hits` is intentionally omitted.  That
    /// flag disables bloom filters on the bottommost SST level — exactly where
    /// most data lives.  It is only safe at 100% positive-lookup rates, which
    /// we cannot guarantee (e.g. lookups for deleted nodes).
    fn cf_options(name: &str, cache: &Cache) -> Options {
        let mut opts = Options::default();
        let mut block_opts = BlockBasedOptions::default();

        // Share the cache and keep bloom/index blocks in it so they survive
        // across block evictions.
        block_opts.set_block_cache(cache);
        block_opts.set_cache_index_and_filter_blocks(true);
        block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);

        match name {
            // Point-lookup CFs: whole-key bloom (no prefix extractor)
            CF_NODES | CF_EDGES | CF_META => {
                block_opts.set_bloom_filter(10.0, false);
            }
            // Fixed-prefix scan CFs: prefix bloom + SliceTransform on the
            // 16-byte node ID prefix so seeks skip non-matching blocks.
            CF_ADJ_OUT | CF_ADJ_IN => {
                block_opts.set_bloom_filter(10.0, true);
                opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(16));
            }
            // Variable-prefix CFs: whole-key bloom (label lengths vary, so no
            // fixed prefix extractor is safe here)
            _ => {
                block_opts.set_bloom_filter(10.0, false);
            }
        }

        opts.set_block_based_table_factory(&block_opts);
        opts.set_write_buffer_size(64 * 1024 * 1024); // 64 MiB memtable
        opts.set_max_write_buffer_number(3);
        opts
    }

    // ── Key encoding ─────────────────────────────────────────────────────────

    /// Encode a `u128` ID as 16 big-endian bytes.
    ///
    /// Big-endian byte order preserves ULID sort order under RocksDB's default
    /// lexicographic key comparison, so prefix scans and range queries over
    /// adjacency CFs naturally return results in ULID (insertion-time) order.
    #[inline]
    pub(crate) fn id_key(id: u128) -> [u8; 16] {
        id.to_be_bytes()
    }

    /// Decode the first 16 big-endian bytes of `b` as a `u128` ID.
    ///
    /// # Panics
    ///
    /// Panics if `b.len() < 16`.  This is intentional: all keys stored by
    /// this module have a known layout; a short key indicates DB corruption.
    #[inline]
    pub(crate) fn bytes_to_id(b: &[u8]) -> u128 {
        u128::from_be_bytes(b[..16].try_into().expect("slice must be ≥16 bytes"))
    }

    /// Build a 32-byte compound adjacency key: `[node_id: 16][edge_id: 16]`.
    ///
    /// This layout lets `scan_adj` seek directly to the first entry for a
    /// given node (16-byte prefix) and stop as soon as the prefix changes,
    /// giving O(degree) scan time without scanning unrelated entries.
    fn adj_key(node_id: u128, edge_id: u128) -> [u8; 32] {
        let mut k = [0u8; 32];
        k[..16].copy_from_slice(&node_id.to_be_bytes());
        k[16..].copy_from_slice(&edge_id.to_be_bytes());
        k
    }

    /// Build a node label-index key: `label | NUL | node_id(16)`.
    ///
    /// The NUL separator prevents a label that is a prefix of another (e.g.,
    /// `"Per"` vs `"Person"`) from producing false prefix matches during scans.
    fn label_key(label: &str, node_id: u128) -> Vec<u8> {
        let mut k = Vec::with_capacity(label.len() + 1 + 16);
        k.extend_from_slice(label.as_bytes());
        k.push(0); // NUL separator between label and node_id
        k.extend_from_slice(&node_id.to_be_bytes());
        k
    }

    /// Build the scan prefix for a node label: `label | NUL`.
    ///
    /// Used as the seek key and prefix guard when iterating over all nodes
    /// that carry a given label.
    fn label_prefix(label: &str) -> Vec<u8> {
        let mut p = Vec::with_capacity(label.len() + 1);
        p.extend_from_slice(label.as_bytes());
        p.push(0); // NUL terminates the label segment
        p
    }

    /// Build a property-index key:
    /// `label \0 prop \0 encoded_val \0 node_id(16)`.
    ///
    /// Each segment is NUL-terminated so that the value segment does not
    /// accidentally match a different value that shares its prefix.  The
    /// trailing 16-byte node ID is not NUL-terminated because it has a fixed
    /// length and is always the final segment.
    fn prop_key(label: &str, prop: &str, encoded_val: &str, node_id: u128) -> Vec<u8> {
        let mut k = Vec::new();
        k.extend_from_slice(label.as_bytes());       k.push(0); // NUL after label
        k.extend_from_slice(prop.as_bytes());         k.push(0); // NUL after prop name
        k.extend_from_slice(encoded_val.as_bytes());  k.push(0); // NUL after encoded value
        k.extend_from_slice(&node_id.to_be_bytes());             // fixed-width ID, no terminator
        k
    }

    /// Build the equality-scan prefix for a property value:
    /// `label \0 prop \0 encoded_val \0`.
    ///
    /// The trailing NUL ensures that only entries with exactly `encoded_val`
    /// are returned — not entries whose encoded value starts with `encoded_val`
    /// as a substring.
    fn prop_prefix(label: &str, prop: &str, encoded_val: &str) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(label.as_bytes());       p.push(0);
        p.extend_from_slice(prop.as_bytes());         p.push(0);
        p.extend_from_slice(encoded_val.as_bytes());  p.push(0);
        p
    }

    /// Build the range-scan prefix covering all entries for `(label, prop)`:
    /// `label \0 prop \0`.
    ///
    /// Used as both the seek start point and the prefix guard for
    /// `scan_prop_range` and `delete_prop_range`, where we want to iterate
    /// over all stored values of a given property regardless of their encoded
    /// value.
    fn prop_range_prefix(label: &str, prop: &str) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(label.as_bytes()); p.push(0); // NUL after label
        p.extend_from_slice(prop.as_bytes());  p.push(0); // NUL after prop name
        p
    }

    // ── Nodes CF ─────────────────────────────────────────────────────────────

    /// Write raw serialized bytes for a node directly to RocksDB (outside a batch).
    ///
    /// Prefer `put_node_batch` when the write is part of a multi-CF atomic
    /// operation.  This direct form is only used during database load / replay
    /// where individual ops are applied in sequence.
    #[cfg(test)]
    pub fn put_node_raw(&self, id: u128, data: &[u8]) -> Result<(), DbError> {
        let cf = self.db.cf_handle(CF_NODES).expect("nodes CF");
        self.db.put_cf(&cf, Self::id_key(id), data)
            .map_err(|e| DbError::RocksDb(e.to_string()))
    }

    /// Read raw serialized bytes for a node by its ID.
    ///
    /// Returns `None` if no node with that ID exists, or `DbError::RocksDb`
    /// on a storage-level failure.  The caller is responsible for
    /// deserializing the bytes with `bincode`.
    pub fn get_node_raw(&self, id: u128) -> Result<Option<Vec<u8>>, DbError> {
        let cf = self.db.cf_handle(CF_NODES).expect("nodes CF");
        self.db.get_cf(&cf, Self::id_key(id))
            .map_err(|e| DbError::RocksDb(e.to_string()))
    }

    /// Stage a node deletion into `batch`.
    ///
    /// The deletion is not visible until `batch` is committed via
    /// [`RocksStore::write`].  This is the preferred form for mutations that
    /// also touch adjacency or index CFs so that the entire logical operation
    /// is atomic.
    pub fn delete_node_batch(&self, batch: &mut WriteBatch, id: u128) {
        let cf = self.db.cf_handle(CF_NODES).expect("nodes CF");
        batch.delete_cf(&cf, Self::id_key(id));
    }

    /// Stage a node write into `batch`.
    ///
    /// Like `delete_node_batch`, the write is deferred until `batch` is
    /// committed.  Use this in conjunction with adjacency and index puts so
    /// that node data and its index entries are always written atomically.
    pub fn put_node_batch(&self, batch: &mut WriteBatch, id: u128, data: &[u8]) {
        let cf = self.db.cf_handle(CF_NODES).expect("nodes CF");
        batch.put_cf(&cf, Self::id_key(id), data);
    }

    /// Collect all node IDs present in the nodes CF.
    ///
    /// Performs a full sequential scan of the CF.  Used during graph load to
    /// reconstruct the in-memory node set from persistent storage.  Not
    /// suitable for hot paths on large graphs.
    pub fn all_node_ids(&self) -> Result<Vec<u128>, DbError> {
        let cf = self.db.cf_handle(CF_NODES).expect("nodes CF");
        let iter = self.db.iterator_cf(&cf, IteratorMode::Start);
        let mut ids = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;
            ids.push(Self::bytes_to_id(&key));
        }
        Ok(ids)
    }

    /// Like `all_node_ids` but stops after collecting `limit` IDs.
    ///
    /// Avoids a full CF scan when only a bounded prefix of nodes is needed
    /// (e.g. MATCH … RETURN … LIMIT N with no ORDER BY or WHERE).
    pub fn all_node_ids_limit(&self, limit: usize) -> Result<Vec<u128>, DbError> {
        let cf = self.db.cf_handle(CF_NODES).expect("nodes CF");
        let iter = self.db.iterator_cf(&cf, IteratorMode::Start);
        let mut ids = Vec::with_capacity(limit.min(1024));
        for item in iter {
            if ids.len() >= limit { break; }
            let (key, _) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;
            ids.push(Self::bytes_to_id(&key));
        }
        Ok(ids)
    }

    // ── Edges CF ─────────────────────────────────────────────────────────────

    /// Write raw serialized bytes for an edge directly to RocksDB (outside a batch).
    ///
    /// Mirrors `put_node_raw`; used during WAL replay where ops are applied
    /// sequentially rather than batched.
    pub fn put_edge_raw(&self, id: u128, data: &[u8]) -> Result<(), DbError> {
        let cf = self.db.cf_handle(CF_EDGES).expect("edges CF");
        self.db.put_cf(&cf, Self::id_key(id), data)
            .map_err(|e| DbError::RocksDb(e.to_string()))
    }

    /// Read raw serialized bytes for an edge by its ID.
    ///
    /// Returns `None` if no edge with that ID exists.  The caller deserializes
    /// the returned bytes with `bincode`.
    pub fn get_edge_raw(&self, id: u128) -> Result<Option<Vec<u8>>, DbError> {
        let cf = self.db.cf_handle(CF_EDGES).expect("edges CF");
        self.db.get_cf(&cf, Self::id_key(id))
            .map_err(|e| DbError::RocksDb(e.to_string()))
    }

    /// Stage an edge deletion into `batch`.
    ///
    /// Should be called alongside `delete_adj_out_batch`, `delete_adj_in_batch`,
    /// and `delete_edge_label_entry` so that all index entries for the edge are
    /// removed in the same atomic write.
    pub fn delete_edge_batch(&self, batch: &mut WriteBatch, id: u128) {
        let cf = self.db.cf_handle(CF_EDGES).expect("edges CF");
        batch.delete_cf(&cf, Self::id_key(id));
    }

    /// Stage an edge write into `batch`.
    ///
    /// Should be paired with `put_adj_out`, `put_adj_in`, and
    /// `put_edge_label_entry` in the same batch so that edge data and all its
    /// index entries are always kept consistent.
    pub fn put_edge_batch(&self, batch: &mut WriteBatch, id: u128, data: &[u8]) {
        let cf = self.db.cf_handle(CF_EDGES).expect("edges CF");
        batch.put_cf(&cf, Self::id_key(id), data);
    }

    /// Collect all edge IDs present in the edges CF.
    ///
    /// Performs a full sequential scan.  Used during graph load and the
    /// one-time `edge_label_idx` migration (phase R6).
    pub fn all_edge_ids(&self) -> Result<Vec<u128>, DbError> {
        let cf = self.db.cf_handle(CF_EDGES).expect("edges CF");
        let iter = self.db.iterator_cf(&cf, IteratorMode::Start);
        let mut ids = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;
            ids.push(Self::bytes_to_id(&key));
        }
        Ok(ids)
    }

    // ── Adjacency CFs ─────────────────────────────────────────────────────────

    /// Stage an outgoing-adjacency entry into `batch`.
    ///
    /// Key: `from(16) | edge_id(16)`.  Value: `to(16) | label(UTF-8)`.
    ///
    /// The value encodes both the destination node ID and the edge label so
    /// that graph traversal can read everything it needs from one CF entry
    /// without a separate lookup in the `edges` CF.
    pub fn put_adj_out(
        &self,
        batch: &mut WriteBatch,
        from: u128,
        edge_id: u128,
        to: u128,
        label: &str,
    ) {
        let cf = self.db.cf_handle(CF_ADJ_OUT).expect("adj_out CF");
        // Pack the destination node ID (fixed 16 bytes) followed by the label
        // (variable-length UTF-8) into a single value.
        let mut val = Vec::with_capacity(16 + label.len());
        val.extend_from_slice(&to.to_be_bytes());
        val.extend_from_slice(label.as_bytes());
        batch.put_cf(&cf, Self::adj_key(from, edge_id), val);
    }

    /// Stage an incoming-adjacency entry into `batch`.
    ///
    /// Key: `to(16) | edge_id(16)`.  Value: `from(16) | label(UTF-8)`.
    ///
    /// Mirrors `put_adj_out` but for the reverse direction, enabling efficient
    /// in-edge traversal without scanning `adj_out` for the entire graph.
    pub fn put_adj_in(
        &self,
        batch: &mut WriteBatch,
        to: u128,
        edge_id: u128,
        from: u128,
        label: &str,
    ) {
        let cf = self.db.cf_handle(CF_ADJ_IN).expect("adj_in CF");
        // Value layout mirrors adj_out: the "other" node ID first, then label.
        let mut val = Vec::with_capacity(16 + label.len());
        val.extend_from_slice(&from.to_be_bytes());
        val.extend_from_slice(label.as_bytes());
        batch.put_cf(&cf, Self::adj_key(to, edge_id), val);
    }

    /// Stage deletion of an outgoing-adjacency entry from `batch`.
    ///
    /// Must be called alongside `delete_adj_in_batch` and `delete_edge_batch`
    /// to keep the two adjacency CFs and the edges CF in sync.
    pub fn delete_adj_out_batch(&self, batch: &mut WriteBatch, from: u128, edge_id: u128) {
        let cf = self.db.cf_handle(CF_ADJ_OUT).expect("adj_out CF");
        batch.delete_cf(&cf, Self::adj_key(from, edge_id));
    }

    /// Stage deletion of an incoming-adjacency entry from `batch`.
    ///
    /// Must be called alongside `delete_adj_out_batch` and `delete_edge_batch`
    /// to keep both adjacency CFs in sync with the edges CF.
    pub fn delete_adj_in_batch(&self, batch: &mut WriteBatch, to: u128, edge_id: u128) {
        let cf = self.db.cf_handle(CF_ADJ_IN).expect("adj_in CF");
        batch.delete_cf(&cf, Self::adj_key(to, edge_id));
    }

    /// Scan all outgoing edges from `from`.
    ///
    /// Returns `(edge_id, to_node_id, label)` for each entry, in ULID order
    /// of edge insertion (because both IDs are big-endian and ULID is
    /// time-ordered).
    pub fn scan_adj_out(&self, from: u128) -> Result<Vec<(u128, u128, String)>, DbError> {
        self.scan_adj(CF_ADJ_OUT, from)
    }

    /// Scan all incoming edges to `to`.
    ///
    /// Returns `(edge_id, from_node_id, label)` for each entry, in ULID order
    /// of edge insertion.
    pub fn scan_adj_in(&self, to: u128) -> Result<Vec<(u128, u128, String)>, DbError> {
        self.scan_adj(CF_ADJ_IN, to)
    }

    /// Scan the entire `adj_out` CF and yield `(from, edge_id, to, label)` for
    /// every entry.  Used at graph-open time to build the in-memory adjacency
    /// list (ARCH-1).  Not suitable for per-node hot paths.
    pub fn iter_all_adj_out(&self) -> Result<Vec<(u128, u128, u128, String)>, DbError> {
        let cf = self.db.cf_handle(CF_ADJ_OUT).expect("adj_out CF");
        let iter = self.db.iterator_cf(&cf, IteratorMode::Start);
        let mut entries = Vec::new();
        for item in iter {
            let (key, val) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;
            if key.len() < 32 || val.len() < 16 {
                continue; // corrupt entry — skip silently
            }
            let from     = Self::bytes_to_id(&key[..16]);
            let edge_id  = Self::bytes_to_id(&key[16..32]);
            let to       = Self::bytes_to_id(&val[..16]);
            let label    = String::from_utf8_lossy(&val[16..]).into_owned();
            entries.push((from, edge_id, to, label));
        }
        Ok(entries)
    }

    /// Shared implementation for `scan_adj_out` and `scan_adj_in`.
    ///
    /// Seeks to the 16-byte big-endian prefix for `node_id` and iterates
    /// forward until the prefix changes, collecting `(edge_id, other_id, label)`
    /// tuples from each entry.
    ///
    /// The `cf_name` parameter is either `CF_ADJ_OUT` or `CF_ADJ_IN`; the key
    /// and value layouts are symmetric — key is always `[node_id:16][edge_id:16]`
    /// and value is always `[other_node_id:16][label:variable]`.
    fn scan_adj(&self, cf_name: &str, node_id: u128) -> Result<Vec<(u128, u128, String)>, DbError> {
        // Seek directly to the first key with this node's 16-byte prefix.
        let prefix = node_id.to_be_bytes();
        let cf = self.db.cf_handle(cf_name).expect("adjacency CF");
        let mode = IteratorMode::From(prefix.as_ref(), Direction::Forward);
        let iter = self.db.iterator_cf(&cf, mode);

        let mut results = Vec::new();
        for item in iter {
            let (key, val) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;

            // Stop as soon as we step past entries for this node.
            // A key shorter than 32 bytes would be malformed; treat it as the
            // end of this node's range.
            if key.len() < 32 || key[..16] != prefix { break; }

            // Extract the edge ID from bytes 16..32 of the key.
            let edge_id = Self::bytes_to_id(&key[16..]);

            // Extract the peer node ID from the first 16 bytes of the value,
            // then decode the label from the remaining variable-length bytes.
            let other_id = Self::bytes_to_id(&val);
            let label = String::from_utf8_lossy(&val[16..]).into_owned();

            results.push((edge_id, other_id, label));
        }
        Ok(results)
    }

    // ── Label index CF ────────────────────────────────────────────────────────

    /// Stage a label-index insertion: `label \0 node_id → []`.
    ///
    /// The empty value `b""` is intentional — all information is encoded in
    /// the key itself, so no value bytes are needed.  This is the standard
    /// pattern for index-only CFs where iteration over keys is sufficient.
    pub fn put_label_entry(&self, batch: &mut WriteBatch, label: &str, node_id: u128) {
        let cf = self.db.cf_handle(CF_LABEL_IDX).expect("label_idx CF");
        batch.put_cf(&cf, Self::label_key(label, node_id), b"");
    }

    /// Stage deletion of a label-index entry.
    ///
    /// Called when a node is removed or when its label is changed (old label
    /// deleted, new label inserted) so that `scan_label` never returns stale IDs.
    pub fn delete_label_entry(&self, batch: &mut WriteBatch, label: &str, node_id: u128) {
        let cf = self.db.cf_handle(CF_LABEL_IDX).expect("label_idx CF");
        batch.delete_cf(&cf, Self::label_key(label, node_id));
    }

    /// Scan all node IDs that carry `label`.
    ///
    /// Seeks to `label \0` and iterates forward, stopping as soon as the key
    /// no longer starts with that prefix.  Returns IDs in ULID order (big-endian
    /// sort order of the trailing 16 bytes).
    pub fn scan_label(&self, label: &str) -> Result<Vec<u128>, DbError> {
        let prefix = Self::label_prefix(label);
        let cf = self.db.cf_handle(CF_LABEL_IDX).expect("label_idx CF");
        let mode = IteratorMode::From(prefix.as_slice(), Direction::Forward);
        let iter = self.db.iterator_cf(&cf, mode);

        let mut ids = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;

            // Stop when the key's label prefix no longer matches.
            if !key.starts_with(prefix.as_slice()) { break; }

            // The node ID occupies the last 16 bytes after the prefix.
            if key.len() >= prefix.len() + 16 {
                ids.push(Self::bytes_to_id(&key[prefix.len()..]));
            }
        }
        Ok(ids)
    }

    /// Like `scan_label` but stops after collecting `limit` IDs.
    pub fn scan_label_limit(&self, label: &str, limit: usize) -> Result<Vec<u128>, DbError> {
        let prefix = Self::label_prefix(label);
        let cf = self.db.cf_handle(CF_LABEL_IDX).expect("label_idx CF");
        let mode = IteratorMode::From(prefix.as_slice(), Direction::Forward);
        let iter = self.db.iterator_cf(&cf, mode);

        let mut ids = Vec::new();
        for item in iter {
            if ids.len() >= limit { break; }
            let (key, _) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;
            if !key.starts_with(prefix.as_slice()) { break; }
            if key.len() >= prefix.len() + 16 {
                ids.push(Self::bytes_to_id(&key[prefix.len()..]));
            }
        }
        Ok(ids)
    }

    // ── Edge label index CF ───────────────────────────────────────────────────

    /// Build an edge-label-index key: `label | NUL | edge_id(16)`.
    ///
    /// Matches the structure of `label_key` but for edges, enabling O(matches)
    /// lookup of all edges with a given label without scanning the full edges CF.
    fn edge_label_key(label: &str, edge_id: u128) -> Vec<u8> {
        let mut k = Vec::with_capacity(label.len() + 1 + 16);
        k.extend_from_slice(label.as_bytes());
        k.push(0); // NUL separator
        k.extend_from_slice(&edge_id.to_be_bytes());
        k
    }

    /// Build the scan prefix for an edge label: `label | NUL`.
    ///
    /// Used by `scan_edge_label` to seek to the first matching entry and as
    /// the prefix guard to stop iteration once we've passed all entries for
    /// this label.
    fn edge_label_prefix(label: &str) -> Vec<u8> {
        let mut p = Vec::with_capacity(label.len() + 1);
        p.extend_from_slice(label.as_bytes());
        p.push(0); // NUL terminates the label segment
        p
    }

    /// Stage an edge-label-index insertion: `label \0 edge_id → []`.
    ///
    /// Must be included in every write batch that inserts a new edge so that
    /// `scan_edge_label` returns accurate results immediately after commit.
    pub fn put_edge_label_entry(&self, batch: &mut WriteBatch, label: &str, edge_id: u128) {
        let cf = self.db.cf_handle(CF_EDGE_LABEL_IDX).expect("edge_label_idx CF");
        batch.put_cf(&cf, Self::edge_label_key(label, edge_id), b"");
    }

    /// Stage deletion of an edge-label-index entry.
    ///
    /// Must be included in every write batch that removes an edge so that the
    /// index does not retain stale entries for deleted edges.
    pub fn delete_edge_label_entry(&self, batch: &mut WriteBatch, label: &str, edge_id: u128) {
        let cf = self.db.cf_handle(CF_EDGE_LABEL_IDX).expect("edge_label_idx CF");
        batch.delete_cf(&cf, Self::edge_label_key(label, edge_id));
    }

    /// Return all edge IDs with the given label (O(matches) scan).
    ///
    /// Seeks to `label \0` in `edge_label_idx` and iterates forward until the
    /// prefix changes.  Much faster than scanning all edges and deserializing
    /// each one to check its label, especially for sparse labels.
    pub fn scan_edge_label(&self, label: &str) -> Result<Vec<u128>, DbError> {
        let prefix = Self::edge_label_prefix(label);
        let cf = self.db.cf_handle(CF_EDGE_LABEL_IDX).expect("edge_label_idx CF");
        let mode = IteratorMode::From(prefix.as_slice(), Direction::Forward);
        let iter = self.db.iterator_cf(&cf, mode);

        let mut ids = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;

            // Stop when the key's label prefix no longer matches.
            if !key.starts_with(prefix.as_slice()) { break; }

            // The edge ID occupies the last 16 bytes after the label prefix.
            if key.len() >= prefix.len() + 16 {
                ids.push(Self::bytes_to_id(&key[prefix.len()..]));
            }
        }
        Ok(ids)
    }

    // ── Property index CF ─────────────────────────────────────────────────────

    /// Stage a property-index insertion:
    /// `label \0 prop \0 encoded_val \0 node_id → []`.
    ///
    /// `encoded_val` must be produced by `value_index_key()` in
    /// `types/value.rs` so that the order-preserving encoding invariant holds
    /// and range scans via `scan_prop_range` return correct results.
    pub fn put_prop_entry(
        &self,
        batch: &mut WriteBatch,
        label: &str,
        prop: &str,
        encoded_val: &str,
        node_id: u128,
    ) {
        let cf = self.db.cf_handle(CF_PROP_IDX).expect("prop_idx CF");
        batch.put_cf(&cf, Self::prop_key(label, prop, encoded_val, node_id), b"");
    }

    /// Stage deletion of a property-index entry.
    ///
    /// Called when a node property is removed or updated (old entry deleted,
    /// new entry inserted) so that equality and range scans never return nodes
    /// with stale property values.
    pub fn delete_prop_entry(
        &self,
        batch: &mut WriteBatch,
        label: &str,
        prop: &str,
        encoded_val: &str,
        node_id: u128,
    ) {
        let cf = self.db.cf_handle(CF_PROP_IDX).expect("prop_idx CF");
        batch.delete_cf(&cf, Self::prop_key(label, prop, encoded_val, node_id));
    }

    /// Scan all node IDs whose `(label, prop)` equals `encoded_val` (equality lookup).
    ///
    /// Seeks to the full `label \0 prop \0 encoded_val \0` prefix and iterates
    /// forward until the prefix changes.  O(matches) — the cost is proportional
    /// to the number of nodes with exactly that property value, not the total
    /// number of nodes or index entries.
    pub fn scan_prop(
        &self,
        label: &str,
        prop: &str,
        encoded_val: &str,
    ) -> Result<Vec<u128>, DbError> {
        let prefix = Self::prop_prefix(label, prop, encoded_val);
        let cf = self.db.cf_handle(CF_PROP_IDX).expect("prop_idx CF");
        let mode = IteratorMode::From(prefix.as_slice(), Direction::Forward);
        let iter = self.db.iterator_cf(&cf, mode);

        let mut ids = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;

            // Stop when we've passed all entries for this exact value.
            if !key.starts_with(prefix.as_slice()) { break; }

            // The node ID is the final 16 bytes of the key.
            if key.len() >= prefix.len() + 16 {
                ids.push(Self::bytes_to_id(&key[prefix.len()..]));
            }
        }
        Ok(ids)
    }

    /// Scan node IDs whose `(label, prop)` index value falls within a range.
    ///
    /// `lo` and `hi` are optional `(encoded_val, inclusive)` bound pairs
    /// produced by `value_index_key()`.  Because that encoding is
    /// order-preserving, a lexicographic string comparison on the encoded
    /// values is equivalent to the natural numeric or string ordering of the
    /// original `Value`.
    ///
    /// - When `lo` is given, the iterator seeks directly to that lower bound
    ///   key, skipping all earlier entries in O(log n) time.
    /// - When `hi` is given, iteration stops as soon as the encoded value
    ///   exceeds the upper bound, giving O(log n + matches) overall cost.
    /// - When both bounds are `None`, all entries for `(label, prop)` are
    ///   returned (equivalent to `scan_prop` with a wildcard value).
    ///
    /// Returns matching node IDs in ascending encoded-value order.
    pub fn scan_prop_range(
        &self,
        label: &str,
        prop: &str,
        lo: Option<(&str, bool)>, // (encoded_val, inclusive)
        hi: Option<(&str, bool)>, // (encoded_val, inclusive)
    ) -> Result<Vec<u128>, DbError> {
        let range_pfx = Self::prop_range_prefix(label, prop);

        // If a lower bound is given, seek directly to it so we skip all
        // entries with smaller values in O(log n).  Otherwise seek to the
        // start of the (label, prop) range.
        let seek_key: Vec<u8> = if let Some((lo_val, _)) = lo {
            let mut k = range_pfx.clone();
            k.extend_from_slice(lo_val.as_bytes());
            k.push(0); // \0 separates the encoded value from the node_id
            k
        } else {
            range_pfx.clone()
        };

        let cf = self.db.cf_handle(CF_PROP_IDX).expect("prop_idx CF");
        let mode = IteratorMode::From(seek_key.as_slice(), Direction::Forward);
        let iter = self.db.iterator_cf(&cf, mode);

        let mut ids = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;

            // Stop when we've left the (label, prop) prefix entirely.
            if !key.starts_with(range_pfx.as_slice()) {
                break;
            }

            // Key layout after range_pfx: encoded_val | \0 | node_id (16 bytes).
            // The node_id is the last 16 bytes; the \0 separator is at
            // key.len() - 17 (one byte before the 16-byte ID).
            if key.len() < range_pfx.len() + 17 {
                continue; // malformed key — skip rather than panic
            }

            // Isolate the encoded value segment by slicing from the end of
            // range_pfx to the position just before the \0|node_id trailer.
            let enc_end = key.len() - 17; // exclusive end of encoded_val bytes
            let enc_bytes = &key[range_pfx.len()..enc_end];
            let enc_val = match std::str::from_utf8(enc_bytes) {
                Ok(s) => s,
                Err(_) => continue, // non-UTF-8 encoded value; should never happen
            };

            // Lower-bound check.  The seek already positions the iterator at
            // or past the lower bound, so most iterations skip this branch.
            // The explicit check is needed to handle the exclusive-lower-bound
            // case (`lo_incl = false`) where the seek lands exactly on the
            // bound key and we must skip it.
            if let Some((lo_val, lo_incl)) = lo {
                let cmp = enc_val.cmp(lo_val);
                if cmp == std::cmp::Ordering::Less {
                    continue; // should not happen after the seek, but be safe
                }
                if cmp == std::cmp::Ordering::Equal && !lo_incl {
                    continue; // exclusive lower bound: skip the exact bound value
                }
            }

            // Upper-bound check.  Break (not continue) because all subsequent
            // entries will have larger encoded values due to the sorted order.
            if let Some((hi_val, hi_incl)) = hi {
                let cmp = enc_val.cmp(hi_val);
                if cmp == std::cmp::Ordering::Greater {
                    break; // past the upper bound; no more matches possible
                }
                if cmp == std::cmp::Ordering::Equal && !hi_incl {
                    break; // exclusive upper bound: stop before the exact bound value
                }
            }

            // The node ID immediately follows the \0 separator at enc_end.
            ids.push(Self::bytes_to_id(&key[enc_end + 1..]));
        }
        Ok(ids)
    }

    /// Scan node IDs whose `(label, prop)` string value starts with `string_prefix`.
    ///
    /// The prop_idx encodes string values as `"S:<value>"`.  This method seeks to
    /// `label\0prop\0S:<string_prefix>` and iterates forward, collecting node IDs
    /// while the encoded value has the `"S:<string_prefix>"` prefix.  When the
    /// encoded value diverges (e.g. the next lexicographic string no longer starts
    /// with `string_prefix`), iteration stops.
    ///
    /// The seek is O(log n); the iteration is O(matches).  Returns an empty vec
    /// if the index has no matching entries.
    pub fn scan_prop_prefix(
        &self,
        label: &str,
        prop: &str,
        string_prefix: &str,
    ) -> Result<Vec<u128>, DbError> {
        let range_pfx = Self::prop_range_prefix(label, prop);
        // String values in the prop_idx are encoded as "S:<original_value>".
        let encoded_prefix = format!("S:{}", string_prefix);

        // Seek directly to the first possible match.
        let mut seek_key = range_pfx.clone();
        seek_key.extend_from_slice(encoded_prefix.as_bytes());

        let cf = self.db.cf_handle(CF_PROP_IDX).expect("prop_idx CF");
        let mode = IteratorMode::From(seek_key.as_slice(), Direction::Forward);
        let iter = self.db.iterator_cf(&cf, mode);

        let mut ids = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;

            // Stop when we've left the (label, prop) namespace entirely.
            if !key.starts_with(range_pfx.as_slice()) {
                break;
            }

            // Key layout: range_pfx | encoded_val | \0 | node_id (16 bytes be).
            if key.len() < range_pfx.len() + 17 {
                continue;
            }

            let enc_end = key.len() - 17;
            let enc_bytes = &key[range_pfx.len()..enc_end];
            let enc_val = match std::str::from_utf8(enc_bytes) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Entries are sorted, so once the encoded value no longer matches
            // our prefix there cannot be any further matches.
            if !enc_val.starts_with(encoded_prefix.as_str()) {
                break;
            }

            ids.push(Self::bytes_to_id(&key[enc_end + 1..]));
        }
        Ok(ids)
    }

    /// Delete every `prop_idx` entry whose key starts with `label \0 prop \0`.
    ///
    /// Used by `DROP INDEX ON :Label(prop)` to remove all stored index data
    /// for a property index in a single atomic write.
    ///
    /// This function uses a two-phase collect-then-delete approach rather than
    /// deleting inside the iterator loop, because RocksDB's iterator is
    /// invalidated if the underlying data changes while it is open.  Collecting
    /// all matching keys first and then deleting them in a single `WriteBatch`
    /// avoids that hazard and also makes the deletion atomic.
    pub fn delete_prop_range(&self, label: &str, prop: &str) -> Result<(), DbError> {
        let prefix = Self::prop_range_prefix(label, prop);
        let cf = self.db.cf_handle(CF_PROP_IDX).expect("prop_idx CF");
        let mode = IteratorMode::From(prefix.as_slice(), Direction::Forward);

        // Phase 1: collect all matching keys without modifying the iterator.
        let keys: Vec<Box<[u8]>> = self.db.iterator_cf(&cf, mode)
            .map_while(|item| item.ok())
            .take_while(|(key, _)| key.starts_with(prefix.as_slice()))
            .map(|(key, _)| key)
            .collect();

        if keys.is_empty() { return Ok(()); }

        // Phase 2: delete all collected keys in a single atomic batch.
        // Re-acquire the CF handle because the borrow from phase 1 has ended.
        let mut batch = WriteBatch::default();
        let cf2 = self.db.cf_handle(CF_PROP_IDX).expect("prop_idx CF");
        for key in &keys {
            batch.delete_cf(&cf2, key.as_ref());
        }
        self.db.write(batch).map_err(|e| DbError::RocksDb(e.to_string()))
    }

    // ── prop_idx entry counter ───────────────────────────────────────────────

    /// Build the meta key for the prop_idx entry-count for `(label, prop)`.
    fn prop_count_meta_key(label: &str, prop: &str) -> Vec<u8> {
        let mut k = META_PIDX_CNT_PREFIX.to_vec();
        k.extend_from_slice(label.as_bytes());
        k.push(0);
        k.extend_from_slice(prop.as_bytes());
        k
    }

    /// Return the persisted entry count for the `(label, prop)` index.
    ///
    /// O(1) meta CF point lookup — replaces the old `count_prop_entries` full scan.
    /// Returns 0 for indexes created before this counter was introduced (they
    /// will be updated on the next write; callers can also call
    /// `rebuild_prop_count` for an immediate repair).
    pub fn get_prop_count(&self, label: &str, prop: &str) -> usize {
        let key = Self::prop_count_meta_key(label, prop);
        self.get_meta(&key)
            .ok()
            .flatten()
            .map(|v| u64_from_le(&v) as usize)
            .unwrap_or(0)
    }


    /// Overwrite the persisted entry count with an exact value.
    ///
    /// Used during `CREATE INDEX` after the full index build loop so a single
    /// accurate count is written rather than incrementing once per node.
    pub fn set_prop_count_batch(&self, batch: &mut WriteBatch, label: &str, prop: &str, count: usize) {
        let key = Self::prop_count_meta_key(label, prop);
        self.put_meta_batch(batch, &key, &(count as u64).to_le_bytes());
    }

    /// Delete the persisted entry count for `(label, prop)` (used on DROP INDEX).
    pub fn delete_prop_count(&self, label: &str, prop: &str) -> Result<(), DbError> {
        let key = Self::prop_count_meta_key(label, prop);
        let cf = self.db.cf_handle(CF_META).expect("meta CF");
        self.db.delete_cf(&cf, &key).map_err(|e| DbError::RocksDb(e.to_string()))
    }

    // ── label_idx node-count counters ────────────────────────────────────────

    /// Build the meta key for the node count under `label`.
    fn label_count_meta_key(label: &str) -> Vec<u8> {
        let mut k = META_LABEL_CNT_PREFIX.to_vec();
        k.extend_from_slice(label.as_bytes());
        k
    }

    /// Return the persisted node count for `label` — O(1) meta point lookup.
    ///
    /// Returns 0 for labels that existed before this counter was introduced;
    /// the counter is lazily initialized on the next write via
    /// `adjust_label_count_batch`.
    pub fn get_label_count(&self, label: &str) -> usize {
        let key = Self::label_count_meta_key(label);
        self.get_meta(&key)
            .ok()
            .flatten()
            .map(|v| u64_from_le(&v) as usize)
            .unwrap_or(0)
    }

    /// Like `get_label_count` but returns `None` when the counter has never
    /// been written (databases created before the label-count meta key was
    /// introduced).  Used by the query executor's COUNT fast path to avoid
    /// incorrectly returning 0 for a label that has nodes but no stored counter.
    pub fn get_label_count_if_known(&self, label: &str) -> Option<usize> {
        let key = Self::label_count_meta_key(label);
        self.get_meta(&key)
            .ok()
            .flatten()
            .map(|v| u64_from_le(&v) as usize)
    }

    /// Count nodes with `label` by doing a full `label_idx` prefix scan.
    ///
    /// Expensive — only called once per label for lazy migration when
    /// `adjust_label_count_batch` finds no persisted counter yet.
    pub(crate) fn count_label_entries(&self, label: &str) -> usize {
        self.scan_label(label).unwrap_or_default().len()
    }

    /// Adjust the persisted node count for `label` by `delta` (+1 or −1).
    ///
    /// Staged into `batch` so the counter update is atomic with the
    /// corresponding `put_label_entry` / `delete_label_entry` call.
    ///
    /// On first write to a label that has no persisted counter (pre-existing
    /// data), initializes the counter via a one-time `label_idx` scan so it
    /// stays accurate going forward.
    pub fn adjust_label_count_batch(
        &self,
        batch: &mut WriteBatch,
        label: &str,
        delta: i64,
    ) {
        let key = Self::label_count_meta_key(label);
        let cur = match self.get_meta(&key) {
            Ok(Some(v)) => u64_from_le(&v) as i64,
            // Counter absent — lazy migration: scan once to initialize.
            Ok(None) => self.count_label_entries(label) as i64,
            Err(_) => 0,
        };
        let new_val = (cur + delta).max(0) as u64;
        self.put_meta_batch(batch, &key, &new_val.to_le_bytes());
    }

    /// Delete the persisted node count for `label`.
    #[allow(dead_code)]
    pub fn delete_label_count(&self, label: &str) -> Result<(), DbError> {
        let key = Self::label_count_meta_key(label);
        let cf = self.db.cf_handle(CF_META).expect("meta CF");
        self.db
            .delete_cf(&cf, &key)
            .map_err(|e| DbError::RocksDb(e.to_string()))
    }

    // ── Meta CF ───────────────────────────────────────────────────────────────

    /// Read a raw value from the `meta` CF by key.
    ///
    /// Returns `None` if the key has not been written yet.  Used for both
    /// structured values (counters encoded as little-endian `u64`) and boolean
    /// sentinels (e.g., [`META_EDGE_LABEL_IDX_BUILT`]) where the presence of
    /// the key rather than its content carries the meaning.
    pub fn get_meta(&self, key: &[u8]) -> Result<Option<Vec<u8>>, DbError> {
        let cf = self.db.cf_handle(CF_META).expect("meta CF");
        self.db.get_cf(&cf, key).map_err(|e| DbError::RocksDb(e.to_string()))
    }

    /// Write a raw value to the `meta` CF immediately (outside a batch).
    ///
    /// Use `put_meta_batch` when the write must be part of an atomic
    /// multi-CF operation.  This direct form is suitable for one-time
    /// administrative writes such as migration sentinels.
    pub fn put_meta(&self, key: &[u8], val: &[u8]) -> Result<(), DbError> {
        let cf = self.db.cf_handle(CF_META).expect("meta CF");
        self.db.put_cf(&cf, key, val).map_err(|e| DbError::RocksDb(e.to_string()))
    }

    /// Stage a meta write into `batch`.
    ///
    /// Used to atomically update counters (node/edge count) in the same batch
    /// as the corresponding node or edge mutation so they never become
    /// inconsistent with the actual CF contents.
    pub fn put_meta_batch(&self, batch: &mut WriteBatch, key: &[u8], val: &[u8]) {
        let cf = self.db.cf_handle(CF_META).expect("meta CF");
        batch.put_cf(&cf, key, val);
    }

    // ── Counters ──────────────────────────────────────────────────────────────

    /// Return the current node count stored in the `meta` CF.
    ///
    /// Returns `0` if the counter has never been written (i.e., on a
    /// freshly-created database).
    pub fn node_count(&self) -> Result<u64, DbError> {
        Ok(self.get_meta(META_NODE_COUNT)?.map(|v| u64_from_le(&v)).unwrap_or(0))
    }

    /// Return the current edge count stored in the `meta` CF.
    ///
    /// Returns `0` if the counter has never been written.
    pub fn edge_count(&self) -> Result<u64, DbError> {
        Ok(self.get_meta(META_EDGE_COUNT)?.map(|v| u64_from_le(&v)).unwrap_or(0))
    }

    /// Persist the node count to the `meta` CF (immediate, outside a batch).
    ///
    /// Prefer updating the counter inside the same `WriteBatch` as the node
    /// mutation via `put_meta_batch` to keep the counter atomic with the data.
    /// This direct form is used during load-time reconstruction.
    #[cfg(test)]
    pub fn set_node_count(&self, n: u64) -> Result<(), DbError> {
        self.put_meta(META_NODE_COUNT, &n.to_le_bytes())
    }

    /// Persist the edge count to the `meta` CF (immediate, outside a batch).
    ///
    /// See `set_node_count` for the same caveat about batch vs. direct writes.
    #[cfg(test)]
    pub fn set_edge_count(&self, n: u64) -> Result<(), DbError> {
        self.put_meta(META_EDGE_COUNT, &n.to_le_bytes())
    }

    // ── Bulk clear ────────────────────────────────────────────────────────────

    /// Delete every key in every data CF and reset the node/edge counters to zero.
    ///
    /// Uses RocksDB range tombstones (`delete_range_cf`) so the cost is O(1)
    /// regardless of how many nodes or edges are stored.  The `meta` CF is left
    /// intact except for the node/edge counters and the edge-label-index sentinel
    /// (which is removed so a subsequent `Graph::open` will rebuild it — a no-op
    /// on an empty database).  Index definitions (`index_defs`) stored in meta
    /// are preserved; their backing data in `prop_idx` is wiped with the rest.
    pub fn clear_all(&self) -> Result<(), DbError> {
        // A key consisting of 17 × 0xFF is guaranteed to sort after every real
        // key (node/edge keys are 16-byte big-endian u128; string-prefixed index
        // keys are always shorter than 256 bytes in practice).
        const MAX_KEY: &[u8] = &[
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        ];

        const MIN_KEY: &[u8] = b"";

        let data_cfs = [
            CF_NODES, CF_EDGES, CF_ADJ_OUT, CF_ADJ_IN,
            CF_LABEL_IDX, CF_EDGE_LABEL_IDX, CF_PROP_IDX, CF_EDGE_PROP_IDX,
        ];

        let mut batch = WriteBatch::default();
        for cf_name in &data_cfs {
            let cf = self.db.cf_handle(cf_name).expect("CF missing");
            batch.delete_range_cf(&cf, MIN_KEY, MAX_KEY);
        }
        // Reset persisted counters.
        self.put_meta_batch(&mut batch, META_NODE_COUNT, &0u64.to_le_bytes());
        self.put_meta_batch(&mut batch, META_EDGE_COUNT,  &0u64.to_le_bytes());
        // Remove the edge-label-index sentinel so Graph::open re-stamps it.
        let meta_cf = self.db.cf_handle(CF_META).expect("meta CF missing");
        batch.delete_cf(&meta_cf, META_EDGE_LABEL_IDX_BUILT);
        self.write(batch)
    }

    // ── Atomic write ──────────────────────────────────────────────────────────

    /// Commit a `WriteBatch` atomically to RocksDB.
    ///
    /// All operations staged into `batch` (across any combination of CFs) are
    /// written as a single atomic unit — either all succeed or none are
    /// visible.  This is the mechanism that keeps node/edge data, adjacency
    /// entries, and index entries consistent with each other.
    pub fn write(&self, batch: WriteBatch) -> Result<(), DbError> {
        self.db.write(batch).map_err(|e| DbError::RocksDb(e.to_string()))
    }

    /// Create a fresh empty `WriteBatch`.
    ///
    /// A convenience constructor so callers do not need to import
    /// `rocksdb::WriteBatch` directly; the batch is then populated via the
    /// `*_batch` staging methods on this struct.
    pub fn batch() -> WriteBatch {
        WriteBatch::default()
    }

    /// Force a flush of all in-memory memtables to SST files on disk.
    ///
    /// Called during graceful shutdown (checkpoint) to ensure that all data
    /// acknowledged to clients is durably on disk before the process exits.
    /// Not needed for crash safety (the WAL covers that), but useful to
    /// minimize recovery time on the next open.
    pub fn flush(&self) -> Result<(), DbError> {
        self.db.flush().map_err(|e| DbError::RocksDb(e.to_string()))
    }

    // ── Bulk startup scan methods (ARCH-2) ────────────────────────────────────

    /// Return the raw serialized bytes of every node in the `nodes` CF.
    /// Used by `Graph::open()` for the single startup pass that populates
    /// the in-memory `nodes` and `label_index` maps.
    pub fn all_nodes_raw(&self) -> Result<Vec<Vec<u8>>, DbError> {
        let cf = self.db.cf_handle(CF_NODES).expect("nodes CF");
        let iter = self.db.iterator_cf(&cf, IteratorMode::Start);
        let mut out = Vec::new();
        for item in iter {
            let (_, val) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;
            out.push(val.to_vec());
        }
        Ok(out)
    }

    /// Return the raw serialized bytes of every edge in the `edges` CF.
    /// Used by `Graph::open()` for the single startup pass that populates
    /// the in-memory `edges` and `edge_label_index` maps.
    pub fn all_edges_raw(&self) -> Result<Vec<Vec<u8>>, DbError> {
        let cf = self.db.cf_handle(CF_EDGES).expect("edges CF");
        let iter = self.db.iterator_cf(&cf, IteratorMode::Start);
        let mut out = Vec::new();
        for item in iter {
            let (_, val) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;
            out.push(val.to_vec());
        }
        Ok(out)
    }

    // ── edge_prop_idx CF write-through methods ───────────────────────────────

    /// Build the key for an edge property index entry.
    /// Format: `label \0 property \0 encoded_value \0 edge_id(16 bytes BE)`
    fn edge_prop_idx_key(label: &str, prop: &str, encoded_val: &str, edge_id: u128) -> Vec<u8> {
        let mut k = Vec::new();
        k.extend_from_slice(label.as_bytes());
        k.push(0);
        k.extend_from_slice(prop.as_bytes());
        k.push(0);
        k.extend_from_slice(encoded_val.as_bytes());
        k.push(0);
        k.extend_from_slice(&edge_id.to_be_bytes());
        k
    }

    /// Stage an edge_prop_idx insertion into `batch`.
    pub fn put_edge_prop_entry(
        &self,
        batch: &mut WriteBatch,
        label: &str,
        prop: &str,
        encoded_val: &str,
        edge_id: u128,
    ) {
        let cf = self.db.cf_handle(CF_EDGE_PROP_IDX).expect("edge_prop_idx CF");
        batch.put_cf(&cf, Self::edge_prop_idx_key(label, prop, encoded_val, edge_id), b"");
    }

    /// Stage an edge_prop_idx deletion into `batch`.
    pub fn delete_edge_prop_entry(
        &self,
        batch: &mut WriteBatch,
        label: &str,
        prop: &str,
        encoded_val: &str,
        edge_id: u128,
    ) {
        let cf = self.db.cf_handle(CF_EDGE_PROP_IDX).expect("edge_prop_idx CF");
        batch.delete_cf(&cf, Self::edge_prop_idx_key(label, prop, encoded_val, edge_id));
    }

    /// Delete all edge_prop_idx entries for (label, prop) — used by DROP INDEX.
    pub fn delete_edge_prop_range(&self, label: &str, prop: &str) -> Result<(), DbError> {
        let cf = self.db.cf_handle(CF_EDGE_PROP_IDX).expect("edge_prop_idx CF");
        // Collect keys first to avoid iterator invalidation.
        let mut prefix = Vec::new();
        prefix.extend_from_slice(label.as_bytes());
        prefix.push(0);
        prefix.extend_from_slice(prop.as_bytes());
        prefix.push(0);

        let mode = IteratorMode::From(prefix.as_slice(), Direction::Forward);
        let iter = self.db.iterator_cf(&cf, mode);
        let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| DbError::RocksDb(e.to_string()))?;
            if !key.starts_with(prefix.as_slice()) { break; }
            keys_to_delete.push(key.to_vec());
        }
        let mut batch = WriteBatch::default();
        for k in keys_to_delete {
            batch.delete_cf(&cf, k);
        }
        self.db.write(batch).map_err(|e| DbError::RocksDb(e.to_string()))
    }
}

/// Decode a little-endian `u64` from the first 8 bytes of `b`.
///
/// Returns `0` if `b` is shorter than 8 bytes, which handles the case of a
/// freshly-created database where the counter key has never been written.
/// Counters are stored little-endian (as opposed to IDs which are big-endian)
/// because counters are not used as key prefixes and do not need to be
/// lexicographically sortable.
fn u64_from_le(b: &[u8]) -> u64 {
    if b.len() < 8 { return 0; }
    u64::from_le_bytes(b[..8].try_into().unwrap())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open() -> (RocksStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = RocksStore::open(dir.path()).unwrap();
        (store, dir)
    }

    // ── Open / reopen ────────────────────────────────────────────────────────

    #[test]
    fn open_creates_all_column_families() {
        let (_, _dir) = open();
        // If any CF were missing, open() would have panicked or returned Err.
    }

    #[test]
    fn reopen_existing_db_succeeds() {
        let dir = TempDir::new().unwrap();
        {
            let s = RocksStore::open(dir.path()).unwrap();
            s.put_node_raw(1, b"data").unwrap();
        }
        // Drop and reopen.
        let s2 = RocksStore::open(dir.path()).unwrap();
        assert_eq!(s2.get_node_raw(1).unwrap().as_deref(), Some(b"data".as_ref()));
    }

    // ── Key encoding ─────────────────────────────────────────────────────────

    #[test]
    fn id_key_round_trips() {
        let id: u128 = 0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210;
        assert_eq!(RocksStore::bytes_to_id(&RocksStore::id_key(id)), id);
    }

    #[test]
    fn id_keys_sort_correctly() {
        // Big-endian encoding: smaller IDs must produce lexicographically smaller keys.
        let a = RocksStore::id_key(1u128);
        let b = RocksStore::id_key(2u128);
        let big = RocksStore::id_key(u128::MAX);
        assert!(a < b);
        assert!(b < big);
    }

    // ── Nodes CF ─────────────────────────────────────────────────────────────

    #[test]
    fn node_put_get_roundtrip() {
        let (s, _d) = open();
        s.put_node_raw(42, b"node-payload").unwrap();
        assert_eq!(s.get_node_raw(42).unwrap().as_deref(), Some(b"node-payload".as_ref()));
    }

    #[test]
    fn missing_node_returns_none() {
        let (s, _d) = open();
        assert!(s.get_node_raw(999).unwrap().is_none());
    }

    #[test]
    fn node_overwrite() {
        let (s, _d) = open();
        s.put_node_raw(1, b"v1").unwrap();
        s.put_node_raw(1, b"v2").unwrap();
        assert_eq!(s.get_node_raw(1).unwrap().as_deref(), Some(b"v2".as_ref()));
    }

    #[test]
    fn node_batch_delete() {
        let (s, _d) = open();
        s.put_node_raw(10, b"x").unwrap();
        let mut batch = RocksStore::batch();
        s.delete_node_batch(&mut batch, 10);
        s.write(batch).unwrap();
        assert!(s.get_node_raw(10).unwrap().is_none());
    }

    #[test]
    fn all_node_ids_empty() {
        let (s, _d) = open();
        assert!(s.all_node_ids().unwrap().is_empty());
    }

    #[test]
    fn all_node_ids_returns_all() {
        let (s, _d) = open();
        for i in 0u128..5 {
            s.put_node_raw(i, b"x").unwrap();
        }
        let mut ids = s.all_node_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    // ── Edges CF ─────────────────────────────────────────────────────────────

    #[test]
    fn edge_put_get_roundtrip() {
        let (s, _d) = open();
        s.put_edge_raw(7, b"edge-data").unwrap();
        assert_eq!(s.get_edge_raw(7).unwrap().as_deref(), Some(b"edge-data".as_ref()));
    }

    #[test]
    fn edge_batch_delete() {
        let (s, _d) = open();
        s.put_edge_raw(5, b"e").unwrap();
        let mut batch = RocksStore::batch();
        s.delete_edge_batch(&mut batch, 5);
        s.write(batch).unwrap();
        assert!(s.get_edge_raw(5).unwrap().is_none());
    }

    // ── Adjacency CFs ─────────────────────────────────────────────────────────

    #[test]
    fn adj_out_put_and_scan() {
        let (s, _d) = open();
        let from = 100u128;
        let to = 200u128;
        let edge_id = 1u128;

        let mut batch = RocksStore::batch();
        s.put_adj_out(&mut batch, from, edge_id, to, "KNOWS");
        s.write(batch).unwrap();

        let results = s.scan_adj_out(from).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (edge_id, to, "KNOWS".to_string()));
    }

    #[test]
    fn adj_in_put_and_scan() {
        let (s, _d) = open();
        let from = 10u128;
        let to = 20u128;
        let edge_id = 99u128;

        let mut batch = RocksStore::batch();
        s.put_adj_in(&mut batch, to, edge_id, from, "LIKES");
        s.write(batch).unwrap();

        let results = s.scan_adj_in(to).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (edge_id, from, "LIKES".to_string()));
    }

    #[test]
    fn adj_scan_multiple_edges_same_source() {
        let (s, _d) = open();
        let from = 1u128;
        let mut batch = RocksStore::batch();
        s.put_adj_out(&mut batch, from, 10, 100, "A");
        s.put_adj_out(&mut batch, from, 11, 101, "B");
        s.put_adj_out(&mut batch, from, 12, 102, "C");
        s.write(batch).unwrap();

        let results = s.scan_adj_out(from).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn adj_scan_does_not_bleed_into_other_nodes() {
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_adj_out(&mut batch, 1, 10, 100, "X");
        s.put_adj_out(&mut batch, 2, 11, 101, "Y");
        s.write(batch).unwrap();

        assert_eq!(s.scan_adj_out(1).unwrap().len(), 1);
        assert_eq!(s.scan_adj_out(2).unwrap().len(), 1);
        assert_eq!(s.scan_adj_out(3).unwrap().len(), 0);
    }

    #[test]
    fn adj_delete() {
        let (s, _d) = open();
        let from = 5u128;
        let edge_id = 50u128;
        let mut batch = RocksStore::batch();
        s.put_adj_out(&mut batch, from, edge_id, 999, "EDGE");
        s.write(batch).unwrap();

        let mut batch2 = RocksStore::batch();
        s.delete_adj_out_batch(&mut batch2, from, edge_id);
        s.write(batch2).unwrap();

        assert!(s.scan_adj_out(from).unwrap().is_empty());
    }

    // ── Label index ───────────────────────────────────────────────────────────

    #[test]
    fn label_put_and_scan() {
        let (s, _d) = open();
        let node_a = 1u128;
        let node_b = 2u128;
        let mut batch = RocksStore::batch();
        s.put_label_entry(&mut batch, "Person", node_a);
        s.put_label_entry(&mut batch, "Person", node_b);
        s.put_label_entry(&mut batch, "Company", 3);
        s.write(batch).unwrap();

        let mut persons = s.scan_label("Person").unwrap();
        persons.sort();
        assert_eq!(persons, vec![node_a, node_b]);

        let companies = s.scan_label("Company").unwrap();
        assert_eq!(companies, vec![3]);
    }

    #[test]
    fn label_delete() {
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_label_entry(&mut batch, "Person", 1);
        s.put_label_entry(&mut batch, "Person", 2);
        s.write(batch).unwrap();

        let mut batch2 = RocksStore::batch();
        s.delete_label_entry(&mut batch2, "Person", 1);
        s.write(batch2).unwrap();

        assert_eq!(s.scan_label("Person").unwrap(), vec![2]);
    }

    #[test]
    fn label_scan_empty_label() {
        let (s, _d) = open();
        assert!(s.scan_label("Ghost").unwrap().is_empty());
    }

    #[test]
    fn label_prefix_isolation() {
        // "Per" must not match "Person" nodes.
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_label_entry(&mut batch, "Person", 1);
        s.write(batch).unwrap();
        assert!(s.scan_label("Per").unwrap().is_empty());
    }

    // ── Edge label index ──────────────────────────────────────────────────────

    #[test]
    fn edge_label_put_and_scan() {
        let (s, _d) = open();
        let e1 = 10u128;
        let e2 = 11u128;
        let mut batch = RocksStore::batch();
        s.put_edge_label_entry(&mut batch, "KNOWS", e1);
        s.put_edge_label_entry(&mut batch, "KNOWS", e2);
        s.put_edge_label_entry(&mut batch, "LIKES", 99);
        s.write(batch).unwrap();

        let mut knows = s.scan_edge_label("KNOWS").unwrap();
        knows.sort();
        assert_eq!(knows, vec![e1, e2]);
        assert_eq!(s.scan_edge_label("LIKES").unwrap(), vec![99]);
        assert!(s.scan_edge_label("HATES").unwrap().is_empty());
    }

    #[test]
    fn edge_label_delete() {
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_edge_label_entry(&mut batch, "KNOWS", 1);
        s.put_edge_label_entry(&mut batch, "KNOWS", 2);
        s.write(batch).unwrap();

        let mut batch2 = RocksStore::batch();
        s.delete_edge_label_entry(&mut batch2, "KNOWS", 1);
        s.write(batch2).unwrap();

        assert_eq!(s.scan_edge_label("KNOWS").unwrap(), vec![2]);
    }

    #[test]
    fn edge_label_prefix_isolation() {
        // "KNO" must not match "KNOWS".
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_edge_label_entry(&mut batch, "KNOWS", 1);
        s.write(batch).unwrap();
        assert!(s.scan_edge_label("KNO").unwrap().is_empty());
    }

    // ── Property index ────────────────────────────────────────────────────────

    #[test]
    fn prop_put_and_scan() {
        let (s, _d) = open();
        let n1 = 1u128;
        let n2 = 2u128;
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "Person", "age", "I:25", n1);
        s.put_prop_entry(&mut batch, "Person", "age", "I:25", n2);
        s.put_prop_entry(&mut batch, "Person", "age", "I:30", 3);
        s.write(batch).unwrap();

        let mut age25 = s.scan_prop("Person", "age", "I:25").unwrap();
        age25.sort();
        assert_eq!(age25, vec![n1, n2]);

        let age30 = s.scan_prop("Person", "age", "I:30").unwrap();
        assert_eq!(age30, vec![3]);
    }

    #[test]
    fn prop_delete() {
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "Person", "name", "S:Alice", 1);
        s.write(batch).unwrap();

        let mut batch2 = RocksStore::batch();
        s.delete_prop_entry(&mut batch2, "Person", "name", "S:Alice", 1);
        s.write(batch2).unwrap();

        assert!(s.scan_prop("Person", "name", "S:Alice").unwrap().is_empty());
    }

    #[test]
    fn prop_scan_isolation() {
        // Different property values must not bleed into each other.
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "Person", "age", "I:25", 1);
        s.write(batch).unwrap();
        assert!(s.scan_prop("Person", "age", "I:26").unwrap().is_empty());
    }

    // ── Meta + counters ───────────────────────────────────────────────────────

    #[test]
    fn meta_put_get() {
        let (s, _d) = open();
        s.put_meta(b"hello", b"world").unwrap();
        assert_eq!(s.get_meta(b"hello").unwrap().as_deref(), Some(b"world".as_ref()));
        assert!(s.get_meta(b"missing").unwrap().is_none());
    }

    #[test]
    fn counters_start_at_zero() {
        let (s, _d) = open();
        assert_eq!(s.node_count().unwrap(), 0);
        assert_eq!(s.edge_count().unwrap(), 0);
    }

    #[test]
    fn counters_set_and_read() {
        let (s, _d) = open();
        s.set_node_count(42).unwrap();
        s.set_edge_count(7).unwrap();
        assert_eq!(s.node_count().unwrap(), 42);
        assert_eq!(s.edge_count().unwrap(), 7);
    }

    #[test]
    fn counters_persist_across_reopen() {
        let dir = TempDir::new().unwrap();
        {
            let s = RocksStore::open(dir.path()).unwrap();
            s.set_node_count(100).unwrap();
            s.set_edge_count(50).unwrap();
        }
        let s2 = RocksStore::open(dir.path()).unwrap();
        assert_eq!(s2.node_count().unwrap(), 100);
        assert_eq!(s2.edge_count().unwrap(), 50);
    }

    // ── WriteBatch atomicity ──────────────────────────────────────────────────

    #[test]
    fn write_batch_is_atomic() {
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_node_batch(&mut batch, 1, b"n1");
        s.put_node_batch(&mut batch, 2, b"n2");
        s.put_edge_batch(&mut batch, 100, b"e1");
        s.write(batch).unwrap();

        assert!(s.get_node_raw(1).unwrap().is_some());
        assert!(s.get_node_raw(2).unwrap().is_some());
        assert!(s.get_edge_raw(100).unwrap().is_some());
    }

    #[test]
    fn write_batch_mixed_ops() {
        let (s, _d) = open();
        s.put_node_raw(1, b"old").unwrap();

        let mut batch = RocksStore::batch();
        s.put_node_batch(&mut batch, 1, b"new");   // overwrite
        s.put_node_batch(&mut batch, 2, b"fresh");
        s.delete_node_batch(&mut batch, 3);        // delete non-existent (no-op)
        s.write(batch).unwrap();

        assert_eq!(s.get_node_raw(1).unwrap().as_deref(), Some(b"new".as_ref()));
        assert_eq!(s.get_node_raw(2).unwrap().as_deref(), Some(b"fresh".as_ref()));
        assert!(s.get_node_raw(3).unwrap().is_none());
    }

    // ── Large-scale sanity ────────────────────────────────────────────────────

    #[test]
    fn ten_thousand_nodes_round_trip() {
        let (s, _d) = open();
        let n = 10_000u128;
        for i in 0..n {
            s.put_node_raw(i, format!("node-{i}").as_bytes()).unwrap();
        }
        assert_eq!(s.all_node_ids().unwrap().len(), n as usize);
        assert_eq!(
            s.get_node_raw(9_999).unwrap().as_deref(),
            Some(b"node-9999".as_ref())
        );
    }

    // ── Range scan ────────────────────────────────────────────────────────────

    #[test]
    fn prop_range_scan_all() {
        // No bounds → returns every entry for (label, prop).
        let (s, _d) = open();
        let enc20 = crate::types::value_index_key(&crate::types::Value::Int(20)).unwrap();
        let enc30 = crate::types::value_index_key(&crate::types::Value::Int(30)).unwrap();
        let enc40 = crate::types::value_index_key(&crate::types::Value::Int(40)).unwrap();
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "P", "age", &enc20, 20);
        s.put_prop_entry(&mut batch, "P", "age", &enc30, 30);
        s.put_prop_entry(&mut batch, "P", "age", &enc40, 40);
        s.write(batch).unwrap();
        let mut ids = s.scan_prop_range("P", "age", None, None).unwrap();
        ids.sort();
        assert_eq!(ids, vec![20, 30, 40]);
    }

    #[test]
    fn prop_range_scan_lower_exclusive() {
        // age > 25 → only 30 and 40
        let (s, _d) = open();
        let enc20 = crate::types::value_index_key(&crate::types::Value::Int(20)).unwrap();
        let enc30 = crate::types::value_index_key(&crate::types::Value::Int(30)).unwrap();
        let enc40 = crate::types::value_index_key(&crate::types::Value::Int(40)).unwrap();
        let lo_enc = crate::types::value_index_key(&crate::types::Value::Int(25)).unwrap();
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "P", "age", &enc20, 20);
        s.put_prop_entry(&mut batch, "P", "age", &enc30, 30);
        s.put_prop_entry(&mut batch, "P", "age", &enc40, 40);
        s.write(batch).unwrap();
        let mut ids = s.scan_prop_range("P", "age", Some((&lo_enc, false)), None).unwrap();
        ids.sort();
        assert_eq!(ids, vec![30, 40]);
    }

    #[test]
    fn prop_range_scan_upper_inclusive() {
        // age <= 30 → 20 and 30
        let (s, _d) = open();
        let enc20 = crate::types::value_index_key(&crate::types::Value::Int(20)).unwrap();
        let enc30 = crate::types::value_index_key(&crate::types::Value::Int(30)).unwrap();
        let enc40 = crate::types::value_index_key(&crate::types::Value::Int(40)).unwrap();
        let hi_enc = crate::types::value_index_key(&crate::types::Value::Int(30)).unwrap();
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "P", "age", &enc20, 20);
        s.put_prop_entry(&mut batch, "P", "age", &enc30, 30);
        s.put_prop_entry(&mut batch, "P", "age", &enc40, 40);
        s.write(batch).unwrap();
        let mut ids = s.scan_prop_range("P", "age", None, Some((&hi_enc, true))).unwrap();
        ids.sort();
        assert_eq!(ids, vec![20, 30]);
    }

    #[test]
    fn prop_range_scan_closed_interval() {
        // 20 <= age < 40 → 20 and 30
        let (s, _d) = open();
        let enc10 = crate::types::value_index_key(&crate::types::Value::Int(10)).unwrap();
        let enc20 = crate::types::value_index_key(&crate::types::Value::Int(20)).unwrap();
        let enc30 = crate::types::value_index_key(&crate::types::Value::Int(30)).unwrap();
        let enc40 = crate::types::value_index_key(&crate::types::Value::Int(40)).unwrap();
        let lo_enc = crate::types::value_index_key(&crate::types::Value::Int(20)).unwrap();
        let hi_enc = crate::types::value_index_key(&crate::types::Value::Int(40)).unwrap();
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "P", "age", &enc10, 10);
        s.put_prop_entry(&mut batch, "P", "age", &enc20, 20);
        s.put_prop_entry(&mut batch, "P", "age", &enc30, 30);
        s.put_prop_entry(&mut batch, "P", "age", &enc40, 40);
        s.write(batch).unwrap();
        let mut ids = s.scan_prop_range("P", "age", Some((&lo_enc, true)), Some((&hi_enc, false))).unwrap();
        ids.sort();
        assert_eq!(ids, vec![20, 30]);
    }

    #[test]
    fn prop_range_scan_negative_integers() {
        // Verify negatives sort below positives.
        let (s, _d) = open();
        let enc_n5 = crate::types::value_index_key(&crate::types::Value::Int(-5)).unwrap();
        let enc_0  = crate::types::value_index_key(&crate::types::Value::Int(0)).unwrap();
        let enc_5  = crate::types::value_index_key(&crate::types::Value::Int(5)).unwrap();
        let lo_enc = crate::types::value_index_key(&crate::types::Value::Int(-3)).unwrap();
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "P", "x", &enc_n5, 1);
        s.put_prop_entry(&mut batch, "P", "x", &enc_0,  2);
        s.put_prop_entry(&mut batch, "P", "x", &enc_5,  3);
        s.write(batch).unwrap();
        // x > -3 → 0 and 5
        let mut ids = s.scan_prop_range("P", "x", Some((&lo_enc, false)), None).unwrap();
        ids.sort();
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn prop_prefix_scan_basic() {
        // Insert names: Alice, Albert, Bob. Prefix "Al" should return Alice and Albert.
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "P", "name", "S:Alice",  1);
        s.put_prop_entry(&mut batch, "P", "name", "S:Albert", 2);
        s.put_prop_entry(&mut batch, "P", "name", "S:Bob",    3);
        s.write(batch).unwrap();

        let mut ids = s.scan_prop_prefix("P", "name", "Al").unwrap();
        ids.sort();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn prop_prefix_scan_empty_prefix_returns_all_strings() {
        // Empty prefix "" matches every string value.
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "P", "name", "S:Alice", 1);
        s.put_prop_entry(&mut batch, "P", "name", "S:Bob",   2);
        s.write(batch).unwrap();

        let mut ids = s.scan_prop_prefix("P", "name", "").unwrap();
        ids.sort();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn prop_prefix_scan_no_match_returns_empty() {
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "P", "name", "S:Alice", 1);
        s.write(batch).unwrap();

        let ids = s.scan_prop_prefix("P", "name", "Z").unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn prop_prefix_scan_does_not_bleed_across_properties() {
        // Entries for a different property must not appear.
        let (s, _d) = open();
        let mut batch = RocksStore::batch();
        s.put_prop_entry(&mut batch, "P", "name",  "S:Alice", 1);
        s.put_prop_entry(&mut batch, "P", "alias", "S:Al",    2); // different prop
        s.write(batch).unwrap();

        let ids = s.scan_prop_prefix("P", "name", "Al").unwrap();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn clear_all_wipes_nodes_and_resets_counts() {
        let (s, _d) = open();
        // Insert a node and an edge.
        let mut batch = RocksStore::batch();
        s.put_node_raw(1, b"node-1").unwrap();
        s.put_node_raw(2, b"node-2").unwrap();
        s.put_label_entry(&mut batch, "Person", 1);
        s.put_label_entry(&mut batch, "Person", 2);
        s.write(batch).unwrap();
        assert!(s.get_node_raw(1).unwrap().is_some());

        s.clear_all().unwrap();

        // Nodes should be gone.
        assert!(s.get_node_raw(1).unwrap().is_none());
        assert!(s.get_node_raw(2).unwrap().is_none());
        // Counts should be zero.
        assert_eq!(s.node_count().unwrap(), 0);
        assert_eq!(s.edge_count().unwrap(), 0);
        // Label index should be empty.
        assert!(s.scan_label("Person").unwrap().is_empty());
    }
}
