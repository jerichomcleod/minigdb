//! ULID generation and the [`NodeId`] / [`EdgeId`] newtype wrappers.
//!
//! ULIDs are 128-bit universally-unique, lexicographically-sortable identifiers.
//! The upper 48 bits encode a millisecond timestamp; the lower 80 bits provide
//! randomness (counter + nanosecond entropy) to guarantee uniqueness within the
//! same millisecond.  They serialise to a 26-character Crockford Base32 string.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ── ULID generation ─────────────────────────────────────────────────────────

/// Crockford Base32 alphabet (no I, L, O, U to avoid ambiguity).
const ENCODING: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Generate a new ULID as a `u128`.
///
/// Layout: upper 48 bits = milliseconds since Unix epoch,
///         lower 80 bits = pseudo-random (counter + nanosecond entropy).
///
/// Uniqueness guarantee: a global atomic counter ensures each call gets a
/// distinct value even if two calls land in the same millisecond.
pub fn ulid_new() -> u128 {
    static SEQ: AtomicU64 = AtomicU64::new(0);

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();

    // 48-bit millisecond timestamp.
    let ts_ms = (now.as_millis() as u128) & 0xFFFF_FFFF_FFFFu128;

    // Monotonic counter for within-millisecond uniqueness.
    let seq = SEQ.fetch_add(1, Ordering::Relaxed) as u128;

    // Sub-millisecond nanos add entropy so IDs from different processes
    // (same counter value) are still distinct.
    let nanos = now.subsec_nanos() as u128;

    // Mix counter + nanos into 80 bits with a couple of multiply-xor rounds.
    let rand_80 = (seq
        .wrapping_mul(6_364_136_223_846_793_005u128)
        .wrapping_add(nanos.wrapping_mul(0x9e37_79b9_7f4a_7c15u128)))
        & ((1u128 << 80) - 1);

    (ts_ms << 80) | rand_80
}

/// Encode a `u128` as a 26-character Crockford Base32 ULID string.
pub fn ulid_encode(v: u128) -> String {
    let mut buf = [0u8; 26];
    let mut n = v;
    for i in (0..26).rev() {
        buf[i] = ENCODING[(n & 0x1F) as usize];
        n >>= 5;
    }
    // SAFETY: ENCODING contains only ASCII bytes.
    unsafe { String::from_utf8_unchecked(buf.to_vec()) }
}

/// Decode a 26-character Crockford Base32 ULID string back to a `u128`.
///
/// Accepts upper- and lower-case letters.  Returns `Err` if the string has
/// the wrong length or contains a character not in the Crockford alphabet.
pub fn ulid_decode(s: &str) -> Result<u128, String> {
    if s.len() != 26 {
        return Err(format!("ULID must be 26 characters, got {}", s.len()));
    }
    let mut v: u128 = 0;
    for c in s.chars() {
        let digit: u128 = match c.to_ascii_uppercase() {
            '0' => 0,  '1' => 1,  '2' => 2,  '3' => 3,  '4' => 4,
            '5' => 5,  '6' => 6,  '7' => 7,  '8' => 8,  '9' => 9,
            'A' => 10, 'B' => 11, 'C' => 12, 'D' => 13, 'E' => 14,
            'F' => 15, 'G' => 16, 'H' => 17, 'J' => 18, 'K' => 19,
            'M' => 20, 'N' => 21, 'P' => 22, 'Q' => 23, 'R' => 24,
            'S' => 25, 'T' => 26, 'V' => 27, 'W' => 28, 'X' => 29,
            'Y' => 30, 'Z' => 31,
            other => return Err(format!("invalid ULID character: '{}'", other)),
        };
        v = (v << 5) | digit;
    }
    Ok(v)
}

// ── ID newtypes ──────────────────────────────────────────────────────────────

/// Stable public identifier for a node (ULID encoded as u128).
/// `Copy` because u128 is a value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub u128);

/// Stable public identifier for an edge (ULID encoded as u128).
/// `Copy` because u128 is a value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgeId(pub u128);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ulid_encode(self.0))
    }
}

impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ulid_encode(self.0))
    }
}
