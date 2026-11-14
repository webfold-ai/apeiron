//! The cold KV store: sealed block rows held outside the dense `KvState`, in the canonical
//! flat/spill [`Table`](crate::map::table::Table).
//!
//! A dense [`KvState`](crate::engine::KvState) holds every position of every layer as a Q32.32 plane:
//! `2 · n_head_kv · head_dim · 8` bytes per token per layer. This store takes a block's frozen key
//! rows and value rows out of that plane, one record per `(layer, kv-head, block)`, and returns them
//! bit-for-bit on [`ColdStore::gather`]. A block the store does not hold is a refusal, never a read
//! of the plane the rows came from.
//!
//! # Bytes, stated separately
//!
//! * **Stored.** `2 · head_dim · 8` payload bytes per stored position, so `O(tokens)` and growing.
//!   At `n_head_kv = 2`, `head_dim = 64` that is 1 024 B/token per layer-plane, 49 152 B/token over
//!   24 layers. **This is not flat, and it is not 9 B/token.** The measured `AC2-MEM` 9 B/token
//!   object is a token-id + hash-chain store (`FinderRecall.ColdStore`): it recalls tokens by exact
//!   recompute and holds no KV vectors, so it cannot serve a row gather. Building that object, or
//!   quantising these rows below 8 B/word, is a different task.
//! * **Resident.** The store's own record per block: packed key, row count and value header,
//!   [`ColdStore::index_bytes`]. `O(blocks)`, i.e. `O(tokens / block)`. It is reported apart from the
//!   payload and is never summed into a "flat" claim.
//!
//! # Lineage
//!
//! `~/fold/bedrock/src/serve/model/cold.rs` `ColdDecode` (full KV off the hot
//! path, gathered by block) and `tier.rs` `ColdTier` (one contiguous record per
//! `(layer, kv-head, block)`, sealed once). Differences: the record lives in this crate's canonical
//! growable [`Table`](crate::map::table::Table) rather than a bespoke int8 array plus a file; the
//! rows stay Q32.32 so a gather is exact without a dequantisation step; there is no file tier, so
//! "cold" here means "outside the dense plane", not yet "on disk".

use crate::cell::Sigma;
use crate::map::table::{Merge, StorageError, Table, WordHome};

/// One gathered block's key and value rows, borrowed from the store.
type KvPair<'a> = (&'a [i64], &'a [i64]);

/// Bits per packed key field: `layer`, `kv_head` and `block` each below `2^21`.
const KEY_FIELD_BITS: u32 = 21;

/// The largest value a packed key field admits.
const KEY_FIELD_MASK: usize = (1 << KEY_FIELD_BITS) - 1;

/// Resident bytes of one stored block's index record: the packed `i64` key, the `usize` row count
/// and the three-word `Vec` header of the payload. The payload itself is not resident here; it lives
/// in the table's flat area or spill region and is counted by [`ColdStore::payload_bytes`].
const INDEX_RECORD_BYTES: usize =
    core::mem::size_of::<i64>() + core::mem::size_of::<usize>() + 3 * core::mem::size_of::<usize>();

/// A refused cold-store write or read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdStoreError {
    /// No block is stored under that `(layer, kv-head, block)`. Reported, never substituted.
    Missing,
    /// A zero geometry, a row width that disagrees with `head_dim`, a short or oversized block, or
    /// a packed key field at or above `2^21`.
    BadGeometry,
    /// The canonical table refused a checked sizing or reservation step.
    Store(StorageError),
}

/// The replacement rule of [`ColdStore`]: a re-put block replaces its rows.
struct ReplaceRows;

impl Merge<Vec<i64>> for ReplaceRows {
    fn merge(&self, new: Vec<i64>, _old: Vec<i64>) -> Vec<i64> {
        new
    }
}

/// Sealed KV block rows outside the dense plane, in the canonical flat/spill
/// [`Table`](crate::map::table::Table).
///
/// One record per `(layer, kv-head, block)` holds the block's key rows followed by its value rows,
/// `2 · rows · head_dim` Q32.32 words. The key packs the three fields into one `i64`; the table's
/// `WordHome` places it and its spill regions hold the payload past the resident extent.
pub struct ColdStore {
    table: Table<i64, Vec<i64>, WordHome>,
    block: usize,
    head_dim: usize,
    positions: usize,
}

impl ColdStore {
    /// A store of `block`-position records of `head_dim`-wide Q32.32 rows.
    ///
    /// # Errors
    ///
    /// [`ColdStoreError::BadGeometry`] when `block` or `head_dim` is zero.
    ///
    /// # Complexity
    ///
    /// `O(1)` time and space.
    pub fn new(block: usize, head_dim: usize) -> Result<ColdStore, ColdStoreError> {
        if block == 0 || head_dim == 0 {
            return Err(ColdStoreError::BadGeometry);
        }
        Ok(ColdStore {
            table: Table::new(Vec::new(), WordHome),
            block,
            head_dim,
            positions: 0,
        })
    }

    /// The packed table key of one block.
    ///
    /// # Errors
    ///
    /// [`ColdStoreError::BadGeometry`] when a field does not fit its 21 bits.
    ///
    /// # Complexity
    ///
    /// `O(1)`.
    fn key(layer: usize, kv_head: usize, block: usize) -> Result<i64, ColdStoreError> {
        if layer > KEY_FIELD_MASK || kv_head > KEY_FIELD_MASK || block > KEY_FIELD_MASK {
            return Err(ColdStoreError::BadGeometry);
        }
        // Pack in the key's width, not the host pointer width (WASM pointers are 32-bit).
        let word = ((layer as u64) << (2 * KEY_FIELD_BITS))
            | ((kv_head as u64) << KEY_FIELD_BITS)
            | block as u64;
        match i64::try_from(word) {
            Ok(k) => Ok(k),
            Err(_) => Err(ColdStoreError::BadGeometry),
        }
    }

    /// Store one block's key and value rows, replacing any earlier copy.
    ///
    /// `k` and `v` are the block's rows in position order, each `head_dim` wide; a final partial
    /// block may be stored (`rows ≤ block`).
    ///
    /// # Errors
    ///
    /// [`ColdStoreError::BadGeometry`] when `k` and `v` differ in length, are not a whole number of
    /// `head_dim` rows, are empty, or exceed the block width; [`ColdStoreError::Store`] when the
    /// table's checked sizing or reservation refuses.
    ///
    /// # Complexity
    ///
    /// `O(rows · head_dim)` time and space to copy the record, plus the table's `O(1)` amortized
    /// insertion; `2 · rows · head_dim` words copied.
    pub fn put(
        &mut self,
        layer: usize,
        kv_head: usize,
        block: usize,
        k: &[i64],
        v: &[i64],
    ) -> Result<(), ColdStoreError> {
        if k.len() != v.len() || !k.len().is_multiple_of(self.head_dim) {
            return Err(ColdStoreError::BadGeometry);
        }
        let rows = k.len() / self.head_dim;
        if rows == 0 || rows > self.block {
            return Err(ColdStoreError::BadGeometry);
        }
        let key = Self::key(layer, kv_head, block)?;
        let fresh = self.table.get_exact_at(Sigma::Parabolic, &key).is_none();
        let mut record = Vec::with_capacity(2 * k.len());
        record.extend_from_slice(k);
        record.extend_from_slice(v);
        self.table
            .insert_with_at(&ReplaceRows, Sigma::Parabolic, key, record)
            .map_err(ColdStoreError::Store)?;
        if fresh {
            self.positions += rows;
        }
        Ok(())
    }

    /// The `(key rows, value rows)` of one stored block, or [`ColdStoreError::Missing`].
    ///
    /// There is no dense-plane fallback: an unsealed block is an error, not a read of the plane the
    /// rows were moved out of.
    ///
    /// # Errors
    ///
    /// [`ColdStoreError::Missing`] for an unstored block; [`ColdStoreError::BadGeometry`] for a key
    /// outside the packed range.
    ///
    /// # Complexity
    ///
    /// `O(1)` probe plus one borrowed split; no copy.
    pub fn gather(
        &self,
        layer: usize,
        kv_head: usize,
        block: usize,
    ) -> Result<(&[i64], &[i64]), ColdStoreError> {
        let key = Self::key(layer, kv_head, block)?;
        let record = self
            .table
            .get_exact_at(Sigma::Parabolic, &key)
            .ok_or(ColdStoreError::Missing)?;
        let half = record.len() / 2;
        Ok((&record[..half], &record[half..]))
    }

    /// The rows of exactly `blocks`, in that order: one record per named block, no substitution and
    /// no silent skip. A block the store does not hold refuses the whole read.
    ///
    /// # Errors
    ///
    /// The first refusal of [`ColdStore::gather`], identically.
    ///
    /// # Complexity
    ///
    /// `O(blocks.len())` probes and `O(blocks.len())` space for the returned borrow list.
    pub fn gather_many(
        &self,
        layer: usize,
        kv_head: usize,
        blocks: &[usize],
    ) -> Result<Vec<KvPair<'_>>, ColdStoreError> {
        let mut out = Vec::with_capacity(blocks.len());
        let mut err: Option<ColdStoreError> = None;
        let mut i = 0usize;
        while i < blocks.len() && err.is_none() {
            match self.gather(layer, kv_head, blocks[i]) {
                Ok(pair) => out.push(pair),
                Err(e) => err = Some(e),
            }
            i += 1;
        }
        match err {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }

    /// Stored block records.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.table.len()
    }

    /// Stored positions, summed over the distinct blocks put.
    #[must_use]
    pub fn positions(&self) -> usize {
        self.positions
    }

    /// Payload bytes of the stored rows, `positions · 2 · head_dim · 8`: `O(tokens)`. The table's
    /// slot and key bytes are not included; this is the KV payload the store owns.
    #[must_use]
    pub fn payload_bytes(&self) -> usize {
        self.positions * self.bytes_per_position()
    }

    /// Payload bytes one stored position costs on one `(layer, kv-head)` plane: `2 · head_dim · 8`
    /// for K and V together. This is the per-token figure the store measures; it is not flat and it
    /// is not the 9 B/token token-store figure.
    #[must_use]
    pub fn bytes_per_position(&self) -> usize {
        2 * self.head_dim * core::mem::size_of::<i64>()
    }

    /// Resident index bytes, `blocks · INDEX_RECORD_BYTES`: the packed key, row count and value
    /// header per block. `O(blocks)`; reported apart from [`ColdStore::payload_bytes`].
    #[must_use]
    pub fn index_bytes(&self) -> usize {
        self.blocks() * INDEX_RECORD_BYTES
    }
}

#[cfg(test)]
mod wasm_key_tests {
    use super::*;

    #[test]
    fn packed_key_retains_all_three_fields_on_32_bit_targets() {
        assert_eq!(ColdStore::key(1, 2, 3), Ok((1_i64 << 42) | (2_i64 << 21) | 3));
        assert_eq!(ColdStore::key(KEY_FIELD_MASK, KEY_FIELD_MASK, KEY_FIELD_MASK), Ok(i64::MAX));
    }

    #[test]
    fn oversized_key_fields_are_refused() {
        for key in [(KEY_FIELD_MASK + 1, 0, 0), (0, KEY_FIELD_MASK + 1, 0), (0, 0, KEY_FIELD_MASK + 1)] {
            assert_eq!(ColdStore::key(key.0, key.1, key.2), Err(ColdStoreError::BadGeometry));
        }
    }
}
