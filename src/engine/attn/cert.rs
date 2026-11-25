//! Exact-on-certificate block summaries for the attention read.
//!
//! The full scan in [`KvState::attn_head`](super::KvState::attn_head) reads every key and value of
//! the causal prefix. A block of positions can be skipped without changing one output word when
//! every key in it is far enough below the softmax maximum that its `fixed_exp` word is exactly
//! zero. This module supplies the summary and the sound bound that decide that.
//!
//! # The bound
//!
//! For a block `B` of keys, store the rounded component mean `μ` (truncating integer division of
//! the component sums by the block width) and
//!
//! ```text
//! radius = max_{j ∈ B} Σ_i |k_{j,i} − μ_i|.
//! ```
//!
//! For any query `q`, `Σ_i q_i (k_{j,i} − μ_i) ≤ max_i |q_i| · radius` by Hölder's inequality
//! (∞-norm against 1-norm). The engine's score is two round-to-nearest shifts, so
//!
//! ```text
//! |score_j − round((q·μ / 2^32) · scale / 2^32)| ≤ (floor(qmax·radius / 2^32) + 1) · scale / 2^32 + 1
//! ```
//!
//! and [`bounds`](KvBlockIndex::bounds) adds two units of slack. Every step is checked `i128`
//! arithmetic; a summary that cannot be built or a bound that leaves `i128` is reported as `None`,
//! and a block without a bound is always scanned. The bound is therefore an upper bound on a real
//! score for every query, whether or not the block is later skipped.
//!
//! # Cost
//!
//! One `head_dim` dot product per block per query, against `block` full score dots: a `1/block`
//! overhead on the read. A sealed block stores `head_dim` `i64` mean words and one `i128` radius.

use std::sync::atomic::{AtomicU64, Ordering};

/// What one certified read skipped.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CertStats {
    /// Blocks the causal prefix spans.
    pub blocks: u64,
    /// Blocks whose positions were scanned.
    pub scanned: u64,
    /// Blocks certified out of every pass.
    pub skipped: u64,
    /// Positions in the skipped blocks: the keys and values that were not read.
    pub skipped_positions: u64,
}

/// The running totals of every certified read through one index.
#[derive(Debug, Default)]
struct CertTotals {
    blocks: AtomicU64,
    scanned: AtomicU64,
    skipped: AtomicU64,
    skipped_positions: AtomicU64,
}

impl Clone for CertTotals {
    fn clone(&self) -> Self {
        Self {
            blocks: AtomicU64::new(self.blocks.load(Ordering::Relaxed)),
            scanned: AtomicU64::new(self.scanned.load(Ordering::Relaxed)),
            skipped: AtomicU64::new(self.skipped.load(Ordering::Relaxed)),
            skipped_positions: AtomicU64::new(self.skipped_positions.load(Ordering::Relaxed)),
        }
    }
}

impl CertTotals {
    fn add(&self, stats: &CertStats) {
        self.blocks.fetch_add(stats.blocks, Ordering::Relaxed);
        self.scanned.fetch_add(stats.scanned, Ordering::Relaxed);
        self.skipped.fetch_add(stats.skipped, Ordering::Relaxed);
        self.skipped_positions
            .fetch_add(stats.skipped_positions, Ordering::Relaxed);
    }

    fn get(&self) -> CertStats {
        CertStats {
            blocks: self.blocks.load(Ordering::Relaxed),
            scanned: self.scanned.load(Ordering::Relaxed),
            skipped: self.skipped.load(Ordering::Relaxed),
            skipped_positions: self.skipped_positions.load(Ordering::Relaxed),
        }
    }
}

/// One block's sound summary: the component mean of its keys and the largest 1-norm deviation.
#[derive(Debug, Clone)]
struct BlockSummary {
    /// Truncating component mean of the block's keys, one word per head component.
    mu: Vec<i64>,
    /// `max_j Σ_i |k_{j,i} − μ_i|`, non-negative.
    radius: i128,
}

/// The per-`(layer, kv_head, block)` summaries of one [`KvState`](super::KvState).
#[derive(Debug, Clone)]
pub(crate) struct KvBlockIndex {
    block: usize,
    n_layer: usize,
    n_head_kv: usize,
    head_dim: usize,
    /// `summaries[layer][kv_head][block]`, pushed in block order.
    summaries: Vec<Vec<Vec<BlockSummary>>>,
    /// Accumulated over every certified read; relaxed loads and adds (the counters order no data).
    totals: CertTotals,
}

impl KvBlockIndex {
    /// An empty index for the cache geometry, with `block` positions per block.
    ///
    /// # Complexity
    ///
    /// `O(n_layer · n_head_kv)` empty vectors; no summary is allocated until a block seals.
    pub(crate) fn new(n_layer: usize, n_head_kv: usize, head_dim: usize, block: usize) -> Self {
        Self {
            block,
            n_layer,
            n_head_kv,
            head_dim,
            summaries: vec![vec![Vec::new(); n_head_kv]; n_layer],
            totals: CertTotals::default(),
        }
    }

    /// Add one read's counters to the running totals.
    pub(crate) fn add_read(&self, stats: &CertStats) {
        self.totals.add(stats);
    }

    /// The running totals of every certified read through this index.
    pub(crate) fn totals(&self) -> CertStats {
        self.totals.get()
    }

    /// Zero the running totals.
    pub(crate) fn reset_totals(&mut self) {
        self.totals = CertTotals::default();
    }

    /// The block width in positions.
    pub(crate) fn block(&self) -> usize {
        self.block
    }

    /// Rebuild `layer`'s summaries from the key plane for every completed block of `len`
    /// positions. Blocks that are not complete are not summarised.
    ///
    /// # Complexity
    ///
    /// `O(n_head_kv · (len / block) · block · head_dim)` additions and comparisons: two passes
    /// over each completed block for each KV head.
    pub(crate) fn rebuild_layer(&mut self, layer: usize, kplane: &[i64], len: usize) {
        let full = len / self.block;
        let mut kvh = 0usize;
        while kvh < self.n_head_kv {
            let mut summaries: Vec<BlockSummary> = Vec::with_capacity(full);
            let mut b = 0usize;
            while b < full {
                match summarise(kplane, self.n_head_kv, kvh, self.head_dim, self.block, b) {
                    Some(s) => summaries.push(s),
                    None => break,
                }
                b += 1;
            }
            self.summaries[layer][kvh] = summaries;
            kvh += 1;
        }
    }

    /// Seal block `b` of `layer` from the key plane, for every KV head.
    ///
    /// A block that cannot be summarised (an arithmetic refusal) is left without a summary, and
    /// [`KvBlockIndex::bounds`] returns `None` for it; the read then scans it.
    ///
    /// # Complexity
    ///
    /// `O(n_head_kv · block · head_dim)` additions and comparisons.
    pub(crate) fn seal(&mut self, layer: usize, kplane: &[i64], b: usize) {
        let mut kvh = 0usize;
        while kvh < self.n_head_kv {
            let slot = &mut self.summaries[layer][kvh];
            match summarise(kplane, self.n_head_kv, kvh, self.head_dim, self.block, b) {
                Some(s) => {
                    // Blocks seal in order; a re-seal after truncation may replace the slot.
                    if b < slot.len() {
                        slot[b] = s;
                    } else {
                        slot.push(s);
                    }
                }
                None => {
                    if b < slot.len() {
                        slot.truncate(b);
                    }
                }
            }
            kvh += 1;
        }
    }

    /// Drop every summary whose block is not wholly inside the first `m` positions.
    ///
    /// # Complexity
    ///
    /// `O(n_layer · n_head_kv)` truncations of vectors.
    pub(crate) fn truncate(&mut self, m: usize) {
        let keep = m / self.block;
        let mut layer = 0usize;
        while layer < self.n_layer {
            let mut kvh = 0usize;
            while kvh < self.n_head_kv {
                self.summaries[layer][kvh].truncate(keep);
                kvh += 1;
            }
            layer += 1;
        }
    }

    /// The sound `(lower, upper)` bound on every score of block `b`, or `None` when the block is
    /// not complete (a trailing partial block) or has no summary.
    ///
    /// `len` is the layer's appended position count; a block is complete when its last position is
    /// inside `0..len`.
    ///
    /// # Errors
    ///
    /// `None` on a missing summary or any arithmetic refusal. The caller scans the block in that
    /// case, so the read stays exact.
    ///
    /// # Complexity
    ///
    /// `O(head_dim)` checked `i128` multiply-adds plus one `i128` multiply for the margin.
    pub(crate) fn bounds(
        &self,
        layer: usize,
        kvh: usize,
        b: usize,
        len: usize,
        q: &[i64],
        scale: i64,
    ) -> Option<(i64, i64)> {
        let end = b.checked_add(1)?.checked_mul(self.block)?;
        if end > len {
            return None;
        }
        let summary = self.summaries.get(layer)?.get(kvh)?.get(b)?;
        if summary.mu.len() != q.len() || summary.mu.len() != self.head_dim {
            return None;
        }
        block_bounds(&summary.mu, summary.radius, q, scale)
    }
}

/// The mean/radius summary of block `b` for one KV head, or `None` when the block is outside the
/// plane or an accumulation leaves `i128`.
///
/// # Complexity
///
/// `O(block · head_dim)` additions and comparisons: one sum pass, then one deviation pass.
fn summarise(
    kplane: &[i64],
    n_head_kv: usize,
    kvh: usize,
    head_dim: usize,
    block: usize,
    block_index: usize,
) -> Option<BlockSummary> {
    if block == 0 || head_dim == 0 || kvh >= n_head_kv {
        return None;
    }
    let width = n_head_kv.checked_mul(head_dim)?;
    let first = kvh.checked_mul(head_dim)?;
    let start = block_index.checked_mul(block)?;
    let end = start.checked_add(block)?;
    if end.checked_mul(width)? > kplane.len() {
        return None;
    }
    let mut sum: Vec<i128> = vec![0i128; head_dim];
    let mut pos = start;
    while pos < end {
        let base = pos.checked_mul(width)?.checked_add(first)?;
        let mut col = 0usize;
        while col < head_dim {
            let word = i128::from(*kplane.get(base.checked_add(col)?)?);
            sum[col] = sum[col].checked_add(word)?;
            col += 1;
        }
        pos += 1;
    }
    let width_i = i128::try_from(block).ok()?;
    let mut mu: Vec<i64> = Vec::with_capacity(head_dim);
    let mut col = 0usize;
    while col < head_dim {
        mu.push(clamp_i64(sum[col] / width_i));
        col += 1;
    }
    let mut radius: i128 = 0;
    pos = start;
    while pos < end {
        let base = pos.checked_mul(width)?.checked_add(first)?;
        let mut dev: i128 = 0;
        let mut col = 0usize;
        while col < head_dim {
            let delta = i128::from(*kplane.get(base.checked_add(col)?)?) - i128::from(mu[col]);
            dev = dev.checked_add(delta.abs())?;
            col += 1;
        }
        if dev > radius {
            radius = dev;
        }
        pos += 1;
    }
    Some(BlockSummary { mu, radius })
}

/// The sound score interval of one block: the centred score plus a margin that covers the block's
/// radius and both round-to-nearest epilogues.
///
/// For every key `k_j` of the block, with `q` any query of the plane width,
///
/// ```text
/// center = round_shift_32(round_shift_32(Σ_i q_i μ_i) · scale)
/// |score_j − center| ≤ (floor(qmax · radius / 2^32) + 1) · scale / 2^32 + 1 ≤ margin
/// ```
///
/// where `score_j` is [`score_at`](super::score_at)'s value. The function adds two units of slack.
///
/// # Complexity
///
/// `O(head_dim)` checked `i128` multiply-adds, `O(1)` otherwise.
fn block_bounds(mu: &[i64], radius: i128, q: &[i64], scale: i64) -> Option<(i64, i64)> {
    if radius < 0 || scale < 0 || mu.len() != q.len() {
        return None;
    }
    let mut sum: i128 = 0;
    let mut qmax: i128 = 0;
    let mut i = 0usize;
    while i < q.len() {
        let qi = i128::from(q[i]);
        sum = sum.checked_add(qi.checked_mul(i128::from(mu[i]))?)?;
        let a = qi.abs();
        if a > qmax {
            qmax = a;
        }
        i += 1;
    }
    let c = round_shift_32_checked(sum)?;
    let s0 = round_shift_32_checked(c.checked_mul(i128::from(scale))?)?;
    let spread = qmax.checked_mul(radius)?;
    // `floor(spread / 2^32) + 1` is an integer upper bound on `|raw_j − c|`; a positive value, so
    // the shift is the floor.
    let bound = (spread >> 32).checked_add(1)?;
    let scaled = bound.checked_mul(i128::from(scale))?;
    let margin = (scaled >> 32).checked_add(2)?;
    Some((
        clamp_i64(s0.checked_sub(margin)?),
        clamp_i64(s0.checked_add(margin)?),
    ))
}

/// [`round_shift_32`] with the `i128` addition checked; `None` when it would leave the range.
fn round_shift_32_checked(v: i128) -> Option<i128> {
    v.checked_add(1_i128 << 31).map(|x| x >> 32)
}

/// Narrow an `i128` bound into `i64`, saturating: a bound above `i64::MAX` is still above every
/// representable score, so saturation keeps the interval sound.
#[allow(
    clippy::cast_possible_truncation,
    reason = "both casts are guarded by the comparisons immediately above them"
)]
fn clamp_i64(v: i128) -> i64 {
    if v > i128::from(i64::MAX) {
        i64::MAX
    } else if v < i128::from(i64::MIN) {
        i64::MIN
    } else {
        v as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::attn::KvCarrier;
    use crate::engine::attn::{score_at, KvState, KvStorage};
    use crate::engine::ONE;

    /// A deterministic Q32.32 word in `[-4·ONE, 4·ONE)` from a 64-bit LCG state.
    fn word(state: &mut u64) -> i64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let wide = i64::try_from(*state >> 1).expect("the top 63 bits fit i64");
        wide % (8 * ONE) - 4 * ONE
    }

    #[test]
    fn every_score_of_a_block_is_inside_its_bound() {
        // Two KV heads of width 5 over three 8-position blocks, so the radius is nontrivial and
        // the block is narrower than the plane stride.
        let (n_head_kv, hd, block, keys) = (2usize, 5usize, 8usize, 24usize);
        let mut kv =
            KvState::with_carrier(1, 4, n_head_kv, hd, keys, KvCarrier::Q32).expect("geom");
        let mut seed = 0x1234_5678_9ABC_DEF0_u64;
        let mut s = 0usize;
        while s < keys {
            let k = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let v = [0i64; 10];
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let mut index = KvBlockIndex::new(1, n_head_kv, hd, block);
        let kplane = match &kv.storage {
            KvStorage::Q32 { k, .. } => &k[0],
            KvStorage::F32 { .. } | KvStorage::I8 { .. } => unreachable!(),
        };
        index.rebuild_layer(0, kplane, keys);

        let mut draw = 0usize;
        while draw < 64 {
            let q = [
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
                word(&mut seed),
            ];
            let mut kvh = 0usize;
            while kvh < n_head_kv {
                let mut b = 0usize;
                while b < keys / block {
                    let (lb, ub) = index
                        .bounds(0, kvh, b, keys, &q, kv.scale)
                        .expect("a complete block has a bound");
                    let mut s = b * block;
                    while s < (b + 1) * block {
                        let score =
                            score_at(kplane, &q, kv.scale, n_head_kv, kvh, hd, s).expect("score");
                        assert!(
                            lb <= score && score <= ub,
                            "draw {draw} kvh {kvh} block {b} s {s}: score {score} outside [{lb}, {ub}]"
                        );
                        s += 1;
                    }
                    b += 1;
                }
                kvh += 1;
            }
            draw += 1;
        }
    }

    #[test]
    fn a_partial_block_has_no_bound() {
        let mut index = KvBlockIndex::new(1, 1, 2, 4);
        let plane = vec![0i64; 4 * 2 * 2];
        index.rebuild_layer(0, &plane, 6); // one full block and a partial one
        let q = [0i64; 2];
        assert!(index.bounds(0, 0, 0, 6, &q, ONE).is_some());
        assert!(index.bounds(0, 0, 1, 6, &q, ONE).is_none());
    }

    #[test]
    fn truncate_keeps_only_wholly_inside_blocks() {
        let mut index = KvBlockIndex::new(1, 1, 2, 4);
        let plane = vec![0i64; 8 * 2];
        index.rebuild_layer(0, &plane, 8);
        index.truncate(6);
        let q = [0i64; 2];
        assert!(index.bounds(0, 0, 0, 6, &q, ONE).is_some());
        assert!(index.bounds(0, 0, 1, 6, &q, ONE).is_none());
    }

    #[test]
    fn clamp_keeps_bounds_sound_at_the_i64_edges() {
        assert_eq!(clamp_i64(i128::from(i64::MAX) + 1), i64::MAX);
        assert_eq!(clamp_i64(i128::from(i64::MIN) - 1), i64::MIN);
        assert_eq!(clamp_i64(-3), -3);
    }
}
