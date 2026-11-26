//! KV prefix cache: O(1) table lookup instead of O(prompt) prefill on the crate's Table.
//!
//! # Causality and Exactness
//!
//! In causal autoregressive transformer attention, the key/value state rows for sequence
//! positions `< m` depend exclusively on prefix tokens `t[0..m]`. Specifically, position `p`
//! consumes `t[p]` and previous keys `0..=p` through the causal lower-triangular mask; tokens at
//! positions `≥ m` have no backward impact on the keys, values, or normalization states at positions
//! `< m`. Therefore, a [`KvState`] snapshot captured after stepping tokens `t[0..m]` is bit-for-bit
//! identical to the state that a cold prefill reaches at position `m`.
//!
//! # Architecture and Full-Key Identity
//!
//! [`PrefixCache`] keys snapshots at block boundaries (every `B` tokens, default `B = 16`).
//! Probing checks prefixes from the longest boundary down: `m = (n / B) * B, ..., B`.
//! For prompt length `n`, this requires at most `n / B` lookups.
//!
//! Under Apeiron's Table architecture, every key lookup must respect the full-key identity rule:
//! a 64-bit hash hit alone is never trusted. Two distinct prefix token sequences could collide on
//! the 64-bit hash. A hit is accepted if and only if the full stored token sequence equals the
//! query prefix `t[0..m]`. A hash hit with mismatched tokens is treated as a miss.
//!
//! # Memory Footprint and O(n) Storage
//!
//! To avoid O(n²) memory overhead from cloning an independent [`KvState`] at every block boundary,
//! the cache retains exactly ONE [`KvState`] per distinct cached prompt in an arena. The Apeiron
//! [`Table`] maps 64-bit boundary keys to [`BoundaryRef`] `(entry_id, m)`.
//!
//! On a hit at boundary `m ≤ n`:
//! - If `m == entry.tokens.len()`, the stored state already holds exactly `m` occupied positions.
//! - If `m < entry.tokens.len()`, a clone of the prompt's state is truncated in-place to `m`
//!   positions via [`truncate_kv_state`], which sets the layer occupied counters to `m`.
//!
//! Snapshot memory is dominated by the key and value planes, at the state's carrier width:
//! - Per position, the footprint is `2 · n_head_kv · head_dim · n_layer` values times the carrier
//!   width: 4 B for `F32`, 8 B for `Q32`, and 1 B plus one `f32` scale per row per plane for `I8`.
//! - The charge is that times `cap`, because a clone carries whole planes, not the occupied prefix.
//! - Qwen3-0.6B (28 layers, 8 kv heads × 128): 57,344 values/position, 229,376 B at `F32`, so
//!   405 MiB at capacity 1,851.
//! - The 8B (36 layers, 8 kv heads × 128): 73,728 values/position, 294,912 B at `F32`, so
//!   580 MiB at capacity 2,064.
//!
//! # Eviction
//!
//! [`PrefixCache::insert`] admits a snapshot only while `current_bytes + needed ≤ byte_budget`. When
//! a snapshot would exceed the budget, the cache evicts whole prompt entries in least-recently-used
//! order until it fits. Recency is a monotone `u64` tick stored per arena entry; both a successful
//! [`PrefixCache::insert`] and a confirmed [`PrefixCache::longest_prefix`] hit advance the tick. A
//! snapshot whose own footprint exceeds the whole budget is refused with
//! [`PrefixCacheError::BudgetExceeded`], and the cache is left unchanged.
//!
//! The Apeiron [`Table`] exposes no removal operation, so eviction frees an entry's arena slot and
//! its bytes but leaves that entry's boundary keys in the table. Stale keys are safe misses:
//! [`PrefixCache::longest_prefix`] confirms every candidate by comparing the stored token sequence
//! with the queried prefix `t[0..m]`, so a key whose slot is freed or reused by a *different* prompt
//! misses. When the reused slot's prompt shares the first `m` tokens the state is causally identical,
//! so the hit is correct.
//!
//! # Complexity
//!
//! - `PrefixCache::longest_prefix`: `O(n / B)` probes down boundary candidates, each probe an `O(1)`
//!   table lookup with `O(m)` full-key token comparison on hit, where `m ≤ n`, plus one `O(1)` tick
//!   update on a confirmed hit.
//! - `PrefixCache::insert`: `O(m)` hashing and token cloning, `O(m / B)` boundary insertions, plus
//!   eviction: one `O(entries)` LRU scan per evicted entry, so `O(entries²)` when the whole cache is
//!   recycled.
//! - `prefill_with_cache`: resumes from prefix `m` (saving `m` positions of `O(n_layer · ...)`
//!   compute), then solves the remaining `n - m - 1` positions in block passes and steps only the
//!   last token. Overall speedup is proportional to `m / n`.

use crate::engine::attn::KvState;
use crate::engine::loader::Engine;
use crate::engine::step::{StepError, StepResult};
use crate::engine::{argmax_fixed, fixed_vector_checksum, KernelMode};
use crate::map::coord::locator::ExactIdentity;
use crate::map::freq::{Coord, Table};

/// Default block boundary step in tokens.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// A hasher abstraction for 64-bit prefix hashing.
pub trait PrefixHasher: Send + Sync {
    /// Hashes a slice of token ids into a 64-bit key.
    fn hash_tokens(&self, tokens: &[u32]) -> u64;
}

/// FNV-1a 64-bit hasher: standard deterministic default.
#[derive(Debug, Clone, Copy, Default)]
pub struct FnvPrefixHasher;

impl PrefixHasher for FnvPrefixHasher {
    /// Temporal complexity: O(len) byte iterations; spatial: O(1) in registers.
    fn hash_tokens(&self, tokens: &[u32]) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0100_0000_01b3;

        let mut hash = FNV_OFFSET_BASIS;
        for &tok in tokens {
            for b in tok.to_le_bytes() {
                hash ^= u64::from(b);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        hash
    }
}

/// A reference from a block boundary to a cached prompt entry and token count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BoundaryRef {
    /// Arena slot index of the parent prompt entry.
    pub entry_id: u32,
    /// Token boundary length.
    pub m: u32,
}

/// A stored entry for a distinct cached prompt in the arena.
#[derive(Clone)]
pub struct PromptEntry {
    /// Full token sequence for full-key identity confirmation.
    pub tokens: Vec<u32>,
    /// The captured KV snapshot after prefilling `tokens`.
    pub state: KvState,
    /// Estimated memory footprint in bytes.
    pub byte_size: usize,
    /// Monotone recency tick of the last use (insert or confirmed hit); lower means older.
    pub last_used: u64,
}

/// Key type for the Apeiron Table: carries the primary 64-bit hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrefixKey(pub u64);

impl ExactIdentity for PrefixKey {
    type Retained = u64;
    type Reserved = ();

    fn retain(&self) -> u64 {
        self.0
    }

    fn reserve_retain(&self) -> Result<(), crate::map::coord::locator::Refusal> {
        Ok(())
    }

    fn retain_reserved(&self, (): ()) -> u64 {
        self.0
    }

    fn matches_retained(&self, retained: &u64) -> bool {
        self.0 == *retained
    }

    fn encoding_len(&self) -> usize {
        8
    }

    fn encoding_byte(&self, index: usize) -> u8 {
        self.0.to_be_bytes()[index]
    }

    fn retained_encoding_len(_: &u64) -> usize {
        8
    }

    fn retained_encoding_byte(retained: &u64, index: usize) -> u8 {
        retained.to_be_bytes()[index]
    }
}

/// [`PrefixKey`]'s address in the executed map face: the key's 64-bit hash IS the coordinate.
///
/// Injectivity: `PrefixKey(a).coord() == PrefixKey(b).coord()` iff `a == b`, because the coordinate
/// is exactly the stored field and `PrefixKey`'s derived equality compares that field. `INJECTIVE`
/// is therefore `true`, which lets `Stored` skip the stored-key compare on a coordinate match —
/// sound here because no two distinct keys share a coordinate.
impl Coord for PrefixKey {
    const INJECTIVE: bool = true;

    #[inline]
    fn coord(&self) -> u64 {
        self.0
    }
}

/// Refusal and runtime errors for prefix cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixCacheError {
    /// Inserting the snapshot would exceed the configured byte budget.
    BudgetExceeded {
        /// Requested snapshot footprint in bytes.
        needed: usize,
        /// Current cache usage in bytes.
        current: usize,
        /// Maximum allowed bytes.
        budget: usize,
    },
    /// Prefill decode execution error.
    Step(StepError),
    /// Prompt tokens slice was empty.
    EmptyPrompt,
    /// Cache snapshot capacity was smaller than required prompt length.
    CapacityExceeded,
}

impl From<StepError> for PrefixCacheError {
    fn from(err: StepError) -> Self {
        Self::Step(err)
    }
}

/// The prefix cache using the executed map face.
///
/// Stores ONE [`KvState`] per distinct cached prompt in an arena, and indexes all
/// block boundaries in the [`Table`] pointing to `(entry_id, m)`.
pub struct PrefixCache<H: PrefixHasher = FnvPrefixHasher> {
    /// Boundary block size in tokens.
    block_size: usize,
    /// Maximum allowed byte footprint across all retained snapshots.
    byte_budget: usize,
    /// Currently committed byte footprint.
    current_bytes: usize,
    /// Prompt arena holding one entry per distinct prompt.
    arena: Vec<Option<PromptEntry>>,
    /// Free slot indices in the arena.
    free_slots: Vec<usize>,
    /// Monotone recency clock; each use stamps an entry with the next value.
    tick: u64,
    /// The executed map face mapping 64-bit hash keys to boundary references.
    table: Table<PrefixKey, Option<BoundaryRef>>,
    /// Pluggable hasher for token slices.
    hasher: H,
}

impl PrefixCache<FnvPrefixHasher> {
    /// Creates a new prefix cache with default FNV-1a hasher and default block size (16).
    #[must_use]
    pub fn new(byte_budget: usize) -> Self {
        Self::with_block_size_and_hasher(byte_budget, DEFAULT_BLOCK_SIZE, FnvPrefixHasher)
    }

    /// Creates a new prefix cache with custom block size and default FNV-1a hasher.
    #[must_use]
    pub fn with_block_size(byte_budget: usize, block_size: usize) -> Self {
        Self::with_block_size_and_hasher(byte_budget, block_size, FnvPrefixHasher)
    }
}

impl<H: PrefixHasher> PrefixCache<H> {
    /// Creates a new prefix cache with custom budget, block size, and hasher.
    #[must_use]
    pub fn with_block_size_and_hasher(byte_budget: usize, block_size: usize, hasher: H) -> Self {
        let block_size = block_size.max(1);
        Self {
            block_size,
            byte_budget,
            current_bytes: 0,
            arena: Vec::new(),
            free_slots: Vec::new(),
            tick: 0,
            table: Table::new(),
            hasher,
        }
    }

    /// Returns the configured block size in tokens.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the configured byte budget.
    #[must_use]
    pub fn byte_budget(&self) -> usize {
        self.byte_budget
    }

    /// Returns the current total byte usage of stored snapshots.
    #[must_use]
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Returns the number of distinct prompt entries stored in the arena.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.arena.iter().filter(|e| e.is_some()).count()
    }

    /// The resident bytes of a [`KvState`] snapshot: the clone carries whole planes at `cap`, so
    /// the charge is the carrier's bytes per position times the capacity, not the occupied prefix.
    /// Temporal: O(1); spatial: O(1).
    #[must_use]
    pub fn estimate_snapshot_bytes(kv: &KvState) -> usize {
        kv.bytes_per_position().saturating_mul(kv.cap())
    }

    /// Probes for the longest cached prefix matching `tokens`.
    ///
    /// Checks boundaries `m = (len / B) * B` down to `B`.
    /// On a 64-bit key hit in the table, verifies the full token sequence `t[0..m]`.
    /// Returns `Some((m, &KvState))` on confirmed match, or `None`. A confirmed hit counts as a use
    /// and advances the entry's recency tick.
    ///
    /// # Complexity
    ///
    /// `O(len / B)` Table lookups. Each lookup is `O(1)` in probe width, followed by
    /// an `O(m)` full-key token comparison on hit, plus one `O(1)` tick update on a confirmed hit.
    #[must_use]
    pub fn longest_prefix(&mut self, tokens: &[u32]) -> Option<(usize, &KvState)> {
        let n = tokens.len();
        if n < self.block_size {
            return None;
        }

        let max_blocks = n / self.block_size;
        let mut b = max_blocks;
        let mut hit: Option<(usize, usize)> = None;
        while b > 0 {
            let m = b * self.block_size;
            let prefix = &tokens[..m];
            let hash = self.hasher.hash_tokens(prefix);
            let key = PrefixKey(hash);

            if let Some(b_ref) = self.table.get(&key).copied().flatten() {
                if let Some(Some(entry)) = self.arena.get_mut(b_ref.entry_id as usize) {
                    // Full-key identity confirmation: hash alone is never trusted!
                    if entry.tokens.len() >= m && &entry.tokens[..m] == prefix {
                        let tick = self.tick.wrapping_add(1);
                        entry.last_used = tick;
                        self.tick = tick;
                        hit = Some((m, b_ref.entry_id as usize));
                        break;
                    }
                }
            }
            b -= 1;
        }

        let (m, slot) = hit?;
        Some((m, &self.arena[slot].as_ref()?.state))
    }

    /// Returns the arena slot of the least-recently-used entry, if any.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(entries)` full scan of the arena; spatial: `O(1)`.
    fn lru_victim(&self) -> Option<usize> {
        let mut victim: Option<(usize, u64)> = None;
        for (slot, entry) in self.arena.iter().enumerate() {
            if let Some(e) = entry {
                if victim.is_none_or(|(_, t)| e.last_used < t) {
                    victim = Some((slot, e.last_used));
                }
            }
        }
        victim.map(|(slot, _)| slot)
    }

    /// Evicts the entry in `slot`, dropping its state and releasing its bytes.
    ///
    /// The entry's boundary keys stay in the [`Table`] (it exposes no removal); a later probe
    /// resolves them through full-key token confirmation, so a stale key is a miss unless a reused
    /// slot holds an entry with the same prefix tokens.
    ///
    /// # Complexity
    ///
    /// Temporal: `O(bytes)` drop of the [`KvState`]; spatial: `O(1)` beyond the freed entry.
    fn evict_entry(&mut self, slot: usize) {
        if let Some(entry) = self.arena[slot].take() {
            self.current_bytes = self.current_bytes.saturating_sub(entry.byte_size);
            self.free_slots.push(slot);
        }
    }

    /// Inserts a snapshot of `kv` corresponding to `tokens` into the cache.
    ///
    /// Stores exactly ONE [`KvState`] in the prompt arena and indexes each block boundary
    /// `b · B ≤ m` into the map face as `(entry_id, b · B)`.
    ///
    /// When the snapshot would exceed `byte_budget`, whole prompt entries are evicted in
    /// least-recently-used order until it fits.
    ///
    /// # Errors
    ///
    /// - Returns [`PrefixCacheError::BudgetExceeded`] when the snapshot alone exceeds `byte_budget`,
    ///   leaving the cache completely unchanged.
    /// - Returns [`PrefixCacheError::Step`] with [`StepError::BadConfig`] when `kv` holds fewer
    ///   positions than `tokens`.
    /// - Returns [`PrefixCacheError::CapacityExceeded`] if boundary length exceeds `u32::MAX`.
    ///
    /// # Complexity
    ///
    /// `O(m)` where `m = tokens.len()` for hashing and token cloning, plus `O(m / B)`
    /// map boundary insertions each taking `O(1)` time, plus eviction at `O(entries)` per evicted
    /// entry (`O(entries²)` when the whole cache is recycled).
    pub fn insert(&mut self, tokens: &[u32], kv: &KvState) -> Result<(), PrefixCacheError> {
        let m = tokens.len();
        if m == 0 {
            return Ok(());
        }

        // The snapshot must hold the prompt it is filed under; a shorter state would resume at
        // the wrong offset.
        let occ = kv
            .min_occupied()
            .map_err(|e| PrefixCacheError::Step(StepError::Kv(e)))?;
        if occ < m {
            return Err(PrefixCacheError::Step(StepError::BadConfig(format!(
                "the snapshot holds {occ} positions for a {m}-token prompt"
            ))));
        }

        let needed = Self::estimate_snapshot_bytes(kv);

        // A snapshot that cannot fit even in an empty cache is refused before any mutation.
        if needed > self.byte_budget {
            return Err(PrefixCacheError::BudgetExceeded {
                needed,
                current: self.current_bytes,
                budget: self.byte_budget,
            });
        }

        // Evict least-recently-used whole entries until the snapshot fits. The loop terminates
        // because `needed ≤ byte_budget`: after evicting every entry the free space is the budget.
        while needed > self.byte_budget.saturating_sub(self.current_bytes) {
            let Some(victim) = self.lru_victim() else {
                break;
            };
            self.evict_entry(victim);
        }

        let slot = if let Some(free_slot) = self.free_slots.pop() {
            free_slot
        } else {
            let s = self.arena.len();
            self.arena.push(None);
            s
        };

        let entry_id = u32::try_from(slot).map_err(|_| PrefixCacheError::CapacityExceeded)?;

        let tick = self.tick.wrapping_add(1);
        let entry = PromptEntry {
            tokens: tokens.to_vec(),
            state: kv.clone(),
            byte_size: needed,
            last_used: tick,
        };

        self.arena[slot] = Some(entry);
        self.current_bytes += needed;
        self.tick = tick;

        let max_b = m / self.block_size;
        for b in 1..=max_b {
            let boundary_m = b * self.block_size;
            let m_u32 =
                u32::try_from(boundary_m).map_err(|_| PrefixCacheError::CapacityExceeded)?;
            let prefix = &tokens[..boundary_m];
            let hash = self.hasher.hash_tokens(prefix);
            let key = PrefixKey(hash);
            let b_ref = BoundaryRef { entry_id, m: m_u32 };

            // The map face merges with last-write-wins: an existing key for the same hash is
            // replaced by the newest snapshot, and insertion cannot refuse.
            self.table.insert_with(key, Some(b_ref), |new, _old| new);
        }

        Ok(())
    }
}

/// Truncates a cloned [`KvState`] to `m` occupied positions through [`KvState::truncate`];
/// `false` when a layer holds fewer than `m` positions.
///
/// # Complexity
///
/// `O(n_layer)`, as [`KvState::truncate`].
pub fn truncate_kv_state(kv: &mut KvState, m: usize) -> bool {
    kv.truncate(m).is_ok()
}

/// Prefills a sequence of tokens with prefix caching.
///
/// 1. Queries `cache` for the longest cached prefix of `tokens` strictly less than `tokens.len()`.
/// 2. If a prefix of length `m` is found, clones its `KvState` as the starting state,
///    and truncates it in-place to `m` positions via [`truncate_kv_state`].
///    Otherwise, creates a fresh cold `KvState` with capacity `tokens.len() + extra_capacity`.
/// 3. Solves the remaining tokens `tokens[m..n-1]` in block passes over the weights, then steps
///    only the last token for its [`StepResult`].
/// 4. If `insert_boundaries` is true, stores the completed prompt snapshot into the cache.
/// 5. Returns the final `KvState` and the last `StepResult`.
///
/// This wrapper runs [`KernelMode::Exact`]; [`prefill_with_cache_mode`] selects the reader.
///
/// # Errors
///
/// - Returns [`PrefixCacheError::EmptyPrompt`] if `tokens` is empty.
/// - Returns [`PrefixCacheError::CapacityExceeded`] if cache snapshot capacity is smaller
///   than needed.
/// - Returns [`PrefixCacheError::Step`] on a projection or decode refusal.
///
/// # Complexity
///
/// `O(n / B)` lookup time + `O(⌈(n − m) / w⌉)` block passes over the weights, `w` the engine's
/// prefill block width, and one single-token step. When `m == n - 1`, only the step runs.
pub fn prefill_with_cache<H: PrefixHasher>(
    engine: &Engine,
    cache: &mut PrefixCache<H>,
    tokens: &[u32],
    extra_capacity: usize,
    insert_boundaries: bool,
) -> Result<(KvState, StepResult), PrefixCacheError> {
    prefill_with_cache_mode(
        engine,
        cache,
        tokens,
        extra_capacity,
        insert_boundaries,
        KernelMode::Exact,
    )
}

/// [`prefill_with_cache`] under a projection kernel: the cached prefix is the solved part, and the
/// missed tail is one block pass, not one step per token.
///
/// # Errors
///
/// As [`prefill_with_cache`], plus [`PrefixCacheError::Step`] with [`StepError::BadConfig`] when
/// the model carries affine layer state, which a [`KvState`] snapshot does not represent.
///
/// # Complexity
///
/// As [`prefill_with_cache`].
pub fn prefill_with_cache_mode<H: PrefixHasher>(
    engine: &Engine,
    cache: &mut PrefixCache<H>,
    tokens: &[u32],
    extra_capacity: usize,
    insert_boundaries: bool,
    mode: KernelMode,
) -> Result<(KvState, StepResult), PrefixCacheError> {
    let n = tokens.len();
    if n == 0 {
        return Err(PrefixCacheError::EmptyPrompt);
    }

    let total_cap = n
        .checked_add(extra_capacity)
        .ok_or(PrefixCacheError::Step(StepError::LengthOverflow))?;

    // Probe for a prefix strictly less than `n` so at least the last token is stepped to produce
    // the step result.
    let prefix_bound = &tokens[..n.saturating_sub(1)];
    let (start_pos, kv) = if let Some((m, cached_state)) = cache.longest_prefix(prefix_bound) {
        if cached_state.cap() < total_cap {
            return Err(PrefixCacheError::CapacityExceeded);
        }
        let mut kv = cached_state.clone();
        let occ = kv
            .min_occupied()
            .map_err(|e| PrefixCacheError::Step(StepError::Kv(e)))?;
        if occ < m {
            return Err(PrefixCacheError::Step(StepError::BadConfig(format!(
                "the cached state holds {occ} positions, its boundary claims {m}"
            ))));
        }
        if occ > m && !truncate_kv_state(&mut kv, m) {
            return Err(PrefixCacheError::CapacityExceeded);
        }
        (m, kv)
    } else {
        (
            0,
            engine
                .decode_state(total_cap)
                .map_err(PrefixCacheError::Step)?,
        )
    };

    let mut st = engine.state_with_kv(kv);
    if st.affine.is_some() {
        return Err(PrefixCacheError::Step(StepError::BadConfig(String::from(
            "the prefix cache stores softmax KV only; the model has affine layer state",
        ))));
    }

    // The missed tail is one closed pass of the layer stack at the engine's prefill width; only
    // the last token is a single step, because its logits are the caller's result.
    if start_pos < n - 1 {
        engine
            .prefill_state_with_width(
                &mut st,
                &tokens[start_pos..n - 1],
                mode,
                crate::engine::step::prefill_block(),
            )
            .map_err(PrefixCacheError::Step)?;
    }
    let logits = engine
        .step_logits_with_mode(&mut st.kv, tokens[n - 1], mode)
        .map_err(PrefixCacheError::Step)?;
    let token = argmax_fixed(&logits)
        .map_err(StepError::Sampler)
        .map_err(PrefixCacheError::Step)?;
    let result = StepResult {
        token,
        logits_checksum: fixed_vector_checksum(&logits),
    };

    if insert_boundaries && n >= cache.block_size() && start_pos < n {
        let _ = cache.insert(tokens, &st.kv);
    }

    Ok((st.kv, result))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
