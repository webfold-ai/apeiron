//! The NB face — `FreqMap` and the `FreqLine` group-probe engine it rides on. Split out of `lib.rs`,
//! re-exported from it, so every path is unchanged. A separate structure from `Table`: SwissTable
//! control bytes, no key column, no deletions.

use crate::map::hashtrinity::overflow::Overflow;
use crate::map::hashtrinity::{
    bits_for_slots, is_sentinel, par, Coord, CoordInv, Regen, Table, EMPTY, PHI, PROBE_CAP,
};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// THE COUNT FACE — sequential helpers (the refinable steps) and the parallel fan-out around them.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// The sequential steps of the parallel count. Each is a plain loop over slices with its values
/// carried (no early `return` in a loop, no iterator adapter, no closure, no raw pointer, no
/// `unsafe`), so the Aeneas fragment can extract it. `count_par` calls exactly these; the thread
/// fan-out around them (disjoint windows, one barrier per phase) is the only code that is not
/// sequential, and stays a named premise. Composed single-threaded they compute the group-by count
/// `Apeiron.Table.build (+) 0` over the keys' coordinates.
///
/// Refusal (`false` / `None`) replaces every overflow: a `u32` histogram entry, a `usize` offset
/// and an `i64` count each fail closed.
impl<K: Coord> FreqMap<K, i64> {
    /// Bucket-index width: the `COUNT_BBITS` bits of `c·PHI` the partition reads.
    pub const COUNT_BBITS: u32 = 8;
    /// `2^COUNT_BBITS` home buckets.
    pub const COUNT_BUCKETS: usize = 1 << Self::COUNT_BBITS;

    /// The window above the bucket window reserved for the probe's own `p = f·PHI` reads: the top
    /// `gbits` bits are the home/descent window (`FreqLine::home_group`) and the 7 bits below those
    /// are the tag (`FreqLine::tag`). A phase-3 map holds at most `2^COUNT_HOME_GAP` slots in the
    /// counted population (`gbits ≤ COUNT_HOME_GAP − 4`), so the bucket and tag windows stay
    /// **disjoint** from the home window. That disjointness is the lever: disjoint bits vary freely
    /// inside a bucket, so its keys start in up to `2^gbits` distinct home groups instead of only
    /// `2^(gbits−8)` when the bucket reused the top of the home window.
    pub const COUNT_HOME_GAP: u32 = 16;

    /// The bucket window shift: bits `[COUNT_BUCKET_SHIFT, COUNT_BUCKET_SHIFT + COUNT_BBITS)` of
    /// `c·PHI`. Budget `COUNT_BBITS + COUNT_HOME_GAP + 7 = 31 ≤ 64` (the `+7` is the tag window).
    pub const COUNT_BUCKET_SHIFT: u32 = 64 - Self::COUNT_BBITS - Self::COUNT_HOME_GAP;

    /// L3: the capped start capacity, in keys, of one phase-3 bucket's map
    /// ([`count_aggregate_start`](Self::count_aggregate_start)). Occurrences bound the insertions
    /// but not the distinct keys, so the start is capped here and the final capacity is set by the
    /// map's own growth from the bucket's DISTINCT count. `with_capacity(8)` opens a 16-slot line
    /// (one group). O(1).
    pub const COUNT_AGGREGATE_START: usize = 8;

    /// The home bucket of coordinate `c`: the `COUNT_BBITS` bits of `c·PHI` just below the probe's
    /// home window (bits `[40, 48)`), disjoint from `FreqLine::home_group`'s top `gbits` and from
    /// `FreqLine::tag`'s window. BOUND: the mask states `[0, 2^COUNT_BBITS)` to the optimiser. O(1).
    #[inline]
    pub fn count_home(c: u64) -> usize {
        (c.wrapping_mul(PHI) >> Self::COUNT_BUCKET_SHIFT) as usize & (Self::COUNT_BUCKETS - 1)
    }

    /// The fused count-face read: [`count_home`](Self::count_home)'s bucket select and
    /// [`FreqMap::get`]'s probe formed from ONE `p = k·PHI`. The three windows of `p` are disjoint —
    /// bucket `[COUNT_BUCKET_SHIFT, COUNT_BUCKET_SHIFT + COUNT_BBITS)` = `[40, 48)`, tag
    /// `[TAG_SHIFT, 40)` = `[33, 40)`, home group the top `gbits` bits below the tag — so the single
    /// product feeds all three and the probe's second `k·PHI` is removed. `maps` holds the
    /// `COUNT_BUCKETS` shards of one [`count_par`](Self::count_par) (any `len`; the bucket index is
    /// masked to `[0, COUNT_BUCKETS)` and the slice bound is checked). The single-map serial shape
    /// (`maps.len() == 1`) has no bucket to select and uses `maps[0].get`. O(1).
    #[inline]
    pub fn count_get<'a>(maps: &'a [Self], k: &K) -> Option<&'a i64> {
        let f = Self::freq(k);
        let p = f.wrapping_mul(PHI);
        let b = (p >> Self::COUNT_BUCKET_SHIFT) as usize & (Self::COUNT_BUCKETS - 1);
        maps[b].inner.get_p(p, f)
    }

    /// [`count_get`](Self::count_get) with the shard count in the type (`maps.len()` is
    /// `COUNT_BUCKETS`), so the masked bucket index `[0, COUNT_BUCKETS)` needs no bounds check: the
    /// compiler proves it in range. `maps` MUST be the full `COUNT_BUCKETS` shards of one
    /// [`count_par`](Self::count_par); the single-map serial shape does not fit this type and uses
    /// `maps[0].get`. O(1).
    #[inline]
    pub fn count_get_arr<'a>(
        maps: &'a [Self; FreqMap::<u64, i64>::COUNT_BUCKETS],
        k: &K,
    ) -> Option<&'a i64> {
        let f = Self::freq(k);
        let p = f.wrapping_mul(PHI);
        let b = (p >> Self::COUNT_BUCKET_SHIFT) as usize & (Self::COUNT_BUCKETS - 1);
        maps[b].inner.get_p(p, f)
    }

    /// Admission of an `n`-key count: every per-key count is at most `n`, so `n <= count_max`
    /// (production: `i64::MAX`) keeps every `i64` total in range. O(1).
    pub fn count_admit(n: usize, count_max: u64) -> bool {
        (n as u64) <= count_max
    }

    /// Phase 1, one shard: `homes[i] = home(keys[i])` and `hist[h] += 1`, with every `u32` entry
    /// incremented by `checked_add`. `false` (refusal) if the lengths are wrong
    /// (`homes.len() != keys.len()` or `hist.len() != COUNT_BUCKETS`) or an entry would exceed
    /// `u32::MAX`; the loop stops at the first refusal. Time `Θ(keys.len())` key reads and
    /// increments, space `O(1)` beyond the slices: one pass, one iteration per key.
    pub fn count_histogram(keys: &[K], homes: &mut [u8], hist: &mut [u32]) -> bool {
        let n = keys.len();
        let mut ok = homes.len() == n && hist.len() == Self::COUNT_BUCKETS;
        let mut i = 0usize;
        while ok && i < n {
            let h = Self::count_home(keys[i].coord());
            homes[i] = h as u8;
            match hist[h].checked_add(1) {
                Some(v) => hist[h] = v,
                None => ok = false,
            }
            i += 1;
        }
        ok
    }

    /// Checked prefix sums over the `w × b` histogram (`counts[t*b + j]`, `b = COUNT_BUCKETS`):
    /// `bstart[j]` is the first routed slot of bucket `j` (`bstart[b] = total`) and `woff[t*b + j]`
    /// the first slot worker `t` writes in bucket `j`, ordered by `t`. `false` (refusal) if a
    /// length is wrong, a `usize` sum overflows, or `total != n` (the routed buffer is only fully
    /// written when the entries sum to `n`). Time `Θ(w·b)` additions, space `O(1)` beyond the
    /// outputs: one pass over the `b` buckets × `w` workers.
    pub fn count_offsets(
        counts: &[u32],
        w: usize,
        n: usize,
        bstart: &mut [usize],
        woff: &mut [usize],
    ) -> bool {
        let b = Self::COUNT_BUCKETS;
        let wb = match w.checked_mul(b) {
            Some(x) => x,
            None => return false,
        };
        let mut ok = counts.len() == wb && woff.len() == wb && bstart.len() == b + 1;
        let mut acc = 0usize;
        let mut j = 0usize;
        while ok && j < b {
            bstart[j] = acc;
            let mut t = 0usize;
            while ok && t < w {
                woff[t * b + j] = acc;
                match acc.checked_add(counts[t * b + j] as usize) {
                    Some(v) => acc = v,
                    None => ok = false,
                }
                t += 1;
            }
            j += 1;
        }
        if ok {
            bstart[b] = acc;
        }
        ok && acc == n
    }

    /// Phase 2, one key: the destination slot of a key of home `h` for the shard whose write cursors
    /// are `off`, and the cursor advance. The shard's loop applies it to every key in order
    /// (`routed[count_slot(off, homes[i])] = keys[i]`); the cursors are the `woff` row of that shard,
    /// so slots are disjoint across shards and within a shard. O(1).
    #[inline]
    pub fn count_slot(off: &mut [usize], h: u8) -> usize {
        let j = usize::from(h);
        let s = off[j];
        off[j] = s + 1;
        s
    }

    /// The start capacity passed to [`Self::with_capacity`] for a phase-3 bucket holding `hi - lo`
    /// routed occurrences: `min(hi - lo, COUNT_AGGREGATE_START)`. Taking both bounds in one call
    /// keeps the Aeneas extraction of [`count_aggregate`](Self::count_aggregate) at ONE capacity
    /// bind, so the canonical `Count.Aggregate.count_aggregate_refines_win` — which binds that
    /// capacity generically and feeds it to `with_capacity_rep`/`pot_with_capacity`, never reading
    /// its value — still type-checks without a corpus edit. O(1).
    fn count_aggregate_start(lo: usize, hi: usize) -> usize {
        hi.saturating_sub(lo).min(Self::COUNT_AGGREGATE_START)
    }

    /// Phase 3, one bucket: the count map of `routed[lo..hi]` (`m.add(k, 1)` per key; `Build (+) 0`
    /// over the bucket). A range outside `routed` is clipped to it. Time `Θ(hi - lo)` map
    /// insertions (each amortised O(1) at the group probe's bounded probe length — a hypothesis,
    /// `AggregationBound`), space `Θ(distinct)` slots.
    ///
    /// **Sizing by distinct keys (L3): grow-from-small.** The line opens at the capped start
    /// ([`count_aggregate_start`](Self::count_aggregate_start), 16 slots) and reaches its final
    /// capacity only through [`FreqLine::grow`]'s load-factor doubling: `c₀ = 2⁴`,
    /// `c_{k+1} = 2·c_k` when `len` reaches `(7/8)·c_k`. The final capacity is therefore
    /// `next_pow2((8/7)·distinct)` — a function of the bucket's DISTINCT keys, not its routed
    /// occurrences — with **no second pass** over `routed` and no distinct-count pre-pass. Cost
    /// recurrence: the keys re-placed over all grows total `Σ_k (7/8)·2^k < (7/8)·c_final`, i.e.
    /// **≤ ~1 extra key move per distinct key** (O(1) amortised per insert). Peak extra space: at
    /// one grow the old and new lines are live together, `c_k + 2·c_k = 1.5·c_{k+1}` — 1.5× the
    /// bucket's final line, transient for that grow. A 1%-distinct bucket that routes ~100× its
    /// distinct count therefore holds a distinct-sized (cache-resident) map instead of a ~100×
    /// over-allocated one.
    pub fn count_aggregate(routed: &[K], lo: usize, hi: usize) -> Self {
        let mut m = Self::with_capacity(Self::count_aggregate_start(lo, hi));
        let mut i = lo;
        while i < hi && i < routed.len() {
            m.add(&routed[i], 1);
            i += 1;
        }
        m
    }
}

/// Phases 1–2 of the count: partition `keys` by home bucket across the pool. Returns the keys
/// grouped by bucket and the bucket starts (`b + 1` entries), or `None` if admission refuses.
/// Each task passes only its own disjoint window to the sequential helpers; the raw pointers
/// are the fan-out premise.
fn count_route<K: Coord + Copy + Sync>(
    pool: &par::Pool,
    keys: &[K],
) -> Option<(Vec<K>, Vec<usize>)> {
    type F<K> = FreqMap<K, i64>;
    let w = pool.width();
    let n = keys.len();
    let b = F::<K>::COUNT_BUCKETS;
    let chunk = n / w + usize::from(n % w != 0);
    let refused = AtomicBool::new(false);

    // R1: phase 1 folds every key once and phase 2 reads the bucket back instead of refolding it
    // (`COUNT_BBITS <= 8`, so one byte carries `home` exactly).
    const _: () = assert!(FreqMap::<u64, i64>::COUNT_BBITS <= 8);
    let mut homes: Vec<u8> = vec![0u8; n];
    let hptr = par::SendPtr(homes.as_mut_ptr());
    let mut counts = vec![0u32; w * b];
    let cptr = par::SendPtr(counts.as_mut_ptr());
    pool.dispatch(w, |t| {
        let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
        let (lo, hi) = (lo.min(hi), hi);
        // SAFETY: task t's windows `homes[lo..hi]` and `counts[t*b..t*b+b]` are disjoint across t
        // and in bounds (`hi <= n`, `t < w`); the barrier orders the writes before the reads below.
        let (hw, cw) = unsafe {
            (
                std::slice::from_raw_parts_mut(hptr.add(lo), hi - lo),
                std::slice::from_raw_parts_mut(cptr.add(t * b), b),
            )
        };
        if !F::<K>::count_histogram(&keys[lo..hi], hw, cw) {
            refused.store(true, Ordering::Relaxed);
        }
    });
    if refused.load(Ordering::Relaxed) {
        return None;
    }
    let mut bstart = vec![0usize; b + 1];
    let mut woff = vec![0usize; w * b];
    if !F::<K>::count_offsets(&counts, w, n, &mut bstart, &mut woff) {
        return None;
    }

    // Phase 2 — scatter into `routed` grouped by bucket (disjoint offset ranges ⇒ no contention).
    // `K: Copy`, so writing uninitialised slots by pointer then `set_len` is sound.
    let mut routed: Vec<K> = Vec::with_capacity(n);
    let rptr = par::SendPtr(routed.as_mut_ptr());
    let woff_ref = &woff;
    let homes_ref = &homes;
    pool.dispatch(w, |t| {
        let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
        let lo = lo.min(hi);
        let mut off: Vec<usize> = woff_ref[t * b..t * b + b].to_vec();
        for (&h, &k) in homes_ref[lo..hi].iter().zip(&keys[lo..hi]) {
            let s = F::<K>::count_slot(&mut off, h);
            // SAFETY: `count_offsets` verified the cursors tile `[0, n)` and phase 1 counted exactly
            // this shard's keys per bucket, so `s < n` and each slot is written once across all tasks.
            unsafe { *rptr.add(s) = k };
        }
    });
    // SAFETY: every slot 0..n was written exactly once by phase 2 (Σ counts = n, checked).
    unsafe { routed.set_len(n) };
    Some((routed, bstart))
}

impl<K: Coord + Eq + Copy + Sync> FreqMap<K, i64> {
    /// **The parallel two-level GROUP BY count**, on the group-probe (NB) engine (`GroupProbe.lean`:
    /// uniform golden fold ⇒ no identity-home block, 16-wide tag glance ⇒ miss filter, bounded
    /// terminal sweep). Partition `keys` by home bucket (bits `[40,48)` of `k·PHI`, disjoint from
    /// the probe's home window — balanced, no hot bucket, `RadixDescent.home_balanced`), then aggregate each bucket into its own `FreqMap`
    /// across the pool. Buckets hold disjoint keys ⇒ **no merge**: a key is in exactly one returned
    /// map and `∑ maps[b].get(k)` is its count — the disjoint-fold case of
    /// `SharedFold.interleave_exact` (independent per-bucket folds, any schedule, exact).
    /// Below the crossover (small `n` / single core —
    /// `GatherCrossover`) it falls back to one serial aggregation. Cost is linear in `n` GIVEN a
    /// per-probe bound (`AggregationBound.build_linear_of_bounded_probes`, where that bound is a
    /// hypothesis, refuted worst-case); `2^-64` fold-collision fp (the join/count use case).
    ///
    /// Panics on admission refusal ([`try_count_par`](Self::try_count_par) returns `None`
    /// instead): a `u32` histogram entry, a `usize` offset or an `i64` count overflowing.
    #[must_use]
    pub fn count_par(keys: &[K]) -> Vec<Self> {
        Self::try_count_par(keys).expect(
            "count_par admission: a u32 histogram entry, usize offset or i64 count would overflow",
        )
    }

    /// [`count_par`](Self::count_par) with admission refusal as `None` instead of a panic. The
    /// sequential steps are the `count_*` helpers; the pool fan-out is outside them.
    #[must_use]
    pub fn try_count_par(keys: &[K]) -> Option<Vec<Self>> {
        let pool = par::Pool::global();
        let w = pool.width();
        let n = keys.len();
        if !Self::count_admit(n, i64::MAX as u64) {
            return None;
        }
        // Serial fallback below the crossover: one map, no partition cost.
        if w <= 1 || n < (1 << 15) {
            return Some(vec![Self::count_aggregate(keys, 0, n)]);
        }
        let (routed, bstart) = count_route(pool, keys)?;
        let b = Self::COUNT_BUCKETS;

        // Phase 3 — aggregate each bucket into its own group-probe map, in parallel (disjoint keys
        // ⇒ no merge; schedule-free, `Web.preduceAt_schedule_free`). Static bucket chunks per worker.
        // R4: `Vec<Option<Self>>` allocates nothing for the `None` placeholders (a vector of
        // pre-built maps cost `2b` allocations per call); the unwrap is one `Vec<Self>` allocation.
        let mut maps: Vec<Option<Self>> = (0..b).map(|_| None).collect();
        let mptr = par::SendPtr(maps.as_mut_ptr());
        let bstart_ref = &bstart;
        let routed_ref = &routed;
        let bchunk = b.div_ceil(w);
        pool.dispatch(w, |t| {
            for j in (t * bchunk)..((t + 1) * bchunk).min(b) {
                let m = Self::count_aggregate(routed_ref, bstart_ref[j], bstart_ref[j + 1]);
                // SAFETY: task t owns bucket range [t*bchunk,..); disjoint j ⇒ disjoint maps[j].
                unsafe { *mptr.add(j) = Some(m) }
            }
        });
        // Every slot is written: the `w` dispatches partition `[0, b)` by `bchunk` with no gap.
        Some(maps.into_iter().map(|slot| slot.expect("phase 3 wrote every bucket")).collect())
    }
}

impl<K: CoordInv + Eq + Copy + Sync> Table<K, i64, Regen<K>> {
    /// **The parallel two-level GROUP BY count, KEYS RETAINED** (the Regen face), on the group-probe (NB) engine (`GroupProbe.lean`:
    /// uniform golden fold ⇒ no identity-home block, 16-wide tag glance ⇒ miss filter, bounded
    /// terminal sweep). Partition `keys` by home bucket (bits `[40,48)` of `k·PHI`, disjoint from
    /// the probe's home window — balanced, no hot bucket, `RadixDescent.home_balanced`), then aggregate each bucket into its own `FreqMap`
    /// across the pool. Buckets hold disjoint keys ⇒ **no merge**: a key is in exactly one returned
    /// map and `∑ maps[b].get(k)` is its count — the disjoint-fold case of
    /// `SharedFold.interleave_exact` (independent per-bucket folds, any schedule, exact).
    /// Below the crossover (small `n` / single core —
    /// `GatherCrossover`) it falls back to one serial aggregation. Cost is linear in `n` GIVEN a
    /// per-probe bound (`AggregationBound.build_linear_of_bounded_probes`, where that bound is a
    /// hypothesis, refuted worst-case); `2^-64` fold-collision fp (the join/count use case).
    /// Phases 1–2 are the shared `FreqMap` count helpers; panics on admission refusal.
    #[must_use]
    pub fn count_par(keys: &[K]) -> Vec<Self> {
        let pool = par::Pool::global();
        let w = pool.width();
        let n = keys.len();
        assert!(
            FreqMap::<K, i64>::count_admit(n, i64::MAX as u64),
            "count_par admission: i64 count would overflow"
        );
        // Serial fallback below the crossover: one map, no partition cost.
        if w <= 1 || n < (1 << 15) {
            let mut m = Self::coordinate_with_capacity(n);
            for &k in keys {
                m.add(k, 1);
            }
            return vec![m];
        }
        let (routed, bstart) = count_route(pool, keys).expect(
            "count_par admission: a u32 histogram entry, usize offset or i64 count would overflow",
        );
        let b = FreqMap::<K, i64>::COUNT_BUCKETS;

        // Phase 3 — aggregate each bucket into its own group-probe map, in parallel (disjoint keys
        // ⇒ no merge; schedule-free, `Web.preduceAt_schedule_free`). Static bucket chunks per worker.
        let mut maps: Vec<Self> = (0..b).map(|_| Self::coordinate_with_capacity(0)).collect();
        let mptr = par::SendPtr(maps.as_mut_ptr());
        let bstart_ref = &bstart;
        let routed_ref = &routed;
        let bchunk = b.div_ceil(w);
        pool.dispatch(w, |t| {
            for j in (t * bchunk)..((t + 1) * bchunk).min(b) {
                let (lo, hi) = (bstart_ref[j], bstart_ref[j + 1]);
                let mut m = Self::coordinate_with_capacity(hi - lo);
                for &k in &routed_ref[lo..hi] {
                    m.add(k, 1);
                }
                // SAFETY: task t owns bucket range [t*bchunk,..); disjoint j ⇒ disjoint maps[j].
                unsafe { *mptr.add(j) = m };
            }
        });
        maps
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// THE NB FACE — the frequency map, on a control-filtered frequency line.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// The side store for reserved-frequency keys (`EMPTY`/`TOMB`) in the count face, the [`FreqLine`]
/// twin of the map face's [`SideStore`](super::SideStore). A frequency equal to a reserved marker
/// cannot live in a primary lane — `slots[i].0 == EMPTY` is the occupancy flag — so the count for
/// the keys that literally own those frequencies is held here, outside the slot arrays. At most the
/// two reserved coordinates can appear, so the store is a packed linear scan that is empty on every
/// ordinary map.
struct FreqSide<V> {
    coords: Vec<u64>,
    vals: Vec<MaybeUninit<V>>,
    // `live[i]` is false only while a merge owns the value (panic safety, exactly as the primary
    // slot blanks its coordinate before the merge and restores it after).
    live: Vec<bool>,
}

impl<V> FreqSide<V> {
    fn empty() -> Self {
        FreqSide {
            coords: Vec::new(),
            vals: Vec::new(),
            live: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.coords.len()
    }

    fn find(&self, coord: u64) -> Option<usize> {
        let mut i = 0;
        while i < self.coords.len() {
            if self.coords[i] == coord {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    fn get(&self, coord: u64) -> Option<&V> {
        let i = self.find(coord)?;
        // SAFETY: `find` only returns an entry whose `live` marker is set.
        Some(unsafe { self.vals.get_unchecked(i).assume_init_ref() })
    }

    /// Set `coord → merge(new, old)`; a fresh coordinate stores `new` unmerged.
    fn insert_with(&mut self, coord: u64, new: V, merge: impl FnOnce(V, V) -> V) {
        if let Some(i) = self.find(coord) {
            // SAFETY: `live[i]` ⇒ value initialized. Blank the marker first so a panicking merge
            // leaves the entry skipped by `Drop` (the primary slot's blank/restore dance).
            let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
            self.live[i] = false;
            let merged = merge(new, old);
            // SAFETY: the value was moved out; re-writing and re-marking restores the invariant.
            unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(merged) };
            self.live[i] = true;
        } else {
            self.coords.push(coord);
            self.vals.push(MaybeUninit::new(new));
            self.live.push(true);
        }
    }

    fn iter_live(&self) -> impl Iterator<Item = (u64, &V)> {
        self.coords
            .iter()
            .copied()
            .zip(self.vals.iter())
            .zip(self.live.iter())
            .filter_map(|((c, v), live)| {
                if *live {
                    // SAFETY: `live[i]` ⇒ value initialized.
                    Some((c, unsafe { v.assume_init_ref() }))
                } else {
                    None
                }
            })
    }

    /// Drop every live value (the side twin of the primary slot walk). An indexed `while`, not an
    /// iterator adapter, so the extracted count face reaches it.
    fn drop_live(&mut self) {
        if std::mem::needs_drop::<V>() {
            let mut i = 0;
            while i < self.vals.len() {
                if self.live[i] {
                    // SAFETY: `live[i]` ⇒ value initialized.
                    unsafe { self.vals.get_unchecked_mut(i).assume_init_drop() };
                }
                i += 1;
            }
        }
    }
}

impl<V: Clone> Clone for FreqSide<V> {
    fn clone(&self) -> Self {
        FreqSide {
            coords: self.coords.clone(),
            // Every steady-state entry is initialized (`live` is false only during a merge).
            vals: self
                .vals
                .iter()
                .map(|v| MaybeUninit::new(unsafe { v.assume_init_ref() }.clone()))
                .collect(),
            live: self.live.clone(),
        }
    }
}

/// The frequency line: coords (frequencies) + values + a **packed byte control**, group probe.
///
/// The control is the first of the two flaw readings the Null cell forces on any lookup of opaque
/// data (sanguine `GoldenRouting.golden_control_is_the_first_flaw`, standing on
/// `NullReadout.one_flaw_is_blind`/`readout_injective`): a coarse reading computed from the query
/// itself, then an exact confirm against the stored residual (the Seed — the only bits that must
/// persist). Its group-granular control flow is `GroupProbe.lean` — SwissTable's three bare
/// assertions, proved:
///
/// * **the glance** — one `u64` ctrl word holds 8 byte lanes (a group of 8 slots, whose 8 coords
///   are exactly one 64-byte cache line); an exact SWAR haszero detector winnows all 8 at once,
///   and a filter is allowed to be winnowed sloppily because no-false-negatives + exact confirm =
///   exact (`GroupProbe.winnow_exact`);
/// * **the stop** — a probe halts at the first group holding an empty lane; sound because
///   placement is first-fit and occupancy is monotone — `FreqMap` never deletes
///   (`GroupProbe.stop_not_before_placement`; one deletion voids the licence,
///   `deletion_breaks_the_stop` — SwissTable's DELETED≠EMPTY tombstone, forced);
/// * **the tag** — each lane is `{0 = empty, else a 7-bit window of the fold, 0 remapped to 1}`. The
///   tag must be a window DISJOINT from the address windows: a top-window tag is a function of the
///   home address and rejects nothing exactly where the probe starts
///   (`GroupProbe.top_window_is_home_determined` — the bug the earlier top-bits band carried);
///   the deployed window ([`FreqLine::tag`], `TAG_SHIFT`) has exact false-positive mass `2⁻ᵗ` globally and within every home fiber
///   (`low_window_balanced`, `home_and_tag_jointly_balanced`). Its width is the `F = D + λR` rate
///   dial, and a full byte is the point that MATCHES SwissTable's h2: the earlier 4-bit nibble
///   was measured slower on misses (its `2⁻⁴` false-positive mass forced a cold coord read on
///   ~1/3 of misses), the dial's distortion term made visible — the byte pays it down to `2⁻⁸`.
///
/// The **group sequence is the PHI-descent** read at group granularity
/// (`GroupProbe.group_descent_dichotomy` — the closure: separate within the digit walk, or
/// share a pile of exactly `2^(64 mod gbits)` folds (`pile_card`), tag-resolved when the residual
/// fits the tag window (`pile_resolved_by_tag`); no mention of the tag in the walk itself): an
/// adversary who defeats the filter still meets the descent's SEPARATION — which bounds aliasing,
/// not probe length (`AggregationBound.descent_can_be_wholly_blocked`). The control can only
/// shortcut, never lengthen; that part is unaffected.
///
/// **The readout depth is a constant, not the word-length walk.** `probe`'s glance-then-confirm
/// is exactly `SpinorReadout.resolve₂` (proven in sanguine
/// `proof/Closures/Engineering/CoordinateReadout.lean`; retired from this crate's corpus — see
/// THEOREMS.md External): flaw one = the home group, flaw
/// two = the low-window `tag`, and the stored-coord confirm (`slots[i].0 == f`) fires ONLY when
/// both collide. So the readout is a **spinor — it resolves in depth ≤ 2, the double cover**
/// (`SpinorReadout.spinor_depth_two`), not the variable radix length `D = ⌈W/b⌉` that
/// `RadixDescent.chain_separates_within_radix` bounds the single-projection walk by. Three facts,
/// none assuming injectivity: the verdict is the exact membership test for every key
/// (`resolve₂_correct`); the confirm fires iff the two windows jointly collide
/// (`resolve₂_touch_iff_double_collision`), which is the top-`(t+b)` fiber — of exactly
/// `2^(W−t−b)` folds (`home_and_tag_jointly_balanced`; the adjacent-window twin is
/// `SpinorReadout.concrete_spinor_depth_two`'s `top_pair_fiber_card`); everywhere else the depth-2
/// coordinate read IS the whole lookup, no key touched (`resolve₂_no_touch_of_separated`). This is
/// the readout's proof — depth-2, exact, flood-bounded — superseding the older
/// "separate-within-`D+1`" sketch for the deployed descend.
struct FreqLine<V> {
    ctrl: Box<[u8]>, // one control byte per slot; group g = bytes 16g..16g+16, the
    // SSE register SwissTable scans. `ctrl[i] == EMPTY_CTRL (0x80)`
    // ⟺ slot i empty; a full slot's byte is its 7-bit tag.
    slots: Box<[(u64, MaybeUninit<V>)]>, // (frequency, value) fused — one cache line per confirm/hit;
    // `slots[i].0 == EMPTY` ⟺ empty (the in-band Null the
    // Drop/Clone/grow liveness test reads). The miss path never
    // reads it (the tag lives in `ctrl`); a hit reads one line:
    // the payload is a projection of the confirm's own cell
    // (`GroupProbe.fused_payload_from_one_cell`), where split
    // arrays FORCE a second load (`split_payload_needs_second_cell`).
    gshift: u32,  // home group = (p >> gshift) & gmask — precomputed, one shift+and
    gmask: usize, // = (#groups) − 1 = (2^bits / 16) − 1
    bits: u32,
    len: usize,
    side: FreqSide<V>, // reserved-frequency keys (`EMPTY`/`TOMB`) held out of band
    overflow: Option<Overflow<u64, V>>, // keys blocked past the probe cap, one AVL tree by frequency
}

/// The group width — 16 control bytes scanned in one glance, exactly SwissTable's SSE group. The
/// `GroupProbe.winnow_exact` licence is `∀ w`, so the width is a free efficiency knob; 16 is the
/// SSE register, and the byte tag makes the scan one `_mm_cmpeq_epi8` + `_mm_movemask_epi8`.
pub const GROUP: usize = 16;

/// The tag window of `p = f·PHI`: the 7 bits just below the bucket window, `[TAG_SHIFT, 40)` with
/// the current [`FreqMap::COUNT_HOME_GAP`]/[`FreqMap::COUNT_BBITS`]. Disjoint from
/// [`FreqLine::home_group`]'s top `gbits` window and from [`FreqMap::count_home`]. The low 7 bits
/// the tag used to read are all zero on an `i · 2³²` key family (`(i·2³²)·PHI` has its low 32 bits
/// zero), so that tag was blind exactly there; this window is inside the varying high half of every
/// counted key family. Budget: folded into `COUNT_BBITS + COUNT_HOME_GAP + 7 ≤ 64`.
const TAG_SHIFT: u32 = 64 - FreqMap::<u64, i64>::COUNT_HOME_GAP - FreqMap::<u64, i64>::COUNT_BBITS - 7;

/// **The glance** — 16 tag lanes compared to `tag` in one movemask (`SimtSelect.movemask`,
/// `GroupProbe.winnow_exact`): the exact SwissTable move (`_mm_cmpeq_epi8` + `_mm_movemask_epi8`,
/// SSE2 baseline), a portable byte loop off x86_64. ONE implementation, shared by the hashmap
/// ([`FreqLine::match_tag`]) and the codec's row match-finder (`codec::lz`) — the SAME coordinate
/// readout / group-probe at both call sites (this is the point of `CoordinateReadout`: SwissTable
/// lookup and zstd-style match finding are one operation). `base + GROUP ≤ ctrl.len()`.
#[inline]
pub fn group_tag_mask(ctrl: &[u8], base: usize, tag: u8) -> u16 {
    debug_assert!(base + GROUP <= ctrl.len());
    // SAFETY (both paths): base + GROUP ≤ ctrl.len() (caller-guaranteed), so ptr..ptr+16 is in-bounds.
    let ptr = unsafe { ctrl.as_ptr().add(base) };
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        _mm_movemask_epi8(_mm_cmpeq_epi8(
            _mm_loadu_si128(ptr.cast::<__m128i>()),
            _mm_set1_epi8(tag as i8),
        )) as u16
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut m = 0u16;
        for k in 0..GROUP {
            m |= u16::from(unsafe { *ptr.add(k) } == tag) << k;
        }
        m
    }
}
/// The empty-slot control byte: high bit set. A full slot's tag is 7-bit (high bit clear), so
/// empty detection is a single `_mm_movemask_epi8` of the sign bits — no compare (SwissTable's
/// `0xFF`-empty encoding, the reason the miss-deciding op is one instruction). Proven: the
/// sentinel can never collide with a tag (`GroupProbe.sentinel_never_a_tag`) and the sign bit
/// reads emptiness exactly (`flag_bit_reads_emptiness`) — the byte is the Null cell in 8 bits,
/// one Being/Non-Being flag + the coarse flaw, and 7 is the byte's maximum tag width, the
/// minimum false-positive mass (`tag_width_is_maximal`), not a tuning.
const EMPTY_CTRL: u8 = 0x80;
/// `0x80` per byte (the SWAR-fallback sign-bit mask); only used off x86_64.
#[cfg(not(target_arch = "x86_64"))]
const LANE_HI: u64 = 0x8080_8080_8080_8080;

impl<V> FreqLine<V> {
    fn with_bits(bits: u32) -> Self {
        let bits = bits.max(4); // at least one full 16-lane group
        let n = 1usize << bits;
        let gbits = bits - 4; // groups = 2^gbits
        let mut slots = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            slots.push((EMPTY, MaybeUninit::uninit()));
            i += 1;
        }
        FreqLine {
            ctrl: vec![EMPTY_CTRL; n].into_boxed_slice(),
            slots: slots.into_boxed_slice(),
            gshift: if gbits == 0 { 0 } else { 64 - gbits },
            gmask: (n / GROUP) - 1,
            bits,
            len: 0,
            side: FreqSide::empty(),
            overflow: None,
        }
    }

    /// The lane tag of a frequency: 7 bits of `f·PHI` below the home window and the bucket window
    /// (`TAG_SHIFT`), high bit clear so it never aliases the `EMPTY_CTRL` sentinel. 128 states = 7
    /// bits — exactly SwissTable's h2 width, false-positive mass `2⁻⁷`. The window is disjoint from
    /// the address windows the descent consumes (`GroupProbe.tagWindow`; a top window would be
    /// home-determined and blind, `top_window_is_home_determined`, and the old LOW window was
    /// identically 0 on the `i·2³²` family).
    #[inline]
    fn tag(f: u64) -> u8 {
        ((f.wrapping_mul(PHI) >> TAG_SHIFT) & 0x7F) as u8
    }

    #[inline]
    fn set_lane(&mut self, i: usize, s: u8) {
        // SAFETY: i < slots.len() = ctrl.len() = 2^bits.
        unsafe { *self.ctrl.get_unchecked_mut(i) = s };
    }

    /// **The glance** — the 16-lane tag mask of group `g` (`GroupProbe.winnow_exact`): the exact
    /// SwissTable move (`_mm_cmpeq_epi8` + `_mm_movemask_epi8`, SSE2 baseline), a portable byte loop
    /// elsewhere. Split from the empty mask so a *hit* pays only this one movemask — the empty mask
    /// (the stop) is computed lazily by [`match_empty`](Self::match_empty), only when no tag matched
    /// (exactly hashbrown's `match_tag` / `match_empty` split). The group is 16-aligned, so the load
    /// never straddles a cache line.
    #[inline]
    fn match_tag(&self, g: usize, tag: u8) -> u16 {
        // g < ctrl.len()/16 (descend_group masks to that), so base + 16 ≤ ctrl.len().
        group_tag_mask(&self.ctrl, g * GROUP, tag)
    }

    /// The empty-lane mask of group `g` — the stop signal. A full slot's tag has the high bit clear
    /// and `EMPTY_CTRL` has it set, so this is a single sign-bit `_mm_movemask_epi8` (no compare).
    /// Called only after [`match_tag`](Self::match_tag) finds no candidate: the group's cache line is
    /// already resident from that load, so the reload is an L1 hit, and the hit path skips it entirely.
    #[inline]
    fn match_empty(&self, g: usize) -> u16 {
        let base = g * GROUP;
        // SAFETY: g < ctrl.len()/16, so base + 16 ≤ ctrl.len().
        let ptr = unsafe { self.ctrl.as_ptr().add(base) };
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            // SAFETY: SSE2 baseline; ptr..ptr+16 in-bounds.
            unsafe { _mm_movemask_epi8(_mm_loadu_si128(ptr.cast::<__m128i>())) as u16 }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut empties = 0u16;
            for k in 0..GROUP {
                // SAFETY: base + k < base + 16 ≤ ctrl.len().
                let b = unsafe { *ptr.add(k) };
                empties |= (((b & 0x80) != 0) as u16) << k;
            }
            empties
        }
    }

    /// The home group — the common case (`j = 0`), one precomputed shift + and. The PHI-descent's
    /// top window read at group granularity.
    #[inline]
    fn home_group(&self, p: u64) -> usize {
        (p >> self.gshift) as usize & self.gmask
    }

    /// The group descent for the overflow path (`j ≥ 1`): successive `(bits−4)`-bit windows of
    /// `p` (`GroupProbe.topDigit`), then a linear sweep once the word is spent. `gbits` need not
    /// divide 64, so this is NOT a bare `chain_separates_within_radix` instantiation — the
    /// closure is `GroupProbe.group_descent_dichotomy`: distinct folds either separate within the
    /// digit walk, or share the whole consumed prefix and land in a pile of exactly `2^(64 mod
    /// gbits)` folds (`pile_card`), which the tag resolves at the first glance when that residual
    /// fits the tag window (`pile_resolved_by_tag` — the 4M table has `r = 7 = t`, the word fully
    /// consumed). The sweep offset is `j − full + 1`: offset 0 re-glanced the group the last
    /// digit step had just scanned (`zero_offset_duplicates`, the bug the proof caught); distinct
    /// offsets below `2^gbits` never repeat (`sweep_period_no_repeat`). Cold — group overflow only.
    #[inline]
    fn descend_group(&self, p: u64, j: u32) -> usize {
        let gbits = self.bits - 4;
        if gbits == 0 {
            return 0; // a single group
        }
        let used = gbits * (j + 1);
        if used <= 64 {
            (p >> (64 - used)) as usize & self.gmask
        } else {
            let full = 64 / gbits;
            ((p >> (64 - gbits * full)) as usize + (j - full) as usize + 1) & self.gmask
        }
    }

    /// The primary group-probe cap: a lookup, insert, update or grow examines at most this many
    /// groups (the home group, then the fresh PHI-descent windows) before the frequency routes to
    /// the [`Overflow`] tree. One constant, mirroring
    /// [`PROBE_CAP`](crate::map::hashtrinity::PROBE_CAP) in the map face. The cap also removes
    /// [`descend_group`](Self::descend_group)'s unbounded index growth: `j < PROBE_CAP` keeps
    /// `gbits * (j + 1)` and the sweep offset inside `u32`/`usize`.
    #[inline]
    fn probe_limit(&self) -> u32 {
        PROBE_CAP
    }

    /// **The capped group probe.** Per group: one SSE glance winnows all 16 lanes against the tag,
    /// then confirm candidates on the coordinate (the exact flaw); stop at the first group with an
    /// empty lane (sound: first-fit placement + monotone occupancy, `stop_not_before_placement`).
    ///
    /// Returns `(slot, found, blocked, tag)`: a hit (`found`) at `slot`; a vacant insert point
    /// (`!found && !blocked`) at `slot` — the first empty lane in probe order, exactly where an
    /// insert must place (first-fit); or `blocked` — the whole prefix of
    /// [`probe_limit`](Self::probe_limit) groups held no empty lane and no identity, so the
    /// frequency lives (or must be placed) in the [`Overflow`].
    ///
    /// **The overflow invariant (HashDOS cap), the count-face twin of [`Table`]'s.** A frequency is
    /// in `overflow` only if, when it was placed, its `PROBE_CAP` primary groups held no empty lane
    /// and no lane with its identity. It is preserved because a lane only ever changes from empty
    /// to occupied — `FreqLine` never deletes ([`Table`](super::Table) keeps `TOMB` for that), and
    /// [`grow`](Self::grow) re-places every primary and overflow frequency through this same capped
    /// placement — so the absence of an empty lane in the prefix is stable. Hence a lookup that
    /// finds an empty lane within the cap is absent without consulting the overflow; a hit is found
    /// in the prefix; only an exhausted prefix consults the overflow.
    ///
    /// R2: the tag is returned, not recomputed. `insert_with`'s fresh-lane path wrote
    /// `Self::tag(f) = ((f·PHI) >> TAG_SHIFT) & 0x7F`, a second application of the multiply `probe` had already
    /// done; returning the value removes the `lea;lea;and` triple the multiplier constant folds to
    /// from every fresh insert, on the serial `add` path and on `count_par`'s phase 3 alike.
    /// Callers that do not place a lane (`get`, `grow`'s re-place probe) drop it.
    /// Structured with flags and a single exit (no `return` inside the nested scan loop), the form
    /// the Aeneas fragment translates — the flag discipline `count_aggregate`'s extraction depends
    /// on.
    fn probe_tagged(&self, f: u64) -> (usize, bool, bool, u8) {
        let p = f.wrapping_mul(PHI);
        let tag = ((p >> TAG_SHIFT) & 0x7F) as u8; // = tag(f), reusing p
        let limit = self.probe_limit();
        let mut g = self.home_group(p); // j = 0, the common case: one shift + and
        let mut j = 0u32;
        let mut done = false;
        let mut found = false;
        let mut blocked = false;
        let mut slot = 0usize;
        while !done {
            let base = g * GROUP;
            let mut matches = self.match_tag(g, tag); // the hit path: one movemask, no empty scan
            while matches != 0 {
                let i = base + matches.trailing_zeros() as usize;
                // SAFETY: base + lane < ctrl.len() = slots.len().
                if unsafe { self.slots.get_unchecked(i).0 } == f {
                    slot = i;
                    found = true;
                    matches = 0;
                } else {
                    matches &= matches - 1;
                }
            }
            if found {
                done = true;
            } else {
                // No tag candidate here — only NOW read the empty mask (the stop): an empty lane
                // means the key, had it been inserted, would sit at or before this group ⇒ absent,
                // and the first empty lane is the insert point. The group's line is already resident
                // from the tag load, so this reload is an L1 hit; the hit path never runs it.
                let empties = self.match_empty(g);
                if empties != 0 {
                    slot = base + empties.trailing_zeros() as usize;
                    done = true;
                } else {
                    j += 1;
                    if j >= limit {
                        blocked = true;
                        done = true;
                    } else {
                        g = self.descend_group(p, j);
                    }
                }
            }
        }
        (slot, found, blocked, tag)
    }

    /// The number of primary group probes a lookup of frequency `f` makes (at most
    /// [`probe_limit`](Self::probe_limit)); `0` for a reserved frequency (resolved in the side
    /// store). Diagnostic — not part of the stable surface.
    fn probe_examinations(&self, f: u64) -> u32 {
        let mut result = 0u32;
        if is_sentinel(f) {
            return 0;
        }
        let p = f.wrapping_mul(PHI);
        let tag = ((p >> TAG_SHIFT) & 0x7F) as u8;
        let limit = self.probe_limit();
        let mut g = self.home_group(p);
        let mut j = 0u32;
        // One carried flag and one exit after the loop: a `return` inside the inner `while` is a
        // continue-to-outer-loop shape the extraction refuses.
        let mut done = false;
        while !done {
            let base = g * GROUP;
            let mut matches = self.match_tag(g, tag);
            while matches != 0 {
                let i = base + matches.trailing_zeros() as usize;
                // SAFETY: base + lane < ctrl.len() = slots.len().
                if unsafe { self.slots.get_unchecked(i).0 } == f {
                    result = j + 1;
                    done = true;
                }
                matches &= matches - 1;
            }
            if done {
                // The tag walk already resolved this probe.
            } else if self.match_empty(g) != 0 {
                result = j + 1;
                done = true;
            } else {
                j += 1;
                if j >= limit {
                    result = j;
                    done = true;
                } else {
                    g = self.descend_group(p, j);
                }
            }
        }
        result
    }

    #[inline]
    fn get(&self, f: u64) -> Option<&V> {
        self.get_p(f.wrapping_mul(PHI), f)
    }

    /// [`get`](Self::get) with the fold `p = f·PHI` already formed. The count face's bucket-select
    /// read forms `p` once and feeds it to both the bucket window ([`FreqMap::count_home`]) and this
    /// probe; the home group and tag are disjoint windows of that same `p`, so the second multiply
    /// is not needed. `p` MUST equal `f.wrapping_mul(PHI)` — otherwise the home group and tag are
    /// read from the wrong windows and a present key is reported absent. The reserved-frequency
    /// check is here so a sentinel frequency resolves in the side store before any group probe. O(1).
    #[inline(always)]
    fn get_p(&self, p: u64, f: u64) -> Option<&V> {
        if is_sentinel(f) {
            return self.side.get(f);
        }
        // Home-group fast path (j = 0): the common case for a present key. One tag glance;
        // a confirmed lane returns WITHOUT the general capped probe's call, tuple and
        // descent setup. A home group with an empty lane proves absence (first-fit
        // placement + monotone occupancy, `stop_not_before_placement`) — no lane can hold
        // the key beyond an empty one — and a blocked key's prefix is still full, so the
        // overflow cannot hold it either. Only a full home group with no identity falls
        // through to `probe_tagged`, which owns the descent and overflow.
        let tag = ((p >> TAG_SHIFT) & 0x7F) as u8;
        let g = self.home_group(p);
        let base = g * GROUP;
        let mut matches = self.match_tag(g, tag);
        while matches != 0 {
            let i = base + matches.trailing_zeros() as usize;
            // SAFETY: base + lane < ctrl.len() = slots.len().
            if unsafe { self.slots.get_unchecked(i).0 } == f {
                // SAFETY: found ⇒ slot i is live ⇒ its value is initialized.
                return Some(unsafe { self.slots.get_unchecked(i).1.assume_init_ref() });
            }
            matches &= matches - 1;
        }
        if self.match_empty(g) != 0 {
            return None;
        }
        let (i, found, blocked, _tag) = self.probe_tagged(f);
        if found {
            Some(unsafe { self.slots.get_unchecked(i).1.assume_init_ref() })
        } else if blocked {
            // A `match`, not `Option::and_then(closure)`: a closure argument is an arrow type,
            // which the extraction's first-order boundary does not carry.
            match &self.overflow {
                Some(o) => o.get(f, &f),
                None => None,
            }
        } else {
            None
        }
    }

    /// Prefetch the home group's control line + slot line for fold `p` — the pipeline warm-up.
    #[inline]
    fn prefetch_group(&self, p: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let base = self.home_group(p) * GROUP;
            _mm_prefetch(self.ctrl.as_ptr().add(base).cast::<i8>(), _MM_HINT_T0);
            _mm_prefetch(self.slots.as_ptr().add(base).cast::<i8>(), _MM_HINT_T0);
        }
        let _ = p;
    }

    /// Iterate live `(coord, &value)` entries — a live primary slot is `slots[i].0 != EMPTY`;
    /// the side store and the overflow contribute their own entries. Arbitrary order.
    fn iter_live(&self) -> impl Iterator<Item = (u64, &V)> {
        let main = self.slots.iter().filter_map(|(c, v)| {
            if *c == EMPTY {
                None
            } else {
                // SAFETY: coord ≠ EMPTY ⇒ the value was initialized on insert.
                Some((*c, unsafe { v.assume_init_ref() }))
            }
        });
        let side = self.side.iter_live();
        let overflow = self
            .overflow
            .iter()
            .flat_map(|o| o.iter().map(|(c, _k, v)| (c, v)));
        main.chain(side).chain(overflow)
    }

    /// Reduce the whole table in parallel over live values on the group-probe engine
    /// (schedule-free, `Web.preduceAt_schedule_free`).
    fn encircle<M, R, C>(&self, id: M, residual: R, combine: C) -> M
    where
        M: Send + Sync + Clone,
        R: Fn(&V) -> M + Sync,
        C: Fn(M, M) -> M + Sync,
        V: Sync,
    {
        let fold_range = |start: usize, end: usize| -> M {
            let mut acc = id.clone();
            for i in start..end {
                if self.slots[i].0 != EMPTY {
                    // SAFETY: live coord ⇒ value initialized.
                    acc = combine(
                        acc,
                        residual(unsafe { self.slots.get_unchecked(i).1.assume_init_ref() }),
                    );
                }
            }
            acc
        };
        let n = self.slots.len();
        let pool = par::Pool::global();
        let main = if pool.width() <= 1 || n < 1 << 14 {
            fold_range(0, n)
        } else {
            let tasks = (pool.width() * 4).min(n);
            let chunk = n.div_ceil(tasks);
            let mut partials: Vec<M> = (0..tasks).map(|_| id.clone()).collect();
            let pptr = par::SendPtr(partials.as_mut_ptr());
            pool.dispatch(tasks, |t| {
                let start = t * chunk;
                if start >= n {
                    return;
                }
                let end = (start + chunk).min(n);
                // SAFETY: task t writes only partials[t]; disjoint by t; barrier orders before merge.
                unsafe { *pptr.add(t) = fold_range(start, end) };
            });
            partials.into_iter().fold(id, &combine)
        };
        let mut acc = main;
        // Side entries (reserved frequencies) and overflow entries are O(1) on ordinary maps.
        for (i, live) in self.side.live.iter().enumerate() {
            if *live {
                // SAFETY: `live[i]` ⇒ value initialized.
                acc = combine(acc, residual(unsafe {
                    self.side.vals.get_unchecked(i).assume_init_ref()
                }));
            }
        }
        if let Some(o) = &self.overflow {
            for (_c, _k, v) in o.iter() {
                acc = combine(acc, residual(v));
            }
        }
        acc
    }

    fn insert_with(&mut self, f: u64, new: V, merge: impl FnOnce(V, V) -> V) {
        // A reserved frequency cannot live in a primary lane (`slots[i].0 == EMPTY` is the
        // occupancy flag); its count is held in the side store, exactly as the map face holds a
        // reserved-coordinate key in its `SideStore`.
        if is_sentinel(f) {
            // Inline the side merge rather than passing `merge` across a call boundary: an
            // `impl FnOnce` argument is an arrow type the Aeneas fragment refuses, and this body is
            // reached from the extracted `count_aggregate`.
            if let Some(i) = self.side.find(f) {
                // SAFETY: `find` only returns an entry whose `live` marker is set.
                let old = unsafe { self.side.vals.get_unchecked_mut(i).assume_init_read() };
                self.side.live[i] = false;
                let merged = merge(new, old);
                // SAFETY: the value was moved out; re-writing restores the invariant.
                unsafe { *self.side.vals.get_unchecked_mut(i) = MaybeUninit::new(merged) };
                self.side.live[i] = true;
            } else {
                self.side.coords.push(f);
                self.side.vals.push(MaybeUninit::new(new));
                self.side.live.push(true);
            }
            return;
        }
        if (self.len + 1) * 8 >= self.slots.len() * 7 {
            self.grow();
        }
        let (i, found, blocked, tag) = self.probe_tagged(f); // R2: the tag `probe` already computed
        if found {
            // SAFETY: found ⇒ slot i is live ⇒ its value is initialized.
            let slot = unsafe { self.slots.get_unchecked_mut(i) };
            let old = unsafe { slot.1.assume_init_read() };
            if std::mem::needs_drop::<V>() {
                // panic-safe: blank the COORD only, so a merge panic leaves a TOMBSTONE — Drop/
                // Clone/grow skip it (coord == EMPTY, no double-free) but the control byte keeps
                // its tag, so probes for OTHER keys do not stop here. Blanking the lane too would
                // be a deletion mid-chain, exactly `GroupProbe.deletion_breaks_the_stop`: a key
                // placed past this group would false-miss after the panic. DELETED ≠ EMPTY. Only
                // when `V` has a destructor — a non-drop `V` (the counter/accumulator case)
                // cannot double-free, so the dance is skipped and the merge writes in place.
                slot.0 = EMPTY;
                let merged = merge(new, old);
                let slot = unsafe { self.slots.get_unchecked_mut(i) };
                slot.1 = MaybeUninit::new(merged);
                slot.0 = f;
            } else {
                slot.1 = MaybeUninit::new(merge(new, old));
            }
        } else if blocked {
            // The whole capped prefix was occupied: the frequency lives here (merge) or is new
            // here (insert), keyed by the frequency itself. The merge runs in THIS body — `merge`
            // is not passed into the overflow, where the arrow type would be refused.
            if self.overflow.is_none() {
                self.overflow = Some(Overflow::new());
            }
            if let Some(ov) = self.overflow.as_mut() {
                match ov.find_entry(f, &f) {
                    Some((node, entry)) => {
                        let old = ov.take_value(node, entry);
                        let merged = merge(new, old);
                        ov.write_back(node, entry, merged);
                    }
                    None => {
                        ov.insert_new_at(f, f, new);
                    }
                }
            }
        } else {
            // SAFETY: i < slots.len() (probe returns an in-range empty slot).
            let slot = unsafe { self.slots.get_unchecked_mut(i) };
            slot.0 = f;
            slot.1 = MaybeUninit::new(new);
            self.set_lane(i, tag); // R2: not `Self::tag(f)`, which refolds f·PHI
            self.len += 1;
        }
    }

    /// Re-place one distinct frequency through the capped placement every insert uses: a vacant
    /// lane within the cap, or the overflow. Used only by [`grow`](Self::grow), where frequencies
    /// are distinct, so a `found` (identity already present) verdict is impossible; it is handled
    /// defensively.
    fn place_capped(&mut self, f: u64, v: V) {
        if is_sentinel(f) {
            // The side store is carried over separately; a primary lane never holds a sentinel.
            if let Some(i) = self.side.find(f) {
                let slot = unsafe { self.side.vals.get_unchecked_mut(i) };
                let old = unsafe { slot.assume_init_read() };
                drop(old);
                *slot = MaybeUninit::new(v);
            } else {
                self.side.coords.push(f);
                self.side.vals.push(MaybeUninit::new(v));
                self.side.live.push(true);
            }
            return;
        }
        let (i, found, blocked, tag) = self.probe_tagged(f);
        if blocked {
            if self.overflow.is_none() {
                self.overflow = Some(Overflow::new());
            }
            if let Some(ov) = self.overflow.as_mut() {
                ov.insert_new_at(f, f, v);
            }
        } else if found {
            // Defensive: the rebuild carries distinct frequencies. Overwrite and drop the old.
            // SAFETY: a live lane ⇒ value initialized; `assume_init_read` moves it out.
            let slot = unsafe { self.slots.get_unchecked_mut(i) };
            let old = unsafe { slot.1.assume_init_read() };
            drop(old);
            slot.1 = MaybeUninit::new(v);
        } else {
            let slot = unsafe { self.slots.get_unchecked_mut(i) };
            slot.0 = f;
            slot.1 = MaybeUninit::new(v);
            self.set_lane(i, tag);
            self.len += 1;
        }
    }

    #[cold]
    fn grow(&mut self) {
        let mut bigger = Self::with_bits(self.bits + 1);
        let mut i = 0;
        let n = self.slots.len();
        while i < n {
            let c = self.slots[i].0;
            if c != EMPTY {
                // Move the value out and blank the source so the old line's Drop skips it; re-place
                // through the SAME capped placement (a grown table may hold it, or route it on).
                let v = unsafe { self.slots.get_unchecked_mut(i).1.assume_init_read() };
                self.slots[i].0 = EMPTY;
                bigger.place_capped(c, v);
            }
            i += 1;
        }
        // Carry the side store over; the rebuilt line is fresh (empty side store).
        bigger.side = std::mem::replace(&mut self.side, FreqSide::empty());
        // Re-place every overflow frequency through the capped placement too: a grown line may now
        // hold it in the primary array, and the invariant is re-established for the new generation.
        if let Some(ov) = self.overflow.take() {
            for (c, _k, v) in ov.into_entries() {
                bigger.place_capped(c, v);
            }
        }
        *self = bigger;
    }

    /// Test-only: the worst probe length (in group steps) over all present primary keys — the flood
    /// bound. A flood clusters keys at one home group; the capped probe holds every present primary
    /// key within [`PROBE_CAP`] groups, and blocked keys live in the overflow.
    #[cfg(test)]
    fn max_probe_groups(&self) -> u32 {
        let mut worst = 0;
        for i in 0..self.slots.len() {
            let f = self.slots[i].0;
            if f == EMPTY {
                continue;
            }
            worst = worst.max(self.probe_examinations(f));
        }
        worst
    }
}

impl<V> Drop for FreqLine<V> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<V>() {
            for (c, v) in self.slots.iter_mut() {
                if *c != EMPTY {
                    // SAFETY: coord != EMPTY ⇒ value initialized.
                    unsafe { v.assume_init_drop() };
                }
            }
            // Side entries: reserved frequencies, held in `side`; the overflow tree's `Vec` drops
            // its own `Option<V>` values.
            self.side.drop_live();
        }
    }
}

impl<V: Clone> Clone for FreqLine<V> {
    fn clone(&self) -> Self {
        let slots = self
            .slots
            .iter()
            .map(|(c, v)| {
                if *c == EMPTY {
                    (EMPTY, MaybeUninit::uninit())
                } else {
                    (*c, MaybeUninit::new(unsafe { v.assume_init_ref() }.clone()))
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        FreqLine {
            ctrl: self.ctrl.clone(),
            slots,
            gshift: self.gshift,
            gmask: self.gmask,
            bits: self.bits,
            len: self.len,
            side: self.side.clone(),
            overflow: self.overflow.clone(),
        }
    }
}

/// **The frequency map — the NB (Non-Being) face of the storage trinity.**
///
/// The opaque golden router. An opaque key routes by its FOLD (its frequency, a `u64`); the map
/// keeps no key at all — the frequency IS the address (`key_routes_iff_fold_distinguishes`). The
/// 16-byte pointer and the key bytes are gone; only the 8-byte frequency remains, and its storage is
/// **independent of key length** (17 B/slot — 8-byte coord + 1-byte tag + value — whether the key
/// is 8 bytes or a kilobyte).
///
/// The three storage faces, the trinity `x = B/U`:
/// * [`CoordMap`] / [`Regen`] — **B** (Being): coordinate keys, address round-trips, stores nothing.
/// * [`Table`] / [`Stored`] — **B/U**: opaque keys, keeps the key to confirm exactly.
/// * `FreqMap` — **NB** (Non-Being): opaque keys, the fold collapses the key, only the frequency
///   survives; stores nothing, and rejects misses through the control — the Null cell's coarse
///   first reading (`golden_control_is_the_first_flaw`), whose cache-resident density is the
///   derived description — while the PHI-descent underneath is untouched (the filter adds no probe
///   surface; the descent's own worst case is the measured one, not a proved one).
///
/// **The one tradeoff.** The fold is lossy, so two distinct keys with the same 64-bit fold collide,
/// and a non-member query whose fold equals a member's is a false positive — both `2^-64` (zero over
/// millions of probes). This is the map for dedup, joins, caches, counters, indexes on high-entropy
/// keys, where `2^-64` is "never". A map that must reject arbitrary non-members exactly, or return
/// the key, wants [`Table`]. `FreqMap` carries frequencies, not keys.
#[derive(Clone, Debug, Default)]
pub struct FreqMap<K, V> {
    inner: FreqLine<V>,
    _k: std::marker::PhantomData<K>,
}

impl<V> std::fmt::Debug for FreqLine<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FreqLine")
            .field("bits", &self.bits)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}
impl<V> Default for FreqLine<V> {
    fn default() -> Self {
        Self::with_bits(3)
    }
}

impl<K: Coord, V> FreqMap<K, V> {
    /// The key's frequency (its fold): the raw coordinate, with no clamp. A key whose coordinate is
    /// a reserved marker (`EMPTY`/`TOMB`) is held out of band in the [`FreqSide`] store — the
    /// count-face twin of the map face's [`SideStore`](super::SideStore) — so distinct top-value
    /// keys no longer fold onto one count and every `u64` key has an exact count. O(1).
    #[inline]
    fn freq(k: &K) -> u64 {
        k.coord()
    }

    /// A frequency map sized to hold `cap` keys below the 7/8 grow threshold: `≈ 8/7·cap` slots
    /// (`next_pow2(8·cap/7)`; load ≤ 7/8). `8/7` is the smallest factor that keeps
    /// `7/8·slots ≥ cap`, so a `with_capacity(cap)` line does not grow while it fills to `cap`;
    /// `3/2` rounded the same request up a whole power of two more often (e.g. `with_capacity(1e5)`
    /// opened 2^18 slots under `3/2`, 2^17 under `8/7`). The byte-tag group probe
    /// (`GroupProbe.lean`) rejects a miss with one control word regardless of load — probe work is
    /// occupancy over the group width, `w`-fold amplified against load
    /// (`GroupProbe.full_groups_bounded_by_occupancy`, the load inversion) — so a smaller footprint
    /// is strictly better: fewer cold lines on every out-of-cache access. The load knob is
    /// `LoadFactorFrontier`'s, not a security flag — the probe is always immune.
    /// At `cap = 1e5`, 2^18 slots
    /// (4 MB, past a 2 MB L2) took 0.855 L2 misses per hit read, 2^17 takes 0.221.
    pub fn with_capacity(cap: usize) -> Self {
        FreqMap {
            inner: FreqLine::with_bits(bits_for_slots(cap.saturating_mul(8) / 7)),
            _k: std::marker::PhantomData,
        }
    }

    /// Alias of [`with_capacity`](Self::with_capacity) — kept for source compatibility. The group probe's optimum is the
    /// same dense sizing for read- and stream-heavy work alike; the former sparse/dense split was
    /// calibrated for the retired direct-slot probe.
    pub fn stream_with_capacity(cap: usize) -> Self {
        Self::with_capacity(cap)
    }

    /// Look up a key by its frequency. `2^-64` false-positive on non-members (see the type docs).
    #[inline]
    pub fn get(&self, k: &K) -> Option<&V> {
        self.inner.get(Self::freq(k))
    }

    /// Insert `k → new`, resolving a frequency clash with `f` (`|new, old| …`).
    #[inline]
    pub fn insert_with(&mut self, k: &K, new: V, f: impl FnOnce(V, V) -> V) {
        self.inner.insert_with(Self::freq(k), new, f);
    }

    /// Insert `k → v`, last write wins.
    #[inline]
    pub fn insert(&mut self, k: &K, v: V) {
        self.inner.insert_with(Self::freq(k), v, |new, _| new);
    }

    /// The number of live frequencies: primary lanes, reserved-frequency side entries and overflow
    /// entries.
    pub fn len(&self) -> usize {
        // A `match`, not `Option::map_or(Overflow::len)`: a `fn` item argument is an arrow type,
        // which the extraction's first-order boundary does not carry.
        let overflow_len = match &self.inner.overflow {
            Some(o) => o.len(),
            None => 0,
        };
        self.inner.len + self.inner.side.len() + overflow_len
    }
    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The slot capacity (the frequency line length).
    pub fn capacity(&self) -> usize {
        self.inner.slots.len()
    }

    /// The number of primary group probes a lookup of `k` makes (at most
    /// [`PROBE_CAP`](crate::map::hashtrinity::PROBE_CAP)); `0` for a key whose frequency is a reserved
    /// marker, which is resolved in the side store. Diagnostic — the `tests/hashdos_count.rs`
    /// HashDOS bound reads it.
    #[doc(hidden)]
    pub fn primary_examinations(&self, k: &K) -> u32 {
        self.inner.probe_examinations(Self::freq(k))
    }

    /// The worst primary group-probe over the primary lane entries. Diagnostic.
    #[doc(hidden)]
    pub fn longest_probe(&self) -> usize {
        let mut worst = 0u32;
        for i in 0..self.inner.slots.len() {
            let f = self.inner.slots[i].0;
            if f != EMPTY {
                worst = worst.max(self.inner.probe_examinations(f));
            }
        }
        worst as usize
    }

    /// The number of frequencies held in the overflow tree (blocked past the probe cap). Diagnostic.
    #[doc(hidden)]
    pub fn overflow_len(&self) -> usize {
        match &self.inner.overflow {
            Some(o) => o.len(),
            None => 0,
        }
    }

    /// The overflow tree's height (`0` when no frequency was blocked). Diagnostic.
    #[doc(hidden)]
    pub fn overflow_height(&self) -> u32 {
        match &self.inner.overflow {
            Some(o) => o.height(),
            None => 0,
        }
    }

    /// Iterate the live values (arbitrary order). The key is not recoverable from the NB fold —
    /// `values` is the reduce input; use [`CoordMap`] if you need the keys back.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.inner.iter_live().map(|(_, v)| v)
    }

    /// **The stream read, pipelined** ([`Table::get_each`] on the group probe): probe each key,
    /// prefetching the home group `P` folds ahead so the miss latency is hidden
    /// (`MemoryLevelParallelism.optimal_depth_hides`). Visits `(key, Option<&value>)` in order.
    pub fn get_each<'a, F>(&'a self, keys: &'a [K], mut visit: F)
    where
        F: FnMut(&'a K, Option<&'a V>),
    {
        const P: usize = 16;
        let m = &self.inner;
        let n = keys.len();
        if n <= P || m.slots.len() <= (1 << 15) {
            for k in keys {
                visit(k, m.get(Self::freq(k)));
            }
            return;
        }
        for k in &keys[..P] {
            m.prefetch_group(Self::freq(k).wrapping_mul(PHI));
        }
        for i in 0..n {
            visit(&keys[i], m.get(Self::freq(&keys[i])));
            if i + P < n {
                m.prefetch_group(Self::freq(&keys[i + P]).wrapping_mul(PHI));
            }
        }
    }

    /// Reduce the whole table in parallel over live values on the group probe.
    pub fn encircle<M, R, C>(&self, id: M, residual: R, combine: C) -> M
    where
        M: Send + Sync + Clone,
        R: Fn(&V) -> M + Sync,
        C: Fn(M, M) -> M + Sync,
        V: Sync,
    {
        self.inner.encircle(id, residual, combine)
    }

    /// **The threaded stream read** on the group probe: split the keys one slice
    /// per core, each pipelines internally into a disjoint output slice. A read can't affect another,
    /// so the split is safe and `out[i] = get(keys[i])`, order preserved.
    pub fn get_all<'a>(&'a self, keys: &'a [K]) -> Vec<Option<&'a V>>
    where
        K: Sync,
        V: Sync,
    {
        let n = keys.len();
        let mut out: Vec<Option<&'a V>> = vec![None; n];
        let pool = par::Pool::global();
        if pool.width() <= 1 || n < 1 << 15 {
            let mut w = 0;
            self.get_each(keys, |_, v| {
                out[w] = v;
                w += 1;
            });
            return out;
        }
        let tasks = pool.width();
        let chunk = n.div_ceil(tasks);
        let optr = par::SendPtr(out.as_mut_ptr());
        pool.dispatch(tasks, |t| {
            let start = t * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            let mut w = start;
            self.get_each(&keys[start..end], |_, v| {
                // SAFETY: worker t writes only its disjoint [start,end); barrier before the read.
                unsafe { *optr.add(w) = v };
                w += 1;
            });
        });
        out
    }

    /// **The fused parallel key-stream reduce** on the group probe: each core
    /// pipelines its slice and folds locally, partials combined once — no `N`-alloc.
    pub fn get_reduce<'a, M, Rf, C>(&'a self, keys: &'a [K], id: M, map: Rf, combine: C) -> M
    where
        K: Sync,
        V: Sync,
        M: Send + Sync + Clone,
        Rf: Fn(Option<&'a V>) -> M + Sync,
        C: Fn(M, M) -> M + Sync,
    {
        let reduce_range = |start: usize, end: usize| -> M {
            let mut acc = id.clone();
            self.get_each(&keys[start..end], |_, v| {
                acc = combine(core::mem::replace(&mut acc, id.clone()), map(v));
            });
            acc
        };
        let n = keys.len();
        let pool = par::Pool::global();
        if pool.width() <= 1 || n < 1 << 15 {
            return reduce_range(0, n);
        }
        let tasks = (pool.width() * 4).min(n);
        let chunk = n.div_ceil(tasks);
        let mut partials: Vec<M> = (0..tasks).map(|_| id.clone()).collect();
        let pptr = par::SendPtr(partials.as_mut_ptr());
        pool.dispatch(tasks, |t| {
            let start = t * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            // SAFETY: task t writes only partials[t]; disjoint by t; barrier before merge.
            unsafe { *pptr.add(t) = reduce_range(start, end) };
        });
        partials.into_iter().fold(id, &combine)
    }
}

impl<K: Coord> FreqMap<K, i64> {
    /// Count: add `v` to the running total at `k`'s frequency (the B/U accumulator face).
    #[inline]
    pub fn add(&mut self, k: &K, v: i64) {
        self.inner
            .insert_with(Self::freq(k), v, |new, old| new + old);
    }
}

#[cfg(test)]
mod freqline_flood {
    //! Phase-4 gate: the ·PHI-collapse reads home/tag/descent from the fold's own bits, so an
    //! adversary who controls the fold's top window controls the home group directly. Flood immunity
    //! must then rest ENTIRELY on the descent separating distinct folds within the radix depth (not on
    //! any home-spread). These build the sharpest floods and assert the probe stays bounded.
    use super::{FreqMap, PHI};

    fn assert_bounded(keys: &[u64], label: &str) {
        let mut m: FreqMap<u64, i64> = FreqMap::with_capacity(keys.len());
        for &k in keys {
            m.insert(&k, 1);
        }
        assert_eq!(m.len(), keys.len(), "{label}: all distinct keys stored");
        for &k in keys {
            assert!(m.get(&k).is_some(), "{label}: present key resolves");
        }
        let worst = m.inner.max_probe_groups();
        assert!(
            worst < 128,
            "{label}: flood must stay radix-bounded, got worst {worst} groups (n = {})",
            keys.len()
        );
    }

    /// Worst case for a top-window-first descent read from the fold's own bits: small sequential
    /// integers. `freq(k) = k`, so every key's home window (top bits) is 0 AND the descent's first
    /// several windows are 0 — they differ only in the low bits. If the descent still separates them
    /// within the radix depth, the collapse is flood-safe for the hardest input.
    #[test]
    fn small_sequential_ints_stay_bounded() {
        let keys: Vec<u64> = (0..12_000u64).collect();
        assert_bounded(&keys, "small-sequential");
    }

    /// The classic multiples attack — all keys share the low bits; here also the top bits (they are
    /// `i << 40`, so for i < 2^12 the top ~12 bits vary but the middle is zero). Distinct folds,
    /// so the descent separates.
    #[test]
    fn multiples_attack_stays_bounded() {
        let keys: Vec<u64> = (1..=12_000u64).map(|i| i << 20).collect();
        assert_bounded(&keys, "multiples");
    }

    /// Keys whose top window collides directly (the collapse's home is the fold's top window): pin the
    /// top 16 bits to 0, vary the rest. Every key homes to group 0.
    #[test]
    fn pinned_home_window_stays_bounded() {
        // top 16 bits zero; the low 48 bits are a PHI-scrambled counter (distinct, dense).
        let keys: Vec<u64> = (0..12_000u64).map(|i| i.wrapping_mul(PHI) >> 16).collect();
        assert_bounded(&keys, "pinned-home");
    }
}

#[cfg(test)]
mod count_par_regen {
    use super::{FreqMap, Regen, Table};
    use std::collections::HashMap;

    /// The Regen face's `count_par` (shared phases 1-2) agrees with a `HashMap` oracle and with the
    /// `FreqMap` face on a key set above the crossover.
    #[test]
    fn regen_count_par_matches_oracle() {
        let mut x = 0x1234_5678_9abc_def0u64;
        let keys: Vec<u64> = (0..200_000)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (x >> 33) % 50_000
            })
            .collect();
        let mut oracle: HashMap<u64, i64> = HashMap::new();
        for &k in &keys {
            *oracle.entry(k).or_insert(0) += 1;
        }
        let maps = Table::<u64, i64, Regen<u64>>::count_par(&keys);
        let fmaps = FreqMap::<u64, i64>::count_par(&keys);
        for (&k, &c) in &oracle {
            let r: i64 = maps.iter().filter_map(|m| m.get(&k).copied()).sum();
            let f: i64 = fmaps.iter().filter_map(|m| m.get(&k).copied()).sum();
            assert_eq!((r, f), (c, c), "key {k}");
        }
    }
}
