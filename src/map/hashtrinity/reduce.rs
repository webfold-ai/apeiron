//! Parallel reads over a [`Table`]: a stream of keys resolved across every core
//! ([`get_all`](Table::get_all), [`get_reduce`](Table::get_reduce)) and the whole table folded
//! across every core ([`encircle`](Table::encircle), [`encircle_floor`](Table::encircle_floor)).
//!
//! These are the methods of the `hashtrinity` crate's `Table` that this port did not carry. The
//! main slot array is split across the pool as before; the side store (keys whose coordinate is a
//! reserved marker) and the overflow tree (blocked keys) are small by construction and are read
//! serially, so no entry is skipped.

use super::{is_live, par, Coord, CoordInv, KeyReadout, KeyStore, Regen, Table};

/// Constructors for a [`CoordMap`](super::CoordMap) — coordinate keys, no key array.
impl<K: CoordInv + Eq, V> Table<K, V, Regen<K>> {
    /// An empty coordinate map. The key regenerates from the address — nothing but the coordinate
    /// stored.
    pub fn coordinate() -> Self {
        Self::with_bits(3)
    }
}

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    /// Resolve a stream of keys across every core — the threaded version of
    /// [`get_each`](Self::get_each). Split the keys into one slice per core, resolve each on its own
    /// thread into a disjoint output slice; each lane still pipelines internally, so this stacks core
    /// parallelism on top of per-core memory-level parallelism. A read of one key can't affect
    /// another, so the split is safe, and `out[i] = get(keys[i])` with order preserved.
    ///
    /// Small/serial inputs just pipeline on this thread (spawn isn't worth it), so it never loses.
    pub fn get_all<'a>(&'a self, keys: &'a [K]) -> Vec<Option<&'a V>>
    where
        K: Sync,
        V: Sync,
        KS: Sync,
    {
        let n = keys.len();
        let mut out: Vec<Option<&'a V>> = vec![None; n];
        let pool = par::Pool::global();
        // Below ~a page of keys per executor the dispatch/partition cost dominates — pipeline serially.
        if pool.width() <= 1 || n < 1 << 15 {
            let mut w = out.iter_mut();
            self.get_each(keys, |_, v| {
                if let Some(slot) = w.next() {
                    *slot = v;
                }
            });
            return out;
        }
        // Static equal partition, self-balanced by the pool's task counter. Each task runs the
        // *pipelined* `get_each` over its window, so core parallelism stacks on per-core
        // memory-level parallelism.
        let tasks = (pool.width() * 4).min(n);
        let chunk = n.div_ceil(tasks);
        // Disjoint output windows keyed by task index: task t writes out[t*chunk .. (t+1)*chunk].
        let out_ptr = par::SendPtr(out.as_mut_ptr());
        pool.dispatch(tasks, |t| {
            let start = t * chunk;
            if start >= n {
                return;
            }
            let end = (start + chunk).min(n);
            // SAFETY: task windows are disjoint (distinct `t` ⇒ distinct [start,end)), so no two tasks
            // touch the same `out` slot; the dispatch barrier joins before `out` is read/returned.
            let oc = unsafe { std::slice::from_raw_parts_mut(out_ptr.add(start), end - start) };
            let mut w = oc.iter_mut();
            self.get_each(&keys[start..end], |_, v| {
                if let Some(slot) = w.next() {
                    *slot = v;
                }
            });
        });
        out
    }

    /// Reduce a **stream of keys** in parallel: look each key up, `map` its result (`None` if absent),
    /// and `combine` across all cores — without materializing the per-key results. This is the fused
    /// face of [`get_all`](Self::get_all): where `get_all` writes an `N`-vector the caller then folds,
    /// `get_reduce` folds inline. Each worker runs the pipelined [`get_each`](Self::get_each) over its
    /// key-chunk into a local accumulator, and the partials combine. `combine` must be a commutative
    /// monoid with identity `id`, so the split is schedule-free — any task→core assignment and merge
    /// order give the same answer.
    pub fn get_reduce<'a, M, Rf, C>(&'a self, keys: &'a [K], id: M, map: Rf, combine: C) -> M
    where
        M: Send + Sync + Clone,
        Rf: Fn(Option<&'a V>) -> M + Sync,
        C: Fn(M, M) -> M + Sync,
        K: Sync,
        V: Sync,
        KS: Sync,
    {
        let reduce_range = |start: usize, end: usize| -> M {
            let mut acc = id.clone();
            // pipelined lookups (get_each) folded into the local accumulator, in stream order.
            self.get_each(&keys[start..end], |_, v| {
                acc = combine(core::mem::replace(&mut acc, id.clone()), map(v));
            });
            acc
        };
        let n = keys.len();
        let pool = par::Pool::global();
        // Below a page of keys per executor the dispatch cost dominates — reduce serially (still
        // pipelined by get_each).
        if pool.width() <= 1 || n < 1 << 15 {
            return reduce_range(0, n);
        }
        // One partial per task; more tasks than executors so uneven work self-balances.
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
            // SAFETY: task `t` writes only `partials[t]`; distinct tasks ⇒ disjoint slots, and the
            // dispatch barrier orders the write before the merge below.
            unsafe { *pptr.add(t) = reduce_range(start, end) };
        });
        partials.into_iter().fold(id, &combine)
    }

    /// Reduce the whole table in parallel: apply `residual` to every value and `combine` the results
    /// across all cores. Where [`get`](Self::get) reads one key and [`get_each`](Self::get_each) a
    /// stream, this reads the entire table — the aggregate a point-lookup map makes you write a serial
    /// loop for. `combine` must be a commutative monoid with identity `id`, so the split across cores
    /// gives the same answer regardless of how the work is partitioned.
    ///
    /// Every entry is folded: the main slots across the pool, then the side store and the overflow
    /// tree serially.
    pub fn encircle<M, R, C>(&self, id: M, residual: R, combine: C) -> M
    where
        M: Send + Sync + Clone,
        R: Fn(&V) -> M + Sync,
        C: Fn(M, M) -> M + Sync,
        K: Sync,
        V: Sync,
        KS: Sync,
    {
        let fold_range = |start: usize, end: usize| -> M {
            let mut acc = id.clone();
            for i in start..end {
                if is_live(self.coords[i]) {
                    // SAFETY: live coord ⇒ value initialized.
                    acc = combine(
                        acc,
                        residual(unsafe { self.vals.get_unchecked(i).assume_init_ref() }),
                    );
                }
            }
            acc
        };
        let n = self.coords.len();
        let pool = par::Pool::global();
        let main = if pool.width() <= 1 || n < 1 << 14 {
            fold_range(0, n) // no core to spread onto — one pass
        } else {
            // One partial per task; more tasks than executors so uneven live-density self-balances.
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
                // SAFETY: task `t` writes only `partials[t]`; distinct tasks ⇒ disjoint slots, and
                // the dispatch barrier orders the write before the merge below.
                unsafe { *pptr.add(t) = fold_range(start, end) };
            });
            partials.into_iter().fold(id.clone(), &combine)
        };
        let mut acc = main;
        for i in 0..self.side.coords.len() {
            // SAFETY: every side entry is occupied ⇒ its value is initialized.
            let v = unsafe { self.side.vals.get_unchecked(i).assume_init_ref() };
            acc = combine(acc, residual(v));
        }
        if let Some(o) = self.overflow.as_ref() {
            for (_coord, _k, v) in o.iter() {
                acc = combine(acc, residual(v));
            }
        }
        acc
    }

    /// Like [`encircle`](Self::encircle), but return *where* the minimum is: the entry with the
    /// smallest `residual`, and its key. Ties break toward the lower slot (main slots first, then
    /// the side store, then the overflow tree), so the result is deterministic. `None` for an empty
    /// table.
    pub fn encircle_floor<M, R>(&self, residual: R) -> Option<(KS::Key<'_>, &V)>
    where
        M: Ord + Copy + Send,
        R: Fn(&V) -> M + Sync,
        K: Sync,
        V: Sync,
        KS: Sync + KeyReadout<K>,
    {
        // argmin over one chunk: (residual, absolute slot), strict `<` so the lower slot wins ties.
        let range_argmin = |start: usize, end: usize| -> Option<(M, usize)> {
            let mut best: Option<(M, usize)> = None;
            for i in start..end {
                if is_live(self.coords[i]) {
                    // SAFETY: live coord ⇒ value initialized.
                    let m = residual(unsafe { self.vals.get_unchecked(i).assume_init_ref() });
                    if best.map_or(true, |(bm, _)| m < bm) {
                        best = Some((m, i));
                    }
                }
            }
            best
        };
        let n = self.coords.len();
        let pool = par::Pool::global();
        let main = if pool.width() <= 1 || n < 1 << 14 {
            range_argmin(0, n)
        } else {
            let tasks = (pool.width() * 4).min(n);
            let chunk = n.div_ceil(tasks);
            let mut partials: Vec<Option<(M, usize)>> = vec![None; tasks];
            let pptr = par::SendPtr(partials.as_mut_ptr());
            pool.dispatch(tasks, |t| {
                let start = t * chunk;
                if start >= n {
                    return;
                }
                let end = (start + chunk).min(n);
                // SAFETY: task `t` writes only `partials[t]`; disjoint across tasks, barrier-ordered.
                unsafe { *pptr.add(t) = range_argmin(start, end) };
            });
            // least residual, then least slot — deterministic, independent of the task→executor
            // assignment.
            partials
                .into_iter()
                .flatten()
                .min_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)))
        };

        enum Pick<'a, K, V> {
            Main(usize),
            Side(usize),
            Overflow(&'a K, &'a V),
        }
        let mut best: Option<(M, Pick<'_, K, V>)> = main.map(|(m, i)| (m, Pick::Main(i)));
        for i in 0..self.side.coords.len() {
            // SAFETY: every side entry is occupied ⇒ its value is initialized.
            let m = residual(unsafe { self.side.vals.get_unchecked(i).assume_init_ref() });
            if best.as_ref().map_or(true, |(bm, _)| m < *bm) {
                best = Some((m, Pick::Side(i)));
            }
        }
        if let Some(o) = self.overflow.as_ref() {
            for (_coord, k, v) in o.iter() {
                let m = residual(v);
                if best.as_ref().map_or(true, |(bm, _)| m < *bm) {
                    best = Some((m, Pick::Overflow(k, v)));
                }
            }
        }
        best.map(|(_, pick)| match pick {
            Pick::Main(i) => {
                let c = self.coords[i];
                // SAFETY: the argmin ranged over occupied slots only ⇒ value initialized.
                (unsafe { self.keys.key(i, c) }, unsafe {
                    self.vals.get_unchecked(i).assume_init_ref()
                })
            }
            Pick::Side(i) => {
                let c = self.side.coords[i];
                // SAFETY: a side entry is occupied ⇒ value initialized.
                (unsafe { self.side.keys.key(i, c) }, unsafe {
                    self.side.vals.get_unchecked(i).assume_init_ref()
                })
            }
            Pick::Overflow(k, v) => (self.keys.key_of(k), v),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::CoordMap;
    use super::*;

    fn doubled(n: u64) -> Table<u64, u64> {
        let mut t = Table::new();
        for k in 0..n {
            t.insert(k, k * 2);
        }
        t
    }

    #[test]
    fn coordinate_builds_an_empty_coord_map() {
        let mut t: CoordMap<u64, u64> = Table::coordinate();
        assert!(t.is_empty());
        t.insert(5, 6);
        assert_eq!(t.get(&5), Some(&6));
    }

    #[test]
    fn get_all_matches_get_on_the_serial_and_the_parallel_path() {
        for n in [10u64, 70_000] {
            let t = doubled(n);
            let keys: Vec<u64> = (0..n + 5).collect();
            let got = t.get_all(&keys);
            assert_eq!(got.len(), keys.len());
            for (k, v) in keys.iter().zip(got) {
                assert_eq!(v, t.get(k));
            }
        }
    }

    #[test]
    fn get_reduce_sums_present_values_on_both_paths() {
        for n in [10u64, 70_000] {
            let t = doubled(n);
            let keys: Vec<u64> = (0..n + 5).collect();
            let got = t.get_reduce(&keys, 0u64, |v| v.map_or(0, |v| *v), |a, b| a + b);
            assert_eq!(got, (0..n).map(|k| k * 2).sum::<u64>());
        }
    }

    #[test]
    fn encircle_folds_every_entry_including_the_side_store() {
        for n in [100u64, 40_000] {
            let mut t: Table<u64, u64> = Table::new();
            for k in 0..n {
                t.insert(k, 1);
            }
            // These coordinates are the reserved EMPTY and TOMB markers, so they live in the side
            // store and not in the main slots.
            t.insert(u64::MAX, 1);
            t.insert(u64::MAX - 1, 1);
            assert_eq!(t.encircle(0u64, |v| *v, |a, b| a + b), n + 2);
        }
    }

    #[test]
    fn encircle_floor_finds_the_smallest_residual_and_its_key() {
        let mut t: Table<u64, i64> = Table::new();
        for k in 0..100u64 {
            t.insert(k, (k as i64 - 50).abs() + 1);
        }
        let (k, v) = t.encircle_floor(|v| *v).expect("non-empty");
        assert_eq!((*k, *v), (50, 1));
        // A side-store entry is a candidate too, and wins when its residual is smaller.
        t.insert(u64::MAX, 0);
        let (k, v) = t.encircle_floor(|v| *v).expect("non-empty");
        assert_eq!((*k, *v), (u64::MAX, 0));
    }

    #[test]
    fn encircle_floor_breaks_ties_toward_the_main_slots() {
        let mut t: Table<u64, i64> = Table::new();
        t.insert(7, 3);
        t.insert(u64::MAX, 3);
        let (k, _) = t.encircle_floor(|v| *v).expect("non-empty");
        assert_eq!(*k, 7);
    }

    #[test]
    fn encircle_floor_of_an_empty_table_is_none() {
        let t: Table<u64, i64> = Table::new();
        assert!(t.encircle_floor(|v| *v).is_none());
    }
}
