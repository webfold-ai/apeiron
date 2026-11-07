//! The side store for reserved-coordinate keys (`EMPTY`/`TOMB`), and the merge guard every merge
//! face runs under, moved out of `mod.rs` to keep it under the 1500-line bound.

use super::*;

/// The side store for keys whose own coordinate is a reserved marker (`EMPTY` or `TOMB`). Such a key
/// cannot live in `coords` — its coordinate would alias the occupancy flag — so its identity is held
/// here, outside the slot arrays. `coords[i]` is the reserved coordinate, `vals[i]` the value
/// (initialized for every entry `i < coords.len()`; a merge moves it out only while a
/// [`MergeGuard`] holds the table, and a panicking merge swap-removes the entry), and `keys` the
/// identity exactly as the main store carries it: the full key for `Stored`, nothing for `Regen`
/// (which regenerates the key from `coords[i]`; a keyless store refuses reserved-coordinate keys, see
/// [`push`](Self::push)). The store is packed and scanned linearly; it holds only the keys that literally ask
/// for a sentinel, so it is empty (one zero-length `Vec` and a zero-sized key store) on every
/// ordinary table. Capacity is O(side entries), never O(table slots).
pub(super) struct SideStore<K: Coord, V, KS: KeyStore<K>> {
    pub(super) coords: Vec<u64>,
    pub(super) vals: Vec<MaybeUninit<V>>,
    pub(super) keys: KS, // allocated to `cap`; entries `0..coords.len()` initialized
    pub(super) cap: usize,
    pub(super) _k: std::marker::PhantomData<K>,
}

impl<K: Coord, V, KS: KeyStore<K>> SideStore<K, V, KS> {
    pub(super) fn empty() -> Self {
        SideStore {
            coords: Vec::new(),
            vals: Vec::new(),
            keys: KS::alloc(0),
            cap: 0,
            _k: std::marker::PhantomData,
        }
    }

    /// Append `(coord, k, v)`, growing the key store's fixed-capacity allocation by doubling. The
    /// values and coordinates live in plain `Vec`s and grow themselves.
    pub(super) fn push(&mut self, coord: u64, k: K, v: V) {
        // A keyless store cannot hold a reserved-coordinate key: `Regen` regenerates the key from
        // its coordinate, and the reserved words are the very values that were refused as keys. A
        // refusal here, before anything is stored, leaves the table unchanged.
        assert!(
            !KS::REGEN,
            "a CoordMap cannot store a key whose coordinate is u64::MAX or u64::MAX - 1 (reserved \
             slot markers); use a Table, which keeps the key"
        );
        let need = self.coords.len() + 1;
        if need > self.cap {
            let new_cap = (self.cap * 2).max(4).max(need);
            let mut grown = KS::alloc(new_cap);
            let mut i = 0;
            while i < self.coords.len() {
                let c = self.coords[i];
                let key = self.keys.take_key(i, c);
                grown.set(i, key);
                i += 1;
            }
            self.keys = grown;
            self.cap = new_cap;
        }
        let i = self.coords.len();
        self.keys.set(i, k);
        self.coords.push(coord);
        self.vals.push(MaybeUninit::new(v));
    }

    /// The entry holding key `k`: coordinate equality plus the key compare (`matches` is the
    /// coordinate round-trip for `Regen`), so distinct keys with equal reserved coordinates stay
    /// distinct here exactly as they do in the main array.
    pub(super) fn find(&self, coord: u64, k: &K) -> Option<usize> {
        let mut i = 0;
        while i < self.coords.len() {
            if self.coords[i] == coord && self.keys.matches(i, coord, k) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Remove entry `i`; its value must already have been moved out by the caller. The removed key is
    /// dropped, and the last entry is swapped into `i` — its key explicitly, since `KS` has no swap.
    pub(super) fn remove_entry(&mut self, i: usize) {
        let gone = self.keys.take_key(i, self.coords[i]);
        let last = self.coords.len() - 1;
        if i != last {
            let moved = self.keys.take_key(last, self.coords[last]);
            self.keys.set(i, moved);
            self.coords.swap(i, last);
            self.vals.swap(i, last);
        }
        self.coords.pop();
        self.vals.pop();
        drop(gone);
    }

    /// Move entry `i`'s `(coord, key, value)` out and swap-remove it. The caller owns all three;
    /// nothing is dropped here. The owner-consuming faces (`fuse`, `drain`, `IntoIterator`) take an
    /// entry apart with this rather than leave it in the store.
    pub(super) fn take_entry(&mut self, i: usize) -> (u64, K, V) {
        let c = self.coords[i];
        // SAFETY: entry `i` is occupied ⇒ its key and value are initialized.
        let key = self.keys.take_key(i, c);
        let val = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
        let last = self.coords.len() - 1;
        if i != last {
            let moved = self.keys.take_key(last, self.coords[last]);
            self.keys.set(i, moved);
        }
        self.coords.swap(i, last);
        self.vals.swap(i, last);
        self.coords.pop();
        self.vals.pop();
        (c, key, val)
    }

    /// Drop every entry, keeping the key store's allocation — the side twin of `Table::clear`.
    ///
    /// The entry tables are detached before any destructor runs: `self`'s live-entry length is zero
    /// from the first line on, so a panicking `V::drop` or `K::drop` cannot leave `coords`/`vals`
    /// marking an entry whose value or key is already gone. On the unwind the entries not yet
    /// dropped leak with the local `Vec`s; a later `clear` (from `Drop for Table`) sees an empty
    /// store and drops nothing. Without the detach, the lengths stay long through the panicking
    /// drop, and the later `clear` reads a moved-out value or drops entry 0 twice.
    pub(super) fn clear(&mut self) {
        let coords = std::mem::take(&mut self.coords);
        let mut vals = std::mem::take(&mut self.vals);
        if std::mem::needs_drop::<V>() {
            let mut i = 0;
            while i < vals.len() {
                // SAFETY: every entry's value is initialized.
                unsafe { vals[i].assume_init_drop() };
                i += 1;
            }
        }
        let mut i = 0;
        while i < coords.len() {
            let c = coords[i];
            drop(self.keys.take_key(i, c));
            i += 1;
        }
    }
}

/// A merge in flight on one entry of a [`Table`]. A merge face moves the entry's old value out,
/// marks a main slot `TOMB`, arms this guard, runs the caller's merge, and hands the result to
/// [`finish`](Self::finish), which writes it back and disarms. If the merge unwinds instead, `Drop`
/// removes the key, so a panicking merge leaves a consistent map:
///
/// - main slot: the slot stays `TOMB` (never `EMPTY`, so the `EMPTY` → occupied invariant the capped
///   probe and the [`Overflow`] rest on holds across the unwind), its key is dropped, `len` falls by
///   one and `tombs` rises by one;
/// - side entry: swap-removed with its key ([`SideStore::remove_entry`]);
/// - overflow entry: deleted with its key, and its node when it empties
///   ([`Overflow::discard_entry`]).
///
/// The old and new values belong to the merge's arguments, so the unwind drops each once; the guard
/// drops only the key. While the guard lives it holds the table's only borrow, so no read can see
/// the moved-out value. Cost: O(1) on the main and side paths; the overflow removal is O(log n) node
/// rebalances, paid only on the unwind.
pub(super) struct MergeGuard<'a, K: Coord, V, KS: KeyStore<K>> {
    pub(super) t: &'a mut Table<K, V, KS>,
    pub(super) store: Store,
    /// The main slot or the side entry.
    pub(super) slot: usize,
    /// The overflow node and entry.
    pub(super) node: u32,
    pub(super) entry: u32,
    /// The entry's coordinate, restored on a main slot by `finish`.
    pub(super) coord: u64,
}

impl<K: Coord, V, KS: KeyStore<K>> MergeGuard<'_, K, V, KS> {
    /// Write `merged` into the held entry, re-occupy a main slot with its coordinate, and disarm.
    pub(super) fn finish(self, merged: V) {
        let t = &mut *self.t;
        match self.store {
            // SAFETY: the slot index is in bounds (it came from the probe) and its value was moved
            // out when the guard was armed, so the write initializes it; restoring the coordinate
            // makes the slot live again over an initialized value and key.
            Store::Main => unsafe {
                *t.vals.get_unchecked_mut(self.slot) = MaybeUninit::new(merged);
                *t.coords.get_unchecked_mut(self.slot) = self.coord;
            },
            // SAFETY: as above, for the side entry.
            Store::Side => unsafe {
                *t.side.vals.get_unchecked_mut(self.slot) = MaybeUninit::new(merged);
            },
            Store::Overflow => t
                .overflow
                .as_mut()
                .expect("overflow merge ⇒ store present")
                .write_back(self.node, self.entry, merged),
        }
        std::mem::forget(self);
    }
}

/// Runs only on an unwind out of the merge (`finish` forgets the guard): remove the key.
impl<K: Coord, V, KS: KeyStore<K>> Drop for MergeGuard<'_, K, V, KS> {
    fn drop(&mut self) {
        let t = &mut *self.t;
        match self.store {
            Store::Main => {
                // The slot is already `TOMB` and its value is gone; drop the key it still holds.
                drop(t.keys.take_key(self.slot, self.coord));
                t.len -= 1;
                t.tombs += 1;
            }
            Store::Side => t.side.remove_entry(self.slot),
            Store::Overflow => {
                if let Some(o) = t.overflow.as_mut() {
                    o.discard_entry(self.node, self.entry);
                }
            }
        }
    }
}
