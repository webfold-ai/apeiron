// Growth at load LOAD_NUM/LOAD_DEN = 3/4.
//! The std-shaped faces of [`Table`] — the surface a *mechanical* `HashMap` port lands on.
//!
//! [`Table::insert_with`] is the whole algebra: one probe, one merge, four faces. That is the right
//! primitive and it is not what a porter reaches for. A porter reaches for `entry`, `get_mut`,
//! `insert`, `retain`, `for (k, v) in &map`, and — most expensively — `map.get(s)` where `s: &str`
//! and the key type is `String`. When those are absent the port still *compiles*, because the porter
//! writes the workaround: two probes instead of one, a `to_string()` per lookup, a rebuild instead
//! of a `retain`. Every one of those is slower than the `HashMap` it replaced, and none of them is a
//! user error — a primitive that is only fast in expert hands has failed at the layer that matters.
//!
//! Everything here is additive and expressed in the existing fold. Nothing re-implements the probe.
//!
//! # The two cliffs
//!
//! **Read-modify-write.** `*m.entry(k).or_insert(0) += 1` is ONE probe in std because `entry` hands
//! back the `&mut`. [`Table::entry`] does the same: probe once, hand back the slot.
//! [`Table::insert_fetch`] is the merge-shaped form of it.
//!
//! **Borrowed and composite keys.** `HashMap<String, V>` is probed with `&str` through
//! `Borrow<Q>`; `Table<String, V>`'s [`get`](Table::get) takes `&K`, so the mechanical port writes
//! `t.get(&s.to_string())` — an allocation on every lookup. Worse, `Table<(String, String), V>`
//! *compiles* when probed with an owned tuple, so the cost is two `String` clones per probe and no
//! compiler error to notice it. [`Locate`] is the fix: a borrowed view that addresses the same
//! coordinate as the owned key, so `(&str, &str)` probes a `(String, String)` table with **zero**
//! allocation. It extends the crate's existing tuple fold (`c ← c·PHI ⊕ xᵢ`) rather than adding a
//! second scheme — see [`Locate`] for the identity that makes that exact.
//!
//! ```ignore
//! use apeiron::map::hashtrinity::Table;
//! let mut t: Table<(String, String), i64> = Table::new();
//! t.put(("lo".into(), "hi".into()), 7);
//! assert_eq!(t.get_at(("lo", "hi")), Some(&7)); // no String materialized
//! *t.entry_at(("lo", "hi")).or_insert(0) += 1;  // one probe, no String materialized
//! assert_eq!(t.get_at(("lo", "hi")), Some(&8));
//! ```

// A `Locate` view IS the borrow — `&K`, `&str`, `(&str, &str)`. Taking it by value is the whole
// point (it is one or two machine words, and `Absorb` consumes it on the vacant path); taking it by
// reference would put a second indirection on the hot probe.
#![allow(clippy::needless_pass_by_value)]

use crate::map::hashtrinity::{
    bits_for, is_live, is_sentinel, side::MergeGuard, Capped, Coord, KeyReadout, KeyStore, Overflow,
    SideStore, Store, Stored, Table, EMPTY, PHI, TOMB,
};
use std::mem::MaybeUninit;

// ── Locate / Absorb: a borrowed key that addresses the same slot as the owned one. ─────────────

/// A borrowed view of a key that lands on the **same coordinate** as the owned key `K`.
///
/// This is `Borrow<Q>`'s job in `std`, done through the fold instead of through `Hash`. The
/// contract is one equation:
///
/// > `q.coord() == owned.coord()` whenever `q` locates `owned`
///
/// and it holds *by construction*, not by convention, because every impl reduces to the coordinate
/// the crate already computes for that shape:
///
/// * `&K` → `K::coord` (the identity impl).
/// * `&str` → `fold_bytes(bytes)`, which is *literally* `<String as Coord>::coord`'s body.
/// * `(PA, PB)` → `pa.coord()·PHI ⊕ pb.coord()`, the SAME expression as
///   `<(A, B) as Coord>::coord`, with each component's coordinate supplied by that component's own
///   `Locate`. By induction the composite equals `<(A, B) as Coord>::coord` of the absorbed key.
///
/// So this is the existing tuple fold, not a second scheme, and nothing downstream of the `u64`
/// changes: the radix descent, the `⌈64/bits⌉` separation bound, the HashDOS immunity, and the
/// `2⁻⁶⁴` coordinate-identity risk are all untouched. `tests/locate.rs` pins the equation.   CITATION-EXEMPT: deleted by `d84bda3`, 25 crates → 12
///
/// [`same`](Locate::same) is the exact key compare, used only by the `_exact` faces — the ordinary
/// borrowed read trusts the coordinate exactly as [`Table::get`] does, and costs exactly the same.
pub trait Locate<K: Coord + Eq> {
    /// This view's coordinate — equal to `K::coord` of the key it locates.
    fn coord(&self) -> u64;
    /// Whether this view names exactly `k` (the exact compare; `_exact` faces only).
    fn same(&self, k: &K) -> bool;
}

/// A borrowed view that can be turned into the owned key — paid **only** when a probe misses and the
/// table must take ownership. On a hit, nothing is allocated.
pub trait Absorb<K: Coord + Eq>: Locate<K> {
    /// Materialize the owned key. Called on the vacant path and nowhere else.
    fn absorb(self) -> K;
}

impl<K: Coord + Eq> Locate<K> for &K {
    #[inline]
    fn coord(&self) -> u64 {
        (**self).coord()
    }
    #[inline]
    fn same(&self, k: &K) -> bool {
        *self == k
    }
}
impl<K: Coord + Eq + Clone> Absorb<K> for &K {
    #[inline]
    fn absorb(self) -> K {
        self.clone()
    }
}

impl Locate<String> for &str {
    #[inline]
    fn coord(&self) -> u64 {
        // `<str as Coord>::coord` and `<String as Coord>::coord` are the same `fold_bytes` call.
        <str as Coord>::coord(self)
    }
    #[inline]
    fn same(&self, k: &String) -> bool {
        // Byte equality is `str` equality; the byte form is the one the model defines
        // (`Str.as_bytes` + `Slice` eq), where `PartialEq for str` is not.
        self.as_bytes() == k.as_bytes()
    }
}
impl Absorb<String> for &str {
    #[inline]
    fn absorb(self) -> String {
        self.to_owned()
    }
}

impl<A: Coord + Eq, B: Coord + Eq, PA: Locate<A>, PB: Locate<B>> Locate<(A, B)> for (PA, PB) {
    #[inline]
    fn coord(&self) -> u64 {
        // Identical to `<(A, B) as Coord>::coord` — the crate's tuple fold, on borrowed parts.
        self.0.coord().wrapping_mul(PHI) ^ self.1.coord()
    }
    #[inline]
    fn same(&self, k: &(A, B)) -> bool {
        self.0.same(&k.0) && self.1.same(&k.1)
    }
}
impl<A: Coord + Eq, B: Coord + Eq, PA: Absorb<A>, PB: Absorb<B>> Absorb<(A, B)> for (PA, PB) {
    #[inline]
    fn absorb(self) -> (A, B) {
        (self.0.absorb(), self.1.absorb())
    }
}

impl<A: Coord + Eq, B: Coord + Eq, C: Coord + Eq, PA: Locate<A>, PB: Locate<B>, PC: Locate<C>>
    Locate<(A, B, C)> for (PA, PB, PC)
{
    #[inline]
    fn coord(&self) -> u64 {
        (self.0.coord().wrapping_mul(PHI) ^ self.1.coord()).wrapping_mul(PHI) ^ self.2.coord()
    }
    #[inline]
    fn same(&self, k: &(A, B, C)) -> bool {
        self.0.same(&k.0) && self.1.same(&k.1) && self.2.same(&k.2)
    }
}
impl<A: Coord + Eq, B: Coord + Eq, C: Coord + Eq, PA: Absorb<A>, PB: Absorb<B>, PC: Absorb<C>>
    Absorb<(A, B, C)> for (PA, PB, PC)
{
    #[inline]
    fn absorb(self) -> (A, B, C) {
        (self.0.absorb(), self.1.absorb(), self.2.absorb())
    }
}

// ── Entry: the one-probe read-modify-write. ────────────────────────────────────────────────────

/// A single probe's result, held open so the caller can decide what to do with the slot. The std
/// `Entry` shape, on the fold's one probe: [`Table::entry`] walks the table exactly once, and
/// nothing here walks it again.
pub enum Entry<'a, K: Coord + Eq, V, KS: KeyStore<K>> {
    /// The key is present; the slot is live.
    Occupied(OccupiedEntry<'a, K, V, KS>),
    /// The key is absent; the slot is where it would go.
    Vacant(VacantEntry<'a, K, V, KS>),
}

/// A live entry, already located. Every method here is zero further probes. `store` selects the main
/// slot array, the side store (a reserved-coordinate key), or the overflow tree (`node`/`entry`).
pub struct OccupiedEntry<'a, K: Coord + Eq, V, KS: KeyStore<K>> {
    t: &'a mut Table<K, V, KS>,
    slot: usize,
    node: u32,
    entry: u32,
    coord: u64,
    store: Store,
}

/// The slot an absent key would occupy, already located, holding the key that would go there.
pub struct VacantEntry<'a, K: Coord + Eq, V, KS: KeyStore<K>> {
    t: &'a mut Table<K, V, KS>,
    slot: usize,
    coord: u64,
    key: K,
    store: Store,
}

/// Where [`locate_at`](Table::locate_at) found a borrowed view — the main array, the side store, or
/// the overflow.
enum Located {
    Main(usize),
    Side(usize),
    Overflow(u32, u32),
}

impl<'a, K: Coord + Eq, V, KS: KeyStore<K>> OccupiedEntry<'a, K, V, KS> {
    /// This entry's key — borrowed when stored, regenerated from the address when not.
    #[inline]
    pub fn key(&self) -> KS::Key<'_>
    where
        KS: KeyReadout<K>,
    {
        match self.store {
            // SAFETY: slot is live (the probe matched a live coordinate here).
            Store::Side => unsafe { self.t.side.keys.key(self.slot, self.coord) },
            // SAFETY: slot is live (the probe matched a live coordinate here).
            Store::Main => unsafe { self.t.keys.key(self.slot, self.coord) },
            Store::Overflow => {
                let ov = self.t.overflow.as_ref().expect("overflow entry ⇒ store present");
                self.t.keys.key_of(&ov.nodes[self.node as usize].entries[self.entry as usize].0)
            }
        }
    }
    /// The value at this entry.
    #[inline]
    pub fn get(&self) -> &V {
        match self.store {
            // SAFETY: an occupied side entry ⇒ its value is initialized.
            Store::Side => unsafe { self.t.side.vals.get_unchecked(self.slot).assume_init_ref() },
            // SAFETY: an occupied entry names a live slot ⇒ its value is initialized.
            Store::Main => unsafe { self.t.vals.get_unchecked(self.slot).assume_init_ref() },
            Store::Overflow => {
                let ov = self.t.overflow.as_ref().expect("overflow entry ⇒ store present");
                ov.nodes[self.node as usize].entries[self.entry as usize]
                    .1
                    .as_ref()
                    .expect("overflow entry ⇒ live value")
            }
        }
    }
    /// The value at this entry, mutably.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        match self.store {
            // SAFETY: an occupied side entry ⇒ its value is initialized.
            Store::Side => unsafe {
                self.t.side.vals.get_unchecked_mut(self.slot).assume_init_mut()
            },
            // SAFETY: an occupied entry names a live slot ⇒ its value is initialized.
            Store::Main => unsafe { self.t.vals.get_unchecked_mut(self.slot).assume_init_mut() },
            Store::Overflow => {
                let ov = self
                    .t
                    .overflow
                    .as_mut()
                    .expect("overflow entry ⇒ store present");
                ov.nodes[self.node as usize].entries[self.entry as usize]
                    .1
                    .as_mut()
                    .expect("overflow entry ⇒ live value")
            }
        }
    }
    /// The value at this entry, mutably, for as long as the table borrow lives.
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        self.get_mut_owned()
    }
    fn get_mut_owned(self) -> &'a mut V {
        // SAFETY: the entry is occupied, so the value is initialized; the borrow is the table's.
        match self.store {
            Store::Side => unsafe {
                self.t.side.vals.get_unchecked_mut(self.slot).assume_init_mut()
            },
            Store::Main => unsafe { self.t.vals.get_unchecked_mut(self.slot).assume_init_mut() },
            Store::Overflow => {
                let ov = self
                    .t
                    .overflow
                    .as_mut()
                    .expect("overflow entry ⇒ store present");
                ov.nodes[self.node as usize].entries[self.entry as usize]
                    .1
                    .as_mut()
                    .expect("overflow entry ⇒ live value")
            }
        }
    }
    /// Overwrite the value and return the old one. The `Set` face, at a located entry.
    #[inline]
    pub fn insert(&mut self, v: V) -> V {
        std::mem::replace(self.get_mut(), v)
    }
    /// Take the value out, tombstoning the slot / removing the overflow entry (see [`Table::remove`]).
    #[inline]
    pub fn remove(self) -> V {
        self.remove_entry().1
    }
    /// Take the key and the value out, tombstoning the slot / removing the overflow entry.
    #[inline]
    pub fn remove_entry(self) -> (K, V) {
        let (i, c) = (self.slot, self.coord);
        match self.store {
            Store::Side => {
                // SAFETY: an occupied side entry ⇒ key and value initialized; `take_entry` moves both
                // out and swap-removes the entry, so nothing double-drops.
                let (_c, k, v) = self.t.side.take_entry(i);
                (k, v)
            }
            Store::Overflow => {
                let ov = self
                    .t
                    .overflow
                    .as_mut()
                    .expect("overflow entry ⇒ store present");
                ov.take_entry(self.node, self.entry)
            }
            Store::Main => {
                // SAFETY: live slot ⇒ value initialized. Read it out and take the key, then tombstone:
                // the slot's value is now uninitialized and `TOMB` fails the liveness test, so
                // Drop/Clone/iter skip it — no double free, and the chain past it stays intact.
                let v = unsafe { self.t.vals.get_unchecked_mut(i).assume_init_read() };
                // SAFETY: slot is live (its coordinate is checked above); the caller stops using it afterwards.
                let k = unsafe { self.t.keys.take_key(i, c) };
                unsafe { *self.t.coords.get_unchecked_mut(i) = TOMB };
                self.t.len -= 1;
                self.t.tombs += 1;
                (k, v)
            }
        }
    }
}

impl<'a, K: Coord + Eq, V, KS: KeyStore<K>> VacantEntry<'a, K, V, KS> {
    /// The key that would be inserted here.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }
    /// Take the key back out (the entry is dropped).
    #[inline]
    pub fn into_key(self) -> K {
        self.key
    }
    /// Fill the entry and hand back the value. The table was already sized for this by
    /// [`Table::entry`], so the slot index is still the right one — no second probe.
    #[inline]
    pub fn insert(self, v: V) -> &'a mut V {
        match self.store {
            Store::Side => {
                self.t.side.push(self.coord, self.key, v);
                let i = self.t.side.vals.len() - 1;
                // SAFETY: just pushed ⇒ initialized.
                unsafe { self.t.side.vals.get_unchecked_mut(i).assume_init_mut() }
            }
            Store::Overflow => {
                let ov = self
                    .t
                    .overflow
                    .get_or_insert_with(Overflow::new);
                ov.insert_new(self.coord, self.key, v)
            }
            Store::Main => {
                let (i, c) = (self.slot, self.coord);
                // SAFETY: `entry` grew the table BEFORE probing, so `i` is a valid EMPTY slot in the
                // current arrays; nothing between here and there can panic.
                unsafe { *self.t.coords.get_unchecked_mut(i) = c };
                unsafe { *self.t.vals.get_unchecked_mut(i) = MaybeUninit::new(v) };
                // SAFETY: slot is in bounds and vacant (the probe's EMPTY slot), so nothing is dropped.
                unsafe { self.t.keys.set(i, self.key) }; // Stored: write the key; Regen: no-op
                self.t.len += 1;
                // SAFETY: just written ⇒ initialized.
                unsafe { self.t.vals.get_unchecked_mut(i).assume_init_mut() }
            }
        }
    }
}

impl<'a, K: Coord + Eq, V, KS: KeyStore<K>> Entry<'a, K, V, KS> {
    /// The value, inserting `v` if the key was absent. `*t.entry(k).or_insert(0) += 1` — the RMW a
    /// porter writes, at ONE probe.
    #[inline]
    pub fn or_insert(self, v: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(v),
        }
    }
    /// The value, inserting `f()` if the key was absent (the default is not built on a hit).
    #[inline]
    pub fn or_insert_with(self, f: impl FnOnce() -> V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(f()),
        }
    }
    /// The value, inserting `f(&key)` if the key was absent.
    #[inline]
    pub fn or_insert_with_key(self, f: impl FnOnce(&K) -> V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let v = f(&e.key);
                e.insert(v)
            }
        }
    }
    /// The value, inserting `V::default()` if the key was absent.
    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }
    /// Run `f` on the value if the key was present, then carry on.
    #[must_use]
    #[inline]
    pub fn and_modify(mut self, f: impl FnOnce(&mut V)) -> Self {
        if let Entry::Occupied(e) = &mut self {
            f(e.get_mut());
        }
        self
    }
}

// ── The table faces. ───────────────────────────────────────────────────────────────────────────

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    /// Locate `k` in ONE probe and hold the slot open. The std `entry` face:
    /// `*t.entry(k).or_insert(0) += 1` is one probe, one merge, no second lookup.
    ///
    /// Grows first, then probes, so [`VacantEntry::insert`] never has to re-probe. A reserved
    /// coordinate resolves in the [`SideStore`] (which grows itself); a non-reserved coordinate is
    /// the keyed probe, so two distinct keys with equal coordinates take two slots.
    #[inline]
    pub fn entry(&mut self, k: K) -> Entry<'_, K, V, KS> {
        let coord = Self::coord_of(&k);
        if is_sentinel(coord) {
            if let Some(slot) = self.side.find(coord, &k) {
                return Entry::Occupied(OccupiedEntry {
                    t: self,
                    slot,
                    node: 0,
                    entry: 0,
                    coord,
                    store: Store::Side,
                });
            }
            return Entry::Vacant(VacantEntry {
                t: self,
                slot: 0,
                coord,
                key: k,
                store: Store::Side,
            });
        }
        self.grow_for_one();
        match self.probe_capped_keyed(coord, &k) {
            Capped::Hit(slot) => Entry::Occupied(OccupiedEntry {
                t: self,
                slot,
                node: 0,
                entry: 0,
                coord,
                store: Store::Main,
            }),
            Capped::Vacant(slot) => Entry::Vacant(VacantEntry {
                t: self,
                slot,
                coord,
                key: k,
                store: Store::Main,
            }),
            Capped::Blocked => {
                let found = self
                    .overflow
                    .as_ref()
                    .and_then(|o| o.find_entry(coord, &k));
                match found {
                    Some((node, entry)) => Entry::Occupied(OccupiedEntry {
                        t: self,
                        slot: 0,
                        node,
                        entry,
                        coord,
                        store: Store::Overflow,
                    }),
                    None => Entry::Vacant(VacantEntry {
                        t: self,
                        slot: 0,
                        coord,
                        key: k,
                        store: Store::Overflow,
                    }),
                }
            }
        }
    }

    /// [`entry`](Self::entry) from a BORROWED key — the owned key is materialized only if the probe
    /// misses and the table has to take ownership. `t.entry_at(("a", "b")).or_insert(0)` on a
    /// `Table<(String, String), _>` allocates nothing on the hit path.
    #[inline]
    pub fn entry_at<Q: Absorb<K>>(&mut self, q: Q) -> Entry<'_, K, V, KS>
    where
        KS: KeyReadout<K>,
    {
        let coord = q.coord();
        if is_sentinel(coord) {
            if let Some(slot) = self.side_find_locate(coord, &q) {
                return Entry::Occupied(OccupiedEntry {
                    t: self,
                    slot,
                    node: 0,
                    entry: 0,
                    coord,
                    store: Store::Side,
                });
            }
            return Entry::Vacant(VacantEntry {
                t: self,
                slot: 0,
                coord,
                key: q.absorb(), // the ONLY allocation, and only when the key is genuinely new
                store: Store::Side,
            });
        }
        self.grow_for_one();
        match self.probe_capped_locate(coord, &q) {
            Capped::Hit(slot) => Entry::Occupied(OccupiedEntry {
                t: self,
                slot,
                node: 0,
                entry: 0,
                coord,
                store: Store::Main,
            }),
            Capped::Vacant(slot) => Entry::Vacant(VacantEntry {
                t: self,
                slot,
                coord,
                key: q.absorb(), // the ONLY allocation, and only when the key is genuinely new
                store: Store::Main,
            }),
            Capped::Blocked => {
                let found = self
                    .overflow
                    .as_ref()
                    .and_then(|o| o.find_locate(coord, &q));
                match found {
                    Some((node, entry)) => Entry::Occupied(OccupiedEntry {
                        t: self,
                        slot: 0,
                        node,
                        entry,
                        coord,
                        store: Store::Overflow,
                    }),
                    None => Entry::Vacant(VacantEntry {
                        t: self,
                        slot: 0,
                        coord,
                        key: q.absorb(), // the ONLY allocation, and only when the key is genuinely new
                        store: Store::Overflow,
                    }),
                }
            }
        }
    }

    /// The value for `k`, mutably. One probe.
    #[inline]
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let coord = Self::coord_of(k);
        if is_sentinel(coord) {
            let i = self.side.find(coord, k)?;
            // SAFETY: `find` returns an occupied side entry ⇒ value initialized.
            return Some(unsafe { self.side.vals.get_unchecked_mut(i).assume_init_mut() });
        }
        match self.probe_capped_keyed(coord, k) {
            // SAFETY: found ⇒ value initialized.
            Capped::Hit(i) => Some(unsafe { self.vals.get_unchecked_mut(i).assume_init_mut() }),
            Capped::Vacant(_) => None,
            Capped::Blocked => self.overflow.as_mut().and_then(|o| o.get_mut(coord, k)),
        }
    }

    /// [`get`](Table::get) from a borrowed key — `t.get_at("word")` on a `Table<String, _>`,
    /// `t.get_at((a, b))` on a `Table<(String, String), _>`. Zero allocation, one probe. The stored
    /// key is confirmed on a coordinate match (free on `Regen`, where the coordinate round-trips),
    /// so a coordinate collision reads as absent instead of returning another key's value; a
    /// reserved coordinate resolves in the [`SideStore`].
    #[inline]
    pub fn get_at<Q: Locate<K>>(&self, q: Q) -> Option<&V>
    where
        KS: KeyReadout<K>,
    {
        match self.locate_at(&q)? {
            // SAFETY: a located entry is occupied ⇒ value initialized.
            Located::Main(i) => Some(unsafe { self.vals.get_unchecked(i).assume_init_ref() }),
            Located::Side(i) => Some(unsafe { self.side.vals.get_unchecked(i).assume_init_ref() }),
            Located::Overflow(n, e) => self
                .overflow
                .as_ref()
                .and_then(|o| o.nodes[n as usize].entries[e as usize].1.as_ref()),
        }
    }

    /// [`get_mut`](Self::get_mut) from a borrowed key.
    #[inline]
    pub fn get_mut_at<Q: Locate<K>>(&mut self, q: Q) -> Option<&mut V>
    where
        KS: KeyReadout<K>,
    {
        match self.locate_at(&q)? {
            // SAFETY: a located entry is occupied ⇒ value initialized.
            Located::Main(i) => Some(unsafe { self.vals.get_unchecked_mut(i).assume_init_mut() }),
            Located::Side(i) => {
                Some(unsafe { self.side.vals.get_unchecked_mut(i).assume_init_mut() })
            }
            Located::Overflow(n, e) => self
                .overflow
                .as_mut()
                .and_then(|o| o.nodes[n as usize].entries[e as usize].1.as_mut()),
        }
    }

    /// Whether `k` is present. One probe, no value touched.
    #[inline]
    pub fn contains_key(&self, k: &K) -> bool {
        let coord = Self::coord_of(k);
        if is_sentinel(coord) {
            return self.side.find(coord, k).is_some();
        }
        match self.probe_capped_keyed(coord, k) {
            Capped::Hit(_) => true,
            Capped::Vacant(_) => false,
            // A `match`, not `Option::is_some_and`: the closure form is not in the model.
            Capped::Blocked => match self.overflow.as_ref() {
                Some(o) => o.get(coord, k).is_some(),
                None => false,
            },
        }
    }

    /// Whether a borrowed key is present. One probe, no allocation, full-key confirmed.
    #[inline]
    pub fn contains_at<Q: Locate<K>>(&self, q: Q) -> bool
    where
        KS: KeyReadout<K>,
    {
        self.locate_at(&q).is_some()
    }

    /// Set `k` to `v` and return the value it replaced. The `Set` face with its old value handed
    /// back — std's `insert`. One probe.
    #[inline]
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        match self.entry(k) {
            Entry::Occupied(mut e) => Some(e.insert(v)),
            Entry::Vacant(e) => {
                e.insert(v);
                None
            }
        }
    }

    /// [`insert`](Self::insert) from a borrowed key — no allocation unless the key is new.
    #[inline]
    pub fn insert_at<Q: Absorb<K>>(&mut self, q: Q, v: V) -> Option<V>
    where
        KS: KeyReadout<K>,
    {
        match self.entry_at(q) {
            Entry::Occupied(mut e) => Some(e.insert(v)),
            Entry::Vacant(e) => {
                e.insert(v);
                None
            }
        }
    }

    /// [`insert_fetch`](Self::insert_fetch) from a borrowed key: merge and hand back the result, in
    /// ONE probe, materializing the key only when it is new. This is the pair-counter shape —
    /// `counts[(a, b)] += 1; push(counts[(a, b)])` — with neither a second probe nor a `String`.
    #[inline]
    pub fn insert_fetch_at<Q: Absorb<K>>(&mut self, q: Q, new: V, f: impl FnOnce(V, V) -> V) -> &V
    where
        KS: KeyReadout<K>,
    {
        let coord = q.coord();
        if is_sentinel(coord) {
            if let Some(i) = self.side_find_locate(coord, &q) {
                // SAFETY: every side entry's value is initialized; it is moved out once, and the
                // guard removes the entry if `f` unwinds (as `insert_with` does).
                let old = unsafe { self.side.vals.get_unchecked_mut(i).assume_init_read() };
                let guard = MergeGuard {
                    t: self,
                    store: Store::Side,
                    slot: i,
                    node: 0,
                    entry: 0,
                    coord,
                };
                guard.finish(f(new, old));
                // SAFETY: just written ⇒ initialized.
                return unsafe { self.side.vals.get_unchecked(i).assume_init_ref() };
            }
            self.side.push(coord, q.absorb(), new);
            let i = self.side.vals.len() - 1;
            // SAFETY: just pushed ⇒ initialized.
            return unsafe { self.side.vals.get_unchecked(i).assume_init_ref() };
        }
        self.grow_for_one();
        match self.probe_capped_locate(coord, &q) {
            Capped::Hit(i) => {
                // SAFETY: found ⇒ value initialized. Move it out and mark the slot `TOMB` for the
                // duration of `f` (never `EMPTY`: that would cut other keys' chains); the guard
                // removes the key if `f` unwinds, as in `insert_with`.
                let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
                unsafe { *self.coords.get_unchecked_mut(i) = TOMB };
                let guard = MergeGuard {
                    t: self,
                    store: Store::Main,
                    slot: i,
                    node: 0,
                    entry: 0,
                    coord,
                };
                guard.finish(f(new, old));
                // SAFETY: the slot is live again ⇒ its value is initialized.
                unsafe { self.vals.get_unchecked(i).assume_init_ref() }
            }
            Capped::Vacant(i) => {
                // Materialize the key BEFORE any slot is marked live: `q.absorb()` can run user code
                // and panic, and a live `coords[i]` over an uninitialized `keys[i]` would make
                // `Drop` drop uninitialized memory.
                let key = q.absorb();
                // SAFETY: `grow_for_one` ran, so `i` is an EMPTY slot in the current arrays.
                unsafe { *self.coords.get_unchecked_mut(i) = coord };
                unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(new) };
                // SAFETY: slot is in bounds and vacant (the probe's EMPTY slot), so nothing is dropped.
                unsafe { self.keys.set(i, key) };
                self.len += 1;
                // SAFETY: just written ⇒ initialized.
                unsafe { self.vals.get_unchecked(i).assume_init_ref() }
            }
            // The capped prefix was full: the entry lives in the overflow, upserted there.
            Capped::Blocked => self
                .overflow
                .get_or_insert_with(Overflow::new)
                .upsert(coord, q.absorb(), new, f),
        }
    }

    /// [`remove`](Self::remove) from a borrowed key.
    #[inline]
    pub fn remove_at<Q: Locate<K>>(&mut self, q: Q) -> Option<V>
    where
        KS: KeyReadout<K>,
    {
        let coord = q.coord();
        match self.locate_at(&q)? {
            Located::Side(i) => {
                // SAFETY: a located side entry is occupied ⇒ key and value initialized; `take_entry`
                // moves both out and swap-removes the entry (the `len()` total drops with the store).
                let (_c, k, v) = self.side.take_entry(i);
                drop(k);
                Some(v)
            }
            Located::Overflow(n, e) => {
                let (_k, v) = self
                    .overflow
                    .as_mut()
                    .expect("located overflow entry ⇒ store present")
                    .take_entry(n, e);
                drop(_k);
                Some(v)
            }
            Located::Main(i) => {
                // SAFETY: found ⇒ value initialized; see `Table::remove` for the tombstone argument.
                // Commit the slot's state before the key is dropped: a panicking `K::drop` must not
                // leave a live slot whose value was moved out.
                let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
                unsafe { *self.coords.get_unchecked_mut(i) = TOMB };
                self.len -= 1;
                self.tombs += 1;
                // SAFETY: slot is live (its coordinate is checked above); the caller stops using it afterwards.
                drop(unsafe { self.keys.take_key(i, coord) });
                Some(old)
            }
        }
    }

    /// Remove `k` and return both the key and the value.
    #[inline]
    pub fn remove_entry(&mut self, k: &K) -> Option<(K, V)> {
        self.entry_ref(k).map(OccupiedEntry::remove_entry)
    }

    /// The stored key and its value. `Regen` regenerates the key from the address.
    #[inline]
    pub fn get_key_value(&self, k: &K) -> Option<(KS::Key<'_>, &V)>
    where
        KS: KeyReadout<K>,
    {
        let coord = Self::coord_of(k);
        if is_sentinel(coord) {
            let i = self.side.find(coord, k)?;
            // SAFETY: `find` returns an occupied side entry ⇒ key and value initialized.
            return Some((
                // SAFETY: slot is live (the probe matched a live coordinate here).
                unsafe { self.side.keys.key(i, coord) },
                unsafe { self.side.vals.get_unchecked(i).assume_init_ref() },
            ));
        }
        match self.probe_capped_keyed(coord, k) {
            Capped::Hit(i) => {
                // SAFETY: found ⇒ value initialized.
                Some((unsafe { self.keys.key(i, coord) }, unsafe {
                    self.vals.get_unchecked(i).assume_init_ref()
                }))
            }
            Capped::Vacant(_) => None,
            Capped::Blocked => {
                let (n, e) = self
                    .overflow
                    .as_ref()
                    .and_then(|o| o.find_entry(coord, k))?;
                let ov = self.overflow.as_ref().expect("found ⇒ store present");
                let entry = &ov.nodes[n as usize].entries[e as usize];
                Some((
                    self.keys.key_of(&entry.0),
                    entry.1.as_ref().expect("overflow entry ⇒ live value"),
                ))
            }
        }
    }

    /// The occupied entry for `k`, if present — the read side of [`entry`](Self::entry) that never
    /// grows the table (no key is being added).
    #[inline]
    fn entry_ref(&mut self, k: &K) -> Option<OccupiedEntry<'_, K, V, KS>> {
        let coord = Self::coord_of(k);
        if is_sentinel(coord) {
            let slot = self.side.find(coord, k)?;
            return Some(OccupiedEntry {
                t: self,
                slot,
                node: 0,
                entry: 0,
                coord,
                store: Store::Side,
            });
        }
        match self.probe_capped_keyed(coord, k) {
            Capped::Hit(slot) => Some(OccupiedEntry {
                t: self,
                slot,
                node: 0,
                entry: 0,
                coord,
                store: Store::Main,
            }),
            Capped::Vacant(_) => None,
            Capped::Blocked => {
                let (node, entry) = self
                    .overflow
                    .as_ref()
                    .and_then(|o| o.find_entry(coord, k))?;
                Some(OccupiedEntry {
                    t: self,
                    slot: 0,
                    node,
                    entry,
                    coord,
                    store: Store::Overflow,
                })
            }
        }
    }

    /// Iterate the live keys.
    pub fn keys(&self) -> impl Iterator<Item = KS::Key<'_>>
    where
        KS: KeyReadout<K>,
    {
        Iter::new(self).map(|(k, _)| k)
    }

    /// Iterate the live values.
    pub fn values(&self) -> impl Iterator<Item = &V>
    where
        KS: KeyReadout<K>,
    {
        Iter::new(self).map(|(_, v)| v)
    }

    /// Iterate the live values mutably — main slots, side entries, then overflow entries.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        let Table {
            coords,
            vals,
            side,
            overflow,
            ..
        } = self;
        let SideStore { vals: s_vals, .. } = side;
        let main = vals
            .iter_mut()
            .zip(coords.iter())
            .filter(|(_, c)| is_live(**c))
            // SAFETY: live coord ⇒ value initialized.
            .map(|(v, _)| unsafe { v.assume_init_mut() });
        // SAFETY: every side entry's value is initialized.
        let side_it = s_vals.iter_mut().map(|v| unsafe { v.assume_init_mut() });
        let ov_it = overflow
            .as_mut()
            .into_iter()
            .flat_map(|o| o.nodes.iter_mut())
            .filter(|n| n.alive)
            .flat_map(|n| n.entries.iter_mut())
            .filter_map(|(_, v)| v.as_mut());
        main.chain(side_it).chain(ov_it)
    }

    /// Iterate the live `(key, &mut value)` pairs — main slots, side entries, then overflow entries.
    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (KS::Key<'a>, &'a mut V)>
    where
        KS: KeyReadout<K> + 'a,
        K: 'a,
    {
        // Shared reads of every field the walk needs, mutable borrows only of the value arrays.
        let (coords, keys, s_coords, s_keys) = (
            &self.coords,
            &self.keys,
            &self.side.coords,
            &self.side.keys,
        );
        let (vals, s_vals, overflow) = (&mut self.vals, &mut self.side.vals, &mut self.overflow);
        let main = vals.iter_mut().enumerate().filter_map(move |(i, v)| {
            if is_live(coords[i]) {
                // SAFETY: live coord ⇒ value initialized.
                Some((unsafe { keys.key(i, coords[i]) }, unsafe {
                    v.assume_init_mut()
                }))
            } else {
                None
            }
        });
        // SAFETY: every side entry's key and value are initialized.
        let side_it = s_vals
            .iter_mut()
            .enumerate()
            // SAFETY: slot is live (the probe matched a live coordinate here).
            .map(move |(i, v)| {
                (unsafe { s_keys.key(i, s_coords[i]) }, unsafe {
                    v.assume_init_mut()
                })
            });
        let ov_it = overflow
            .as_mut()
            .into_iter()
            .flat_map(|o| o.nodes.iter_mut())
            .filter(|n| n.alive)
            .flat_map(|n| n.entries.iter_mut())
            .filter_map(move |(k, v)| v.as_mut().map(|v| (keys.key_of(k), v)));
        main.chain(side_it).chain(ov_it)
    }

    /// Keep only the entries `keep` returns true for. Tombstones the rest — one pass, no rebuild.
    /// Side entries are retained the same way (drop value and key, swap-remove the entry).
    pub fn retain(&mut self, mut keep: impl FnMut(KS::Key<'_>, &mut V) -> bool)
    where
        KS: KeyReadout<K>,
    {
        for i in 0..self.coords.len() {
            let c = self.coords[i];
            if !is_live(c) {
                continue;
            }
            let alive = {
                // Disjoint field borrows: the key store is read, the value array is written.
                // SAFETY: slot is live (the probe matched a live coordinate here).
                let k = unsafe { self.keys.key(i, c) };
                // SAFETY: live coord ⇒ value initialized.
                let v = unsafe { self.vals.get_unchecked_mut(i).assume_init_mut() };
                keep(k, v)
            };
            if !alive {
                // Commit `TOMB`/`len`/`tombs` BEFORE any destructor runs: a panicking `V::drop`
                // or `K::drop` then leaves the slot non-live, so nothing reads the moved-out value
                // again and the table stays consistent.
                // SAFETY: live coord ⇒ value initialized; the value is moved out exactly once.
                let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
                self.coords[i] = TOMB;
                self.len -= 1;
                self.tombs += 1;
                // SAFETY: slot is live (its coordinate is checked above); the caller stops using it afterwards.
                drop(unsafe { self.keys.take_key(i, c) });
                drop(old);
            }
        }
        // Side entries: a removal swap-removes the entry, so re-examine the index it swapped in.
        let mut i = 0;
        while i < self.side.coords.len() {
            let c = self.side.coords[i];
            let alive = {
                // SAFETY: slot is live (the probe matched a live coordinate here).
                let k = unsafe { self.side.keys.key(i, c) };
                // SAFETY: side entries are occupied ⇒ value initialized.
                let v = unsafe { self.side.vals.get_unchecked_mut(i).assume_init_mut() };
                keep(k, v)
            };
            if alive {
                i += 1;
            } else {
                // SAFETY: occupied side entry ⇒ key and value initialized; `take_entry` moves both
                // out exactly once, then the entry is gone.
                let (_c, k, v) = self.side.take_entry(i);
                drop(k);
                drop(v);
            }
        }
        // Overflow entries: retained through the same keep, keyed as the store holds them.
        let keys = &self.keys;
        if let Some(o) = self.overflow.as_mut() {
            o.retain(|k, v| keep(keys.key_of(k), v));
        }
    }

    /// Drop every entry, keeping the allocation. Each slot is committed non-live (`EMPTY`, with
    /// `len`/`tombs` adjusted) BEFORE its value's or key's destructor runs, so a destructor that
    /// panics mid-pass leaves a consistent table: the slots already visited are empty and are not
    /// dropped again, the slots not yet reached are still live and are dropped once (by a later
    /// `clear`, by `Drop`, or as the table is reused), and no value or key is dropped twice.
    pub fn clear(&mut self) {
        if self.len == 0 && self.tombs == 0 && self.side.coords.is_empty() && self.overflow.is_none()
        {
            return;
        }
        let drop_v = std::mem::needs_drop::<V>();
        let drop_k = std::mem::needs_drop::<K>();
        let mut i = 0;
        while i < self.coords.len() {
            let c = self.coords[i];
            if c != EMPTY {
                // Commit the slot non-live first: a panicking destructor below then finds a slot
                // that `Drop`/iteration skip, and the value it dropped is not reached again.
                self.coords[i] = EMPTY;
                if is_live(c) {
                    self.len -= 1;
                    if drop_v {
                        // SAFETY: live coord ⇒ value initialized; dropped exactly once, here.
                        unsafe { self.vals[i].assume_init_drop() };
                    }
                    if drop_k {
                        // The key store rides the same liveness line; its destructor may panic.
                        // SAFETY: slot is live (its coordinate is checked above); the caller stops using it afterwards.
                        drop(unsafe { self.keys.take_key(i, c) });
                    }
                } else {
                    self.tombs -= 1; // the slot was a `TOMB`, now `EMPTY`
                }
            }
            i += 1;
        }
        self.side.clear();
        self.overflow = None;
    }

    /// Make room for `additional` more keys without growing again. The same one sizing as
    /// [`Table::with_capacity`] — a no-op when the table is already big enough. Main capacity is
    /// sized from the main-entry count; side entries live outside the slot arrays and need no room.
    pub fn reserve(&mut self, additional: usize) {
        let want = bits_for(self.len + additional);
        if want > self.bits {
            self.resize(want);
        }
    }

    /// Empty the table into an iterator of its `(key, value)` pairs, keeping the table usable.
    pub fn drain(&mut self) -> IntoIter<K, V, KS> {
        std::mem::replace(self, Self::with_bits(3)).into_iter()
    }

    /// The side-store twin of [`probe_capped_locate`](Self::probe_capped_locate): the entry whose
    /// reserved coordinate is `coord` and whose stored key the view `q` names.
    #[inline]
    fn side_find_locate<Q: Locate<K>>(&self, coord: u64, q: &Q) -> Option<usize>
    where
        KS: KeyReadout<K>,
    {
        let mut i = 0;
        while i < self.side.coords.len() {
            // SAFETY: slot is live (the probe matched a live coordinate here).
            if self.side.coords[i] == coord && unsafe { self.side.keys.matches_locate(i, coord, q) } {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Locate a borrowed view: the capped keyed probe for a free coordinate, the side store for a
    /// reserved one, the overflow for an exhausted prefix. `None` when absent.
    #[inline]
    fn locate_at<Q: Locate<K>>(&self, q: &Q) -> Option<Located>
    where
        KS: KeyReadout<K>,
    {
        let coord = q.coord();
        if is_sentinel(coord) {
            return self.side_find_locate(coord, q).map(Located::Side);
        }
        match self.probe_capped_locate(coord, q) {
            Capped::Hit(i) => Some(Located::Main(i)),
            Capped::Vacant(_) => None,
            Capped::Blocked => self
                .overflow
                .as_ref()
                .and_then(|o| o.find_locate(coord, q))
                .map(|(n, e)| Located::Overflow(n, e)),
        }
    }

    /// [`probe_capped_keyed`](Table::probe_capped_keyed) over a borrowed view: a coordinate match is
    /// a hit only when the stored key is the one the view names. Only called with a non-reserved
    /// `coord`.
    #[inline]
    fn probe_capped_locate<Q: Locate<K>>(&self, coord: u64, q: &Q) -> Capped
    where
        KS: KeyReadout<K>,
    {
        let home = self.home(coord);
        // SAFETY: home ∈ [0, len).
        let sc = unsafe { *self.coords.get_unchecked(home) };
        // SAFETY: slot is live (the probe matched a live coordinate here).
        if sc == coord && unsafe { self.keys.matches_locate(home, coord, q) } {
            return Capped::Hit(home);
        }
        if sc == EMPTY {
            return Capped::Vacant(home);
        }
        self.probe_capped_with(coord, home, &ByLocate { coord, q })
    }
}

/// The exact borrowed read — only on the [`Stored`] key store, which is the only one that HAS a key
/// to compare. On [`Regen`](crate::map::hashtrinity::Regen) the coordinate round-trips to the key, so
/// [`get_at`](Table::get_at) already IS the exact read (`CoordinateReadout.exact_read_is_read`); the
/// face is retained under its historical name.
impl<K: Eq + Coord, V> Table<K, V, Stored<K>> {
    /// [`get_exact`](Table::get_exact) from a borrowed key — also compares the stored key, so a
    /// `2⁻⁶⁴` coordinate collision reads as absent instead of returning someone else's value.
    #[inline]
    pub fn get_exact_at<Q: Locate<K>>(&self, q: Q) -> Option<&V> {
        self.get_at(q)
    }
}

// ── The live-lane scan: the read side of iteration, branchless. ────────────────────────────────
//
// A drain (`for (k, v) in &map`) walks every SLOT, not every entry, and at the grow law's steady
// state half of them are empty. The obvious loop — `if is_live(coords[i])` per slot — puts a
// ~50/50 unpredictable branch on that walk, and a mispredict costs ~15 cycles where the load costs
// ~0.5. That is the whole of the measured 4.4x iteration loss to std/hashbrown, and it is NOT a
// bandwidth story: the coordinate line is 8 B/slot against SwissTable's 1 B/slot control array, a
// factor 8 in traffic that would explain 8x if traffic were the binding constraint — it explains
// almost none of it, because the scan is branch-bound, not memory-bound (measured: 2.4 ns/slot,
// ~10 cycles, where a streaming 8 B/slot read is ~0.5).
//
// So take SwissTable's ACTUAL trick, which is not the byte array — it is the movemask. Glance eight
// coordinates (one 64-byte cache line) against `EMPTY` and `TOMB` in two AVX2 compares, invert to a
// live-lane bitmask, and walk the SET BITS. One branch per eight slots instead of one per slot, and
// the empty slots cost a shift each. No new array, no control byte, no layout change, so `iter()`
// keeps its signature, its slot order, and its footprint.

/// The live-lane bitmask for the eight coordinates at `base` — bit `k` set iff slot `base + k` holds
/// a live entry. `bits >= 3` and the slot count is a power of two, so a table is always a whole
/// number of eight-slot blocks and there is no tail case.
#[inline]
fn live_mask8(coords: &[u64], base: usize) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: the caller only calls this with `base + 8 <= coords.len()`.
    return unsafe { live_mask8_avx2(coords.as_ptr().add(base)) };

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: the caller only calls this with `base + 8 <= coords.len()`.
            return unsafe { live_mask8_avx2(coords.as_ptr().add(base)) };
        }
        let mut m = 0u32;
        for k in 0..8 {
            // branchless by construction — the compiler autovectorizes this shape
            m |= u32::from(is_live(coords[base + k])) << k;
        }
        m
    }
}

/// AVX2 body of [`live_mask8`]: two 256-bit loads (one 64-byte cache line), and ONE `vpcmpgtq` each.
///
/// `is_live(c)` is `c < TOMB` (the sentinels are the top two `u64`s), so a lane is live iff its
/// coordinate is *below* `TOMB`. AVX2 has no unsigned 64-bit compare, so flip the sign bit — an
/// order-preserving `u64 → i64` map — and compare signed against `TOMB` flipped: live ⟺
/// `flip(TOMB) > flip(c)`. Two vector ops per four slots (`vpxor`, `vpcmpgtq`) where the
/// equality form needed four (`vpcmpeqq` ×2, `vpor`, and the final invert).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn live_mask8_avx2(p: *const u64) -> u32 {
    use std::arch::x86_64::{
        __m256i, _mm256_castsi256_pd, _mm256_cmpgt_epi64, _mm256_loadu_si256, _mm256_movemask_pd,
        _mm256_set1_epi64x, _mm256_xor_si256,
    };
    // SAFETY: caller guarantees 8 readable lanes at `p`.
    let (lo, hi) = unsafe {
        (
            _mm256_loadu_si256(p.cast::<__m256i>()),
            _mm256_loadu_si256(p.add(4).cast::<__m256i>()),
        )
    };
    let sign = _mm256_set1_epi64x(i64::MIN);
    let bound = _mm256_set1_epi64x((TOMB ^ (1u64 << 63)) as i64); // flip(TOMB)
    let live = |v: __m256i| -> u32 {
        let f = _mm256_xor_si256(v, sign);
        _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(bound, f))) as u32
    };
    live(lo) | (live(hi) << 4)
}

/// The eight-slot cursor: `base` is the block whose `mask` is loaded, `next` the block after it.
#[derive(Clone, Copy)]
struct Lanes {
    base: usize,
    next: usize,
    mask: u32,
}

impl Lanes {
    const START: Lanes = Lanes {
        base: 0,
        next: 0,
        mask: 0,
    };
    /// The next live slot index, or `None` at the end. One branch per eight slots.
    #[inline(always)]
    fn step(&mut self, coords: &[u64]) -> Option<usize> {
        while self.mask == 0 {
            if self.next >= coords.len() {
                return None;
            }
            self.base = self.next;
            self.next += 8;
            self.mask = live_mask8(coords, self.base);
        }
        let k = self.mask.trailing_zeros() as usize;
        self.mask &= self.mask - 1; // clear the lowest set bit
        Some(self.base + k)
    }
    /// How many live slots are left at or after the cursor — for `size_hint`/`Drop`.
    fn remaining(&self, coords: &[u64]) -> usize {
        let mut c = self.mask.count_ones() as usize;
        let mut b = self.next;
        while b < coords.len() {
            c += live_mask8(coords, b).count_ones() as usize;
            b += 8;
        }
        c
    }
}

// ── Iteration: `for (k, v) in &t`, `for (k, v) in t`, `t.extend(..)`. ──────────────────────────

/// `(key, &value)` over the live slots — the `IntoIterator` form of [`Table::iter`], on the
/// eight-slot live-lane glance. The main slots come first, then the side entries, then the overflow
/// entries (the in-order node list is collected once at construction; empty when no key was blocked).
pub struct Iter<'a, K: Coord + Eq, V, KS: KeyStore<K>> {
    t: &'a Table<K, V, KS>,
    lanes: Lanes,
    side: usize,
    ov_nodes: Vec<u32>,
    ov_node: usize,
    ov_entry: usize,
    seen: usize,
}

impl<'a, K: Coord + Eq, V, KS: KeyStore<K>> Iter<'a, K, V, KS> {
    /// The walker for `t` — used by [`Table::iter`] itself.
    #[inline]
    pub(crate) fn new(t: &'a Table<K, V, KS>) -> Self {
        let mut ov_nodes = Vec::new();
        if let Some(o) = &t.overflow {
            o.in_order(&mut ov_nodes);
        }
        Iter {
            t,
            lanes: Lanes::START,
            side: 0,
            ov_nodes,
            ov_node: 0,
            ov_entry: 0,
            seen: 0,
        }
    }
}

impl<'a, K: Coord + Eq, V, KS: KeyReadout<K>> Iterator for Iter<'a, K, V, KS>
where
    KS: 'a,
    K: 'a,
{
    type Item = (KS::Key<'a>, &'a V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let t: &'a Table<K, V, KS> = self.t; // copy the borrow so the yielded key lives for 'a
        if let Some(i) = self.lanes.step(&t.coords) {
            self.seen += 1;
            // SAFETY: `step` only yields in-bounds live lanes ⇒ this slot's value (and, for
            // `Stored`, its key) is initialized, and the index needs no bounds check.
            let c = unsafe { *t.coords.get_unchecked(i) };
            // SAFETY: slot is live (the probe matched a live coordinate here).
            return Some((unsafe { t.keys.key(i, c) }, unsafe {
                t.vals.get_unchecked(i).assume_init_ref()
            }));
        }
        if self.side < t.side.coords.len() {
            let i = self.side;
            self.side += 1;
            self.seen += 1;
            let c = t.side.coords[i];
            // SAFETY: every side entry's key and value are initialized.
            return Some((unsafe { t.side.keys.key(i, c) }, unsafe {
                t.side.vals.get_unchecked(i).assume_init_ref()
            }));
        }
        let ov = t.overflow.as_ref()?;
        while self.ov_node < self.ov_nodes.len() {
            let node = &ov.nodes[self.ov_nodes[self.ov_node] as usize];
            while self.ov_entry < node.entries.len() {
                let e = self.ov_entry;
                self.ov_entry += 1;
                if let Some(v) = node.entries[e].1.as_ref() {
                    self.seen += 1;
                    return Some((t.keys.key_of(&node.entries[e].0), v));
                }
            }
            self.ov_node += 1;
            self.ov_entry = 0;
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.t.len() - self.seen;
        (left, Some(left))
    }
}

impl<K: Coord + Eq, V, KS: KeyReadout<K>> ExactSizeIterator for Iter<'_, K, V, KS> {}

impl<'a, K: Coord + Eq, V, KS: KeyReadout<K> + 'a> IntoIterator for &'a Table<K, V, KS>
where
    K: 'a,
{
    type Item = (KS::Key<'a>, &'a V);
    type IntoIter = Iter<'a, K, V, KS>;
    fn into_iter(self) -> Iter<'a, K, V, KS> {
        Iter::new(self)
    }
}

/// Owned `(key, value)` pairs — [`Table::drain`] and `for (k, v) in table`. The side entries and the
/// overflow entries are moved out eagerly into `side`/`overflow` (small stores; ordinary tables have
/// no overflow).
pub struct IntoIter<K: Coord + Eq, V, KS: KeyStore<K>> {
    coords: Box<[u64]>,
    vals: Box<[MaybeUninit<V>]>,
    keys: KS,
    lanes: Lanes,
    side: std::vec::IntoIter<(K, V)>,
    overflow: std::vec::IntoIter<(K, V)>,
    _k: std::marker::PhantomData<K>,
}

impl<K: Coord + Eq, V, KS: KeyStore<K>> Iterator for IntoIter<K, V, KS> {
    type Item = (K, V);
    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        if let Some(i) = self.lanes.step(&self.coords) {
            // SAFETY: `step` only yields in-bounds live lanes.
            let c = unsafe { *self.coords.get_unchecked(i) };
            // SAFETY: a live lane ⇒ value and (for Stored) key initialized. Both are moved out
            // exactly once, then the slot is blanked so this iterator's Drop skips it.
            let v = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
            // SAFETY: slot is live (its coordinate is checked above); the caller stops using it afterwards.
            let k = unsafe { self.keys.take_key(i, c) };
            self.coords[i] = EMPTY;
            return Some((k, v));
        }
        if let Some(kv) = self.side.next() {
            return Some(kv);
        }
        self.overflow.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.lanes.remaining(&self.coords) + self.side.len() + self.overflow.len();
        (left, Some(left))
    }
}

impl<K: Coord + Eq, V, KS: KeyStore<K>> Drop for IntoIter<K, V, KS> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<V>() {
            for (c, v) in self.coords.iter().zip(self.vals.iter_mut()) {
                if is_live(*c) {
                    // SAFETY: still-live coord ⇒ never yielded ⇒ value initialized.
                    unsafe { v.assume_init_drop() };
                }
            }
        }
        let coords = std::mem::take(&mut self.coords);
        // SAFETY: slot is live (its coordinate is checked above); the caller stops using it afterwards.
        unsafe { self.keys.drop_live(&coords) };
        // `self.side` / `self.overflow` are Vec iterators; their remaining `(K, V)` pairs drop
        // themselves.
    }
}

impl<K: Coord + Eq, V, KS: KeyStore<K>> IntoIterator for Table<K, V, KS> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, KS>;
    fn into_iter(self) -> IntoIter<K, V, KS> {
        // Move the slot arrays out, leaving `t` an empty table (no slots, so its own Drop walks
        // nothing) that still owns — and frees — the side store's and overflow's buffers.
        let mut t = self;
        let coords = std::mem::take(&mut t.coords);
        let vals = std::mem::replace(&mut t.vals, Box::new([]));
        let keys = std::mem::replace(&mut t.keys, KS::alloc(0));
        // Move the side entries out eagerly (a small store; ordinary tables have none). Each
        // `take_entry` moves one key and value out and swap-removes it, so nothing double-drops.
        let mut side: Vec<(K, V)> = Vec::with_capacity(t.side.coords.len());
        while let Some(last) = t.side.coords.len().checked_sub(1) {
            let (_c, k, v) = t.side.take_entry(last);
            side.push((k, v));
        }
        // Move the overflow entries out eagerly too; the iterator now owns the keys and values.
        let overflow: Vec<(K, V)> = match t.overflow.take() {
            Some(ov) => ov
                .into_entries()
                .into_iter()
                .map(|(_c, k, v)| (k, v))
                .collect(),
            None => Vec::new(),
        };
        IntoIter {
            coords,
            vals,
            keys,
            lanes: Lanes::START,
            side: side.into_iter(),
            overflow: overflow.into_iter(),
            _k: std::marker::PhantomData,
        }
    }
}

impl<K: Coord + Eq, V, KS: KeyStore<K>> Extend<(K, V)> for Table<K, V, KS> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        let it = iter.into_iter();
        self.reserve(it.size_hint().0);
        for (k, v) in it {
            self.put(k, v);
        }
    }
}

impl<'x, K: Coord + Eq + Clone, V: Clone, KS: KeyStore<K>> Extend<(&'x K, &'x V)>
    for Table<K, V, KS>
{
    fn extend<I: IntoIterator<Item = (&'x K, &'x V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.put(k.clone(), v.clone());
        }
    }
}

impl<Q: Locate<K>, K: Coord + Eq, V, KS: KeyReadout<K>> std::ops::Index<Q> for Table<K, V, KS> {
    type Output = V;
    /// # Panics
    /// If the key is absent — std `HashMap`'s `Index` contract.
    fn index(&self, q: Q) -> &V {
        self.get_at(q).expect("no entry found for key")
    }
}

/// Confirmation against a borrowed [`Locate`] view: [`KeyReadout::matches_locate`].
struct ByLocate<'a, Q> {
    coord: u64,
    q: &'a Q,
}

impl<'a, K: Eq + Coord, KS: KeyReadout<K>, Q: Locate<K>> super::overflow::Confirm<K, KS>
    for ByLocate<'a, Q>
{
    #[inline]
    fn confirm(&self, keys: &KS, i: usize) -> bool {
        // SAFETY: slot is live (the probe matched a live coordinate here).
        unsafe { keys.matches_locate(i, self.coord, self.q) }
    }
}
