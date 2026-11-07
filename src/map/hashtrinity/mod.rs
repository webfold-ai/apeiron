// Only the table core, its extraction reach and webfold's three names remain.
//! hashtrinity — a `K → V` table where the key is its own address.
//!
//! Not a wrapper over `std::HashMap`. One flat power-of-two array, one insert, and four merge
//! functions that pick what kind of table you get. No random seed, so the layout is deterministic.
//!
//! # One structure, four merges
//!
//! [`Table::insert_with`] is the only mutator: it merges the new value with whatever was already
//! there. The merge function is the *only* difference between a set, a map, a counter, and a graph:
//!
//! | table | methods                     | merge      | absent reads as |
//! |-------|-----------------------------|------------|-----------------|
//! | Set   | [`Table::mark`]/`has`       | `\|\|`     | `false`         |
//! | Map   | [`Table::put`]/`get`        | last write | `None`          |
//! | Count | [`Table::add`]/`count`      | `+`        | `0`             |
//! | Graph | [`Table::push`]/`neighbors` | append     | `[]`            |
//!
//! An absent key reads as the merge's identity, not a sentinel. Commutative merges (Set, Count) are
//! order-free, so their tables can be split across threads and recombined ([`Table::fuse`]); the
//! directional merges (Map, Graph) cannot.
//!
//! [`Table`] keeps the key (exact). Two keyless variants trade that for speed and footprint:
//! [`CoordMap`] for coordinate keys (the address regenerates the key), and [`FreqMap`] for opaque
//! keys it routes by fold (`2⁻⁶⁴` false-positive on non-members) — dedup / count / index.
//!
//! # Why it's fast
//!
//! The key is its own address when it can be. An integer key smaller than the array indexes directly
//! — no hash, no collision, one probe. Larger or byte keys fold to a 64-bit coordinate via `k·PHI`
//! (Fibonacci hashing) and take the top `bits` as the home slot; a collision re-addresses by the next
//! window of `k·PHI`. Multiply-by-odd is a bijection on 2⁶⁴, so distinct keys always separate within
//! `⌈64/bits⌉` steps — no two keys can alias forever. Coordinate keys are HashDOS-immune by
//! construction (`home(c) = c`: the only key that can occupy slot `s` is `s`). On the FOLD path the
//! immunity is measured, not proved: separation bounds aliasing, not probe length, and
//! `AggregationBound.descent_can_be_wholly_blocked` refutes the worst-case reading — `PHI` is public,
//! so `PHI⁻¹` aims one crafted key at each slot a victim would try. `tests/hashdos.rs` is the red
//! team and carries the numbers. No SipHash tax.
//!
//! # Which face, and when — see `FACES.md`
//!
//! Two choices, every time: the **keystore** and the **face**. **Bulk work is parallel by default.**
//!
//! **Map (the trinity `x = B/U`), by what you need from a key:**
//!
//! | map | face | when | confirm |
//! |-----|------|------|---------|
//! | [`FreqMap`] | **NB** | join / count / dedup / cache / index — the analytical work | trusts the fold (`2^-64` false-positive) — the group probe |
//! | [`CoordMap`] | **B** | dense *coordinate* keys you must resolve EXACTLY, worst-probe-1 | the address round-trips (regenerates the key) |
//! | [`Table`] `Stored` | **B/U** | opaque keys you must resolve exactly OR read back | stores + compares the key |
//!
//! Default to **`FreqMap`** for the analytical faces (GROUP BY, JOIN, DISTINCT, counters): it is the
//! proven group-probe engine (`GroupProbe.lean` — uniform fold, 16-wide tag glance miss filter, bounded
//! terminal sweep), and `2^-64` is "never" on those workloads. Reach for `CoordMap`/`Table` only when a
//! `2^-64` false positive is unacceptable or you must recover the key.
//!
//! **Operation shape** (on any of them):
//!
//! | you have          | you want                | face |
//! |-------------------|-------------------------|------|
//! | one key           | its value               | [`get`](Table::get) / [`FreqMap::get`] |
//! | a stream of keys  | each value (visit)      | [`get_each`](Table::get_each) (pipelined) |
//! | a stream of rows  | a per-key table (GROUP BY) | [`FreqMap::count_par`] (two-level parallel, group probe) |
//! | one row at a time | update a key            | [`add`](Table::add) / [`FreqMap::add`] |
//!
//! **Anti-patterns** (each produced a false "we lose"): a serial `for k { add(k,1) }` (or the [`FoldExt`] builders) as a *bulk* GROUP BY —
//! use [`FreqMap::count_par`]; `CoordMap` for a miss-heavy join — use [`FreqMap`] (its uniform fold has
//! no identity-home block); benchmarking any bulk pass single-threaded. If you're about to report a loss,
//! you have probably picked the wrong face — check the parallel group-probe one first. Full map: `FACES.md`.
//!
//! ```ignore
//! use apeiron::map::hashtrinity::FoldExt;
//! let freq = ["a", "b", "a"].iter().copied().count_by(|w| w);
//! assert_eq!(freq.count(&"a"), 2);
//! ```

#![allow(missing_docs)] // the port is public now; its items keep the upstream docs as written
#![warn(clippy::pedantic)]

// ---------------------------------------------------------------------------
// Host-policy allowance (apeiron). Upstream's own `#![allow]` list is below and is left byte-
// identical to rev `4cc14471`; this block is separate because these five entries exist only to
// reconcile the ported tree with *apeiron's* crate-level policy, which upstream never saw.
//
// `clippy::pedantic` + `clippy::style` — apeiron raises pedantic crate-wide through
// `[lints.clippy]` in `Cargo.toml`, so it reaches this tree too; `style` is on by default in every
// build. Upstream instead tuned against its own 18-entry list. Measured on clippy 0.1.96 against
// this tree, the host's two groups account for 38 distinct lints (28 pedantic + 10 style:
// `explicit_iter_loop` … `items_after_test_module`). Suppressing the two groups rather than 38
// names is what keeps a future upstream re-sync from re-deriving this. `dead_code` and
// `unused_imports` are separate because they are rustc lints, not clippy groups.
// `manual_is_multiple_of` and `manual_memcpy` are named one by one because they are in neither
// group — `manual_is_multiple_of` is `complexity`, and allowing that whole group for two hits is a
// wider hammer than naming them. Measured: 6 and 2 sites respectively.
// `dead_code`/`unused_imports` — this is a whole upstream crate, imported whole at a pinned
// revision so the count face can be read against its real source. Only `FreqMap` is wired
// (`src/map/freq.rs`); the rest is unreachable from this crate by construction, not by oversight, and
// deleting it would break the re-sync identity. Scoped to this module: it silences nothing outside
// `src/map/hashtrinity/`.
// `clippy::allow_attributes_without_reason` — apeiron requires a reason on every `#[allow]`. The 19
// bare `#[allow]`s here are upstream's, in files this task does not edit; requiring a reason would
// mean editing ported upstream source, so the lint is silenced for the subtree instead.
// ---------------------------------------------------------------------------
#![allow(
    clippy::pedantic,
    clippy::style,
    clippy::manual_is_multiple_of,
    clippy::manual_memcpy,
    dead_code,
    unused_imports,
    clippy::allow_attributes_without_reason
)]

// The truncating/wrapping casts are the mechanism, not a bug: a fold *is* a `u64`→slot truncation and
// the coordinate math wraps mod 2⁶⁴ on purpose. The doc/style lints below are noise for this crate.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::must_use_candidate,
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::similar_names,
    clippy::unreadable_literal,
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::inline_always,
    clippy::wildcard_imports,
    clippy::many_single_char_names,
    clippy::cast_ptr_alignment
)]

use std::mem::MaybeUninit;

/// The `hex` masquerade — encode/decode hex strings, the same public API as the real crate.
/// Zero dependencies; see the module doc for what's ported.
#[allow(missing_docs)]
pub mod hex;

/// The `base64` masquerade — the same `Engine` API as the real crate. Zero dependencies; see the
/// module doc for what's ported.
#[allow(missing_docs)]
pub mod base64;

pub mod seed;
pub mod tregex;
pub use seed::{CoordSeed, PHI_INV};

// The ML primitives (emit·carrier·fold — simt/simt_nn/gpu, gptq/seal, foldtrain/foldnet/ridgefold/
// finetune/countlm) and the serving surface were carved out of this core into the `modelfold` and
// `servefold` crates. hashtrinity is the K→V Table and its faces, nothing model-specific.

pub mod par;

/// The std-shaped faces — `entry`/`get_mut`/`insert`/`retain`/iteration, and borrowed-key probes
/// ([`Locate`]) so `&str` reads a `Table<String, _>` and `(&str, &str)` reads a
/// `Table<(String, String), _>` with no key materialized. Purely additive; see [`face`].
pub mod face;
pub use face::{Absorb, Entry, IntoIter, Iter, Locate, OccupiedEntry, VacantEntry};

// ═══════════════════════════════════════════════════════════════════
// SUBSUMED — hashtrinity-crypto is now a module of the one crate:
//   hashtrinity-crypto   → crypto    (the standard hashes as the fold)
pub mod crypto;

// ── The two files this root was split into. ────────────────────────────────────────────────────
//
// PRIVATE modules, glob-re-exported, so every path stays exactly where it was: `Coord`, `Monoid`,
// `FreqMap` and friends are `hashtrinity::Coord`/`Monoid`/`FreqMap` as before, and no NEW public
// module path (`hashtrinity::algebra::…`) is created. `Table` itself, the sentinels, `is_live`,
// `prefix_masks` and `bits_for` stay HERE: `face` reaches them as `crate::map::hashtrinity::is_live` etc.,
// which is legal only because they are private items of the crate ROOT and those modules are its
// descendants. Moving them down one level would force `pub(crate)` on all of them.
//
// algebra — the trait vocabulary `Table<K, V, KS>` is parameterized by: how a key becomes an
//   address (`Coord`/`CoordInv`/`KeyStore`/`Stored`/`Regen`, the golden-128 byte fold) and how two
//   values merge (`Face`/`Species`/`keep`·`set`·`count`·`link`/`Monoid`/`FoldExt`).
// freq — the NB face: `FreqMap` and its `FreqLine` group-probe engine, a whole separate structure
//   (SwissTable control bytes, no key column) that shares nothing with `Table`'s probe.
mod side;
use side::SideStore;
mod overflow;
use overflow::Overflow;
mod reduce;
mod algebra;
mod freq;
pub use algebra::*;
pub use freq::*;

/// Golden ratio · 2⁶⁴ — the Fibonacci-hashing multiplier. Odd, so `c ↦ c·PHI` is a bijection on 2⁶⁴
/// and loses nothing: splitting `c·PHI` into (top `bits` = home slot | low bits = residual) is
/// reversible. Public so the [`seed`] codec can store just the residual and regrow the coordinate.
pub const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

// ── The table: coordinates + values + (for opaque keys) a tag line and a key store. ────────────

/// The empty-slot marker. A live coordinate never equals it: a key whose own coordinate is `EMPTY`
/// (or `TOMB`) is held out of band in the [`SideStore`], so `coord == EMPTY` unambiguously means
/// "empty".
const EMPTY: u64 = u64::MAX;

/// Linear-prefix width of the probe ([`Table::descend`]): 8 `u64` coordinates = one 64-byte cache
/// line. The collision walk's first `LINEAR_G` steps are contiguous (`home + j`), so a miss or short
/// chain touches one coordinate line instead of one scattered line per radix step
/// (`RadixDescent.linear_prefix_chain_separates`).
const LINEAR_G: u32 = 8;

// The AVX2 prefix scan below consumes the `LINEAR_G` lanes as two 256-bit (4×u64) loads; if the
// prefix width ever changes, the scan must change with it, so pin the coupling at compile time.
const _: () = assert!(LINEAR_G == 8, "prefix_masks scans exactly 8 lanes");

/// The primary probe cap: a lookup, insert, update or remove examines at most this many primary
/// slots (home, the linear prefix, then the fresh radix windows, in the order
/// [`Table::descend`] gives). Anything unresolved — the whole capped prefix occupied with no
/// identity — routes to the table-wide [`Overflow`] tree instead of the removed terminal sweep.
/// One named constant so the bench can tune it.
pub const PROBE_CAP: u32 = 16;

/// Growth threshold as a fraction `LOAD_NUM / LOAD_DEN`: an insert grows (or, with tombstones and
/// `len + 1 ≤ capacity / 2`, compacts) when `(len + tombs + 1) · LOAD_DEN ≥ capacity · LOAD_NUM`
/// ([`Table::grow_for_one`]), and
/// [`bits_for`] sizes a `with_capacity(n)` table so `n` keys stay under the same threshold. Named
/// so the load factor is one edit and the bench can sweep it.
const LOAD_NUM: usize = 3;
const LOAD_DEN: usize = 4;

/// The capped primary probe's verdict — the three outcomes a probe can have without walking the
/// whole slot space.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Capped {
    /// The key's coordinate matched at this slot and the caller's confirmation held.
    Hit(usize),
    /// An `EMPTY` slot was reached — the insertion point for an absent key.
    Vacant(usize),
    /// The whole capped prefix was occupied: the key may live in the [`Overflow`].
    Blocked,
}

/// Which store an entry lives in — the primary slot array, the reserved-coordinate side store, or
/// the overflow tree.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Store {
    /// A main `coords`/`vals` slot.
    Main,
    /// A [`SideStore`] entry (a key whose own coordinate is `EMPTY`/`TOMB`).
    Side,
    /// An [`Overflow`] entry, addressed by `(node, entry)`.
    Overflow,
}

/// Deleted-slot marker (a tombstone), distinct from `EMPTY`. A removed slot's coordinate is set to
/// `TOMB`: it is not a live coordinate (a key whose own coordinate is `EMPTY` or `TOMB` is held out
/// of band in the [`SideStore`]) and not `EMPTY`, so every probe walks *through* it (chains stay
/// intact — see `probe`) while a resize reclaims it. Its value is uninitialized, exactly like an
/// `EMPTY` slot — the liveness invariant below excludes it. There is no tag line for a tombstone to
/// mark (see [`Table`]'s doc): the coordinate alone carries it.
const TOMB: u64 = EMPTY - 1;

/// A slot holds an initialized value iff its coordinate is neither `EMPTY` nor `TOMB`. This is the one
/// invariant every `unsafe` value read rests on; with deletions it replaces the old `!= EMPTY` test.
#[inline]
fn is_live(c: u64) -> bool {
    c != EMPTY && c != TOMB
}

/// Whether `c` is one of the two coordinates reserved as occupancy markers. Such a coordinate is
/// never written into [`Table`]'s `coords`; the key that owns it is held in the [`SideStore`].
#[inline]
fn is_sentinel(c: u64) -> bool {
    c == EMPTY || c == TOMB
}

/// `K -> V`. The slot's coordinate is the identity, the filter, and the occupancy flag at once:
/// `coord == wanted` is a hit (no key deref, no string compare) and `coord == EMPTY` is empty.
/// Values live in a parallel array so a probe can scan a tight coordinate line without dragging
/// payloads through cache; the key (when stored) lives in a third array, off the hot path.
///
/// No `Option` anywhere — the coordinate already says whether the slot is live, so each slot is
/// 8 bytes of coordinate, and value/key are touched only after a match. `Drop`/`Clone` walk the
/// live slots by hand. All the `unsafe` rests on ONE invariant, upheld at every write:
/// **`vals[i]` (and, for `Stored`, `keys[i]`) is initialized iff `coords[i]` is live**
/// ([`is_live`] — neither `EMPTY` nor a [`remove`](Self::remove)d slot's `TOMB` tombstone).
///
/// There is no tag line. The lossy 7-bit tag (SwissTable h2) is argued against by construction
/// (`CoordinateReadout.readout_beats_compare_by_construction`: a non-injective fingerprint cannot
/// confirm a hit; the full coordinate can, confirmed by the stored key at one compare) and its one
/// measured win — the opaque-key miss stream — is not a workload the framework runs. The probe is
/// the pure radix descent; its first two steps are the spinor readout (see [`Table::get`]).
///
/// The reserved coordinates (`EMPTY`/`TOMB`) never appear in `coords`: keys that own them live in
/// `side`, where every entry's value is initialized (a merge moves it out only under a
/// [`MergeGuard`](side::MergeGuard), which removes the entry if the merge unwinds).
///
/// # The overflow invariant (HashDOS cap)
///
/// The primary probe is capped at [`PROBE_CAP`] slot examinations ([`Self::probe_limit`]); the
/// terminal sweep is removed. A key whose capped prefix held no `EMPTY` and no identity is stored
/// in `overflow`, a deterministic AVL tree keyed by coordinate. The invariant is: **a key is in the
/// overflow only if, when it was placed, its `PROBE_CAP` primary slots held no `EMPTY` slot and no
/// slot with its identity.** It is preserved because a primary slot only ever changes from `EMPTY`
/// to occupied — `remove` leaves a `TOMB` (not `EMPTY`), and `resize` re-places every entry,
/// primary and overflow alike, through this same capped placement — so the absence of an `EMPTY` in
/// the prefix is stable. Therefore a lookup that finds an `EMPTY` in the capped prefix is absent
/// without consulting the overflow; a hit in the prefix is found there; only an exhausted prefix
/// consults `overflow`. `overflow` is `None` until the first blocked key, so a table that never
/// overflows pays one `is_none()` branch on a capped miss and nothing on a hit.
///
/// The HashDOS claim covers keys whose coordinate is injective — `u64` and the other [`CoordInv`]
/// types. For an arbitrary `Eq + Coord` key whose coordinate is not injective, the overflow still
/// stores entries correctly (equal coordinates share a node, compared by full key), but the
/// flood bound is a statement about the coordinate, not about the key.
pub struct Table<K, V, KS = Stored<K>>
where
    K: Coord,
    KS: KeyStore<K>,
{
    coords: Box<[u64]>, // identity + occupancy; coord == EMPTY ⇒ empty, value uninit
    vals: Box<[MaybeUninit<V>]>, // payloads; initialized exactly where coords[i] != EMPTY
    keys: KS,           // the identity store — a vector (Stored) or nothing (Regen)
    side: SideStore<K, V, KS>, // keys whose coordinate is a reserved marker (EMPTY/TOMB)
    overflow: Option<Overflow<K, V>>, // blocked keys, one table-wide AVL tree, lazily allocated
    bits: u32,
    len: usize,   // live PRIMARY main-slot entries (excludes tombstones and the overflow)
    tombs: usize, // deleted-but-not-yet-reclaimed slots; cleared on resize
    // `Some` ⇒ every coordinate is the keyed fold ([`Coord::coord_keyed`]); set by `with_seed`.
    seed: Option<FoldKey>,
    _k: std::marker::PhantomData<K>,
}

/// Values AND stored keys are `MaybeUninit`, so drop exactly the live ones (`coord != EMPTY`).
/// No-op for types without destructors, the common integer case.
impl<K: Coord, V, KS: KeyStore<K>> Drop for Table<K, V, KS> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<V>() {
            let mut i = 0;
            while i < self.coords.len() {
                if is_live(self.coords[i]) {
                    // SAFETY: the invariant — live coord ⇒ value initialized.
                    unsafe { self.vals[i].assume_init_drop() };
                }
                i += 1;
            }
        }
        // the key store rides the same invariant; internally gated on needs_drop::<K>
        let coords = std::mem::take(&mut self.coords);
        // SAFETY: slot is live (its coordinate is checked above); the caller stops using it afterwards.
        unsafe { self.keys.drop_live(&coords) };
        // Side entries: `coords` holds reserved markers (never `is_live`), so every entry's key and
        // value is dropped by the side store itself.
        self.side.clear();
    }
}

/// Copy live values and keys through `Clone`, leave empty slots uninitialized.
impl<K: Coord + Clone, V: Clone, KS: KeyStore<K>> Clone for Table<K, V, KS> {
    fn clone(&self) -> Self {
        let coords = self.coords.clone();
        let vals = self
            .coords
            .iter()
            .zip(self.vals.iter())
            .map(|(c, v)| {
                if is_live(*c) {
                    // SAFETY: the invariant — live coord ⇒ value initialized.
                    MaybeUninit::new(unsafe { v.assume_init_ref() }.clone())
                } else {
                    MaybeUninit::uninit()
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        // Side entries: every entry is live (their coordinates are reserved markers, so
        // `clone_live` needs an all-live mask).
        let side_vals = self
            .side
            .vals
            .iter()
            // SAFETY: every side entry's value is initialized.
            .map(|v| MaybeUninit::new(unsafe { v.assume_init_ref() }.clone()))
            .collect::<Vec<_>>();
        let side_live_mask = vec![0u64; self.side.coords.len()]; // 0 is live (not EMPTY/TOMB)
        let side = SideStore {
            coords: self.side.coords.clone(),
            vals: side_vals,
            // SAFETY: slot is live (its coordinate is checked above); the caller stops using it afterwards.
            keys: unsafe { self.side.keys.clone_live(&side_live_mask) },
            cap: self.side.coords.len(),
            _k: std::marker::PhantomData,
        };
        Table {
            // SAFETY: slot is live (its coordinate is checked above); the caller stops using it afterwards.
            keys: unsafe { self.keys.clone_live(&self.coords) },
            coords,
            vals,
            side,
            overflow: self.overflow.clone(),
            bits: self.bits,
            len: self.len,
            tombs: self.tombs,
            seed: self.seed,
            _k: std::marker::PhantomData,
        }
    }
}

/// Show the shape (bits/len/capacity), not the raw slots.
impl<K: Coord, V, KS: KeyStore<K>> std::fmt::Debug for Table<K, V, KS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Table")
            .field("bits", &self.bits)
            .field("len", &self.len)
            .field("capacity", &self.coords.len())
            .finish_non_exhaustive()
    }
}

/// A table for coordinate keys ([`CoordInv`], integers): the key regenerates from the address, so no
/// key array. Builds a big map faster than hashbrown (no key writes) and uses less memory.
///
/// The two reserved coordinates `u64::MAX` (`EMPTY`) and `u64::MAX - 1` (`TOMB`) cannot be stored
/// without a key, so inserting either panics before the table changes; reads report them absent.
/// For `i64` keys these are `-1` and `-2`. Use a [`Table`], which keeps the key, for those values.
pub type CoordMap<K, V> = Table<K, V, Regen<K>>;

impl<K: Eq + Coord, V, KS: KeyStore<K>> Default for Table<K, V, KS> {
    fn default() -> Self {
        Self::with_bits(3)
    }
}

/// THE one sizing: the grow law (`LOAD_NUM / LOAD_DEN` headroom) applied at construction — the next
/// power of two holding `cap` keys under the grow threshold. No extra doubling, no second load
/// factor: `LoadFactorFrontier` proves there is no magic load factor. The measured frontier
/// (BENCHMARK.md §Discharged inequalities) showed the old `+1` doubling losing the 1M build AND
/// the 1M hit stream (footprint beats probe length out of cache, `CoordinateReadout.footprint_mono`)
/// while winning only sub-millisecond in-cache hits; the current 3/4 is the measured choice of the
/// 2026-10-06 random-key sweep (7/8 vs 3/4 vs 5/8 at n = 1e5, 1e6 — 3/4 smallest on every random
/// cell outside spread). One law, stated once, same for both key species.
#[inline]
fn bits_for(cap: usize) -> u32 {
    let need = cap.saturating_mul(LOAD_DEN) / LOAD_NUM + 1;
    (usize::BITS - need.max(8).next_power_of_two().leading_zeros() - 1).max(3)
}

/// Bits for a slot line of at least `slots` (the next power of two ≥ `slots`). Used by [`FreqMap`],
/// which sizes its oversize directly — the oversize is its own headroom, no extra grow factor.
#[inline]
fn bits_for_slots(slots: usize) -> u32 {
    (usize::BITS - slots.max(8).next_power_of_two().leading_zeros() - 1).max(3)
}

/// Constructors for the default (opaque-key) table — keys stored. Concrete `Stored` so `Table::new()`
/// is unambiguous; for coordinate keys use [`CoordMap`]'s constructors, which store no key.
impl<K: Eq + Coord, V> Table<K, V, Stored<K>> {
    /// An empty table (8 slots; grows as needed).
    pub fn new() -> Self {
        Self::with_bits(3) // 8 slots
    }

    /// An empty table presized to hold `cap` keys without growing.
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_bits(bits_for(cap))
    }
}

/// Seeded constructors, for keys an attacker chooses. The default table is **unseeded**: its fold is
/// a public function of the key (`PHI` is public), so keys can be computed that all land on the slots
/// a victim key probes. A seeded table keys every coordinate with a secret [`FoldKey`]
/// ([`Coord::coord_keyed`]), so those keys cannot be aimed in advance. It stores the key and is exact.
/// It exists only for the [`Stored`] species: a [`CoordMap`] regenerates the key from its address and
/// so cannot hide it. The cost is one keyed fold per operation. The keyed fold is a multiply-fold,
/// not a cryptographic PRF.
impl<K: Eq + Coord, V> Table<K, V, Stored<K>> {
    /// An empty seeded table with an explicit [`FoldKey`] (reproducible; for tests and replay). Against
    /// an attacker the key must be unpredictable: see [`with_random_seed`](Self::with_random_seed).
    pub fn with_seed(seed: FoldKey) -> Self {
        let mut t = Self::with_bits(3);
        t.seed = Some(seed);
        t
    }

    /// A seeded table presized to hold `cap` keys.
    pub fn with_seed_and_capacity(seed: FoldKey, cap: usize) -> Self {
        let mut t = Self::with_bits(bits_for(cap));
        t.seed = Some(seed);
        t
    }

    /// An empty table with a fresh random [`FoldKey`] from `std`'s per-process random state: the
    /// opt-in flood-resistant constructor for untrusted keys.
    pub fn with_random_seed() -> Self {
        Self::with_seed(FoldKey::random())
    }

    /// [`with_random_seed`](Self::with_random_seed), presized to hold `cap` keys.
    pub fn with_random_seed_and_capacity(cap: usize) -> Self {
        Self::with_seed_and_capacity(FoldKey::random(), cap)
    }

    /// Whether this table folds its keys under a secret ([`with_seed`](Self::with_seed)).
    pub fn is_seeded(&self) -> bool {
        self.seed.is_some()
    }
}

/// Constructors for a [`CoordMap`] — coordinate keys, no key array (regenerated from the address).
impl<K: CoordInv + Eq, V> Table<K, V, Regen<K>> {
    /// A coordinate map sized for `cap` keys — the same one sizing as [`Table::with_capacity`].
    pub fn coordinate_with_capacity(cap: usize) -> Self {
        Self::with_bits(bits_for(cap))
    }
}

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    fn with_bits(bits: u32) -> Self {
        let n = 1usize << bits;
        Table {
            coords: vec![EMPTY; n].into_boxed_slice(),
            vals: alloc_uninit_boxed(n),
            keys: KS::alloc(n),
            side: SideStore::empty(),
            overflow: None,
            bits,
            len: 0,
            tombs: 0,
            seed: None,
            _k: std::marker::PhantomData,
        }
    }

    /// A key's coordinate: the key's own [`Coord::coord`], or its keyed coordinate when the table is
    /// seeded ([`with_seed`](Table::with_seed)). The two values reserved as occupancy markers
    /// (`EMPTY`, `TOMB`) are held out of band (see [`SideStore`]), so no coordinate is remapped here.
    #[inline]
    fn coord_of(&self, k: &K) -> u64 {
        match self.seed {
            None => k.coord(),
            Some(s) => k.coord_keyed(s),
        }
    }

    /// The home slot. A key that fits the array indexes directly (no hash); a larger key takes the
    /// top `bits` of `coord·PHI`.
    #[inline]
    fn home(&self, coord: u64) -> usize {
        let len = self.coords.len();
        if (coord as usize) < len {
            coord as usize
        } else {
            (coord.wrapping_mul(PHI) >> (64 - self.bits)) as usize
        }
    }

    /// The `j`-th probe slot for a collision — the shipped `seqLin` probe
    /// (`RadixDescent.linear_prefix_chain_separates`). Steps `1..LINEAR_G` are a **linear prefix**:
    /// `home + j`, contiguous slots that stay in the home cache line, so a miss (or a short chain)
    /// reads one 64-byte line of coordinates instead of one scattered line per radix step — the cache
    /// locality THE CUT gave up when it reverted to the pure scattered descent. Steps `≥ LINEAR_G`
    /// resume the MSB-first radix windows of `coord·PHI`, which break any linear cluster: distinct
    /// keys still separate within `LINEAR_G + D − 1` steps — a constant fixed by the word and the
    /// group width, never `n` (the prefix is an injective translation mod `2^b`; the digits after it
    /// are the proven walk re-indexed). That is ALIASING, not probe length: corrected 2026-08-06,
    /// this comment read "so flood immunity is preserved", and
    /// `AggregationBound.descent_can_be_wholly_blocked` refutes that step — `PHI⁻¹` aims one crafted
    /// key at each slot a victim would try. The capped probe ([`Self::probe_limit`]) never takes the
    /// last arm below (the terminal sweep): a key unresolved within [`PROBE_CAP`] window steps is
    /// routed to the [`Overflow`] instead, so this function's index `j` always satisfies
    /// `j < window_steps()`. The sweep arm remains for the negative control that restores it. No tag
    /// line, no extra array — only the probe *order* changes.
    #[cold]
    fn descend(&self, coord: u64, j: u32) -> usize {
        let (bits, mask) = (self.bits, self.coords.len() - 1);
        if j < LINEAR_G {
            // seqLin, j < G: the home slot walked linearly, staying in the home cache line.
            return self.home(coord).wrapping_add(j as usize) & mask;
        }
        // seqLin, j ≥ G: fresh radix windows, digit index shifted past the prefix.
        let p = coord.wrapping_mul(PHI);
        let jj = j - LINEAR_G + 1; // radix step 1, 2, … (window 0 is home, already probed)
        let used = bits * (jj + 1);
        if used <= 64 {
            // A window of the first product `coord·PHI` — the proven radix digits
            // (`linear_prefix_chain_separates`, separation within D+G−1).
            (p >> (64 - used)) as usize & mask
        } else {
            // Past the first product's digits: the proven TERMINAL SWEEP (`GroupProbe.group_descent_dichotomy`
            // at slot granularity). `sweep_period_no_repeat`: the `+1` offset makes `1..2^bits` a
            // permutation of the slots, so a miss finds EMPTY within one period (guaranteed
            // termination); `zero_offset_duplicates` proves the `+1` is required (offset 0 re-glances
            // the last digit's slot). Base carries the low product bits (the mirror of `pile_card`'s
            // residual) where keys that collide every high window still differ. The dense-block scan is
            // routed away by construction: joins/counts/mixed keys go to the NB face (`FreqMap`), whose
            // uniform fold has no identity-home block; this B-face serves dense coordinate keys whose
            // misses hit EMPTY at their own slot and never reach here.
            let full = 64 / bits; // radix windows per product
            ((p >> (64 - bits * full)) as usize + (jj - full) as usize + 1) & mask
        }
    }

    /// Read one key. `None` = absent. ONE algorithm for both key stores: the home coordinate, then
    /// (on a collision) the `seqLin` walk — a linear prefix in the home cache line, then radix windows
    /// (`RadixDescent.linear_prefix_chain_separates`). The near-certain home hit reads one coordinate,
    /// confirms the stored key, and reads its value; a chain or miss stays in the home line. `KS`
    /// selects only key *storage* (regenerated vs kept), never the algorithm.
    ///
    /// The stored key is confirmed on a coordinate match (free on `Regen`, where the coordinate
    /// round-trips), so a coordinate collision returns `None` instead of another key's value;
    /// [`get_exact`](Self::get_exact) is the same read under its historical name.
    #[inline]
    pub fn get(&self, k: &K) -> Option<&V> {
        self.get_coord(k)
    }

    /// The coordinate read: one scalar coordinate at `home`, a predicted-taken compare, then the
    /// collision tail only on the rare miss. The scalar home probe is what keeps the *coordinate*-key
    /// path (CoordMap: near-certain home hit) free of the SIMD glance's overhead — the glance is paid
    /// only where it wins, in the collision tail that opaque hashed keys actually reach.
    ///
    /// A coordinate match is confirmed against the stored key (free on `Regen`), so two distinct keys
    /// with equal coordinates are two entries; a reserved coordinate is resolved in the [`SideStore`].
    #[inline]
    fn get_coord(&self, k: &K) -> Option<&V> {
        let coord = self.coord_of(k);
        if is_sentinel(coord) {
            let i = self.side.find(coord, k)?;
            // SAFETY: `find` only returns an entry whose `live` marker is set.
            return Some(unsafe { self.side.vals.get_unchecked(i).assume_init_ref() });
        }
        let home = self.home(coord);
        // SAFETY: home ∈ [0, len). The home hit reads one coord, confirms the key, then one value.
        let sc = unsafe { *self.coords.get_unchecked(home) };
        // SAFETY: slot is live (the probe matched a live coordinate here).
        if sc == coord && unsafe { self.keys.matches(home, coord, k) } {
            return Some(unsafe { self.vals.get_unchecked(home).assume_init_ref() });
        }
        // The direct-addressed face (`home == coord`, i.e. a coordinate below the array length) is
        // a local, biased scan and this `EMPTY` return predicts. A hashed key lands at a scattered
        // address, so the `EMPTY`/occupied split is near even and mispredicts; there the early
        // return is skipped and `probe_capped_with` reads lane 0 (== `home`) in its own prefix
        // glance, whose `empties` branch is near-certain on the common miss. The examined set and
        // the returned value are unchanged — the guard selects only which equivalent early-out runs.
        if home == coord as usize && sc == EMPTY {
            return None;
        }
        // Collision tail, capped at `PROBE_CAP`: a hit in the prefix is found; an EMPTY lane
        // terminates; an exhausted prefix consults the overflow (a key whose capped prefix was full
        // when it was placed lives there, and a miss here is absent).
        match self.probe_capped_with(coord, home, &overflow::ByKey { coord, k }) {
            Capped::Hit(i) => Some(unsafe { self.vals.get_unchecked(i).assume_init_ref() }),
            Capped::Vacant(_) => None,
            Capped::Blocked => match self.overflow.as_ref() {
                Some(o) => o.get(coord, k),
                None => None,
            },
        }
    }

    /// Resolve a stream of keys, software-pipelined. Compute each key's coordinate, prefetch its home
    /// line, and resolve the lookup `P` keys later — so each key's cache miss overlaps the probes of
    /// the keys ahead of it. A plain [`get`](Self::get) loop stalls on each miss with nothing else in
    /// flight; pipelining a stream lets the misses run concurrently (memory-level parallelism). For a
    /// lookup-all pass over a table larger than cache this is ~2× a `get` loop **and ~2× hashbrown**;
    /// a small (cache-resident) table falls back to the plain loop, so it never loses.
    ///
    /// `visit(key, value)` runs once per key in order; absence is `None`.
    #[inline]
    pub fn get_each<'a, F>(&'a self, keys: &'a [K], mut visit: F)
    where
        F: FnMut(&'a K, Option<&'a V>),
    {
        const P: usize = 16; // power of two ⇒ ring index is a mask, not a divide
        let n = keys.len();
        // Small / cache-resident: no miss to hide, so skip the pipeline.
        if n <= P || self.footprint_bytes() <= (1 << 19) {
            for k in keys {
                visit(k, self.get(k));
            }
            return;
        }
        let mut coords = [0u64; P];
        for (i, c) in coords.iter_mut().enumerate() {
            *c = self.coord_of(&keys[i]);
            self.prefetch_coord(*c);
        }
        for i in 0..n {
            let c = coords[i & (P - 1)];
            let v = self.get_with_coord(c, &keys[i]);
            visit(&keys[i], v);
            let j = i + P;
            if j < n {
                let cj = self.coord_of(&keys[j]);
                coords[j & (P - 1)] = cj; // j ≡ i (mod P): the slot we just consumed
                self.prefetch_coord(cj);
            }
        }
    }
}

/// A landed insert: the entry's value is moved out (a main slot reads `TOMB`) and the caller owns
/// the merge between `new` and the value `old` that was already there. Returned by
/// [`Table::insert_core`] instead of writing, so every insert face shares one probe/reserve core and
/// differs only in what it does with the two values. The caller MUST arm a
/// [`MergeGuard`](side::MergeGuard) on it before running any user code.
struct InsertHit<V> {
    /// The primary slot ([`Store::Main`]) or side entry ([`Store::Side`]); in bounds by construction.
    slot: usize,
    /// The overflow node ([`Store::Overflow`]).
    node: u32,
    /// The overflow entry within `node` ([`Store::Overflow`]).
    entry: u32,
    /// The slot's coordinate, to restore after the write — identity is unchanged by an overwrite.
    coord: u64,
    /// Which store the hit landed in.
    store: Store,
    /// The incoming value, handed back because a hit does not consume it.
    new: V,
    /// The value that was in the entry, moved out so exactly one of the faces owns its drop.
    old: V,
}

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    /// Remove `k` and return its value, or `None` if absent. The bijection face — inverse of `put`.
    ///
    /// A removed main slot becomes a **tombstone** (`TOMB`), not `EMPTY`: because the probe re-addresses
    /// by windows of `coord·PHI` (not a contiguous run), blanking a slot to `EMPTY` mid-chain would make
    /// every key that probed *past* it unreachable. A tombstone reads as occupied-but-non-matching, so
    /// chains stay intact; the next `resize` (triggered on live + tombstone occupancy) reclaims it. A
    /// reserved-coordinate key is removed from the [`SideStore`].
    #[inline]
    pub fn remove(&mut self, k: &K) -> Option<V> {
        let coord = self.coord_of(k);
        if is_sentinel(coord) {
            let i = self.side.find(coord, k)?;
            // SAFETY: `find` only returns an entry whose `live` marker is set.
            let old = unsafe { self.side.vals.get_unchecked_mut(i).assume_init_read() };
            self.side.remove_entry(i);
            return Some(old);
        }
        match self.probe_capped_keyed(coord, k) {
            Capped::Hit(i) => {
                // SAFETY: found ⇒ value initialized. Commit the slot's state (`TOMB`, `len`,
                // `tombs`) BEFORE the key is moved out and dropped: a panicking `K::drop` then
                // finds the slot already non-live, so `Drop`/iteration skip its moved-out value
                // and no read sees the tombstone's gone value. `old` is dropped on the unwind.
                let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
                unsafe { *self.coords.get_unchecked_mut(i) = TOMB };
                self.len -= 1;
                self.tombs += 1;
                // SAFETY: slot is live (its coordinate is checked above); the caller stops using it afterwards.
                drop(unsafe { self.keys.take_key(i, coord) });
                Some(old)
            }
            Capped::Vacant(_) => None,
            // A key whose capped prefix is full lives in the overflow; a genuine miss there is absent.
            Capped::Blocked => self
                .overflow
                .as_mut()
                .and_then(|o| o.remove(coord, k)),
        }
    }

    /// The reserve/probe/store core every insert face shares. On a miss `new` is stored and this
    /// returns `None`. On a hit the old value is moved out and the caller receives the
    /// [`InsertHit`] — so [`insert_with`](Self::insert_with) (merge applied by the caller under a
    /// [`MergeGuard`](side::MergeGuard)) and [`insert_merge_last_write`](Self::insert_merge_last_write)
    /// (merge specialised to its own body) are the same probe over one body, not two copies of it.
    ///
    /// For the general merge (`last_write == false`) a hit leaves the entry ready for the guard: a
    /// main slot is marked `TOMB` (never `EMPTY`, which would cut every probe chain through it), a
    /// side or overflow entry holds a moved-out value. The caller arms the guard before running any
    /// user code, so an unwinding merge removes the key. For `set(new, old) = new`
    /// (`last_write == true`) there is nothing to merge: in every store `new` is written in place
    /// before `old` is dropped, and `None` is returned — a panicking `V::drop` finds the map
    /// already consistent. The incoming key is dropped before any entry is touched (a hit keeps the
    /// stored key). A key whose coordinate is a reserved marker (`EMPTY`/`TOMB`) is inserted into
    /// the side store instead of the slot arrays.
    #[inline]
    fn insert_core(&mut self, k: K, new: V, last_write: bool) -> Option<InsertHit<V>> {
        let coord = self.coord_of(&k);
        if is_sentinel(coord) {
            if let Some(i) = self.side.find(coord, &k) {
                drop(k);
                // SAFETY: every side entry's value is initialized; it is moved out exactly once.
                let old = unsafe { self.side.vals.get_unchecked_mut(i).assume_init_read() };
                if last_write {
                    unsafe { *self.side.vals.get_unchecked_mut(i) = MaybeUninit::new(new) };
                    drop(old);
                    return None;
                }
                return Some(InsertHit {
                    slot: i,
                    node: 0,
                    entry: 0,
                    coord,
                    store: Store::Side,
                    new,
                    old,
                });
            }
            self.side.push(coord, k, new);
            return None;
        }
        self.grow_for_one();
        match self.probe_capped_keyed(coord, &k) {
            Capped::Hit(i) => {
                drop(k);
                // SAFETY: found ⇒ value initialized; it is moved out exactly once.
                let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
                if last_write {
                    // `set(new, old) = new`: the slot's identity is unchanged, so nothing is marked
                    // and nothing is handed back. The slot holds `new` before `old`'s destructor runs.
                    unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(new) };
                    drop(old);
                    return None;
                }
                // `TOMB` for the merge: the value is moved out, and the slot still reads as
                // occupied-but-non-matching, so no other key's chain is cut.
                unsafe { *self.coords.get_unchecked_mut(i) = TOMB };
                Some(InsertHit {
                    slot: i,
                    node: 0,
                    entry: 0,
                    coord,
                    store: Store::Main,
                    new,
                    old,
                })
            }
            Capped::Vacant(i) => {
                // SAFETY: probe returns an in-bounds EMPTY slot; coords, vals, keys share the index.
                unsafe { *self.coords.get_unchecked_mut(i) = coord };
                unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(new) };
                // SAFETY: slot is in bounds and vacant (the probe's EMPTY slot), so nothing is dropped.
                unsafe { self.keys.set(i, k) }; // Stored: write the key; Regen: no-op
                self.len += 1;
                None
            }
            Capped::Blocked => {
                // The capped prefix was full: the key lives here or it is new here.
                if self.overflow.is_none() {
                    self.overflow = Some(Overflow::new());
                }
                match self.overflow.as_mut() {
                    Some(ov) => match ov.find_entry(coord, &k) {
                        Some((node, entry)) => {
                            drop(k);
                            let old = ov.take_value(node, entry);
                            if last_write {
                                ov.write_back(node, entry, new);
                                drop(old);
                                return None;
                            }
                            Some(InsertHit {
                                slot: 0,
                                node,
                                entry,
                                coord,
                                store: Store::Overflow,
                                new,
                                old,
                            })
                        }
                        None => {
                            ov.insert_new_at(coord, k, new);
                            None
                        }
                    },
                    None => unreachable!(), // ensured `Some` just above
                }
            }
        }
    }

    /// The one mutator: set `k` to `f(new, old)`. An absent key has no old value, so you get
    /// `f(new, identity) = new`. Every table face is just this with a different `f`.
    ///
    /// If `f` panics, the key is removed (`len` falls by one) and the panic propagates; every
    /// other key is untouched and every value is dropped once ([`MergeGuard`](side::MergeGuard)).
    #[inline]
    pub fn insert_with(&mut self, k: K, new: V, f: impl FnOnce(V, V) -> V) {
        if let Some(hit) = self.insert_core(k, new, false) {
            let guard = side::MergeGuard {
                t: self,
                store: hit.store,
                slot: hit.slot,
                node: hit.node,
                entry: hit.entry,
                coord: hit.coord,
            };
            let merged = f(hit.new, hit.old);
            guard.finish(merged);
        }
    }

    /// The Map face's merge, concretely: set `k` to `new`, overwriting whatever was there — the same
    /// semantics as [`put`](Self::put) (`insert_with(k, v, set)`, with [`set`](crate::map::hashtrinity::set)
    /// `= new`), specialised into its own body instead of taking the merge as a `fn` item.
    ///
    /// Merge-free by construction, not by duplication: it is [`insert_core`](Self::insert_core) plus
    /// the merge `set(new, old) = new`, which consumes `old` and keeps `new`. A first-order translator
    /// cannot take an arrow-typed merge argument, so the concrete face is what the extraction roots.
    /// `insert_core(k, new, LAST_WRITE)` writes every hit in place (main, side and overflow), so no
    /// hit comes back and no guard is armed. The probe still runs once, in the shared core, so
    /// nothing is duplicated.
    #[inline]
    pub fn insert_merge_last_write(&mut self, k: K, new: V) {
        match self.insert_core(k, new, true) {
            None => {}
            Some(_) => unreachable!(), // a last-write hit is written in place by `insert_core`
        }
    }

    /// Re-place one carried entry through the same capped placement every insert uses: a primary
    /// `EMPTY` within the cap, or the overflow. Used only by [`resize`](Self::resize), where keys are
    /// distinct, so a `Hit` (an already-present identity) is impossible and handled defensively.
    fn place_capped(&mut self, coord: u64, k: K, v: V) {
        match self.probe_capped_keyed(coord, &k) {
            Capped::Hit(i) => {
                // Defensive: the rebuild carries distinct keys. Overwrite, dropping the old value.
                // SAFETY: a live slot ⇒ value initialized; `assume_init_read` moves it out.
                let old = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
                drop(old);
                unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(v) };
            }
            Capped::Vacant(i) => {
                // SAFETY: probe returns an in-bounds EMPTY slot.
                unsafe { *self.coords.get_unchecked_mut(i) = coord };
                unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(v) };
                // SAFETY: slot is in bounds and vacant (the probe's EMPTY slot), so nothing is dropped.
                unsafe { self.keys.set(i, k) }; // Stored: write the key; Regen: no-op
                self.len += 1;
            }
            Capped::Blocked => {
                if self.overflow.is_none() {
                    self.overflow = Some(Overflow::new());
                }
                match self.overflow.as_mut() {
                    Some(ov) => {
                        ov.insert_new_at(coord, k, v);
                    }
                    None => unreachable!(), // ensured `Some` just above
                }
            }
        }
    }

    /// The grow law, applied before an insert that may take a fresh slot: when live + tombstone
    /// occupancy would reach `LOAD_NUM / LOAD_DEN` of the slots, rebuild — at the same size only when
    /// `len + 1 ≤ capacity / 2` (hashbrown's `RawTable::reserve_rehash` rule), otherwise doubled.
    /// The probe walks through tombstones, so they count toward the trigger; without the rebuild a
    /// churn workload would fill the slots with tombstones and a miss would never reach `EMPTY`.
    ///
    /// Cost: O(1) when no rebuild runs; a rebuild is O(cap) slot moves ([`resize`](Self::resize)).
    /// Amortised O(1) per insert: only an insert into an `EMPTY` slot raises `len + tombs` (a
    /// remove trades one live slot for one tombstone), and a rebuild leaves `tombs = 0`. After a
    /// same-size rebuild `len + 1 ≤ cap/2`, so the next trigger at `len + tombs + 1 ≥ 3/4·cap` is
    /// at least `cap/4` inserts away; after a doubling `len ≤ 3/4·cap = 3/8·(2·cap)`, so it is at
    /// least `3/8·(2·cap)` inserts away. Either way ≤ 4 slot moves per insert. The rule it
    /// replaces (same size whenever `len + 1 < 3/4·cap`) let a steady size of `3/4·cap − 1`
    /// rebuild the whole table on every remove+insert pair.
    #[inline]
    pub(super) fn grow_for_one(&mut self) {
        let cap = self.coords.len();
        if (self.len + self.tombs + 1) * LOAD_DEN >= cap * LOAD_NUM {
            let new_bits = if self.len < cap / 2 {
                // `len + 1 ≤ cap/2`: crowded by tombstones, a same-size rebuild reclaims them.
                self.bits
            } else {
                self.bits + 1
            };
            self.resize(new_bits);
        }
    }

    /// Rebuild into a fresh table of `new_bits` slots, carrying only live entries — so tombstones
    /// vanish. `new_bits > bits` grows; `new_bits == bits` is an in-place compaction that reclaims
    /// tombstones without doubling. The old `grow` is `resize(bits + 1)` with no tombstones present.
    /// Every entry — primary and overflow — is re-placed through the same capped placement
    /// ([`place_capped`](Self::place_capped)); an uncapped probe is never called. The [`SideStore`]
    /// is carried over unchanged (it does not depend on `bits`).
    #[cold]
    fn resize(&mut self, new_bits: u32) {
        let mut bigger = Self::with_bits(new_bits);
        // The stored coordinates are keyed: the rebuilt table must key them the same way.
        bigger.seed = self.seed;
        let mut i = 0;
        while i < self.coords.len() {
            let c = self.coords[i];
            if is_live(c) {
                // Move the key out first so the keyed probe can confirm it: two distinct keys with
                // equal coordinates must land on two slots of the rebuilt table, not one.
                // SAFETY: slot is live (its coordinate is checked above); the caller stops using it afterwards.
                let key = unsafe { self.keys.take_key(i, c) };
                // SAFETY: live coord ⇒ value initialized. Move the value out and blank the source
                // coord so the old table's Drop skips it.
                let v = unsafe { self.vals.get_unchecked_mut(i).assume_init_read() };
                self.coords[i] = EMPTY;
                bigger.place_capped(c, key, v);
            }
            i += 1;
        }
        // Carry the side store over; the rebuilt table is fresh (empty side store).
        bigger.side = std::mem::replace(&mut self.side, SideStore::empty());
        // Re-place every overflow entry through the capped placement too: a grown table may now hold
        // it in the primary array, and the invariant is re-established for the new generation.
        if let Some(ov) = self.overflow.take() {
            for (c, k, v) in ov.into_entries() {
                bigger.place_capped(c, k, v);
            }
        }
        *self = bigger; // tombstones dropped: bigger.tombs == 0
    }

    /// The number of live keys: primary slots, side entries and overflow entries.
    pub fn len(&self) -> usize {
        // A `match`, not `Option::map_or(Overflow::len)`: a `fn` item argument is an arrow type,
        // which the extraction's first-order boundary does not carry.
        let overflow_len = match &self.overflow {
            Some(o) => o.len(),
            None => 0,
        };
        self.len + self.side.coords.len() + overflow_len
    }
    /// Whether the table holds no keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    fn footprint_bytes(&self) -> usize {
        self.coords.len() * (std::mem::size_of::<u64>() + std::mem::size_of::<MaybeUninit<V>>())
    }
    /// Slot count (the power-of-two array length). `len / capacity` is the load factor.
    pub fn capacity(&self) -> usize {
        self.coords.len()
    }
}

/// The key-readback face of iteration — projects [`KeyReadout::Key`], so it requires the extension
/// trait; the storage and probe bodies stay on [`KeyStore`].
impl<K: Eq + Coord, V, KS: KeyReadout<K>> Table<K, V, KS> {
    /// Iterate the live `(key, &value)` pairs in arbitrary order — the main slots, then the side
    /// entries, then the overflow entries (the walker [`Iter`], which includes all three stores).
    pub fn iter(&self) -> Iter<'_, K, V, KS> {
        Iter::new(self)
    }
}

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    /// Bit-exact read — never returns a wrong value. The stored key is compared on a coordinate
    /// match (free on `Regen`, where the coordinate round-trips), so a coordinate collision returns
    /// `None` instead of someone else's value; a reserved-coordinate key is resolved in the
    /// [`SideStore`]. Same read as [`get`](Self::get).
    #[inline]
    pub fn get_exact(&self, k: &K) -> Option<&V> {
        self.get_coord(k)
    }

    /// Prefetch (T0) the coordinate home line for the scalar pipeline. Issued for a key `P` steps
    /// ahead so its cache miss overlaps the probes already in flight.
    #[inline]
    fn prefetch_coord(&self, coord: u64) {
        let i = self.home(coord);
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            _mm_prefetch(self.coords.as_ptr().add(i).cast::<i8>(), _MM_HINT_T0);
        }
        let _ = i;
    }

    /// The coordinate seed matched to this table's slot count (see the [`seed`] module). `seed.home(c)`
    /// is this table's own home slot and `seed.residual(c)` is the rest of `c·PHI` the slot doesn't
    /// carry — what a compact readout would store instead of the full coordinate. It describes the
    /// unkeyed coordinate, so it does not apply to a seeded table. Diagnostic — not part of the
    /// stable surface.
    #[doc(hidden)]
    #[inline]
    pub fn coord_seed(&self) -> CoordSeed {
        CoordSeed::new(self.bits)
    }

    /// `get` from a pre-computed coordinate — the pipelined path computes the coordinate during
    /// prefetch and resolves `k` here. A reserved coordinate is resolved in the [`SideStore`].
    #[inline]
    fn get_with_coord(&self, coord: u64, k: &K) -> Option<&V> {
        if is_sentinel(coord) {
            let i = self.side.find(coord, k)?;
            // SAFETY: `find` only returns an entry whose `live` marker is set.
            return Some(unsafe { self.side.vals.get_unchecked(i).assume_init_ref() });
        }
        // Same shape as `get_coord`: the near-certain home hit, then the shared SIMD collision tail.
        let home = self.home(coord);
        // SAFETY: home ∈ [0, len).
        let sc = unsafe { *self.coords.get_unchecked(home) };
        // SAFETY: slot is live (the probe matched a live coordinate here).
        if sc == coord && unsafe { self.keys.matches(home, coord, k) } {
            return Some(unsafe { self.vals.get_unchecked(home).assume_init_ref() });
        }
        if sc == EMPTY {
            return None;
        }
        match self.probe_capped_with(coord, home, &overflow::ByKey { coord, k }) {
            Capped::Hit(i) => Some(unsafe { self.vals.get_unchecked(i).assume_init_ref() }),
            Capped::Vacant(_) => None,
            Capped::Blocked => match self.overflow.as_ref() {
                Some(o) => o.get(coord, k),
                None => None,
            },
        }
    }
}

/// Equal iff the same entries: every live entry of `self` (main, side or overflow) is present in
/// `other` under the same KEY with an equal value, and the counts agree. Keys are compared, not
/// coordinates: two distinct keys may share a coordinate (`u128` `5` and `5 << 64`), and they are
/// different entries. O(n) key lookups, each O(1) expected.
impl<K: Eq + Coord, V: PartialEq, KS: KeyStore<K>> PartialEq for Table<K, V, KS> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for (i, c) in self.coords.iter().enumerate() {
            if !is_live(*c) {
                continue;
            }
            // SAFETY: live coord ⇒ slot i's key and value are initialized.
            let same = unsafe {
                self.keys.with_key(i, *c, |k| {
                    other.get(k) == Some(self.vals.get_unchecked(i).assume_init_ref())
                })
            };
            if !same {
                return false;
            }
        }
        for i in 0..self.side.coords.len() {
            let c = self.side.coords[i];
            // SAFETY: side entries are occupied ⇒ keys and values are initialized.
            let same = unsafe {
                self.side.keys.with_key(i, c, |k| {
                    other.get(k) == Some(self.side.vals.get_unchecked(i).assume_init_ref())
                })
            };
            if !same {
                return false;
            }
        }
        if let Some(o) = &self.overflow {
            for (_c, k, v) in o.iter() {
                if other.get(k) != Some(v) {
                    return false;
                }
            }
        }
        true
    }
}
impl<K: Eq + Coord, V: Eq, KS: KeyStore<K>> Eq for Table<K, V, KS> {}
