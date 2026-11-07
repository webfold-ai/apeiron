//! The trait vocabulary `Table<K, V, KS>` is parameterized by — split out of `lib.rs`, re-exported
//! from it, so every path is unchanged. Two halves: how a KEY becomes a 64-bit address
//! (`Coord`/`CoordInv`/`KeyStore`/`Stored`/`Regen`, and the golden-128 byte fold underneath), and
//! how two VALUES merge (`Face`/`Species`/the four merges/`Monoid`/`FoldExt`).

use crate::map::hashtrinity::{is_live, is_sentinel, Capped, Locate, Overflow, Table, EMPTY, PHI};
use std::mem::MaybeUninit;

/// `n` uninitialized [`MaybeUninit<T>`] slots as a boxed slice — RESERVE, never initialize.
///
/// The table's slot arrays are allocated full-length and left unwritten: liveness lives in the
/// coordinate line, so a build writes a slot only when it stores into it. That is the property
/// `../sanguine-ref/proof/Closures/Engineering/ConcreteProbeCost.lean:607` `slot_tax_exact` charges a `capacity`-scaled per-slot tax for; an
/// element-wise fill would move `capacity` writes into the build and forfeit it.
///
/// One allocation of `n` elements, zero per-slot writes: O(n) bytes allocated, 0 element writes,
/// 0 element reads, 1 allocation call. The loop is absent, so nothing scales with a write count.
///
/// The equivalent `repeat_with(MaybeUninit::uninit).take(n).collect()` allocates identically but
/// states the same property through `Iterator::collect`, which the extraction's pinned model does
/// not carry; this form is the reserve itself, spelled as the two operations it performs.
///
/// # Safety of the `set_len`
///
/// `Vec::with_capacity(n)` reserves at least `n` elements, and an uninitialized `MaybeUninit<T>`
/// is not a value of `T`, so exposing the length neither reads a slot nor claims a drop
/// obligation: the drop glue for `MaybeUninit<T>` is a no-op. `T` is therefore never dropped
/// uninitialized and never read before [`KeyStore::set`] writes it.
#[inline]
pub fn alloc_uninit_boxed<T>(n: usize) -> Box<[MaybeUninit<T>]> {
    let mut v: Vec<MaybeUninit<T>> = Vec::with_capacity(n);
    // SAFETY: the capacity above is ≥ n and `MaybeUninit<T>` carries no drop obligation, so a
    // length of n over the reserved-but-unwritten storage is sound.
    unsafe { v.set_len(n) };
    v.into_boxed_slice()
}

// ── Coord: the key's 64-bit address. ─────────────────────────────────────────────────────────

/// A key's 64-bit address. Integers are their own coordinate (a direct, collision-free index);
/// byte/opaque keys fold to one. This is the only thing the table needs to place a key.
pub trait Coord {
    /// Whether `coord` is injective on `Self`: `a.coord() == b.coord()` implies `a == b`. `true`
    /// only for the impls whose coordinate IS the key (the sign/zero-extending integer casts, and
    /// the delegating reference); `false` (the default) for every lossy fold. When it holds, a
    /// coordinate match at a slot already implies key equality, so [`Stored`] can skip the stored
    /// key compare on the hot read (`CoordinateReadout.exact_read_is_read`).
    ///
    /// # Soundness
    ///
    /// Overriding this to `true` for a lossy coordinate is unsound: two distinct keys with equal
    /// coordinates would then alias, and the table would return one's value for the other.
    const INJECTIVE: bool = false;
    /// This key's 64-bit address.
    fn coord(&self) -> u64;
}

macro_rules! int_coord {
    ($($t:ty),*) => {$( impl Coord for $t {
        const INJECTIVE: bool = true; // `*self as u64` is a bijection onto its image
        #[inline] fn coord(&self) -> u64 { *self as u64 } // an integer is already its own address
    } )*};
}
int_coord!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

/// Keys whose coordinate *is* their identity — integers, where `from_coord(k.coord()) == k`. If the
/// address round-trips to the key, the key never has to be stored: it regenerates from the slot it
/// sits in ([`Regen`]). Opaque keys (`str`/`String`) fold lossily and have no inverse, so their key
/// must be kept ([`Stored`]). Storing a key that could regenerate is wasted space and a wasted write.
pub trait CoordInv: Coord + Sized {
    /// `from_coord(k.coord()) == k` for every `k`.
    fn from_coord(c: u64) -> Self;
}
macro_rules! int_coord_inv {
    ($($t:ty),*) => {$( impl CoordInv for $t {
        #[inline] fn from_coord(c: u64) -> Self { c as Self } // two's complement makes `as u64` round-trip
    } )*};
}
int_coord_inv!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

// ── KeyStore: how a key's identity is held, or regenerated. ────────────────────────────────────

/// How a table holds each key's identity. Opaque keys are stored ([`Stored`]); coordinate keys
/// regenerate from the address and store nothing ([`Regen`]).
///
/// This trait is deliberately **generic-associated-type-free**: the borrowed/owned readback
/// (`Key<'a>`, [`key`](KeyReadout::key)) lives on the extension trait [`KeyReadout`]. Aeneas's
/// `translate_fwd_ty` refuses a nonempty-generic `TTraitType`, so a `type Key<'a>` here keeps the
/// whole storage trait — and therefore every `Table` body bounded by it — out of the extraction.
/// Storage, probe and mutation need only this trait; readback takes [`KeyReadout`].
pub trait KeyStore<K: Coord>: Sized {
    /// True when keys regenerate from the address ([`Regen`]) instead of being stored ([`Stored`]).
    /// Selects key *storage* only — never the read algorithm (there is one probe). `get_exact`
    /// uses it to skip the redundant key compare (`CoordinateReadout.exact_read_is_read`).
    const REGEN: bool = false;
    /// Allocate a store for `n` slots.
    fn alloc(n: usize) -> Self;
    /// Record slot `i`'s key (Stored: keep it; Regen: a no-op — the address carries it).
    fn set(&mut self, i: usize, k: K);
    /// Whether slot `i` holds key `q` (an exact compare; trivially true on the Regen path).
    fn matches(&self, i: usize, coord: u64, q: &K) -> bool;
    /// Run `f` on a reference to slot `i`'s key (borrowed if stored, regenerated if not). The
    /// key-borrowing face the exact comparisons use without taking ownership of the key.
    ///
    /// # Safety
    /// Slot `i` must be live.
    unsafe fn with_key<R>(&self, i: usize, coord: u64, f: impl FnOnce(&K) -> R) -> R;
    /// Move slot `i`'s key into slot `j` of `dst` (used while growing).
    fn migrate(&mut self, i: usize, dst: &mut Self, j: usize);
    /// Move slot `i`'s key out (Stored: take it; Regen: regenerate it).
    fn take_key(&mut self, i: usize, coord: u64) -> K;
    /// Clone the LIVE keys (`is_live(coords[i])` — excludes `EMPTY` and a removed slot's `TOMB`)
    /// into a fresh store. Liveness is the caller's coordinate line — the store carries no
    /// discriminant of its own (see [`Stored`]).
    #[must_use]
    fn clone_live(&self, coords: &[u64]) -> Self
    where
        K: Clone;
    /// Drop the LIVE keys. Called by `Table`'s `Drop`; a no-op when `K` has no destructor or
    /// nothing is stored.
    fn drop_live(&mut self, coords: &[u64]);
}

/// Key readback — the borrowed/owned key projection, split off [`KeyStore`] so the storage trait
/// carries no generic associated type. `Key<'a>` is a borrow when stored, an owned key when
/// regenerated.
///
/// The split is the route-1 extraction adaptation: the executing `new`/`with_capacity`/`get`/
/// `insert_with`/growth bodies never read a key back, so they need only [`KeyStore`] and translate
/// once the GAT is out of that trait. Only the readback faces (`Table::iter` and the `face`
/// iteration surfaces) require this extension trait.
pub trait KeyReadout<K: Coord>: KeyStore<K> {
    /// The key as read back — `&K` when stored, `K` (regenerated) when not.
    type Key<'a>
    where
        Self: 'a,
        K: 'a;
    /// The key at slot `i` — borrowed if stored, regenerated from `coord` if not.
    fn key(&self, i: usize, coord: u64) -> Self::Key<'_>;
    /// The readout of an already-materialized key `k` — the [`Overflow`](crate::map::hashtrinity::Overflow)
    /// tree stores keys inline rather than in the index-addressed store, so its iteration projects
    /// a key it holds directly. `&K` when stored; regenerated from the key's
    /// coordinate when not (exact: a `Regen` coordinate round-trips).
    fn key_of<'a>(&'a self, k: &'a K) -> Self::Key<'a>
    where
        Self: 'a,
        K: 'a;
    /// Whether the key at slot `i` is exactly the key a borrowed [`Locate`] view names. The borrowed
    /// faces' full-key confirmation — the twin of [`KeyStore::matches`], which takes an owned `&K`.
    fn matches_locate<Q>(&self, i: usize, coord: u64, q: &Q) -> bool
    where
        K: Eq,
        Q: Locate<K>;
}

/// Stores each key in a bare `MaybeUninit<K>` slot — for opaque keys whose fold has no inverse.
///
/// No `Option`, no per-slot clearing: the key store rides the SAME invariant as the value array —
/// **`keys[i]` is initialized iff `coords[i]` is live** (neither `EMPTY` nor a tombstone) — so
/// allocation writes nothing
/// (`../sanguine-ref/proof/Closures/Engineering/ConcreteProbeCost.lean:607` `slot_tax_exact`: per-slot clearing is a named, removable build
/// cost; this is the move that removed the `Option<V>` discriminant, applied to `K`). Liveness
/// lives in the coordinate line, so `clone_live`/`drop_live` take it from the caller.
///
/// A merge in flight marks its slot `TOMB` while the key stays initialized; the merge guard
/// (`MergeGuard` in `side.rs`) restores the coordinate on success and drops the key on an unwind,
/// so the invariant holds again before any other operation can observe the table.
pub struct Stored<K>(Box<[MaybeUninit<K>]>);

impl<K> std::fmt::Debug for Stored<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stored({} slots)", self.0.len())
    }
}

impl<K: Coord + Eq> KeyStore<K> for Stored<K> {
    #[inline]
    fn alloc(n: usize) -> Self {
        // like `vals`: reserve, never initialize — no per-slot write at build time
        Stored(alloc_uninit_boxed(n))
    }
    #[inline]
    fn set(&mut self, i: usize, k: K) {
        // only ever called on a previously-EMPTY slot — nothing to drop
        unsafe { *self.0.get_unchecked_mut(i) = MaybeUninit::new(k) };
    }
    #[inline]
    fn matches(&self, i: usize, _coord: u64, q: &K) -> bool {
        // When `coord` is injective, `coords[i] == q.coord()` (the caller's precondition) already
        // implies `self.0[i] == q`, so the stored-key load is redundant and is skipped. The branch
        // is on a const, so it folds away per `K` — no cost on the opaque-key path.
        if K::INJECTIVE {
            return true;
        }
        // SAFETY: only called on an occupied slot (after a successful probe).
        unsafe { self.0.get_unchecked(i).assume_init_ref() == q }
    }
    #[inline]
    unsafe fn with_key<R>(&self, i: usize, _coord: u64, f: impl FnOnce(&K) -> R) -> R {
        // SAFETY: slot `i` live ⇒ in bounds and initialized (the caller's contract).
        f(unsafe { self.0.get_unchecked(i).assume_init_ref() })
    }
    #[inline]
    fn migrate(&mut self, i: usize, dst: &mut Self, j: usize) {
        // SAFETY: src occupied ⇒ initialized; the caller blanks src's coord, so src's Drop skips it.
        unsafe {
            *dst.0.get_unchecked_mut(j) =
                MaybeUninit::new(self.0.get_unchecked(i).assume_init_read());
        };
    }
    #[inline]
    fn take_key(&mut self, i: usize, _coord: u64) -> K {
        // SAFETY: occupied ⇒ initialized; the caller blanks the coord, so Drop skips it.
        unsafe { self.0.get_unchecked_mut(i).assume_init_read() }
    }
    fn clone_live(&self, coords: &[u64]) -> Self
    where
        K: Clone,
    {
        let mut out = <Self as KeyStore<K>>::alloc(coords.len());
        let mut i = 0usize;
        while i < coords.len() {
            if is_live(coords[i]) {
                // SAFETY: live coord ⇒ this slot's key initialized (a tombstone's key was already
                // taken by `remove` and must NOT be read again); writing an uninit slot.
                unsafe {
                    *out.0.get_unchecked_mut(i) =
                        MaybeUninit::new(self.0.get_unchecked(i).assume_init_ref().clone());
                };
            }
            i += 1;
        }
        out
    }
    fn drop_live(&mut self, coords: &[u64]) {
        if std::mem::needs_drop::<K>() {
            let mut i = 0usize;
            while i < coords.len() {
                if is_live(coords[i]) {
                    // SAFETY: the invariant — live coord ⇒ key initialized, dropped once. A
                    // tombstone's key was already moved out by `remove`; dropping it again here
                    // would be a double-drop, so it must be excluded (not just != EMPTY).
                    unsafe { self.0.get_unchecked_mut(i).assume_init_drop() };
                }
                i += 1;
            }
        }
    }
}

impl<K: Coord + Eq> KeyReadout<K> for Stored<K> {
    type Key<'a>
        = &'a K
    where
        K: 'a;
    #[inline]
    fn key(&self, i: usize, _coord: u64) -> &K {
        // SAFETY: only called on an occupied slot — the invariant says initialized.
        unsafe { self.0.get_unchecked(i).assume_init_ref() }
    }
    #[inline]
    fn key_of<'a>(&'a self, k: &'a K) -> &'a K
    where
        Self: 'a,
        K: 'a,
    {
        k
    }
    #[inline]
    fn matches_locate<Q: Locate<K>>(&self, i: usize, coord: u64, q: &Q) -> bool {
        q.same(self.key(i, coord))
    }
}

/// Stores nothing: the key is `from_coord` of the slot's coordinate. For integer keys the address
/// *is* the key, so build writes one array instead of two and the table is smaller.
#[derive(Clone, Debug, Default)]
pub struct Regen<K>(std::marker::PhantomData<K>);

impl<K: CoordInv + Eq> KeyStore<K> for Regen<K> {
    const REGEN: bool = true; // coordinate keys: the address round-trips, nothing stored
    #[inline]
    fn alloc(_n: usize) -> Self {
        Regen(std::marker::PhantomData)
    }
    #[inline]
    fn set(&mut self, _i: usize, _k: K) {} // the address is the key — nothing to store
    #[inline]
    fn matches(&self, _i: usize, _coord: u64, _q: &K) -> bool {
        // `CoordInv` round-trips (`from_coord(k.coord()) == k`), so whenever this runs with
        // `coords[i] == coord` and `q.coord() == coord` — every caller's precondition — the key at
        // slot `i` IS `q`, and the compare is redundant. This is the same redundancy [`Stored`] skips
        // under `K::INJECTIVE`, exact unconditionally here; it removes one compare per confirmed hit.
        // The soundness rests on `set` being a no-op: `coords[i]` is written from `k.coord()` and the
        // key is regenerated as `from_coord(coords[i])`.
        true
    }
    #[inline]
    unsafe fn with_key<R>(&self, _i: usize, coord: u64, f: impl FnOnce(&K) -> R) -> R {
        f(&K::from_coord(coord))
    }
    #[inline]
    fn migrate(&mut self, _i: usize, _dst: &mut Self, _j: usize) {} // regenerated in the new table
    #[inline]
    fn take_key(&mut self, _i: usize, coord: u64) -> K {
        K::from_coord(coord)
    }
    #[inline]
    fn clone_live(&self, _coords: &[u64]) -> Self
    where
        K: Clone,
    {
        Regen(std::marker::PhantomData) // nothing stored, nothing to clone
    }
    #[inline]
    fn drop_live(&mut self, _coords: &[u64]) {} // nothing stored, nothing to drop
}

impl<K: CoordInv + Eq> KeyReadout<K> for Regen<K> {
    type Key<'a>
        = K
    where
        K: 'a;
    #[inline]
    fn key(&self, _i: usize, coord: u64) -> K {
        K::from_coord(coord)
    }
    #[inline]
    fn key_of<'a>(&'a self, k: &'a K) -> K
    where
        Self: 'a,
        K: 'a,
    {
        // The coordinate round-trips by `CoordInv`, so regenerating from it is the same key.
        K::from_coord(k.coord())
    }
    #[inline]
    fn matches_locate<Q: Locate<K>>(&self, _i: usize, coord: u64, q: &Q) -> bool {
        q.same(&self.key(_i, coord))
    }
}
impl Coord for u128 {
    #[inline]
    fn coord(&self) -> u64 {
        (*self as u64) ^ ((*self >> 64) as u64)
    }
}
impl Coord for i128 {
    #[inline]
    fn coord(&self) -> u64 {
        (*self as u128).coord()
    }
}
impl Coord for char {
    const INJECTIVE: bool = true; // codepoints are distinct
    #[inline]
    fn coord(&self) -> u64 {
        *self as u64 // a codepoint is already an address
    }
}
impl Coord for bool {
    const INJECTIVE: bool = true; // two keys, two coordinates
    #[inline]
    fn coord(&self) -> u64 {
        *self as u64
    }
}

/// The 128-bit golden constant's low half — odd, so the pair `(PHI, PHI_HI)` is a unit mod `2^128`
/// (`FoldDepth.mulUnit_bijective`). The second seed of the folded multiply.
const PHI_HI: u64 = 0xF39C_C060_5CED_C835;

/// **The folded multiply — one boundary, a counted `2^64`-to-one fiber
/// (`FoldDepth.xorHalves_balanced`).** `hi ⊕ lo` of the full 64×64→128 product: ONE `mulx`, consuming 128 bits
/// of input where the `wrapping_mul(PHI)` chain consumed 64. The `hi ⊕ lo` reduction is EXACTLY
/// `2^64`-to-one — a COUNTED balance (proven), not a hoped-for ARX distribution — and the full-width
/// multiply admits no closed-form collision family (`golden128_breaks_the_exploit`). This is
/// foldhash's core, here the proven-balanced boundary x = B/U always pointed at.
#[inline]
fn folded_mul(a: u64, b: u64) -> u64 {
    let wide = (a as u128).wrapping_mul(b as u128);
    (wide as u64) ^ ((wide >> 64) as u64)
}

/// Fold bytes to a coordinate — for keys that aren't already addresses.
///
/// **The golden-128 fold (`FoldDepth.xorHalves_balanced`, `FoldDepth.pack_injective`).** Those two
/// are what is proved; the headline `golden128_fold_forced` is not cited, because sanguine's copy
/// of the module writes `Asserted (not proved)` of it — its bijection `g` is never exhibited as the
/// golden multiply, its width `m` is arbitrary, and its exploit clause is one `decide` witness. The
/// fold shipped here is chosen, not forced. The residual was one serial
/// multiply of gap; the two-sheet fold could not beat the m=3 floor (`three_words_tie`) because it
/// stayed at 8 bytes per `·PHI` with a rotate/xor/branch around each. This folds 16 bytes per
/// `folded_mul` — twice the input per multiply, and each `folded_mul` is a single `mulx` with none
/// of that scaffolding. A ≤16-byte key is ONE folded multiply; the benchmark's 17–24-byte key is
/// two; both far fewer instructions than the striped chain. Each `folded_mul` is the proven
/// counted-`2^64`-fiber boundary (`xorHalves_balanced`). **Only keys of at most 7 bytes are
/// injective** (length and bytes packed into one word, then a bijective mix; across lengths too).
/// From 8 bytes up the input space exceeds 2^64, so collisions exist by counting; `2^-64` is the
/// per-pair rate for *random* keys, not a guarantee against chosen ones: a 9–16 byte collision is a
/// closed-form factorisation of the 128-bit product (`(2m)·n == m·(2n)`), and a birthday search
/// needs about 2^32 evaluations by the birthday bound (not measured here). [`Table`] stays exact
/// regardless (it compares the key); [`FreqMap`] trusts the fold.
/// Measured: all six FreqMap-vs-hashbrown quadrants win, the in-cache-miss residual (once +28%,
/// last +9%) now −3%. Everything proven downstream of the u64 (routing, descent, hashdos) untouched.
#[inline]
fn fold_bytes(b: &[u8]) -> u64 {
    let len = b.len();
    if len < 8 {
        // Short keys (< 8 bytes) pack INJECTIVELY across lengths: the bytes, zero-padded into the low
        // 7 bytes of one word, and the length in the top byte, then a bijective mix (xor a constant,
        // xor-shift). Distinct (length, bytes) ⇒ distinct word ⇒ distinct coordinate. A length-only
        // seed made the map injective per length alone: a 4-byte and a 5-byte key could share a
        // coordinate in closed form (audit CLM-02).
        //
        // Per length the packing is injective: ≥4 bytes take the first 4 and the last `len` − 4 bytes
        // (positions 4..len, realigned to bits 32..); 1..=3 bytes take bytes 0, len/2, len−1 (for
        // len 1 and 2 that repeats a byte, which is still injective within the length).
        let packed = if len >= 4 {
            let lo = u32::from_le_bytes(b[..4].try_into().unwrap()) as u64;
            let hi = u32::from_le_bytes(b[len - 4..].try_into().unwrap()) as u64;
            // `hi` holds bytes len-4..len; shifting right by 8·(8−len) keeps exactly positions 4..len.
            lo | ((hi >> (8 * (8 - len))) << 32)
        } else if len > 0 {
            (b[0] as u64) | ((b[len / 2] as u64) << 8) | ((b[len - 1] as u64) << 16)
        } else {
            0
        };
        let x = (packed | ((len as u64) << 56)) ^ PHI;
        return x ^ (x >> 32);
    }
    // 8 bytes and up: `len` is 8..; the seed differentiates lengths, folded in below.
    let seed = PHI ^ (len as u64).wrapping_mul(PHI_HI);
    {
        if len <= 16 {
            // two overlapping words, ONE folded multiply (16 bytes → 64 bits, was 2 muls striped).
            let a = u64::from_le_bytes(b[..8].try_into().unwrap());
            let c = u64::from_le_bytes(b[len - 8..].try_into().unwrap()); // overlap, no copy
            folded_mul(a ^ seed, c ^ PHI_HI)
        } else if len <= 24 {
            // three words, TWO folded multiplies run INDEPENDENTLY (`FoldDepth.synthFold`): `s0`
            // reads the head, `s1` reads the tail, neither reads the other's output — serial multiply
            // depth 1 not 2 (`parallel_split_is_shallower`), which fills the multiply port instead of
            // stalling on the chain (throughput, the FreqMap workload). They join by the two-sheet's
            // asymmetric rotate-xor (zero multiplies, `combine2_order_sensitive` — deferred into the
            // probe's `·PHI`). Security is unchanged: every word still passes a multiply, so the
            // `(a⊕k,mid⊕k,c⊕k)` family that collides a linear recombine dies at each sheet's
            // `folded_mul`, and the single join stays safe to leave multiply-free
            // (`synth_breaks_the_exploit`).
            let a = u64::from_le_bytes(b[..8].try_into().unwrap());
            let mid = u64::from_le_bytes(b[8..16].try_into().unwrap());
            let c = u64::from_le_bytes(b[len - 8..].try_into().unwrap());
            let s0 = folded_mul(a ^ seed, mid ^ PHI_HI);
            let s1 = folded_mul(c ^ seed, mid ^ PHI_HI);
            s0.rotate_left(5) ^ s1
        } else {
            // > 24 bytes: nonlinear 16-byte-block accumulator (half the per-word chain's depth,
            // every word through a multiply, NO linear family).
            let mut acc = seed;
            let mut s = b;
            while s.len() > 16 {
                let x = u64::from_le_bytes(s[..8].try_into().unwrap());
                let y = u64::from_le_bytes(s[8..16].try_into().unwrap());
                acc = folded_mul(acc ^ x, y ^ PHI_HI);
                s = &s[16..];
            }
            let x = u64::from_le_bytes(b[len - 16..len - 8].try_into().unwrap()); // overlap tail
            let y = u64::from_le_bytes(b[len - 8..].try_into().unwrap());
            folded_mul(acc ^ x, y ^ PHI_HI)
        }
    }
}

impl Coord for str {
    #[inline]
    fn coord(&self) -> u64 {
        fold_bytes(self.as_bytes())
    }
}
impl Coord for String {
    #[inline]
    fn coord(&self) -> u64 {
        fold_bytes(self.as_bytes())
    }
}
impl Coord for [u8] {
    #[inline]
    fn coord(&self) -> u64 {
        fold_bytes(self)
    }
}
impl<T: Coord + ?Sized> Coord for &T {
    const INJECTIVE: bool = T::INJECTIVE; // the reference is transparent: it delegates `coord`
    #[inline]
    fn coord(&self) -> u64 {
        (**self).coord()
    }
}
impl<A: Coord, B: Coord> Coord for (A, B) {
    #[inline]
    fn coord(&self) -> u64 {
        self.0.coord().wrapping_mul(PHI) ^ self.1.coord()
    }
}
impl<A: Coord, B: Coord, C: Coord> Coord for (A, B, C) {
    #[inline]
    fn coord(&self) -> u64 {
        (self.0.coord().wrapping_mul(PHI) ^ self.1.coord()).wrapping_mul(PHI) ^ self.2.coord()
    }
}

// ── The four merges: set, map, count, graph. ───────────────────────────────────────────────────

/// Which merge a table uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Face {
    /// Set membership (`||`).
    Keep,
    /// Map: last write wins.
    Set,
    /// Counter: commutative combine (`+`).
    Count,
    /// Graph: append.
    Link,
}

/// Whether a merge is order-free.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Species {
    /// Commutative (Set, Count) — order-free, so tables can be split across threads and recombined.
    Being,
    /// Directional (Map, Graph) — order matters, not fusible.
    Null,
}

impl Face {
    /// Commutative (fusible) vs directional (not).
    pub const fn species(self) -> Species {
        match self {
            Face::Keep | Face::Count => Species::Being,
            Face::Set | Face::Link => Species::Null,
        }
    }
}

// The four merges. An absent key reads as the merge's identity: false / (none) / 0 / [].

/// Set membership: `new || old`. identity = `false`.
#[inline]
pub fn keep(new: bool, old: bool) -> bool {
    new || old
}
/// Map: last write wins. identity = none (the `Option`).
#[inline]
pub fn set<V>(new: V, _old: V) -> V {
    new
}
/// Counter: combine via the monoid. identity = `Monoid::identity()`.
#[inline]
pub fn count<V: Monoid>(new: V, old: V) -> V {
    new.combine(old)
}
/// Graph: prepend `new` to `old` (newest-first). identity = `[]`.
#[inline]
pub fn link<T>(mut new: Vec<T>, old: Vec<T>) -> Vec<T> {
    new.extend(old);
    new
}

/// A commutative monoid for the count face. Commutativity is what makes counting order-free and
/// fusible.
pub trait Monoid {
    /// The identity element (what an absent key reads as).
    fn identity() -> Self;
    /// Combine two values for the same key (commutative and associative).
    #[must_use]
    fn combine(self, other: Self) -> Self;
}

macro_rules! int_monoid {
    ($($t:ty),*) => {$(
        impl Monoid for $t {
            #[inline] fn identity() -> Self { 0 }
            #[inline] fn combine(self, other: Self) -> Self { self + other }
        }
    )*};
}
int_monoid!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

macro_rules! float_monoid {
    ($($t:ty),*) => {$(
        impl Monoid for $t {
            #[inline] fn identity() -> Self { 0.0 }
            #[inline] fn combine(self, other: Self) -> Self { self + other }
        }
    )*};
}
// Honest scope of the float instance: `+` on floats is commutative but only approximately
// associative, so a float is not a monoid in the exact sense the integers are. What the count face
// and `Dense` need is weaker and true: `fuse` is elementwise (one addition per cell, each cell's
// summands fixed), so no regrouping of one cell's history happens and the result does not depend on
// the order cells are visited in.
float_monoid!(f32, f64);

// ── The table faces. Each is one `insert_with` with a fixed merge; the reader returns the identity
//    on absence. ────────────────────────────────────────────────────────────────────────────────

/// Map: overwrite (last write wins). Read with `get`.
impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    /// The map face's merge tag. (Named `FACE_SET`, not `FACE`, to avoid clashing with the per-`V` faces.)
    pub const FACE_SET: Face = Face::Set; // not `FACE`: would clash with the per-V faces below
    /// Insert or overwrite (last write wins).
    ///
    /// Routed through [`Table::insert_merge_last_write`](crate::map::hashtrinity::Table::insert_merge_last_write),
    /// the merge-free concrete face, so the body the extraction roots IS the executed map face and
    /// not a parallel path. It is the same `set(new, old) = new` merge `insert_with(k, v, set)`
    /// would apply, inlined: `insert_core` moves `old` out and blanks the slot either way, and
    /// both forms drop `old`, store `new`, and re-occupy with the same coordinate.
    #[inline]
    pub fn put(&mut self, k: K, v: V) {
        self.insert_merge_last_write(k, v);
    }
}

/// Set: membership. absent = `false`.
impl<K: Eq + Coord, KS: KeyStore<K>> Table<K, bool, KS> {
    /// The merge tag for the set face.
    pub const FACE: Face = Face::Keep;
    /// Mark `k` present.
    #[inline]
    pub fn mark(&mut self, k: K) {
        self.insert_with(k, true, keep);
    }
    /// Whether `k` is present.
    #[inline]
    pub fn has(&self, k: &K) -> bool {
        *self.get(k).unwrap_or(&false)
    }
}

/// Counter: `i64` (or any commutative monoid via `accumulate`). absent = `0`.
impl<K: Eq + Coord, KS: KeyStore<K>> Table<K, i64, KS> {
    /// The merge tag for the count face.
    pub const FACE: Face = Face::Count;
    /// Add `v` to `k`'s running total (starting from 0).
    #[inline]
    pub fn add(&mut self, k: K, v: i64) {
        let coord = Self::coord_of(&k);
        self.add_with_coord(coord, k, v);
    }

    /// [`add`](Self::add) with the coordinate precomputed — the pipelined build calls this so the
    /// `k·PHI` fold happens `P` keys ahead of the load it gates. `coord` is table-size-independent, so
    /// it stays valid across a resize (`probe_keyed` re-homes it), which keeps
    /// [`count_each`](Self::count_each) correct even if the batch grows the table.
    #[inline]
    fn add_with_coord(&mut self, coord: u64, k: K, v: i64) {
        if is_sentinel(coord) {
            // A reserved coordinate (`EMPTY`/`TOMB`) is held in the side store. Route it through the
            // shared core: the old coordinate-only path wrote the marker into `coords`, where `probe`
            // and every liveness test read it back as empty, silently dropping the key.
            self.insert_with(k, v, count::<i64>);
            return;
        }
        self.grow_for_one();
        match self.probe_capped_keyed(coord, &k) {
            Capped::Hit(i) => {
                // SAFETY: found ⇒ value initialized; i64 add in place — no move, no panic, no drop.
                *unsafe { self.vals.get_unchecked_mut(i).assume_init_mut() } += v; // entry().or_insert() += v
            }
            Capped::Vacant(i) => {
                unsafe { *self.coords.get_unchecked_mut(i) = coord };
                unsafe { *self.vals.get_unchecked_mut(i) = MaybeUninit::new(v) };
                self.keys.set(i, k);
                self.len += 1;
            }
            Capped::Blocked => {
                self.overflow
                    .get_or_insert_with(Overflow::new)
                    .upsert(coord, k, v, count::<i64>);
            }
        }
    }

    /// **The pipelined aggregation build** — `add(k, 1)` over a key stream with the same `P`-deep
    /// prefetch pipeline that makes [`get_each`](Self::get_each) win: the home line for key `i+P` is
    /// prefetched while key `i` is counted, so the per-key load-to-use miss (`optimal_depth_hides`,
    /// `MemoryLevelParallelism`) is hidden behind independent work instead of stalling. The reads are
    /// counted in program order, so duplicate keys (the same group) accumulate correctly — the prefetch
    /// only warms the cache line, it never reorders the read-modify-write. Falls back to the plain loop
    /// when there is no miss to hide (small or cache-resident table). Pre-size the table
    /// (`with_capacity`/`coordinate_with_capacity`) so a resize doesn't waste the in-flight prefetches.
    pub fn count_each(&mut self, keys: &[K])
    where
        K: Copy,
    {
        const P: usize = 16; // power of two ⇒ ring index is a mask
        let n = keys.len();
        if n <= P || self.footprint_bytes() <= (1 << 19) {
            for &k in keys {
                self.add(k, 1);
            }
            return;
        }
        let mut coords = [0u64; P];
        for (i, c) in coords.iter_mut().enumerate() {
            *c = Self::coord_of(&keys[i]);
            self.prefetch_coord(*c);
        }
        for i in 0..n {
            let c = coords[i & (P - 1)];
            self.add_with_coord(c, keys[i], 1);
            let j = i + P;
            if j < n {
                let cj = Self::coord_of(&keys[j]);
                coords[j & (P - 1)] = cj; // j ≡ i (mod P): the ring slot we just consumed
                self.prefetch_coord(cj);
            }
        }
    }
    /// `k`'s running total (0 if absent).
    #[inline]
    pub fn count(&self, k: &K) -> i64 {
        *self.get(k).unwrap_or(&0)
    }
}

impl<K: Eq + Coord, V: Monoid, KS: KeyStore<K>> Table<K, V, KS> {
    /// Fold `v` into `k`'s value under the monoid (the generic count face).
    #[inline]
    pub fn accumulate(&mut self, k: K, v: V) {
        self.insert_with(k, v, count);
    }
}

/// Graph: `push` appends in insertion order; `link` prepends (newest-first). absent = `[]`.
impl<K: Eq + Coord, T, KS: KeyStore<K>> Table<K, Vec<T>, KS> {
    /// The merge tag for the graph face.
    pub const FACE: Face = Face::Link;
    /// Append `t` to `k`'s adjacency list (insertion order).
    #[inline]
    pub fn push(&mut self, k: K, t: T) {
        self.insert_with(k, vec![t], |mut new, mut old| {
            old.append(&mut new);
            old
        });
    }
    /// Prepend `t` to `k`'s adjacency list (newest-first).
    #[inline]
    pub fn link(&mut self, k: K, t: T) {
        self.insert_with(k, vec![t], link);
    }
    /// `k`'s adjacency list (empty if absent).
    #[inline]
    pub fn neighbors(&self, k: &K) -> &[T] {
        self.get(k).map_or(&[], Vec::as_slice)
    }
}

// ── Loop replacement: collect an iterator straight into the right table. ───────────────────────

/// Build a `Table` from an iterator in one call — replaces the `for x { entry().or_insert() }` loop.
///
/// Each collector pre-sizes from the iterator's `size_hint`, so a sized source builds in one
/// allocation — no repeated `grow` (which, for byte keys, would re-fold every key each doubling).
pub trait FoldExt: Iterator + Sized {
    /// Count by key. Replaces `for x { *m.entry(k(x)).or_insert(0) += 1 }`.
    fn count_by<K: Eq + Coord>(self, mut key: impl FnMut(Self::Item) -> K) -> Table<K, i64> {
        let mut t = Table::with_capacity(self.size_hint().0);
        for x in self {
            t.add(key(x), 1);
        }
        t
    }
    /// Last value per key. Replaces `for x { let (k,v)=kv(x); m.insert(k,v); }`.
    fn index_by<K: Eq + Coord, V>(self, mut kv: impl FnMut(Self::Item) -> (K, V)) -> Table<K, V> {
        let mut t = Table::with_capacity(self.size_hint().0);
        for x in self {
            let (k, v) = kv(x);
            t.put(k, v);
        }
        t
    }
    /// Group values per key, insertion order. Replaces `entry(k).or_default().push(v)`.
    fn group_by<K: Eq + Coord, V>(
        self,
        mut kv: impl FnMut(Self::Item) -> (K, V),
    ) -> Table<K, Vec<V>> {
        let mut t = Table::with_capacity(self.size_hint().0);
        for x in self {
            let (k, v) = kv(x);
            t.push(k, v);
        }
        t
    }
    /// Membership set. Replaces a `HashSet` insert loop.
    fn set_by<K: Eq + Coord>(self, mut key: impl FnMut(Self::Item) -> K) -> Table<K, bool> {
        let mut t = Table::with_capacity(self.size_hint().0);
        for x in self {
            t.mark(key(x));
        }
        t
    }
    /// Arbitrary merge — the escape hatch when none of the four faces fit.
    fn fold_table<K: Eq + Coord, V>(
        self,
        mut kv: impl FnMut(Self::Item) -> (K, V),
        merge: impl Fn(V, V) -> V,
    ) -> Table<K, V> {
        let mut t = Table::with_capacity(self.size_hint().0);
        for x in self {
            let (k, v) = kv(x);
            t.insert_with(k, v, &merge);
        }
        t
    }
}

impl<I: Iterator> FoldExt for I {}

/// `iter.collect::<Table<_,_>>()` builds the map face.
impl<K: Eq + Coord, V> FromIterator<(K, V)> for Table<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut t = Table::with_capacity(iter.size_hint().0);
        for (k, v) in iter {
            t.put(k, v);
        }
        t
    }
}

/// `iter.collect::<CoordMap<_,_>>()` builds the coordinate map face — same collector, the
/// Regen/no-stored-keys path (sized at the dense load, `home(c)=c` collision-free).
impl<K: CoordInv + Eq, V> FromIterator<(K, V)> for Table<K, V, Regen<K>> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut t = Self::coordinate_with_capacity(iter.size_hint().0);
        for (k, v) in iter {
            t.put(k, v);
        }
        t
    }
}

// ── Total read and fusion. ─────────────────────────────────────────────────────────────────────

/// Total read: an absent key returns the monoid identity instead of `None`.
impl<K: Eq + Coord, V: Monoid + Clone, KS: KeyStore<K>> Table<K, V, KS> {
    /// `k`'s value, or the monoid identity if absent (a total read — never `None`).
    #[inline]
    pub fn fold(&self, k: &K) -> V {
        // A `match`, not `unwrap_or_else(V::identity)`: a `fn` item argument is an arrow type,
        // which the extraction's first-order boundary does not carry.
        match self.get(k) {
            Some(v) => v.clone(),
            None => V::identity(),
        }
    }
}

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    /// Fuse `other` into `self`: each key becomes `f(other, self)`. Only meaningful for a commutative
    /// merge (`keep`, `count`); `set` and `link` are directional, so the result would depend on the
    /// order the two tables were combined.
    pub fn fuse(&mut self, mut other: Table<K, V, KS>, mut f: impl FnMut(V, V) -> V) {
        for i in 0..other.coords.len() {
            let c = other.coords[i];
            if is_live(c) {
                // SAFETY: live coord ⇒ value initialized. Move it out and blank the source coord so
                // `other`'s Drop skips it (and any remaining slots still drop once if `f` panics).
                let v = unsafe { other.vals.get_unchecked_mut(i).assume_init_read() };
                other.coords[i] = EMPTY;
                let k = other.keys.take_key(i, c); // Stored: move; Regen: regenerate
                self.insert_with(k, v, &mut f);
            }
        }
        // The other table's reserved-coordinate keys live in its side store. Consume them too: a
        // `fuse` that walked only `coords` would drop one side's counts on the floor.
        while let Some(last) = other.side.coords.len().checked_sub(1) {
            let (_c, k, v) = other.side.take_entry(last);
            self.insert_with(k, v, &mut f);
        }
        // Its overflow entries as well — a `fuse` that walked the primary and side stores only would
        // drop the blocked keys' counts.
        if let Some(ov) = other.overflow.take() {
            for (_c, k, v) in ov.into_entries() {
                self.insert_with(k, v, &mut f);
            }
        }
    }
}
