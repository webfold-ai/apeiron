//! The Count face at the identity address function — the table whose key *is* its address.
//!
//! The crate header's claim is *"the key is its own address when it can be. An integer key smaller
//! than the array indexes directly — no hash, no collision, one probe."* [`Dense`] is that claim
//! taken to its limit and made into a type: when the key space is a known-bounded coordinate range
//! that the caller has already linearised, the address function is the **identity**, there is no
//! probe at all, and liveness is not a separate fact — an absent cell *is* the merge's identity
//! element, which is the value already sitting there.
//!
//! This is the face [`super::Table`] and [`super::CoordMap`] cannot degenerate to. Both keep a
//! `coords: Box<[u64]>` liveness/identity column (8 bytes per **slot**, not per live key) in a
//! power-of-two array that must run below full to keep probes short. That is the right trade when
//! the occupied coordinates are a sparse and unknown subset of a large space. It is the wrong trade
//! when the caller can enumerate the whole space and most of it is live: the coordinate column then
//! stores what the index already says, and the load-factor slack buys nothing.
//!
//! `Dense` is therefore not an escape from the trinity — it is the trinity's B-face keystore
//! ([`super::Regen`]) with the fold collapsed to `id`, which deletes the coordinate column and the
//! probe together. What it keeps is exactly what the accumulator is *for*:
//!
//! | | |
//! |---|---|
//! | the merge | [`Monoid::combine`] — one `f`, the only freedom |
//! | absence | [`Monoid::identity`] — an absent key reads as the identity, never a sentinel |
//! | the merge of two tables | [`Dense::fuse`] — elementwise, order-free by construction |
//! | the group inverse | [`Dense::unfuse`] — exact, or refused ([`Group::remove`]) |
//! | locality | [`Dense::span`] — the contiguous-range GET, free because addresses are coordinates |
//!
//! `fuse`/`unfuse` are the two properties a count fold is bought for: **exactly mergeable** and
//! **exactly unfoldable**. Getting them from here rather than from a hand-written `for` loop per
//! call site is the whole point of the type — a hand-rolled `checked_sub` loop is where a group
//! inverse silently stops being one.
//!
//! ```
//! use hashtrinity::Dense;
//! let mut a = Dense::<u32>::zeros(4);
//! a.add(1, 5);
//! a.add(3, 2);
//! let mut b = Dense::<u32>::zeros(4);
//! b.add(1, 7);
//!
//! a.fuse(&b);
//! assert_eq!(a.get(1), 12);
//! assert_eq!(a.get(0), 0); // absent reads as the identity
//!
//! a.unfuse(&b); // the exact group inverse
//! assert_eq!(a.get(1), 5);
//! ```

use super::Monoid;

/// A [`Monoid`] whose combine has an inverse — what makes a count fold *unfoldable*.
///
/// [`remove`](Group::remove) is partial on purpose. `u32` under `+` is a monoid but only a
/// *cancellative* one: `a − b` exists exactly when `b ≤ a`. Returning `None` rather than wrapping
/// or clamping is the difference between an inverse and a fabrication — a clamped subtraction
/// leaves a table that is not `self − other` and never says so.
pub trait Group: Monoid {
    /// `self − other`, or `None` when the difference leaves the carrier.
    #[must_use]
    fn remove(self, other: Self) -> Option<Self>
    where
        Self: Sized;
}

macro_rules! uint_group {
    ($($t:ty),*) => {$(
        impl Group for $t {
            #[inline]
            fn remove(self, other: Self) -> Option<Self> { self.checked_sub(other) }
        }
    )*};
}
uint_group!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

macro_rules! float_group {
    ($($t:ty),*) => {$(
        impl Group for $t {
            #[inline] fn remove(self, other: Self) -> Option<Self> { Some(self - other) }
        }
    )*};
}
// The float `Monoid` impls live in apeiron (`apeiron::map::hashtrinity::Monoid`). Honest scope of
// the float instance (Axiom XIII): `+` on floats is commutative but only
// approximately associative, so `f64` is not a monoid in the exact sense the integer instances are.
// What `Dense` needs, and all it needs, is weaker and *is* true: `fuse`/`unfuse` are **elementwise**
// — one addition per cell, each cell's summands fixed — so no regrouping of summands can occur and
// the result is bit-reproducible regardless of the order cells are visited in. Associativity would
// be needed to re-bracket a *single cell's* history; `Dense` never does that. `unfuse` is likewise
// only a bit-exact inverse when the additions it undoes were exact (integers, or float values that
// summed without rounding); callers relying on exact unfolding of rounded floats must say so.
float_group!(f32, f64);

/// **The Count face at the identity address function.** A flat cell array where the key is the
/// index, absence is the monoid identity, and the merge is the only freedom.
///
/// See the [module docs](self) for when this is the right face and when [`super::CoordMap`] is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dense<V> {
    cells: Box<[V]>,
}

impl<V: Monoid + Clone> Dense<V> {
    /// A table of `n` cells, every one holding the merge's identity.
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Dense {
            cells: vec![V::identity(); n].into_boxed_slice(),
        }
    }
}

impl<V> Dense<V> {
    /// The number of addressable cells — the size of the key space, not the live count.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether the key space is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// The whole table as a slice, in coordinate order. Addresses are coordinates, so the
    /// enumeration order is the key order and is the same on every run.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[V] {
        &self.cells
    }

    /// **The contiguous-range GET** — locality, free, because the address *is* the coordinate.
    /// This is the read a keyed table cannot give you: `hi − lo` probes collapse to one slice.
    ///
    /// # Panics
    /// If the range is out of bounds or inverted.
    #[inline]
    #[must_use]
    pub fn span(&self, lo: usize, hi: usize) -> &[V] {
        &self.cells[lo..hi]
    }

    /// Iterate the cells in coordinate order.
    pub fn iter(&self) -> std::slice::Iter<'_, V> {
        self.cells.iter()
    }
}

impl<V: Monoid + Copy> Dense<V> {
    /// **`insert_with` at the Count face** — merge `v` into cell `i` via [`Monoid::combine`].
    ///
    /// # Panics
    /// If `i` is outside the key space.
    #[inline]
    pub fn add(&mut self, i: usize, v: V) {
        self.cells[i] = self.cells[i].combine(v);
    }

    /// Cell `i`'s value. An address nothing was ever added to reads as [`Monoid::identity`] — the
    /// identity is not a sentinel and there is no `Option`.
    ///
    /// # Panics
    /// If `i` is outside the key space.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize) -> V {
        self.cells[i]
    }

    /// **`self ← self ⊕ other`** — the monoid merge of two whole tables.
    ///
    /// Order-free by construction: the merge is elementwise, so each cell's result depends only on
    /// that cell's two summands and never on the order cells are visited in.
    ///
    /// # Panics
    /// If the two tables address different key spaces — tables over different coordinate ranges are
    /// not summable, and saying so loudly is the safety of the merge.
    pub fn fuse(&mut self, other: &Dense<V>) {
        assert_eq!(
            self.cells.len(),
            other.cells.len(),
            "Dense::fuse: different key spaces ({} vs {} cells)",
            self.cells.len(),
            other.cells.len()
        );
        for (a, &b) in self.cells.iter_mut().zip(other.cells.iter()) {
            *a = a.combine(b);
        }
    }
}

impl<V: Group + Copy> Dense<V> {
    /// **`self ← self ⊖ other`** — the group inverse of [`fuse`](Dense::fuse).
    ///
    /// Returns the first address at which the difference left the carrier, having applied nothing;
    /// `Ok(())` means every cell was subtracted exactly. Checking before writing is what keeps a
    /// refused unfold from leaving a half-subtracted table behind.
    ///
    /// # Panics
    /// If the two tables address different key spaces.
    ///
    /// # Errors
    /// The address of the first cell whose difference leaves the carrier (e.g. a count that would
    /// go negative) — the caller is removing something that was never folded in.
    pub fn try_unfuse(&mut self, other: &Dense<V>) -> Result<(), usize> {
        assert_eq!(
            self.cells.len(),
            other.cells.len(),
            "Dense::unfuse: different key spaces ({} vs {} cells)",
            self.cells.len(),
            other.cells.len()
        );
        for (i, (&a, &b)) in self.cells.iter().zip(other.cells.iter()).enumerate() {
            if a.remove(b).is_none() {
                return Err(i);
            }
        }
        for (a, &b) in self.cells.iter_mut().zip(other.cells.iter()) {
            *a = a.remove(b).expect("Dense::unfuse: checked above");
        }
        Ok(())
    }

    /// [`try_unfuse`](Dense::try_unfuse), panicking on a difference that leaves the carrier.
    ///
    /// # Panics
    /// If the tables address different key spaces, or if any cell's difference leaves the carrier.
    pub fn unfuse(&mut self, other: &Dense<V>) {
        if let Err(i) = self.try_unfuse(other) {
            panic!("Dense::unfuse: cell {i} would go negative — nothing to remove");
        }
    }
}

impl<'a, V> IntoIterator for &'a Dense<V> {
    type Item = &'a V;
    type IntoIter = std::slice::Iter<'a, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.cells.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_reads_as_the_identity() {
        let t = Dense::<u32>::zeros(8);
        assert_eq!(t.get(5), 0);
        assert_eq!(Dense::<f64>::zeros(3).get(0), 0.0);
    }

    #[test]
    fn fuse_is_order_free() {
        let mk = |v: &[u32]| {
            let mut t = Dense::<u32>::zeros(v.len());
            for (i, &x) in v.iter().enumerate() {
                t.add(i, x);
            }
            t
        };
        let (a, b, c) = (mk(&[1, 0, 3]), mk(&[4, 5, 0]), mk(&[0, 2, 7]));
        let mut abc = a.clone();
        abc.fuse(&b);
        abc.fuse(&c);
        let mut cba = c.clone();
        cba.fuse(&b);
        cba.fuse(&a);
        assert_eq!(abc, cba);
        let mut bc = b.clone();
        bc.fuse(&c);
        let mut a_bc = a.clone();
        a_bc.fuse(&bc);
        assert_eq!(abc, a_bc);
    }

    #[test]
    fn unfuse_inverts_fuse_exactly() {
        let mut a = Dense::<u64>::zeros(4);
        a.add(0, 9);
        a.add(2, 4);
        let before = a.clone();
        let mut b = Dense::<u64>::zeros(4);
        b.add(0, 3);
        b.add(3, 11);
        a.fuse(&b);
        a.unfuse(&b);
        assert_eq!(a, before);
    }

    #[test]
    fn unfuse_refuses_what_was_never_fused_and_writes_nothing() {
        let mut a = Dense::<u32>::zeros(3);
        a.add(0, 1);
        let mut b = Dense::<u32>::zeros(3);
        b.add(0, 1);
        b.add(2, 5);
        let before = a.clone();
        assert_eq!(a.try_unfuse(&b), Err(2));
        assert_eq!(a, before, "a refused unfold must leave the table untouched");
    }

    #[test]
    fn span_is_the_contiguous_get() {
        let mut t = Dense::<u32>::zeros(6);
        for i in 0..6 {
            t.add(i, i as u32);
        }
        assert_eq!(t.span(2, 5), &[2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "different key spaces")]
    fn fusing_different_key_spaces_is_refused() {
        Dense::<u32>::zeros(3).fuse(&Dense::<u32>::zeros(4));
    }
}
