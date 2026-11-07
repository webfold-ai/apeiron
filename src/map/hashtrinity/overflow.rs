// The deterministic overflow behind the probe cap.
//! The deterministic overflow store for [`Table`](super::Table) — one table-wide AVL tree in an
//! arena, keyed by a key's *coordinate*.
//!
//!
//! The primary probe is capped at [`PROBE_CAP`](super::PROBE_CAP) slot examinations
//! ([`Table::probe_limit`](super::Table)). A key whose whole capped prefix held no `EMPTY` slot and
//! no identity is *blocked*: it is stored here instead, and every capped miss consults this store.
//! Placement is deterministic (no seed); the tree is balanced so the depth is `O(log n)` even when
//! an adversary aims every key at one home.
//!
//! **The invariant.** A key is here iff, when it was placed, its `PROBE_CAP` primary slots held no
//! `EMPTY` and no slot carrying its identity. Primary slots only ever change from `EMPTY` to
//! occupied (`TOMB`, not `EMPTY`, is what `remove` leaves; `resize` re-places everything through
//! this same capped placement), so the property holds for the life of a table generation. Hence on
//! lookup: a hit in the capped prefix is found; an `EMPTY` in it means absent (no lookup here); the
//! prefix exhausted means consult this store.
//!
//! **Arena, not `Box`.** Nodes live in one `Vec` indexed by `u32`; children are indices and a free
//! list recycles removed nodes. No `unsafe`, no recursive `Box`. The tree is AVL: every node's
//! subtree heights differ by at most one, so [`height`](Overflow::height) is `O(log len)`.
//!
//! **Coordinate key, full-key identity.** Entries with equal coordinates share a node (a `Coord`
//! that is not injective makes two distinct keys collide here exactly as they do in the primary
//! array; `Regen` keys, whose coordinate round-trips, have at most one entry per node).

use super::{
    is_sentinel, Capped, Coord, KeyStore, Locate, Table, EMPTY, LINEAR_G, PROBE_CAP,
};

/// `LINEAR_G + 64 / bits - 1` for `bits ∈ [0, 64]`, precomputed so the radix-walk width on the
/// hot capped probe is an indexed load, not a `divl`. `bits` is fixed at construction and never
/// `0` (the smallest table is `1 << 3`), so index `0` is a typed placeholder.
const WINDOW_STEPS_BY_BITS: [u32; 65] = {
    let mut t = [0u32; 65];
    let mut b = 1usize;
    while b < 65 {
        t[b] = LINEAR_G + 64 / b as u32 - 1;
        b += 1;
    }
    t
};

/// Whether primary slot `i` holds the probed identity, once its coordinate matched. A trait with two
/// first-order implementors instead of a closure argument: the extraction does not translate arrow
/// types, and the capped probe stays one body.
pub(super) trait Confirm<K: Coord, KS: KeyStore<K>> {
    fn confirm(&self, keys: &KS, i: usize) -> bool;
}

/// Confirmation against an owned-key query: [`KeyStore::matches`].
pub(super) struct ByKey<'a, K> {
    pub(super) coord: u64,
    pub(super) k: &'a K,
}

impl<'a, K: Eq + Coord, KS: KeyStore<K>> Confirm<K, KS> for ByKey<'a, K> {
    #[inline]
    fn confirm(&self, keys: &KS, i: usize) -> bool {
        // SAFETY: slot is live (the probe matched a live coordinate here).
        unsafe { keys.matches(i, self.coord, self.k) }
    }
}

/// The linear-prefix glance: compare the `LINEAR_G` contiguous coordinates `coords[home + k]` against
/// `coord` (a candidate hit) and against `EMPTY` (a chain terminator) in one branchless pass,
/// returning `(hits, empties)` as lane-bit masks. This is the read-side analogue of
/// [`group_tag_mask`], but on the coordinates *themselves*. The coordinate is identity over the live
/// space ([`coord_of`] is the raw coordinate; the reserved markers live in the [`SideStore`]), but two
/// distinct keys may share a coordinate, so a lane match is confirmed against the stored key (free on
/// `Regen`) before it is a hit. No second structure: the tag line THE CUT removed is not reintroduced;
/// the collision tail's own contiguous cache line is filtered in place, turning the mispredict-prone
/// scalar walk into one movemask. The hot home hit never calls this.
///
/// The prefix is addressed as a range of the coordinate slice (`home + k`), not as a raw pointer: the
/// scalar tail reads an in-bounds element, so the fallback needs no `unsafe` and the extraction sees no
/// raw-pointer dereference. The caller ensures `home + LINEAR_G ≤ coords.len()`.
#[inline]
fn prefix_masks(coords: &[u64], home: usize, coord: u64) -> (u32, u32) {
    // Compiled for AVX2 (`-C target-feature=+avx2`, e.g. `-C target-cpu=x86-64-v3`) the glance
    // resolves at *compile* time — the `_mm256` pair on the hot read with no per-call feature probe.
    // This repository ships no `.cargo/config.toml`, so the default build is portable (sub-v3) and
    // pays one `is_x86_feature_detected!` per collision read into the out-of-line AVX2 body, then
    // falls back to a branchless scalar loop the compiler still autovectorizes.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: caller guarantees `home + LINEAR_G ≤ len`; AVX2 is statically enabled here.
    return unsafe { prefix_masks_avx2(coords.as_ptr().add(home), coord) };

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: caller's read-length contract is exactly the two 4-lane loads avx2 does.
            return unsafe { prefix_masks_avx2(coords.as_ptr().add(home), coord) };
        }
        let (mut hits, mut empties) = (0u32, 0u32);
        // A `while`, not `for k in 0..LINEAR_G`: a range loop monomorphizes `Range<usize>::next`,
        // which resolves the `core::iter::range::Step` trait, and the extraction's pinned model
        // carries no `Step::forward_overflowing`. Bound, order, indexing and body are unchanged.
        let mut k = 0usize;
        while k < LINEAR_G as usize {
            let sc = coords[home + k];
            hits |= u32::from(sc == coord) << k;
            empties |= u32::from(sc == EMPTY) << k;
            k += 1;
        }
        (hits, empties)
    }
}

/// AVX2 body of [`prefix_masks`]: two `_mm256_cmpeq_epi64` + `_mm256_movemask_pd` pairs give one bit
/// per 64-bit lane, so the eight coordinates fold to two 4-bit masks each.
///
/// `#[inline]` is what makes the `#[cfg(target_feature = "avx2")]` arm of [`prefix_masks`] actually
/// inline: a `#[target_feature(enable = "avx2")]` callee can be inlined only into a caller that also
/// has AVX2, so on an AVX2 build this folds into the hot read, while on the portable default build
/// the runtime-checked call correctly stays out of line. (`#[inline(always)]` is rejected on a
/// `#[target_feature]` callee.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn prefix_masks_avx2(base: *const u64, coord: u64) -> (u32, u32) {
    use std::arch::x86_64::{
        __m256i, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_loadu_si256, _mm256_movemask_pd,
        _mm256_set1_epi64x,
    };
    // SAFETY: caller guarantees 8 readable lanes; the two loads cover lanes 0..4 and 4..8.
    let (lo, hi) = unsafe {
        (
            _mm256_loadu_si256(base.cast::<__m256i>()),
            _mm256_loadu_si256(base.add(4).cast::<__m256i>()),
        )
    };
    let key = _mm256_set1_epi64x(coord as i64);
    let emp = _mm256_set1_epi64x(EMPTY as i64);
    let hits_lo = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(lo, key))) as u32;
    let hits_hi = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(hi, key))) as u32;
    let empties_lo = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(lo, emp))) as u32;
    let empties_hi = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(hi, emp))) as u32;
    let hits = hits_lo | (hits_hi << 4);
    let empties = empties_lo | (empties_hi << 4);
    (hits, empties)
}

/// The [`Overflow::upsert`] merge guard: armed after the value is moved out, forgotten after the
/// merged value is written back. Its `Drop` runs only on an unwind out of the merge and deletes the
/// entry, the overflow twin of [`MergeGuard`](super::side::MergeGuard).
struct UpsertGuard<'a, K: Coord, V> {
    o: &'a mut Overflow<K, V>,
    n: u32,
    e: u32,
}

impl<K: Coord, V> Drop for UpsertGuard<'_, K, V> {
    fn drop(&mut self) {
        self.o.discard_entry(self.n, self.e);
    }
}

/// The `NIL` child index — no node. `u32::MAX` is not a valid arena index (the arena would need
/// `2^32` live nodes).
const NIL: u32 = u32::MAX;

/// One AVL node: a coordinate, the entries sharing it, and the child indices.
#[derive(Clone)]
pub(super) struct Node<K, V> {
    pub(super) key: u64,
    /// The entries at this coordinate. The `Option` is `None` only for the duration of a merge that
    /// has moved the value out; a merge that unwinds deletes the entry ([`Overflow::discard_entry`]),
    /// so every entry outside a merge is `Some`.
    pub(super) entries: Vec<(K, Option<V>)>,
    pub(super) left: u32,
    pub(super) right: u32,
    /// Subtree height: `0` for `NIL`, `1` for a leaf.
    pub(super) height: u32,
    /// Whether this arena slot currently holds a tree node (a freed slot is reused).
    pub(super) alive: bool,
}

/// The overflow store: an AVL tree in a `Vec` arena. No bounds on `K`/`V` at the type level so it
/// can be a field of [`Table<K, V, KS>`](super::Table) under the table's own `K: Coord` bound; the
/// coordinate-comparing operations carry `K: Eq`.
#[derive(Clone)]
pub(super) struct Overflow<K: Coord, V> {
    pub(super) nodes: Vec<Node<K, V>>,
    free: Vec<u32>,
    root: u32,
    len: usize,
}

impl<K: Coord, V> Default for Overflow<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Coord, V> Overflow<K, V> {
    pub(super) fn new() -> Self {
        Overflow {
            nodes: Vec::new(),
            free: Vec::new(),
            root: NIL,
            len: 0,
        }
    }

    /// The number of stored entries (not nodes): `len()` for the table.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// The tree height: `0` empty, `1` a single node. `O(log len)` by the AVL invariant.
    pub(super) fn height(&self) -> u32 {
        self.h(self.root)
    }

    fn h(&self, i: u32) -> u32 {
        if i == NIL {
            0
        } else {
            self.nodes[i as usize].height
        }
    }

    fn update(&mut self, i: u32) {
        let l = self.nodes[i as usize].left;
        let r = self.nodes[i as usize].right;
        let hl = self.h(l);
        let hr = self.h(r);
        let taller = if hl >= hr { hl } else { hr };
        let height = taller + 1;
        self.nodes[i as usize].height = height;
    }

    fn rotate_right(&mut self, y: u32) -> u32 {
        let x = self.nodes[y as usize].left;
        let t = self.nodes[x as usize].right;
        self.nodes[x as usize].right = y;
        self.nodes[y as usize].left = t;
        self.update(y);
        self.update(x);
        x
    }

    fn rotate_left(&mut self, x: u32) -> u32 {
        let y = self.nodes[x as usize].right;
        let t = self.nodes[y as usize].left;
        self.nodes[y as usize].left = x;
        self.nodes[x as usize].right = t;
        self.update(x);
        self.update(y);
        y
    }

    /// Restore the AVL invariant at `i` (whose children's heights are current), returning the root
    /// of the rebalanced subtree.
    fn rebalance(&mut self, i: u32) -> u32 {
        self.update(i);
        let l = self.nodes[i as usize].left;
        let r = self.nodes[i as usize].right;
        let hl = self.h(l);
        let hr = self.h(r);
        // The balance factor `hl - hr` as two comparisons: heights are unsigned, so the signed
        // difference (`bf > 1`, `bf < -1`) is never formed.
        if hl > hr + 1 {
            let lr = self.nodes[l as usize].right;
            if self.h(lr) > self.h(self.nodes[l as usize].left) {
                let nl = self.rotate_left(l);
                self.nodes[i as usize].left = nl;
            }
            return self.rotate_right(i);
        }
        if hr > hl + 1 {
            let rl = self.nodes[r as usize].left;
            if self.h(rl) > self.h(self.nodes[r as usize].right) {
                let nr = self.rotate_right(r);
                self.nodes[i as usize].right = nr;
            }
            return self.rotate_left(i);
        }
        i
    }

    /// Rebalance a root-to-deepest path bottom-up, re-linking each parent to the rotated subtree
    /// root. `chain[0]` is the tree root (its new root is written to `self.root`).
    fn rebalance_chain(&mut self, chain: &[u32]) {
        let mut idx = chain.len();
        while idx > 0 {
            idx -= 1; // visits chain.len()-1 down to 0
            let n = chain[idx];
            let newroot = self.rebalance(n);
            if idx == 0 {
                self.root = newroot;
            } else {
                let p = chain[idx - 1];
                if self.nodes[p as usize].left == n {
                    self.nodes[p as usize].left = newroot;
                } else {
                    self.nodes[p as usize].right = newroot;
                }
            }
        }
    }

    fn alloc(&mut self, key: u64, k: K, v: V) -> u32 {
        let node = Node {
            key,
            entries: vec![(k, Some(v))],
            left: NIL,
            right: NIL,
            height: 1,
            alive: true,
        };
        if let Some(i) = self.free.pop() {
            self.nodes[i as usize] = node;
            i
        } else {
            self.nodes.push(node);
            (self.nodes.len() - 1) as u32
        }
    }

    fn free_node(&mut self, i: u32) {
        let n = &mut self.nodes[i as usize];
        n.entries.clear();
        n.left = NIL;
        n.right = NIL;
        n.height = 0;
        n.alive = false;
        self.free.push(i);
    }

    fn find_node(&self, coord: u64) -> u32 {
        let mut cur = self.root;
        while cur != NIL {
            let ck = self.nodes[cur as usize].key;
            if coord == ck {
                return cur;
            }
            cur = if coord < ck {
                self.nodes[cur as usize].left
            } else {
                self.nodes[cur as usize].right
            };
        }
        NIL
    }

    /// The entry with coordinate `coord` and full key `k` — coordinate equality plus the exact key
    /// compare (a `Coord` that is not injective keeps its two distinct keys distinct here).
    pub(super) fn find_entry(&self, coord: u64, k: &K) -> Option<(u32, u32)>
    where
        K: Eq,
    {
        let n = self.find_node(coord);
        if n == NIL {
            return None;
        }
        let entries = &self.nodes[n as usize].entries;
        let mut e = 0;
        while e < entries.len() {
            if entries[e].0 == *k {
                return Some((n, e as u32));
            }
            e += 1;
        }
        None
    }

    /// The entry a borrowed view `q` names (coordinate plus the view's own compare).
    pub(super) fn find_locate<Q: Locate<K>>(&self, coord: u64, q: &Q) -> Option<(u32, u32)>
    where
        K: Eq,
    {
        let n = self.find_node(coord);
        if n == NIL {
            return None;
        }
        self.nodes[n as usize]
            .entries
            .iter()
            .position(|(ek, _)| q.same(ek))
            .map(|e| (n, e as u32))
    }

    // Kept out of line: `get_coord`'s inlined hot window check otherwise pulls the whole AVL
    // search into `face_get`, doubling the read body and regressing the cache-resident read cells
    // (measured: present-hit 1e3 +25% without this). The overflow walk is cold on these tables.
    #[inline(never)]
    pub(super) fn get(&self, coord: u64, k: &K) -> Option<&V>
    where
        K: Eq,
    {
        let (n, e) = self.find_entry(coord, k)?;
        self.nodes[n as usize].entries[e as usize].1.as_ref()
    }

    pub(super) fn get_mut(&mut self, coord: u64, k: &K) -> Option<&mut V>
    where
        K: Eq,
    {
        let (n, e) = self.find_entry(coord, k)?;
        self.nodes[n as usize].entries[e as usize].1.as_mut()
    }

    /// Insert a genuinely new entry (the caller has established absence). Returns a borrow of the
    /// stored value.
    pub(super) fn insert_new(&mut self, coord: u64, k: K, v: V) -> &mut V {
        let (node, e) = self.insert_new_at(coord, k, v);
        self.nodes[node as usize].entries[e].1.as_mut().unwrap()
    }

    /// [`insert_new`](Self::insert_new) without the borrow: returns the `(node, entry)` index the
    /// value landed at. The callers that discard the borrow use this form, which is first-order.
    pub(super) fn insert_new_at(&mut self, coord: u64, k: K, v: V) -> (u32, usize) {
        self.len += 1;
        let n = self.find_node(coord);
        let node = if n == NIL {
            let i = self.alloc(coord, k, v);
            self.insert_node(i);
            i
        } else {
            self.nodes[n as usize].entries.push((k, Some(v)));
            n
        };
        let e = self.nodes[node as usize].entries.len() - 1;
        (node, e)
    }

    /// Upsert: merge `new` into the running value at `(coord, k)` when present, else insert. A merge
    /// that unwinds deletes the entry ([`UpsertGuard`]): the key reads absent and is not counted.
    pub(super) fn upsert(&mut self, coord: u64, k: K, new: V, f: impl FnOnce(V, V) -> V) -> &mut V
    where
        K: Eq,
    {
        if let Some((n, e)) = self.find_entry(coord, &k) {
            drop(k); // the stored key is kept
            let old = self.take_value(n, e);
            let guard = UpsertGuard { o: self, n, e };
            let merged = f(new, old);
            std::mem::forget(guard);
            self.write_back(n, e, merged);
            return self.nodes[n as usize].entries[e as usize].1.as_mut().unwrap();
        }
        self.insert_new(coord, k, new)
    }

    /// Delete the entry at `(n, e)` whose value an unwinding merge consumed (it is `None`): drop its
    /// key, and its node when the node empties. O(log n): the node removal rebalances one path.
    pub(super) fn discard_entry(&mut self, n: u32, e: u32) {
        let (k, _consumed) = self.nodes[n as usize].entries.remove(e as usize);
        self.len -= 1;
        if self.nodes[n as usize].entries.is_empty() {
            self.delete_node(n);
        }
        drop(k);
    }

    /// Move the value out of an existing entry, leaving it `None` for the duration of a merge.
    pub(super) fn take_value(&mut self, n: u32, e: u32) -> V {
        self.nodes[n as usize].entries[e as usize]
            .1
            .take()
            .expect("overflow: live entry")
    }

    /// Write a merged value back into an entry blanked by [`take_value`](Self::take_value).
    pub(super) fn write_back(&mut self, n: u32, e: u32, v: V) {
        self.nodes[n as usize].entries[e as usize].1 = Some(v);
    }

    /// Remove the entry at `(n, e)` and return it. Removes the node when it becomes empty.
    pub(super) fn take_entry(&mut self, n: u32, e: u32) -> (K, V) {
        let (k, v) = self.nodes[n as usize].entries.remove(e as usize);
        self.len -= 1;
        if self.nodes[n as usize].entries.is_empty() {
            self.delete_node(n);
        }
        (k, v.expect("overflow: live entry"))
    }

    /// Remove `(coord, k)` and return its value.
    pub(super) fn remove(&mut self, coord: u64, k: &K) -> Option<V>
    where
        K: Eq,
    {
        let (n, e) = self.find_entry(coord, k)?;
        Some(self.take_entry(n, e).1)
    }

    /// Whether any entry at `coord` has value equal to `value` — the `PartialEq` scan.
    pub(super) fn contains_value(&self, coord: u64, value: &V) -> bool
    where
        V: PartialEq,
    {
        let n = self.find_node(coord);
        if n == NIL {
            return false;
        }
        self.nodes[n as usize]
            .entries
            .iter()
            .any(|(_, v)| v.as_ref() == Some(value))
    }

    /// Every live entry in node-key order, as `(coordinate, &key, &value)`.
    pub(super) fn iter(&self) -> impl Iterator<Item = (u64, &K, &V)> + '_ {
        let mut order = Vec::new();
        self.in_order(&mut order);
        order.into_iter().flat_map(move |n| {
            let node = &self.nodes[n as usize];
            node.entries
                .iter()
                .filter_map(move |(k, v)| v.as_ref().map(|v| (node.key, k, v)))
        })
    }

    /// Consume the store, yielding every live `(coordinate, key, value)`.
    pub(super) fn into_entries(mut self) -> Vec<(u64, K, V)> {
        let mut out = Vec::with_capacity(self.len);
        let mut order = Vec::new();
        self.in_order(&mut order);
        for n in order {
            let node = &mut self.nodes[n as usize];
            let key = node.key;
            let entries = std::mem::take(&mut node.entries);
            for (k, v) in entries {
                if let Some(v) = v {
                    out.push((key, k, v));
                }
            }
        }
        out
    }

    /// Keep only the entries `keep` returns true for. Rebuilds the tree (not a hot path).
    pub(super) fn retain(&mut self, mut keep: impl FnMut(&K, &mut V) -> bool) {
        let old = std::mem::replace(self, Overflow::new());
        for (coord, k, mut v) in old.into_entries() {
            if keep(&k, &mut v) {
                self.insert_new(coord, k, v);
            }
        }
    }

    /// The in-order node indices, iteratively (no recursion).
    pub(super) fn in_order(&self, out: &mut Vec<u32>) {
        let mut stack: Vec<u32> = Vec::new();
        let mut cur = self.root;
        while cur != NIL || !stack.is_empty() {
            while cur != NIL {
                stack.push(cur);
                cur = self.nodes[cur as usize].left;
            }
            let n = stack.pop().expect("nonempty stack");
            if self.nodes[n as usize].alive {
                out.push(n);
            }
            cur = self.nodes[n as usize].right;
        }
    }

    /// BST insertion of the fresh node `i` (its key is not yet in the tree) followed by a bottom-up
    /// rebalance.
    fn insert_node(&mut self, i: u32) {
        if self.root == NIL {
            self.root = i;
            return;
        }
        let key = self.nodes[i as usize].key;
        let mut chain: Vec<u32> = Vec::with_capacity(64);
        let mut cur = self.root;
        loop {
            chain.push(cur);
            let ck = self.nodes[cur as usize].key;
            if key < ck {
                let l = self.nodes[cur as usize].left;
                if l == NIL {
                    self.nodes[cur as usize].left = i;
                    break;
                }
                cur = l;
            } else {
                let r = self.nodes[cur as usize].right;
                if r == NIL {
                    self.nodes[cur as usize].right = i;
                    break;
                }
                cur = r;
            }
        }
        self.rebalance_chain(&chain);
    }

    /// Physically remove node `target` from the tree (its entries are already empty) and rebalance.
    fn delete_node(&mut self, target: u32) {
        // Ancestors of `target`, root first.
        let mut chain: Vec<u32> = Vec::new();
        let tkey = self.nodes[target as usize].key;
        let mut cur = self.root;
        while cur != target {
            chain.push(cur);
            let ck = self.nodes[cur as usize].key;
            cur = if tkey < ck {
                self.nodes[cur as usize].left
            } else {
                self.nodes[cur as usize].right
            };
        }
        let l = self.nodes[target as usize].left;
        let r = self.nodes[target as usize].right;
        if l == NIL || r == NIL {
            let replacement = if l != NIL { l } else { r };
            if chain.is_empty() {
                self.root = replacement;
            } else {
                let p = *chain.last().unwrap();
                if self.nodes[p as usize].left == target {
                    self.nodes[p as usize].left = replacement;
                } else {
                    self.nodes[p as usize].right = replacement;
                }
            }
            self.free_node(target);
            self.rebalance_chain(&chain);
            return;
        }
        // Two children: the in-order successor is the leftmost of the right subtree. Move its
        // payload up into `target`, then physically delete the successor (which has no left child).
        let mut succ_chain: Vec<u32> = Vec::new();
        let mut s = r;
        while self.nodes[s as usize].left != NIL {
            succ_chain.push(s);
            s = self.nodes[s as usize].left;
        }
        let succ = s;
        let (skey, sentries) = {
            let n = &mut self.nodes[succ as usize];
            (n.key, std::mem::take(&mut n.entries))
        };
        self.nodes[target as usize].key = skey;
        self.nodes[target as usize].entries = sentries;
        let succ_right = self.nodes[succ as usize].right;
        if succ_chain.is_empty() {
            // The successor was `target`'s right child.
            self.nodes[target as usize].right = succ_right;
        } else {
            let sp = *succ_chain.last().unwrap();
            self.nodes[sp as usize].left = succ_right;
        }
        self.free_node(succ);
        chain.push(target);
        chain.extend_from_slice(&succ_chain);
        self.rebalance_chain(&chain);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// The capped primary probe, on `Table`. Split out of `mod.rs` to keep that file under the 1,500-line
// bound; the executing probe and diagnostics are the same bodies.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

impl<K: Eq + Coord, V, KS: KeyStore<K>> Table<K, V, KS> {
    /// The number of primary probe steps the radix walk offers at this table width: the home slot,
    /// the `LINEAR_G` linear-prefix slots, then the fresh radix windows of `coord·PHI`. Steps at or
    /// past this index are the removed terminal sweep — `descend`'s last arm — and the capped probe
    /// never takes them; such a key is either placed within the walk or routed to the overflow.
    ///
    /// `bits` is fixed per table, so the width is one indexed load from
    /// [`WINDOW_STEPS_BY_BITS`] rather than a `divl` (`64 / self.bits`) on every capped probe.
    /// The value feeds the `limit >= LINEAR_G` branch and the scalar tail's bound on the hot
    /// miss path, where the hardware divide's latency is exposed. The table is 256 bytes.
    #[inline]
    fn window_steps(&self) -> u32 {
        WINDOW_STEPS_BY_BITS[self.bits as usize]
    }

    /// The primary probe limit in slot examinations: [`PROBE_CAP`], or the whole radix walk when the
    /// word is spent sooner (large `bits`). At most `PROBE_CAP`.
    #[inline]
    fn probe_limit(&self) -> u32 {
        PROBE_CAP.min(self.window_steps())
    }

    /// The scalar tail of the capped probe, from step `j` to `limit`: one coordinate compare (plus
    /// the caller's confirmation) per step. Used past the SIMD-scanned linear prefix, and for the
    /// whole walk on the rare array-end wrap.
    #[cold]
    fn probe_rest_with(
        &self,
        coord: u64,
        mut j: u32,
        limit: u32,
        confirm: &impl Confirm<K, KS>,
    ) -> Capped {
        while j < limit {
            let i = self.descend(coord, j);
            // SAFETY: i ∈ [0, len) — descend() masks to the power-of-two array.
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == coord && confirm.confirm(&self.keys, i) {
                return Capped::Hit(i);
            }
            if sc == EMPTY {
                return Capped::Vacant(i);
            }
            j += 1;
        }
        Capped::Blocked
    }

    /// The capped primary probe from the caller-probed `home` slot — an inlined hot window check
    /// plus a cold out-of-line continuation.
    ///
    /// **Hot, no stack frame.** Given `home + LINEAR_G ≤ len` (the window does not wrap), one
    /// branchless glance (`prefix_masks`) decides the two common outcomes and returns them without a
    /// call: (a) a confirmed hit lane — including a hit lane before any `EMPTY` — is
    /// [`Capped::Hit`](super::Capped::Hit); (b) an `EMPTY` lane reached with no hit before it is
    /// [`Capped::Vacant`](super::Capped::Vacant). Everything else — the window wrap, a hit confirm
    /// rejected (no hit at all), and a full window with no `EMPTY` — is
    /// [`probe_capped_cold`](Self::probe_capped_cold).
    ///
    /// **The dispatch invariant.** The old body gated the prefix glance on `limit >= LINEAR_G`; here
    /// the hot path checks only `home + LINEAR_G ≤ len`. `limit >= LINEAR_G` always holds —
    /// `probe_limit = min PROBE_CAP (LINEAR_G + 64/bits - 1)` and `WINDOW_STEPS_BY_BITS[b] =
    /// 8 + 64/b - 1 ≥ 8` for `1 ≤ b ≤ 64`, `bits ∈ [3, 63]` — so the cold continuation re-derives it
    /// from the same field and reaches the identical branch. This is the invariant `pcw_spec`
    /// re-establishes from `bits > 0`. Indexing is unchecked (`home`/`descend` mask to the array); a
    /// reserved coordinate never reaches here.
    #[inline(always)]
    pub(super) fn probe_capped_with(
        &self,
        coord: u64,
        home: usize,
        confirm: &impl Confirm<K, KS>,
    ) -> Capped {
        if home + LINEAR_G as usize <= self.coords.len() {
            // `home + LINEAR_G ≤ len`, so the `LINEAR_G`-lane read stays in bounds. Lane 0 is
            // `home`, already probed by the caller; a match there cannot confirm, but scanning it
            // is free.
            let (mut hits, empties) = prefix_masks(&self.coords, home, coord);
            while hits != 0 {
                let slot = home + hits.trailing_zeros() as usize;
                if confirm.confirm(&self.keys, slot) {
                    return Capped::Hit(slot);
                }
                hits &= hits - 1;
            }
            if empties != 0 {
                return Capped::Vacant(home + empties.trailing_zeros() as usize);
            }
        }
        self.probe_capped_cold(coord, home, confirm)
    }

    /// The out-of-line continuation of [`probe_capped_with`](Self::probe_capped_with) — and the whole
    /// probe for [`probe_capped_keyed`](Self::probe_capped_keyed) (the insert path), which has not
    /// scanned the window. It is the original full body: the linear prefix is scanned in one
    /// branchless glance (`prefix_masks`), then the radix windows walk scalar up to
    /// [`probe_limit`](Self::probe_limit). No `EMPTY` within the cap and no identity ⇒
    /// [`Capped::Blocked`](super::Capped::Blocked), and the caller consults the overflow. Indexing is
    /// unchecked (`home`/`descend` mask to the array); a reserved coordinate never reaches here.
    ///
    /// `#[cold] #[inline(never)]` so this frame is not carried on the read path; the hot path in
    /// [`probe_capped_with`](Self::probe_capped_with) returns the hit and vacant verdicts itself and
    /// only reaches here for the wrapped window or a full window with no hit. On those inputs this
    /// body re-derives the identical masks and returns the identical verdict (`hits == 0` ⇒ the same
    /// `empties` vacancy; every hit lane re-confirmed in the same ascending order), which is what
    /// lets the hot return and this body agree — the statement `pcw_spec`/`pcw_loop_spec` carry.
    #[cold]
    #[inline(never)]
    fn probe_capped_cold(
        &self,
        coord: u64,
        home: usize,
        confirm: &impl Confirm<K, KS>,
    ) -> Capped {
        let limit = self.probe_limit();
        let len = self.coords.len();
        if limit >= LINEAR_G && home + LINEAR_G as usize <= len {
            // `home + LINEAR_G ≤ len`, so the `LINEAR_G`-lane read stays in bounds. Lane 0 is
            // `home`, already probed by the caller; a match there cannot confirm, but scanning it
            // is free.
            let (mut hits, empties) = prefix_masks(&self.coords, home, coord);
            while hits != 0 {
                let slot = home + hits.trailing_zeros() as usize;
                if confirm.confirm(&self.keys, slot) {
                    return Capped::Hit(slot);
                }
                hits &= hits - 1;
            }
            if empties != 0 {
                return Capped::Vacant(home + empties.trailing_zeros() as usize);
            }
            self.probe_rest_with(coord, LINEAR_G, limit, confirm)
        } else {
            // Wrap tail (rare) or a cap below the prefix width: the scalar walk from step 0. The
            // home slot is examined here — this probe is also reached directly from
            // `probe_capped_keyed`, which has not probed `home` yet.
            self.probe_rest_with(coord, 0, limit, confirm)
        }
    }

    /// [`probe_capped_cold`](Self::probe_capped_cold) preceded by a home check: a coordinate match
    /// is a hit only when the stored key equals `k`, so two distinct keys with equal coordinates are
    /// two entries. For `Regen`, `matches` is the coordinate round-trip.
    ///
    /// The home slot is probed with one scalar compare first — the near-certain case for every
    /// insert and a home hit — so the `#[cold]` SIMD prefix glance is paid only on a collision. The
    /// cold body re-examines `home` in its lane 0; that redundant read is on the cold path. This
    /// entry keeps the whole probe out of line (the inlined read fast path is not duplicated into
    /// every insert loop).
    #[inline]
    pub(super) fn probe_capped_keyed(&self, coord: u64, k: &K) -> Capped {
        let home = self.home(coord);
        // SAFETY: home ∈ [0, len).
        let sc = unsafe { *self.coords.get_unchecked(home) };
        // SAFETY: slot is live (the probe matched a live coordinate here).
        if sc == coord && unsafe { self.keys.matches(home, coord, k) } {
            return Capped::Hit(home);
        }
        if sc == EMPTY {
            return Capped::Vacant(home);
        }
        // The insert path keeps the out-of-line full probe: it has not scanned the window, and the
        // inlined read fast path must not be duplicated into every insert loop.
        self.probe_capped_cold(coord, home, &ByKey { coord, k })
    }

    /// Whether this table holds `value` at `coord`, by coordinate: a live primary slot with
    /// `coord == c` and an equal value, or an overflow entry at `c` with an equal value.
    /// Coordinate-only — equal coordinates conflate, exactly as the pre-overflow comparison did.
    pub(super) fn holds_value_at(&self, coord: u64, value: &V) -> bool
    where
        V: PartialEq,
    {
        let limit = self.probe_limit();
        let mut i = self.home(coord);
        let mut j = 0u32;
        loop {
            // SAFETY: i ∈ [0, len) — home()/descend() mask to the array; `sc == coord` names a live
            // slot whose value is initialized.
            let sc = unsafe { *self.coords.get_unchecked(i) };
            if sc == coord && unsafe { self.vals.get_unchecked(i).assume_init_ref() } == value {
                return true;
            }
            if sc == EMPTY || j + 1 >= limit {
                break;
            }
            j += 1;
            i = self.descend(coord, j);
        }
        self.overflow
            .as_ref()
            .is_some_and(|o| o.contains_value(coord, value))
    }

    /// Worst-case probe length, in steps. Coordinate keys: **1** (address = key, nothing to flood).
    /// Folded keys: collisions descend fresh windows, and `coord·PHI` is a bijection, so distinct
    /// keys can't fully collide — a pinned home slot builds no pile of ALIASES. That is not a bound
    /// on this number (`AggregationBound.descent_can_be_wholly_blocked`); the probe is capped at
    /// [`PROBE_CAP`] primary slots, and every key unresolved within the cap lives in the overflow.
    /// Read it as a measurement. Diagnostic — not part of the stable surface.
    #[doc(hidden)]
    pub fn longest_probe(&self) -> usize {
        let limit = self.probe_limit() as usize;
        let mut worst = 0;
        for (slot, sc) in self.coords.iter().enumerate() {
            if super::is_live(*sc) {
                let mut i = self.home(*sc);
                let mut j = 0u32;
                while i != slot && (j as usize) + 1 < limit {
                    j += 1;
                    i = self.descend(*sc, j);
                }
                worst = worst.max(((j as usize) + 1).min(limit));
            }
        }
        worst
    }

    /// The number of primary slots a capped probe of `k` examines: `1` for a home hit or an empty
    /// home; otherwise the step index (≤ [`PROBE_CAP`]) of the hit, the terminating `EMPTY`, or the
    /// exhausted cap. Diagnostic — not part of the stable surface.
    #[doc(hidden)]
    pub fn primary_examinations(&self, k: &K) -> u32 {
        let coord = self.coord_of(k);
        if is_sentinel(coord) {
            return 0; // resolved in the side store, not a primary slot
        }
        let limit = self.probe_limit();
        let mut i = self.home(coord);
        let mut j = 0u32;
        loop {
            // SAFETY: i ∈ [0, len) — home()/descend() mask to the array.
            let sc = unsafe { *self.coords.get_unchecked(i) };
            // SAFETY: slot is live (the probe matched a live coordinate here).
            if sc == EMPTY || (sc == coord && unsafe { self.keys.matches(i, coord, k) }) || j + 1 >= limit {
                return j + 1;
            }
            j += 1;
            i = self.descend(coord, j);
        }
    }

    /// The overflow tree's height: `0` when no key has been blocked. Diagnostic — the AVL bound is
    /// `height ≤ 2·log2(len + 1) + 1`.
    #[doc(hidden)]
    pub fn overflow_height(&self) -> u32 {
        match &self.overflow {
            Some(o) => o.height(),
            None => 0,
        }
    }

    /// The number of keys held in the overflow tree. Diagnostic.
    #[doc(hidden)]
    pub fn overflow_len(&self) -> usize {
        match &self.overflow {
            Some(o) => o.len(),
            None => 0,
        }
    }

    /// The number of `TOMB` slots not yet reclaimed. Only a rebuild lowers it (every `remove` adds
    /// one and no insert reuses one), so it is the observable a rebuild count is read from.
    /// Diagnostic.
    #[doc(hidden)]
    pub fn tombstones(&self) -> usize {
        self.tombs
    }
}
