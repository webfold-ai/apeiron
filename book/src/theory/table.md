# The table

`apeiron::freq::Table<K, V>` is the executed table: a flat array of 64-bit coordinates, a parallel
array of values, and a capped probe with a deterministic overflow tree. It is the port of the
`hashtrinity` table at the revision recorded in `src/map/freq.rs`, and it is the body the benchmarks and
the refinement target. `apeiron::Table` (`src/map/table.rs`) is the generic growable table specified by the corpus
(`Apeiron.Table.HMap`, `Apeiron.Table.get`, `Apeiron.Table.insertWith`, `Apeiron.Table.build`); it is slated for removal once the executed body is
verified, and this chapter does not describe it further.

## Layout

| array | holds | read when |
|---|---|---|
| coordinates (`u64` per slot) | identity and occupancy | every probe |
| values | payloads, initialised exactly where the coordinate is live | after a match |
| keys | the key, for opaque key types (`Stored`); absent for integer keys (`Regen`) | after a match, to confirm |
| side store | keys whose own coordinate is a reserved marker (`EMPTY`, `TOMB`) | only for those keys |
| overflow | keys blocked by the probe cap | only after an exhausted probe |

A slot holds a live value exactly when its coordinate is neither `EMPTY` (`u64::MAX`) nor `TOMB`
(`u64::MAX - 1`, left by a removal). There is no tag line: the full coordinate is the filter. The
table grows when `(len + tombstones + 1) · 4 ≥ capacity · 3`.

## The probe and its cap

A lookup examines the home slot, then the next seven slots (one 64-byte line of coordinates), then
fresh windows of address bits. It examines at most `PROBE_CAP = 16` primary slots. A key whose
capped prefix has no empty slot and no match is stored in one table-wide AVL tree keyed by
coordinate.

$$\text{examinations}(k) \le \mathrm{PROBE\_CAP} = 16,\qquad \text{height(overflow)} \le 2\log_2(n+1)+1$$

- **say:** a primary probe never exceeds 16 slots, and the overflow tree stays balanced.
- **means:** a crafted key set costs a bounded probe plus a logarithmic tree walk, not a linear chain.
- **vars:** `k` is a key; `n` is the number of keys in the overflow.
- **apeiron:** none — anchor. The statement is tested by `cargo test --test hashdos --locked`. The descent argument is the corpus's `HashDos.descent_can_be_wholly_blocked`, which says a crafted set can fill the whole capped prefix; the cap and the tree are what bound the damage.
- **physics:** none.

A lookup that finds an empty slot in the capped prefix is absent without consulting the overflow.
That holds because a primary slot only changes from `EMPTY` to occupied: `remove` leaves a
tombstone, and a resize re-places every entry through the same capped placement.

### Executed

Keys that are multiples of a power of two collide in every `hash & mask` table. Here they stay
within the cap and every key is found.

```rust
extern crate apeiron;
use apeiron::freq::{Table, PROBE_CAP};

let n = 4000u64;
let keys: Vec<u64> = (1..=n).map(|i| i << 20).collect();
let mut t: Table<u64, i64> = Table::new();
for k in &keys {
    t.add(*k, 1);
}
assert_eq!(t.len(), n as usize);
for k in &keys {
    assert_eq!(t.get(k), Some(&1));
    assert!(t.primary_examinations(k) <= PROBE_CAP);
}
```

An attacker who knows `PHI` can aim keys at one home slot: the key `i·PHI⁻¹` has product `i`, so
for small `i` its top bits are zero and it homes to slot 0. The flood below pins 12000 keys to slot
0 of a table sized for them. The primary probe stays within the cap, the excess goes to the
overflow, and every key still resolves.

```rust
extern crate apeiron;
use apeiron::freq::{Table, PHI, PROBE_CAP};

let phi_inv = (0..6).fold(1u64, |x, _| x.wrapping_mul(2u64.wrapping_sub(PHI.wrapping_mul(x))));
let n = 12_000u64;
let keys: Vec<u64> = (1..=n).map(|i| i.wrapping_mul(phi_inv)).collect();
let mut t: Table<u64, i64> = Table::with_capacity(n as usize);
for k in &keys {
    t.add(*k, 1);
}
assert_eq!(t.len(), n as usize);
assert!(t.overflow_len() > 0);
for k in &keys {
    assert_eq!(t.get(k), Some(&1));
    assert!(t.primary_examinations(k) <= PROBE_CAP);
}
```

## What the bound covers

- It covers keys whose coordinate is injective: `u64` and the other integer types. For a key with a lossy coordinate the overflow still stores entries correctly, comparing by full key, but the bound is a statement about the coordinate.
- `PHI` is public and the table has no secret seed, so the layout is deterministic and replays exactly. An adversary can fill a chosen key's capped prefix (`crafted_keys_block_a_victims_whole_capped_prefix` in `tests/hashdos.rs`); the lookup then costs the cap plus a tree walk and returns the correct value.
- The immunity on the fold path is measured and tested. It is not proved.

## Merge-defined kinds

One table body serves four uses. The merge function decides which.

| use | insert | read |
|---|---|---|
| set | `mark` | `has` |
| map | `put` | `get` |
| count | `add` | `count` |
| graph | `push` | `neighbors` |

`insert_with(k, new, f)` is the general form: `f(old, new)` resolves a clash. `CoordMap` keeps
integer keys as coordinates and regenerates them on iteration. `FreqMap` folds a key to 64 bits
and stores no key, so a non-member whose fold is uniform and independent of the member's matches it with probability `2^-64`. That bound is for random keys. The multiplier is public and `FreqMap` has no seeded mode, so a colliding key can be constructed for a chosen non-member, and keys whose `Coord` is not injective collide in closed form. `FreqMap::count_par`
counts a key slice in parallel.

## Complexity

| operation | time (slots examined) | space |
|---|---|---|
| `get`, `insert_with`, `remove` | at most `PROBE_CAP` primary, plus `O(log n_overflow)` after an exhausted probe | `O(1)` |
| growth | `O(len)` re-placements, amortised `O(1)` per insert at load factor 3/4 | `O(capacity)` |
