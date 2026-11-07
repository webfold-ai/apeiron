# A Table

`apeiron::freq::Table<K, V>` is a flat hash table. Its default form has no random seed: a key's
address is a function of the key, so the layout is deterministic. The probe length is capped at
`PROBE_CAP = 16`; a key that cannot be placed within the cap goes to an AVL overflow tree. A crafted
key set therefore cannot drive a lookup into unbounded probing, but its keys are still public-fold
aimed, so each lookup on them costs `O(log n)` in the overflow instead of `O(1)` (`tests/hashdos.rs`).
For keys an attacker chooses, `Table::with_random_seed()` keys every address with a secret.

`insert_with(key, new, merge)` is the only mutator. When the key is absent, `new` is stored. When
it is present, `merge(old, new)` replaces the stored value. The merge function decides what kind
of table you have.

```rust
extern crate apeiron;
use apeiron::freq::Table;

let mut counts = Table::<u64, i64>::new();
for word in [7u64, 9, 7, 7, 9, 4] {
    counts.insert_with(word, 1, |old, new| old + new);
}
assert_eq!(counts.get(&7), Some(&3));
assert_eq!(counts.get(&9), Some(&2));
assert_eq!(counts.get(&5), None);
assert_eq!(counts.len(), 3);
```

A last-write map uses `|_, new| new`. A set stores `()` and merges with `|_, _| ()`.

## Coordinate keys

When keys are dense integer coordinates that you must resolve exactly, `CoordMap` does not store
the key at all. The address regenerates it, so a hit returns the exact key and a miss costs one
probe.

```rust
extern crate apeiron;
use apeiron::freq::CoordMap;

let mut grid = CoordMap::<u64, i64>::coordinate_with_capacity(0);
grid.insert_with(1_000_003, 5, |old, new| old + new);
grid.insert_with(1_000_003, 5, |old, new| old + new);
assert_eq!(grid.get(&1_000_003), Some(&10));
assert_eq!(grid.get(&1_000_004), None);
```

## Which table

| need | type | key storage |
|---|---|---|
| opaque keys, exact lookup, read the keys back | `Table<K, V>` | stored and compared |
| dense coordinate keys, exact lookup | `CoordMap<K, V>` | regenerated from the address |
| counting, grouping, deduplication | `FreqMap<K, V>` | folded to 64 bits, `2^-64` false merge for random keys |

The first two resolve every key exactly. `FreqMap` trades that for speed and footprint and
suits analytical work over keys you trust, where a `2^-64` false merge is negligible for random keys. The
[theory chapter](../theory/table.md) states the address rule and the bound; the
[verification chapter](../verification/evidence.md) states what is proved about this body and what is open.
