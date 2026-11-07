//! The opt-in seeded table (`Table::with_seed` / `with_random_seed`) for attacker-chosen keys.
//!
//! The seeded table must be exact like the default one (it stores and compares the key), must key
//! every path that resolves a key (owned and borrowed), and must keep its secret through growth,
//! clone and drain. The flood resistance itself is a statement about aimed keys and is measured in
//! `hashdos`; these tests pin the equivalences the seeded path rests on.

// Test indices are small (< 2^32); wrapping is impossible here.
#![allow(clippy::cast_possible_wrap)]

use apeiron::map::hashtrinity::{Coord, FoldKey, Table};

#[test]
fn seeded_table_is_exact_on_coordinate_colliding_keys() {
    // 300 u128 keys that all share the unkeyed coordinate 7 (`lo ^ hi == 7`).
    let keys: Vec<u128> = (0..300u128).map(|i| (i << 64) | (i ^ 7)).collect();
    assert!(
        keys.iter().all(|k| k.coord() == 7),
        "premise: unkeyed collision"
    );
    let mut t: Table<u128, i64> = Table::with_random_seed();
    assert!(t.is_seeded());
    for (i, k) in keys.iter().enumerate() {
        t.add(*k, i as i64 + 1); // grows several times
    }
    assert_eq!(t.len(), keys.len());
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(t.get(k), Some(&(i as i64 + 1)), "key {i}");
    }
    assert_eq!(t.remove(&keys[7]), Some(8));
    assert_eq!(t.get(&keys[7]), None);
    assert_eq!(t.get(&keys[8]), Some(&9));
}

#[test]
fn keyed_coordinates_depend_on_the_secret_and_are_pure() {
    let (a, b) = (FoldKey::new(1), FoldKey::new(2));
    let s: &str = "a longer byte string than eight";
    assert_eq!(
        s.coord_keyed(a),
        s.coord_keyed(a),
        "pure in (key, fold key)"
    );
    assert_ne!(
        s.coord_keyed(a),
        s.coord_keyed(b),
        "a different secret moves the coordinate"
    );
    // `k0 == k1` would make the two halves symmetric; use a key that is not.
    let n = FoldKey::new(0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210);
    assert_ne!(
        5u128.coord_keyed(n),
        (5u128 << 64).coord_keyed(n),
        "halves enter separately"
    );
    assert_eq!(FoldKey::new(9), FoldKey::new(9));
    assert_eq!(format!("{a:?}"), "FoldKey(..)", "the secret is not printed");
}

#[test]
fn seeded_borrowed_reads_match_owned_keys() {
    let mut t: Table<String, i64> = Table::with_seed(FoldKey::new(0x5EED));
    for i in 0..200 {
        t.put(format!("key-{i:04}-with-a-long-tail"), i);
    }
    for i in 0..200 {
        let owned = format!("key-{i:04}-with-a-long-tail");
        assert_eq!(t.get(&owned), Some(&i));
        assert_eq!(t.get_at(owned.as_str()), Some(&i), "&str view of key {i}");
        assert!(t.contains_at(owned.as_str()));
    }
    assert_eq!(t.get_at("absent"), None);

    let mut p: Table<(String, String), i64> = Table::with_seed(FoldKey::new(7));
    p.put(("alpha-beta-gamma".into(), "delta".into()), 42);
    assert_eq!(
        p.get_at(("alpha-beta-gamma", "delta")),
        Some(&42),
        "tuple view"
    );
    assert_eq!(
        p.get(&("alpha-beta-gamma".into(), "delta".into())),
        Some(&42)
    );
}

#[test]
fn seed_survives_growth_clone_and_drain() {
    let mut t: Table<u64, i64> = Table::with_seed(FoldKey::new(3));
    for i in 0..5_000u64 {
        t.put(i.wrapping_mul(0x9E37), i as i64); // forces several resizes
    }
    for i in 0..5_000u64 {
        assert_eq!(t.get(&i.wrapping_mul(0x9E37)), Some(&(i as i64)));
    }
    let c = t.clone();
    assert!(c.is_seeded());
    assert_eq!(c, t);
    let before = t.len();
    let drained: Vec<_> = t.drain().collect();
    assert_eq!(drained.len(), before);
    assert!(t.is_seeded(), "a drained table keeps its secret");
    assert!(t.is_empty());
}

#[test]
fn default_table_is_unseeded() {
    let mut t: Table<String, i64> = Table::new();
    assert!(!t.is_seeded());
    t.add("x".to_string(), 1);
    assert_eq!(t.get_at("x"), Some(&1));
}
