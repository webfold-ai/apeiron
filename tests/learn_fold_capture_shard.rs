//! Calibration shards merge exactly: `Grams::merge` over disjoint token shards equals one
//! accumulator over all tokens, bit for bit, in any order and with any shard count, and the wire
//! form round-trips. SYNTHETIC tap rows (seeded integers in a small range), tiny geometry: this
//! tests the integer algebra of the merge, not model quality.
//!
//! Controls: a shard with one flipped word must make the equality FAIL; a mismatched geometry and
//! an overflow must be refused with the target untouched.
//!
//! Grade: `tested`.

use apeiron::learn::fold::stats::capture::{CaptureError, Grams};
use apeiron::learn::grad::model::Config;

fn cfg() -> Config {
    Config {
        n_layer: 2,
        n_head: 2,
        n_head_kv: 1,
        n_embd: 6,
        n_ff: 10,
        vocab: 16,
        head_dim: 3,
        eps: 1,
        rope_theta: 10_000.0,
    }
}

/// `[layer][stage(12)][column][width]` Q32.32 words from a seeded xorshift, `t` columns.
fn taps(seed: u64, t: usize, c: &Config) -> Vec<Vec<Vec<Vec<i64>>>> {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (((s >> 40) as i64) % 2001 - 1000) << 20
    };
    let q_out = c.n_head * c.head_dim;
    (0..c.n_layer)
        .map(|_| {
            (0..12)
                .map(|stage| {
                    let w = match stage {
                        1 | 3 => c.n_embd,
                        2 => q_out,
                        4 => c.n_ff,
                        _ => 1,
                    };
                    (0..t).map(|_| (0..w).map(|_| next()).collect()).collect()
                })
                .collect()
        })
        .collect()
}

fn acc(parts: &[Vec<Vec<Vec<Vec<i64>>>>], c: &Config) -> Grams {
    let mut g = Grams::new(c).unwrap();
    for p in parts {
        g.accumulate_tap(p).unwrap();
    }
    g
}

#[test]
fn shard_merge_is_bit_exact_order_free_and_wire_stable() {
    let c = cfg();
    let shards: Vec<_> = (0..5).map(|i| taps(100 + i, 3 + i as usize, &c)).collect();
    let whole = acc(&shards, &c);

    // each shard accumulated on its own "machine", then merged in two different orders
    let each: Vec<Grams> = shards.iter().map(|s| acc(std::slice::from_ref(s), &c)).collect();
    let mut fwd = Grams::new(&c).unwrap();
    for g in &each {
        fwd.merge(g).unwrap();
    }
    let mut rev = Grams::new(&c).unwrap();
    for g in each.iter().rev() {
        rev.merge(g).unwrap();
    }
    assert_eq!(fwd, whole, "forward merge differs from single accumulation");
    assert_eq!(rev, whole, "reverse merge differs from single accumulation");

    // a tree merge: ((0+1)+(2+3))+4
    let mut a = each[0].clone();
    a.merge(&each[1]).unwrap();
    let mut b = each[2].clone();
    b.merge(&each[3]).unwrap();
    a.merge(&b).unwrap();
    a.merge(&each[4]).unwrap();
    assert_eq!(a, whole, "tree merge differs");

    // the wire form round-trips exactly, and merging deserialised shards is the same
    let mut over_wire = Grams::new(&c).unwrap();
    for g in &each {
        over_wire.merge(&Grams::from_wire(&g.to_wire()).unwrap()).unwrap();
    }
    assert_eq!(over_wire, whole, "merge over the wire differs");
    assert_eq!(Grams::from_wire(&whole.to_wire()).unwrap(), whole);
}

#[test]
fn controls_a_flipped_word_a_wrong_geometry_and_overflow_are_caught() {
    let c = cfg();
    let s0 = taps(1, 4, &c);
    let s1 = taps(2, 4, &c);
    let whole = acc(&[s0.clone(), s1.clone()], &c);

    // flipped word: one tap word changed in shard 1 must break equality (the check can fail)
    let mut bad = s1.clone();
    bad[1][1][0][0] += 1 << 20;
    let mut m = acc(&[s0.clone()], &c);
    m.merge(&acc(&[bad], &c)).unwrap();
    assert_ne!(m, whole, "control: a mutated shard still compared equal");

    // geometry mismatch refused, target untouched
    let mut c2 = cfg();
    c2.n_ff = 12;
    let mut g = acc(&[s0.clone()], &c);
    let before = g.clone();
    assert!(matches!(g.merge(&Grams::new(&c2).unwrap()), Err(CaptureError::Shape(_))));
    assert_eq!(g, before);

    // truncated / corrupt wire refused
    let w = g.to_wire();
    assert!(Grams::from_wire(&w[..w.len() - 1]).is_err());
    let mut junk = w.clone();
    junk[0] = b'X';
    assert!(Grams::from_wire(&junk).is_err());

    // overflow: merging a shard with itself until a word would pass i64::MAX is refused and
    // leaves the target unchanged
    let mut big = acc(&[s0], &c);
    let mut guard = 0;
    loop {
        let snapshot = big.clone();
        let other = big.clone();
        match big.merge(&other) {
            Ok(()) => {}
            Err(e) => {
                assert!(matches!(e, CaptureError::Overflow));
                assert_eq!(big, snapshot, "overflowing merge mutated the target");
                break;
            }
        }
        guard += 1;
        assert!(guard < 80, "never overflowed");
    }
}
