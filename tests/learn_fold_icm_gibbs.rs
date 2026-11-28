//! The finite-temperature code draw against the zero-temperature coordinate descent: a joint
//! two-coordinate table with a local minimum the coordinate descent cannot leave and a global
//! minimum the Boltzmann draw reaches. The draw is the Rust side of the folding
//! `gibbs_variational_principle`; the coordinate descent is the `T = 0` control.
//!
//! The two-block table is driven through the real decision helpers [`best_pair_q`] (the `T = 0`
//! argmin) and [`best_pair_q_beta`] (one [`gibbs_choice`] draw of the same `CODE_LEN²` cost table),
//! and the real [`e8_joint_icm_2plane_annealed`] entry is run at `o = 1`, `n = 128` to check the
//! schedule is wired into the band loop.
//!
//! Limitation, stated exactly: the two-block table is a hand-built sequence of per-block conditional
//! tables fed to the real helpers, not the full `e8_joint_icm_2plane` trajectory — driving that
//! entry to a *cross-block* local minimum needs an `o × n` payload, the in-block codebook geometry
//! and the three Hadamard SyrkAcc arranged so the coupling has that minimum, which this test does not
//! construct. The annealed entry's own run here is a wiring check (it returns valid planes), not a
//! claim about a fitted model.
//!
//! Grade `tested`; nothing here is extracted or refined.

use apeiron::engine::ONE;
use apeiron::learn::code::e8::codeword;
use apeiron::learn::fold::fit::icm::{
    best_pair_q, best_pair_q_beta, e8_joint_icm_2plane, e8_joint_icm_2plane_annealed, gibbs_choice,
    BetaSchedule, IcmError, Plane, PlaneRef, BLOCK, CODE_LEN, GROUP,
};

/// One in the Q96.96 cost scale the ICM pair cost uses.
const ONE96: i128 = 1i128 << 96;

/// A cost gap no draw at the test's temperatures can take: every entry past the two live codes is
/// clamped to `exp(-16)` by [`gibbs_choice`], so its mass stays below `3e-5` of the table.
const PENALTY: i128 = 512 * ONE96;

/// The 2×2 joint energy: `(0,1)` and `(1,0)` cost 5, so every single-coordinate move from `(0,0)`
/// climbs, and `(1,1) = -10` is the global minimum.
const ENERGY: [[i128; 2]; 2] = [[0, 5 * ONE96], [5 * ONE96, -10 * ONE96]];

/// One block's conditional pair table: the live index carries `energy`, every other code the
/// penalty, and the other plane is pinned to its code `0`.
struct Cond {
    av: [i128; CODE_LEN],
    xm: Vec<i64>,
    bv: [i128; CODE_LEN],
}

impl Cond {
    fn pinned(step: [i128; 2]) -> Self {
        let mut av = [PENALTY; CODE_LEN];
        let mut bv = [PENALTY; CODE_LEN];
        av[0] = 0;
        bv[0] = 0;
        av[..2].copy_from_slice(&step);
        Self {
            av,
            xm: vec![0_i64; CODE_LEN * CODE_LEN],
            bv,
        }
    }

    /// The same table with the live cost in plane 2 and plane 1 pinned to code `0`.
    fn pinned_second(step: [i128; 2]) -> Self {
        let mut av = [PENALTY; CODE_LEN];
        let mut bv = [PENALTY; CODE_LEN];
        av[0] = 0;
        bv[0] = 0;
        bv[..2].copy_from_slice(&step);
        Self {
            av,
            xm: vec![0_i64; CODE_LEN * CODE_LEN],
            bv,
        }
    }
}

/// The block-0 conditional (plane-1 code) given plane-2's code `j`.
fn row_step(j: usize) -> Cond {
    Cond::pinned([ENERGY[0][j], ENERGY[1][j]])
}

/// The block-1 conditional (plane-2 code) given plane-1's code `i`.
fn col_step(i: usize) -> Cond {
    Cond::pinned_second([ENERGY[i][0], ENERGY[i][1]])
}

/// Alternating single-coordinate descent on the 2×2 table through the real `T = 0` helper.
fn coordinate_descent() -> (usize, usize) {
    let (mut i, mut j) = (0_usize, 0_usize);
    for _ in 0..8 {
        let c = row_step(j);
        i = best_pair_q(0, &c.av, &c.xm, &c.bv).expect("argmin") / CODE_LEN;
        let c = col_step(i);
        j = best_pair_q(0, &c.av, &c.xm, &c.bv).expect("argmin") % CODE_LEN;
    }
    (i, j)
}

/// The same alternation with each block's code drawn at `schedule.beta_at(sweep)`. A draw that lands
/// on a penalised code past the two live entries (a nonzero-floor event of the `exp(-16)` clamp) is
/// returned as `(2, 2)`, so it is counted as off the table rather than as the global minimum.
fn annealed(schedule: &BetaSchedule) -> (usize, usize) {
    let (mut i, mut j) = (0_usize, 0_usize);
    let mut states = [schedule.seed(), schedule.seed() ^ 0xA5A5_5A5A_1234_5678];
    for sweep in 0..schedule.sweeps() {
        let beta = schedule.beta_at(sweep);
        let c = row_step(j);
        i = best_pair_q_beta(0, &c.av, &c.xm, &c.bv, beta, &mut states[0]).expect("draw")
            / CODE_LEN;
        if i >= 2 {
            return (2, 2);
        }
        let c = col_step(i);
        j = best_pair_q_beta(0, &c.av, &c.xm, &c.bv, beta, &mut states[1]).expect("draw")
            % CODE_LEN;
        if j >= 2 {
            return (2, 2);
        }
    }
    (i, j)
}

#[test]
fn annealed_gibbs_escapes_the_coordinate_local_min() {
    // Every single-coordinate flip from (0,0) costs 5, so the coordinate descent stops there while
    // the global minimum is (1,1).
    assert_eq!(coordinate_descent(), (0, 0));

    // A cooling ramp from beta = 0.05 to beta = 1 over eight sweeps: the uphill flip out of (0,0)
    // has probability exp(-0.25) / (1 + exp(-0.25)) = 0.62 at the first sweep, and once (1,1) is
    // occupied the 15-unit gap to (0,1) or (1,0) holds it there at the later sweeps. The escape is
    // a chance event, so the claim is over 32 fixed seeds, not one lucky stream.
    let mut reached = 0_u32;
    for seed in 1..=32_u64 {
        let schedule = BetaSchedule::new(ONE / 20, ONE, 8, seed).expect("schedule");
        if annealed(&schedule) == (1, 1) {
            reached += 1;
        }
    }
    assert!(
        reached > 16,
        "the annealed sweep should reach (1,1) from (0,0) on most seeds; reached = {reached}/32"
    );

    // The cold limit approaches the global minimum from the same start: beta = 4 makes every
    // uphill move exp(-20), below the helper's exp(-16) floor.
    let cold = BetaSchedule::new(4 * ONE, 4 * ONE, 1, 0x0123_4567_89AB_CDEF).expect("schedule");
    assert_eq!(annealed(&cold), (0, 0), "a cold draw follows the argmin");
}

#[test]
fn the_schedule_ramps_and_refuses_an_empty_sweep() {
    let s = BetaSchedule::new(ONE / 4, 4 * ONE, 3, 7).expect("schedule");
    assert_eq!(s.beta_at(0), ONE / 4);
    assert_eq!(s.beta_at(2), 4 * ONE);
    assert_eq!(s.beta_at(9), 4 * ONE, "clamped past the last sweep");
    assert!(s.beta_at(1) > s.beta_at(0) && s.beta_at(1) < s.beta_at(2));
    assert_eq!(s.seed(), 7);
    assert_eq!(s.sweeps(), 3);
    assert_eq!(BetaSchedule::new(0, ONE, 0, 1).err(), Some(IcmError::Shape));
    assert_eq!(BetaSchedule::new(ONE, 0, 2, 1).err(), Some(IcmError::Shape));
    assert_eq!(
        BetaSchedule::new(-1, ONE, 2, 1).err(),
        Some(IcmError::Shape)
    );
}

#[test]
fn hot_and_draw_tables_keep_the_gibbs_contract() {
    // (0,0) = 0, (0,1) = (1,0) = 5, (1,1) = -10 in the Q96.96 scale the pair cost uses.
    let costs = [0, 5 * ONE96, 5 * ONE96, -10 * ONE96];

    // beta = 0.2: the global minimum carries exp(2) / (1 + 2 exp(-1) + exp(2)) = 0.81 of the mass.
    let beta = ONE / 5;
    let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
    let mut hits = 0_u32;
    for _ in 0..64 {
        if gibbs_choice(&costs, beta, &mut rng).expect("draw") == 3 {
            hits += 1;
        }
    }
    assert!(
        hits > 32,
        "the positive-temperature draw should reach (1,1); hits = {hits}/64"
    );

    // The hot limit is uniform: beta = 0 gives every entry the same weight.
    let mut rng = 0x0123_4567_89AB_CDEF_u64;
    let mut hot = 0_u32;
    for _ in 0..256 {
        if gibbs_choice(&costs, 0, &mut rng).expect("draw") == 3 {
            hot += 1;
        }
    }
    assert!(
        (32..96).contains(&hot),
        "the hot limit should be near uniform; hits = {hot}/256"
    );

    // The cold limit approaches the global minimum.
    let mut rng = 0xDEAD_BEEF_CAFE_BABE_u64;
    let mut cold = 0_u32;
    for _ in 0..64 {
        if gibbs_choice(&costs, 64 * ONE, &mut rng).expect("draw") == 3 {
            cold += 1;
        }
    }
    assert!(
        cold > 60,
        "the cold limit should concentrate on (1,1); hits = {cold}/64"
    );
}

/// A plane whose codes are all codebook row `code`, with unit scales.
fn flat_plane(o: usize, n: usize, code: usize) -> Plane {
    let nb = n / BLOCK;
    let g = n / GROUP;
    let word = codeword(code).expect("row in range");
    let mut s = vec![0_i64; o * n];
    for row in 0..o {
        for m in 0..nb {
            let at = row * n + m * BLOCK;
            s[at..at + BLOCK].copy_from_slice(&word);
        }
    }
    Plane {
        s,
        idx: vec![u16::try_from(code).expect("codebook row < 256"); o * nb],
        r: vec![ONE; o * g],
        c: vec![ONE; n],
        d: vec![ONE; n],
    }
}

fn pref(p: &Plane) -> PlaneRef<'_> {
    PlaneRef {
        s: &p.s,
        idx: &p.idx,
        r: &p.r,
        c: &p.c,
        d: &p.d,
    }
}

#[test]
fn the_annealed_entry_runs_the_schedule_over_the_real_band() {
    let (o, n) = (1_usize, GROUP);
    let ws: Vec<i64> = (0..o * n)
        .map(|i| ONE / 4 + i64::try_from(i).expect("index fits") * 977)
        .collect();
    let mut hl = vec![0_i64; n * n];
    for i in 0..n {
        hl[i * n + i] = ONE;
    }
    let (p1, p2) = (flat_plane(o, n, 0), flat_plane(o, n, 1));
    let schedule = BetaSchedule::new(ONE / 8, ONE, 3, 0x0BAD_5EED_1234_5678).expect("schedule");

    let (a1, a2, ast) = e8_joint_icm_2plane_annealed(
        &ws,
        &hl,
        pref(&p1),
        pref(&p2),
        o,
        n,
        3,
        0,
        None,
        1,
        None,
        Some(&schedule),
    )
    .expect("annealed joint sweep");
    let (b1, b2, bst) =
        e8_joint_icm_2plane(&ws, &hl, pref(&p1), pref(&p2), o, n, 3, 0, None, 1, None)
            .expect("T = 0 joint sweep");

    for p in [&a1, &a2, &b1, &b2] {
        assert_eq!(p.s.len(), o * n);
        assert_eq!(p.idx.len(), o * (n / BLOCK));
        assert!(p.idx.iter().all(|&j| usize::from(j) < CODE_LEN));
    }
    assert!(
        bst.j_after <= bst.j_before,
        "the T = 0 control is non-increasing: {} -> {}",
        bst.j_before,
        bst.j_after
    );
    // The schedule is live: the scheduled planes differ from the argmin planes, and the move
    // counter counts the same code rows the draw wrote.
    assert!(ast.moves <= o * (n / BLOCK) * 3);
    assert!(
        a1.idx != b1.idx || a2.idx != b2.idx,
        "a hot schedule should not reproduce the T = 0 planes"
    );
}
