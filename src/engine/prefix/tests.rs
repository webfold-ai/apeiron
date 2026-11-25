//! Tests for the prefix cache and warm-vs-cold prefill.

use super::*;
use crate::engine::attn::KvCarrier;
use crate::engine::attn::KvState;
use crate::engine::loader::Engine;
use crate::format::quant;

// Minimal GGUF writer helper for tests (similar to tests in engine::tests)
struct Gw(Vec<u8>);

impl Gw {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn string(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.bytes(s.as_bytes());
    }
}

fn d1(d0: usize) -> Vec<u64> {
    vec![d0 as u64]
}

fn d2(d0: usize, d1: usize) -> Vec<u64> {
    vec![d0 as u64, d1 as u64]
}

fn push_t(
    ts: &mut Vec<(String, u32, Vec<u64>, Vec<u8>)>,
    omit: Option<&str>,
    name: String,
    ty: u32,
    dims: Vec<u64>,
    payload: Vec<u8>,
) {
    if omit != Some(&name) {
        ts.push((name, ty, dims, payload));
    }
}

/// A `[-2, 2]` deterministic code varying with the row and column.
fn synth_code(r: usize, c: usize, salt: usize) -> i8 {
    let rr = i64::try_from(r).expect("small");
    let cc = i64::try_from(c).expect("small");
    let ss = i64::try_from(salt).expect("small");
    i8::try_from((rr * 3 + cc * 5 + ss) % 5 - 2).expect("in [-2, 2]")
}

fn synth_q8_weight(n_out: usize, d_bits: u16, salt: usize) -> Vec<u8> {
    const N_IN: usize = 32;
    let mut out = Vec::with_capacity(n_out * (2 + N_IN));
    let mut r = 0usize;
    while r < n_out {
        out.extend_from_slice(&d_bits.to_le_bytes());
        let mut c = 0usize;
        while c < N_IN {
            out.push(synth_code(r, c, salt).to_le_bytes()[0]);
            c += 1;
        }
        r += 1;
    }
    out
}

fn synth_f32_vec(n: usize, f: impl Fn(usize) -> f32) -> Vec<u8> {
    let mut b = Vec::with_capacity(n * 4);
    let mut i = 0usize;
    while i < n {
        b.extend_from_slice(&f(i).to_le_bytes());
        i += 1;
    }
    b
}

#[allow(
    clippy::too_many_lines,
    reason = "synthetic GGUF tensor and metadata header test fixture"
)]
fn synth_gguf_bytes() -> Vec<u8> {
    const NL: usize = 2;
    const NE: usize = 32;
    const NF: usize = 32;
    const NH: usize = 4;
    const NKV: usize = 2;
    const HD: usize = 8;
    const V: usize = 16;
    const Q_OUT: usize = NH * HD;
    const KV_OUT: usize = NKV * HD;
    const D_PROJ: u16 = 0x2800; // f16 2^-5
    const D_EMB: u16 = 0x3C00; // f16 1.0
    let gain = |_: usize| 1.0_f32;
    let bias = |i: usize| match i % 3 {
        0 => -1.0_f32,
        1 => 0.0,
        _ => 1.0,
    } / 64.0;

    let mut ts = Vec::new();
    let mut l = 0usize;
    while l < NL {
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.attn_norm.weight"),
            quant::GGML_TYPE_F32,
            d1(NE),
            synth_f32_vec(NE, gain),
        );
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.attn_q.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NE, Q_OUT),
            synth_q8_weight(Q_OUT, D_PROJ, l + 1),
        );
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.attn_q.bias"),
            quant::GGML_TYPE_F32,
            d1(Q_OUT),
            synth_f32_vec(Q_OUT, bias),
        );
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.attn_k.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NE, KV_OUT),
            synth_q8_weight(KV_OUT, D_PROJ, l + 2),
        );
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.attn_k.bias"),
            quant::GGML_TYPE_F32,
            d1(KV_OUT),
            synth_f32_vec(KV_OUT, bias),
        );
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.attn_v.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NE, KV_OUT),
            synth_q8_weight(KV_OUT, D_PROJ, l + 3),
        );
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.attn_v.bias"),
            quant::GGML_TYPE_F32,
            d1(KV_OUT),
            synth_f32_vec(KV_OUT, bias),
        );
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.attn_output.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NE, NE),
            synth_q8_weight(NE, D_PROJ, l + 4),
        );
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.ffn_norm.weight"),
            quant::GGML_TYPE_F32,
            d1(NE),
            synth_f32_vec(NE, gain),
        );
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.ffn_gate.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NE, NF),
            synth_q8_weight(NF, D_PROJ, l + 5),
        );
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.ffn_up.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NE, NF),
            synth_q8_weight(NF, D_PROJ, l + 6),
        );
        push_t(
            &mut ts,
            None,
            format!("blk.{l}.ffn_down.weight"),
            quant::GGML_TYPE_Q8_0,
            d2(NF, NE),
            synth_q8_weight(NE, D_PROJ, l + 7),
        );
        l += 1;
    }
    push_t(
        &mut ts,
        None,
        "output_norm.weight".to_string(),
        quant::GGML_TYPE_F32,
        d1(NE),
        synth_f32_vec(NE, gain),
    );
    push_t(
        &mut ts,
        None,
        "token_embd.weight".to_string(),
        quant::GGML_TYPE_Q8_0,
        d2(NE, V),
        synth_q8_weight(V, D_EMB, 9),
    );

    let mut h = Gw::new();
    h.bytes(b"GGUF");
    h.u32(3);
    h.u64(u64::try_from(ts.len()).expect("small"));
    h.u64(8);
    h.string("general.architecture");
    h.u32(8);
    h.string("qwen2");
    for (k, v) in [
        ("qwen2.block_count", u32::try_from(NL).expect("small")),
        (
            "qwen2.attention.head_count",
            u32::try_from(NH).expect("small"),
        ),
        (
            "qwen2.attention.head_count_kv",
            u32::try_from(NKV).expect("small"),
        ),
        ("qwen2.embedding_length", u32::try_from(NE).expect("small")),
        (
            "qwen2.feed_forward_length",
            u32::try_from(NF).expect("small"),
        ),
    ] {
        h.string(k);
        h.u32(4);
        h.u32(v);
    }
    h.string("qwen2.rope.freq_base");
    h.u32(6);
    h.u32(10_000.0_f32.to_bits());
    h.string("qwen2.attention.layer_norm_rms_epsilon");
    h.u32(6);
    h.u32(1e-5_f32.to_bits());

    let mut off = 0u64;
    for (name, ty, dims, payload) in &ts {
        h.string(name);
        h.u32(u32::try_from(dims.len()).expect("small"));
        for &d in dims {
            h.u64(d);
        }
        h.u32(*ty);
        h.u64(off);
        off += u64::try_from(payload.len()).expect("small");
    }
    let header_len = h.0.len();
    let pad = (32 - header_len % 32) % 32;
    for _ in 0..pad {
        h.0.push(0);
    }
    for (_, _, _, payload) in &ts {
        h.bytes(payload);
    }
    h.0
}

fn write_synth_gguf(name: &str) -> String {
    let path = std::env::temp_dir().join(format!("apeiron_prefix_{}_{name}", std::process::id()));
    std::fs::write(&path, synth_gguf_bytes()).expect("write fixture");
    path.to_str().expect("utf8 path").to_string()
}

/// Helper: checks that two `KvStates` have the same geometry and per-layer occupancy. The value
/// check is the caller's logits checksum; this helper does not compare plane words.
fn assert_kv_states_same_geometry_and_occupancy(a: &KvState, b: &KvState) {
    assert_eq!(a.n_layer(), b.n_layer());
    assert_eq!(a.n_head(), b.n_head());
    assert_eq!(a.n_head_kv(), b.n_head_kv());
    assert_eq!(a.head_dim(), b.head_dim());

    for l in 0..a.n_layer() {
        let occ_a = a.occupied(l).expect("layer a");
        let occ_b = b.occupied(l).expect("layer b");
        assert_eq!(occ_a, occ_b, "occupied count mismatch at layer {l}");
    }
}

/// Constant/forced collision hasher for negative control and collision tests.
#[derive(Debug, Clone, Copy)]
struct CollidingHasher(u64);

impl PrefixHasher for CollidingHasher {
    fn hash_tokens(&self, _tokens: &[u32]) -> u64 {
        self.0
    }
}

#[test]
fn warm_prefill_equals_cold_prefill_bit_for_bit() {
    let path = write_synth_gguf("warm_cold.gguf");
    let engine = Engine::open(&path).expect("open synth engine");

    // Block size B = 4 for testing on short prompt
    let b = 4;
    let mut cache = PrefixCache::with_block_size(10_000_000, b);

    // Prompt 1: 8 tokens
    let prompt1: Vec<u32> = vec![1, 3, 5, 7, 2, 4, 6, 8];
    // Cold prefill prompt 1 and populate cache
    let (kv_cold1, res_cold1) =
        prefill_with_cache(&engine, &mut cache, &prompt1, 16, true).expect("prefill prompt1");

    // Prompt 2: shares first 4 tokens with prompt 1, but has different remaining 4 tokens
    let prompt2: Vec<u32> = vec![1, 3, 5, 7, 9, 11, 13, 15];

    // Warm prefill prompt 2 using cache (should hit boundary 4 and resume from there)
    let (kv_warm2, res_warm2) =
        prefill_with_cache(&engine, &mut cache, &prompt2, 16, false).expect("warm prefill prompt2");

    // Cold prefill prompt 2 with empty cache
    let mut empty_cache = PrefixCache::with_block_size(10_000_000, b);
    let (kv_cold2, res_cold2) = prefill_with_cache(&engine, &mut empty_cache, &prompt2, 16, false)
        .expect("cold prefill prompt2");

    // Verify bit-for-bit equality of StepResult
    assert_eq!(
        res_warm2.token, res_cold2.token,
        "argmax token mismatch between warm and cold"
    );
    assert_eq!(
        res_warm2.logits_checksum, res_cold2.logits_checksum,
        "logits checksum mismatch between warm and cold"
    );

    // The logits checksum above is the value check; these verify the same shape and occupancy.
    assert_kv_states_same_geometry_and_occupancy(&kv_warm2, &kv_cold2);

    // Also verify prompt 1 itself when re-run warm
    let (kv_warm1, res_warm1) =
        prefill_with_cache(&engine, &mut cache, &prompt1, 16, false).expect("warm prefill prompt1");
    assert_eq!(res_warm1, res_cold1);
    assert_kv_states_same_geometry_and_occupancy(&kv_warm1, &kv_cold1);
}

#[test]
fn hash_collision_refusal_via_full_key_confirmation() {
    let path = write_synth_gguf("collision.gguf");
    let engine = Engine::open(&path).expect("open synth engine");

    // Create cache with a colliding hasher that maps every token slice to hash 42
    let colliding_hasher = CollidingHasher(42);
    let mut cache = PrefixCache::with_block_size_and_hasher(10_000_000, 4, colliding_hasher);

    let prefix_a: Vec<u32> = vec![1, 2, 3, 4];
    let prefix_b: Vec<u32> = vec![5, 6, 7, 8];

    // Step prefix A to create a snapshot
    let mut kv_a = engine.decode_state(16).expect("kv a");
    for &tok in &prefix_a {
        engine.step_with_kv(&mut kv_a, tok).expect("step a");
    }

    // Insert prefix A
    cache.insert(&prefix_a, &kv_a).expect("insert prefix a");

    // Probing for prefix A must succeed
    let hit_a = cache.longest_prefix(&prefix_a);
    assert!(hit_a.is_some(), "prefix A should hit");
    let (m_a, _) = hit_a.unwrap();
    assert_eq!(m_a, 4);

    // Probing for prefix B (which shares the same 64-bit hash 42) must MISS
    // because its tokens differ from prefix A!
    let hit_b = cache.longest_prefix(&prefix_b);
    assert!(
        hit_b.is_none(),
        "prefix B must miss due to full-key token mismatch despite hash collision"
    );
}

#[test]
fn budget_refusal_leaves_cache_unchanged() {
    let path = write_synth_gguf("budget.gguf");
    let engine = Engine::open(&path).expect("open synth engine");

    let prefix: Vec<u32> = vec![1, 2, 3, 4];
    let mut kv = engine.decode_state(16).expect("kv");
    for &tok in &prefix {
        engine.step_with_kv(&mut kv, tok).expect("step");
    }

    let snapshot_size = PrefixCache::<FnvPrefixHasher>::estimate_snapshot_bytes(&kv);
    assert!(snapshot_size > 0);

    // Set budget strictly smaller than needed for 1 snapshot
    let budget = snapshot_size - 1;
    let mut cache = PrefixCache::with_block_size(budget, 4);

    let res = cache.insert(&prefix, &kv);
    assert_eq!(
        res,
        Err(PrefixCacheError::BudgetExceeded {
            needed: snapshot_size,
            current: 0,
            budget,
        })
    );

    // Cache remains empty and unchanged
    assert_eq!(cache.current_bytes(), 0);
    assert_eq!(cache.entry_count(), 0);
    assert!(cache.longest_prefix(&prefix).is_none());
}

#[test]
fn cache_stores_one_kv_state_per_prompt_o_n_memory() {
    let path = write_synth_gguf("on_memory.gguf");
    let engine = Engine::open(&path).expect("open synth engine");

    // 16-token prompt with block size 4 -> 4 boundaries (4, 8, 12, 16)
    let prompt: Vec<u32> = (0..16).map(|i| (i * 3 + 1) % 15).collect();
    let mut kv = engine.decode_state(32).expect("kv");
    for &tok in &prompt {
        engine.step_with_kv(&mut kv, tok).expect("step");
    }

    let mut cache = PrefixCache::with_block_size(10_000_000, 4);
    cache.insert(&prompt, &kv).expect("insert prompt");

    // Exactly ONE PromptEntry stored in the arena
    assert_eq!(cache.entry_count(), 1);
    let expected_bytes = PrefixCache::<FnvPrefixHasher>::estimate_snapshot_bytes(&kv);
    assert_eq!(cache.current_bytes(), expected_bytes);

    // All 4 boundaries hit and resolve to their respective lengths
    for b in 1..=4 {
        let m = b * 4;
        let hit = cache.longest_prefix(&prompt[..m]);
        assert!(hit.is_some(), "boundary {m} must hit");
        assert_eq!(hit.unwrap().0, m);
    }
}

#[test]
fn truncate_kv_state_resets_occupied_and_resumes_append() {
    let path = write_synth_gguf("truncate.gguf");
    let engine = Engine::open(&path).expect("open synth engine");

    let mut kv = engine.decode_state(16).expect("kv");
    for tok in 1..=8 {
        engine.step_with_kv(&mut kv, tok).expect("step");
    }
    assert_eq!(kv.occupied(0).expect("layer 0"), 8);

    // Truncate to 4
    assert!(truncate_kv_state(&mut kv, 4));
    assert_eq!(kv.occupied(0).expect("layer 0"), 4);

    // Stepping next token advances occupied to 5
    let rewound = engine.step_with_kv(&mut kv, 7).expect("step at pos 4");
    assert_eq!(kv.occupied(0).expect("layer 0"), 5);

    // The rewound state steps bit-identically to a state that stopped at 4.
    let mut fresh = engine.decode_state(16).expect("kv");
    for tok in 1..=4 {
        engine.step_with_kv(&mut fresh, tok).expect("step");
    }
    let direct = engine.step_with_kv(&mut fresh, 7).expect("step at pos 4");
    assert_eq!(rewound.token, direct.token);
    assert_eq!(rewound.logits_checksum, direct.logits_checksum);

    // Truncating past the occupied length refuses and changes no layer.
    assert!(!truncate_kv_state(&mut kv, 6));
    for l in 0..kv.n_layer() {
        assert_eq!(kv.occupied(l).expect("layer"), 5);
    }
}

/// Steps `tokens` from a fresh state, returning the snapshot to insert.
fn step_prompt(engine: &Engine, tokens: &[u32], cap: usize) -> KvState {
    let mut kv = engine.decode_state(cap).expect("decode state");
    for &tok in tokens {
        engine.step_with_kv(&mut kv, tok).expect("step");
    }
    kv
}

#[test]
fn eviction_evicts_lru_entry_under_budget() {
    let path = write_synth_gguf("evict_lru.gguf");
    let engine = Engine::open(&path).expect("open synth engine");

    let b = 4;
    let a: Vec<u32> = vec![1, 2, 3, 4];
    let b_toks: Vec<u32> = vec![5, 6, 7, 8];
    let c: Vec<u32> = vec![9, 10, 11, 12];

    let kv_a = step_prompt(&engine, &a, 16);
    let kv_b = step_prompt(&engine, &b_toks, 16);
    let kv_c = step_prompt(&engine, &c, 16);

    let snap = PrefixCache::<FnvPrefixHasher>::estimate_snapshot_bytes(&kv_a);
    assert_eq!(
        snap,
        PrefixCache::<FnvPrefixHasher>::estimate_snapshot_bytes(&kv_b)
    );
    let budget = 2 * snap;
    let mut cache = PrefixCache::with_block_size(budget, b);

    cache.insert(&a, &kv_a).expect("insert A");
    cache.insert(&b_toks, &kv_b).expect("insert B");
    assert_eq!(cache.entry_count(), 2);
    assert_eq!(cache.current_bytes(), 2 * snap);

    // Touch A through a confirmed hit; B stays the least recently used.
    assert!(
        cache.longest_prefix(&a).is_some(),
        "A must hit before eviction"
    );

    // Insert C: B is LRU and must be evicted to make room.
    cache.insert(&c, &kv_c).expect("insert C");

    assert!(
        cache.current_bytes() <= budget,
        "budget must hold after eviction"
    );
    assert_eq!(cache.entry_count(), 2);
    assert!(
        cache.longest_prefix(&a).is_some(),
        "recently touched A must survive"
    );
    assert!(
        cache.longest_prefix(&c).is_some(),
        "newly inserted C must hit"
    );
    assert!(
        cache.longest_prefix(&b_toks).is_none(),
        "LRU B must be evicted"
    );
}

#[test]
fn oversized_snapshot_refuses_and_leaves_cache_unchanged() {
    let path = write_synth_gguf("evict_oversize.gguf");
    let engine = Engine::open(&path).expect("open synth engine");

    let b = 4;
    let a: Vec<u32> = vec![1, 2, 3, 4];
    let kv_a = step_prompt(&engine, &a, 16);

    let snap = PrefixCache::<FnvPrefixHasher>::estimate_snapshot_bytes(&kv_a);
    let budget = 3 * snap / 2;
    let mut cache = PrefixCache::with_block_size(budget, b);
    cache.insert(&a, &kv_a).expect("insert A");

    let count_before = cache.entry_count();
    let bytes_before = cache.current_bytes();
    assert_eq!(count_before, 1);

    // 16 tokens at capacity 32: twice the single-snapshot charge, above the whole budget.
    let big: Vec<u32> = (0..16).collect();
    let kv_big = step_prompt(&engine, &big, 32);
    let big_bytes = PrefixCache::<FnvPrefixHasher>::estimate_snapshot_bytes(&kv_big);
    assert!(big_bytes > budget);

    let res = cache.insert(&big, &kv_big);
    assert_eq!(
        res,
        Err(PrefixCacheError::BudgetExceeded {
            needed: big_bytes,
            current: bytes_before,
            budget,
        })
    );

    // The refusal happened before any eviction: the cache is byte- and entry-identical.
    assert_eq!(cache.entry_count(), count_before);
    assert_eq!(cache.current_bytes(), bytes_before);
    assert!(
        cache.longest_prefix(&a).is_some(),
        "resident A must be untouched"
    );
    assert!(
        cache.longest_prefix(&big).is_none(),
        "refused snapshot must not be resident"
    );
}

#[test]
fn stale_boundary_key_never_returns_reused_slot_state() {
    let path = write_synth_gguf("evict_stale.gguf");
    let engine = Engine::open(&path).expect("open synth engine");

    let b = 4;
    let a: Vec<u32> = vec![1, 2, 3, 4];
    let b_toks: Vec<u32> = vec![5, 6, 7, 8];
    let c: Vec<u32> = vec![9, 10, 11, 12];

    let kv_a = step_prompt(&engine, &a, 16);
    let kv_b = step_prompt(&engine, &b_toks, 16);
    let kv_c = step_prompt(&engine, &c, 16);

    let snap = PrefixCache::<FnvPrefixHasher>::estimate_snapshot_bytes(&kv_a);
    let budget = 2 * snap;
    let mut cache = PrefixCache::with_block_size(budget, b);

    cache.insert(&a, &kv_a).expect("insert A");
    cache.insert(&b_toks, &kv_b).expect("insert B");
    assert!(cache.longest_prefix(&a).is_some(), "touch A so B is LRU");
    cache.insert(&c, &kv_c).expect("insert C evicts B");

    // B's boundary key survives eviction (the map face has no removal) ...
    let stale_key = PrefixKey(cache.hasher.hash_tokens(&b_toks));
    let stale = cache
        .table
        .get(&stale_key)
        .copied()
        .flatten()
        .expect("stale boundary key retained after eviction");

    // ... and it now points at the slot reused by C.
    let reused_slot = cache
        .arena
        .iter()
        .position(|e| e.as_ref().is_some_and(|p| p.tokens == c))
        .expect("C resident in arena");
    assert_eq!(stale.entry_id as usize, reused_slot);
    assert_eq!(
        cache.arena[reused_slot].as_ref().expect("C entry").tokens,
        c
    );

    // Probing B's old tokens must MISS, never return C's state for B's tokens.
    assert!(
        cache.longest_prefix(&b_toks).is_none(),
        "stale key must not resolve the reused slot's state for the evicted prompt's tokens"
    );
}

/// The batched tail is the per-token path's own values: one block pass over the missed positions
/// plus the last step equals stepping every token, at [`KernelMode::Exact`] and on the Q8 reader.
#[test]
fn the_batched_tail_equals_the_per_token_path() {
    let path = write_synth_gguf("batched_tail.gguf");
    let engine = Engine::open(&path).expect("open synth engine");
    let prompt: Vec<u32> = vec![2, 3, 5, 7, 11, 13, 1, 4, 6, 8, 9];
    let cap = 16;

    for mode in [KernelMode::Exact, KernelMode::Q8Act] {
        // The oracle: one single-token step per position.
        let mut kv = engine.decode_state(cap).expect("oracle kv");
        let mut want = None;
        for &tok in &prompt {
            let logits = engine
                .step_logits_with_mode(&mut kv, tok, mode)
                .expect("oracle step");
            want = Some(StepResult {
                token: argmax_fixed(&logits).expect("argmax"),
                logits_checksum: fixed_vector_checksum(&logits),
            });
        }
        let want = want.expect("a result");

        // Cold: the whole prompt through the cache's batched tail.
        let mut cold = PrefixCache::with_block_size(10_000_000, 4);
        let (kv_cold, res_cold) =
            prefill_with_cache_mode(&engine, &mut cold, &prompt, cap - prompt.len(), true, mode)
                .expect("cold");
        assert_eq!(res_cold.token, want.token, "{mode:?}: cold token");
        assert_eq!(
            res_cold.logits_checksum, want.logits_checksum,
            "{mode:?}: cold logits"
        );
        for l in 0..kv_cold.n_layer() {
            assert_eq!(kv_cold.occupied(l).expect("cold"), prompt.len());
            assert_eq!(
                kv_cold.occupied(l).expect("cold"),
                kv.occupied(l).expect("oracle")
            );
        }

        // Warm: seed the first 8 tokens, then the same prompt resumes at 8 and solves 8..10.
        let mut warm = PrefixCache::with_block_size(10_000_000, 4);
        // The seed must be stepped under the same kernel: the cache resumes whatever state it
        // stores, and a Q8Act tail over an Exact prefix is a different token stream's state.
        let mut seed = engine.decode_state(cap).expect("seed kv");
        for &tok in &prompt[..8] {
            engine
                .step_logits_with_mode(&mut seed, tok, mode)
                .expect("seed step");
        }
        warm.insert(&prompt[..8], &seed).expect("seed insert");
        let (kv_warm, res_warm) =
            prefill_with_cache_mode(&engine, &mut warm, &prompt, cap - prompt.len(), false, mode)
                .expect("warm");
        assert_eq!(res_warm.token, want.token, "{mode:?}: warm token");
        assert_eq!(
            res_warm.logits_checksum, want.logits_checksum,
            "{mode:?}: warm logits"
        );
        for l in 0..kv_warm.n_layer() {
            assert_eq!(kv_warm.occupied(l).expect("warm"), prompt.len());
        }
    }
}

/// The snapshot charge is the carrier width times the state's capacity, not a fixed eight bytes
/// per value at the occupied length: an `F32` default state is half the `Q32` charge, and the
/// spare capacity the clone carries is paid for.
#[test]
fn the_snapshot_charge_follows_the_carrier_and_the_capacity() {
    let (n_layer, n_head, n_head_kv, head_dim, cap) = (2, 4, 2, 8, 10);
    let values = 2 * n_head_kv * head_dim * n_layer;
    let q32 = KvState::with_carrier(n_layer, n_head, n_head_kv, head_dim, cap, KvCarrier::Q32)
        .expect("q32 state");
    let f32 = KvState::with_carrier(n_layer, n_head, n_head_kv, head_dim, cap, KvCarrier::F32)
        .expect("f32 state");
    let i8 = KvState::with_carrier(n_layer, n_head, n_head_kv, head_dim, cap, KvCarrier::I8)
        .expect("i8 state");

    let charge = |kv: &KvState| PrefixCache::<FnvPrefixHasher>::estimate_snapshot_bytes(kv);
    assert_eq!(charge(&q32), values * 8 * cap);
    assert_eq!(charge(&f32), values * 4 * cap);
    assert_eq!(charge(&i8), (values + 2 * n_head_kv * n_layer * 4) * cap);
    assert_eq!(charge(&f32) * 2, charge(&q32));
}

/// A snapshot shorter than the prompt it is filed under is refused before any mutation: resuming
/// it would append at a layer's short offset.
#[test]
fn a_snapshot_short_of_its_prompt_is_refused() {
    let path = write_synth_gguf("short_snapshot.gguf");
    let engine = Engine::open(&path).expect("open synth engine");
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    let mut kv = engine.decode_state(16).expect("kv");
    for &tok in &prompt[..4] {
        engine.step_with_kv(&mut kv, tok).expect("step");
    }

    let mut cache = PrefixCache::with_block_size(10_000_000, 4);
    let err = cache.insert(&prompt, &kv).expect_err("short snapshot");
    assert!(
        matches!(err, PrefixCacheError::Step(StepError::BadConfig(_))),
        "unexpected refusal: {err:?}"
    );
    assert_eq!(cache.entry_count(), 0);
    assert_eq!(cache.current_bytes(), 0);
}
