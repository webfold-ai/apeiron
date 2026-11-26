//! The attention (softmax-layer) half of the column loop: [`Engine::softmax_cols`]'s projections,
//! its serial per-column path and its pooled block path.
//!
//! Split out of `step/cols.rs` by moving items, not by changing them; module paths in the generated
//! extraction move with it (`step.cols.*` -> `step.softmax.*`, `proof/boundary-axioms.txt`).

use super::cols::tap::{Tap, STAGE_K, STAGE_Q, STAGE_V};
use super::cols::{first_by_key, fold_col};
use super::{
    add_assign_fixed, apply_sigmoid_gate, qk_norm_and_rope, split_query_gate, LayerIndex, Sched,
    StepError,
};
use crate::engine::attn::KvCarrier;
use crate::engine::chart::task::{HeadTask, Task2};
use crate::engine::loader::{opt_slice, DecodeVectors};
use crate::engine::trace::{self, Phase};
use crate::engine::{AttnError, Engine, KernelMode, KvState};
use std::sync::OnceLock;

/// The `APEIRON_EXACT_SCAN` flag: the per-thread test override, else the process value parsed once
/// (`OnceLock`) and defaulting to off. [`Engine::softmax_core_block`] reads it on the calling
/// thread before any pool dispatch, so every worker in a dispatch sees one value.
fn exact_scan_enabled() -> bool {
    crate::engine::attn::exact_scan_override().unwrap_or_else(|| {
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| std::env::var("APEIRON_EXACT_SCAN").is_ok_and(|v| v == "1"))
    })
}

/// The key-tile width of the fused exact-scan attention read. The fold's tile boundaries only
/// re-parenthesise the exact `i128` `den`/`num` sums, so the value changes no result; 32 keys keep
/// the `head_dim`-wide numerator hot through a tile.
const EXACT_SCAN_TILE: usize = 32;

/// Consecutive columns per attention tile on the [`KvCarrier::F32`] path: the tile reads each K/V
/// row once and folds it against all of the tile's queries. The cold read falls from
/// `Σ_c (pos₀ + c + 1)` rows to `Σ_t (pos₀ + t·QUERY_TILE + QUERY_TILE)`, about `1/QUERY_TILE`
/// of it; 32 keeps the score stage (`rows · 64` words) inside L2.
const QUERY_TILE: usize = 32;

/// The `APEIRON_ATTN_TILE` flag: the tiled F32 read unless set to `0`. A same-binary A/B switch,
/// parsed once per process like [`exact_scan_enabled`]'s environment fallback.
fn attn_tile_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("APEIRON_ATTN_TILE").as_deref() != Ok("0"))
}

impl Engine {
    /// The `[query | key | value]` projection columns of [`Engine::softmax_cols`]: the single-column
    /// fast path when `xs` holds one token and there is no [`LayerIndex`], else the grouped
    /// projection over every column. `dims` is `[q_proj, kv_out, n_embd]`.
    ///
    /// # Complexity
    ///
    /// As [`Engine::step_qkv_sched`] (one column) or [`Engine::group_cols`] (the block).
    pub(super) fn softmax_qkv(
        &self,
        li: Option<&LayerIndex>,
        layer: usize,
        xs: &[Vec<i64>],
        dims: [usize; 3],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<Vec<Vec<Vec<i64>>>, StepError> {
        let [q_proj, kv_out, n_embd] = dims;
        if xs.len() == 1 && li.is_none() {
            let mut r =
                self.step_qkv_sched(layer, [q_proj, kv_out, kv_out], n_embd, &xs[0], sched, mode)?;
            let q = std::mem::take(&mut r[0]);
            let k = std::mem::take(&mut r[1]);
            let v = std::mem::take(&mut r[2]);
            Ok(vec![vec![q], vec![k], vec![v]])
        } else {
            self.group_cols(
                li,
                layer,
                &[0, 1, 2],
                &[q_proj, kv_out, kv_out],
                n_embd,
                xs,
                sched,
                mode,
            )
        }
    }

    /// The serial per-column attention loop of [`Engine::softmax_cols`]: column `c` runs at position
    /// `occupied + c`, in token order, on the calling thread.
    ///
    /// With `tap`, each column's post-`RoPE` query and key and its value are appended to
    /// [`STAGE_Q`], [`STAGE_K`] and [`STAGE_V`] of the layer, in column order; the off path adds
    /// one `Option` test per column and copies nothing.
    ///
    /// # Complexity
    ///
    /// `O(Σ_c cost(softmax_col(c)))`; `O(k)` output rows, plus `O(k·(q_out + 2·kv_out))` copied
    /// when the tap records.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer, slot, cache, projections, norm epsilon, rotation table, dispatch and tap are the loop's input"
    )]
    pub(super) fn softmax_serial_cols(
        &self,
        layer: usize,
        slot: usize,
        kv: &mut KvState,
        qkv: &[Vec<Vec<i64>>],
        eps: i64,
        table: &crate::engine::ops::rope::RopeTable,
        parallel_heads: bool,
        tap: &mut Option<Tap>,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let k = qkv[0].len();
        let mut atts = Vec::with_capacity(k);
        let mut err: Option<StepError> = None;
        let mut c = 0usize;
        while c < k && err.is_none() {
            match self.softmax_col(layer, slot, kv, qkv, c, eps, table, parallel_heads, tap) {
                Ok(att) => atts.push(att),
                Err(e) => err = Some(e),
            }
            c += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        Ok(atts)
    }

    /// The attention output projection of [`Engine::softmax_cols`], after the `AttnCore` span.
    ///
    /// # Complexity
    ///
    /// As [`Engine::proj_cols`] on projection part 3.
    #[allow(
        clippy::too_many_arguments,
        reason = "the layer index, layer, embedding width, query output width, attention rows, schedule and dispatch are the projection's input"
    )]
    pub(super) fn softmax_proj_out(
        &self,
        li: Option<&LayerIndex>,
        layer: usize,
        n_embd: usize,
        q_out: usize,
        atts: &[Vec<i64>],
        sched: Sched,
        mode: KernelMode,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let k = atts.len();
        let _out = trace::span(
            Phase::AttnProjOut,
            k,
            trace::q8_bytes(n_embd, q_out),
            trace::macs(n_embd, q_out, k),
        );
        self.proj_cols(li, layer, 3, n_embd, q_out, atts, sched, mode, None)
    }

    /// One column of [`Engine::softmax_cols`]: bias, gate split, q/k norm and `RoPE` at its position,
    /// append, then attention over the prefix including itself. With `tap`, the post-`RoPE` query
    /// and key and the value are appended to [`STAGE_Q`], [`STAGE_K`] and [`STAGE_V`].
    #[allow(
        clippy::too_many_arguments,
        clippy::index_refutable_slice,
        reason = "the per-column inputs of softmax_cols: layer, slot, cache, columns, column, eps, table, dispatch, tap; the three biases are read by constant index because an array pattern does not extract"
    )]
    fn softmax_col(
        &self,
        layer: usize,
        slot: usize,
        kv: &mut KvState,
        qkv: &[Vec<Vec<i64>>],
        c: usize,
        eps: i64,
        table: &crate::engine::ops::rope::RopeTable,
        parallel_heads: bool,
        tap: &mut Option<Tap>,
    ) -> Result<Vec<i64>, StepError> {
        let cfg = &self.config;
        let (n_head, hd) = (cfg.n_head, cfg.head_dim);
        let q_out = n_head.checked_mul(hd).ok_or(StepError::LengthOverflow)?;
        let vectors = self.decode_vectors();
        let (mut query, mut key, value) = Self::softmax_bias_add(qkv, vectors, layer, c)?;
        let gate = Self::softmax_gate_split(&mut query, n_head, hd, cfg.attn_gate)?;
        let pos = kv.occupied(slot).map_err(StepError::Kv)?;
        let pos_u32 = u32::try_from(pos).map_err(|_| StepError::LengthOverflow)?;
        Self::softmax_qk_rope(
            vectors,
            layer,
            table,
            &mut query,
            &mut key,
            hd,
            eps,
            pos_u32,
            parallel_heads,
        )?;
        if let Some(t) = tap {
            t.push(layer, STAGE_Q, &query);
            t.push(layer, STAGE_K, &key);
            t.push(layer, STAGE_V, &value);
        }
        let appended = Self::softmax_kv_append(kv, slot, &key, &value)?;
        let kv_ro: &KvState = kv;
        let mut att = Self::softmax_head_loop(
            kv_ro,
            slot,
            &query,
            [n_head, hd, q_out, appended],
            parallel_heads,
        )?;
        Self::softmax_gate_apply(&mut att, opt_slice(&gate))?;
        Ok(att)
    }

    /// The column's three attention biases, added to the cloned `[query | key | value]` in that
    /// order. A bias entry of `None` leaves its vector unchanged, as the former inline blocks did.
    ///
    /// # Complexity
    ///
    /// `O(q_out + 2·kv_out)` checked adds, `O(1)` besides the three column clones.
    #[allow(
        clippy::type_complexity,
        clippy::index_refutable_slice,
        reason = "the three biased column vectors are the call's result; the three biases are read by constant index because an array pattern does not extract"
    )]
    fn softmax_bias_add(
        qkv: &[Vec<Vec<i64>>],
        vectors: &DecodeVectors,
        layer: usize,
        c: usize,
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>), StepError> {
        let (mut query, mut key, mut value) =
            (qkv[0][c].clone(), qkv[1][c].clone(), qkv[2][c].clone());
        if let Some(bs) = vectors.biases_of(layer) {
            if let Some(bias) = opt_slice(&bs[0]) {
                add_assign_fixed(&mut query, bias)?;
            }
        }
        if let Some(bs) = vectors.biases_of(layer) {
            if let Some(bias) = opt_slice(&bs[1]) {
                add_assign_fixed(&mut key, bias)?;
            }
        }
        if let Some(bs) = vectors.biases_of(layer) {
            if let Some(bias) = opt_slice(&bs[2]) {
                add_assign_fixed(&mut value, bias)?;
            }
        }
        Ok((query, key, value))
    }

    /// The gated-query split: `query` becomes its `n_head·hd` query half and the gate half is
    /// returned; an ungated layer returns `None` and leaves `query` untouched.
    ///
    /// # Complexity
    ///
    /// `O(n_head·hd)` copies; `O(1)` otherwise.
    fn softmax_gate_split(
        query: &mut Vec<i64>,
        n_head: usize,
        hd: usize,
        gated: bool,
    ) -> Result<Option<Vec<i64>>, StepError> {
        if !gated {
            return Ok(None);
        }
        let (q, g) = split_query_gate(query, n_head, hd)?;
        *query = q;
        Ok(Some(g))
    }

    /// The QK-norm and `RoPE` of the column's query and key at `pos`.
    ///
    /// # Complexity
    ///
    /// As [`qk_norm_and_rope`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the two QK gains come from vectors, the rotation table, both vectors, the head \
                  geometry, the norm epsilon, the position and the schedule are the call's input"
    )]
    fn softmax_qk_rope(
        vectors: &DecodeVectors,
        layer: usize,
        table: &crate::engine::ops::rope::RopeTable,
        query: &mut [i64],
        key: &mut [i64],
        hd: usize,
        eps: i64,
        pos: u32,
        parallel: bool,
    ) -> Result<(), StepError> {
        qk_norm_and_rope(vectors, layer, table, query, key, hd, eps, pos, parallel)
    }

    /// Append the column's key and value to `kv` at `slot`, returning the position after the
    /// append, the prefix length this column attends over.
    ///
    /// # Complexity
    ///
    /// As [`KvState::append`].
    fn softmax_kv_append(
        kv: &mut KvState,
        slot: usize,
        key: &[i64],
        value: &[i64],
    ) -> Result<usize, StepError> {
        kv.append(slot, key, value).map_err(StepError::Kv)
    }

    /// The per-KV-head attention read of one column into an `q_out`-wide output; `dims` is
    /// `[n_head, hd, q_out, appended]`. One task owns one KV head and its `group` query heads, so
    /// the K/V rows are read once per group rather than once per query head.
    ///
    /// # Complexity
    ///
    /// `O(n_head·hd·appended)` word reads, `O(q_out)` output.
    fn softmax_head_loop(
        kv: &KvState,
        slot: usize,
        query: &[i64],
        dims: [usize; 4],
        parallel_heads: bool,
    ) -> Result<Vec<i64>, StepError> {
        let [n_head, hd, q_out, appended] = dims;
        let mut att: Vec<i64> = vec![0i64; q_out];
        let group = n_head / kv.n_head_kv();
        let width = group * hd;
        let head = QueryHeadCtx {
            kv,
            slot,
            query,
            width,
            appended,
        };
        if let Some((_, e)) = crate::engine::chart::ffn::attn_heads(
            kv.n_head_kv(),
            width,
            &mut att,
            parallel_heads,
            &head,
        ) {
            return Err(StepError::Kv(e));
        }
        Ok(att)
    }

    /// The attention output gate: `att[i] ← att[i] · sigmoid(gate[i])` when the layer is gated.
    ///
    /// # Complexity
    ///
    /// As [`apply_sigmoid_gate`].
    fn softmax_gate_apply(att: &mut [i64], gate: Option<&[i64]>) -> Result<(), StepError> {
        if let Some(g) = gate {
            apply_sigmoid_gate(att, g)?;
        }
        Ok(())
    }

    /// The attention core of a block of `k > 1` columns in three pooled stages, each column equal
    /// to the per-column loop in [`Engine::softmax_cols`]:
    /// 1. per column (one task each): bias, gate split, q/k norm and `RoPE` at position
    ///    `pos₀ + c`;
    /// 2. in token order: append each column's key and value;
    /// 3. per (column, KV head) (`k·n_head_kv` tasks): the `group = n_head / n_head_kv` query heads
    ///    of that KV head attend over positions `0..=pos₀ + c` in one read of its K/V rows, then
    ///    each head's sigmoid output gate is applied. The pool task index runs over the fold order
    ///    ([`fold_col`]) so a contiguous speed-weighted range alternates cheap and expensive
    ///    columns; a gather restores the block order.
    ///
    /// Column `c` sees exactly the keys the per-column loop had appended when it attended, so the
    /// block is causal and bit-identical (`qwen35_two_columns_equal_two_single_steps`). With
    /// `APEIRON_EXACT_SCAN=1` stage 3 reads each head through [`KvState::attn_head_fused`]
    /// instead of [`KvState::attn_head`]; that read is not bit-identical to the per-position path,
    /// and the deviation is bounded (`qwen35_exact_scan_attention_block_matches_baseline_within_bound`).
    /// A refusal leaves the cache unspecified.
    ///
    /// # Complexity
    ///
    /// `O(k·n_head·(pos₀ + k)·head_dim)` for the attention over `k·n_head_kv` pool tasks, plus
    /// `O(k·(q_proj + 2·kv_out))` for stages 1–2; three dispatches instead of `k`. With `tap`,
    /// stage 1's post-`RoPE` query and key and its value are appended to [`STAGE_Q`], [`STAGE_K`]
    /// and [`STAGE_V`] in block column order, one `O(q_out + 2·kv_out)` copy per column; the off
    /// path tests the `Option` once per block.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::many_single_char_names,
        reason = "the three stages read in order; c, h, n, q are the column, head, fold and query \
                  words; the layer, slot, cache, columns, epsilon, rotation table and tap are the \
                  block's input"
    )]
    pub(super) fn softmax_core_block(
        &self,
        layer: usize,
        slot: usize,
        kv: &mut KvState,
        qkv: &[Vec<Vec<i64>>],
        eps: i64,
        table: &crate::engine::ops::rope::RopeTable,
        tap: &mut Option<Tap>,
    ) -> Result<Vec<Vec<i64>>, StepError> {
        let cfg = &self.config;
        let (n_head, hd) = (cfg.n_head, cfg.head_dim);
        let q_out = n_head * hd;
        let kv_out = cfg.n_head_kv * hd;
        let k = qkv[0].len();
        let gated = cfg.attn_gate;
        let vectors = self.decode_vectors();
        let pos0 = kv.occupied(slot).map_err(StepError::Kv)?;
        // Stage 1: per column `[query | key | value | gate]`.
        let stride = 2 * q_out + 2 * kv_out;
        let mut prep = vec![0i64; k * stride];
        let one = PrepColCtx {
            qkv,
            vectors,
            layer,
            dims: [n_head, hd, q_out, kv_out, pos0],
            gated,
            eps,
            table,
        };
        if let Some((_, e)) =
            crate::engine::chart::ffn::attn_heads(k, stride, &mut prep, true, &one)
        {
            return Err(e);
        }
        // The tap's attention rows: stage 1's `[query | key | value]` per column, in block order.
        // The gate half after them is not an attention input.
        if let Some(t) = tap {
            let mut c = 0usize;
            while c < k {
                let base = c * stride;
                t.push(layer, STAGE_Q, &prep[base..base + q_out]);
                t.push(layer, STAGE_K, &prep[base + q_out..base + q_out + kv_out]);
                t.push(
                    layer,
                    STAGE_V,
                    &prep[base + q_out + kv_out..base + q_out + 2 * kv_out],
                );
                c += 1;
            }
        }
        // Stage 2: append in token order.
        let mut last = Vec::with_capacity(k);
        let mut err: Option<StepError> = None;
        let mut c = 0usize;
        while c < k && err.is_none() {
            let base = c * stride;
            let key = &prep[base + q_out..base + q_out + kv_out];
            let value = &prep[base + q_out + kv_out..base + q_out + 2 * kv_out];
            match kv.append(slot, key, value) {
                Ok(appended) => last.push(appended),
                Err(e) => err = Some(StepError::Kv(e)),
            }
            c += 1;
        }
        if let Some(e) = err {
            return Err(e);
        }
        // Stage 3: attention and gate, one pool task per (column, KV head) — or, on the F32
        // carrier, per (tile of consecutive columns, KV head), where a tile reads each K/V row once
        // for all of its columns. With `APEIRON_EXACT_SCAN=1` the per-column read is the fused
        // `(m, den, num)` fold; the flag is resolved here, on the calling thread, so the pool
        // workers run one path for the whole dispatch.
        //
        // The per-column task `t = n·n_kv + h` owns fold position `n`, whose key column is
        // `fold_col(n, k)`. A contiguous range of the speed-weighted split then alternates cheap and
        // expensive columns instead of taking a monotone run of the cheapest. The tile dispatch
        // applies the same fold at tile granularity. Each task reads only `prep`, the read-only KV
        // cache and `last`, so the permutations leave every word bit-identical; the gather after
        // the dispatch restores the tip's block order.
        let fused = exact_scan_enabled();
        // The tile entry rescales once per key tile (llama.cpp's tiled flash attention), so it
        // replaces the F32 and I8 carriers' per-column dispatch; the exact-scan fold and the Q32
        // oracle keep the per-column path.
        let tile_on = attn_tile_enabled()
            && !fused
            && k > 1
            && matches!(kv.carrier(), KvCarrier::F32 | KvCarrier::I8);
        let kv_ro: &KvState = kv;
        let n_kv = kv.n_head_kv();
        let group = n_head / n_kv;
        let width = group * hd;
        if tile_on {
            let tiles = k.div_ceil(QUERY_TILE);
            let tasks = tiles.checked_mul(n_kv).ok_or(StepError::LengthOverflow)?;
            // One `(fold position, KV head)` task per `QUERY_TILE · width` block. The buffer holds
            // a full block per task; the last tile writes and gathers only its `len · width`
            // prefix.
            let block = QUERY_TILE
                .checked_mul(width)
                .ok_or(StepError::LengthOverflow)?;
            let att_len = tasks.checked_mul(block).ok_or(StepError::LengthOverflow)?;
            let mut att = vec![0i64; att_len];
            // One status slot per task, written only by that task; the slots are reduced after the
            // dispatch (`first_by_key`), so no shared error cell crosses the pool boundary.
            let mut status: Vec<Option<StepError>> = Vec::with_capacity(tasks);
            let mut s = 0usize;
            while s < tasks {
                status.push(None);
                s += 1;
            }
            let dims = [n_kv, group, hd, kv_out, q_out, stride, k, tiles];
            let ctx = AttnTileCtx {
                kv: kv_ro,
                layer: slot,
                gated,
                prep: &prep,
                last: &last,
                dims,
            };
            let dispatched: Result<(), ()> = crate::engine::chart::pool::for_chunks2(
                &mut att,
                QUERY_TILE * width,
                &mut status,
                1,
                &ctx,
            );
            if dispatched.is_err() {
                return Err(StepError::BadConfig(String::from("attention dispatch")));
            }
            if let Some(e) = first_by_key(&mut status, tiles, n_kv) {
                // The reduction orders refusals by fold tile then KV head, not by column: the tile
                // kernel may reach a different faulting column first. The step discards the block
                // on any refusal, so the error kind is the observable, not which column raised it.
                return Err(e);
            }
            // Fold position `n` holds tile `fold_col(n, tiles)`, whose column `j` is block `j` of
            // task `n·n_kv + kvh`; gather it back to column order.
            let mut atts: Vec<Vec<i64>> = Vec::with_capacity(k);
            let mut c = 0usize;
            while c < k {
                atts.push(vec![0i64; q_out]);
                c += 1;
            }
            for n in 0..tiles {
                let t0 = fold_col(n, tiles) * QUERY_TILE;
                let len = (k - t0).min(QUERY_TILE);
                let mut kvh = 0usize;
                while kvh < n_kv {
                    let src = (n * n_kv + kvh) * QUERY_TILE * width;
                    let mut j = 0usize;
                    while j < len {
                        atts[t0 + j][kvh * width..(kvh + 1) * width]
                            .copy_from_slice(&att[src + j * width..src + (j + 1) * width]);
                        j += 1;
                    }
                    kvh += 1;
                }
            }
            return Ok(atts);
        }
        let mut att = vec![0i64; k * q_out];
        // One status slot per task, written only by that task; the slots are reduced after the
        // dispatch (`first_by_key`), so no shared error cell crosses the pool boundary. A task is
        // one (column, KV head): under `i8` one call folds the group's heads, under `q32` and the
        // untiled `f32` the group's heads are read one at a time, and the words are the per-head
        // words either way.
        let tasks = att.len().div_ceil(width.max(1));
        let mut status: Vec<Option<StepError>> = Vec::with_capacity(tasks);
        let mut s = 0usize;
        while s < tasks {
            status.push(None);
            s += 1;
        }
        let dims = [n_kv, group, hd, kv_out, q_out, stride, k];
        let ctx = AttnHeadCtx {
            kv: kv_ro,
            layer: slot,
            flags: [fused, gated],
            prep: &prep,
            last: &last,
            dims,
        };
        let dispatched: Result<(), ()> =
            crate::engine::chart::pool::for_chunks2(&mut att, width, &mut status, 1, &ctx);
        if dispatched.is_err() {
            return Err(StepError::BadConfig(String::from("attention dispatch")));
        }
        if let Some(e) = first_by_key(&mut status, k, n_kv) {
            return Err(e);
        }
        // Block `n` holds column `fold_col(n, k)`; gather it back to index `c` with the same `k`
        // `q_out`-wide contiguous copies the tip's `chunks(q_out).map(to_vec)` performed.
        let mut atts: Vec<Vec<i64>> = Vec::with_capacity(k);
        let mut j = 0usize;
        while j < k {
            atts.push(Vec::new());
            j += 1;
        }
        for n in 0..k {
            atts[fold_col(n, k)] = att[n * q_out..(n + 1) * q_out].to_vec();
        }
        Ok(atts)
    }
}

/// Stage-3 pool task `t = n·n_kv + kvh`: the `group` query heads of KV head `kvh` for key column
/// `c = fold_col(n, k)`, written into `dst` (`group · hd` words), gated when `flags[1]`.
/// `flags = [fused, gated]`, `dims = [n_kv, group, hd, kv_out, q_out, stride, k]`. A refusal goes
/// to `status[0]` and the task returns `Ok`, so the pool never sees an error and every task runs
/// once.
///
/// # Complexity
///
/// Time `O(group · hd · (pos + 1))` words of KV read for the group; no allocation.
#[allow(
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    reason = "flat scalar parameters keep the pool closure one call (Aeneas R2) with no nested region; the pool closure type is Result"
)]
fn attn_head_task(
    kv: &KvState,
    layer: usize,
    flags: [bool; 2],
    prep: &[i64],
    last: &[usize],
    dims: [usize; 7],
    task: usize,
    dst: &mut [i64],
    status: &mut [Option<StepError>],
) -> Result<(), ()> {
    let [n_kv, group, hd, kv_out, q_out, stride, cols] = dims;
    let (fold_pos, kvh) = (task / n_kv.max(1), task % n_kv.max(1));
    let col = fold_col(fold_pos, cols);
    let base = col * stride;
    let width = group * hd;
    let lo = kvh * width;
    // Query head `kvh · group + g` is `prep[base + (kvh · group + g) · hd ..]`, so the group's rows
    // are one contiguous run.
    let query = &prep[base + lo..base + lo + width];
    let tile = if flags[0] { EXACT_SCAN_TILE } else { 0 };
    let done = match kv.attn_head_group(layer, kvh, query, last[col], tile, dst) {
        Ok(()) if flags[1] => {
            let g0 = base + 2 * kv_out + q_out + lo;
            let mut g = 0usize;
            let mut res: Result<(), StepError> = Ok(());
            while g < group && res.is_ok() {
                let off = g * hd;
                res = apply_sigmoid_gate(&mut dst[off..off + hd], &prep[g0 + off..g0 + off + hd]);
                g += 1;
            }
            res
        }
        Ok(()) => Ok(()),
        Err(e) => Err(StepError::Kv(e)),
    };
    if let Err(e) = done {
        status[0] = Some(e);
    }
    Ok(())
}

/// The shared borrows of one attention-group dispatch; `run` is the single call to
/// [`attn_head_task`].
struct AttnHeadCtx<'a> {
    kv: &'a KvState,
    layer: usize,
    flags: [bool; 2],
    prep: &'a [i64],
    last: &'a [usize],
    dims: [usize; 7],
}

impl Task2<i64, Option<StepError>, ()> for AttnHeadCtx<'_> {
    fn run(&self, t: usize, dst: &mut [i64], st: &mut [Option<StepError>]) -> Result<(), ()> {
        attn_head_task(
            self.kv, self.layer, self.flags, self.prep, self.last, self.dims, t, dst, st,
        )
    }
}

/// Stage-3 pool task `t = n·n_kv + kvh` on the F32 carrier: the `QUERY_TILE` columns of fold
/// position `n` for KV head `kvh`, each column's `group` query heads gated when `gated` and written
/// into `dst` (`len · group · hd` words, `len <= QUERY_TILE`).
/// `dims = [n_kv, group, hd, kv_out, q_out, stride, cols, tiles]`. A refusal goes to `status[0]`
/// and the task returns `Ok`, so the pool never sees an error and every task runs once.
///
/// The columns of one tile are consecutive, so they are also consecutive positions, which is the
/// tile kernel's precondition: `last[t0 + j] == last[t0] + j`, checked here rather than assumed.
///
/// # Complexity
///
/// Time `O(len · group · hd · (last[t0] + 1))` words of KV read for the tile plus `O(len · width)`
/// query words copied; space `O(len · width)` query words.
#[allow(
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    reason = "flat scalar parameters keep the pool closure one call (Aeneas R2) with no nested region; the pool closure type is Result"
)]
fn attn_tile_task(
    kv: &KvState,
    layer: usize,
    gated: bool,
    prep: &[i64],
    last: &[usize],
    dims: [usize; 8],
    task: usize,
    dst: &mut [i64],
    status: &mut [Option<StepError>],
) -> Result<(), ()> {
    let [n_kv, group, hd, kv_out, q_out, stride, cols, tiles] = dims;
    let (fold_pos, kvh) = (task / n_kv.max(1), task % n_kv.max(1));
    let t0 = fold_col(fold_pos, tiles) * QUERY_TILE;
    let len = (cols - t0).min(QUERY_TILE);
    let width = group * hd;
    let lo = kvh * width;
    let first = last[t0];
    let mut queries = vec![0i64; len * width];
    let mut j = 0usize;
    while j < len {
        let c = t0 + j;
        if last[c] != first + j {
            status[0] = Some(StepError::Kv(AttnError::BadGeometry));
            return Ok(());
        }
        let base = c * stride;
        queries[j * width..(j + 1) * width].copy_from_slice(&prep[base + lo..base + lo + width]);
        j += 1;
    }
    let read = match kv.carrier() {
        KvCarrier::F32 => {
            kv.attn_head_group_tile_f32(layer, kvh, &queries, first, &mut dst[..len * width])
        }
        KvCarrier::I8 => {
            kv.attn_head_group_tile_i8(layer, kvh, &queries, first, &mut dst[..len * width])
        }
        KvCarrier::Q32 => Err(AttnError::BadGeometry),
    };
    let done = match read {
        Ok(()) if gated => {
            let g0 = t0 * stride + 2 * kv_out + q_out + lo;
            let mut res: Result<(), StepError> = Ok(());
            let mut j = 0usize;
            while j < len && res.is_ok() {
                let gate = &prep[g0 + j * stride..g0 + j * stride + width];
                let mut g = 0usize;
                while g < group && res.is_ok() {
                    let off = j * width + g * hd;
                    res = apply_sigmoid_gate(&mut dst[off..off + hd], &gate[g * hd..(g + 1) * hd]);
                    g += 1;
                }
                j += 1;
            }
            res
        }
        Ok(()) => Ok(()),
        Err(e) => Err(StepError::Kv(e)),
    };
    if let Err(e) = done {
        status[0] = Some(e);
    }
    Ok(())
}

/// The shared borrows of one tiled attention dispatch; `run` is the single call to
/// [`attn_tile_task`].
struct AttnTileCtx<'a> {
    kv: &'a KvState,
    layer: usize,
    gated: bool,
    prep: &'a [i64],
    last: &'a [usize],
    dims: [usize; 8],
}

impl Task2<i64, Option<StepError>, ()> for AttnTileCtx<'_> {
    fn run(&self, t: usize, dst: &mut [i64], st: &mut [Option<StepError>]) -> Result<(), ()> {
        attn_tile_task(
            self.kv, self.layer, self.gated, self.prep, self.last, self.dims, t, dst, st,
        )
    }
}

/// The shared borrows of one KV head's attention read; `run` is the single call to
/// [`query_group_attn`].
struct QueryHeadCtx<'a> {
    kv: &'a KvState,
    slot: usize,
    query: &'a [i64],
    /// `group · hd`: the `n_head / n_head_kv` query heads one KV head serves, as one row.
    width: usize,
    appended: usize,
}

impl HeadTask<i64, AttnError> for QueryHeadCtx<'_> {
    fn run(&self, kv_head: usize, dst: &mut [i64]) -> Result<(), AttnError> {
        query_group_attn(
            self.kv,
            self.slot,
            self.query,
            self.width,
            self.appended,
            kv_head,
            dst,
        )
    }
}

/// One KV head of the query read: its `width`-word query row, attended over `appended` positions
/// into `dst` in the same layout.
fn query_group_attn(
    kv: &KvState,
    slot: usize,
    query: &[i64],
    width: usize,
    appended: usize,
    kv_head: usize,
    dst: &mut [i64],
) -> Result<(), AttnError> {
    let lo = kv_head * width;
    kv.attn_head_group(slot, kv_head, &query[lo..lo + width], appended, 0, dst)
}

/// The shared borrows of one column's stage-1 preparation; `run` is the single call to [`prep_column`].
struct PrepColCtx<'a> {
    qkv: &'a [Vec<Vec<i64>>],
    vectors: &'a DecodeVectors,
    layer: usize,
    /// `[n_head, hd, q_out, kv_out, pos0]`.
    dims: [usize; 5],
    gated: bool,
    eps: i64,
    table: &'a crate::engine::ops::rope::RopeTable,
}

impl HeadTask<i64, StepError> for PrepColCtx<'_> {
    fn run(&self, c: usize, dst: &mut [i64]) -> Result<(), StepError> {
        prep_column(
            c,
            dst,
            self.qkv,
            self.vectors,
            self.layer,
            self.dims,
            self.gated,
            self.eps,
            self.table,
        )
    }
}

/// Stage 1 for column `c`: the biases, the optional gate split, the QK-norm and rotation, written to
/// `dst` as `[query | key | value | gate]`. The stages are the same calls [`Engine::softmax_col`]
/// makes, so the block and the per-column path share one body per stage.
#[allow(
    clippy::too_many_arguments,
    clippy::index_refutable_slice,
    reason = "the column's inputs are the call's full state, the geometry is packed in dims; the three biases are read by constant index because an array pattern does not extract"
)]
fn prep_column(
    c: usize,
    dst: &mut [i64],
    qkv: &[Vec<Vec<i64>>],
    vectors: &DecodeVectors,
    layer: usize,
    dims: [usize; 5],
    gated: bool,
    eps: i64,
    table: &crate::engine::ops::rope::RopeTable,
) -> Result<(), StepError> {
    let [n_head, hd, q_out, kv_out, pos0] = dims;
    let (mut query, mut key, value) = Engine::softmax_bias_add(qkv, vectors, layer, c)?;
    let gate = prep_gate_split(&mut query, n_head, hd, gated)?;
    let pos = u32::try_from(pos0 + c).map_err(|_| StepError::LengthOverflow)?;
    Engine::softmax_qk_rope(
        vectors, layer, table, &mut query, &mut key, hd, eps, pos, false,
    )?;
    prep_copy_out(dst, &query, &key, &value, &gate, [q_out, kv_out], gated)
}

/// The gate half of [`prep_column`]'s query, or `Vec::new()` for an ungated layer. `Engine::softmax_gate_split`
/// returns the option; the pool writes a flat `[query | key | value | gate]` slot, so the absent case is empty.
///
/// # Complexity
///
/// `O(n_head·hd)` copies; `O(1)` otherwise.
fn prep_gate_split(
    query: &mut Vec<i64>,
    n_head: usize,
    hd: usize,
    gated: bool,
) -> Result<Vec<i64>, StepError> {
    match Engine::softmax_gate_split(query, n_head, hd, gated)? {
        Some(g) => Ok(g),
        None => Ok(Vec::new()),
    }
}

/// Write the column's `[query | key | value | gate]` into the pool slot `dst`; `dims` is
/// `[q_out, kv_out]`. A width mismatch is refused before any byte is written.
///
/// # Complexity
///
/// `O(q_out + 2·kv_out)` word copies; `O(1)` besides.
fn prep_copy_out(
    dst: &mut [i64],
    query: &[i64],
    key: &[i64],
    value: &[i64],
    gate: &[i64],
    dims: [usize; 2],
    gated: bool,
) -> Result<(), StepError> {
    let [q_out, kv_out] = dims;
    if query.len() != q_out || key.len() != kv_out || value.len() != kv_out {
        return Err(StepError::BadConfig(String::from("attention block widths")));
    }
    let (dq, rest) = dst.split_at_mut(q_out);
    let (dk, rest) = rest.split_at_mut(kv_out);
    let (dv, dg) = rest.split_at_mut(kv_out);
    dq.copy_from_slice(query);
    dk.copy_from_slice(key);
    dv.copy_from_slice(value);
    if gated {
        dg.copy_from_slice(gate);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic small Q32.32 word: |value| <= 2000/2^12, in range for the gate product.
    #[allow(
        clippy::cast_possible_wrap,
        reason = "the modulo keeps the value below 4001, far inside i64"
    )]
    fn word(seed: usize) -> i64 {
        (((seed.wrapping_mul(2_654_435_761) % 4_001) as i64) - 2_000) << 20
    }

    /// [`attn_tile_task`] agrees with [`attn_head_task`] on the same block within the per-tile
    /// rescale bound, with and without the gate, across a tile boundary and on the last partial
    /// tile; and the gate changes the tile's output, so the comparison is not two ungated paths.
    #[test]
    fn the_attention_tile_task_agrees_with_the_per_column_task() {
        let (n_head, n_head_kv, hd) = (4usize, 2usize, 8usize);
        let group = n_head / n_head_kv;
        let width = group * hd;
        let (q_out, kv_out) = (n_head * hd, n_head_kv * hd);
        let stride = 2 * q_out + 2 * kv_out;
        let cols = QUERY_TILE + 3;
        let tiles = cols.div_ceil(QUERY_TILE);
        assert_eq!(tiles, 2, "the fixture must cross a tile boundary");
        let cap = cols + 4;
        let mut kv = KvState::with_carrier(1, n_head, n_head_kv, hd, cap, KvCarrier::F32)
            .expect("kv geometry");
        let mut s = 0usize;
        while s < cap {
            let k: Vec<i64> = (0..kv_out).map(|i| word(s * kv_out + i)).collect();
            let v: Vec<i64> = (0..kv_out).map(|i| word(s * kv_out + i + 7)).collect();
            assert_eq!(kv.append(0, &k, &v), Ok(s));
            s += 1;
        }
        let last: Vec<usize> = (0..cols).collect();
        let prep: Vec<i64> = (0..cols * stride).map(|i| word(i + 11)).collect();
        let dims8 = [n_head_kv, group, hd, kv_out, q_out, stride, cols, tiles];
        let dims7 = [n_head_kv, group, hd, kv_out, q_out, stride, cols];
        let block = QUERY_TILE * width;
        let mut gathered: Vec<Vec<i64>> = Vec::new();
        for gated in [false, true] {
            let mut tile_buf = vec![0i64; tiles * n_head_kv * block];
            for task in 0..tiles * n_head_kv {
                let mut st = [None];
                attn_tile_task(
                    &kv,
                    0,
                    gated,
                    &prep,
                    &last,
                    dims8,
                    task,
                    &mut tile_buf[task * block..(task + 1) * block],
                    &mut st,
                )
                .expect("tile task");
                assert!(st[0].is_none(), "tile task {task}, gated {gated}");
            }
            // The per-column tasks wrote one `width` block per (fold position, KV head). Lay them
            // out by block column, as the step's gather does.
            let mut by_col = vec![0i64; cols * n_head_kv * width];
            for task in 0..cols * n_head_kv {
                let (fold_pos, kvh) = (task / n_head_kv, task % n_head_kv);
                let col = fold_col(fold_pos, cols);
                let dst = &mut by_col[col * n_head_kv * width..(col + 1) * n_head_kv * width];
                let mut st = [None];
                attn_head_task(
                    &kv,
                    0,
                    [false, gated],
                    &prep,
                    &last,
                    dims7,
                    task,
                    &mut dst[kvh * width..(kvh + 1) * width],
                    &mut st,
                )
                .expect("column task");
                assert!(st[0].is_none(), "column task {task}, gated {gated}");
            }
            let mut out = vec![0i64; cols * n_head_kv * width];
            for n in 0..tiles {
                let t0 = fold_col(n, tiles) * QUERY_TILE;
                let len = (cols - t0).min(QUERY_TILE);
                for kvh in 0..n_head_kv {
                    let src = (n * n_head_kv + kvh) * block;
                    for j in 0..len {
                        let col = t0 + j;
                        let at = col * n_head_kv * width + kvh * width;
                        let got = &tile_buf[src + j * width..src + (j + 1) * width];
                        for i in 0..width {
                            let (x, y) = (got[i], by_col[at + i]);
                            let m = i128::from(x.abs().max(y.abs()));
                            let tol = 64
                                + i64::try_from((m * 512 * (col as i128 + 65)) >> 32)
                                    .expect("the tile bound fits i64");
                            assert!(
                                (x - y).abs() <= tol,
                                "gated {gated}, tile {n}, column {col}, kv head {kvh} word {i}: {x} vs {y} (tol {tol})"
                            );
                        }
                        out[at..at + width].copy_from_slice(got);
                    }
                }
            }
            gathered.push(out);
        }
        assert!(
            gathered[0] != gathered[1],
            "the gate must change the ungated output, or the gated leg proves nothing"
        );
    }
}
