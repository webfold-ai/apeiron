//! The `NanoQuant` sign-rank product in Q32.32:
//! `y = s_post ⊙ U (s_mid ⊙ (V (s_pre ⊙ x))) + Σ_j salient[·, j]·x[idx[j]] + Σ_l s_res_l ⊙ (sign_l x)`.
//!
//! [`NqLinear::project`] is the CPU kernel the packed linears run. `x`, `t = s_pre ⊙ x`, `z = V t`
//! and `y` are fixed-point vectors; `U`, `V` and every residual plane are sign matrices stored as
//! bits, so each product is a sum of add/subtracts and never a multiply. The only rounding is the
//! documented Q32.32 epilogues: one [`mul_q32`] per pre-scaled activation, one per mid-scaled
//! rank, and one on the sum of the post-scaled output, the sparse correction and every dense
//! residual. A factor or output that leaves `i64` refuses with [`NqError::Overflow`].
//!
//! **Accumulator width.** A sum of `n` terms each at most `max` in magnitude is at most `n·max`,
//! checked before the pass. When that bound fits `i64` (every real activation), the sums run in
//! `i64`; otherwise the same sign-dot source runs in `i128` so cancellation still decides the
//! result instead of a conservative refusal. The bound is a guard, not a second kernel.
//!
//! **One face.** This is not a second forward and not a dtype-specific copy: the one dispatch
//! point `Engine::step_proj_sched` selects it for a tensor the pack replaces, exactly as it
//! selects the payload kernels for a stored tensor. A fitted tensor has no GGUF row, so the kernel
//! mode chooses the activation reader over the factors: the exact Q32.32 words
//! ([`NqLinear::project`]), the Q8 block codes ([`NqLinear::project_q8`]) or the ternary block masks
//! ([`NqLinear::project_ternary`]), the same quantized activation contract the stored rows read. The
//! pack reader decodes the
//! three scale vectors and every residual row scale once ([`PackedLinear`], `bf16_to_fixed`) and
//! the sparse correction with the `f16_to_fixed` the other `F16` rows use.
//!
//! **Bit convention.** Bit `1` in `U`/`V` is `+1`, bit `0` is `-1` (the exporter's convention; the
//! dense reference words of `nq-ref7` reproduce with it). The trailing bits of the last word are
//! padding and are ignored. [`sign_dot!`] maps bit `1` to `-t`; `U` and `V` are always read as a
//! product, so the shared inversion cancels, while a residual plane is read once and its dot is
//! negated.
//!
//! **Parallelism.** The `V` rows split across the engine's persistent pool: task `k` owns one
//! `z_k`, so the `rank × inn` adds are shared. The `U` rows then split the same way over output
//! rows. Both dispatches write disjoint chunks through [`Pool::chunks_mut`]; the caller is the
//! step's main thread, never a pool task. A task folds a row block against two adjacent columns
//! and decodes the weight word once for both, each column keeping its own codes, shift vectors and
//! `e_min`. The per-column pack (the `t` build, the `t` and `z` quantizations and the
//! `rank·max|z|` bound) is a column-band dispatch on the same pool, so a packed column's cost does
//! not land on the caller while the workers spin. The residual's `x` reader stays on the caller
//! and is absent from a pack with no residual layer.
//!
//! **Cost.** One projection is `O(rank·inn + out·rank + res_layers·out·inn)` additions,
//! `O(out · inn)` weights replaced, `O(rank)` scratch for `z`, `O(inn)` for `t` and `O(out)`
//! output.

#![allow(
    clippy::many_single_char_names,
    reason = "`x`, `t`, `z`, `y`, `U`, `V` are the derivation's names; the section counters are short"
)]

use crate::engine::chart::pool::{for_chunks2, Pool};
use crate::engine::chart::task::Task2;
use crate::engine::chart::{quant_act, QuantAct};
use crate::engine::kern::pow2::act_scale;
use crate::engine::num::dequant::{bf16_to_fixed, f16_to_fixed};
use crate::engine::num::value::mul_q32;
use crate::format::nq::{PackedLinear, Plane};

/// A refused sign-rank product or factor decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NqError {
    /// The activation width is not the linear's input width.
    Shape {
        /// Activation length.
        x: usize,
        /// Input width.
        n_in: usize,
    },
    /// A packed factor's field lengths disagree with its dimensions.
    Packed {
        /// Which field.
        field: &'static str,
        /// Length found.
        have: usize,
        /// Length the dimensions require.
        expected: usize,
    },
    /// A 16-bit scale (bfloat16 or binary16) is not finite or leaves the Q32.32 grid.
    Scale {
        /// The raw binary16 bit pattern.
        bits: u16,
    },
    /// A salient index is outside the input width or repeats an earlier index.
    SalientIndex {
        /// Index found.
        index: usize,
        /// Input width.
        inn: usize,
    },
    /// A checked fixed-point product, sum or narrowing left its range.
    Overflow,
}

impl core::fmt::Display for NqError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NqError::Shape { x, n_in } => {
                write!(f, "nq projection: activation {x}, want {n_in}")
            }
            NqError::Packed {
                field,
                have,
                expected,
            } => write!(f, "nq factor: {field} has {have}, want {expected}"),
            NqError::Scale { bits } => write!(f, "nq scale bits {bits:#06x}"),
            NqError::SalientIndex { index, inn } => {
                write!(f, "nq salient index {index} outside {inn}")
            }
            NqError::Overflow => write!(f, "nq projection: fixed-point overflow"),
        }
    }
}

impl std::error::Error for NqError {}

/// The sign dot `Σ (1 − 2·bit)·t[i]` of one packed row, written once for both accumulator widths.
///
/// `$acc` is `i64` when the caller's `n·max|t|` bound fits and `i128` otherwise; the bit mask is
/// all ones for `1` and zero for `0`, so `(t ^ mask) - mask` is `t` or `-t` without a branch.
macro_rules! sign_dot {
    ($t:expr, $words:expr, $len:expr, $acc:ty) => {{
        let mut acc: $acc = 0;
        let mut i = 0usize;
        for &wordbits in $words {
            let mut bits = wordbits;
            let end = (i + 32).min($len);
            while i < end {
                let mask = -<$acc>::from(bits & 1);
                acc += ($t[i] as $acc ^ mask) - mask;
                bits >>= 1;
                i += 1;
            }
        }
        acc
    }};
}

/// The narrow sign dot: the AVX2 byte source when the stage's planes are present, the AVX2
/// `i64`-lane source otherwise, the scalar macro when the host has no AVX2.
///
/// `t.len()` is at least `len` and `words` covers those `len` bits; every caller passes the
/// `len` terms the caller's `len·max|t| ≤ i64::MAX` bound covers. `sum` is `Σ t[..len]` and
/// `planes` the four-digit split of the same terms; the `i64`-lane source and the macro read
/// neither.
fn sign_dot_i64(
    t: &[i64],
    planes: Option<&BytePlanes>,
    sum: i64,
    words: &[u32],
    len: usize,
) -> i64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        if let Some(planes) = planes {
            // SAFETY: the detection above names AVX2; `t.len() >= len`, `words` covers `len`
            // bits, `sum` is `Σ t[..len]`, and `Terms` builds planes only for a term set whose
            // four digits and `i32` lane sums are exact.
            return unsafe { sign_dot_planes_avx2(t, planes, sum, words, len) };
        }
        if std::arch::is_x86_feature_detected!("popcnt") {
            // SAFETY: the detection above names AVX2 and POPCNT, and `t.len() >= len` is the
            // callers' precondition, the same prefix the scalar macro indexes.
            return unsafe { sign_dot_avx2(t, words, len) };
        }
    }
    sign_dot!(t, words, len, i64)
}

/// The 64-term ternary cell in safe Rust: `Σ_{i<64} (1 − 2·bit_i(w))·t_i` with `t_i = [p_i] − [n_i]`
/// and the contract `p & n = 0`. Bit `i` of `w` is the stored weight sign, bit `i` of `p`/`n` the
/// positive/negative trit; the three states per term are exact:
/// `00 → 0`, `10 → 1 − 2b`, `01 → 2b − 1`.
///
/// This is the extracted body (the denotation `proof/corpus/WordRefinement.lean` states); the row
/// loop is the same identity with the row-independent `pc(p) − pc(n)` folded into one constant per
/// activation vector. `count_ones` is the intrinsic the twin names.
///
/// # Complexity
///
/// `O(1)` time: two masks, four popcounts, three subtractions. `O(1)` space.
#[allow(
    clippy::cast_possible_truncation,
    reason = "`count_ones` of a `u64` is at most 64, exact in `i32`"
)]
fn ternary_block(w: u64, p: u64, n: u64) -> i32 {
    2 * ((w & n).count_ones().cast_signed()) - 2 * ((w & p).count_ones().cast_signed())
        + (p.count_ones().cast_signed())
        - (n.count_ones().cast_signed())
}

/// Four `i64` lanes per `__m256i`: the vector twin of [`sign_dot!`] for the narrow accumulator.
///
/// AVX2 has no `vpsignq`, so bit `b` of a word becomes the mask `-(b)` and each lane is
/// `(t ^ mask) − mask`: `t` for `0`, `−t` for `1`, the scalar macro's own expression. One word
/// drives eight four-lane groups; the trailing word shorter than 32 terms runs the scalar source,
/// so bits past `len` are padding on both paths.
///
/// `− mask` is `+ b`, and `Σ b` over a word is its `popcount`: the bit sum is a property of the
/// packed word, not of the lanes. The vector accumulator therefore holds only `Σ (t ^ mask)` and
/// the bit count is added once per word in `i64`, which removes the eight `vpsrlq` and eight
/// `vpaddq` the lane-wise form spent rebuilding it. The intermediates are the same ones the
/// lane-wise expression produces, so the narrow-accumulator bound is unchanged.
///
/// # Complexity
///
/// `O(len)` terms: `len/32` words, each eight groups of four loads, masks and adds, plus one
/// scalar popcount per word and at most 31 scalar tail terms. `O(1)` space: one vector
/// accumulator in registers.
///
/// # Safety
///
/// The host CPU supports AVX2, and `t.len() >= len`: the groups read `t[..len]`, the same prefix
/// [`sign_dot!`] indexes. The caller's `len·max|t| ≤ i64::MAX` bound keeps every lane sum exact,
/// because each lane and horizontal fold is a subset sum of `±t`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "`_mm256_loadu_si256` is the unaligned load; the cast target's alignment is not a \
              requirement of the instruction"
)]
unsafe fn sign_dot_avx2(t: &[i64], words: &[u32], len: usize) -> i64 {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_castsi256_si128,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_set1_epi64x, _mm256_setr_epi64x,
        _mm256_setzero_si256, _mm256_srli_epi64, _mm256_srlv_epi64, _mm256_sub_epi64,
        _mm256_xor_si256, _mm_add_epi64, _mm_cvtsi128_si64, _mm_unpackhi_epi64,
    };
    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi64x(1);
    // `bits` is shifted right four bits per group, so these are the four bit positions the
    // group's lanes read relative to its first term.
    let lane = _mm256_setr_epi64x(0, 1, 2, 3);
    let mut acc = zero;
    let mut i = 0usize;
    let mut bits_total = 0i64;
    for &word in words {
        if i + 32 > len {
            break;
        }
        bits_total += i64::from(word.count_ones());
        let mut bits = _mm256_set1_epi64x(i64::from(word));
        let mut g = 0usize;
        while g < 8 {
            let at = i + g * 4;
            // SAFETY: `i + 32 <= len <= t.len()`, so this load reads inside `t`.
            unsafe {
                let v = _mm256_loadu_si256(t.as_ptr().add(at).cast::<__m256i>());
                let bit = _mm256_and_si256(_mm256_srlv_epi64(bits, lane), one);
                let mask = _mm256_sub_epi64(zero, bit);
                acc = _mm256_add_epi64(acc, _mm256_xor_si256(v, mask));
            }
            bits = _mm256_srli_epi64::<4>(bits);
            g += 1;
        }
        i += 32;
    }
    let half = _mm_add_epi64(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    let half = _mm_add_epi64(half, _mm_unpackhi_epi64(half, half));
    let mut total = _mm_cvtsi128_si64(half) + bits_total;
    if i < len {
        // One partial word is left and it starts at `i`, a multiple of 32; its bits past `len`
        // are padding. The scalar source cuts at the same bit, so both paths skip them.
        let tail_words = &words[(i / 32).min(words.len())..];
        total += sign_dot!(&t[i..], tail_words, len - i, i64);
    }
    total
}

/// The four signed byte digits of one stage's terms, little-endian two's complement:
/// `Σ_b 2^{8b}·plane[b][i] == exact[i]` for every term `i`.
///
/// Four planes are the same bytes a 32-bit term array would hold, split into four streams the
/// vector source reads one `__m256i` at a time. [`BytePlanes::of`] returns `None` when a term
/// needs a fifth digit, so the split is exact over its whole domain rather than rounded.
struct BytePlanes {
    /// `plane[b][i]` is digit `b` of term `i`, as its two's-complement byte.
    plane: [Box<[u8]>; 4],
}

impl BytePlanes {
    /// Split `exact` into four signed byte digits, or `None` when a term leaves the range four
    /// digits cover: `−2,155,905,152 … 2,139,062,143`, the `i32` range less its top 8,421,504
    /// values.
    ///
    /// # Complexity
    ///
    /// `O(n)` time, four shift-and-subtract steps per term; `O(n)` space, four bytes per term.
    fn of(exact: &[i64]) -> Option<Self> {
        let n = exact.len();
        let mut plane: [Box<[u8]>; 4] = std::array::from_fn(|_| vec![0u8; n].into_boxed_slice());
        for (i, &value) in exact.iter().enumerate() {
            let mut rest = value;
            for digit in &mut plane {
                let byte = rest.to_le_bytes()[0];
                // The byte read back as `i8` without a cast: the pattern is the digit.
                let signed = i64::from(i8::from_ne_bytes([byte]));
                digit[i] = byte;
                rest = (rest - signed) >> 8;
            }
            if rest != 0 {
                return None;
            }
        }
        Some(Self { plane })
    }
}

/// Whether the byte source's `i32` lanes hold every partial sum: each lane takes four digits per
/// word and a digit is at most `128` below zero, so `512·ceil(len/32) ≤ i32::MAX` bounds a lane,
/// and the `i64` epilogue stays inside `i64` for every `len` this admits.
fn lane_sums_fit(len: usize) -> bool {
    i32::try_from(len.div_ceil(32).saturating_mul(512)).is_ok()
}

/// Thirty-two terms per `__m256i`: the byte-plane twin of [`sign_dot!`] for the narrow accumulator.
///
/// A word's 32 bits become 32 `i8` selectors in five instructions: one broadcast, one `vpsrlvd`
/// that puts bits `4k..4k+3` in the low nibble of dword `k`, one `vpshufb` that repeats that
/// nibble across the dword's four bytes, one `vpand` with `[1,2,4,8]`, and one `vpshufb` that maps
/// `2^j` to `1`. Each digit plane then costs one `vpmaddubsw` (unsigned selector × signed digit,
/// adjacent pairs in `i16`), one `vpmaddwd` with ones (pairs to four-term `i32` lanes) and one
/// `vpaddd`; the load folds into the `vpmaddubsw`. The four plane sums are combined as
/// `Σ_b 2^{8b}·(Σ_{bit=1} digit_b) = Σ_{bit=1} t`, and the result is `Σ t − 2·Σ_{bit=1} t`, the
/// macro's `Σ (1 − 2·bit)·t`.
///
/// # Complexity
///
/// `O(len)` terms: `len/32` words of five selector instructions and four planes of three vector
/// operations, plus at most 31 scalar tail terms. `O(1)` space: four accumulator registers.
///
/// # Safety
///
/// The host CPU supports AVX2; `t.len() >= len`; each of `planes.plane` holds at least `len`
/// bytes; `words` covers `len` bits; `sum == Σ t[..len]`; every term splits into four digits (the
/// [`BytePlanes`] invariant); and `lane_sums_fit(len)` holds, so each `i32` lane sum, the four
/// `i64` plane sums and `2·Σ_{bit=1} t` are exact.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "`_mm256_loadu_si256` is the unaligned load; the cast target's alignment is not a \
              requirement of the instruction"
)]
unsafe fn sign_dot_planes_avx2(
    t: &[i64],
    planes: &BytePlanes,
    sum: i64,
    words: &[u32],
    len: usize,
) -> i64 {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_and_si256, _mm256_castsi256_si128,
        _mm256_cvtepi32_epi64, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32,
        _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srlv_epi32,
        _mm_add_epi64, _mm_cvtsi128_si64, _mm_unpackhi_epi64,
    };
    // `[1,2,4,8]` in each dword: byte `j` of dword `k` tests bit `4k + j` of the word.
    let digit_mask = _mm256_set1_epi32(0x0804_0201);
    // `word >> 4k` in dword `k`, so its low nibble holds bits `4k..4k+3`.
    let bit_at = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    // Byte `j` of dword `k` takes lane byte `4k`: the low nibble, four times.
    let repeat = _mm256_setr_epi8(
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12,
        12, 12, 12,
    );
    // The byte `2^j` maps to `1`; every other byte maps to `0`.
    let select_one = _mm256_setr_epi8(
        0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0,
    );
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_si256(); 4];
    let mut i = 0usize;
    for &word in words {
        if i + 32 > len {
            break;
        }
        let spread = _mm256_srlv_epi32(_mm256_set1_epi32(word.cast_signed()), bit_at);
        let nibble = _mm256_shuffle_epi8(spread, repeat);
        let select = _mm256_shuffle_epi8(select_one, _mm256_and_si256(nibble, digit_mask));
        let mut b = 0usize;
        while b < 4 {
            // SAFETY: `i + 32 <= len <= planes.plane[b].len()`, so the load reads inside it.
            let digits =
                unsafe { _mm256_loadu_si256(planes.plane[b].as_ptr().add(i).cast::<__m256i>()) };
            let pair = _mm256_maddubs_epi16(select, digits);
            acc[b] = _mm256_add_epi32(acc[b], _mm256_madd_epi16(pair, ones));
            b += 1;
        }
        i += 32;
    }
    // The four plane sums to `i64`, then `Σ_b 2^{8b}·(Σ_{bit=1} digit_b)`.
    let mut selected = 0i64;
    let mut b = 0usize;
    while b < 4 {
        let lane = acc[b];
        let lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(lane));
        let hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(lane));
        let quad = _mm256_add_epi64(lo, hi);
        let half = _mm_add_epi64(
            _mm256_castsi256_si128(quad),
            _mm256_extracti128_si256::<1>(quad),
        );
        let half = _mm_add_epi64(half, _mm_unpackhi_epi64(half, half));
        selected += _mm_cvtsi128_si64(half) << (8 * b);
        b += 1;
    }
    let mut total = sum;
    if i < len {
        // The partial word starts at `i`, a multiple of 32; its bits past `len` are padding.
        // The plane identity covers `t[..i]`, so the tail's own `Σ t` leaves `total` and the
        // scalar source supplies the tail's signed sum.
        let tail_len = len - i;
        let tail = &t[i..i + tail_len];
        total -= tail.iter().copied().sum::<i64>();
        total += sign_dot!(tail, &words[(i / 32).min(words.len())..], tail_len, i64);
    }
    total - 2 * selected
}

/// One stage's terms in the forms the sign dot reads: the exact `i64` values, their four signed
/// byte digits when the split and the lane sums are exact, and the exact row sum.
struct Terms<'a> {
    /// The exact terms: the scalar source's values and the `i64`-lane source's input.
    exact: &'a [i64],
    /// The four byte digits, present when every term splits into four and the lanes fit.
    planes: Option<BytePlanes>,
    /// `Σ exact`, the byte source's `Σ t` term.
    sum: i64,
}

impl<'a> Terms<'a> {
    /// Read `exact` into its three forms: one `O(n)` pass for the sum and one for the digits.
    ///
    /// The caller has checked `n·max|exact| ≤ i64::MAX`, so the sum and the plane source's
    /// intermediates are exact.
    fn new(exact: &'a [i64]) -> Self {
        let sum = exact.iter().copied().sum();
        let planes = if lane_sums_fit(exact.len()) {
            BytePlanes::of(exact)
        } else {
            None
        };
        Self { exact, planes, sum }
    }

    /// The narrow sign dot of `words` over `exact[..len]`; `len` is `exact.len()`.
    fn dot(&self, words: &[u32], len: usize) -> i64 {
        sign_dot_i64(self.exact, self.planes.as_ref(), self.sum, words, len)
    }
}

/// The Q8 block reader's safe twin: `Σ_b S_b · Σ_{bit=1} codes[j]` over `len` terms, folded block by
/// block in `i128` with no intrinsic. This is the selected sum, one half of the sign dot
/// (`Σ x̂ − 2·selected`); the block scale is applied once per 32 terms, so the twin is the
/// definition of the quantized selection, not a rearrangement of the exact Q32.32 dot.
fn code_select_twin(codes: &[i8], scales: &[i64], words: &[u32], len: usize) -> i128 {
    let mut acc = 0i128;
    let mut i = 0usize;
    while i < len {
        let block = i / 32;
        let end = (i + 32).min(len);
        let mut inner = 0i64;
        while i < end {
            if (words[block] >> (i % 32)) & 1 == 1 {
                inner += i64::from(codes[i]);
            }
            i += 1;
        }
        acc += i128::from(scales[block]) * i128::from(inner);
    }
    acc
}

/// One activation in the Q8 block reader: the int8 codes, their per-32 block scales, and the
/// exponent shifts that let one `i32` lane accumulator add every block's selected sum before a
/// single `e_min` shift at the row's end.
///
/// The scales are powers of two (`quant_act`'s `act_scale`), so block `b` contributes
/// `2^{e_min + shifts[b]} · inner_b`; normalizing by the smallest exponent turns the per-block
/// multiply into a per-lane shift. [`CodeTerms::new`] admits only the terms whose lane sum fits
/// `i32` and whose total fits `i64`, and refuses the rest, so the reader is exact over its whole
/// domain rather than truncated.
struct CodeTerms<'a> {
    /// The int8 codes, one per activation element.
    codes: &'a [i8],
    /// The verbatim per-block scale, the scalar dispatch's input.
    scales: &'a [i64],
    /// `e_b − e_min`, one `i32` per 32-term block. The body broadcasts it with one `vpbroadcastd`
    /// from memory; an eight-lane copy per block is four times the activation reader's bytes and
    /// does not fit the L1 working set of a wide tile.
    shift_vecs: Vec<i32>,
    /// The smallest block exponent; the row's lane sum shifts left by it once.
    e_min: u32,
    /// `Σ_b S_b·Σ_j codes[j]`, the dequantized activation's total, the sign dot's `Σ x̂` term.
    sum: i64,
    /// Whether the host runs the AVX2 selector: detected once per activation instead of once per
    /// output row, where the same cached load competed with the row's own work.
    vector: bool,
}

/// One activation column as the wide column tile reads it: the codes, the verbatim block scales,
/// the normalized shift vectors, the smallest block exponent and the host's selector flag.
#[derive(Clone, Copy)]
struct ColRef<'a> {
    codes: &'a [i8],
    scales: &'a [i64],
    shift_vecs: &'a [i32],
    e_min: u32,
    vector: bool,
}

impl<'a> CodeTerms<'a> {
    /// Read `q` into the normalized block form, or `None` when a scale is not a power of two, the
    /// lane sum leaves `i32`, or the total leaves `i64`.
    ///
    /// # Complexity
    ///
    /// `O(n/32)` time and space in the activation length `n`: two passes over the blocks (the
    /// smallest exponent, then the shifts and the two lane bounds) and one vector pass over the
    /// codes for the total.
    fn new(q: &'a QuantAct) -> Option<Self> {
        let codes = q.codes();
        let scales = q.scales();
        let blocks = codes.len().div_ceil(32);
        if scales.len() < blocks {
            return None;
        }
        let mut e_min = u32::MAX;
        let mut b = 0usize;
        while b < blocks {
            let s = scales[b];
            if s != 0 {
                if s < 0 || !s.unsigned_abs().is_power_of_two() {
                    return None;
                }
                e_min = e_min.min(s.unsigned_abs().trailing_zeros());
            }
            b += 1;
        }
        let e_min = if e_min == u32::MAX { 0 } else { e_min };
        let mut shift_vecs: Vec<i32> = Vec::with_capacity(blocks);
        let mut norm: u64 = 0;
        let mut scale_sum: u128 = 0;
        b = 0;
        while b < blocks {
            let s = scales[b].unsigned_abs();
            let sh = if s == 0 {
                0
            } else {
                s.trailing_zeros() - e_min
            };
            shift_vecs.push(i32::try_from(sh).ok()?);
            norm = norm.saturating_add(1u64.checked_shl(sh).unwrap_or(u64::MAX));
            scale_sum = scale_sum.saturating_add(u128::from(s));
            b += 1;
        }
        // A `vpmaddwd` lane sums four selected codes of at most 127, so `508` bounds it before the
        // block shift; `4064` bounds one block's selected sum before its scale. Both are the
        // reader's whole domain check, not padding: outside them the caller keeps the exact path.
        if norm.saturating_mul(508) > u64::from(i32::MAX.cast_unsigned()) {
            return None;
        }
        if scale_sum.saturating_mul(4064) > i64::MAX as u128 {
            return None;
        }
        #[cfg(target_arch = "x86_64")]
        let vector = std::arch::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let vector = false;
        // The dequantized total is the selector over every bit set: one vector pass over the codes
        // rather than a scalar walk, and exactly the body each output row then runs. The bounds
        // above make it exact, so its own tail and shift agree with the twin.
        let ones = vec![u32::MAX; blocks];
        #[cfg(target_arch = "x86_64")]
        let sum = if vector {
            // SAFETY: `vector` names AVX2; `codes.len()` covers the terms, `ones` covers every
            // block bit, `shift_vecs` covers `blocks` (checked above), and the lane bound above
            // keeps every lane sum inside `i32` with the total inside `i64`.
            unsafe { code_select_avx2(codes, scales, &shift_vecs, e_min, &ones, codes.len())? }
        } else {
            i64::try_from(code_select_twin(codes, scales, &ones, codes.len())).ok()?
        };
        #[cfg(not(target_arch = "x86_64"))]
        let sum = i64::try_from(code_select_twin(codes, scales, &ones, codes.len())).ok()?;
        Some(Self {
            codes,
            scales,
            shift_vecs,
            e_min,
            sum,
            vector,
        })
    }

    /// The quantized selection `Σ_b S_b·Σ_{bit=1} codes[j]` of `words` over `codes[..len]`, or
    /// `None` when a caller's length is outside this reader (the caller then keeps the exact
    /// form). The sign dot is `sum − 2·selected`.
    fn selected(&self, words: &[u32], len: usize) -> Option<i64> {
        if len > self.codes.len() || words.len() < len.div_ceil(32) {
            return None;
        }
        #[cfg(target_arch = "x86_64")]
        if self.vector {
            // SAFETY: `vector` names AVX2; `len <= codes.len()` and `words` covers `len` bits
            // (checked above); `shift_vecs` and `scales` cover `len.div_ceil(32)` blocks, and
            // `CodeTerms::new` admitted only a lane sum inside `i32` and a total inside `i64`.
            return unsafe {
                code_select_avx2(
                    self.codes,
                    self.scales,
                    &self.shift_vecs,
                    self.e_min,
                    words,
                    len,
                )
            };
        }
        i64::try_from(code_select_twin(self.codes, self.scales, words, len)).ok()
    }

    /// The quantized selection of `rows` factor rows against this activation: row `r` reads
    /// `words[r·stride..]`. `true` when every row is served; `false` when the reader cannot serve
    /// `len`, which leaves the caller on its per-row entry. On `false` a prefix of `out` may hold
    /// rows already folded, so the caller must not read `out`.
    ///
    /// # Complexity
    ///
    /// `O(rows·len)` terms in the vector source, which reads this activation's codes and shift
    /// vectors once per 32-term block instead of once per row; `O(1)` space beyond `out`.
    fn selected_rows(
        &self,
        words: &[u32],
        stride: usize,
        rows: usize,
        len: usize,
        out: &mut [i64],
    ) -> bool {
        let blocks = len.div_ceil(32);
        if len > self.codes.len() || self.shift_vecs.len() < blocks || stride < blocks {
            return false;
        }
        let Some(span) = rows.checked_mul(stride) else {
            return false;
        };
        if words.len() < span || out.len() < rows {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        if self.vector {
            // SAFETY: `vector` names AVX2; the length, row-span, shift and code bounds were
            // checked above; `shift_vecs` and `scales` cover `blocks`; and `CodeTerms::new`
            // admitted only a lane sum inside `i32` and a total inside `i64`, per row.
            if unsafe {
                code_select_rows_avx2(
                    self.codes,
                    self.scales,
                    &self.shift_vecs,
                    self.e_min,
                    words,
                    stride,
                    rows,
                    len,
                    out,
                )
            }
            .is_some()
            {
                return true;
            }
        }
        code_select_rows_twin(self.codes, self.scales, words, stride, rows, len, out)
    }

    /// The same domain check as [`CodeTerms::selected_rows`] for `rows` rows written to an output
    /// of `out_len`; the two-column entry uses it for both stages before it dispatches.
    fn serves(
        &self,
        words: &[u32],
        stride: usize,
        rows: usize,
        len: usize,
        out_len: usize,
    ) -> bool {
        let blocks = len.div_ceil(32);
        let Some(span) = rows.checked_mul(stride) else {
            return false;
        };
        len <= self.codes.len()
            && self.shift_vecs.len() >= blocks
            && stride >= blocks
            && words.len() >= span
            && out_len >= rows
    }

    /// This reader as the wide column tile's per-column input.
    fn col_ref(&self) -> ColRef<'_> {
        ColRef {
            codes: self.codes,
            scales: self.scales,
            shift_vecs: &self.shift_vecs,
            e_min: self.e_min,
            vector: self.vector,
        }
    }

    /// The quantized selections of **eight** activations against one factor row's `words`, one
    /// bit expansion of each 32-term block shared by all eight columns.
    ///
    /// `true` when every column's reader served `len`; `false` otherwise, leaving `out` partially
    /// written, so the caller stays on its blocked loop and does not read it.
    ///
    /// # Complexity
    ///
    /// `O(8·len)` terms: `len/32` blocks of one word load, one expansion and `8·6` vector
    /// operations, then one horizontal fold per column. `O(1)` space beyond `out`.
    fn selected_row_cols<const N: usize>(
        cols: &[ColRef<'_>; N],
        words: &[u32],
        len: usize,
        out: &mut [i64; N],
    ) -> bool {
        let blocks = len.div_ceil(32);
        let mut c = 0usize;
        while c < N {
            let col = &cols[c];
            if len > col.codes.len()
                || col.shift_vecs.len() < blocks
                || col.scales.len() < blocks
                || words.len() < blocks
            {
                return false;
            }
            c += 1;
        }
        #[cfg(target_arch = "x86_64")]
        if cols.iter().all(|c| c.vector) {
            // SAFETY: `vector` names AVX2 on every column; the length, block and word bounds were
            // checked above, and `CodeTerms::new` admitted only a lane sum inside `i32` and a
            // total inside `i64`, per column.
            if unsafe { code_select_rowN_avx2(cols, words, len, out) }.is_some() {
                return true;
            }
        }
        false
    }

    /// The quantized selection of `rows` rows against **two** activations at once, both readers
    /// over the same weight words: the bit expansion of a word is shared, so one call costs
    /// less than two [`CodeTerms::selected_rows`] calls. Every row of each output is the
    /// single-column entry's value. `false` when either reader cannot serve the domain; on
    /// `false` a prefix of either output may hold rows already folded, so the caller must not read
    /// them.
    ///
    /// # Complexity
    ///
    /// `O(rows·len)` terms for the pair, with one bit expansion per `(row, 32-term block)`
    /// instead of one per column; `O(1)` space beyond the two outputs.
    #[allow(
        clippy::too_many_arguments,
        reason = "two columns' readers and outputs; the arguments are the pair's input"
    )]
    fn selected_rows2(
        &self,
        other: &Self,
        words: &[u32],
        stride: usize,
        rows: usize,
        len: usize,
        out_a: &mut [i64],
        out_b: &mut [i64],
    ) -> bool {
        if !self.serves(words, stride, rows, len, out_a.len())
            || !other.serves(words, stride, rows, len, out_b.len())
        {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        if self.vector && other.vector {
            // SAFETY: `vector` names AVX2; both stages passed `serves`, so each has its length,
            // row span, shift and code bounds; `CodeTerms::new` admitted only a lane sum inside
            // `i32` and a total inside `i64`, per column and per row.
            if unsafe {
                code_select_rows2_avx2(
                    self.codes,
                    self.scales,
                    &self.shift_vecs,
                    self.e_min,
                    other.codes,
                    other.scales,
                    &other.shift_vecs,
                    other.e_min,
                    words,
                    stride,
                    rows,
                    len,
                    out_a,
                    out_b,
                )
            }
            .is_some()
            {
                return true;
            }
        }
        code_select_rows_twin(self.codes, self.scales, words, stride, rows, len, out_a)
            && code_select_rows_twin(other.codes, other.scales, words, stride, rows, len, out_b)
    }
}

/// One 64-term block of the ternary activation reader: the positive and negative trit masks and
/// the block scale's exponent.
///
/// The masks are the quantiser's, not the pack's: bit `i` of `p` is `τ_i = +1`, bit `i` of `n` is
/// `τ_i = −1`, and `p & n = 0`. Bits past the block's real term count are zero, so a tail word's
/// padding bits meet only zero masks.
struct TernaryBlock {
    /// `τ_i = +1` at bit `i`.
    p: u64,
    /// `τ_i = −1` at bit `i`.
    n: u64,
    /// `k_b`: the block scale is `2^{k_b}` (`0` for an all-zero block).
    shift: u32,
}

/// One activation in the ternary block reader: the per-64-term masks of
/// `t̂_i = 2^{k_b+6}·τ_i`, the pure trit level with the code residual dropped.
///
/// The reader is exact for `t̂`; the error against the pre-scaled activation is the dropped code
/// residual, which the 0.6B PPL arm measures.
struct TernaryTerms {
    blocks: Vec<TernaryBlock>,
    len: usize,
}

impl TernaryTerms {
    /// Quantize `x` into the 64-term ternary form. `x` is the engine's pre-scaled Q32.32
    /// activation, so the block scale is chosen on the words the projection actually reads.
    ///
    /// Per block: `S = 2^k` the smallest power of two with `127·S ≥ absmax`, `c = round(x/S)` by
    /// the same ties-away rule as `quant_act`, and `τ = sign(c)` when `2|c| > 64` else `0`. The
    /// code residual `r = c − 64·τ` is dropped.
    ///
    /// # Complexity
    ///
    /// `O(n)` time and `O(n/64)` space in the activation length `n`: one absmax pass and one code
    /// pass per block.
    fn new(x: &[i64]) -> Option<Self> {
        let blocks = x.len().div_ceil(64);
        let mut out: Vec<TernaryBlock> = Vec::with_capacity(blocks);
        let mut b = 0usize;
        while b < blocks {
            let lo = b * 64;
            let take = (x.len() - lo).min(64);
            let mut absmax = 0u64;
            let mut j = lo;
            while j < lo + take {
                absmax = absmax.max(x[j].unsigned_abs());
                j += 1;
            }
            let scale = act_scale(absmax)?;
            let shift = if scale == 0 {
                0
            } else {
                scale.trailing_zeros()
            };
            let mut p = 0u64;
            let mut n = 0u64;
            j = 0;
            while j < take {
                let t = x[lo + j];
                let tau = if scale == 0 {
                    0i64
                } else {
                    let mag = u128::from(t.unsigned_abs());
                    let q = (2 * mag + u128::from(scale)) / (2 * u128::from(scale));
                    let q = i64::try_from(q).ok()?;
                    let code = if t < 0 { -q } else { q };
                    if 2 * code.unsigned_abs() > 64 {
                        code.signum()
                    } else {
                        0
                    }
                };
                if tau > 0 {
                    p |= 1u64 << j;
                } else if tau < 0 {
                    n |= 1u64 << j;
                }
                j += 1;
            }
            out.push(TernaryBlock { p, n, shift });
            b += 1;
        }
        Some(Self {
            blocks: out,
            len: x.len(),
        })
    }

    /// The ternary sign dot `Σ_i (1 − 2·bit_i(w))·t̂_i` over the first `len` terms, or `None` when
    /// `len` is not this reader's activation (the caller then keeps the exact source).
    ///
    /// Per block the stored weight word and the trit masks meet [`ternary_block`]'s `2d + c`
    /// identity, shifted once by the block exponent: `t̂_i = 2^{k_b+6}·τ_i`, so the block sum is
    /// `(2d + c) << (k_b + 6)`.
    ///
    /// # Complexity
    ///
    /// `O(len/64)` time: four popcounts, three subtractions and one shift per block. `O(1)` space.
    fn dot(&self, words: &[u32], len: usize) -> Option<i128> {
        if len != self.len || words.len() < len.div_ceil(32) {
            return None;
        }
        let mut acc: i128 = 0;
        let mut b = 0usize;
        while b < self.blocks.len() {
            let block = &self.blocks[b];
            let low = u64::from(words[2 * b]);
            let w = if 2 * b + 1 < words.len() && 64 * b + 32 < len {
                low | (u64::from(words[2 * b + 1]) << 32)
            } else {
                low
            };
            // The masks are zero past the block's real length, so the padding bits of `w`
            // contribute nothing.
            acc += i128::from(ternary_block(w, block.p, block.n)) << (block.shift + 6);
            b += 1;
        }
        Some(acc)
    }
}

/// The Q8 block selection, 32 terms per `__m256i`: the byte selector of [`sign_dot_planes_avx2`]
/// against one int8 activation plane instead of four digit planes.
///
/// A word's 32 bits become 32 `i8` selectors in five instructions (broadcast, `vpsrlvd`,
/// `vpshufb`, `vpand`, `vpshufb`), each block's `vpmaddubsw` selects the codes of its set bits,
/// `vpmaddwd` folds pairs into four-term `i32` lanes, and one `vpsllvd` by the normalized block
/// exponent plus one `vpaddd` adds the block before the row's single `e_min` shift. The block scale
/// multiply that the exact reader spends on four plane accumulators is one shift per block here.
///
/// # Complexity
///
/// `O(len)` terms: `len/32` words of eleven vector operations, plus at most 31 scalar tail terms.
/// `O(1)` space: one `i32` accumulator in registers.
///
/// # Safety
///
/// The host CPU supports AVX2; `len <= codes.len()`; `words` covers `len` bits; `shift_vecs` and
/// `scales` cover `len.div_ceil(32)` blocks; `e_min` is the smallest block exponent; every scale is
/// a non-negative power of two; and the [`CodeTerms::new`] bounds hold (`508·Σ 2^{shifts} ≤
/// i32::MAX`, `4064·Σ scales ≤ i64::MAX`), so every lane shift and the final total are exact.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "`_mm256_loadu_si256` is the unaligned load; the cast target's alignment is not a \
              requirement of the instruction"
)]
unsafe fn code_select_avx2(
    codes: &[i8],
    scales: &[i64],
    shift_vecs: &[i32],
    e_min: u32,
    words: &[u32],
    len: usize,
) -> Option<i64> {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_and_si256, _mm256_castsi256_si128,
        _mm256_cvtepi32_epi64, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32,
        _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_sllv_epi32,
        _mm256_srlv_epi32, _mm_add_epi64, _mm_cvtsi128_si64, _mm_unpackhi_epi64,
    };
    let full = len / 32;
    if words.len() < full || shift_vecs.len() < full {
        return None;
    }
    let digit_mask = _mm256_set1_epi32(0x0804_0201);
    let bit_at = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    let repeat = _mm256_setr_epi8(
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12,
        12, 12, 12,
    );
    let select_one = _mm256_setr_epi8(
        0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0,
    );
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    // The full blocks iterate as three exact-length zips: no per-block bounds check, no `Option`,
    // no scalar `try_from`, and the shift arrives as one load rather than a port-5 broadcast.
    for ((chunk, &word), shift) in codes[..full * 32]
        .as_chunks::<32>()
        .0
        .iter()
        .zip(&words[..full])
        .zip(&shift_vecs[..full])
    {
        let spread = _mm256_srlv_epi32(_mm256_set1_epi32(word.cast_signed()), bit_at);
        let nibble = _mm256_shuffle_epi8(spread, repeat);
        let select = _mm256_shuffle_epi8(select_one, _mm256_and_si256(nibble, digit_mask));
        // SAFETY: `chunk` is one 32-byte block of `codes`.
        let plane = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast::<__m256i>()) };
        let pair = _mm256_maddubs_epi16(select, plane);
        let part = _mm256_madd_epi16(pair, ones);
        let sv = _mm256_set1_epi32(*shift);
        acc = _mm256_add_epi32(acc, _mm256_sllv_epi32(part, sv));
    }
    let lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc));
    let hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(acc));
    let quad = _mm256_add_epi64(lo, hi);
    let half = _mm_add_epi64(
        _mm256_castsi256_si128(quad),
        _mm256_extracti128_si256::<1>(quad),
    );
    let half = _mm_add_epi64(half, _mm_unpackhi_epi64(half, half));
    let norm = _mm_cvtsi128_si64(half);
    let mut total = i128::from(norm) << e_min;
    let tail = full * 32;
    if tail < len {
        // The partial block starts at `tail`; its bits past `len` are padding. The full blocks'
        // scales were normalized by `e_min`, the tail keeps its own scales.
        total += code_select_twin(&codes[tail..], &scales[full..], &words[full..], len - tail);
    }
    i64::try_from(total).ok()
}

/// The largest factor-row block one column read folds: two `__m256i` accumulators per row (one
/// per activation column), so the blocked Q8 reader's register budget is twice this plus its five
/// constants.
const NQ_BLOCK_MAX: usize = 8;

/// The factor rows folded per column read when `APEIRON_NQ_BLOCK` is absent.
///
/// Eight divides every 0.6B shape (rank 736, output widths 1024/2048/3072), so no stage pays a
/// short remainder block, and it is the two-column reader's best cold arm on the box:
/// 13.56–14.23 s at 8 participants against 15.62–15.70 s at 4
/// (`bench/cpu/1795557505-nq-two-column.md`). The laptop confirmation is open;
/// `bench/cpu/1795518973-nq-blocking.md` kept 4 because the laptop's single-column block-8
/// difference was inside the block-4 arm's spread, not because 8 lost.
const NQ_BLOCK: usize = 8;

/// The activation columns one weight-word pass folds when `APEIRON_NQ_COLS` admits the wide
/// tile: eight, the register file's limit at one factor row.
const NQ_TILE_COLS: usize = 8;

/// The columns one weight-word pass folds: `APEIRON_NQ_COLS`, parsed once per process.
///
/// A value of `NQ_TILE_COLS` or more enables the eight-column tile; anything else (`0`, `2`, a
/// malformed value) keeps the two-column pair, which is also what serves a column group the tile
/// refuses. Absent, the tile is the default at four or more participants and the pair below that:
/// the tile's one-row, eight-column pass wins 1.19x at four participants and 1.22x at eight in
/// `1795569181-nq-eight-column.md`, and loses at one and two because its eight accumulators carry
/// less instruction-level parallelism than the pair's sixteen. A same-binary A/B switch, not a
/// quality dial: every arm is exact.
///
/// # Complexity
///
/// `O(1)`: one environment read, one parse and one pool-width read, cached in a `OnceLock`.
fn nq_cols() -> usize {
    static COLS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *COLS.get_or_init(|| {
        parse_cols(
            std::env::var("APEIRON_NQ_COLS").ok().as_deref(),
            Pool::global().width(),
        )
    })
}

/// [`nq_cols`]'s parse, separate so a test can cover every value and width without the process
/// cache: an explicit value at or above [`NQ_TILE_COLS`] is the tile and anything else is the
/// pair; absent, the tile needs at least four participants.
///
/// # Complexity
///
/// `O(1)`.
fn parse_cols(value: Option<&str>, width: usize) -> usize {
    match value.and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n) if n >= NQ_TILE_COLS => NQ_TILE_COLS,
        None if width >= 4 => NQ_TILE_COLS,
        Some(_) | None => 2,
    }
}

/// The factor rows folded per column read: `APEIRON_NQ_BLOCK`, parsed once per process.
///
/// `0` or `1` keeps the per-row order (one row read per dot), a value in `2..=NQ_BLOCK_MAX` folds
/// that many factor rows against one read of a column's codes and shifts, and an absent or
/// malformed value is [`NQ_BLOCK`]. Values above [`NQ_BLOCK_MAX`] clamp: the blocked reader holds
/// two vector accumulators per row (one per activation column), and it needs at least two rows to
/// share a read.
///
/// # Complexity
///
/// `O(1)`: one environment read and one parse, cached in a `OnceLock`.
fn nq_block() -> usize {
    static BLOCK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BLOCK.get_or_init(|| parse_block(std::env::var("APEIRON_NQ_BLOCK").ok().as_deref()))
}

/// [`nq_block`]'s parse, separate so a test can cover every value without the process cache: `0`
/// and `1` are the per-row loop, `2..=NQ_BLOCK_MAX` the block, malformed or absent [`NQ_BLOCK`],
/// and a value above the register budget clamps.
///
/// # Complexity
///
/// `O(1)`.
fn parse_block(value: Option<&str>) -> usize {
    value
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map_or(NQ_BLOCK, |n| if n == 0 { 0 } else { n.min(NQ_BLOCK_MAX) })
}

/// The blocked Q8 reader's safe twin: the per-row selection of [`code_select_twin`], one row at a
/// time, over `words[r·stride..]`. This is the definition the vector block implements; `false`
/// when the reader cannot serve `len` or a row's total leaves `i64`. On `false` a prefix of `out`
/// may hold rows already folded, so the caller stays on its per-row entry and does not read it.
///
/// # Complexity
///
/// `O(rows·len)` time and `O(1)` space beyond `out`.
fn code_select_rows_twin(
    codes: &[i8],
    scales: &[i64],
    words: &[u32],
    stride: usize,
    rows: usize,
    len: usize,
    out: &mut [i64],
) -> bool {
    let blocks = len.div_ceil(32);
    if len > codes.len() || scales.len() < blocks || stride < blocks || out.len() < rows {
        return false;
    }
    let Some(span) = rows.checked_mul(stride) else {
        return false;
    };
    if words.len() < span {
        return false;
    }
    for (r, slot) in out.iter_mut().take(rows).enumerate() {
        let row = &words[r * stride..];
        let Ok(selected) = i64::try_from(code_select_twin(codes, scales, row, len)) else {
            return false;
        };
        *slot = selected;
    }
    true
}

/// `rows` selections against one activation: the byte selector and the block shift are read once
/// per 32-term block and folded against every row's word before the next block. The per-row
/// arithmetic is [`code_select_avx2`]'s, block for block, so each row's total is unchanged; only
/// the code load and the shift load are shared, and the constants are built once per block.
///
/// # Complexity
///
/// `O(rows·len)` terms: `len/32` blocks of one code load, one shift load and `rows·9` vector
/// operations, then one horizontal fold per row. `O(rows)` space: `R` accumulators in registers.
///
/// # Safety
///
/// The host CPU supports AVX2; `len <= codes.len()`; every row's words cover `len` bits at
/// `words[r·stride..]`; `shift_vecs` and `scales` cover `len.div_ceil(32)` blocks; `R ≤
/// NQ_BLOCK_MAX` and `out.len() >= R`; and the [`CodeTerms::new`] bounds hold, so every lane shift
/// and each row's total are exact.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_arguments,
    clippy::cast_ptr_alignment,
    reason = "the reader and the row block are one call; splitting it would re-read the codes, and loadu admits the unaligned block pointer"
)]
unsafe fn code_select_block_avx2<const R: usize>(
    codes: &[i8],
    scales: &[i64],
    shift_vecs: &[i32],
    e_min: u32,
    words: &[u32],
    stride: usize,
    len: usize,
    out: &mut [i64],
) -> Option<()> {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_and_si256, _mm256_castsi256_si128,
        _mm256_cvtepi32_epi64, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32,
        _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_sllv_epi32,
        _mm256_srlv_epi32, _mm_add_epi64, _mm_cvtsi128_si64, _mm_unpackhi_epi64,
    };
    let full = len / 32;
    let digit_mask = _mm256_set1_epi32(0x0804_0201);
    let bit_at = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    let repeat = _mm256_setr_epi8(
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12,
        12, 12, 12,
    );
    let select_one = _mm256_setr_epi8(
        0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0,
    );
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_si256(); R];
    let mut b = 0usize;
    while b < full {
        // SAFETY: `b < full <= codes.len()/32`, so the load reads inside the codes array.
        let plane = unsafe { _mm256_loadu_si256(codes.as_ptr().add(b * 32).cast::<__m256i>()) };
        let sv = _mm256_set1_epi32(shift_vecs[b]);
        let mut r = 0usize;
        while r < R {
            let word = words[r * stride + b];
            let spread = _mm256_srlv_epi32(_mm256_set1_epi32(word.cast_signed()), bit_at);
            let nibble = _mm256_shuffle_epi8(spread, repeat);
            let select = _mm256_shuffle_epi8(select_one, _mm256_and_si256(nibble, digit_mask));
            let pair = _mm256_maddubs_epi16(select, plane);
            let part = _mm256_madd_epi16(pair, ones);
            acc[r] = _mm256_add_epi32(acc[r], _mm256_sllv_epi32(part, sv));
            r += 1;
        }
        b += 1;
    }
    let tail = full * 32;
    let mut r = 0usize;
    while r < R {
        let lane = acc[r];
        let lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(lane));
        let hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(lane));
        let quad = _mm256_add_epi64(lo, hi);
        let half = _mm_add_epi64(
            _mm256_castsi256_si128(quad),
            _mm256_extracti128_si256::<1>(quad),
        );
        let half = _mm_add_epi64(half, _mm_unpackhi_epi64(half, half));
        let mut total = i128::from(_mm_cvtsi128_si64(half)) << e_min;
        if tail < len {
            // The partial block starts at `tail`, a multiple of 32; its bits past `len` are
            // padding, and the full blocks were folded above.
            total += code_select_twin(
                &codes[tail..],
                &scales[full..],
                &words[r * stride + full..],
                len - tail,
            );
        }
        out[r] = i64::try_from(total).ok()?;
        r += 1;
    }
    Some(())
}

/// Dispatch the blocked Q8 reader over a row count this monomorphises for: 2 to
/// [`NQ_BLOCK_MAX`]. Any other count returns `None` and the caller keeps the per-row entry.
///
/// # Safety
///
/// As [`code_select_block_avx2`]: the host supports AVX2 and the reader, `words`, `stride`, `len`
/// and `out` cover `rows` rows of `len` terms.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_arguments,
    reason = "one dispatch over the reader's arguments; see `code_select_block_avx2`"
)]
unsafe fn code_select_rows_avx2(
    codes: &[i8],
    scales: &[i64],
    shift_vecs: &[i32],
    e_min: u32,
    words: &[u32],
    stride: usize,
    rows: usize,
    len: usize,
    out: &mut [i64],
) -> Option<()> {
    macro_rules! dispatch {
        ($($r:literal),*) => {
            match rows {
                $($r => {
                    // SAFETY: the caller's contract is `rows` rows; the arm fixes `R = rows`.
                    unsafe {
                        code_select_block_avx2::<$r>(
                            codes, scales, shift_vecs, e_min, words, stride, len, out,
                        )
                    }
                })*
                _ => None,
            }
        };
    }
    dispatch!(2, 3, 4, 5, 6, 7, 8)
}

/// The horizontal sum of one `i32` lane accumulator, widened to `i64` without an `i32` wrap.
///
/// The [`CodeTerms::new`] lane bound makes every lane's sum fit `i32`, so the eight widened
/// values sum to at most `8·i32::MAX < i64::MAX` in absolute value and the widening is exact.
///
/// # Safety
///
/// The host CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lane_sum(lane: std::arch::x86_64::__m256i) -> i64 {
    use std::arch::x86_64::{
        _mm256_add_epi64, _mm256_castsi256_si128, _mm256_cvtepi32_epi64, _mm256_extracti128_si256,
        _mm_add_epi64, _mm_cvtsi128_si64, _mm_unpackhi_epi64,
    };
    let lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(lane));
    let hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(lane));
    let quad = _mm256_add_epi64(lo, hi);
    let half = _mm_add_epi64(
        _mm256_castsi256_si128(quad),
        _mm256_extracti128_si256::<1>(quad),
    );
    let half = _mm_add_epi64(half, _mm_unpackhi_epi64(half, half));
    _mm_cvtsi128_si64(half)
}

/// Fold one factor row's weight words against **eight** activation columns: the word's bit
/// expansion (broadcast, `vpsrlvd`, two `vpshufb`, `vpand`) is built once per 32-term block and
/// folded against all eight columns, and only the `vpmaddubsw`/`vpmaddwd`/`vpsllvd`/`vpaddd`
/// chains repeat per column.
///
/// This is the same arithmetic as eight [`code_select_block_avx2`] calls: each column's codes,
/// shift vectors and `e_min` are read independently, and [`CodeTerms::new`]'s per-column bounds
/// make every lane sum exact, so each column's output equals the single-column kernel's word for
/// word. The row count is one by construction: eight column accumulators plus the five selector
/// constants and the two per-column loads fill the vector register file, and a second row spills.
///
/// # Safety
///
/// The host CPU supports AVX2. For each column: `len <= codes.len()`, `scales.len() >=
/// len.div_ceil(32)`, `shift_vecs` covers `len.div_ceil(32)` blocks, `e_min` is the smallest
/// block exponent, `words.len() >= len.div_ceil(32)`, and the [`CodeTerms::new`] bounds hold.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "loadu admits the unaligned block pointer"
)]
unsafe fn code_select_rowN_avx2<const N: usize>(
    cols: &[ColRef<'_>; N],
    words: &[u32],
    len: usize,
    out: &mut [i64; N],
) -> Option<()> {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32,
        _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_sllv_epi32,
        _mm256_srlv_epi32,
    };
    let full = len / 32;
    let digit_mask = _mm256_set1_epi32(0x0804_0201);
    let bit_at = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    let repeat = _mm256_setr_epi8(
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12,
        12, 12, 12,
    );
    let select_one = _mm256_setr_epi8(
        0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0,
    );
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_si256(); N];
    let mut b = 0usize;
    while b < full {
        let word = words[b];
        let spread = _mm256_srlv_epi32(_mm256_set1_epi32(word.cast_signed()), bit_at);
        let nibble = _mm256_shuffle_epi8(spread, repeat);
        let select = _mm256_shuffle_epi8(select_one, _mm256_and_si256(nibble, digit_mask));
        let mut c = 0usize;
        while c < N {
            // SAFETY: the caller's per-column bounds cover both planes at block `b`.
            let plane =
                unsafe { _mm256_loadu_si256(cols[c].codes.as_ptr().add(b * 32).cast::<__m256i>()) };
            let sv = _mm256_set1_epi32(cols[c].shift_vecs[b]);
            let pair = _mm256_maddubs_epi16(select, plane);
            let part = _mm256_madd_epi16(pair, ones);
            acc[c] = _mm256_add_epi32(acc[c], _mm256_sllv_epi32(part, sv));
            c += 1;
        }
        b += 1;
    }
    let tail = full * 32;
    let mut c = 0usize;
    while c < N {
        // SAFETY: AVX2 is the caller's contract; the lane bound makes the sum exact.
        let mut total = i128::from(unsafe { lane_sum(acc[c]) }) << cols[c].e_min;
        if tail < len {
            total += code_select_twin(
                &cols[c].codes[tail..],
                &cols[c].scales[full..],
                &words[full..],
                len - tail,
            );
        }
        out[c] = i64::try_from(total).ok()?;
        c += 1;
    }
    Some(())
}

/// Fold one row block's weight words against **two** activation columns: the word's bit
/// expansion (broadcast, `vpsrlvd`, two `vpshufb`, `vpand`) is shared, and only the two
/// `vpmaddubsw`/`vpmaddwd`/`vpsllvd`/`vpaddd` chains are per column.
///
/// This is the same arithmetic as two [`code_select_block_avx2`] calls: each column's codes,
/// shift vectors and `e_min` are read independently, and [`CodeTerms::new`]'s per-column bounds
/// make both lane sums exact. Each column's result therefore equals the single-column kernel's
/// word for word.
///
/// # Safety
///
/// The host CPU supports AVX2. For each column: `len <= codes.len()`, `scales.len() >=
/// len.div_ceil(32)`, `shift_vecs` covers `len.div_ceil(32)` blocks, `e_min` is the smallest
/// block exponent, `words` covers `rows` rows of `len` terms at `stride >= len.div_ceil(32)`,
/// `out_a.len() >= R` and `out_b.len() >= R`, and the [`CodeTerms::new`] bounds hold.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_arguments,
    clippy::cast_ptr_alignment,
    reason = "two columns' readers and their outputs; the arguments are the pair's input, and loadu admits the unaligned block pointer"
)]
unsafe fn code_select_block2_avx2<const R: usize>(
    codes_a: &[i8],
    scales_a: &[i64],
    shift_a: &[i32],
    e_min_a: u32,
    codes_b: &[i8],
    scales_b: &[i64],
    shift_b: &[i32],
    e_min_b: u32,
    words: &[u32],
    stride: usize,
    len: usize,
    out_a: &mut [i64],
    out_b: &mut [i64],
) -> Option<()> {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32,
        _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_sllv_epi32,
        _mm256_srlv_epi32,
    };
    let full = len / 32;
    let digit_mask = _mm256_set1_epi32(0x0804_0201);
    let bit_at = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    let repeat = _mm256_setr_epi8(
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12,
        12, 12, 12,
    );
    let select_one = _mm256_setr_epi8(
        0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0,
    );
    let ones = _mm256_set1_epi16(1);
    let mut acc_a = [_mm256_setzero_si256(); R];
    let mut acc_b = [_mm256_setzero_si256(); R];
    let mut b = 0usize;
    while b < full {
        // SAFETY: `b < full` and the caller's per-column bounds cover both codes arrays at this
        // block.
        let plane_a = unsafe { _mm256_loadu_si256(codes_a.as_ptr().add(b * 32).cast::<__m256i>()) };
        let plane_b = unsafe { _mm256_loadu_si256(codes_b.as_ptr().add(b * 32).cast::<__m256i>()) };
        let sv_a = _mm256_set1_epi32(shift_a[b]);
        let sv_b = _mm256_set1_epi32(shift_b[b]);
        let mut r = 0usize;
        while r < R {
            let word = words[r * stride + b];
            let spread = _mm256_srlv_epi32(_mm256_set1_epi32(word.cast_signed()), bit_at);
            let nibble = _mm256_shuffle_epi8(spread, repeat);
            let select = _mm256_shuffle_epi8(select_one, _mm256_and_si256(nibble, digit_mask));
            let pair = _mm256_maddubs_epi16(select, plane_a);
            let part = _mm256_madd_epi16(pair, ones);
            acc_a[r] = _mm256_add_epi32(acc_a[r], _mm256_sllv_epi32(part, sv_a));
            let pair = _mm256_maddubs_epi16(select, plane_b);
            let part = _mm256_madd_epi16(pair, ones);
            acc_b[r] = _mm256_add_epi32(acc_b[r], _mm256_sllv_epi32(part, sv_b));
            r += 1;
        }
        b += 1;
    }
    let tail = full * 32;
    let mut r = 0usize;
    while r < R {
        // SAFETY: AVX2 is the caller's contract; the lane bound makes each sum exact.
        let mut total_a = i128::from(unsafe { lane_sum(acc_a[r]) }) << e_min_a;
        let mut total_b = i128::from(unsafe { lane_sum(acc_b[r]) }) << e_min_b;
        if tail < len {
            total_a += code_select_twin(
                &codes_a[tail..],
                &scales_a[full..],
                &words[r * stride + full..],
                len - tail,
            );
            total_b += code_select_twin(
                &codes_b[tail..],
                &scales_b[full..],
                &words[r * stride + full..],
                len - tail,
            );
        }
        out_a[r] = i64::try_from(total_a).ok()?;
        out_b[r] = i64::try_from(total_b).ok()?;
        r += 1;
    }
    Some(())
}

/// Dispatch the two-column blocked reader over a row count this monomorphises for: 2 to
/// [`NQ_BLOCK_MAX`]. Any other count returns `None` and the caller keeps the per-column entry.
///
/// # Safety
///
/// As [`code_select_block2_avx2`]: the host supports AVX2 and both readers, `words`, `stride`,
/// `len` and the two outputs cover `rows` rows of `len` terms.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    clippy::too_many_arguments,
    reason = "two columns' readers and their outputs; see `code_select_block2_avx2`"
)]
unsafe fn code_select_rows2_avx2(
    codes_a: &[i8],
    scales_a: &[i64],
    shift_a: &[i32],
    e_min_a: u32,
    codes_b: &[i8],
    scales_b: &[i64],
    shift_b: &[i32],
    e_min_b: u32,
    words: &[u32],
    stride: usize,
    rows: usize,
    len: usize,
    out_a: &mut [i64],
    out_b: &mut [i64],
) -> Option<()> {
    macro_rules! dispatch {
        ($($r:literal),*) => {
            match rows {
                $($r => {
                    // SAFETY: the caller's contract is `rows` rows; the arm fixes `R = rows`.
                    unsafe {
                        code_select_block2_avx2::<$r>(
                            codes_a, scales_a, shift_a, e_min_a, codes_b, scales_b, shift_b,
                            e_min_b, words, stride, len, out_a, out_b,
                        )
                    }
                })*
                _ => None,
            }
        };
    }
    dispatch!(2, 3, 4, 5, 6, 7, 8)
}

/// One stage's activation in the forms [`NqLinear::project_with`] reads: the exact Q32.32 words,
/// their four-plane reader when the narrow bound holds, and the Q8 or ternary block reader when
/// the mode and its bounds admit it.
struct Stage<'a> {
    /// The exact activation, the wide `i128` source's input.
    wide: &'a [i64],
    /// The four-plane source, present when the narrow bound holds, a block reader is absent and a
    /// residual layer reads this stage (the only reader besides the block readers).
    terms: Option<Terms<'a>>,
    /// The Q8 reader, present under [`Reader::Q8`] when its bounds hold.
    codes: Option<CodeTerms<'a>>,
    /// The ternary reader, present under [`Reader::Ternary`] when its quantizer admits the
    /// activation.
    ternary: Option<TernaryTerms>,
}

impl Stage<'_> {
    /// The stage's sign dot over `words`: the ternary reader when present and in range, else the
    /// Q8 reader when present and in range, else the exact plane source, else the wide scalar
    /// source. Each fast form is exact over its own domain: the Q8 form is the quantized sign dot
    /// `Σ x̂ − 2·selected`, the ternary form the trit dot of [`TernaryTerms::dot`].
    fn dot(&self, words: &[u32], len: usize) -> i128 {
        if let Some(ternary) = &self.ternary {
            if let Some(dot) = ternary.dot(words, len) {
                return dot;
            }
        }
        if let Some(codes) = &self.codes {
            if let Some(selected) = codes.selected(words, len) {
                return i128::from(codes.sum) - 2 * i128::from(selected);
            }
        }
        if let Some(terms) = &self.terms {
            i128::from(terms.dot(words, len))
        } else {
            sign_dot!(self.wide, words, len, i128)
        }
    }

    /// The sign dots of `rows` factor rows against this stage, row `r` reading
    /// `words[r·stride..]`; the Q8 reader folds the whole row block against one read of its codes,
    /// and every other reader takes [`Stage::dot`] row by row, which is the same arithmetic in the
    /// same order.
    ///
    /// # Complexity
    ///
    /// `O(rows·len)` terms. The blocked Q8 source reads `len/32` code and shift words per block
    /// rather than per row; every other reader reads `rows·len/32` words, as its per-row entry
    /// does. `O(rows)` space.
    fn dot_rows(&self, words: &[u32], stride: usize, rows: usize, len: usize, out: &mut [i128]) {
        if rows > 1 && self.ternary.is_none() {
            if let Some(codes) = &self.codes {
                let mut selected = [0i64; NQ_BLOCK_MAX];
                if codes.selected_rows(words, stride, rows, len, &mut selected[..rows]) {
                    for (slot, &sel) in out.iter_mut().take(rows).zip(&selected[..rows]) {
                        *slot = i128::from(codes.sum) - 2 * i128::from(sel);
                    }
                    return;
                }
            }
        }
        for r in 0..rows {
            out[r] = self.dot(&words[r * stride..(r + 1) * stride], len);
        }
    }

    /// The sign dots of `rows` rows against **two** stages at once: the shared weight read and
    /// bit expansion of [`CodeTerms::selected_rows2`]. `true` when both stages are Q8 readers
    /// that served every row; `false` when the caller must take [`Stage::dot_rows`] per stage,
    /// which is the same arithmetic.
    ///
    /// # Complexity
    ///
    /// `O(rows·len)` terms for the pair; `O(1)` space beyond the two outputs.
    #[allow(
        clippy::too_many_arguments,
        reason = "two stages' readers and outputs; the arguments are the pair's input"
    )]
    fn dot_rows2(
        &self,
        other: &Self,
        words: &[u32],
        stride: usize,
        rows: usize,
        len: usize,
        out_a: &mut [i128],
        out_b: &mut [i128],
    ) -> bool {
        if !(2..=NQ_BLOCK_MAX).contains(&rows)
            || out_a.len() < rows
            || out_b.len() < rows
            || self.ternary.is_some()
            || other.ternary.is_some()
        {
            return false;
        }
        let (Some(a), Some(b)) = (self.codes.as_ref(), other.codes.as_ref()) else {
            return false;
        };
        let mut sel_a = [0i64; NQ_BLOCK_MAX];
        let mut sel_b = [0i64; NQ_BLOCK_MAX];
        if !a.selected_rows2(
            b,
            words,
            stride,
            rows,
            len,
            &mut sel_a[..rows],
            &mut sel_b[..rows],
        ) {
            return false;
        }
        for i in 0..rows {
            out_a[i] = i128::from(a.sum) - 2 * i128::from(sel_a[i]);
            out_b[i] = i128::from(b.sum) - 2 * i128::from(sel_b[i]);
        }
        true
    }

    /// Assemble the wide tile's eight readers once per column group: the stages, the views and
    /// the dequantized totals are group properties, and rebuilding them for every output row was
    /// 10.3% of the `apeiron-q8act` worker's instruction stream on the box at width 8, block 256
    /// (`bench/cpu/1795573862-prefill-block-and-group.md`). `None` when a stage is not a Q8
    /// reader; the caller then keeps its pair loop, which is the same arithmetic. The assembly
    /// copies views; the reader's length and block bounds are [`CodeTerms::selected_row_cols`]'s
    /// admission, re-checked on every [`StageCols::dot_row`] call.
    ///
    /// # Complexity
    ///
    /// `O(1)`: eight view reads and no work in `len`.
    fn cols<'s, const N: usize>(stages: &[&'s Self; N]) -> Option<StageCols<'s, N>> {
        let mut cols = [ColRef {
            codes: &[],
            scales: &[],
            shift_vecs: &[],
            e_min: 0,
            vector: false,
        }; N];
        let mut sums = [0i64; N];
        let mut c = 0usize;
        while c < N {
            let Some(codes) = &stages[c].codes else {
                return None;
            };
            cols[c] = codes.col_ref();
            sums[c] = codes.sum;
            c += 1;
        }
        Some(StageCols { cols, sums })
    }
}

/// `N` adjacent activation readers of a column group, assembled once for all of a row band's
/// output rows; see [`Stage::cols`].
struct StageCols<'a, const N: usize> {
    cols: [ColRef<'a>; N],
    sums: [i64; N],
}

impl<const N: usize> StageCols<'_, N> {
    /// The sign dots of one factor row's `words` against the group's `N` columns, one bit
    /// expansion of each 32-term block shared by all of them. `false` leaves `out` partially
    /// written, and the caller keeps its two-column or per-row loop, which is the same
    /// arithmetic.
    ///
    /// # Complexity
    ///
    /// `O(N·len)` terms with one word expansion per 32 terms; `O(1)` space beyond `out`.
    fn dot_row(&self, words: &[u32], len: usize, out: &mut [i128; N]) -> bool {
        let mut sel = [0i64; N];
        if !CodeTerms::selected_row_cols(&self.cols, words, len, &mut sel) {
            return false;
        }
        let mut c = 0usize;
        while c < N {
            out[c] = i128::from(self.sums[c]) - 2 * i128::from(sel[c]);
            c += 1;
        }
        true
    }
}

/// The activation reader a projection runs: the exact Q32.32 words, the Q8 block codes, or the
/// ternary block masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reader {
    /// The exact words through [`Terms`].
    Exact,
    /// Int8 block codes through [`CodeTerms`].
    Q8,
    /// Ternary trit masks through [`TernaryTerms`].
    Ternary,
}

/// One fitted linear in the form the kernel executes: bits, Q32.32 scales, the sparse residual and
/// the dense sign-residual layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NqLinear {
    out: usize,
    inn: usize,
    rank: usize,
    u_words: usize,
    v_words: usize,
    k: usize,
    u: Plane<u32>,
    v: Plane<u32>,
    s_pre: Box<[i64]>,
    s_mid: Box<[i64]>,
    s_post: Box<[i64]>,
    salient_idx: Box<[u32]>,
    salient: Box<[i64]>,
    res_layers: usize,
    /// The packed section stored no shared block; its `v`, `s_pre`, `s_mid` and `salient_idx`
    /// point at the layer's `q` section in the same mapping.
    shared: bool,
    res_words: usize,
    res: Plane<u32>,
    s_res: Box<[i64]>,
}

impl NqLinear {
    /// Decode a packed linear: every scale word through [`bf16_to_fixed`], every sign word copied.
    ///
    /// # Errors
    ///
    /// [`NqError::Packed`] for a field whose length disagrees with its dimensions, and
    /// [`NqError::Scale`] for a bfloat16 scale that is not finite or does not fit Q32.32.
    pub fn from_packed(p: PackedLinear) -> Result<Self, NqError> {
        let shape = checked_shape(&p)?;
        let mut s_pre = Vec::with_capacity(p.inn);
        for &bits in p.s_pre.iter() {
            s_pre.push(bf16_to_fixed(bits).map_err(|_| NqError::Scale { bits })?);
        }
        let mut s_mid = Vec::with_capacity(p.rank);
        for &bits in p.s_mid.iter() {
            s_mid.push(bf16_to_fixed(bits).map_err(|_| NqError::Scale { bits })?);
        }
        let mut s_post = Vec::with_capacity(p.out);
        for &bits in p.s_post.iter() {
            s_post.push(bf16_to_fixed(bits).map_err(|_| NqError::Scale { bits })?);
        }
        let mut salient = Vec::with_capacity(shape.salient_len);
        for &bits in p.salient.iter() {
            salient.push(f16_to_fixed(bits).map_err(|_| NqError::Scale { bits })?);
        }
        let mut s_res = Vec::with_capacity(p.s_res.len());
        for &bits in p.s_res.iter() {
            s_res.push(bf16_to_fixed(bits).map_err(|_| NqError::Scale { bits })?);
        }
        let mut seen = vec![false; p.inn];
        for &j in p.salient_idx.iter() {
            let j = j as usize;
            if j >= p.inn || seen[j] {
                return Err(NqError::SalientIndex {
                    index: j,
                    inn: p.inn,
                });
            }
            seen[j] = true;
        }
        Ok(Self {
            out: p.out,
            inn: p.inn,
            rank: p.rank,
            u_words: shape.u_words,
            v_words: shape.v_words,
            k: shape.k,
            u: p.u,
            v: p.v,
            s_pre: s_pre.into_boxed_slice(),
            s_mid: s_mid.into_boxed_slice(),
            s_post: s_post.into_boxed_slice(),
            salient_idx: p.salient_idx.to_vec().into_boxed_slice(),
            salient: salient.into_boxed_slice(),
            res_layers: shape.res_layers,
            shared: p.shared,
            res_words: shape.res_words,
            res: p.res,
            s_res: s_res.into_boxed_slice(),
        })
    }

    /// Output rows.
    #[must_use]
    pub fn out(&self) -> usize {
        self.out
    }

    /// Input columns.
    #[must_use]
    pub fn inn(&self) -> usize {
        self.inn
    }

    /// The dense weights the linear replaces.
    #[must_use]
    pub fn weights(&self) -> u64 {
        (self.out as u64).saturating_mul(self.inn as u64)
    }

    /// The stored bits: the two sign planes, the three scale vectors, the index list, the sparse
    /// residual and every dense sign-residual layer, omitting the block a shared member takes from
    /// its layer's `q` section.
    #[must_use]
    pub fn stored_bits(&self) -> u64 {
        let (v, pre, mid, idx) = if self.shared {
            (0, 0, 0, 0)
        } else {
            (
                self.v.len(),
                self.s_pre.len(),
                self.s_mid.len(),
                self.salient_idx.len(),
            )
        };
        let signs = self
            .u
            .len()
            .saturating_add(v)
            .saturating_add(self.res.len());
        let scales = pre
            .saturating_add(mid)
            .saturating_add(self.s_post.len())
            .saturating_add(self.salient.len())
            .saturating_add(self.s_res.len());
        (signs as u64)
            .saturating_mul(32)
            .saturating_add((idx as u64).saturating_mul(32))
            .saturating_add((scales as u64).saturating_mul(16))
    }

    /// The exact Q32.32 projection over the exact activation words: the reference reader.
    ///
    /// `round(s_post_i · Σ_k U_ik · z_k / 2^32)` plus `Σ_j salient_ij·x_idx[j]` and
    /// `Σ_l s_res_li·Σ_j sign_lij·x_j`, for every output row, with
    /// `z = s_mid ⊙ (V (s_pre ⊙ x))`.
    ///
    /// # Errors
    ///
    /// [`NqError::Shape`] for an activation of the wrong width and [`NqError::Overflow`] when a
    /// pre-scaled activation, a mid-scaled rank, a rank sum, a dense residual or a post-scaled
    /// output leaves `i64`. No value is clamped.
    ///
    /// # Complexity
    ///
    /// `O(rank·inn + out·rank + out·k + res_layers·out·inn)` additions and products; scratch
    /// `O(inn + rank)` words and `O(out)` output.
    pub fn project(&self, x: &[i64]) -> Result<Vec<i64>, NqError> {
        self.project_with(x, Reader::Exact)
    }

    /// The same projection with the Q8 block reader for the `V`, `U` and dense-residual
    /// activations: each stage activation is quantized to one int8 code per element and one
    /// power-of-two scale per 32 elements, and every sign row folds those codes through
    /// [`code_select_avx2`], the block shape `matvec_q8act` reads for a stored quantized row.
    /// The result is the exact integer projection of the quantized activation, not of the exact
    /// Q32.32 words: this is the fast reader and [`NqLinear::project`] is its oracle. A pack with
    /// no residual layer reads `x` only through the sparse correction, so its `x` stage is not
    /// quantized at all.
    ///
    /// # Errors
    ///
    /// As [`NqLinear::project`]. A stage whose activation cannot be quantized, or whose codes
    /// leave [`CodeTerms`]' domain, keeps the exact reader for that stage; an actual fixed-point
    /// overflow still refuses.
    ///
    /// # Complexity
    ///
    /// As [`NqLinear::project`] plus `O(inn + rank)` quantization work, and `O(inn)` less when the
    /// pack has no residual layer.
    pub fn project_q8(&self, x: &[i64]) -> Result<Vec<i64>, NqError> {
        self.project_with(x, Reader::Q8)
    }

    /// The same projection with the ternary block reader for the `V`, `U` and dense-residual
    /// activations: each stage activation is quantized to one trit per term and one power-of-two
    /// scale per 64 terms, the code residual is dropped, and every sign row folds the trit masks
    /// with the [`ternary_block`] identity. The result is the exact integer projection of the
    /// ternary-quantized activation, not of the exact Q32.32 words: this is the fast carrier and
    /// [`NqLinear::project`] is its oracle.
    ///
    /// # Errors
    ///
    /// As [`NqLinear::project`]. A stage whose activation cannot be quantized keeps the exact
    /// reader for that stage; an actual fixed-point overflow still refuses.
    ///
    /// # Complexity
    ///
    /// As [`NqLinear::project`] plus `O(inn + rank)` quantization work.
    pub fn project_ternary(&self, x: &[i64]) -> Result<Vec<i64>, NqError> {
        self.project_with(x, Reader::Ternary)
    }

    fn project_with(&self, x: &[i64], reader: Reader) -> Result<Vec<i64>, NqError> {
        if x.len() != self.inn {
            return Err(NqError::Shape {
                x: x.len(),
                n_in: self.inn,
            });
        }
        // t_j = round(s_pre_j · x_j / 2^32).
        let mut t: Vec<i64> = Vec::with_capacity(self.inn);
        let mut max_abs_t = 0u64;
        let mut j = 0usize;
        while j < self.inn {
            match mul_q32(x[j], self.s_pre[j]) {
                Some(v) => {
                    max_abs_t = max_abs_t.max(v.unsigned_abs());
                    t.push(v);
                }
                None => return Err(NqError::Overflow),
            }
            j += 1;
        }
        // Every `V` row sums `inn` terms of at most `max_abs_t`, so `inn·max_abs_t` bounds it.
        // When the bound fits `i64`, every partial sum does too and the narrow source is exact. The
        // product saturates because the factors can leave `u128`; a saturated bound is simply not
        // narrow and takes the wide source.
        let bound_v = u128::from(max_abs_t).saturating_mul(self.inn as u128);
        let narrow_v = bound_v <= i64::MAX as u128;
        let q8 = reader == Reader::Q8;
        let ternary = reader == Reader::Ternary;
        let t_ternary = if ternary { TernaryTerms::new(&t) } else { None };
        let t_q = if q8 { quant_act(&t) } else { None };
        let t_codes = t_q.as_ref().and_then(CodeTerms::new);
        let t_stage = Stage {
            wide: &t,
            terms: (t_codes.is_none() && t_ternary.is_none() && narrow_v).then(|| Terms::new(&t)),
            codes: t_codes,
            ternary: t_ternary,
        };
        let pool = Pool::global();
        let width = pool.width().max(1);
        let mut z: Vec<i64> = vec![0; self.rank];
        let z_chunk = self.rank.div_ceil(width.saturating_mul(4)).max(1);
        let overflow = std::sync::atomic::AtomicBool::new(false);
        pool.chunks_mut(&mut z, z_chunk, &|_, at, zs| {
            let mut d = 0usize;
            while d < zs.len() {
                let k = at + d;
                let vrow = &self.v[k * self.v_words..(k + 1) * self.v_words];
                let Some(dot) = i64::try_from(t_stage.dot(vrow, self.inn)).ok() else {
                    overflow.store(true, std::sync::atomic::Ordering::Relaxed);
                    d += 1;
                    continue;
                };
                match mul_q32(dot, self.s_mid[k]) {
                    Some(v) => zs[d] = v,
                    None => overflow.store(true, std::sync::atomic::Ordering::Relaxed),
                }
                d += 1;
            }
        });
        if overflow.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(NqError::Overflow);
        }
        // The mid scale changes the bound: every `U` row sums `rank` terms of at most
        // `max_abs_z`, so `rank·max_abs_z` bounds it. The serial scan is `O(rank)` against the
        // `O(out·rank)` product that follows.
        let mut max_abs_z = 0u64;
        for &v in &z {
            max_abs_z = max_abs_z.max(v.unsigned_abs());
        }
        let bound_u = u128::from(max_abs_z).saturating_mul(self.rank as u128);
        let narrow_u = bound_u <= i64::MAX as u128;
        let z_ternary = if ternary { TernaryTerms::new(&z) } else { None };
        let z_q = if q8 { quant_act(&z) } else { None };
        let z_codes = z_q.as_ref().and_then(CodeTerms::new);
        let z_stage = Stage {
            wide: &z,
            terms: (z_codes.is_none() && z_ternary.is_none() && narrow_u).then(|| Terms::new(&z)),
            codes: z_codes,
            ternary: z_ternary,
        };
        // The dense residual is the only reader of the `x` stage's codes and planes, and the
        // sparse correction indexes `x` directly. A pack with no residual layer therefore skips
        // the whole `x` setup; its stage keeps the exact words as the fallback source.
        let residual = self.res_layers > 0;
        let max_abs_x = if residual {
            x.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0)
        } else {
            0
        };
        // The dense sign dot sums `inn` activation words; it is narrow exactly when that bound is.
        let bound_x = u128::from(max_abs_x).saturating_mul(self.inn as u128);
        let narrow_x = residual && bound_x <= i64::MAX as u128;
        let x_ternary = if ternary && residual {
            TernaryTerms::new(x)
        } else {
            None
        };
        let x_q = if q8 && residual { quant_act(x) } else { None };
        let x_codes = x_q.as_ref().and_then(CodeTerms::new);
        let x_stage = Stage {
            wide: x,
            terms: (x_codes.is_none() && x_ternary.is_none() && narrow_x).then(|| Terms::new(x)),
            codes: x_codes,
            ternary: x_ternary,
        };
        let y = self.project_rows(pool, &z_stage, &x_stage, &overflow);
        if overflow.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(NqError::Overflow);
        }
        Ok(y)
    }

    /// The exact product of every column in `xs`: [`NqLinear::project`] on each column, with one
    /// pooled `V` pass and one pooled `U` pass instead of one dispatch pair per column.
    ///
    /// The arithmetic is the single-column kernel's: each column keeps its own pre-scaled words,
    /// reader and accumulation order, and every output equals the per-column call bit for bit.
    /// The pooled pass reads each `V` or `U` row once and folds it against every column its task
    /// owns, so the pack's factor words are read once per column group. A block of
    /// [`nq_block`] factor rows is folded against each column's codes before the next block, so the
    /// column's selector planes are read once per block rather than once per row; the integer
    /// dot is exact under either nesting and the per-row epilogues keep their order.
    ///
    /// # Errors
    ///
    /// As [`NqLinear::project`], for every column. No column is written when any refuses. The group's
    /// widths are checked before the arithmetic, so a group carrying both a wrong width and an
    /// overflow reports the width; the per-column entry would report the earlier column's overflow
    /// first. `proj_cols` never presents columns of differing widths.
    ///
    /// # Complexity
    ///
    /// `O(k·(rank·inn + out·rank + res_layers·out·inn))` additions and products; scratch
    /// `O(k·(inn + rank + out))` words. Two pool dispatches per call.
    pub fn project_pooled(&self, xs: &[&[i64]]) -> Result<Vec<Vec<i64>>, NqError> {
        self.project_pooled_with(xs, Reader::Exact)
    }

    /// The Q8 reader over every column in `xs`: [`NqLinear::project_q8`] on each column, pooled
    /// as [`NqLinear::project_pooled`] pools the exact reader.
    ///
    /// # Errors
    ///
    /// As [`NqLinear::project_q8`], for every column.
    ///
    /// # Complexity
    ///
    /// As [`NqLinear::project_pooled`] plus `O(k·(inn + rank))` quantization work.
    pub fn project_q8_pooled(&self, xs: &[&[i64]]) -> Result<Vec<Vec<i64>>, NqError> {
        self.project_pooled_with(xs, Reader::Q8)
    }

    /// The ternary reader over every column in `xs`: [`NqLinear::project_ternary`] on each
    /// column, pooled as [`NqLinear::project_pooled`] pools the exact reader.
    ///
    /// # Errors
    ///
    /// As [`NqLinear::project_ternary`], for every column.
    ///
    /// # Complexity
    ///
    /// As [`NqLinear::project_pooled`] plus `O(k·(inn + rank))` quantization work.
    pub fn project_ternary_pooled(&self, xs: &[&[i64]]) -> Result<Vec<Vec<i64>>, NqError> {
        self.project_pooled_with(xs, Reader::Ternary)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the two stages, their per-column readers and the refusal flag are one protocol"
    )]
    fn project_pooled_with(&self, xs: &[&[i64]], reader: Reader) -> Result<Vec<Vec<i64>>, NqError> {
        self.project_pooled_block(xs, reader, nq_block())
    }

    /// [`NqLinear::project_pooled_with`] under an explicit factor-row block: module-private so the
    /// tests can step every block size in one process, while production takes `APEIRON_NQ_BLOCK`.
    ///
    /// The block selects the loop nesting only. Each column's reader is read once per block of
    /// factor rows instead of once per row, and every per-(row, column) dot, `mul_q32` epilogue and
    /// output-row sum keeps its own order. A block of `0` or `1` is the per-row loop.
    fn project_pooled_block(
        &self,
        xs: &[&[i64]],
        reader: Reader,
        block: usize,
    ) -> Result<Vec<Vec<i64>>, NqError> {
        self.project_pooled_shape(xs, reader, block, nq_cols())
    }

    /// [`NqLinear::project_pooled_block`] under an explicit column tile: `tile` at or above
    /// [`NQ_TILE_COLS`] runs the eight-column wide pass on whole groups of eight columns, and
    /// every remaining column keeps the block's two-column or per-row loop. The tile is a loop
    /// nesting, not an arithmetic: both arms compute each column's dot exactly, so their words
    /// are equal, as `the_pooled_block_is_bit_equal_at_every_block` asserts.
    #[allow(
        clippy::too_many_lines,
        reason = "the two stages, their per-column readers and the refusal flag are one protocol"
    )]
    fn project_pooled_shape(
        &self,
        xs: &[&[i64]],
        reader: Reader,
        block: usize,
        tile: usize,
    ) -> Result<Vec<Vec<i64>>, NqError> {
        let block = block.min(NQ_BLOCK_MAX);
        let wide = tile >= NQ_TILE_COLS;
        let k = xs.len();
        if k == 0 {
            return Ok(Vec::new());
        }
        for x in xs {
            if x.len() != self.inn {
                return Err(NqError::Shape {
                    x: x.len(),
                    n_in: self.inn,
                });
            }
        }
        // `t_j = round(s_pre_j · x_j / 2^32)` for every column, and each column's `V` bound.
        let pool = Pool::global();
        let width = pool.width().max(1);
        let cols = pack_chunk(k, width);
        let mut t: Vec<i64> = vec![0; k * self.inn];
        let mut narrow_t: Vec<bool> = vec![false; k];
        let build = TBuild {
            xs,
            s_pre: &self.s_pre,
            inn: self.inn,
            cols,
        };
        for_chunks2(&mut t, self.inn * cols, &mut narrow_t, cols, &build)?;
        let t_q = quantized_columns(&t, self.inn, reader);
        let mut t_masks = ternary_columns(&t, self.inn, reader);
        let t_stages = build_stages(t.chunks_exact(self.inn), &narrow_t, &t_q, &mut t_masks);
        let overflow = std::sync::atomic::AtomicBool::new(false);
        // The `V` stage: every factor row is read once and folded against all `k` activation
        // columns. A task owns a contiguous row band and writes its columns into a row-major
        // buffer; `z` below is the reader's column-major form of it. llama.cpp's
        // `ggml_compute_forward_mul_mat` takes the same partition when `nr0 > nr1`
        // (`nchunk0 = nth`, every column inside a task), and it is what keeps one weight-word
        // read shared by every column. The column-partitioned loop re-read the whole stage once
        // per column, so at width `k` it paid `k` times the weight bytes.
        let row_task = self.rank.div_ceil(width.saturating_mul(2)).max(1);
        let mut zr: Vec<i64> = vec![0; self.rank * k];
        pool.chunks_mut(&mut zr, row_task * k, &|_, at, zs| {
            let r0 = at / k;
            let rows = zs.len() / k;
            let mut dots = [0i128; NQ_BLOCK_MAX];
            let mut dots2 = [0i128; NQ_BLOCK_MAX];
            let mut dots8 = [0i128; 8];
            // Group outer, row inner: a group's readers are assembled once for the band, and a
            // group that refuses leaves the previous groups complete. The pair loop below redoes
            // the refused columns for the band; its stores are assignments, so a re-run is the
            // same value.
            let mut wide_to = 0usize;
            if wide {
                while wide_to + NQ_TILE_COLS <= k {
                    let base = wide_to;
                    let tiles: [&Stage<'_>; NQ_TILE_COLS] =
                        std::array::from_fn(|c| &t_stages[base + c]);
                    let Some(group) = Stage::cols(&tiles) else {
                        break;
                    };
                    let mut r = 0usize;
                    let mut served = true;
                    while r < rows {
                        let row = &self.v[(r0 + r) * self.v_words..];
                        if !group.dot_row(row, self.inn, &mut dots8) {
                            served = false;
                            break;
                        }
                        let mut c = 0usize;
                        while c < NQ_TILE_COLS {
                            match i64::try_from(dots8[c])
                                .ok()
                                .and_then(|dot| mul_q32(dot, self.s_mid[r0 + r]))
                            {
                                Some(v) => zs[r * k + base + c] = v,
                                None => {
                                    overflow.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                            c += 1;
                        }
                        r += 1;
                    }
                    if !served {
                        break;
                    }
                    wide_to = base + NQ_TILE_COLS;
                }
            }
            let mut i0 = 0usize;
            while i0 < rows {
                let n = block.max(1).min(rows - i0);
                let plane = &self.v[(r0 + i0) * self.v_words..];
                let mut store = |col: usize, ds: &[i128]| {
                    for (d, &dot) in ds.iter().enumerate() {
                        let r = r0 + i0 + d;
                        let Some(dot) = i64::try_from(dot).ok() else {
                            overflow.store(true, std::sync::atomic::Ordering::Relaxed);
                            continue;
                        };
                        match mul_q32(dot, self.s_mid[r]) {
                            Some(v) => zs[(i0 + d) * k + col] = v,
                            None => overflow.store(true, std::sync::atomic::Ordering::Relaxed),
                        }
                    }
                };
                let mut cc = wide_to;
                while cc < k {
                    if cc + 1 < k
                        && t_stages[cc].dot_rows2(
                            &t_stages[cc + 1],
                            plane,
                            self.v_words,
                            n,
                            self.inn,
                            &mut dots[..n],
                            &mut dots2[..n],
                        )
                    {
                        store(cc, &dots[..n]);
                        store(cc + 1, &dots2[..n]);
                        cc += 2;
                    } else {
                        t_stages[cc].dot_rows(plane, self.v_words, n, self.inn, &mut dots[..n]);
                        store(cc, &dots[..n]);
                        cc += 1;
                    }
                }
                i0 += n;
            }
        });
        if overflow.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(NqError::Overflow);
        }
        // The readers take a column as one slice, so the band buffer is transposed once per
        // stage. Parallel over columns: each task reads one strided column and writes it
        // contiguously.
        let mut z: Vec<i64> = vec![0; k * self.rank];
        pool.chunks_mut(&mut z, self.rank, &|_, at, zs| {
            let c = at / self.rank;
            let mut r = 0usize;
            for slot in zs.iter_mut() {
                *slot = zr[r * k + c];
                r += 1;
            }
        });
        // The `U` stage's per-column bound: `rank·max|z|` decides the narrow source.
        let mut max_abs_z: Vec<u64> = vec![0; k];
        pool.chunks_mut(&mut max_abs_z, cols, &|_, c0, ms| {
            for (j, m) in ms.iter_mut().enumerate() {
                let col = &z[(c0 + j) * self.rank..(c0 + j + 1) * self.rank];
                *m = col.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
            }
        });
        let narrow_z: Vec<bool> = max_abs_z
            .iter()
            .map(|&m| u128::from(m).saturating_mul(self.rank as u128) <= i64::MAX as u128)
            .collect();
        let z_q = quantized_columns(&z, self.rank, reader);
        let mut z_masks = ternary_columns(&z, self.rank, reader);
        let z_stages = build_stages(z.chunks_exact(self.rank), &narrow_z, &z_q, &mut z_masks);
        // The dense residual reads `x`; a pack without a residual layer skips that stage. The
        // sparse correction indexes `x` directly and needs no reader at all.
        let residual = self.res_layers > 0;
        let mut narrow_x = vec![false; k];
        if residual {
            for (c, x) in xs.iter().enumerate() {
                let max_abs_x = x.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
                let bound = u128::from(max_abs_x).saturating_mul(self.inn as u128);
                narrow_x[c] = residual && bound <= i64::MAX as u128;
            }
        }
        let x_q = quantized_slices(xs, reader, residual);
        let mut x_masks = ternary_slices(xs, reader, residual);
        let x_stages = build_stages(xs.iter().copied(), &narrow_x, &x_q, &mut x_masks);
        // The `U` stage: as in `V`, a task owns a contiguous band of output rows and folds every
        // column inside it, so each factor row is read once for all `k` columns. `yr` is
        // row-major; the gather below writes the per-column outputs.
        let row_task = self.out.div_ceil(width.saturating_mul(2)).max(1);
        let mut yr: Vec<i64> = vec![0; self.out * k];
        pool.chunks_mut(&mut yr, row_task * k, &|_, at, ys| {
            let r0 = at / k;
            let rows = ys.len() / k;
            let mut dots = [0i128; NQ_BLOCK_MAX];
            let mut dots2 = [0i128; NQ_BLOCK_MAX];
            let mut dots8 = [0i128; 8];
            let mut wide_to = 0usize;
            if wide {
                while wide_to + NQ_TILE_COLS <= k {
                    let base = wide_to;
                    let tiles: [&Stage<'_>; NQ_TILE_COLS] =
                        std::array::from_fn(|c| &z_stages[base + c]);
                    let Some(group) = Stage::cols(&tiles) else {
                        break;
                    };
                    let mut i = 0usize;
                    let mut served = true;
                    while i < rows {
                        let row = &self.u[(r0 + i) * self.u_words..];
                        if !group.dot_row(row, self.rank, &mut dots8) {
                            served = false;
                            break;
                        }
                        let mut c = 0usize;
                        while c < NQ_TILE_COLS {
                            match self.project_row(r0 + i, dots8[c], &x_stages[base + c]) {
                                Some(v) => ys[i * k + base + c] = v,
                                None => {
                                    overflow.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                            c += 1;
                        }
                        i += 1;
                    }
                    if !served {
                        break;
                    }
                    wide_to = base + NQ_TILE_COLS;
                }
            }
            let mut i0 = 0usize;
            while i0 < rows {
                let n = block.max(1).min(rows - i0);
                let plane = &self.u[(r0 + i0) * self.u_words..];
                let mut store = |col: usize, ds: &[i128]| {
                    let x_stage = &x_stages[col];
                    for (d, &dot) in ds.iter().enumerate() {
                        match self.project_row(r0 + i0 + d, dot, x_stage) {
                            Some(v) => ys[(i0 + d) * k + col] = v,
                            None => overflow.store(true, std::sync::atomic::Ordering::Relaxed),
                        }
                    }
                };
                let mut cc = wide_to;
                while cc < k {
                    if cc + 1 < k
                        && z_stages[cc].dot_rows2(
                            &z_stages[cc + 1],
                            plane,
                            self.u_words,
                            n,
                            self.rank,
                            &mut dots[..n],
                            &mut dots2[..n],
                        )
                    {
                        store(cc, &dots[..n]);
                        store(cc + 1, &dots2[..n]);
                        cc += 2;
                    } else {
                        z_stages[cc].dot_rows(plane, self.u_words, n, self.rank, &mut dots[..n]);
                        store(cc, &dots[..n]);
                        cc += 1;
                    }
                }
                i0 += n;
            }
        });
        if overflow.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(NqError::Overflow);
        }
        let mut out = Vec::with_capacity(k);
        for c in 0..k {
            let col: Vec<i64> = (0..self.out).map(|r| yr[r * k + c]).collect();
            out.push(col);
        }
        Ok(out)
    }

    /// The `U` stage of [`NqLinear::project_with`]: `round(s_post_i·Σ_k U_ik·z_k / 2^32)` plus
    /// the sparse residual, one row per output element, parallel over `pool` chunks.
    ///
    /// A refusal sets `overflow` and leaves the row zero rather than unwinding, because the body
    /// runs on pool workers; the first stage shares the flag, so the caller reads it once after
    /// both stages.
    fn project_rows(
        &self,
        pool: &Pool,
        z: &Stage<'_>,
        x: &Stage<'_>,
        overflow: &std::sync::atomic::AtomicBool,
    ) -> Vec<i64> {
        let width = pool.width().max(1);
        let mut y: Vec<i64> = vec![0; self.out];
        let y_chunk = self.out.div_ceil(width.saturating_mul(4)).max(1);
        pool.chunks_mut(&mut y, y_chunk, &|_, at, ys| {
            for (d, slot) in ys.iter_mut().enumerate() {
                match self.project_one_row(at + d, z, x) {
                    Some(v) => *slot = v,
                    None => overflow.store(true, std::sync::atomic::Ordering::Relaxed),
                }
            }
        });
        y
    }

    /// Output row `i`: the `U` sign dot, the post-scale, the sparse correction and every dense
    /// residual, rounded once. `None` is the checked refusal the caller records in its flag.
    ///
    /// # Complexity
    ///
    /// `O(rank + k + res_layers·inn)` checked operations; `O(1)` space.
    fn project_one_row(&self, i: usize, z: &Stage<'_>, x: &Stage<'_>) -> Option<i64> {
        let urow = &self.u[i * self.u_words..(i + 1) * self.u_words];
        self.project_row(i, z.dot(urow, self.rank), x)
    }

    /// Output row `i` from its `U` sign dot: the post-scale, the sparse correction and every dense
    /// residual, rounded once. The pooled `U` stage supplies `dot` from the blocked reader; the
    /// per-column entry supplies [`Stage::dot`]. The order below is the row's, not the block's.
    ///
    /// A wide `U` sum is not a refusal by itself: the post-scale can bring it back into `i64`, so
    /// only the scaled output is checked. The sparse correction lives in the weight's units:
    /// `Σ_j w_ij·x_idx[j]` is a Q64.32 sum like the `U` term, and one rounding decides the row.
    /// Both the term products and the sums are checked, so an extreme factor refuses instead of
    /// wrapping in release.
    ///
    /// # Complexity
    ///
    /// `O(1 + k + res_layers·inn)` checked operations; `O(1)` space.
    fn project_row(&self, i: usize, dot: i128, x: &Stage<'_>) -> Option<i64> {
        let low = dot.checked_mul(i128::from(self.s_post[i]))?;
        let mut corr: i128 = 0;
        let mut c = 0usize;
        while c < self.k {
            let src = self.salient_idx[c] as usize;
            let term =
                i128::from(self.salient[i * self.k + c]).checked_mul(i128::from(x.wide[src]))?;
            corr = corr.checked_add(term)?;
            c += 1;
        }
        // The dense sign residual: `Σ_l s_res[l,i] · Σ_j sign_lij·x_j` is a Q64.32 sum like
        // `corr`, so the row keeps one rounding after every term is added. The bits in `U` and
        // `V` are consumed as a product, so their shared inversion cancels; a residual plane is
        // read once, and the negated stage dot restores the format's bit-1-is-`+1` convention.
        let mut res: i128 = 0;
        let mut l = 0usize;
        while l < self.res_layers {
            let at = (l * self.out + i) * self.res_words;
            let srow = &self.res[at..at + self.res_words];
            let dot = -x.dot(srow, self.inn);
            res = res.checked_add(dot.checked_mul(i128::from(self.s_res[l * self.out + i]))?)?;
            l += 1;
        }
        let v = low
            .checked_add(corr)?
            .checked_add(res)?
            .checked_add(1_i128 << 31)?;
        i64::try_from(v >> 32).ok()
    }
}

/// The activation-pack band: `k` columns are packed in chunks of `cols`, so one pool task owns a
/// frame of columns and the dispatch has about `4·width` tasks.
///
/// # Complexity
///
/// `O(1)`.
fn pack_chunk(k: usize, width: usize) -> usize {
    k.div_ceil(width.saturating_mul(4)).max(1)
}

/// The `t` stage's per-column build, one pool task per band of `cols` columns: the flat `t`
/// buffer's `[c0·inn, (c0 + len)·inn)` piece and the matching `narrow` piece.
///
/// # Complexity
///
/// `O(cols·inn)` checked products in the task, `O(k·inn)` over the dispatch.
struct TBuild<'a> {
    xs: &'a [&'a [i64]],
    s_pre: &'a [i64],
    inn: usize,
    cols: usize,
}

impl Task2<i64, bool, NqError> for TBuild<'_> {
    fn run(&self, t: usize, ts: &mut [i64], narrow: &mut [bool]) -> Result<(), NqError> {
        t_build(self.xs, self.s_pre, self.inn, t * self.cols, ts, narrow)
    }
}

/// [`TBuild`]'s body: `t_j = round(s_pre_j·x_j / 2^32)` for column `c0 + c`, and the
/// `inn·max|t| ≤ i64::MAX` bound its `V` stage reads. The per-column order, the checked product and
/// the rounding are the serial loop's.
///
/// # Complexity
///
/// `O(narrow.len() · inn)` checked products; `O(1)` space beyond the caller's buffers.
fn t_build(
    xs: &[&[i64]],
    s_pre: &[i64],
    inn: usize,
    c0: usize,
    ts: &mut [i64],
    narrow: &mut [bool],
) -> Result<(), NqError> {
    for (c, slot) in narrow.iter_mut().enumerate() {
        let x = xs[c0 + c];
        let at = c * inn;
        let mut max_abs_t = 0u64;
        for j in 0..inn {
            let v = mul_q32(x[j], s_pre[j]).ok_or(NqError::Overflow)?;
            max_abs_t = max_abs_t.max(v.unsigned_abs());
            ts[at + j] = v;
        }
        let bound = u128::from(max_abs_t).saturating_mul(inn as u128);
        *slot = bound <= i64::MAX as u128;
    }
    Ok(())
}

/// The Q8 reader over `flat`'s columns of `n` words each, or one empty slot per column under a
/// reader that does not quantize. Task `t` quantizes columns `t·cols .. (t + 1)·cols` with the
/// same per-column [`quant_act`] the serial loop calls, so each column's `Option` is unchanged.
///
/// # Complexity
///
/// `O(k·n)` time and space in the column count `k` and width `n`, split over `⌈k/cols⌉` tasks.
fn quantized_columns(flat: &[i64], n: usize, reader: Reader) -> Vec<Option<QuantAct>> {
    let k = flat.len() / n;
    let mut out: Vec<Option<QuantAct>> = (0..k).map(|_| None).collect();
    if reader != Reader::Q8 || k == 0 {
        return out;
    }
    let cols = pack_chunk(k, Pool::global().width().max(1));
    Pool::global().chunks_mut(&mut out, cols, &|_, c0, qs| {
        for (j, slot) in qs.iter_mut().enumerate() {
            *slot = quant_act(&flat[(c0 + j) * n..(c0 + j + 1) * n]);
        }
    });
    out
}

/// The ternary reader over `flat`'s columns of `n` words each, or one empty slot per column under
/// a reader that does not quantize.
///
/// # Complexity
///
/// `O(k·n)` time and `O(k·n/64)` space.
fn ternary_columns(flat: &[i64], n: usize, reader: Reader) -> Vec<Option<TernaryTerms>> {
    if reader == Reader::Ternary {
        flat.chunks_exact(n).map(TernaryTerms::new).collect()
    } else {
        (0..flat.len() / n).map(|_| None).collect()
    }
}

/// The Q8 reader over separate column slices, or one empty slot per column when the reader does
/// not quantize or the caller has no residual layer to read.
///
/// # Complexity
///
/// `O(k·n)` time and space.
fn quantized_slices(xs: &[&[i64]], reader: Reader, residual: bool) -> Vec<Option<QuantAct>> {
    if reader == Reader::Q8 && residual {
        xs.iter().map(|x| quant_act(x)).collect()
    } else {
        (0..xs.len()).map(|_| None).collect()
    }
}

/// The ternary reader over separate column slices, or one empty slot per column when the reader
/// does not quantize or the caller has no residual layer to read.
///
/// # Complexity
///
/// `O(k·n)` time and `O(k·n/64)` space.
fn ternary_slices(xs: &[&[i64]], reader: Reader, residual: bool) -> Vec<Option<TernaryTerms>> {
    if reader == Reader::Ternary && residual {
        xs.iter().map(|x| TernaryTerms::new(x)).collect()
    } else {
        (0..xs.len()).map(|_| None).collect()
    }
}

/// The readers of a column group: one [`Stage`] per column, its Q8 codes built from `quant`, its
/// ternary masks taken from `masks`, and its exact planes when no block reader applies and the
/// column's narrow bound holds. The slices `quant` and `masks` are indexed by column and are
/// consumed column by column, so the caller keeps owning the quantized words.
///
/// # Complexity
///
/// `O(k·n)` time for the codes, which is the only reader built here; `O(k·n)` space.
fn build_stages<'a>(
    cols: impl Iterator<Item = &'a [i64]>,
    narrow: &[bool],
    quant: &'a [Option<QuantAct>],
    masks: &mut [Option<TernaryTerms>],
) -> Vec<Stage<'a>> {
    let mut codes: Vec<Option<CodeTerms<'a>>> = quant
        .iter()
        .map(|q| q.as_ref().and_then(CodeTerms::new))
        .collect();
    let mut stages = Vec::with_capacity(narrow.len());
    for (c, wide) in cols.enumerate() {
        let code = codes[c].take();
        let ternary = masks[c].take();
        let terms = (code.is_none() && ternary.is_none() && narrow[c]).then(|| Terms::new(wide));
        stages.push(Stage {
            wide,
            terms,
            codes: code,
            ternary,
        });
    }
    stages
}

/// The decoded geometry of one packed linear: strides, sparse columns and residual planes.
struct Shape {
    u_words: usize,
    v_words: usize,
    salient_len: usize,
    k: usize,
    res_words: usize,
    res_layers: usize,
}

/// Validate a [`PackedLinear`]'s dimensions and section lengths.
///
/// Returns the U/V strides, the sparse residual's row count and column count, and the dense
/// residual's layer count and row stride, all checked against the declared `out`, `inn` and
/// `rank`.
fn checked_shape(p: &PackedLinear) -> Result<Shape, NqError> {
    if p.out == 0 || p.inn == 0 || p.rank == 0 {
        return Err(NqError::Packed {
            field: "dimensions",
            have: 0,
            expected: 1,
        });
    }
    let u_words = p.rank.div_ceil(32);
    let v_words = p.inn.div_ceil(32);
    let u_len = p.out.checked_mul(u_words).ok_or(NqError::Overflow)?;
    let v_len = p.rank.checked_mul(v_words).ok_or(NqError::Overflow)?;
    if p.u.len() != u_len {
        return Err(NqError::Packed {
            field: "U",
            have: p.u.len(),
            expected: u_len,
        });
    }
    if p.v.len() != v_len {
        return Err(NqError::Packed {
            field: "V",
            have: p.v.len(),
            expected: v_len,
        });
    }
    if p.s_pre.len() != p.inn {
        return Err(NqError::Packed {
            field: "s_pre",
            have: p.s_pre.len(),
            expected: p.inn,
        });
    }
    if p.s_mid.len() != p.rank {
        return Err(NqError::Packed {
            field: "s_mid",
            have: p.s_mid.len(),
            expected: p.rank,
        });
    }
    if p.s_post.len() != p.out {
        return Err(NqError::Packed {
            field: "s_post",
            have: p.s_post.len(),
            expected: p.out,
        });
    }
    let k = p.salient_idx.len();
    if k > p.inn {
        return Err(NqError::Packed {
            field: "salient_idx",
            have: k,
            expected: p.inn,
        });
    }
    let salient_len = p.out.checked_mul(k).ok_or(NqError::Overflow)?;
    if p.salient.len() != salient_len {
        return Err(NqError::Packed {
            field: "salient",
            have: p.salient.len(),
            expected: salient_len,
        });
    }
    let res_words = p.inn.div_ceil(32);
    let res_len = p
        .res_layers
        .checked_mul(p.out)
        .and_then(|x| x.checked_mul(res_words))
        .ok_or(NqError::Overflow)?;
    if p.res.len() != res_len {
        return Err(NqError::Packed {
            field: "res",
            have: p.res.len(),
            expected: res_len,
        });
    }
    let s_res_len = p.res_layers.checked_mul(p.out).ok_or(NqError::Overflow)?;
    if p.s_res.len() != s_res_len {
        return Err(NqError::Packed {
            field: "s_res",
            have: p.s_res.len(),
            expected: s_res_len,
        });
    }
    Ok(Shape {
        u_words,
        v_words,
        salient_len,
        k,
        res_words,
        res_layers: p.res_layers,
    })
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test-only small indices and the staged oracle index their vectors as written"
)]
mod tests {
    use super::*;
    use crate::engine::num::value::round_shift_32;
    use crate::format::nq::PackedLinear;

    /// The stated relative bound on [`NqLinear::project_q8`] against [`NqLinear::project`] on the
    /// `packed(11)` fixture: `max_dev·32 ≤ max_abs`. The production shapes measure 0.96–1.04% of
    /// `|y|` at eight output rows per activation; one int8 activation code per element bounds the
    /// error at about `1/127` of the activation's magnitude, so `1/32` of `|y|` is a threefold
    /// margin and a route mix-up (a different reader's whole-unit error) leaves it.
    const Q8_REL_BOUND: u64 = 32;

    /// A deterministic xorshift word source.
    fn next_u32(state: &mut u64) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state as u32
    }

    /// `f32` to bfloat16 bits, round-to-nearest (test-only; scales only).
    fn bf16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let round = 0x7fff + ((bits >> 16) & 1);
        ((bits.wrapping_add(round)) >> 16) as u16
    }

    /// A random `PackedLinear` at a caller's shape, for the timing harness; [`packed`] is the
    /// fixed small shape the equality tests use. No residual layer, as the 0.6B pack has none.
    fn packed_shape(out: usize, inn: usize, rank: usize) -> PackedLinear {
        let u_words = rank.div_ceil(32);
        let v_words = inn.div_ceil(32);
        let mut state =
            0x9e37_79b9_7f4a_7c15u64 ^ ((out as u64) << 40) ^ ((inn as u64) << 20) ^ rank as u64;
        let mut u = Vec::with_capacity(out * u_words);
        let mut v = Vec::with_capacity(rank * v_words);
        for _ in 0..out * u_words {
            u.push(next_u32(&mut state));
        }
        for _ in 0..rank * v_words {
            v.push(next_u32(&mut state));
        }
        PackedLinear {
            layer: 0,
            kind: 0,
            out,
            inn,
            rank,
            u: u.into(),
            v: v.into(),
            s_pre: (0..inn)
                .map(|i| bf16_bits(0.01 + 0.001 * (i % 7) as f32))
                .collect::<Vec<u16>>()
                .into(),
            s_mid: (0..rank)
                .map(|i| bf16_bits(0.5 + 0.01 * (i % 3) as f32))
                .collect::<Vec<u16>>()
                .into(),
            s_post: (0..out)
                .map(|i| bf16_bits(0.02 + 0.002 * (i % 5) as f32))
                .collect::<Vec<u16>>()
                .into(),
            salient_idx: vec![1, 7].into(),
            salient: (0..out * 2)
                .map(|i| 0x2400u16 + (i % 9) as u16)
                .collect::<Vec<u16>>()
                .into(),
            res_layers: 0,
            shared: false,
            res: Vec::new().into(),
            s_res: Vec::new().into(),
        }
    }

    fn packed(reps: u64) -> PackedLinear {
        let (out, inn, rank) = (37usize, 29usize, 23usize);
        let u_words = rank.div_ceil(32);
        let v_words = inn.div_ceil(32);
        let mut state = 0x1234_5678_9abc_def0u64 ^ reps;
        let mut u = Vec::with_capacity(out * u_words);
        let mut v = Vec::with_capacity(rank * v_words);
        for _ in 0..out * u_words {
            u.push(next_u32(&mut state));
        }
        for _ in 0..rank * v_words {
            v.push(next_u32(&mut state));
        }
        let res_layers = 1usize;
        let res_words = inn.div_ceil(32);
        let mut planes = Vec::with_capacity(res_layers * out * res_words);
        for _ in 0..res_layers * out * res_words {
            planes.push(next_u32(&mut state));
        }
        PackedLinear {
            layer: 0,
            kind: 0,
            out,
            inn,
            rank,
            u: u.into(),
            v: v.into(),
            s_pre: (0..inn)
                .map(|i| bf16_bits(0.01 + 0.001 * (i % 7) as f32))
                .collect::<Vec<u16>>()
                .into(),
            s_mid: (0..rank)
                .map(|i| bf16_bits(0.5 + 0.01 * (i % 3) as f32))
                .collect::<Vec<u16>>()
                .into(),
            s_post: (0..out)
                .map(|i| bf16_bits(0.02 + 0.002 * (i % 5) as f32))
                .collect::<Vec<u16>>()
                .into(),
            salient_idx: vec![1, 7].into(),
            salient: (0..out * 2)
                .map(|i| 0x2400u16 + (i % 9) as u16)
                .collect::<Vec<u16>>()
                .into(),
            res_layers,
            shared: false,
            res: planes.into(),
            s_res: (0..res_layers * out)
                .map(|i| bf16_bits(0.0005 + 0.0001 * (i % 7) as f32))
                .collect::<Vec<u16>>()
                .into(),
        }
    }

    /// The staged product written out one term at a time: the kernel must equal it word for word.
    fn sequential(p: &PackedLinear, x: &[i64]) -> Vec<i64> {
        let s_pre: Vec<i64> = p
            .s_pre
            .iter()
            .map(|&b| bf16_to_fixed(b).expect("scale"))
            .collect();
        let s_mid: Vec<i64> = p
            .s_mid
            .iter()
            .map(|&b| bf16_to_fixed(b).expect("scale"))
            .collect();
        let s_post: Vec<i64> = p
            .s_post
            .iter()
            .map(|&b| bf16_to_fixed(b).expect("scale"))
            .collect();
        let salient: Vec<i64> = p
            .salient
            .iter()
            .map(|&b| f16_to_fixed(b).expect("salient"))
            .collect();
        let s_res: Vec<i64> = p
            .s_res
            .iter()
            .map(|&b| bf16_to_fixed(b).expect("residual scale"))
            .collect();
        let t: Vec<i64> = (0..p.inn)
            .map(|j| mul_q32(x[j], s_pre[j]).expect("t"))
            .collect();
        let v_words = p.inn.div_ceil(32);
        let z: Vec<i64> = (0..p.rank)
            .map(|k| {
                let mut acc: i128 = 0;
                for j in 0..p.inn {
                    let bit = (p.v[k * v_words + j / 32] >> (j % 32)) & 1;
                    acc += if bit == 1 {
                        i128::from(t[j])
                    } else {
                        -i128::from(t[j])
                    };
                }
                mul_q32(i64::try_from(acc).expect("z"), s_mid[k]).expect("mid")
            })
            .collect();
        let u_words = p.rank.div_ceil(32);
        (0..p.out)
            .map(|i| {
                let mut acc: i128 = 0;
                for k in 0..p.rank {
                    let bit = (p.u[i * u_words + k / 32] >> (k % 32)) & 1;
                    acc += if bit == 1 {
                        i128::from(z[k])
                    } else {
                        -i128::from(z[k])
                    };
                }
                let mut corr: i128 = 0;
                for j in 0..p.salient_idx.len() {
                    corr += i128::from(salient[i * p.salient_idx.len() + j])
                        * i128::from(x[p.salient_idx[j] as usize]);
                }
                let mut res: i128 = 0;
                for l in 0..p.res_layers {
                    let mut acc = 0i128;
                    for j in 0..p.inn {
                        let bit = (p.res[(l * p.out + i) * p.res_words() + j / 32] >> (j % 32)) & 1;
                        acc += if bit == 1 {
                            i128::from(x[j])
                        } else {
                            -i128::from(x[j])
                        };
                    }
                    res += acc * i128::from(s_res[l * p.out + i]);
                }
                i64::try_from(round_shift_32(acc * i128::from(s_post[i]) + corr + res)).expect("y")
            })
            .collect()
    }

    #[test]
    fn the_kernel_equals_the_sequential_staged_product() {
        let p = packed(1);
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        let x: Vec<i64> = (0..p.inn)
            .map(|i| ((i as i64 * 37 + 11) % 101 - 50) << 24)
            .collect();
        assert_eq!(linear.project(&x).expect("project"), sequential(&p, &x));
    }

    /// The eight-column source against its per-column twins over the whole vector domain: each
    /// column carries its own codes, scale heights and `e_min`, `len` covers whole 32-term blocks
    /// and a tail whose padding bits are set, and the word span leaves slack past the blocks. The
    /// group must give every column its single-column sum, and must refuse a group whose columns
    /// cannot serve the length without writing the output.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_wide_source_equals_eight_per_column_sources() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x51c3_9e27_44d1_0b6au64;
        for len in [32usize, 33, 63, 64, 95, 1000, 1023] {
            let blocks = len.div_ceil(32);
            let mut acts: Vec<QuantAct> = Vec::with_capacity(NQ_TILE_COLS);
            for c in 0..NQ_TILE_COLS {
                #[allow(clippy::cast_possible_truncation, reason = "column index < 8")]
                let e = 3 * c as u32;
                let period = c + 2;
                let codes: Vec<i8> = (0..len)
                    .map(|_| {
                        i8::try_from(i32::try_from(next_u32(&mut state) % 255).expect("byte") - 127)
                            .expect("[-127, 127]")
                    })
                    .collect();
                let scales: Vec<i64> = (0..blocks)
                    .map(|b| 1i64 << (e + u32::from(b % period == 0)))
                    .collect();
                acts.push(QuantAct::from_parts(scales, codes));
            }
            let terms: Vec<CodeTerms<'_>> = acts
                .iter()
                .map(|q| CodeTerms::new(q).expect("the fixture admits"))
                .collect();
            let cols: [ColRef<'_>; NQ_TILE_COLS] = std::array::from_fn(|c| terms[c].col_ref());
            let stride = blocks + 1;
            let mut words: Vec<u32> = (0..stride).map(|_| next_u32(&mut state)).collect();
            if len % 32 != 0 {
                words[blocks - 1] |= !0u32 << (len % 32);
            }
            let mut got = [0i64; NQ_TILE_COLS];
            assert!(
                CodeTerms::selected_row_cols(&cols, &words, len, &mut got),
                "len {len}: the group refused"
            );
            for c in 0..NQ_TILE_COLS {
                let want = code_select_twin(terms[c].codes, terms[c].scales, &words, len);
                assert_eq!(i128::from(got[c]), want, "len {len} column {c}");
            }
        }
        // A column shorter than the length refuses the group, and the caller's fallback is what
        // the domain check protects.
        let q = QuantAct::from_parts(vec![1], vec![1; 32]);
        let terms = CodeTerms::new(&q).expect("the fixture admits");
        let cols: [ColRef<'_>; NQ_TILE_COLS] = std::array::from_fn(|_| terms.col_ref());
        let mut got = [0i64; NQ_TILE_COLS];
        assert!(!CodeTerms::selected_row_cols(
            &cols, &[0u32; 2], 64, &mut got
        ));
        assert_eq!(got, [0i64; NQ_TILE_COLS]);
    }

    /// The two-column Q8 source against its two per-row twins over the whole vector domain: `len`
    /// covers whole 32-term blocks and a tail, the two activations carry different `e_min`s and
    /// shift patterns, `stride` is padded past the block count, the tail word's padding bits are
    /// set, and the row count steps every block size the pair admits. The pair must give each
    /// column the single-column sum, and the pair entry must refuse a row count outside
    /// `2..=NQ_BLOCK_MAX` without writing either output.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the lengths, exponents, rows, twin comparisons and the refusal guard are one fixture"
    )]
    fn the_two_column_source_equals_two_per_row_sources() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x2c01_7e57_9d3b_44a1u64;
        for len in [32usize, 33, 63, 64, 65, 127, 128, 129, 1000, 1023] {
            let blocks = len.div_ceil(32);
            for e in [0u32, 20, 40] {
                let codes_a: Vec<i8> = (0..len)
                    .map(|_| {
                        i8::try_from(i32::try_from(next_u32(&mut state) % 255).expect("byte") - 127)
                            .expect("[-127, 127]")
                    })
                    .collect();
                let codes_b: Vec<i8> = (0..len)
                    .map(|_| {
                        i8::try_from(i32::try_from(next_u32(&mut state) % 255).expect("byte") - 127)
                            .expect("[-127, 127]")
                    })
                    .collect();
                // Blocks alternate their scale height within each activation, and the second
                // activation's smallest exponent is three above the first's, so the two
                // `shift_vecs` and `e_min`s differ.
                let scales_a: Vec<i64> = (0..blocks)
                    .map(|b| 1i64 << (e + u32::from(b % 2 == 0) * 2))
                    .collect();
                let scales_b: Vec<i64> = (0..blocks)
                    .map(|b| 1i64 << (e + 3 + u32::from(b % 3 == 0)))
                    .collect();
                let qa = QuantAct::from_parts(scales_a.clone(), codes_a.clone());
                let qb = QuantAct::from_parts(scales_b.clone(), codes_b.clone());
                let a = CodeTerms::new(&qa).expect("a admits the fixture");
                let b = CodeTerms::new(&qb).expect("b admits the fixture");
                for rows in 2..=NQ_BLOCK_MAX {
                    // A stride one word past the block count leaves one word of slack on every
                    // row, as an `inn` that is not a multiple of 32 does.
                    let stride = blocks + rows % 2;
                    let mut words: Vec<u32> =
                        (0..rows * stride).map(|_| next_u32(&mut state)).collect();
                    if len % 32 != 0 {
                        for r in 0..rows {
                            words[r * stride + blocks - 1] |= !0u32 << (len % 32);
                        }
                    }
                    let mut got_a = vec![0i64; rows];
                    let mut got_b = vec![0i64; rows];
                    assert!(
                        a.selected_rows2(&b, &words, stride, rows, len, &mut got_a, &mut got_b),
                        "len {len} e {e} rows {rows}: the pair refused"
                    );
                    let mut want_a = vec![0i64; rows];
                    let mut want_b = vec![0i64; rows];
                    assert!(
                        code_select_rows_twin(
                            &codes_a,
                            &scales_a,
                            &words,
                            stride,
                            rows,
                            len,
                            &mut want_a
                        ),
                        "len {len} e {e} rows {rows}: the a twin refused"
                    );
                    assert!(
                        code_select_rows_twin(
                            &codes_b,
                            &scales_b,
                            &words,
                            stride,
                            rows,
                            len,
                            &mut want_b
                        ),
                        "len {len} e {e} rows {rows}: the b twin refused"
                    );
                    assert_eq!(got_a, want_a, "len {len} e {e} rows {rows}: column a");
                    assert_eq!(got_b, want_b, "len {len} e {e} rows {rows}: column b");
                }
            }
        }
        // The pair entry's row-count guard runs before its stack scratch is indexed, so a count
        // outside the block is a refusal, not a panic, and it leaves both outputs alone.
        let codes: Vec<i8> = vec![1; 32];
        let scales: Vec<i64> = vec![1];
        let q = QuantAct::from_parts(scales, codes);
        let terms = CodeTerms::new(&q).expect("the fixture admits");
        let stage_a = Stage {
            wide: &[],
            terms: None,
            codes: Some(terms),
            ternary: None,
        };
        let stage_b = Stage {
            wide: &[],
            terms: None,
            codes: Some(CodeTerms::new(&q).expect("the fixture admits")),
            ternary: None,
        };
        let words = vec![u32::MAX; NQ_BLOCK_MAX + 1];
        let mut out_a = vec![0i128; NQ_BLOCK_MAX + 1];
        let mut out_b = vec![0i128; NQ_BLOCK_MAX + 1];
        assert!(!stage_a.dot_rows2(
            &stage_b,
            &words,
            1,
            NQ_BLOCK_MAX + 1,
            32,
            &mut out_a,
            &mut out_b
        ));
        assert!(!stage_a.dot_rows2(&stage_b, &words, 1, 1, 32, &mut out_a[..1], &mut out_b[..1]));
        assert!(!stage_a.dot_rows2(&stage_b, &words, 1, 2, 32, &mut out_a[..1], &mut out_b[..2]));
        assert_eq!(out_a, vec![0i128; NQ_BLOCK_MAX + 1]);
        assert_eq!(out_b, vec![0i128; NQ_BLOCK_MAX + 1]);
    }

    /// The blocked Q8 reader, its safe twin and the per-row entry must agree for every row count
    /// the block admits, at every word boundary and at two block-scale heights, with a padded
    /// stride and with the tail word's padding bits set. The blocked source shares one code load
    /// and one shift load across the rows; that sharing must not move a single row's sum.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_blocked_code_source_equals_the_per_row_source() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x5eed_b10c_0f17_1357u64;
        for len in [1usize, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 1023] {
            for e in [0u32, 40] {
                let blocks = len.div_ceil(32);
                let codes: Vec<i8> = (0..len)
                    .map(|_| {
                        i8::try_from(i32::try_from(next_u32(&mut state) % 255).expect("byte") - 127)
                            .expect("[-127, 127]")
                    })
                    .collect();
                let scales: Vec<i64> = (0..blocks)
                    .map(|b| 1i64 << (e + u32::from(b % 2 == 0) * 2))
                    .collect();
                let q = QuantAct::from_parts(scales.clone(), codes.clone());
                let terms = CodeTerms::new(&q).expect("the code plan admits the fixture");
                for rows in 1..=NQ_BLOCK_MAX {
                    // A stride one word past the block count leaves one word of slack on every
                    // row, as an `inn` that is not a multiple of 32 does.
                    let stride = blocks + rows % 2;
                    let mut words: Vec<u32> =
                        (0..rows * stride).map(|_| next_u32(&mut state)).collect();
                    if len % 32 != 0 {
                        for r in 0..rows {
                            words[r * stride + blocks - 1] |= !0u32 << (len % 32);
                        }
                    }
                    let want: Vec<i64> = (0..rows)
                        .map(|r| {
                            terms
                                .selected(&words[r * stride..], len)
                                .expect("the admitted plan is in range")
                        })
                        .collect();
                    let mut twin = vec![0i64; rows];
                    assert!(
                        code_select_rows_twin(
                            &codes, &scales, &words, stride, rows, len, &mut twin
                        ),
                        "len {len} e {e} rows {rows}: the twin refused"
                    );
                    assert_eq!(twin, want, "len {len} e {e} rows {rows}: twin");
                    let mut blocked = vec![0i64; rows];
                    assert!(
                        terms.selected_rows(&words, stride, rows, len, &mut blocked),
                        "len {len} e {e} rows {rows}: the blocked source refused"
                    );
                    assert_eq!(blocked, want, "len {len} e {e} rows {rows}: blocked source");
                }
            }
        }
    }

    /// The pooled entries are the single-column entries over a group: same pre-scale, same
    /// quantization, same reader and the same accumulation order, so every column is equal bit
    /// for bit at every group width. The fixture carries a sparse correction and a dense
    /// residual, so both are compared too.
    #[test]
    fn the_pooled_readers_equal_the_per_column_readers() {
        let p = packed(12);
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        for k in [1usize, 2, 3, 7, 32, 33, 256] {
            let xs: Vec<Vec<i64>> = (0..k)
                .map(|c| {
                    (0..p.inn)
                        .map(|j| ((j * 37 + c * 11 + 5) as i64 % 101 - 50) << 24)
                        .collect()
                })
                .collect();
            let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
            let exact: Vec<Vec<i64>> = xs
                .iter()
                .map(|x| linear.project(x).expect("exact"))
                .collect();
            let q8: Vec<Vec<i64>> = xs
                .iter()
                .map(|x| linear.project_q8(x).expect("q8"))
                .collect();
            let ternary: Vec<Vec<i64>> = xs
                .iter()
                .map(|x| linear.project_ternary(x).expect("ternary"))
                .collect();
            assert_eq!(
                linear.project_pooled(&refs).expect("pooled exact"),
                exact,
                "exact, k {k}"
            );
            assert_eq!(
                linear.project_q8_pooled(&refs).expect("pooled q8"),
                q8,
                "q8, k {k}"
            );
            assert_eq!(
                linear
                    .project_ternary_pooled(&refs)
                    .expect("pooled ternary"),
                ternary,
                "ternary, k {k}"
            );
        }
    }

    /// Reports the pooled block reader's rate by row block on the `ffn` gate shape (3072×1024,
    /// rank 736) at the prefill block's 228 columns; the rates are printed, not asserted, and
    /// every arm must agree exactly with block 0.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "reports timing; run in release with --ignored --nocapture"]
    fn the_pooled_block_rate_by_block() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("avx2 unavailable; there is no blocked reader to time");
            return;
        }
        let (out, inn, rank, k) = (3072usize, 1024usize, 736usize, 228usize);
        let linear = NqLinear::from_packed(packed_shape(out, inn, rank)).expect("decode");
        let xs: Vec<Vec<i64>> = (0..k)
            .map(|c| {
                (0..inn)
                    .map(|j| ((j * 37 + c * 11 + 5) as i64 % 101 - 50) << 24)
                    .collect()
            })
            .collect();
        let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
        let terms = (rank * inn + out * rank) as f64 * k as f64;
        println!(
            "pool width {}; {k} columns, {out}x{inn} r{rank}, {:.3} G terms/call",
            Pool::global().width(),
            terms / 1e9
        );
        let mut first: Option<Vec<Vec<i64>>> = None;
        for tile in [2usize, NQ_TILE_COLS] {
            for block in 0..=NQ_BLOCK_MAX {
                let mut ns = [0u128; 3];
                let mut got = Vec::new();
                for slot in &mut ns {
                    let start = std::time::Instant::now();
                    got = linear
                        .project_pooled_shape(&refs, Reader::Q8, block, tile)
                        .expect("q8 pooled block");
                    *slot = start.elapsed().as_nanos();
                }
                match &first {
                    None => first = Some(got),
                    Some(want) => assert_eq!(
                        &got, want,
                        "tile {tile} block {block} disagrees with tile 2 block 0"
                    ),
                }
                let mean = ns.iter().map(|&n| n as f64).sum::<f64>() / 3.0;
                let min = ns.iter().copied().min().unwrap_or(0) as f64;
                println!(
                    "tile {tile} block {block}: {:8.2} ms (min {:8.2})  {:6.2} G terms/s",
                    mean / 1e6,
                    min / 1e6,
                    terms / mean
                );
            }
        }
    }

    /// The pooled block selects the loop nesting, not the arithmetic: every block size from the
    /// per-row order through [`NQ_BLOCK_MAX`], plus the clamp edges, reproduces the per-column
    /// entries bit for bit for all three readers. The fixture carries a sparse correction and a
    /// dense residual, and the group widths exercise a short and a wide column group.
    #[test]
    fn the_pooled_block_is_bit_equal_at_every_block() {
        let p = packed(12);
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        for k in [1usize, 7, 33] {
            let xs: Vec<Vec<i64>> = (0..k)
                .map(|c| {
                    (0..p.inn)
                        .map(|j| ((j * 37 + c * 11 + 5) as i64 % 101 - 50) << 24)
                        .collect()
                })
                .collect();
            let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
            let exact: Vec<Vec<i64>> = xs
                .iter()
                .map(|x| linear.project(x).expect("exact"))
                .collect();
            let q8: Vec<Vec<i64>> = xs
                .iter()
                .map(|x| linear.project_q8(x).expect("q8"))
                .collect();
            let ternary: Vec<Vec<i64>> = xs
                .iter()
                .map(|x| linear.project_ternary(x).expect("ternary"))
                .collect();
            // `0` is the per-row loop and a value past the register budget clamps, as the
            // environment switch parses them; every value between is a block of factor rows.
            // The column tile is a loop nesting like the row block: the eight-column wide pass
            // and the two-column pair give the same words, alone and together.
            for tile in [2usize, 5, NQ_TILE_COLS, NQ_TILE_COLS + 1] {
                for block in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, NQ_BLOCK_MAX + 5] {
                    assert_eq!(
                        linear
                            .project_pooled_shape(&refs, Reader::Exact, block, tile)
                            .expect("pooled exact"),
                        exact,
                        "exact, k {k}, block {block}, tile {tile}"
                    );
                    assert_eq!(
                        linear
                            .project_pooled_shape(&refs, Reader::Q8, block, tile)
                            .expect("pooled q8"),
                        q8,
                        "q8, k {k}, block {block}, tile {tile}"
                    );
                    assert_eq!(
                        linear
                            .project_pooled_shape(&refs, Reader::Ternary, block, tile)
                            .expect("pooled ternary"),
                        ternary,
                        "ternary, k {k}, block {block}, tile {tile}"
                    );
                }
            }
        }
    }

    /// The environment switch's parse: `0` is the per-row order, a block up to the register budget
    /// is kept, a larger value clamps, and an absent or malformed value is the default.
    #[test]
    fn the_block_switch_parses_the_per_row_order_and_the_clamp() {
        assert_eq!(parse_block(None), NQ_BLOCK);
        assert_eq!(parse_block(Some("")), NQ_BLOCK);
        assert_eq!(parse_block(Some("x")), NQ_BLOCK);
        assert_eq!(parse_block(Some("-1")), NQ_BLOCK);
        assert_eq!(parse_block(Some("0")), 0);
        assert_eq!(parse_block(Some(" 1 ")), 1);
        assert_eq!(parse_block(Some("4")), 4);
        assert_eq!(parse_block(Some("8")), NQ_BLOCK_MAX);
        assert_eq!(parse_block(Some("9")), NQ_BLOCK_MAX);
        assert_eq!(parse_block(Some("999999999999999999999")), NQ_BLOCK);
    }

    /// The column dial: the default is the wide tile, a value at or above it is the tile, and
    /// everything else keeps the two-column pair.
    #[test]
    fn the_column_switch_parses_the_pair_and_the_tile() {
        assert_eq!(parse_cols(None, 1), 2);
        assert_eq!(parse_cols(None, 2), 2);
        assert_eq!(parse_cols(None, 4), NQ_TILE_COLS);
        assert_eq!(parse_cols(None, 8), NQ_TILE_COLS);
        assert_eq!(parse_cols(Some(""), 8), NQ_TILE_COLS);
        assert_eq!(parse_cols(Some("x"), 8), NQ_TILE_COLS);
        assert_eq!(parse_cols(Some("0"), 8), 2);
        assert_eq!(parse_cols(Some(" 2 "), 8), 2);
        assert_eq!(parse_cols(Some("5"), 8), 2);
        assert_eq!(parse_cols(Some("8"), 2), NQ_TILE_COLS);
        assert_eq!(parse_cols(Some("64"), 1), NQ_TILE_COLS);
        assert_eq!(parse_cols(Some("999999999999999999999"), 8), NQ_TILE_COLS);
    }

    /// The served pack has no residual layer and no sparse correction: the pooled reader must
    /// skip the `x` stage and still equal the single-column reader.
    #[test]
    fn the_pooled_reader_skips_the_x_stage_without_a_residual() {
        let mut p = packed(14);
        p.res_layers = 0;
        p.res.clear();
        p.s_res.clear();
        p.salient_idx.clear();
        p.salient.clear();
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        let xs: Vec<Vec<i64>> = (0..33)
            .map(|c| {
                (0..p.inn)
                    .map(|j| (((j * 17 + c * 7) % 61) as i64 - 30) << 25)
                    .collect()
            })
            .collect();
        let refs: Vec<&[i64]> = xs.iter().map(Vec::as_slice).collect();
        let single: Vec<Vec<i64>> = xs
            .iter()
            .map(|x| linear.project_q8(x).expect("q8"))
            .collect();
        assert_eq!(linear.project_q8_pooled(&refs).expect("pooled q8"), single);
        // The group's widths are checked before the arithmetic: the whole call refuses.
        let short = &xs[0][..p.inn - 1];
        assert!(matches!(
            linear.project_q8_pooled(&[refs[0], short]),
            Err(NqError::Shape { .. })
        ));
    }

    #[test]
    fn the_wide_accumulator_agrees_when_the_bound_leaves_i64() {
        // Unit scales, sign planes all `-1`, and an alternating activation that cancels inside
        // `V`: `z` is one 2^60 term, the `U` sum is 23·2^60 (past `i64`), and the 2^-8 post-scale
        // brings the output back. `29·2^60` is past `i64`, so both wide sources run.
        let mut p = packed(4);
        p.v.fill(0);
        p.u.fill(0);
        p.s_pre.fill(bf16_bits(1.0));
        p.s_mid.fill(bf16_bits(1.0));
        p.s_post.fill(bf16_bits(1.0 / 256.0));
        p.salient.fill(0);
        p.res_layers = 0;
        p.res.clear();
        p.s_res.clear();
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        let x: Vec<i64> = (0..p.inn)
            .map(|i| {
                if i % 2 == 0 {
                    1i64 << 60
                } else {
                    -(1i64 << 60)
                }
            })
            .collect();
        assert_eq!(linear.project(&x).expect("project"), sequential(&p, &x));
        let refs = [x.as_slice(), x.as_slice()];
        let want = vec![
            linear.project(&x).expect("per-column a"),
            linear.project(&x).expect("per-column b"),
        ];
        assert_eq!(linear.project_pooled(&refs).expect("pooled"), want);
    }

    #[test]
    fn wrong_width_and_bad_scale_refuse() {
        let mut p = packed(2);
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        assert!(matches!(
            linear.project(&[0; 28]),
            Err(NqError::Shape { x: 28, n_in: 29 })
        ));
        p.s_pre[0] = 0x7f80; // +inf
        assert!(matches!(
            NqLinear::from_packed(p),
            Err(NqError::Scale { .. })
        ));
    }

    #[test]
    fn a_single_sign_flip_moves_the_output() {
        let p = packed(3);
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        let x: Vec<i64> = (0..p.inn).map(|i| (i as i64 + 1) << 26).collect();
        let a = linear.project(&x).expect("a");
        let mut q = p;
        q.u[0] ^= 1;
        let b = NqLinear::from_packed(q)
            .expect("decode")
            .project(&x)
            .expect("b");
        assert_ne!(a, b);
    }

    #[test]
    fn a_residual_sign_flip_moves_the_output() {
        let p = packed(6);
        assert_eq!(p.res_layers, 1);
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        let x: Vec<i64> = (0..p.inn).map(|i| (i as i64 + 1) << 26).collect();
        let a = linear.project(&x).expect("a");
        let mut q = p;
        q.res[0] ^= 1;
        let b = NqLinear::from_packed(q)
            .expect("decode")
            .project(&x)
            .expect("b");
        assert_ne!(a, b);
    }

    /// The vector source and the scalar macro must agree at every boundary: empty, below one
    /// word, one word, one word plus a bit, several words, and a word count that leaves a
    /// partial tail. The padding bits of the tail are then set and both must still agree.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_vector_source_equals_the_scalar_source() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("popcnt")
        {
            return;
        }
        let mut state = 0x5eed_0f17_1234_9876u64;
        for len in [0usize, 1, 31, 32, 33, 64, 127, 128, 1000] {
            let t: Vec<i64> = (0..len)
                .map(|_| i64::from(next_u32(&mut state) as i32))
                .collect();
            let mut words: Vec<u32> = (0..len.div_ceil(32))
                .map(|_| next_u32(&mut state))
                .collect();
            let scalar = sign_dot!(&t, &words, len, i64);
            // SAFETY: AVX2 is detected above and `t.len() == len`.
            let vector = unsafe { sign_dot_avx2(&t, &words, len) };
            assert_eq!(scalar, vector, "len {len}");
            if len % 32 != 0 {
                *words.last_mut().expect("a partial word") |= !0u32 << (len % 32);
            }
            let padded_scalar = sign_dot!(&t, &words, len, i64);
            // SAFETY: as above.
            let padded_vector = unsafe { sign_dot_avx2(&t, &words, len) };
            assert_eq!(
                scalar, padded_scalar,
                "len {len}: padding moved the scalar sum"
            );
            assert_eq!(
                padded_scalar, padded_vector,
                "len {len}: padding moved only the vector sum"
            );
        }
    }

    /// The byte-plane source's safe twin: the same four signed digits and the same
    /// `Σ t − 2·Σ_{bit=1} t`, folded term by term in `i64` without an intrinsic. The vector
    /// source's `i32` lane grouping is exact under `lane_sums_fit`, so the two fold one sum.
    fn sign_dot_planes_twin(t: &[i64], words: &[u32], len: usize) -> i64 {
        let planes = BytePlanes::of(&t[..len]).expect("four digits");
        let mut sum = 0i64;
        let mut selected = 0i64;
        let mut i = 0usize;
        while i < len {
            let mut term = 0i64;
            let mut b = 0usize;
            while b < 4 {
                term += i64::from(planes.plane[b][i] as i8) << (8 * b);
                b += 1;
            }
            assert_eq!(term, t[i], "digits of term {i}");
            sum += term;
            selected += term * i64::from((words[i / 32] >> (i % 32)) & 1);
            i += 1;
        }
        sum - 2 * selected
    }

    /// The byte-plane source, its safe twin and the scalar macro must agree at every boundary:
    /// empty, below one word, one word, one word plus a bit, several words, and a partial tail,
    /// at three term scales up to the four-digit edge. The tail word's padding bits are then set
    /// and all three must still agree.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_plane_source_equals_the_scalar_source() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x5eed_1234_0f17_abcd_u64;
        for len in [
            0usize, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 1000, 1023,
        ] {
            assert!(lane_sums_fit(len), "len {len}");
            for bound in [1i64 << 8, 1 << 20, 2_139_062_143] {
                let t: Vec<i64> = (0..len)
                    .map(|_| i64::from(next_u32(&mut state) as i32) % bound)
                    .collect();
                let mut words: Vec<u32> = (0..len.div_ceil(32))
                    .map(|_| next_u32(&mut state))
                    .collect();
                let planes = BytePlanes::of(&t).expect("four digits");
                let padded = |words: &[u32]| {
                    let scalar = sign_dot!(&t, words, len, i64);
                    let twin = sign_dot_planes_twin(&t, words, len);
                    let sum = t.iter().copied().sum();
                    // SAFETY: AVX2 is detected above, `t.len() == len`, `words` covers `len`
                    // bits, and `planes` splits every term into four digits.
                    let vector = unsafe { sign_dot_planes_avx2(&t, &planes, sum, words, len) };
                    assert_eq!(scalar, twin, "len {len} bound {bound}: twin");
                    assert_eq!(scalar, vector, "len {len} bound {bound}: plane source");
                };
                padded(&words);
                if len % 32 != 0 {
                    *words.last_mut().expect("a partial word") |= !0u32 << (len % 32);
                }
                padded(&words);
            }
        }
    }

    /// The Q8 reader's safe twin, its AVX2 body and the existing sign dot over the dequantized
    /// codes must agree at every boundary, at block scales from `2^0` to `2^42`. The tail word's
    /// padding bits are then set and all three must still agree.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_code_source_equals_the_scalar_source() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x5eed_c0de_0f17_2468u64;
        for len in [
            0usize, 1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 1023,
        ] {
            for e in [0u32, 1, 12, 40] {
                let blocks = len.div_ceil(32);
                let codes: Vec<i8> = (0..len)
                    .map(|_| {
                        i8::try_from(i32::try_from(next_u32(&mut state) % 255).expect("byte") - 127)
                            .expect("[-127, 127]")
                    })
                    .collect();
                // Adjacent blocks differ by four exponents, so the normalized shifts are
                // exercised on every length, not only the long ones.
                let scales: Vec<i64> = (0..blocks)
                    .map(|b| 1i64 << (e + u32::from(b % 2 == 0) * 2))
                    .collect();
                let mut words: Vec<u32> = (0..blocks).map(|_| next_u32(&mut state)).collect();
                let q = QuantAct::from_parts(scales.clone(), codes.clone());
                let terms = CodeTerms::new(&q).expect("the code plan admits the fixture");
                let run = |words: &[u32]| {
                    let selected = code_select_twin(&codes, &scales, words, len);
                    // The exact sign dot over the dequantized codes, and the activation total the
                    // selection is subtracted from: `Σ x̂ − 2·selected`.
                    let t: Vec<i64> = (0..len)
                        .map(|j| scales[j / 32] * i64::from(codes[j]))
                        .collect();
                    let exact = i128::from(sign_dot!(&t, words, len, i64));
                    let sum: i128 = t.iter().map(|&v| i128::from(v)).sum();
                    // SAFETY: AVX2 is detected above; `len == codes.len()`, `words` covers `len`
                    // bits, and `terms` was admitted by `CodeTerms::new`.
                    let vector = unsafe {
                        code_select_avx2(
                            &codes,
                            &scales,
                            &terms.shift_vecs,
                            terms.e_min,
                            words,
                            len,
                        )
                        .expect("the admitted plan is in range")
                    };
                    assert_eq!(
                        sum - 2 * selected,
                        exact,
                        "len {len} e {e}: selection vs the dequantized dot"
                    );
                    assert_eq!(
                        vector,
                        i64::try_from(selected).expect("fits"),
                        "len {len} e {e}: vector"
                    );
                };
                run(&words);
                if len % 32 != 0 {
                    *words.last_mut().expect("a partial word") |= !0u32 << (len % 32);
                }
                run(&words);
            }
        }
    }

    /// The Q8 projection runs off the exact reader and stays inside [`Q8_REL_BOUND`] on the
    /// fixture: the q8 route is visible (the deviation is nonzero) and a route mix-up is not.
    #[test]
    fn q8_projection_is_within_the_bound_of_exact() {
        let p = packed(11);
        let linear = NqLinear::from_packed(p.clone()).expect("decode");
        let x: Vec<i64> = (0..p.inn)
            .map(|i| ((i as i64 * 37 + 11) % 101 - 50) << 24)
            .collect();
        let exact = linear.project(&x).expect("exact");
        let q8 = linear.project_q8(&x).expect("q8");
        let mut max_dev = 0u64;
        let mut max_abs = 0u64;
        for (a, b) in exact.iter().zip(&q8) {
            max_dev = max_dev.max(a.abs_diff(*b));
            max_abs = max_abs.max(a.unsigned_abs());
        }
        assert!(max_dev > 0, "the Q8 reader was bit-identical to exact");
        assert!(
            max_dev.saturating_mul(Q8_REL_BOUND) <= max_abs,
            "max deviation {max_dev} words against |y| {max_abs}, bound 1/{Q8_REL_BOUND}"
        );
    }

    /// The 64-term ternary twin against the term-by-term sum over the boundary lengths. `p` and
    /// `n` are disjoint and masked to `len`; `w` keeps arbitrary padding above `len`, which the
    /// masks make irrelevant. The lengths land on the word and block boundaries and on the
    /// multiword tail.
    #[test]
    fn the_ternary_block_equals_the_term_sum() {
        let mut state = 0x5eed_7e12_0f17_9abcu64;
        for len in [
            0usize, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 1023,
        ] {
            let blocks = len.div_ceil(64);
            let mut p = vec![0u64; blocks];
            let mut n = vec![0u64; blocks];
            let mut w = vec![0u64; blocks];
            for b in 0..blocks {
                for j in 0..64 {
                    match next_u32(&mut state) % 3 {
                        1 => p[b] |= 1u64 << j,
                        2 => n[b] |= 1u64 << j,
                        _ => {}
                    }
                }
                w[b] = u64::from(next_u32(&mut state)) | (u64::from(next_u32(&mut state)) << 32);
            }
            if len % 64 != 0 {
                let keep = (1u64 << (len % 64)) - 1;
                let last = blocks - 1;
                p[last] &= keep;
                n[last] &= keep;
            }
            let mut got = 0i128;
            let mut b = 0usize;
            while b < blocks {
                got += i128::from(ternary_block(w[b], p[b], n[b]));
                b += 1;
            }
            let mut want = 0i128;
            let mut i = 0usize;
            while i < len {
                let (b, j) = (i / 64, i % 64);
                let trit = i128::from((p[b] >> j) & 1) - i128::from((n[b] >> j) & 1);
                let sign = 1 - 2 * i128::from((w[b] >> j) & 1);
                want += sign * trit;
                i += 1;
            }
            assert_eq!(got, want, "len {len}");
        }
    }

    /// A trit-exact activation through the whole staged product: unit scales, every `V`/`U` row a
    /// single sign bit, one `x` word `+2^26` and the rest zero. Its ternary code is `±64` at
    /// `S = 2^20`, so the dropped residual is zero; every `V` output is that one word with a sign,
    /// so the second stage's activation is trit-exact too. Both readers must agree word for word.
    /// This checks the masks, the level shift, the block and tail boundaries, and the wiring of
    /// both dot stages.
    #[test]
    fn the_ternary_projection_equals_the_exact_reader_on_trit_exact_words() {
        let (out, inn, rank) = (8usize, 130usize, 9usize);
        let mut state = 0x5eed_7e12_0f17_c0deu64;
        let v_words = inn.div_ceil(32);
        let u_words = rank.div_ceil(32);
        let mut v = vec![0u32; rank * v_words];
        for k in 0..rank {
            let j = (next_u32(&mut state) as usize) % inn;
            v[k * v_words + j / 32] |= 1u32 << (j % 32);
        }
        let mut u = vec![0u32; out * u_words];
        for i in 0..out {
            let k = (next_u32(&mut state) as usize) % rank;
            u[i * u_words + k / 32] |= 1u32 << (k % 32);
        }
        let p = PackedLinear {
            layer: 0,
            kind: 0,
            out,
            inn,
            rank,
            u: u.into(),
            v: v.into(),
            s_pre: vec![bf16_bits(1.0); inn].into(),
            s_mid: vec![bf16_bits(1.0); rank].into(),
            s_post: vec![bf16_bits(1.0); out].into(),
            salient_idx: Vec::<u32>::new().into(),
            salient: Vec::<u16>::new().into(),
            res_layers: 0,
            shared: false,
            res: Vec::<u32>::new().into(),
            s_res: Vec::<u16>::new().into(),
        };
        let linear = NqLinear::from_packed(p).expect("decode");
        // One nonzero input word: every `V` row then reads `±a` (one term, so no residual), and
        // every entry of the second stage's activation `z` is one of those `V` outputs, so the
        // sign-only reader is exact there too. A dense `x` breaks the premise: stage 2's
        // activation is `Σx − 2x_j` (3–7 words here), whose ternary code carries a nonzero
        // residual that this reader deliberately drops.
        let a = 1i64 << 26;
        let mut x = vec![0i64; inn];
        x[(next_u32(&mut state) as usize) % inn] = a;
        let exact = linear.project(&x).expect("exact");
        let ternary = linear.project_ternary(&x).expect("ternary");
        assert!(
            exact.iter().any(|&v| v != 0),
            "the fixture must exercise a nonzero output"
        );
        assert_eq!(exact, ternary);
    }

    /// The four-digit split is exact up to `2,139,062,143` and down to `−2,155,905,152`, refuses
    /// the next value, and a refused split leaves the dispatch on the `i64`-lane source, which
    /// still equals the macro.
    #[test]
    fn the_four_digit_split_refuses_a_fifth_digit() {
        for v in [
            0i64,
            1,
            -1,
            127,
            128,
            255,
            256,
            -255,
            -256,
            -(1 << 24),
            1 << 30,
            -(1 << 30),
            2_139_062_143,
            -2_155_905_152,
            i64::from(i32::MIN),
        ] {
            let planes = BytePlanes::of(&[v]).expect("four digits");
            let mut term = 0i64;
            for (b, plane) in planes.plane.iter().enumerate() {
                term += i64::from(plane[0] as i8) << (8 * b);
            }
            assert_eq!(term, v, "digits of {v}");
        }
        for v in [
            2_139_062_144i64,
            -2_155_905_153,
            i64::from(i32::MAX),
            1 << 40,
        ] {
            assert!(BytePlanes::of(&[v]).is_none(), "{v} split into four digits");
        }
        let t = [
            2_139_062_144i64,
            -1,
            5,
            2_139_062_143,
            -2_155_905_152,
            7,
            1 << 30,
            -8,
        ];
        let words = [0b1010_1010u32];
        let terms = Terms::new(&t);
        assert!(terms.planes.is_none());
        assert_eq!(terms.sum, t.iter().copied().sum());
        assert_eq!(
            terms.dot(&words, t.len()),
            sign_dot!(&t, &words, t.len(), i64)
        );
    }

    /// Reports scalar and AVX2 nanoseconds for one `V` stage of a 4096-wide linear over three
    /// interleaved runs; the speedup and its spread are printed, not asserted.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "reports timing; run in release with --ignored --nocapture"]
    fn sign_dot_microbenchmark() {
        const RANK: usize = 1024;
        const INN: usize = 4096;
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("popcnt")
        {
            println!("avx2 or popcnt unavailable; there is no vector source to time");
            return;
        }
        let words_per_row = INN.div_ceil(32);
        let mut state = 0x00c0_ffee_0bad_f00du64;
        let t: Vec<i64> = (0..INN)
            .map(|_| i64::from(next_u32(&mut state) as i32))
            .collect();
        let planes: Vec<u32> = (0..RANK * words_per_row)
            .map(|_| next_u32(&mut state))
            .collect();
        let pass = |vector: bool| -> i64 {
            let mut acc = 0i64;
            let mut row = 0usize;
            while row < RANK {
                let w = &planes[row * words_per_row..(row + 1) * words_per_row];
                let dot = if vector {
                    // SAFETY: AVX2 is detected above and the plane and `t` cover `INN`.
                    unsafe { sign_dot_avx2(&t, w, INN) }
                } else {
                    sign_dot!(&t, w, INN, i64)
                };
                acc = acc.wrapping_add(dot);
                row += 1;
            }
            acc
        };
        // One untimed pass each, so both timed sources see warm caches.
        std::hint::black_box(pass(false));
        std::hint::black_box(pass(true));
        let mut scalar_ns = [0u128; 3];
        let mut vector_ns = [0u128; 3];
        for run in 0..3 {
            let start = std::time::Instant::now();
            std::hint::black_box(pass(false));
            scalar_ns[run] = start.elapsed().as_nanos();
            let start = std::time::Instant::now();
            std::hint::black_box(pass(true));
            vector_ns[run] = start.elapsed().as_nanos();
        }
        println!(
            "sign dot microbenchmark: rank {RANK} x inn {INN}, {words_per_row} words/row, {RANK} rows"
        );
        println!("run  scalar ns  avx2 ns  ratio");
        for run in 0..3 {
            println!(
                "{run}  {}  {}  {:.2}",
                scalar_ns[run],
                vector_ns[run],
                scalar_ns[run] as f64 / vector_ns[run] as f64
            );
        }
        let mean = |v: &[u128; 3]| v.iter().map(|&n| n as f64).sum::<f64>() / 3.0;
        let spread = |v: &[u128; 3]| {
            let min = *v.iter().min().expect("three runs");
            let max = *v.iter().max().expect("three runs");
            (max - min) as f64 / mean(v) * 100.0
        };
        println!(
            "mean  scalar {:.0} ns  avx2 {:.0} ns  ratio {:.2}",
            mean(&scalar_ns),
            mean(&vector_ns),
            mean(&scalar_ns) / mean(&vector_ns)
        );
        println!(
            "spread  scalar {:.1}%  avx2 {:.1}%",
            spread(&scalar_ns),
            spread(&vector_ns)
        );
    }

    /// The mean of three runs, in the runs' own unit.
    fn mean3(v: &[u128; 3]) -> f64 {
        v.iter().map(|&n| n as f64).sum::<f64>() / 3.0
    }

    /// The `(max − min)/mean` spread of three runs, in percent.
    fn spread3(v: &[u128; 3]) -> f64 {
        let min = *v.iter().min().expect("three runs");
        let max = *v.iter().max().expect("three runs");
        (max - min) as f64 / mean3(v) * 100.0
    }

    /// The 7B pack's four linear geometries, `(kind, out, inn, rank, per_token)`, and how many of
    /// each the 28 layers hold (`bench/cpu/1794781140-nq7b-118-full-pack.md`).
    const PRODUCTION: [(&str, usize, usize, usize, usize); 4] = [
        ("q,o", 3584, 3584, 2080, 56),
        ("k,v", 512, 3584, 512, 56),
        ("gate,up", 18944, 3584, 3520, 56),
        ("down", 3584, 18944, 3520, 28),
    ];

    /// A production-shaped packed linear: random sign planes, the staged test's scale pattern, no
    /// sparse column and no dense residual. The 1.174 b/w pack stores neither: its bit count is
    /// `32·(o·⌈r/32⌉ + r·⌈n/32⌉) + 16·(n + r + o)`, with no `salient` or `res` term.
    fn production_packed(out: usize, inn: usize, rank: usize, seed: u64) -> PackedLinear {
        let u_words = rank.div_ceil(32);
        let v_words = inn.div_ceil(32);
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ seed;
        let u: Vec<u32> = (0..out * u_words).map(|_| next_u32(&mut state)).collect();
        let v: Vec<u32> = (0..rank * v_words).map(|_| next_u32(&mut state)).collect();
        PackedLinear {
            layer: 0,
            kind: 0,
            out,
            inn,
            rank,
            u: u.into(),
            v: v.into(),
            s_pre: (0..inn)
                .map(|i| bf16_bits(0.01 + 0.001 * (i % 7) as f32))
                .collect::<Vec<u16>>()
                .into(),
            s_mid: (0..rank)
                .map(|i| bf16_bits(0.5 + 0.01 * (i % 3) as f32))
                .collect::<Vec<u16>>()
                .into(),
            s_post: (0..out)
                .map(|i| bf16_bits(0.02 + 0.002 * (i % 5) as f32))
                .collect::<Vec<u16>>()
                .into(),
            salient_idx: Vec::new().into(),
            salient: Vec::new().into(),
            res_layers: 0,
            shared: false,
            res: Vec::new().into(),
            s_res: Vec::new().into(),
        }
    }

    /// Reports the production geometries: the `project` cost in ms, sign terms/s and replaced
    /// weights/s over three runs, the term magnitudes and whether the byte source is admitted, and
    /// the paired `i64`-lane / byte-plane throughput with the arms alternating order. The per-token
    /// totals close with the one-core rate each arm implies. Prints; the only assertion is that
    /// the two arms agree.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "reports timing; run in release with --ignored --nocapture"]
    #[allow(
        clippy::too_many_lines,
        reason = "one reporting body: the geometry table, the paired arms and the closure totals \
                  share the fixture and are printed together"
    )]
    fn nq_production_shape_throughput() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("popcnt")
        {
            println!("avx2 or popcnt unavailable; there is no vector source to time");
            return;
        }
        println!(
            "nq production shapes; pool width {}",
            Pool::global().width()
        );
        let mut token_terms = 0f64;
        let mut token_weights = 0f64;
        let mut token_project_ns = 0f64;
        let mut token_lane_ns = 0f64;
        let mut token_byte_ns = 0f64;
        let mut token_code_ns = 0f64;
        let mut token_q8_ns = 0f64;
        for &(name, out, inn, rank, per_token) in &PRODUCTION {
            let seed = (out as u64) << 20 ^ (inn as u64) << 8 ^ rank as u64;
            let linear =
                NqLinear::from_packed(production_packed(out, inn, rank, seed)).expect("decode");
            let v_words = inn.div_ceil(32);
            let u_words = rank.div_ceil(32);
            // A Q32.32 activation of magnitude ~2^24, the staged test's pattern.
            let x: Vec<i64> = (0..inn)
                .map(|i| ((i as i64 * 37 + 11) % 101 - 50) << 24)
                .collect();
            let t: Vec<i64> = (0..inn)
                .map(|j| mul_q32(x[j], linear.s_pre[j]).expect("t"))
                .collect();
            let t_terms = Terms::new(&t);
            let z: Vec<i64> = (0..rank)
                .map(|k| {
                    let w = &linear.v[k * v_words..(k + 1) * v_words];
                    mul_q32(t_terms.dot(w, inn), linear.s_mid[k]).expect("z")
                })
                .collect();
            let z_terms = Terms::new(&z);
            let t_q = quant_act(&t);
            let z_q = quant_act(&z);
            let t_code = t_q.as_ref().and_then(CodeTerms::new).expect("t code plan");
            let z_code = z_q.as_ref().and_then(CodeTerms::new).expect("z code plan");
            let max_t = t.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
            let max_z = z.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
            let terms = (rank * inn + out * rank) as f64;
            let weights = (out * inn) as f64;
            // One full `V` + `U` sign-dot pass through one arm, both over the same planes.
            let pass = |byte: bool| -> i64 {
                let mut acc = 0i64;
                let mut row = 0usize;
                while row < rank {
                    let w = &linear.v[row * v_words..(row + 1) * v_words];
                    let dot = if byte {
                        // SAFETY: AVX2 is detected above; `t` and its planes cover `inn` terms.
                        unsafe {
                            sign_dot_planes_avx2(
                                &t,
                                t_terms.planes.as_ref().expect("t planes"),
                                t_terms.sum,
                                w,
                                inn,
                            )
                        }
                    } else {
                        // SAFETY: as above, for the `i64`-lane source.
                        unsafe { sign_dot_avx2(&t, w, inn) }
                    };
                    acc = acc.wrapping_add(dot);
                    row += 1;
                }
                let mut row = 0usize;
                while row < out {
                    let w = &linear.u[row * u_words..(row + 1) * u_words];
                    let dot = if byte {
                        // SAFETY: as above, over `rank` terms.
                        unsafe {
                            sign_dot_planes_avx2(
                                &z,
                                z_terms.planes.as_ref().expect("z planes"),
                                z_terms.sum,
                                w,
                                rank,
                            )
                        }
                    } else {
                        // SAFETY: as above.
                        unsafe { sign_dot_avx2(&z, w, rank) }
                    };
                    acc = acc.wrapping_add(dot);
                    row += 1;
                }
                acc
            };
            std::hint::black_box(pass(false));
            std::hint::black_box(pass(true));
            let pass_codes = || -> i64 {
                let mut acc = 0i64;
                let mut row = 0usize;
                while row < rank {
                    let w = &linear.v[row * v_words..(row + 1) * v_words];
                    acc = acc.wrapping_add(t_code.selected(w, inn).expect("t selection"));
                    row += 1;
                }
                let mut row = 0usize;
                while row < out {
                    let w = &linear.u[row * u_words..(row + 1) * u_words];
                    acc = acc.wrapping_add(z_code.selected(w, rank).expect("z selection"));
                    row += 1;
                }
                acc
            };
            std::hint::black_box(pass_codes());
            let mut lane_ns = [0u128; 3];
            let mut byte_ns = [0u128; 3];
            for run in 0..3 {
                // Alternate the arm order: even runs time the `i64` lanes first.
                let first = run % 2 == 0;
                let start = std::time::Instant::now();
                let a = std::hint::black_box(pass(!first));
                let mid = start.elapsed().as_nanos();
                let b = std::hint::black_box(pass(first));
                let end = start.elapsed().as_nanos();
                assert_eq!(a, b, "{name}: the two arms disagree");
                let (lane, byte) = if first {
                    (mid, end - mid)
                } else {
                    (end - mid, mid)
                };
                lane_ns[run] = lane;
                byte_ns[run] = byte;
            }
            let mut code_ns = [0u128; 3];
            for ns in &mut code_ns {
                let start = std::time::Instant::now();
                std::hint::black_box(pass_codes());
                *ns = start.elapsed().as_nanos();
            }
            let mut q8_ns = [0u128; 3];
            for ns in &mut q8_ns {
                let start = std::time::Instant::now();
                std::hint::black_box(linear.project_q8(&x).expect("project_q8"));
                *ns = start.elapsed().as_nanos();
            }
            let exact_out = linear.project(&x).expect("exact");
            let q8_out = linear.project_q8(&x).expect("q8");
            let mut max_dev = 0u64;
            let mut max_abs = 0u64;
            for (a, b) in exact_out.iter().zip(&q8_out) {
                max_dev = max_dev.max(a.abs_diff(*b));
                max_abs = max_abs.max(a.unsigned_abs());
            }
            let mut project_ns = [0u128; 3];
            for ns in &mut project_ns {
                let start = std::time::Instant::now();
                std::hint::black_box(linear.project(&x).expect("project"));
                *ns = start.elapsed().as_nanos();
            }
            println!(
                "{name:8} {out:5}x{inn:5} r{rank:4}: project {:8.2} ms (spread {:4.1}%)  {:6.2} G terms/s  {:6.2} G weight/s",
                mean3(&project_ns) / 1e6,
                spread3(&project_ns),
                terms / mean3(&project_ns),
                weights / mean3(&project_ns),
            );
            println!(
                "         max|t| {max_t} (2^{:.1}) byte {}  max|z| {max_z} (2^{:.1}) byte {}  four-digit z headroom {:.2}x  | lanes {:8.2} ms {:6.2} G terms/s | bytes {:8.2} ms {:6.2} G terms/s | ratio {:.2} (spread {:4.1}% / {:4.1}%)",
                (max_t as f64).log2(),
                t_terms.planes.is_some(),
                (max_z as f64).log2(),
                z_terms.planes.is_some(),
                2_139_062_143.0 / max_z.max(1) as f64,
                mean3(&lane_ns) / 1e6,
                terms / mean3(&lane_ns),
                mean3(&byte_ns) / 1e6,
                terms / mean3(&byte_ns),
                mean3(&lane_ns) / mean3(&byte_ns),
                spread3(&lane_ns),
                spread3(&byte_ns),
            );
            println!(
                "         q8 codes {:8.2} ms {:6.2} G terms/s | project_q8 {:8.2} ms | byte/q8 {:.2}x | deviation {max_dev} words of |y| {max_abs}",
                mean3(&code_ns) / 1e6,
                terms / mean3(&code_ns),
                mean3(&q8_ns) / 1e6,
                mean3(&byte_ns) / mean3(&code_ns),
            );
            let per_token = per_token as f64;
            token_terms += terms * per_token;
            token_weights += weights * per_token;
            token_project_ns += mean3(&project_ns) * per_token;
            token_lane_ns += mean3(&lane_ns) * per_token;
            token_byte_ns += mean3(&byte_ns) * per_token;
            token_code_ns += mean3(&code_ns) * per_token;
            token_q8_ns += mean3(&q8_ns) * per_token;
        }
        println!(
            "\nper token: {:.3} G sign terms, {:.3} G replaced weights",
            token_terms / 1e9,
            token_weights / 1e9
        );
        for (arm, ns, one_core) in [
            ("project, pool", token_project_ns, false),
            ("project_q8", token_q8_ns, false),
            ("i64 lanes", token_lane_ns, true),
            ("byte planes", token_byte_ns, true),
            ("q8 codes", token_code_ns, true),
        ] {
            println!(
                "{arm:14} {:8.2} ms/token  {:6.2} G terms/s  {:6.2} G weights/s  {:.2} tok/s{}",
                ns / 1e6,
                token_terms / ns,
                token_weights / ns,
                1e9 / ns,
                if one_core {
                    format!(" on one core ({:.2} tok/s at four cores)", 4e9 / ns)
                } else {
                    format!(" as run (pool width {})", Pool::global().width())
                },
            );
        }
    }

    /// The reference's own served factors through the served kernel.
    ///
    /// `APEIRON_NQ_OP_DIR` is `export_op.py`'s directory: `dims.txt`, `U.i8`, `V.i8`,
    /// `s_pre.u16`, `s_mid.u16`, `s_post.u16`, `x.f32`, `y_f64.f64`, `y_torch.f32`. The assertion
    /// is against the exact product over the reference's bf16 words, so it isolates this kernel
    /// from the reference's bf16 matmul; `y_torch`'s gap is printed for the bf16 part.
    #[test]
    #[ignore = "reads the reference trace at $APEIRON_NQ_OP_DIR; run with --ignored"]
    fn the_reference_factors_product_exactly() {
        let dir = std::env::var("APEIRON_NQ_OP_DIR").expect("APEIRON_NQ_OP_DIR");
        let dims = std::fs::read_to_string(format!("{dir}/dims.txt")).expect("dims.txt");
        let d: Vec<usize> = dims
            .split_whitespace()
            .map(|s| s.parse().expect("dimension"))
            .collect();
        let (out, inn, rank) = (d[0], d[1], d[2]);
        let read = |name: &str| std::fs::read(format!("{dir}/{name}")).expect(name);
        let u8s = read("U.i8");
        let v8s = read("V.i8");
        let u16s = |b: &[u8]| -> Vec<u16> {
            b.as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect()
        };
        let f32s = |b: &[u8]| -> Vec<f32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let f64s = |b: &[u8]| -> Vec<f64> {
            b.as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect()
        };
        let (u_words, v_words) = (rank.div_ceil(32), inn.div_ceil(32));
        assert_eq!(u8s.len(), out * rank, "U bytes");
        assert_eq!(v8s.len(), rank * inn, "V bytes");
        let mut up = vec![0u32; out * u_words];
        for i in 0..out {
            for k in 0..rank {
                if (u8s[i * rank + k] as i8) > 0 {
                    up[i * u_words + k / 32] |= 1 << (k % 32);
                }
            }
        }
        let mut vp = vec![0u32; rank * v_words];
        for k in 0..rank {
            for j in 0..inn {
                if (v8s[k * inn + j] as i8) > 0 {
                    vp[k * v_words + j / 32] |= 1 << (j % 32);
                }
            }
        }
        let p = PackedLinear {
            layer: 0,
            kind: 0,
            out,
            inn,
            rank,
            u: up.into(),
            v: vp.into(),
            s_pre: u16s(&read("s_pre.u16")).into(),
            s_mid: u16s(&read("s_mid.u16")).into(),
            s_post: u16s(&read("s_post.u16")).into(),
            salient_idx: Vec::new().into(),
            salient: Vec::new().into(),
            res_layers: 0,
            shared: false,
            res: Vec::new().into(),
            s_res: Vec::new().into(),
        };
        let lin = NqLinear::from_packed(p).expect("decode reference factors");
        let xf = f32s(&read("x.f32"));
        assert_eq!(xf.len(), inn, "x bytes");
        let x: Vec<i64> = xf
            .iter()
            .map(|&v| (f64::from(v) * 4_294_967_296.0).round() as i64)
            .collect();
        let y = lin.project(&x).expect("project");
        let y64 = f64s(&read("y_f64.f64"));
        let yt = f32s(&read("y_torch.f32"));
        assert_eq!(y.len(), out, "output rows");
        let (mut max_exact, mut max_torch, mut max_plus) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..out {
            let yr = y[i] as f64 / 4_294_967_296.0;
            max_exact = max_exact.max((yr - y64[i]).abs());
            max_torch = max_torch.max((f64::from(yt[i]) - y64[i]).abs());
            max_plus = max_plus.max((yr + y64[i]).abs());
        }
        println!(
            "reference operator: max|rust-f64| {max_exact:.3e}  max|torch-f64| {max_torch:.3e}  max|rust+f64| {max_plus:.3e}"
        );
        for i in 0..4 {
            let yr = y[i] as f64 / 4_294_967_296.0;
            println!(
                "  y[{i}] rust {yr:.8e} f64 {:.8e} torch {:.8e}",
                y64[i], yt[i]
            );
        }
        assert!(
            max_exact < 1e-6,
            "the kernel misses the reference product by {max_exact:.3e}"
        );
    }
}
