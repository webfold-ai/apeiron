//! The `NanoQuant` sign-rank pack (`APNQPK01`): the binary low-rank factors of every served linear,
//! stored at their rate.
//!
//! A fitted linear is `Ŵ = diag(s_post)·U·diag(s_mid)·V·diag(s_pre)`, `U ∈ {±1}^{out×rank}`,
//! `V ∈ {±1}^{rank×in}`, three bfloat16 scale vectors, and a sparse correction
//! `S[o, idx[j]] = salient[o, j]` with binary16 (f16) values (`SamsungLabs/NanoQuant` @ `a9e0a43`,
//! `src/nanoquant/core/compress_model.py`; the `s_mid` and salient terms are the newer export
//! this engine serves). Version 3 adds `N` dense sign-residual layers,
//! `Ŵ + Σ_l diag(s_res_l)·sign_l`, each `sign_l ∈ {±1}^{out×in}` with one bfloat16 row scale.
//! This module stores exactly the bits the rate counts:
//! `32·(|U| + |V| + |idx| + |res|) + 16·(|s_pre| + |s_mid| + |s_post| + |salient| + |s_res|)` per
//! linear, with every sign plane bit-packed in little-endian `u32` words. A version-4 `k`/`v`
//! section that shares its layer's `q` block omits `V`, `s_pre`, `s_mid` and `idx`, which the three
//! linears hold byte-identically because they read the same input; those bytes are counted once.
//! The section says nothing about how the factors were fitted; it is the format the exporter writes
//! and the engine reads.
//!
//! **Layout.** An eight-byte magic, a `u32` version, a `u32` header length, the layer count, the
//! kind count, the linear count and one reserved word (32 bytes), then one section per fitted
//! linear in layer-major, kind order (`q`, `k`, `v`, `o`, `gate`, `up`, `down`). A pack may carry a
//! subset of the `layers · kinds` linears; an omitted one keeps its dense skeleton payload. Each
//! section is the layer, the kind, the
//! `out`/`in`/`rank` dimensions, the two packed row strides, the salient count, the `U` words, the
//! `V` words, the `s_pre`/`s_mid`/`s_post` bfloat16 words, the `salient_idx` `u32` words and the
//! `salient` binary16 (f16) words. Version 3 inserts the residual layer count after the salient count
//! and appends the residual sign words and row scales. Version 4 inserts a `group` word after the
//! residual count; a `group = 1` section is a `k` or `v` member and carries only `U`, `s_post`,
//! `salient`, the residual and its scales, binding `V`/`s_pre`/`s_mid`/`idx` to the head `q`
//! section of the same layer. A pack whose sections carry no residual is
//! written as version 2, which a reader that predates the residual accepts; any residual writes
//! version 3, and any sharing writes version 4. Every dimension, word count and byte extent is
//! checked; a section that leaves the buffer, repeats a linear, or disagrees with its own stride
//! refuses, and so does a member without a matching head.
//!
//! **Round trip.** [`write`] emits the bytes [`read`] parses; neither rounds a value. The factors
//! and the scale words are copied verbatim, so a write-read cycle is byte-identical.
//!
//! **One reader, two 16-bit decoders.** This module is the only pack reader. The kernel
//! `crate::engine::kern::nq::NqLinear` decodes the three scale vectors and the residual row scales
//! with the engine's existing `bf16_to_fixed` and the sparse residual with `f16_to_fixed`, the
//! same conversions the `BF16` and `F16` GGUF rows use. The scales are bfloat16 (`nq_scale_*`,
//! GGUF type 30), not binary16: the smallest is 2.47e-6, below binary16's smallest normal
//! (6.1e-5), where binary16 would keep about five significand bits. Both formats are 16 bits, so
//! this costs no storage and no second decode path.

use core::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// The pack magic: `APNQPK01`.
pub const NQ_MAGIC: [u8; 8] = *b"APNQPK01";
/// The pack version this module writes when a section shares its `V`/`s_pre`/`s_mid`/`idx` block
/// with a preceding `q` section.
pub const NQ_VERSION: u32 = 4;
/// The residual version without sharing.
const NQ_VERSION_RESIDUAL: u32 = 3;
/// The version without a dense sign residual.
const NQ_VERSION_BASE: u32 = 2;
/// The header length in bytes: magic, version, header length, layers, kinds, linears, reserved.
pub const NQ_HEADER_BYTES: usize = 32;
/// The per-linear section header in bytes: layer, kind, out, in, rank, `u_words`, `v_words`, `k`,
/// and in version 3 `res_layers`, and in version 4 `res_layers` and `group`.
pub const NQ_LINEAR_HEADER_BYTES: usize = 40;
/// The linear kinds, in the writer's order and the engine's directory-part order.
pub const NQ_KIND_NAMES: [&str; 7] = ["q", "k", "v", "o", "gate", "up", "down"];
/// The number of linear kinds in a section.
pub const NQ_KINDS: usize = 7;

/// A refused pack read or write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NqError {
    /// The buffer is shorter than the header or than a section it declares.
    Truncated {
        /// Bytes needed.
        need: usize,
        /// Bytes supplied.
        have: usize,
    },
    /// The magic is not [`NQ_MAGIC`].
    BadMagic,
    /// The version is not 2, 3 or 4.
    BadVersion {
        /// Version found.
        version: u32,
    },
    /// The declared header length is not [`NQ_HEADER_BYTES`].
    BadHeaderLength {
        /// Length found.
        bytes: u32,
    },
    /// The kind count is not [`NQ_KINDS`].
    BadKinds {
        /// Count found.
        kinds: usize,
    },
    /// The linear count is more than `layers · kinds`.
    BadCount {
        /// Count found.
        linears: usize,
        /// Most sections the layer and kind counts allow.
        expected: usize,
    },
    /// A layer or kind field is out of range or not after the previous section.
    BadOrder {
        /// Section index.
        at: usize,
        /// Layer found.
        layer: usize,
        /// Kind found.
        kind: usize,
    },
    /// A version-4 group word is neither 0 (a head or a standalone section) nor 1 (a member).
    BadGroup {
        /// Group word found.
        group: u32,
    },
    /// A member section has no `q` head in its layer, or that head's dimensions or shared block
    /// disagree with the member's.
    SharedHead {
        /// Section index.
        at: usize,
        /// Layer of the member.
        layer: usize,
        /// Kind of the member.
        kind: usize,
    },
    /// A dimension or rank is zero.
    ZeroDimension,
    /// A packed row stride is not `ceil(column count / 32)`.
    BadStride {
        /// Which of `U`/`V`.
        which: &'static str,
        /// Stride found.
        words: usize,
        /// Stride the column count requires.
        expected: usize,
    },
    /// A field vector's length disagrees with the dimensions it declares.
    Length {
        /// Which field.
        field: &'static str,
        /// Length found.
        have: usize,
        /// Length the dimensions require.
        expected: usize,
    },
    /// A salient index is outside the input width or repeats an earlier index.
    SalientIndex {
        /// Index found.
        index: usize,
        /// Input width.
        inn: usize,
    },
    /// Bytes remain after the last declared section.
    Trailing {
        /// Bytes remaining.
        bytes: usize,
    },
    /// A dimension, stride or byte extent leaves `usize`.
    LengthOverflow,
}

impl fmt::Display for NqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NqError::Truncated { need, have } => {
                write!(f, "nq pack: needs {need} bytes, has {have}")
            }
            NqError::BadMagic => write!(f, "nq pack: bad magic"),
            NqError::BadVersion { version } => write!(f, "nq pack: version {version}"),
            NqError::BadHeaderLength { bytes } => write!(f, "nq pack: header length {bytes}"),
            NqError::BadKinds { kinds } => write!(f, "nq pack: {kinds} kinds, want {NQ_KINDS}"),
            NqError::BadCount { linears, expected } => {
                write!(f, "nq pack: {linears} linears, at most {expected}")
            }
            NqError::BadOrder { at, layer, kind } => {
                write!(f, "nq pack: section {at} is blk.{layer} kind {kind}")
            }
            NqError::BadGroup { group } => write!(f, "nq pack: group {group}, want 0 or 1"),
            NqError::SharedHead { at, layer, kind } => write!(
                f,
                "nq pack: section {at} is blk.{layer} kind {kind} without a matching q block"
            ),
            NqError::ZeroDimension => write!(f, "nq pack: zero dimension or rank"),
            NqError::BadStride {
                which,
                words,
                expected,
            } => write!(f, "nq pack: {which} stride {words}, want {expected}"),
            NqError::Length {
                field,
                have,
                expected,
            } => write!(f, "nq pack: {field} has {have}, want {expected}"),
            NqError::SalientIndex { index, inn } => {
                write!(f, "nq pack: salient index {index} outside {inn}")
            }
            NqError::Trailing { bytes } => write!(f, "nq pack: {bytes} trailing bytes"),
            NqError::LengthOverflow => write!(f, "nq pack: length overflow"),
        }
    }
}

impl std::error::Error for NqError {}

/// A stored plane: owned for a pack parsed from a byte buffer, or a slice of the read-only file
/// mapping for a pack opened from a path.
///
/// The mapped variant is the serving path: a v3 pack is 2.7 GB of sign words, and copying those
/// words into `Vec`s makes the process's anonymous set the size of the pack. Keeping them in the
/// mapping leaves them file-backed, so the kernel can evict and re-read them under pressure; the
/// anonymous set is the skeleton plus the decoded scales. The arena is immutable, so every shared
/// slice is read-only.
pub enum Plane<T> {
    /// A copy owned by the pack.
    Owned(Vec<T>),
    /// A slice of a shared, read-only mapping at `off` with `len` elements.
    #[cfg(target_os = "linux")]
    Mapped {
        /// The mapping the slice points into.
        map: Arc<MappedFile>,
        /// Byte offset of the first element.
        off: usize,
        /// Element count.
        len: usize,
        /// The element type, without owning one.
        elem: core::marker::PhantomData<T>,
    },
}

impl<T> Plane<T> {
    /// A mapped plane; `off` must be aligned for `T` and `off + len·size_of::<T>()` must be
    /// inside `map`.
    #[cfg(target_os = "linux")]
    fn mapped(map: &Arc<MappedFile>, off: usize, len: usize) -> Self {
        debug_assert_eq!(off % core::mem::align_of::<T>(), 0);
        Self::Mapped {
            map: Arc::clone(map),
            off,
            len,
            elem: core::marker::PhantomData,
        }
    }

    /// Drop every element of an owned plane; refuses a read-only mapped plane.
    ///
    /// # Panics
    ///
    /// Panics when the plane is mapped: a mapped plane is read-only and cannot be cleared. Only
    /// the owned path (a pack parsed from a byte buffer) calls this.
    pub fn clear(&mut self) {
        match self {
            Self::Owned(v) => v.clear(),
            #[cfg(target_os = "linux")]
            Self::Mapped { .. } => panic!("a mapped pack plane is read-only"),
        }
    }
}

impl<T> From<Vec<T>> for Plane<T> {
    fn from(v: Vec<T>) -> Self {
        Self::Owned(v)
    }
}

impl<T> Deref for Plane<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            Self::Owned(v) => v,
            #[cfg(target_os = "linux")]
            Self::Mapped { map, off, len, .. } => {
                // SAFETY: `mapped` checked the offset's alignment and the caller checked the
                // extent against `map.len()` before constructing it; the mapping is immutable,
                // and the returned slice is bounded by `&self`.
                unsafe {
                    core::slice::from_raw_parts(map.ptr.cast::<u8>().add(*off).cast::<T>(), *len)
                }
            }
        }
    }
}

impl<T> DerefMut for Plane<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Self::Owned(v) => v,
            #[cfg(target_os = "linux")]
            Self::Mapped { .. } => panic!("a mapped pack plane is read-only"),
        }
    }
}

impl<T: Clone> Clone for Plane<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Owned(v) => Self::Owned(v.clone()),
            #[cfg(target_os = "linux")]
            Self::Mapped { map, off, len, .. } => Self::Mapped {
                map: Arc::clone(map),
                off: *off,
                len: *len,
                elem: core::marker::PhantomData,
            },
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Plane<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Owned(v) => v.fmt(f),
            #[cfg(target_os = "linux")]
            Self::Mapped { len, .. } => write!(f, "Mapped({len} elements)"),
        }
    }
}

impl<T: PartialEq> PartialEq for Plane<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for Plane<T> {}

/// One linear's stored factors: the packed signs and the 16-bit scale and residual words,
/// verbatim (bfloat16 scales, binary16 salient residuals).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedLinear {
    /// Block index.
    pub layer: usize,
    /// Kind index in [`NQ_KIND_NAMES`].
    pub kind: usize,
    /// Output rows.
    pub out: usize,
    /// Input columns.
    pub inn: usize,
    /// Rank.
    pub rank: usize,
    /// `U`, row-major `[out, ceil(rank/32)]`, bit `1` is `+1`.
    pub u: Plane<u32>,
    /// `V`, row-major `[rank, ceil(inn/32)]`, bit `1` is `+1`.
    pub v: Plane<u32>,
    /// `s_pre`, bfloat16 bits, length `inn`.
    pub s_pre: Plane<u16>,
    /// `s_mid`, bfloat16 bits, length `rank`.
    pub s_mid: Plane<u16>,
    /// `s_post`, bfloat16 bits, length `out`.
    pub s_post: Plane<u16>,
    /// The salient input columns, length `k`; each below `inn`.
    pub salient_idx: Plane<u32>,
    /// The salient residuals, binary16 (f16) bits, row-major `[out, k]`.
    pub salient: Plane<u16>,
    /// The dense sign-residual layers; 0 for a version-2 section.
    pub res_layers: usize,
    /// Version-4 member: `V`, `s_pre`, `s_mid` and `salient_idx` are not stored in this section and
    /// are read from the same layer's `q` section. The in-memory value carries them either way, so
    /// the kernel sees a complete linear; [`PackedLinear::stored_bits`] counts only this section's
    /// own bytes.
    pub shared: bool,
    /// The residual signs, row-major `[res_layers, out, ceil(inn/32)]`, bit `1` is `+1`.
    pub res: Plane<u32>,
    /// The residual row scales, bfloat16 bits, row-major `[res_layers, out]`.
    pub s_res: Plane<u16>,
}

impl PackedLinear {
    /// Words per `U` row: `ceil(rank / 32)`.
    #[must_use]
    pub fn u_words(&self) -> usize {
        self.rank.div_ceil(32)
    }

    /// Words per `V` row: `ceil(inn / 32)`.
    #[must_use]
    pub fn v_words(&self) -> usize {
        self.inn.div_ceil(32)
    }

    /// The salient-column count.
    #[must_use]
    pub fn k(&self) -> usize {
        self.salient_idx.len()
    }

    /// Words per residual row: `ceil(inn / 32)`.
    #[must_use]
    pub fn res_words(&self) -> usize {
        self.inn.div_ceil(32)
    }

    /// The stored bits: `32·(|U| + |V| + |idx| + |res|) + 16·(|s_pre| + |s_mid| + |s_post| +
    /// |salient| + |s_res|)`, omitting the block a [`shared`](Self::shared) member takes from its
    /// layer's `q` section.
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

    /// The dense weights the linear replaces: `out · inn`.
    #[must_use]
    pub fn weights(&self) -> u64 {
        (self.out as u64).saturating_mul(self.inn as u64)
    }

    /// The storage rate in bits per weight, or `None` for a zero-weight linear.
    #[must_use]
    pub fn bits_per_weight(&self) -> Option<f64> {
        let weights = self.weights();
        if weights == 0 {
            return None;
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "the count is below 2^53 for every model this format serves"
        )]
        Some(self.stored_bits() as f64 / weights as f64)
    }
}

/// A parsed pack: the layer and kind counts and every linear in section order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NqPack {
    /// Block count the sections cover.
    pub layers: usize,
    /// Kind count; always [`NQ_KINDS`].
    pub kinds: usize,
    /// The sections in layer-major, kind order.
    pub linears: Vec<PackedLinear>,
}

/// Check that `linears` is strictly layer-major, kind order, without repeats.
///
/// A pack may carry fewer than `layers · kinds` sections: a linear it leaves out is served by the
/// dense skeleton payload, so a sparse pack is a mixed-precision model rather than a malformed one.
fn check_order(layers: usize, linears: &[PackedLinear]) -> Result<(), NqError> {
    if layers == 0 {
        return Err(NqError::ZeroDimension);
    }
    let most = layers
        .checked_mul(NQ_KINDS)
        .ok_or(NqError::LengthOverflow)?;
    if linears.len() > most {
        return Err(NqError::BadCount {
            linears: linears.len(),
            expected: most,
        });
    }
    let mut previous: Option<(usize, usize)> = None;
    for (at, linear) in linears.iter().enumerate() {
        let key = (linear.layer, linear.kind);
        if key.0 >= layers || key.1 >= NQ_KINDS || previous.is_some_and(|prev| key <= prev) {
            return Err(NqError::BadOrder {
                at,
                layer: key.0,
                kind: key.1,
            });
        }
        previous = Some(key);
    }
    Ok(())
}

/// Check every [`shared`](PackedLinear::shared) member against its layer's `q` head.
///
/// A member is only `k` or `v`; its layer must carry a `q` section, and that head's input width,
/// rank and shared block (`V`, `s_pre`, `s_mid`, `salient_idx`) must be the member's. The file
/// stores the block once, so a writer that dropped it from an unequal section would lose bytes.
///
/// # Complexity
///
/// `O(members · (linears + shared_bytes))`: each member scans the section list for its layer's `q`
/// and then compares a block of `4·rank·ceil(inn/32) + 2·inn + 2·rank + 4·k` bytes. Space is
/// `O(1)`.
fn check_shared(linears: &[PackedLinear]) -> Result<(), NqError> {
    for (at, l) in linears.iter().enumerate() {
        if !l.shared {
            continue;
        }
        let head = if l.kind == 1 || l.kind == 2 {
            linears.iter().find(|h| h.layer == l.layer && h.kind == 0)
        } else {
            None
        };
        let matched = head.is_some_and(|h| {
            h.inn == l.inn
                && h.rank == l.rank
                && h.v == l.v
                && h.s_pre == l.s_pre
                && h.s_mid == l.s_mid
                && h.salient_idx == l.salient_idx
        });
        if !matched {
            return Err(NqError::SharedHead {
                at,
                layer: l.layer,
                kind: l.kind,
            });
        }
    }
    Ok(())
}

impl NqPack {
    /// Build a pack from sections in the required order.
    ///
    /// A pack may carry a subset of `layers · kinds`; an omitted linear keeps its dense skeleton
    /// payload. Every section present must name a distinct linear, in layer-major kind order.
    ///
    /// # Errors
    ///
    /// [`NqError::ZeroDimension`] for a zero layer count, [`NqError::BadCount`] for more sections
    /// than `layers · kinds`, [`NqError::BadOrder`] for a section out of range or out of order, and
    /// [`NqError::SharedHead`] for a member whose layer has no matching `q` head.
    pub fn new(layers: usize, linears: Vec<PackedLinear>) -> Result<Self, NqError> {
        check_order(layers, &linears)?;
        check_shared(&linears)?;
        Ok(Self {
            layers,
            kinds: NQ_KINDS,
            linears,
        })
    }

    /// The stored bits of every section.
    #[must_use]
    pub fn stored_bits(&self) -> u64 {
        self.linears
            .iter()
            .fold(0u64, |acc, l| acc.saturating_add(l.stored_bits()))
    }

    /// The dense weights every section replaces.
    #[must_use]
    pub fn weights(&self) -> u64 {
        self.linears
            .iter()
            .fold(0u64, |acc, l| acc.saturating_add(l.weights()))
    }

    /// The storage rate in bits per weight, or `None` when the pack carries no weights.
    #[must_use]
    pub fn bits_per_weight(&self) -> Option<f64> {
        let weights = self.weights();
        if weights == 0 {
            return None;
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "the count is below 2^53 for every model this format serves"
        )]
        Some(self.stored_bits() as f64 / weights as f64)
    }

    /// The linear at `layer`, `kind`.
    #[must_use]
    pub fn linear(&self, layer: usize, kind: usize) -> Option<&PackedLinear> {
        if kind >= NQ_KINDS {
            return None;
        }
        self.linears
            .get(layer.checked_mul(NQ_KINDS)?.checked_add(kind)?)
    }
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32, NqError> {
    let end = at.checked_add(4).ok_or(NqError::LengthOverflow)?;
    let raw = bytes.get(at..end).ok_or(NqError::Truncated {
        need: end,
        have: bytes.len(),
    })?;
    Ok(u32::from_le_bytes(
        raw.try_into().map_err(|_| NqError::LengthOverflow)?,
    ))
}

fn read_u16(bytes: &[u8], at: usize) -> Result<u16, NqError> {
    let end = at.checked_add(2).ok_or(NqError::LengthOverflow)?;
    let raw = bytes.get(at..end).ok_or(NqError::Truncated {
        need: end,
        have: bytes.len(),
    })?;
    Ok(u16::from_le_bytes(
        raw.try_into().map_err(|_| NqError::LengthOverflow)?,
    ))
}

/// Where a parsed plane's elements come from: a copy of the buffer, or a slice of its mapping.
trait Planes {
    fn u32s(&self, at: usize, len: usize) -> Result<Plane<u32>, NqError>;
    fn u16s(&self, at: usize, len: usize) -> Result<Plane<u16>, NqError>;
}

/// The copy source: every plane is materialised in the pack's own `Vec`.
struct OwnedPlanes<'a>(&'a [u8]);

impl Planes for OwnedPlanes<'_> {
    fn u32s(&self, at: usize, len: usize) -> Result<Plane<u32>, NqError> {
        let mut v = Vec::with_capacity(len);
        let mut j = 0usize;
        while j < len {
            v.push(read_u32(self.0, at + 4 * j)?);
            j += 1;
        }
        Ok(Plane::Owned(v))
    }

    fn u16s(&self, at: usize, len: usize) -> Result<Plane<u16>, NqError> {
        let mut v = Vec::with_capacity(len);
        let mut j = 0usize;
        while j < len {
            v.push(read_u16(self.0, at + 2 * j)?);
            j += 1;
        }
        Ok(Plane::Owned(v))
    }
}

/// The mapping source: planes stay in the file-backed mapping and are not copied.
#[cfg(target_os = "linux")]
struct MappedPlanes<'a> {
    bytes: &'a [u8],
    map: &'a Arc<MappedFile>,
}

#[cfg(target_os = "linux")]
impl Planes for MappedPlanes<'_> {
    fn u32s(&self, at: usize, len: usize) -> Result<Plane<u32>, NqError> {
        if at % core::mem::align_of::<u32>() == 0 {
            Ok(Plane::mapped(self.map, at, len))
        } else {
            OwnedPlanes(self.bytes).u32s(at, len)
        }
    }

    fn u16s(&self, at: usize, len: usize) -> Result<Plane<u16>, NqError> {
        if at % core::mem::align_of::<u16>() == 0 {
            Ok(Plane::mapped(self.map, at, len))
        } else {
            OwnedPlanes(self.bytes).u16s(at, len)
        }
    }
}

/// Parse a pack from a byte buffer, copying every plane into the pack.
///
/// This is the path for a buffer the caller already owns (the wasm ABI); [`open`] keeps planes in
/// the file mapping instead.
///
/// # Errors
///
/// Every [`NqError`] the layout can raise: a short buffer, a bad magic/version/header/kind count, a
/// section out of order, a zero dimension or rank, a stride or vector length that disagrees with the
/// dimensions, trailing bytes, or a size that leaves `usize`.
pub fn read(bytes: &[u8]) -> Result<NqPack, NqError> {
    read_with(bytes, &OwnedPlanes(bytes))
}

/// Parse a pack, taking each plane from `planes`.
#[allow(
    clippy::too_many_lines,
    reason = "the one pack reader: header, directory and every section's checked extent in file order"
)]
fn read_with<P: Planes>(bytes: &[u8], planes: &P) -> Result<NqPack, NqError> {
    if bytes.len() < NQ_HEADER_BYTES {
        return Err(NqError::Truncated {
            need: NQ_HEADER_BYTES,
            have: bytes.len(),
        });
    }
    if bytes.get(0..8) != Some(NQ_MAGIC.as_slice()) {
        return Err(NqError::BadMagic);
    }
    let version = read_u32(bytes, 8)?;
    if version != NQ_VERSION && version != NQ_VERSION_RESIDUAL && version != NQ_VERSION_BASE {
        return Err(NqError::BadVersion { version });
    }
    let header_bytes = read_u32(bytes, 12)?;
    if header_bytes as usize != NQ_HEADER_BYTES {
        return Err(NqError::BadHeaderLength {
            bytes: header_bytes,
        });
    }
    let layers = read_u32(bytes, 16)? as usize;
    let kinds = read_u32(bytes, 20)? as usize;
    let linears = read_u32(bytes, 24)? as usize;
    let _reserved = read_u32(bytes, 28)?;
    if kinds != NQ_KINDS {
        return Err(NqError::BadKinds { kinds });
    }
    if layers == 0 {
        return Err(NqError::ZeroDimension);
    }
    let most = layers
        .checked_mul(NQ_KINDS)
        .ok_or(NqError::LengthOverflow)?;
    if linears > most {
        return Err(NqError::BadCount {
            linears,
            expected: most,
        });
    }
    // A per-linear section is at least its header, so a count whose sections cannot fit the buffer
    // is refused before any allocation. Version 2 has neither the residual word nor the group word;
    // version 3 adds the residual; version 4 adds the group.
    let section_header = match version {
        NQ_VERSION => NQ_LINEAR_HEADER_BYTES,
        NQ_VERSION_RESIDUAL => NQ_LINEAR_HEADER_BYTES - 4,
        _ => NQ_LINEAR_HEADER_BYTES - 8,
    };
    let need = linears
        .checked_mul(section_header)
        .ok_or(NqError::LengthOverflow)?;
    if need > bytes.len() {
        return Err(NqError::Truncated {
            need,
            have: bytes.len(),
        });
    }
    let mut out: Vec<PackedLinear> = Vec::with_capacity(linears);
    let mut at = NQ_HEADER_BYTES;
    let mut i = 0usize;
    while i < linears {
        let layer = read_u32(bytes, at)? as usize;
        at += 4;
        let kind = read_u32(bytes, at)? as usize;
        at += 4;
        let out_rows = read_u32(bytes, at)? as usize;
        at += 4;
        let inn = read_u32(bytes, at)? as usize;
        at += 4;
        let rank = read_u32(bytes, at)? as usize;
        at += 4;
        let u_words = read_u32(bytes, at)? as usize;
        at += 4;
        let v_words = read_u32(bytes, at)? as usize;
        at += 4;
        let k = read_u32(bytes, at)? as usize;
        at += 4;
        let res_layers = if version == NQ_VERSION_BASE {
            0
        } else {
            let value = read_u32(bytes, at)? as usize;
            at += 4;
            value
        };
        let group = if version == NQ_VERSION {
            let value = read_u32(bytes, at)?;
            at += 4;
            value
        } else {
            0
        };
        if group > 1 {
            return Err(NqError::BadGroup { group });
        }
        let shared = group == 1;
        if out_rows == 0 || inn == 0 || rank == 0 {
            return Err(NqError::ZeroDimension);
        }
        if k > inn {
            return Err(NqError::Length {
                field: "salient_idx",
                have: k,
                expected: inn,
            });
        }
        let want_u = rank.div_ceil(32);
        if u_words != want_u {
            return Err(NqError::BadStride {
                which: "U",
                words: u_words,
                expected: want_u,
            });
        }
        let want_v = inn.div_ceil(32);
        if v_words != want_v {
            return Err(NqError::BadStride {
                which: "V",
                words: v_words,
                expected: want_v,
            });
        }
        let u_len = out_rows
            .checked_mul(u_words)
            .ok_or(NqError::LengthOverflow)?;
        let v_len = rank.checked_mul(v_words).ok_or(NqError::LengthOverflow)?;
        let u_bytes = u_len.checked_mul(4).ok_or(NqError::LengthOverflow)?;
        let v_bytes = v_len.checked_mul(4).ok_or(NqError::LengthOverflow)?;
        let pre_bytes = inn.checked_mul(2).ok_or(NqError::LengthOverflow)?;
        let mid_bytes = rank.checked_mul(2).ok_or(NqError::LengthOverflow)?;
        let post_bytes = out_rows.checked_mul(2).ok_or(NqError::LengthOverflow)?;
        let idx_bytes = k.checked_mul(4).ok_or(NqError::LengthOverflow)?;
        let salient_len = out_rows.checked_mul(k).ok_or(NqError::LengthOverflow)?;
        let salient_bytes = salient_len.checked_mul(2).ok_or(NqError::LengthOverflow)?;
        let res_words = inn.div_ceil(32);
        let res_len = res_layers
            .checked_mul(out_rows)
            .and_then(|x| x.checked_mul(res_words))
            .ok_or(NqError::LengthOverflow)?;
        let s_res_len = res_layers
            .checked_mul(out_rows)
            .ok_or(NqError::LengthOverflow)?;
        let res_bytes = res_len.checked_mul(4).ok_or(NqError::LengthOverflow)?;
        let s_res_bytes = s_res_len.checked_mul(2).ok_or(NqError::LengthOverflow)?;
        // A member's header still declares the shared block's width and stride; bind them to the
        // layer's `q` section so the in-memory linear is complete and the writer can compare it.
        let shared_block = if shared {
            let Some(head) = out.iter().rev().find(|h| h.layer == layer && h.kind == 0) else {
                return Err(NqError::SharedHead { at: i, layer, kind });
            };
            if head.inn != inn || head.rank != rank || head.k() != k || head.v_words() != v_words {
                return Err(NqError::SharedHead { at: i, layer, kind });
            }
            Some((
                head.v.clone(),
                head.s_pre.clone(),
                head.s_mid.clone(),
                head.salient_idx.clone(),
            ))
        } else {
            None
        };
        // A member's payload is `U`, `s_post`, `salient` and its residual; the shared block is
        // absent from the file.
        let payload = if shared {
            u_bytes
                .checked_add(post_bytes)
                .and_then(|x| x.checked_add(salient_bytes))
                .and_then(|x| x.checked_add(res_bytes))
                .and_then(|x| x.checked_add(s_res_bytes))
        } else {
            u_bytes
                .checked_add(v_bytes)
                .and_then(|x| x.checked_add(pre_bytes))
                .and_then(|x| x.checked_add(mid_bytes))
                .and_then(|x| x.checked_add(post_bytes))
                .and_then(|x| x.checked_add(idx_bytes))
                .and_then(|x| x.checked_add(salient_bytes))
                .and_then(|x| x.checked_add(res_bytes))
                .and_then(|x| x.checked_add(s_res_bytes))
        };
        let end = at
            .checked_add(payload.ok_or(NqError::LengthOverflow)?)
            .ok_or(NqError::LengthOverflow)?;
        if end > bytes.len() {
            return Err(NqError::Truncated {
                need: end,
                have: bytes.len(),
            });
        }
        let u = planes.u32s(at, u_len)?;
        at += u_bytes;
        let (v, s_pre, s_mid) = if let Some((v, s_pre, s_mid, _)) = &shared_block {
            (v.clone(), s_pre.clone(), s_mid.clone())
        } else {
            let v = planes.u32s(at, v_len)?;
            at += v_bytes;
            let s_pre = planes.u16s(at, inn)?;
            at += pre_bytes;
            let s_mid = planes.u16s(at, rank)?;
            at += mid_bytes;
            (v, s_pre, s_mid)
        };
        let s_post = planes.u16s(at, out_rows)?;
        at += post_bytes;
        let (salient_idx, salient) = if let Some((_, _, _, idx)) = shared_block {
            let salient = planes.u16s(at, salient_len)?;
            at += salient_bytes;
            (idx, salient)
        } else {
            let idx = planes.u32s(at, k)?;
            at += idx_bytes;
            let salient = planes.u16s(at, salient_len)?;
            at += salient_bytes;
            (idx, salient)
        };
        let res = planes.u32s(at, res_len)?;
        at += res_bytes;
        let s_res = planes.u16s(at, s_res_len)?;
        at += s_res_bytes;
        out.push(PackedLinear {
            layer,
            kind,
            out: out_rows,
            inn,
            rank,
            u,
            v,
            s_pre,
            s_mid,
            s_post,
            salient_idx,
            salient,
            res_layers,
            shared,
            res,
            s_res,
        });
        i += 1;
    }
    if at != bytes.len() {
        return Err(NqError::Trailing {
            bytes: bytes.len().saturating_sub(at),
        });
    }
    NqPack::new(layers, out)
}

fn push_u32(out: &mut Vec<u8>, v: usize) -> Result<(), NqError> {
    let v = u32::try_from(v).map_err(|_| NqError::LengthOverflow)?;
    out.extend_from_slice(&v.to_le_bytes());
    Ok(())
}

fn check_linear(l: &PackedLinear) -> Result<(), NqError> {
    if l.out == 0 || l.inn == 0 || l.rank == 0 {
        return Err(NqError::ZeroDimension);
    }
    let want_u = l.u_words();
    let want_v = l.v_words();
    let u_len = l.out.checked_mul(want_u).ok_or(NqError::LengthOverflow)?;
    if l.u.len() != u_len {
        return Err(NqError::Length {
            field: "U",
            have: l.u.len(),
            expected: u_len,
        });
    }
    let v_len = l.rank.checked_mul(want_v).ok_or(NqError::LengthOverflow)?;
    if l.v.len() != v_len {
        return Err(NqError::Length {
            field: "V",
            have: l.v.len(),
            expected: v_len,
        });
    }
    if l.s_pre.len() != l.inn {
        return Err(NqError::Length {
            field: "s_pre",
            have: l.s_pre.len(),
            expected: l.inn,
        });
    }
    if l.s_mid.len() != l.rank {
        return Err(NqError::Length {
            field: "s_mid",
            have: l.s_mid.len(),
            expected: l.rank,
        });
    }
    if l.s_post.len() != l.out {
        return Err(NqError::Length {
            field: "s_post",
            have: l.s_post.len(),
            expected: l.out,
        });
    }
    let k = l.salient_idx.len();
    if k > l.inn {
        return Err(NqError::Length {
            field: "salient_idx",
            have: k,
            expected: l.inn,
        });
    }
    let salient_len = l.out.checked_mul(k).ok_or(NqError::LengthOverflow)?;
    if l.salient.len() != salient_len {
        return Err(NqError::Length {
            field: "salient",
            have: l.salient.len(),
            expected: salient_len,
        });
    }
    let res_len = l
        .res_layers
        .checked_mul(l.out)
        .and_then(|x| x.checked_mul(l.res_words()))
        .ok_or(NqError::LengthOverflow)?;
    if l.res.len() != res_len {
        return Err(NqError::Length {
            field: "res",
            have: l.res.len(),
            expected: res_len,
        });
    }
    let s_res_len = l
        .res_layers
        .checked_mul(l.out)
        .ok_or(NqError::LengthOverflow)?;
    if l.s_res.len() != s_res_len {
        return Err(NqError::Length {
            field: "s_res",
            have: l.s_res.len(),
            expected: s_res_len,
        });
    }
    let mut seen = vec![false; l.inn];
    for &j in l.salient_idx.iter() {
        let j = j as usize;
        if j >= l.inn || seen[j] {
            return Err(NqError::SalientIndex {
                index: j,
                inn: l.inn,
            });
        }
        seen[j] = true;
    }
    Ok(())
}

/// Write a pack.
///
/// # Errors
///
/// [`NqError::BadCount`] or [`NqError::BadOrder`] for a section list that is not layer-major,
/// [`NqError::SharedHead`] for a member that cannot bind to its layer's `q` section, and every
/// [`NqError`] [`check_linear`] raises for a field whose length disagrees with its dimensions. A
/// subset pack writes only the sections it carries.
pub fn write(pack: &NqPack) -> Result<Vec<u8>, NqError> {
    check_order(pack.layers, &pack.linears)?;
    check_shared(&pack.linears)?;
    // Sharing writes version 4; without it a residual writes version 3 and a residual-free pack
    // version 2, which a reader that predates the residual still accepts.
    let version = if pack.linears.iter().any(|l| l.shared) {
        NQ_VERSION
    } else if pack.linears.iter().any(|l| l.res_layers != 0) {
        NQ_VERSION_RESIDUAL
    } else {
        NQ_VERSION_BASE
    };
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&NQ_MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    push_u32(&mut out, NQ_HEADER_BYTES)?;
    push_u32(&mut out, pack.layers)?;
    push_u32(&mut out, NQ_KINDS)?;
    push_u32(&mut out, pack.linears.len())?;
    push_u32(&mut out, 0)?;
    for l in &pack.linears {
        check_linear(l)?;
        push_u32(&mut out, l.layer)?;
        push_u32(&mut out, l.kind)?;
        push_u32(&mut out, l.out)?;
        push_u32(&mut out, l.inn)?;
        push_u32(&mut out, l.rank)?;
        push_u32(&mut out, l.u_words())?;
        push_u32(&mut out, l.v_words())?;
        push_u32(&mut out, l.salient_idx.len())?;
        if version != NQ_VERSION_BASE {
            push_u32(&mut out, l.res_layers)?;
        }
        if version == NQ_VERSION {
            push_u32(&mut out, usize::from(l.shared))?;
        }
        for &w in l.u.iter() {
            out.extend_from_slice(&w.to_le_bytes());
        }
        if !l.shared {
            for &w in l.v.iter() {
                out.extend_from_slice(&w.to_le_bytes());
            }
            for &s in l.s_pre.iter() {
                out.extend_from_slice(&s.to_le_bytes());
            }
            for &s in l.s_mid.iter() {
                out.extend_from_slice(&s.to_le_bytes());
            }
        }
        for &s in l.s_post.iter() {
            out.extend_from_slice(&s.to_le_bytes());
        }
        if !l.shared {
            for &j in l.salient_idx.iter() {
                out.extend_from_slice(&j.to_le_bytes());
            }
        }
        for &s in l.salient.iter() {
            out.extend_from_slice(&s.to_le_bytes());
        }
        if version != NQ_VERSION_BASE {
            for &w in l.res.iter() {
                out.extend_from_slice(&w.to_le_bytes());
            }
            for &s in l.s_res.iter() {
                out.extend_from_slice(&s.to_le_bytes());
            }
        }
    }
    Ok(out)
}

/// Read a pack from a file path.
///
/// On Linux the file is mapped read-only and the sign planes stay in the mapping: the pack's
/// 2.7 GB of words are file-backed and reclaimable, not a second anonymous copy. The scales are
/// decoded to Q32.32 by the kernel after this returns; the small index and scale planes are
/// mapped too and dropped once decoded. Other platforms read the file into memory first.
///
/// # Errors
///
/// The file open or map's own [`std::io::Error`], or [`NqError`] for the bytes it returned.
pub fn open(path: &str) -> Result<NqPack, OpenError> {
    #[cfg(target_os = "linux")]
    {
        let map = Arc::new(MappedFile::open(path)?);
        let bytes = map.bytes();
        read_with(bytes, &MappedPlanes { bytes, map: &map }).map_err(OpenError::Pack)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let bytes = std::fs::read(path)?;
        read(&bytes).map_err(OpenError::Pack)
    }
}

/// A read-only file mapping whose pages are file-backed and therefore reclaimable.
#[cfg(target_os = "linux")]
#[doc(hidden)]
pub struct MappedFile {
    ptr: *mut libc::c_void,
    len: usize,
}

// SAFETY: the mapping is immutable for its lifetime; `open` is the only constructor and it never
// exposes a writable view. `munmap` runs once, when the last `Arc` drops.
#[cfg(target_os = "linux")]
unsafe impl Send for MappedFile {}
#[cfg(target_os = "linux")]
unsafe impl Sync for MappedFile {}

#[cfg(target_os = "linux")]
impl MappedFile {
    pub(crate) fn open(path: &str) -> std::io::Result<Self> {
        use std::os::unix::io::AsRawFd;

        let file = std::fs::File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::FileTooLarge))?;
        if len == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::InvalidData));
        }
        // SAFETY: the descriptor is open for the call, `len` is the file's current length, and
        // the mapping is private and read-only. The returned pointer is checked against
        // `MAP_FAILED` before it is stored.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self { ptr, len })
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: `ptr` and `len` are the live mapping made by `open`, read-only, and the slice
        // does not outlive `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }
}

#[cfg(target_os = "linux")]
impl fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedFile")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

#[cfg(target_os = "linux")]
impl Drop for MappedFile {
    fn drop(&mut self) {
        // SAFETY: `ptr` and `len` are the mapping made by `open`, unmapped once here.
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}

/// A refused [`open`]: the file read or the pack parse.
#[derive(Debug)]
pub enum OpenError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The bytes are not a valid pack.
    Pack(NqError),
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenError::Io(e) => write!(f, "nq pack: {e}"),
            OpenError::Pack(e) => write!(f, "nq pack: {e}"),
        }
    }
}

impl std::error::Error for OpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OpenError::Io(e) => Some(e),
            OpenError::Pack(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for OpenError {
    fn from(e: std::io::Error) -> Self {
        OpenError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(layer: usize, kind: usize) -> PackedLinear {
        let (out, inn, rank) = match kind {
            0 => (7usize, 5usize, 6usize),
            1 | 2 => (4, 5, 3),
            3 => (5, 7, 6),
            4 | 5 => (9, 5, 8),
            _ => (5, 9, 8),
        };
        let u_words = rank.div_ceil(32);
        let v_words = inn.div_ceil(32);
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ ((layer * 16 + kind) as u64);
        let mut word = |n: usize| {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                v.push(state as u32);
            }
            v
        };
        let u = word(out * u_words);
        let v = word(rank * v_words);
        let res_layers = 2usize;
        let res_words = inn.div_ceil(32);
        let res = word(res_layers * out * res_words);
        let k = 2usize.min(inn);
        PackedLinear {
            layer,
            kind,
            out,
            inn,
            rank,
            u: u.into(),
            v: v.into(),
            s_pre: (0..inn)
                .map(|i| 0x3c00 + i as u16)
                .collect::<Vec<u16>>()
                .into(),
            s_mid: (0..rank)
                .map(|i| 0x3c00 + 3 * i as u16)
                .collect::<Vec<u16>>()
                .into(),
            s_post: (0..out)
                .map(|i| 0x3c00 + 2 * i as u16)
                .collect::<Vec<u16>>()
                .into(),
            salient_idx: (0..k).map(|i| (i * 2) as u32).collect::<Vec<u32>>().into(),
            salient: (0..out * k)
                .map(|i| 0x3800 + i as u16)
                .collect::<Vec<u16>>()
                .into(),
            res_layers,
            shared: false,
            res: res.into(),
            s_res: (0..res_layers * out)
                .map(|i| 0x3000 + i as u16)
                .collect::<Vec<u16>>()
                .into(),
        }
    }

    fn pack(layers: usize) -> NqPack {
        let mut linears = Vec::new();
        for layer in 0..layers {
            for kind in 0..NQ_KINDS {
                linears.push(linear(layer, kind));
            }
        }
        NqPack::new(layers, linears).expect("pack")
    }

    /// A `k`/`v` member that takes `head`'s `V`, `s_pre`, `s_mid` and `salient_idx`: the same
    /// `inn` and `rank`, its own `out` rows and residual.
    fn member(layer: usize, kind: usize, head: &PackedLinear) -> PackedLinear {
        let out = 4usize;
        let res_layers = 2usize;
        let k = head.salient_idx.len();
        let words =
            |n: usize, seed: u32| -> Vec<u32> { (0..n).map(|i| seed ^ (i as u32)).collect() };
        PackedLinear {
            layer,
            kind,
            out,
            inn: head.inn,
            rank: head.rank,
            u: words(out * head.rank.div_ceil(32), 0x1000_0000).into(),
            v: head.v.clone(),
            s_pre: head.s_pre.clone(),
            s_mid: head.s_mid.clone(),
            s_post: (0..out)
                .map(|i| 0x3c00 + i as u16)
                .collect::<Vec<u16>>()
                .into(),
            salient_idx: head.salient_idx.clone(),
            salient: (0..out * k)
                .map(|i| 0x3900 + i as u16)
                .collect::<Vec<u16>>()
                .into(),
            res_layers,
            shared: true,
            res: words(res_layers * out * head.inn.div_ceil(32), 0x2000_0000).into(),
            s_res: (0..res_layers * out)
                .map(|i| 0x3100 + i as u16)
                .collect::<Vec<u16>>()
                .into(),
        }
    }

    fn pack_v4() -> NqPack {
        let mut p = pack(1);
        let head = p.linears[0].clone();
        p.linears[1] = member(0, 1, &head);
        p.linears[2] = member(0, 2, &head);
        NqPack::new(1, p.linears).expect("v4 pack")
    }

    /// The bytes one section occupies in the file: its header plus the payload its `shared` flag
    /// says it stores.
    fn section_len(l: &PackedLinear) -> usize {
        let shared = if l.shared {
            0
        } else {
            4 * l.v.len() + 2 * (l.s_pre.len() + l.s_mid.len()) + 4 * l.salient_idx.len()
        };
        NQ_LINEAR_HEADER_BYTES
            + 4 * l.u.len()
            + 4 * l.res.len()
            + 2 * (l.s_post.len() + l.salient.len() + l.s_res.len())
            + shared
    }

    #[test]
    fn round_trip_is_byte_identical_and_keeps_every_bit() {
        let p = pack(2);
        let bytes = write(&p).expect("write");
        let back = read(&bytes).expect("read");
        assert_eq!(back, p);
        assert_eq!(write(&back).expect("rewrite"), bytes);
    }

    /// The file-backed path parses the same bytes to the same pack: each mapped plane resolves to
    /// its section's extent, and `Plane`'s equality compares the elements, not the storage.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_mapped_pack_equals_the_owned_parse() {
        let p = pack(2);
        let bytes = write(&p).expect("write");
        let path = std::env::temp_dir().join(format!(
            "apeiron_nq_map_{}_{}.nqp",
            std::process::id(),
            bytes.len()
        ));
        std::fs::write(&path, &bytes).expect("write pack");
        let opened = open(path.to_str().expect("utf8 path")).expect("open mapped pack");
        std::fs::remove_file(&path).expect("remove pack");
        assert_eq!(opened, read(&bytes).expect("read owned pack"));
        assert_eq!(opened, p);
    }

    #[test]
    fn a_pack_without_a_residual_writes_version_two() {
        let mut p = pack(1);
        for l in &mut p.linears {
            l.res_layers = 0;
            l.res.clear();
            l.s_res.clear();
        }
        let bytes = write(&p).expect("write");
        assert_eq!(read_u32(&bytes, 8).expect("version"), NQ_VERSION_BASE);
        assert_eq!(read(&bytes).expect("read"), p);
    }

    #[test]
    fn a_residual_pack_without_sharing_writes_version_three() {
        let bytes = write(&pack(1)).expect("write");
        assert_eq!(read_u32(&bytes, 8).expect("version"), NQ_VERSION_RESIDUAL);
    }

    #[test]
    fn a_version_four_pack_round_trips_and_stores_the_shared_block_once() {
        let p = pack_v4();
        let bytes = write(&p).expect("write");
        assert_eq!(read_u32(&bytes, 8).expect("version"), NQ_VERSION);
        assert_eq!(read(&bytes).expect("read"), p);
        let saved: usize = p.linears[1..=2]
            .iter()
            .map(|l| 4 * l.v.len() + 2 * (l.s_pre.len() + l.s_mid.len()) + 4 * l.salient_idx.len())
            .sum();
        let mut unshared = p.clone();
        for l in &mut unshared.linears {
            l.shared = false;
        }
        let unshared_bytes = write(&unshared).expect("unshared");
        assert_eq!(
            bytes.len() + saved,
            unshared_bytes.len() + p.linears.len() * 4
        );
        assert_eq!(p.stored_bits() + saved as u64 * 8, unshared.stored_bits());
    }

    #[test]
    fn a_member_without_its_head_refuses() {
        let p = pack_v4();
        let mut linears = p.linears.clone();
        linears.remove(0);
        assert!(matches!(
            NqPack::new(1, linears),
            Err(NqError::SharedHead {
                at: 0,
                layer: 0,
                kind: 1
            })
        ));
        let mut edited = p;
        edited.linears.remove(0);
        assert!(matches!(
            write(&edited),
            Err(NqError::SharedHead { at: 0, .. })
        ));
    }

    #[test]
    fn a_member_whose_shared_block_differs_refuses() {
        let mut p = pack_v4();
        p.linears[1].s_pre[0] ^= 1;
        assert!(matches!(write(&p), Err(NqError::SharedHead { at: 1, .. })));
        let mut p = pack_v4();
        p.linears[2].salient_idx[0] ^= 1;
        assert!(matches!(write(&p), Err(NqError::SharedHead { at: 2, .. })));
    }

    #[test]
    fn only_k_or_v_may_share() {
        let mut p = pack_v4();
        p.linears[0].shared = true;
        assert!(matches!(write(&p), Err(NqError::SharedHead { at: 0, .. })));
    }

    #[test]
    fn a_group_word_above_one_refuses() {
        let p = pack_v4();
        let mut bytes = write(&p).expect("write");
        let at = NQ_HEADER_BYTES + section_len(&p.linears[0]) + 9 * 4;
        bytes[at..at + 4].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(read(&bytes), Err(NqError::BadGroup { group: 2 }));
    }

    #[test]
    fn rate_is_the_stored_bits_over_the_replaced_weights() {
        let p = pack(1);
        let want: u64 = p
            .linears
            .iter()
            .map(|l| {
                32 * (l.u.len() + l.v.len() + l.salient_idx.len() + l.res.len()) as u64
                    + 16 * (l.inn + l.rank + l.out + l.salient.len() + l.s_res.len()) as u64
            })
            .sum();
        assert_eq!(p.stored_bits(), want);
        let weights: u64 = p.linears.iter().map(|l| (l.out * l.inn) as u64).sum();
        assert_eq!(p.weights(), weights);
        let bpw = p.bits_per_weight().expect("weights");
        assert!((bpw - want as f64 / weights as f64).abs() < 1e-12);
    }

    #[test]
    fn a_flipped_scale_word_changes_the_pack() {
        let p = pack(1);
        let mut q = p.clone();
        q.linears[0].s_pre[0] ^= 1;
        assert_ne!(write(&p).expect("a"), write(&q).expect("b"));
    }

    #[test]
    fn bad_magic_version_and_trailing_bytes_refuse() {
        let bytes = write(&pack(1)).expect("write");
        let mut bad = bytes.clone();
        bad[0] ^= 1;
        assert_eq!(read(&bad), Err(NqError::BadMagic));
        let mut bad = bytes.clone();
        bad[8..12].copy_from_slice(&5u32.to_le_bytes());
        assert_eq!(read(&bad), Err(NqError::BadVersion { version: 5 }));
        let mut bad = bytes.clone();
        bad.extend_from_slice(&[0]);
        assert_eq!(read(&bad), Err(NqError::Trailing { bytes: 1 }));
    }

    #[test]
    fn a_short_section_refuses_without_panicking() {
        let bytes = write(&pack(1)).expect("write");
        let short = &bytes[..bytes.len() - 1];
        assert!(matches!(read(short), Err(NqError::Truncated { .. })));
    }

    #[test]
    fn a_stride_that_disagrees_with_the_rank_refuses() {
        let bytes = write(&pack(1)).expect("write");
        let at = NQ_HEADER_BYTES + 5 * 4;
        let mut bad = bytes;
        bad[at..at + 4].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            read(&bad),
            Err(NqError::BadStride { which: "U", .. })
        ));
    }

    #[test]
    fn a_pack_may_omit_kinds() {
        let p = pack(1);
        let mut linears = p.linears.clone();
        linears.pop();
        let q = NqPack::new(1, linears).expect("subset pack");
        assert_eq!(q.linears.len(), NQ_KINDS - 1);
        assert_eq!(read(&write(&q).expect("write")).expect("read"), q);
    }

    #[test]
    fn a_repeated_or_out_of_order_section_refuses() {
        let p = pack(1);
        let mut linears = p.linears.clone();
        linears.swap(0, 1);
        assert!(matches!(
            NqPack::new(1, linears),
            Err(NqError::BadOrder { .. })
        ));
        let mut linears = p.linears.clone();
        linears[1] = linears[0].clone();
        assert!(matches!(
            NqPack::new(1, linears),
            Err(NqError::BadOrder { .. })
        ));
        let mut linears = p.linears;
        linears[0].layer = 1;
        assert!(matches!(
            NqPack::new(1, linears),
            Err(NqError::BadOrder { .. })
        ));
    }

    #[test]
    fn a_salient_index_outside_the_input_refuses() {
        let p = pack(1);
        let mut q = p.clone();
        q.linears[0].salient_idx[0] = q.linears[0].inn as u32;
        assert!(matches!(
            write(&q),
            Err(NqError::SalientIndex { index, .. }) if index == q.linears[0].inn
        ));
        let mut q = p;
        q.linears[0].salient_idx[1] = q.linears[0].salient_idx[0];
        assert!(matches!(write(&q), Err(NqError::SalientIndex { .. })));
    }
}
