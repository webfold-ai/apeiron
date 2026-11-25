//! GGUF container parsing, ggml quant block unpacking and byte-level BPE tokenization for the
//! inference-parity load path.
//!
//! [`gguf`] reads the header, metadata table and tensor directory; [`quant`] unpacks the quant
//! formats the served recipe carries. A HuggingFace `tokenizer.json` is NOT one of them: reading a
//! foreign tokenizer is the `tokenizer` crate's `artifact` module, which owns that job.

#![allow(
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::manual_is_multiple_of,
    clippy::identity_op,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    reason = "the format-layer ports (GGUF, quant, byte-level BPE) are byte-level transliterations \
              of the reference readers: the \
              `as` casts follow explicit range or bounds checks (a check plus cast is what \
              extracts, while `TryFrom` emits an unmodeled external), a half-to-float conversion \
              is float by nature, the short names (`n`, `b`, `t`, `d`) are the reference's byte \
              vocabulary, and the small metadata accessors need no `#[must_use]`. The pedantic \
              lints stay on for the rest of the crate."
)]

pub mod gguf;
pub mod packing;
pub mod quant;
