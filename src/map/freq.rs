//! The incumbent's NB count face (`FreqMap`) and map face (`Table`), ported into this crate.
//!
//! `src/map/hashtrinity/` is `hashtrinity` at rev `4cc14471ae24f3f204fb0b0e91736ef75c909358`,
//! copied from the fold Git object at that revision and compiled here as this crate's own module.
//! The table core comes across (`mod.rs`, `algebra`, `freq`, `par`, `face`, `seed`), together with
//! `tregex` and `crypto`, not just `freq.rs`, because:
//!
//! - `E0603`: `bits_for_slots` (defined `hashtrinity/src/lib.rs:501`) and `EMPTY` (defined
//!   `hashtrinity/src/lib.rs:298`) are private, so `freq.rs:5` cannot import them from outside the
//!   crate. Inside `crate::map::hashtrinity` they are in scope.
//! - `E0116`: `freq.rs:112` writes an inherent `impl ... Table<K, i64, Regen<K>>`. An inherent impl
//!   may only be written in the crate that defines the type; here the type is
//!   `crate::map::hashtrinity::Table`, defined in the same crate.
//!
//! So the two refusals that made a bare file copy impossible are answered by porting the whole
//! crate, not by rewriting its logic. The changes to the copied tree are path lines — 66 `crate::`
//! references rewritten to `crate::map::hashtrinity::`, the `hex`/`base64` module gates removed (this
//! crate has no `hex`/`base64` feature and `crypto.rs` is unconditional), `lib.rs` renamed
//! `mod.rs` — and the removal of every module no retained consumer reaches. This in-tree tree is
//! the only copy; the external `hashtrinity` git dependency is gone.
//!
//! Both faces are re-exported here as the public path for the port: `apeiron::map::freq::Table` is the
//! type `src/map/hashtrinity/` defines.

pub use crate::map::hashtrinity::{par, Coord, CoordMap, FoldKey, FreqMap, Table, PHI, PROBE_CAP};
