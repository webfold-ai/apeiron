//! `tregex` — write `regex` syntax, run on the streaming-law membership engine. A drop-in for the
//! common subset of the `regex` crate's API (`Regex::new`, `is_match`, `find`, `find_iter`).
//! Pure-literal patterns route to a one-GET-per-byte prefilter (a single literal to the register
//! XOR-compare, a small alternation to register first-byte compares); every other pattern is
//! compiled to a Thompson NFA and run as a Pike VM.
//!
//! **Complexity.** A search is `O(haystack × program)` time and `O(program)` space, whatever the
//! pattern: the VM keeps one thread per NFA state and never backtracks, so `a*b`, `a*a*a*b` and
//! `\w+@\w+` on a long run of `a` are linear in the haystack. Compiling is `O(pattern)`:
//! alternation groups are NFA branches (not expanded into sequences), repetition counts are capped
//! at [`MAX_REPEAT`] (1000), the compiled program at [`MAX_PROGRAM`] (65 536 steps), the pattern at
//! [`MAX_PATTERN_LEN`] (256 KiB) and group nesting at [`MAX_GROUP_DEPTH`] (128); a pattern over a limit is an `Err` from [`Regex::new`].
//! `find_iter` restarts the search after each match, so the total is `O(program × haystack)` per
//! match in the worst case (an earlier alternative that runs on past a later one's match, as in
//! `a*b|a` on `aaaa…`, is scanned again for the next match): linear for a pattern with no such
//! overlap, quadratic at worst.
//!
//! **Semantics.** Leftmost-first, as the `regex` crate: the match starting leftmost wins, and among
//! matches from that start the one the pattern prefers (alternatives in written order, quantifiers
//! greedy) wins, so `a|ab` on `ab` is `a` and `ab|a` is `ab`. Empty matches are reported as
//! `regex::find_iter` reports them. `^`/`$` are assertions at their own place in the pattern.
//! Matching is on bytes (`.` is any byte but `\n`, `\w`/`\d`/`\s` are ASCII), like
//! `regex::bytes` with Unicode off.
//!
//! The thesis (`mem` module): a DFA's `state[n+1]` depends on `state[n]` — a serial chain that is
//! latency-bound. The membership GET reads each position independently, so the loads pipeline:
//! throughput-bound. That is why the literal scans beat a general regex engine ~2× while staying
//! byte-exact.
//!
//! Supported syntax: literals, top-level alternation `a|b|c`, groups `(…)` / `(?:…)` with
//! alternation inside (nested too), char classes `[...]` / `[^...]`, `.`, the escapes
//! `\d \w \s \D \W \S` + `\.\-\+` etc., quantifiers `* + ? {n} {n,} {n,m}` on a single
//! character or class, and `\b` word boundaries. Captures, backrefs, lazy quantifiers and
//! quantified groups are not supported (a quantified group is a compile error).

pub mod mem {
    //! THE MULTI-PATTERN PRIMITIVE — two faces of the HashTrinity over a literal set.
    //!
    //! `Dfa` is THE TRANSITION FACE — a byte automaton, the merge `(state, byteclass) → next`
    //! (Web's graph face folded into a complete table). One array load per byte, but `state[n+1]`
    //! depends on `state[n]`: a serial chain, latency-bound. This is the cold re-scan in a lab coat.
    //!
    //! `MemMatch` is THE MEMBERSHIP FACE — the streaming law (`Substrate/Collapse.streaming_law`).
    //! Certify the patterns ONCE (a bit-array of their k-byte prefixes); then read each position
    //! INDEPENDENTLY — one window load, one hashed GET, no carried state. The GET loads pipeline
    //! (nothing depends on the previous), so it is throughput-bound, not latency-bound. The rare
    //! bit-set hit verifies the full pattern (the Map face). Measured ~2× the DFA, ~1.35× aho-corasick.

    use std::collections::VecDeque;

    const ACCEPT: u32 = 0x8000_0000;
    const STATE: u32 = 0x7fff_ffff;

    pub struct Dfa {
        trans: Vec<u32>,   // [num_states * nclass]; top bit = accept
        class: [u32; 256], // byte -> class (alphabet equivalence; 0 = "other")
        nclass: usize,
        outputs: Vec<Vec<u32>>, // [num_states]; pattern ids ending in this state
        pub lens: Vec<usize>,
    }

    impl Dfa {
        #[must_use]
        pub fn build(patterns: &[&[u8]]) -> Dfa {
            // --- byte classes: significant bytes (used in patterns) each get a class; rest -> 0.
            let mut sig = [false; 256];
            for p in patterns {
                for &b in p.iter() {
                    sig[b as usize] = true;
                }
            }
            let mut class = [0u32; 256];
            let mut nclass = 1usize; // class 0 = "other"
            let mut rep: Vec<usize> = vec![{
                // a representative byte for class 0: the first non-significant byte (exists for ASCII patterns)
                (0..256).find(|&b| !sig[b]).unwrap_or(0)
            }];
            for b in 0..256 {
                if sig[b] {
                    class[b] = nclass as u32;
                    rep.push(b);
                    nclass += 1;
                }
            }

            // --- trie (goto), sparse during construction (keyed by raw byte).
            let mut goto: Vec<[i64; 256]> = vec![[-1; 256]];
            let mut outputs: Vec<Vec<u32>> = vec![vec![]];
            let lens: Vec<usize> = patterns.iter().map(|p| p.len()).collect();
            for (pid, pat) in patterns.iter().enumerate() {
                let mut s = 0usize;
                for &b in pat.iter() {
                    let nxt = goto[s][b as usize];
                    s = if nxt >= 0 {
                        nxt as usize
                    } else {
                        let n = goto.len();
                        goto.push([-1; 256]);
                        outputs.push(vec![]);
                        goto[s][b as usize] = n as i64;
                        n
                    };
                }
                outputs[s].push(pid as u32);
            }
            let n = goto.len();

            // --- failure links (BFS), merging outputs along the fail chain (overlapping matches).
            let mut fail = vec![0u32; n];
            let mut q = VecDeque::new();
            for b in 0..256 {
                let t = goto[0][b];
                if t > 0 {
                    fail[t as usize] = 0;
                    q.push_back(t as usize);
                }
            }
            while let Some(s) = q.pop_front() {
                for b in 0..256 {
                    let t = goto[s][b];
                    if t < 0 {
                        continue;
                    }
                    let t = t as usize;
                    q.push_back(t);
                    let mut f = fail[s] as usize;
                    while f != 0 && goto[f][b] < 0 {
                        f = fail[f] as usize;
                    }
                    let cand = goto[f][b];
                    fail[t] = if cand >= 0 && cand as usize != t {
                        cand as u32
                    } else {
                        0
                    };
                    let extra = outputs[fail[t] as usize].clone();
                    outputs[t].extend(extra);
                }
            }

            // --- complete table over CLASSES, in BFS order (fail row already complete).
            let mut trans = vec![0u32; n * nclass];
            let mut order = Vec::with_capacity(n);
            let mut seen = vec![false; n];
            let mut q2 = VecDeque::new();
            q2.push_back(0usize);
            seen[0] = true;
            while let Some(s) = q2.pop_front() {
                order.push(s);
                for b in 0..256 {
                    let t = goto[s][b];
                    if t > 0 && !seen[t as usize] {
                        seen[t as usize] = true;
                        q2.push_back(t as usize);
                    }
                }
            }
            for &s in &order {
                for c in 0..nclass {
                    let b = rep[c];
                    let next = if goto[s][b] >= 0 {
                        goto[s][b] as u32
                    } else if s == 0 {
                        0
                    } else {
                        trans[fail[s] as usize * nclass + c] & STATE
                    };
                    let tag = if outputs[next as usize].is_empty() {
                        0
                    } else {
                        ACCEPT
                    };
                    trans[s * nclass + c] = next | tag;
                }
            }
            Dfa {
                trans,
                class,
                nclass,
                outputs,
                lens,
            }
        }

        /// THE HOT LOOP — two L1 loads per byte (class map, transition), nothing else off the accept path.
        #[must_use]
        pub fn count(&self, hay: &[u8]) -> usize {
            let trans = &self.trans;
            let class = &self.class;
            let nclass = self.nclass;
            let mut s = 0usize;
            let mut c = 0usize;
            for &b in hay {
                let cls = unsafe { *class.get_unchecked(b as usize) } as usize;
                let cell = unsafe { *trans.get_unchecked(s * nclass + cls) };
                s = (cell & STATE) as usize;
                if cell & ACCEPT != 0 {
                    c += unsafe { self.outputs.get_unchecked(s) }.len();
                }
            }
            c
        }

        pub fn find_all(&self, hay: &[u8], out: &mut Vec<(usize, u32)>) {
            let trans = &self.trans;
            let class = &self.class;
            let nclass = self.nclass;
            let mut s = 0usize;
            for (i, &b) in hay.iter().enumerate() {
                let cls = unsafe { *class.get_unchecked(b as usize) } as usize;
                let cell = unsafe { *trans.get_unchecked(s * nclass + cls) };
                s = (cell & STATE) as usize;
                if cell & ACCEPT != 0 {
                    for &pid in &self.outputs[s] {
                        out.push((i, pid));
                    }
                }
            }
        }

        #[must_use]
        pub fn num_states(&self) -> usize {
            self.outputs.len()
        }
        #[must_use]
        pub fn num_classes(&self) -> usize {
            self.nclass
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // THE MEMBERSHIP FACE — the streaming law, NOT the re-scan (Substrate/Collapse).
    // ═══════════════════════════════════════════════════════════════════════════

    const C: u64 = 0x9E37_79B9_7F4A_7C15;

    pub struct MemMatch {
        seen: Vec<u64>, // bit-array indexed by hash(prefix); the certified rest state
        bits: u32,
        k: usize, // window = the shortest pattern's length (every pattern has a k-prefix)
        kmask: u64,
        /// exact k-prefix value -> the patterns sharing it (the rare verify path; Map face)
        verify: std::collections::HashMap<u64, Vec<(Vec<u8>, u32)>>,
        pub npat: usize,
    }

    impl MemMatch {
        #[inline]
        fn slot(&self, w: u64) -> usize {
            (w.wrapping_mul(C) >> (64 - self.bits)) as usize
        }

        #[must_use]
        pub fn build(patterns: &[&[u8]]) -> MemMatch {
            let k = patterns.iter().map(|p| p.len()).min().unwrap_or(1).min(8);
            let kmask = if k >= 8 {
                u64::MAX
            } else {
                (1u64 << (8 * k)) - 1
            };
            let bits = 22u32; // 2^22 bits = 512 KB; sparse for a real (rare-pattern) set
            let mut m = MemMatch {
                seen: vec![0u64; (1usize << bits) / 64],
                bits,
                k,
                kmask,
                verify: std::collections::HashMap::new(),
                npat: patterns.len(),
            };
            for (pid, pat) in patterns.iter().enumerate() {
                let mut pre = [0u8; 8];
                pre[..k].copy_from_slice(&pat[..k]);
                let w = u64::from_le_bytes(pre) & kmask;
                let s = m.slot(w);
                m.seen[s >> 6] |= 1u64 << (s & 63);
                m.verify
                    .entry(w)
                    .or_default()
                    .push((pat.to_vec(), pid as u32));
            }
            m
        }

        /// Leftmost match at or after `from`: scan with the fast membership GET, return
        /// `(start, longest_len)` of the first real match. The drop-in `find_iter`'s candidate finder.
        #[must_use]
        pub fn find_at(&self, hay: &[u8], from: usize) -> Option<(usize, usize)> {
            let n = hay.len();
            if n < self.k {
                return None;
            }
            let kmask = self.kmask;
            let shift = 64 - self.bits;
            let seen = &self.seen;
            let p = hay.as_ptr();
            let fast = n.saturating_sub(7);
            let scan_end = n - self.k;
            let mut i = from;
            while i <= scan_end {
                let w = if i < fast {
                    (unsafe { (p.add(i) as *const u64).read_unaligned() }) & kmask
                } else {
                    let mut pre = [0u8; 8];
                    let avail = (n - i).min(8);
                    pre[..avail].copy_from_slice(&hay[i..i + avail]);
                    u64::from_le_bytes(pre) & kmask
                };
                let s = (w.wrapping_mul(C) >> shift) as usize;
                if unsafe { *seen.get_unchecked(s >> 6) } & (1u64 << (s & 63)) != 0 {
                    if let Some(cands) = self.verify.get(&w) {
                        let mut best = 0usize;
                        for (pat, _) in cands {
                            if pat.len() > best
                                && i + pat.len() <= n
                                && &hay[i..i + pat.len()] == pat.as_slice()
                            {
                                best = pat.len();
                            }
                        }
                        if best > 0 {
                            return Some((i, best));
                        }
                    }
                }
                i += 1;
            }
            None
        }

        /// Non-overlapping leftmost-longest matches as `(start, end)`, in ONE pass.
        #[must_use]
        pub fn matches_nonoverlapping(&self, hay: &[u8]) -> Vec<(usize, usize)> {
            let mut out = Vec::new();
            let n = hay.len();
            if n < self.k {
                return out;
            }
            let kmask = self.kmask;
            let shift = 64 - self.bits;
            let seen = &self.seen;
            let p = hay.as_ptr();
            let fast = n.saturating_sub(7);
            let scan_end = n - self.k;
            let mut i = 0usize;
            while i <= scan_end {
                let w = if i < fast {
                    (unsafe { (p.add(i) as *const u64).read_unaligned() }) & kmask
                } else {
                    let mut pre = [0u8; 8];
                    let avail = (n - i).min(8);
                    pre[..avail].copy_from_slice(&hay[i..i + avail]);
                    u64::from_le_bytes(pre) & kmask
                };
                let s = (w.wrapping_mul(C) >> shift) as usize;
                if unsafe { *seen.get_unchecked(s >> 6) } & (1u64 << (s & 63)) != 0 {
                    if let Some(cands) = self.verify.get(&w) {
                        let mut best = 0usize;
                        for (pat, _) in cands {
                            if pat.len() > best
                                && i + pat.len() <= n
                                && &hay[i..i + pat.len()] == pat.as_slice()
                            {
                                best = pat.len();
                            }
                        }
                        if best > 0 {
                            out.push((i, i + best));
                            i += best;
                            continue;
                        }
                    }
                }
                i += 1;
            }
            out
        }

        /// Count all occurrences (overlapping, by start position). The hot loop: one window load,
        /// one membership GET per position — both independent across positions, so they pipeline.
        #[must_use]
        pub fn count(&self, hay: &[u8]) -> usize {
            let n = hay.len();
            if n < self.k {
                return 0;
            }
            let kmask = self.kmask;
            let shift = 64 - self.bits;
            let seen = &self.seen;
            let mut c = 0usize;
            let fast = n.saturating_sub(7);
            let p = hay.as_ptr();
            for i in 0..fast {
                let w = unsafe { (p.add(i) as *const u64).read_unaligned() } & kmask;
                let s = (w.wrapping_mul(C) >> shift) as usize;
                if unsafe { *seen.get_unchecked(s >> 6) } & (1u64 << (s & 63)) != 0 {
                    if let Some(cands) = self.verify.get(&w) {
                        for (pat, _) in cands {
                            if i + pat.len() <= n && &hay[i..i + pat.len()] == pat.as_slice() {
                                c += 1;
                            }
                        }
                    }
                }
            }
            // tail (last <8 positions): assemble the window safely
            for i in fast..=n.saturating_sub(self.k) {
                let mut pre = [0u8; 8];
                let avail = (n - i).min(8);
                pre[..avail].copy_from_slice(&hay[i..i + avail]);
                let w = u64::from_le_bytes(pre) & kmask;
                let s = self.slot(w);
                if seen[s >> 6] & (1u64 << (s & 63)) != 0 {
                    if let Some(cands) = self.verify.get(&w) {
                        for (pat, _) in cands {
                            if i + pat.len() <= n && &hay[i..i + pat.len()] == pat.as_slice() {
                                c += 1;
                            }
                        }
                    }
                }
            }
            c
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn overlapping_and_basic() {
            let pats: &[&[u8]] = &[b"he", b"she", b"his", b"hers"];
            let dfa = Dfa::build(pats);
            let mut out = Vec::new();
            dfa.find_all(b"ushers", &mut out);
            let ids: Vec<u32> = out.iter().map(|&(_, p)| p).collect();
            assert!(ids.contains(&1));
            assert!(ids.contains(&0));
            assert!(ids.contains(&3));
        }
        #[test]
        fn membership_parity() {
            let pats: &[&[u8]] = &[b"he", b"she", b"his", b"hers", b"the"];
            let dfa = Dfa::build(pats);
            let mem = MemMatch::build(pats);
            let hay = b"ushers the cashier hisses where he hershey";
            assert_eq!(
                dfa.count(hay),
                mem.count(hay),
                "membership must match DFA count"
            );
        }
    }
}
pub mod simd {
    //! The register-resident literal scan — the XOR-compare floor. The model (the first byte) is a
    //! register; `cmpeq` is the Axiom-V flaw injection; the only data movement is the one haystack load.
    //! On data bigger than cache that load saturates memory bandwidth (SSE2 already at the floor,
    //! measured ~19 GB/s in `~/projects/floortest`, past memchr); on cache-resident data the load width
    //! matters, so we run AVX2 (32 B) when present, else SSE2 (16 B, baseline x86-64).
    //! Non-overlapping leftmost spans, byte-exact with `memchr::memmem::find_iter`.

    use crate::map::hashtrinity::tregex::Match;

    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[cfg(target_arch = "x86_64")]
    pub fn find_literal(hay: &[u8], needle: &[u8], out: &mut Vec<Match>) {
        if is_x86_feature_detected!("avx2") {
            unsafe { find_avx2(hay, needle, out) }
        } else {
            find_sse2(hay, needle, out)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_avx2(hay: &[u8], needle: &[u8], out: &mut Vec<Match>) {
        let (n, nlen) = (hay.len(), needle.len());
        if nlen == 0 || n < nlen {
            return;
        }
        let first = _mm256_set1_epi8(needle[0] as i8);
        let p = hay.as_ptr();
        let mut i = 0usize;
        let lim = if n >= 32 { n - 31 } else { 0 };
        'outer: while i < lim {
            let chunk = _mm256_loadu_si256(p.add(i) as *const __m256i);
            let eq = _mm256_cmpeq_epi8(chunk, first);
            let mut mask = _mm256_movemask_epi8(eq) as u32;
            while mask != 0 {
                let j = i + mask.trailing_zeros() as usize;
                if j + nlen <= n && hay.get_unchecked(j..j + nlen) == needle {
                    out.push(Match {
                        start: j,
                        end: j + nlen,
                    });
                    i = j + nlen;
                    continue 'outer;
                }
                mask &= mask - 1;
            }
            i += 32;
        }
        while i + nlen <= n {
            if hay.get_unchecked(i..i + nlen) == needle {
                out.push(Match {
                    start: i,
                    end: i + nlen,
                });
                i += nlen;
            } else {
                i += 1;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn find_sse2(hay: &[u8], needle: &[u8], out: &mut Vec<Match>) {
        let (n, nlen) = (hay.len(), needle.len());
        if nlen == 0 || n < nlen {
            return;
        }
        unsafe {
            let first = _mm_set1_epi8(needle[0] as i8);
            let p = hay.as_ptr();
            let mut i = 0usize;
            let lim = if n >= 16 { n - 15 } else { 0 };
            'outer: while i < lim {
                let chunk = _mm_loadu_si128(p.add(i) as *const __m128i);
                let eq = _mm_cmpeq_epi8(chunk, first);
                let mut mask = _mm_movemask_epi8(eq) as u32;
                while mask != 0 {
                    let j = i + mask.trailing_zeros() as usize;
                    if j + nlen <= n && hay.get_unchecked(j..j + nlen) == needle {
                        out.push(Match {
                            start: j,
                            end: j + nlen,
                        });
                        i = j + nlen;
                        continue 'outer;
                    }
                    mask &= mask - 1;
                }
                i += 16;
            }
            while i + nlen <= n {
                if hay.get_unchecked(i..i + nlen) == needle {
                    out.push(Match {
                        start: i,
                        end: i + nlen,
                    });
                    i += nlen;
                } else {
                    i += 1;
                }
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn find_literal(hay: &[u8], needle: &[u8], out: &mut Vec<Match>) {
        let (n, nlen) = (hay.len(), needle.len());
        if nlen == 0 || n < nlen {
            return;
        }
        let first = needle[0];
        let mut i = 0;
        while i + nlen <= n {
            if hay[i] == first && &hay[i..i + nlen] == needle {
                out.push(Match {
                    start: i,
                    end: i + nlen,
                });
                i += nlen;
            } else {
                i += 1;
            }
        }
    }

    /// First occurrence only (for `find` / `is_match`).
    pub fn find_one(hay: &[u8], needle: &[u8]) -> Option<Match> {
        let mut v = Vec::with_capacity(1);
        find_literal(hay, needle, &mut v);
        v.into_iter().next()
    }

    // ── multi-literal: the same subtraction, the model as register first-byte compares ───────────────
    // A small literal alternation. The certified rest is the set of DISTINCT first bytes, splatted into
    // registers; the prefilter is an OR of `cmpeq`s (no hash table, no L2). On a candidate we verify the
    // literals sharing that first byte and take the first in written order (leftmost-first, non-overlapping — the `regex` crate's
    // alternation semantics).

    /// The model: the distinct first bytes, and the literals grouped by first byte (≤8 distinct bytes,
    /// or this isn't built and the hash path runs).
    pub struct MultiLit {
        pub fbs: Vec<u8>,
        pub by_first: Box<[Vec<Vec<u8>>; 256]>,
    }

    /// Length of the literal that wins at `j`: the first one, in written order, that is there
    /// (leftmost-first, as the `regex` crate orders an alternation), or 0.
    #[inline]
    fn verify_first(by_first: &[Vec<Vec<u8>>; 256], hay: &[u8], j: usize, n: usize) -> usize {
        for lit in unsafe { by_first.get_unchecked(*hay.get_unchecked(j) as usize) } {
            let l = lit.len();
            if j + l <= n && &hay[j..j + l] == lit.as_slice() {
                return l;
            }
        }
        0
    }

    #[cfg(target_arch = "x86_64")]
    pub fn find_multi(hay: &[u8], m: &MultiLit, out: &mut Vec<Match>) {
        if is_x86_feature_detected!("avx2") {
            unsafe { find_multi_avx2(hay, m, out) }
        } else {
            find_multi_sse2(hay, m, out)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_multi_avx2(hay: &[u8], m: &MultiLit, out: &mut Vec<Match>) {
        let n = hay.len();
        let mut splat = [_mm256_setzero_si256(); 8];
        for (k, &b) in m.fbs.iter().enumerate().take(8) {
            splat[k] = _mm256_set1_epi8(b as i8);
        }
        let nfb = m.fbs.len().min(8);
        let p = hay.as_ptr();
        let mut i = 0usize;
        let lim = if n >= 32 { n - 31 } else { 0 };
        'outer: while i < lim {
            let chunk = _mm256_loadu_si256(p.add(i) as *const __m256i);
            let mut eq = _mm256_cmpeq_epi8(chunk, splat[0]);
            for s in splat.iter().take(nfb).skip(1) {
                eq = _mm256_or_si256(eq, _mm256_cmpeq_epi8(chunk, *s));
            }
            let mut mask = _mm256_movemask_epi8(eq) as u32;
            while mask != 0 {
                let j = i + mask.trailing_zeros() as usize;
                let best = verify_first(&m.by_first, hay, j, n);
                if best > 0 {
                    out.push(Match {
                        start: j,
                        end: j + best,
                    });
                    i = j + best;
                    continue 'outer;
                }
                mask &= mask - 1;
            }
            i += 32;
        }
        while i < n {
            let best = verify_first(&m.by_first, hay, i, n);
            if best > 0 {
                out.push(Match {
                    start: i,
                    end: i + best,
                });
                i += best;
            } else {
                i += 1;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn find_multi_sse2(hay: &[u8], m: &MultiLit, out: &mut Vec<Match>) {
        let n = hay.len();
        unsafe {
            let mut splat = [_mm_setzero_si128(); 8];
            for (k, &b) in m.fbs.iter().enumerate().take(8) {
                splat[k] = _mm_set1_epi8(b as i8);
            }
            let nfb = m.fbs.len().min(8);
            let p = hay.as_ptr();
            let mut i = 0usize;
            let lim = if n >= 16 { n - 15 } else { 0 };
            'outer: while i < lim {
                let chunk = _mm_loadu_si128(p.add(i) as *const __m128i);
                let mut eq = _mm_cmpeq_epi8(chunk, splat[0]);
                for s in splat.iter().take(nfb).skip(1) {
                    eq = _mm_or_si128(eq, _mm_cmpeq_epi8(chunk, *s));
                }
                let mut mask = _mm_movemask_epi8(eq) as u32;
                while mask != 0 {
                    let j = i + mask.trailing_zeros() as usize;
                    let best = verify_first(&m.by_first, hay, j, n);
                    if best > 0 {
                        out.push(Match {
                            start: j,
                            end: j + best,
                        });
                        i = j + best;
                        continue 'outer;
                    }
                    mask &= mask - 1;
                }
                i += 16;
            }
            while i < n {
                let best = verify_first(&m.by_first, hay, i, n);
                if best > 0 {
                    out.push(Match {
                        start: i,
                        end: i + best,
                    });
                    i += best;
                } else {
                    i += 1;
                }
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn find_multi(hay: &[u8], m: &MultiLit, out: &mut Vec<Match>) {
        let n = hay.len();
        let mut i = 0;
        while i < n {
            let best = verify_first(&m.by_first, hay, i, n);
            if best > 0 {
                out.push(Match {
                    start: i,
                    end: i + best,
                });
                i += best;
            } else {
                i += 1;
            }
        }
    }
}
pub mod term {
    //! THE REGEX TYPE SET — fallen out of `Polymorphic.Term`, not invented.
    //!
    //! `proof/Substrate/Algebra/Bulk/UniversalEncoding.lean` proves there is ONE encoding everything reduces
    //! to — the free bulk algebra `Term` (constants `K`, the two routings `S_routeF`/`S_routeG`) — and
    //! that **meaning is the applicative, not structure** (`applicative_is_the_only_freedom`; Axiom XX:
    //! α is a type choice, never a code path). A regex is an instance. regex-syntax's HIR — the regex
    //! crate's own post-desugar smallest form — has EIGHT constructors; under the universal encoding
    //! they collapse to the trinity plus the reflexive fourth:
    //!
    //! | HIR constructor          | Term reading                | this module          |
    //! |--------------------------|-----------------------------|----------------------|
    //! | `Empty`/`Literal`/`Class`/`Look` | **one** `K` (constant boundary; predicate is the only freedom) | [`K`] |
    //! | `Concat`                 | `S_routeF` (sequence)       | [`Re::Concat`]       |
    //! | `Alternation`            | `S_routeG` (branch)         | [`Re::Alt`]          |
    //! | `Repetition{min,max}`    | the **spinor** `B/(1-A)` — folded into the `K` predicate, NOT a 4th form | `K::Class(_,min,max)` |
    //! | `Capture`                | inner Term + span side-channel (no recognizer change) | (dropped) |
    //!
    //! So [`Re`] is EXACTLY `Term`'s **three** constructors — `Atom`=`K`, `Concat`=`S_routeF`,
    //! `Alt`=`S_routeG`. The spinor (repetition) is not a fourth: a count is a parameter of the
    //! boundary predicate (Axiom XX: part of the applicative, never structure), so it lives inside `K`.
    //! THE REFLEXIVE-FOURTH PATH: the reflexive fourth is a doorway into a face, not a new structure — "3 or 4
    //! resolves to 3." And the FOUR HIR atoms are one `K` for the same reason — predicate is the only
    //! freedom (`UniversalEncoding` fact 3). The engine in `lib.rs` is the lowering: `Re` → the flat
    //! membership program (`fromTerm` at the metal applicative; the compiler is topological drag,
    //! `CompilerIsDrag` — no separate codegen, only `Term` run under the membership applicative).

    /// THE K-ATOM (`Polymorphic.Term.K`) — a constant boundary; the ONE atom. HIR's
    /// `Empty`/`Literal`/`Class`/`Look` are this under different predicates (the only freedom). The
    /// three variants are themselves the trinity of the boundary: `Run` = **B** (a determined run),
    /// `Class` = **B/U** (a choice — the division, and where the spinar count lives), `Look` = **NB**
    /// (the zero-width null that consumes nothing).
    #[derive(Clone)]
    pub enum K {
        /// **B** — `Literal`: consume these exact bytes, once (a fused run of single-byte `K`s — kept as
        /// a run so the membership prefilter has a ≥1-byte window).
        Run(Vec<u8>),
        /// **B/U** — `Class` + `Repetition`: consume `[min,max]` bytes each in this set. The spinor is
        /// FOLDED IN here (the count is a predicate parameter, Axiom XX), so `Re` keeps its native 3
        /// forms. (`Literal` of length 1, unquantified, is `Run([b])` instead — the run-fused form.)
        Class(Box<[bool; 256]>, usize, usize),
        /// **NB** — `Look`/`Empty`: consume nothing; a position predicate (the zero-width `K`).
        Look(Look),
    }

    /// The zero-width `K`s: HIR's `Look` (here, the subset tregex supports). `Start`/`End` are pattern-
    /// global (the anchors `^ $`); `WordB` is positional (`\b`).
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Look {
        Start,
        End,
        WordB,
    }

    /// THE TERM (`Polymorphic.Term`) — EXACTLY THREE constructors, the trinity `{NB,B,B/U}` = `{0,1,/}`,
    /// one-for-one with `UniversalEncoding`'s `{K, S_routeF, S_routeG}`. A regex is one of these and
    /// adds nothing: the spinor (repetition) folds into the `K` predicate above, not a fourth form.
    #[derive(Clone)]
    pub enum Re {
        /// `K` — the constant boundary (its payload is the [`K`] atom).
        Atom(K),
        /// `S_routeF` — a sequence (one alternative's body).
        Concat(Vec<Re>),
        /// `S_routeG` — a branch (top-level alternation).
        Alt(Vec<Re>),
    }

    // ── THE LOWERED FORM (the metal applicative) ─────────────────────────────────
    // `Seg` is `Re` run under the membership applicative: a flat per-alternative segment list the
    // engine in `lib.rs` scans with the stateless GET. It is `fromTerm` at the target — the only thing
    // that distinguishes one regex from another is this lowering, never a new code path.

    /// One lowered segment: a literal run, a class with its spinor `[min,max]` fused, or the `\b` `K`.
    #[derive(Clone)]
    pub enum Seg {
        Lit(Vec<u8>),
        Class {
            mask: Box<[bool; 256]>,
            min: usize,
            max: usize,
        },
        WordB,
    }

    pub(crate) fn is_word(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_'
    }

    /// Largest repetition count (`{n}`, `{n,m}`, the `n` of `{n,}`) a pattern may use. Each counted
    /// repetition is unrolled into that many NFA steps, so the count is a cost, not a number.
    pub const MAX_REPEAT: usize = 1000;
    /// Deepest `(…)` nesting a pattern may use; the parser and the NFA compiler recurse per level.
    pub const MAX_GROUP_DEPTH: usize = 128;
    /// Longest pattern, in bytes. Parsing a character builds its 256-entry byte set, so an unbounded
    /// pattern is an unbounded amount of work before any other limit can apply.
    pub const MAX_PATTERN_LEN: usize = 1 << 18;
    /// Most flat sequences [`lower`] will build before it stops expanding groups (see there).
    const MAX_LOWERED_SEQUENCES: usize = 256;

    // ── parser: surface syntax → `Re` (the universal Term) ───────────────────────
    struct P<'a> {
        s: &'a [u8],
        i: usize,
        /// Open groups right now, bounded by [`MAX_GROUP_DEPTH`].
        depth: usize,
    }
    impl<'a> P<'a> {
        fn peek(&self) -> Option<u8> {
            self.s.get(self.i).copied()
        }
        fn bump(&mut self) -> Option<u8> {
            let c = self.peek();
            if c.is_some() {
                self.i += 1;
            }
            c
        }

        /// parse one class atom -> a 256 mask (also a single literal byte)
        fn class_atom(&mut self) -> Result<[bool; 256], String> {
            let mut m = [false; 256];
            match self.bump().ok_or("unexpected end")? {
                b'.' => {
                    for x in &mut m {
                        *x = true;
                    }
                    m[b'\n' as usize] = false;
                }
                b'\\' => {
                    let e = self.bump().ok_or("dangling escape")?;
                    match e {
                        b'd' => {
                            for x in b'0'..=b'9' {
                                m[x as usize] = true;
                            }
                        }
                        b'D' => {
                            for x in &mut m {
                                *x = true;
                            }
                            for x in b'0'..=b'9' {
                                m[x as usize] = false;
                            }
                        }
                        b'w' => {
                            for x in 0..=255u8 {
                                if is_word(x) {
                                    m[x as usize] = true;
                                }
                            }
                        }
                        b'W' => {
                            for x in 0..=255u8 {
                                if !is_word(x) {
                                    m[x as usize] = true;
                                }
                            }
                        }
                        b's' => {
                            for &x in b" \t\n\r\x0c" {
                                m[x as usize] = true;
                            }
                        }
                        b'S' => {
                            for x in &mut m {
                                *x = true;
                            }
                            for &x in b" \t\n\r\x0c" {
                                m[x as usize] = false;
                            }
                        }
                        other => m[other as usize] = true, // \. \- \+ \\ etc -> literal
                    }
                }
                b'[' => {
                    let neg = self.peek() == Some(b'^');
                    if neg {
                        self.i += 1;
                    }
                    let mut set = [false; 256];
                    while let Some(c) = self.peek() {
                        if c == b']' {
                            self.i += 1;
                            break;
                        }
                        let lo = if c == b'\\' {
                            self.i += 1;
                            self.bump().ok_or("bad class esc")?
                        } else {
                            self.bump().unwrap()
                        };
                        if self.peek() == Some(b'-') && self.s.get(self.i + 1) != Some(&b']') {
                            self.i += 1; // '-'
                            let hi = self.bump().ok_or("bad range")?;
                            for x in lo..=hi {
                                set[x as usize] = true;
                            }
                        } else {
                            set[lo as usize] = true;
                        }
                    }
                    if neg {
                        for x in 0..256 {
                            m[x] = !set[x];
                        }
                    } else {
                        m = set;
                    }
                }
                lit => m[lit as usize] = true,
            }
            Ok(m)
        }

        fn quant(&mut self) -> Result<(usize, usize), String> {
            Ok(match self.peek() {
                Some(b'*') => {
                    self.i += 1;
                    (0, usize::MAX)
                }
                Some(b'+') => {
                    self.i += 1;
                    (1, usize::MAX)
                }
                Some(b'?') => {
                    self.i += 1;
                    (0, 1)
                }
                Some(b'{') => {
                    self.i += 1;
                    let mut n = String::new();
                    while let Some(c) = self.peek() {
                        if c.is_ascii_digit() {
                            n.push(c as char);
                            self.i += 1;
                        } else {
                            break;
                        }
                    }
                    // a count too big for `usize` is over `MAX_REPEAT` all the same
                    let too_big = || "repetition count is too large".to_string();
                    let min: usize = if n.is_empty() {
                        0
                    } else {
                        n.parse().map_err(|_| too_big())?
                    };
                    let max = if self.peek() == Some(b',') {
                        self.i += 1;
                        let mut mx = String::new();
                        while let Some(c) = self.peek() {
                            if c.is_ascii_digit() {
                                mx.push(c as char);
                                self.i += 1;
                            } else {
                                break;
                            }
                        }
                        if mx.is_empty() {
                            usize::MAX
                        } else {
                            mx.parse().map_err(|_| too_big())?
                        }
                    } else {
                        min
                    };
                    if self.peek() == Some(b'}') {
                        self.i += 1;
                    }
                    if min > MAX_REPEAT || (max != usize::MAX && max > MAX_REPEAT) {
                        return Err(format!("repetition count exceeds {MAX_REPEAT}"));
                    }
                    if max < min {
                        return Err("repetition range is empty (max < min)".to_string());
                    }
                    (min, max)
                }
                _ => (1, 1),
            })
        }

        /// parse one alternative into a `Concat` of `K`-atoms (`S_routeF` of the trinity boundaries).
        /// `^ $` become `Look(Start/End)` atoms (pattern-global anchors, extracted at lowering); `\b`
        /// a `Look(WordB)` atom; runs of single-byte `{1,1}` classes fuse into one `Run` (the K-fusion
        /// the membership prefilter needs). `in_group` is `true` when parsing a `(...)` body, so a
        /// `)` here ends the group rather than falling through to `class_atom`'s literal catch-all —
        /// at the top level (`in_group = false`) a bare `)` stays a literal, unchanged from before,
        /// so a pattern that never uses `(` at all is byte-for-byte the same as pre-group-support.
        ///
        /// A mid-sequence group — `Re::Alt` pushed here as one atom among literals/classes — is not
        /// a fourth `Seg` case (the trinity stays three): `lower_concat_multi` Cartesian-expands it
        /// into multiple full flat sequences at lowering time instead, one per branch combination.
        fn alt(&mut self, in_group: bool) -> Result<Re, String> {
            let mut segs: Vec<Re> = Vec::new();
            while let Some(c) = self.peek() {
                if c == b'|' {
                    break;
                }
                if in_group && c == b')' {
                    break;
                }
                if c == b'^' {
                    self.i += 1;
                    segs.push(Re::Atom(K::Look(Look::Start)));
                    continue;
                }
                if c == b'$' {
                    self.i += 1;
                    segs.push(Re::Atom(K::Look(Look::End)));
                    continue;
                }
                if c == b'\\' && self.s.get(self.i + 1) == Some(&b'b') {
                    self.i += 2;
                    segs.push(Re::Atom(K::Look(Look::WordB)));
                    continue;
                }
                if c == b'(' {
                    self.i += 1;
                    // optional non-capturing marker `(?:...)` — tregex never captures, so `(...)`
                    // and `(?:...)` are the same construct; both just group an alternation.
                    if self.peek() == Some(b'?') && self.s.get(self.i + 1) == Some(&b':') {
                        self.i += 2;
                    }
                    if self.depth >= MAX_GROUP_DEPTH {
                        return Err(format!("groups nested deeper than {MAX_GROUP_DEPTH}"));
                    }
                    self.depth += 1;
                    let mut branches = Vec::new();
                    loop {
                        branches.push(self.alt(true)?);
                        if self.peek() == Some(b'|') {
                            self.i += 1;
                        } else {
                            break;
                        }
                    }
                    self.depth -= 1;
                    if self.peek() != Some(b')') {
                        return Err("unterminated group".to_string());
                    }
                    self.i += 1;
                    // Quantifying the GROUP itself (`(a|b)+`, `(a|b){2,3}`) is a real, separate
                    // feature (repeat N times, not "pick one branch once") this doesn't implement —
                    // fail to compile rather than silently treat the quantifier as a no-op or as
                    // applying to whatever comes next.
                    if matches!(self.peek(), Some(b'*' | b'+' | b'?' | b'{')) {
                        return Err("quantified groups are not supported".to_string());
                    }
                    segs.push(Re::Alt(branches));
                    continue;
                }
                let mask = self.class_atom()?;
                let (min, max) = self.quant()?;
                let single = (0..256).filter(|&x| mask[x]).collect::<Vec<_>>();
                if min == 1 && max == 1 && single.len() == 1 {
                    // a one-byte exact boundary — fuse into the preceding run (the membership prefix)
                    let b = single[0] as u8;
                    if let Some(Re::Atom(K::Run(v))) = segs.last_mut() {
                        v.push(b);
                    } else {
                        segs.push(Re::Atom(K::Run(vec![b])));
                    }
                } else {
                    // every other boundary is one `K::Class` with its spinor `[min,max]` folded in
                    segs.push(Re::Atom(K::Class(Box::new(mask), min, max)));
                }
            }
            Ok(Re::Concat(segs))
        }
    }

    /// surface syntax → the universal Term (`Re`). Strips one optional non-capturing outer group.
    pub fn parse(pat: &str) -> Result<Re, String> {
        if pat.len() > MAX_PATTERN_LEN {
            return Err(format!("pattern is longer than {MAX_PATTERN_LEN} bytes"));
        }
        let trimmed = pat
            .strip_prefix("(?:")
            .or_else(|| pat.strip_prefix('('))
            .and_then(|s| s.strip_suffix(')'))
            .filter(|s| !s.contains('('))
            .unwrap_or(pat);
        let mut p = P {
            s: trimmed.as_bytes(),
            i: 0,
            depth: 0,
        };
        let mut alts = Vec::new();
        loop {
            alts.push(p.alt(false)?);
            if p.peek() == Some(b'|') {
                p.i += 1;
            } else {
                break;
            }
        }
        Ok(if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Re::Alt(alts)
        })
    }

    /// LOWER the universal Term to flat segment sequences — the *summary* the literal-index hooks
    /// ([`super::Regex::required_literals`], [`super::Regex::alt_literals`]) read; matching does not
    /// use it (see `nfa`): `(alts, anchored_start, anchored_end)`. `Look(Start/End)` are extracted
    /// to the global anchor flags; every other `K` becomes a `Seg`, a `Class` carrying its
    /// `[min,max]` and a `{1,1}` single-byte literal staying a `Run`.
    ///
    /// A group is listed as one sequence per branch combination, up to `MAX_LOWERED_SEQUENCES`; a
    /// concat with more combinations than that is summarised as ONE sequence of its own literals
    /// with each group as a single "any bytes" segment, which only makes the summary less precise
    /// (the group's text is no longer reported as a required literal), never wrong.
    pub fn lower(re: &Re) -> (Vec<Vec<Seg>>, bool, bool) {
        let mut as0 = false;
        let mut ae0 = false;
        let alts = match re {
            Re::Alt(branches) => branches
                .iter()
                .flat_map(|b| lower_concat_multi(b, &mut as0, &mut ae0))
                .collect(),
            other => lower_concat_multi(other, &mut as0, &mut ae0),
        };
        (alts, as0, ae0)
    }

    /// Lower ONE alternative (concat) to all its expanded flat segment-sequences — plural, because
    /// a mid-sequence group (`Re::Alt` pushed as one atom by `P::alt`) Cartesian-expands into
    /// multiple full sequences, one per branch combination: `Seg` stays the three-case trinity
    /// (no Group case), so `foo(a|b)bar` becomes the two flat sequences for `fooabar`/`foobbar`
    /// here, at lowering time, rather than at match time.
    ///
    /// A concat with NO embedded group still produces exactly one sequence (the `sequences` fold
    /// starts at `vec![Vec::new()]` and every literal/class/`\b` atom extends every
    /// sequence-in-progress in place, so a plain literal/class pattern is unaffected — same output
    /// as the old single-sequence `lower_concat` for that case).
    fn lower_concat_multi(re: &Re, as0: &mut bool, ae0: &mut bool) -> Vec<Vec<Seg>> {
        let body: &[Re] = match re {
            Re::Concat(v) => v,
            single => std::slice::from_ref(single),
        };
        let mut sequences: Vec<Vec<Seg>> = vec![Vec::new()];
        for atom in body {
            match atom {
                Re::Atom(K::Look(Look::Start)) => *as0 = true,
                Re::Atom(K::Look(Look::End)) => *ae0 = true,
                Re::Atom(K::Look(Look::WordB)) => {
                    for seq in &mut sequences {
                        seq.push(Seg::WordB);
                    }
                }
                Re::Atom(K::Run(v)) => {
                    for seq in &mut sequences {
                        seq.push(Seg::Lit(v.clone()));
                    }
                }
                Re::Atom(K::Class(m, min, max)) => {
                    for seq in &mut sequences {
                        seq.push(Seg::Class {
                            mask: m.clone(),
                            min: *min,
                            max: *max,
                        });
                    }
                }
                Re::Alt(branches) => {
                    let branch_seqs: Vec<Vec<Vec<Seg>>> = branches
                        .iter()
                        .map(|b| lower_concat_multi(b, as0, ae0))
                        .collect();
                    let per_prefix: usize = branch_seqs.iter().map(Vec::len).sum();
                    if sequences.len().saturating_mul(per_prefix) > MAX_LOWERED_SEQUENCES {
                        return vec![summarise(body, as0, ae0)];
                    }
                    let mut next = Vec::with_capacity(sequences.len() * per_prefix.max(1));
                    for branch_seqs in &branch_seqs {
                        for prefix in &sequences {
                            for suffix in branch_seqs {
                                let mut combined = prefix.clone();
                                combined.extend(suffix.iter().cloned());
                                next.push(combined);
                            }
                        }
                    }
                    sequences = next;
                }
                Re::Concat(_) => {
                    // A bare nested Concat (not produced by the current parser, which only ever
                    // pushes `Re::Alt` for a group) — handled the same structural way for safety.
                    let inner = lower_concat_multi(atom, as0, ae0);
                    if sequences.len().saturating_mul(inner.len()) > MAX_LOWERED_SEQUENCES {
                        return vec![summarise(body, as0, ae0)];
                    }
                    let mut next = Vec::with_capacity(sequences.len() * inner.len().max(1));
                    for prefix in &sequences {
                        for suffix in &inner {
                            let mut combined = prefix.clone();
                            combined.extend(suffix.iter().cloned());
                            next.push(combined);
                        }
                    }
                    sequences = next;
                }
            }
        }
        sequences
    }

    /// `.*` over every byte: the stand-in for a group the lowering declines to expand.
    fn any_bytes() -> Seg {
        Seg::Class {
            mask: Box::new([true; 256]),
            min: 0,
            max: usize::MAX,
        }
    }

    /// ONE sequence for a concat whose groups have too many combinations to list: its own literals,
    /// classes and `\b`s in order, each group (or nested concat) replaced by a single "anything".
    fn summarise(body: &[Re], as0: &mut bool, ae0: &mut bool) -> Vec<Seg> {
        let mut seq: Vec<Seg> = Vec::new();
        let mut last_any = false;
        for atom in body {
            last_any = match atom {
                Re::Atom(K::Look(Look::Start)) => {
                    *as0 = true;
                    last_any
                }
                Re::Atom(K::Look(Look::End)) => {
                    *ae0 = true;
                    last_any
                }
                Re::Atom(K::Look(Look::WordB)) => {
                    seq.push(Seg::WordB);
                    false
                }
                Re::Atom(K::Run(v)) => {
                    seq.push(Seg::Lit(v.clone()));
                    false
                }
                Re::Atom(K::Class(m, min, max)) => {
                    seq.push(Seg::Class {
                        mask: m.clone(),
                        min: *min,
                        max: *max,
                    });
                    false
                }
                Re::Alt(_) | Re::Concat(_) => {
                    if !last_any {
                        seq.push(any_bytes());
                    }
                    true
                }
            };
        }
        seq
    }

    /// the engine's parse entry: surface syntax → the lowered membership program, via the Term.
    pub fn parse_lower(pat: &str) -> Result<(Vec<Vec<Seg>>, bool, bool), String> {
        Ok(lower(&parse(pat)?))
    }
}

/// The matcher: `Re` compiled to a Thompson NFA and run as a Pike VM (no backtracking).
mod nfa {
    use super::term::{is_word, Look, Re, K};

    /// Largest compiled program, in steps. A counted repetition is unrolled (`a{1000}` is 1000
    /// steps), so this bounds the time and memory a pattern can cost at compile time and the work
    /// per haystack byte at match time.
    pub const MAX_PROGRAM: usize = 1 << 16;

    /// A byte set, one bit per byte value.
    type ByteSet = [u64; 4];

    fn contains(set: &ByteSet, b: u8) -> bool {
        set[(b >> 6) as usize] >> (b & 63) & 1 == 1
    }

    #[derive(Clone)]
    enum Inst {
        Byte(u8),
        Class(ByteSet),
        /// Try the first target, then the second (the order is the match priority).
        Split(usize, usize),
        Jmp(usize),
        Assert(Look),
        Match,
    }

    pub struct Program {
        insts: Vec<Inst>,
        /// Bytes a match can begin with; `None` when a match can be empty (or so the closure of the
        /// start state cannot tell), which rules out skipping ahead.
        first: Option<ByteSet>,
        /// No match can start past the first byte (every path begins with `^`).
        start_only_at_zero: bool,
    }

    struct Compiler {
        insts: Vec<Inst>,
    }

    impl Compiler {
        fn push(&mut self, i: Inst) -> Result<usize, String> {
            if self.insts.len() >= MAX_PROGRAM {
                return Err(format!(
                    "pattern is too large (compiled program over {MAX_PROGRAM} steps)"
                ));
            }
            self.insts.push(i);
            Ok(self.insts.len() - 1)
        }

        fn re(&mut self, re: &Re) -> Result<(), String> {
            match re {
                Re::Atom(K::Run(bytes)) => {
                    for &b in bytes {
                        self.push(Inst::Byte(b))?;
                    }
                }
                Re::Atom(K::Look(l)) => {
                    self.push(Inst::Assert(*l))?;
                }
                Re::Atom(K::Class(mask, min, max)) => self.class(mask, *min, *max)?,
                Re::Concat(items) => {
                    for item in items {
                        self.re(item)?;
                    }
                }
                Re::Alt(branches) => self.alt(branches)?,
            }
            Ok(())
        }

        /// `b0 | b1 | …`: `Split(b0, rest)`, `b0`, `Jmp end`, `Split(b1, rest')`, … The earlier a
        /// branch, the higher its priority.
        fn alt(&mut self, branches: &[Re]) -> Result<(), String> {
            let mut jumps = Vec::new();
            for (i, branch) in branches.iter().enumerate() {
                if i + 1 < branches.len() {
                    let split = self.push(Inst::Split(0, 0))?;
                    self.re(branch)?;
                    jumps.push(self.push(Inst::Jmp(0))?);
                    let next = self.insts.len();
                    self.insts[split] = Inst::Split(split + 1, next);
                } else {
                    self.re(branch)?;
                }
            }
            let end = self.insts.len();
            for j in jumps {
                self.insts[j] = Inst::Jmp(end);
            }
            Ok(())
        }

        /// A greedy `class{min,max}`: `min` mandatory steps, then either a loop (`max` unbounded) or
        /// `max - min` optional steps that all leave to the same exit.
        fn class(&mut self, mask: &[bool; 256], min: usize, max: usize) -> Result<(), String> {
            let mut set: ByteSet = [0; 4];
            for (b, &on) in mask.iter().enumerate() {
                if on {
                    set[b >> 6] |= 1 << (b & 63);
                }
            }
            let step = |set: ByteSet| -> Inst {
                let members: Vec<u8> = (0..=255u8).filter(|&b| contains(&set, b)).collect();
                match members.as_slice() {
                    [b] => Inst::Byte(*b),
                    _ => Inst::Class(set),
                }
            };
            let step = step(set);
            for _ in 0..min {
                self.push(step.clone())?;
            }
            if max == usize::MAX {
                let split = self.push(Inst::Split(0, 0))?;
                self.push(step)?;
                self.push(Inst::Jmp(split))?;
                self.insts[split] = Inst::Split(split + 1, self.insts.len());
            } else {
                let mut splits = Vec::new();
                for _ in 0..max - min {
                    splits.push(self.push(Inst::Split(0, 0))?);
                    self.push(step.clone())?;
                }
                let end = self.insts.len();
                for s in splits {
                    self.insts[s] = Inst::Split(s + 1, end);
                }
            }
            Ok(())
        }
    }

    pub fn compile(re: &Re) -> Result<Program, String> {
        let mut c = Compiler { insts: Vec::new() };
        c.re(re)?;
        c.push(Inst::Match)?;
        let insts = c.insts;

        // What can a match begin with? Walk the epsilon closure of the start state, letting every
        // assertion pass (so the set can only be too big, never too small).
        let mut first: ByteSet = [0; 4];
        let mut empty_ok = false;
        let mut any_consumer = false;
        let mut seen = vec![false; insts.len()];
        let mut stack = vec![0usize];
        while let Some(pc) = stack.pop() {
            if std::mem::replace(&mut seen[pc], true) {
                continue;
            }
            match &insts[pc] {
                Inst::Byte(b) => {
                    first[(*b >> 6) as usize] |= 1 << (*b & 63);
                    any_consumer = true;
                }
                Inst::Class(set) => {
                    for (w, bits) in first.iter_mut().zip(set) {
                        *w |= *bits;
                    }
                    any_consumer = true;
                }
                Inst::Split(a, b) => {
                    stack.push(*b);
                    stack.push(*a);
                }
                Inst::Jmp(t) => stack.push(*t),
                Inst::Assert(_) => stack.push(pc + 1),
                Inst::Match => empty_ok = true,
            }
        }

        // Could a match start at a position > 0? Same walk, but `^` now fails.
        let mut seen = vec![false; insts.len()];
        let mut stack = vec![0usize];
        let mut live_past_zero = false;
        while let Some(pc) = stack.pop() {
            if std::mem::replace(&mut seen[pc], true) {
                continue;
            }
            match &insts[pc] {
                Inst::Byte(_) | Inst::Class(_) | Inst::Match => live_past_zero = true,
                Inst::Split(a, b) => {
                    stack.push(*b);
                    stack.push(*a);
                }
                Inst::Jmp(t) => stack.push(*t),
                Inst::Assert(Look::Start) => {}
                Inst::Assert(_) => stack.push(pc + 1),
            }
        }

        Ok(Program {
            insts,
            first: if empty_ok || !any_consumer {
                None
            } else {
                Some(first)
            },
            start_only_at_zero: !live_past_zero,
        })
    }

    /// The live threads at one haystack position, in priority order, one per NFA state.
    struct Threads {
        dense: Vec<usize>,
        sparse: Vec<usize>,
        /// Where each thread's match began, indexed by state.
        start: Vec<usize>,
    }

    impl Threads {
        fn new(n: usize) -> Threads {
            Threads {
                dense: Vec::with_capacity(n),
                sparse: vec![0; n],
                start: vec![0; n],
            }
        }
        fn contains(&self, pc: usize) -> bool {
            let i = self.sparse[pc];
            i < self.dense.len() && self.dense[i] == pc
        }
        fn insert(&mut self, pc: usize, start: usize) {
            self.sparse[pc] = self.dense.len();
            self.dense.push(pc);
            self.start[pc] = start;
        }
    }

    /// Reusable search state (`O(program)`), so `find_iter` allocates it once.
    pub struct Scratch {
        clist: Threads,
        nlist: Threads,
        stack: Vec<usize>,
    }

    impl Scratch {
        pub fn new(prog: &Program) -> Scratch {
            Scratch {
                clist: Threads::new(prog.insts.len()),
                nlist: Threads::new(prog.insts.len()),
                stack: Vec::new(),
            }
        }
    }

    fn holds(look: Look, hay: &[u8], pos: usize) -> bool {
        match look {
            Look::Start => pos == 0,
            Look::End => pos == hay.len(),
            Look::WordB => {
                let before = pos > 0 && is_word(hay[pos - 1]);
                let after = pos < hay.len() && is_word(hay[pos]);
                before != after
            }
        }
    }

    /// Add the thread `pc` and everything reachable from it without consuming a byte, as seen from
    /// position `pos`, in priority order (depth-first; the first branch of a split before the second).
    fn add(
        prog: &Program,
        list: &mut Threads,
        stack: &mut Vec<usize>,
        pc: usize,
        start: usize,
        hay: &[u8],
        pos: usize,
    ) {
        stack.push(pc);
        while let Some(pc) = stack.pop() {
            if list.contains(pc) {
                continue;
            }
            list.insert(pc, start);
            match &prog.insts[pc] {
                Inst::Jmp(t) => stack.push(*t),
                Inst::Split(a, b) => {
                    stack.push(*b);
                    stack.push(*a);
                }
                Inst::Assert(look) => {
                    if holds(*look, hay, pos) {
                        stack.push(pc + 1);
                    }
                }
                Inst::Byte(_) | Inst::Class(_) | Inst::Match => {}
            }
        }
    }

    /// The leftmost-first match that starts at or after `at` (assertions see the whole haystack).
    /// With `any`, return as soon as some match is known to exist (for `is_match`).
    pub fn search(
        prog: &Program,
        scratch: &mut Scratch,
        hay: &[u8],
        at: usize,
        any: bool,
    ) -> Option<(usize, usize)> {
        let n = hay.len();
        let Scratch {
            clist,
            nlist,
            stack,
        } = scratch;
        clist.dense.clear();
        nlist.dense.clear();
        let mut matched: Option<(usize, usize)> = None;
        let mut pos = at;
        loop {
            let can_start = matched.is_none() && !(pos > 0 && prog.start_only_at_zero);
            if can_start {
                if clist.dense.is_empty() {
                    if let Some(first) = &prog.first {
                        while pos < n && !contains(first, hay[pos]) {
                            pos += 1;
                        }
                        if pos >= n {
                            break;
                        }
                    }
                }
                // the new start has the lowest priority: after every thread already running
                add(prog, clist, stack, 0, pos, hay, pos);
            }
            if clist.dense.is_empty() {
                if matched.is_some() || pos >= n || (pos > 0 && prog.start_only_at_zero) {
                    break;
                }
                pos += 1;
                continue;
            }
            nlist.dense.clear();
            let byte = hay.get(pos).copied();
            for k in 0..clist.dense.len() {
                let pc = clist.dense[k];
                let start = clist.start[pc];
                match &prog.insts[pc] {
                    Inst::Match => {
                        matched = Some((start, pos));
                        if any {
                            return matched;
                        }
                        // every thread after this one has lower priority: drop them
                        break;
                    }
                    Inst::Byte(b) => {
                        if byte == Some(*b) {
                            add(prog, nlist, stack, pc + 1, start, hay, pos + 1);
                        }
                    }
                    Inst::Class(set) => {
                        if byte.is_some_and(|c| contains(set, c)) {
                            add(prog, nlist, stack, pc + 1, start, hay, pos + 1);
                        }
                    }
                    Inst::Split(..) | Inst::Jmp(_) | Inst::Assert(_) => {}
                }
            }
            std::mem::swap(clist, nlist);
            if pos >= n {
                break;
            }
            pos += 1;
        }
        matched
    }
}

pub use crate::map::hashtrinity::tregex::nfa::MAX_PROGRAM;
use crate::map::hashtrinity::tregex::term::{Re, Seg, K};
pub use crate::map::hashtrinity::tregex::term::{MAX_GROUP_DEPTH, MAX_PATTERN_LEN, MAX_REPEAT};

/// A match span (byte offsets, `end` exclusive) — the `regex::Match` shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    pub start: usize,
    pub end: usize,
}
impl Match {
    #[must_use]
    pub fn start(&self) -> usize {
        self.start
    }
    #[must_use]
    pub fn end(&self) -> usize {
        self.end
    }
    #[must_use]
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start..self.end
    }
}

const C: u64 = 0x9E37_79B9_7F4A_7C15;

/// A pattern that is a set of plain literals (`foo|bar|baz`, or one literal): no classes, anchors or
/// groups, every literal non-empty. Searched by the membership prefilter alone, no NFA.
struct Literals {
    /// The literals in written order (their order is their priority, as in any alternation).
    lits: Vec<Vec<u8>>,
    // literal-prefix prefilter: the first `k` bytes of every literal, `k` the shortest length (<= 8)
    k: usize,
    lit_bits: u32,
    lit_seen: Vec<u64>,
    lit_map: std::collections::HashMap<u64, Vec<usize>>,
    /// `Some(bytes)` when the whole pattern is one literal — routes to the register XOR-compare
    /// (`simd::find_literal`) instead of the hash-membership L2 gather.
    single_lit: Option<Vec<u8>>,
    /// `Some` when the pattern is a small (≤8 distinct first bytes) literal alternation — routes to
    /// the register first-byte compares (`simd::find_multi`) instead of the L2 hash membership.
    multi_lit: Option<simd::MultiLit>,
}

/// The compiled pattern: a literal set on the membership prefilter, or anything else as an NFA.
pub struct Regex {
    literals: Option<Literals>,
    prog: nfa::Program,
    /// The flat per-alternative segment summary the literal-index hooks read (see `term::lower`).
    alts: Vec<Vec<Seg>>,
    anchored_start: bool,
    anchored_end: bool,
    src: String,
}

/// the first `k` bytes at `i` as a little-endian u64 (`k <= 8`); caller ensures `i + k <= n`.
#[inline]
fn window(hay: &[u8], i: usize, _k: usize, kmask: u64) -> u64 {
    let n = hay.len();
    if i + 8 <= n {
        (unsafe { (hay.as_ptr().add(i) as *const u64).read_unaligned() }) & kmask
    } else {
        let mut pre = [0u8; 8];
        let avail = (n - i).min(8);
        pre[..avail].copy_from_slice(&hay[i..i + avail]);
        u64::from_le_bytes(pre) & kmask
    }
}

/// index of the first CONSUMING seg (skip leading zero-width `\b`), if any.
fn lead_idx(segs: &[Seg]) -> Option<usize> {
    segs.iter().position(|s| !matches!(s, Seg::WordB))
}

/// The literals of a pattern that is nothing but literals, in written order.
fn pure_literals(re: &Re) -> Option<Vec<Vec<u8>>> {
    fn one(re: &Re) -> Option<Vec<u8>> {
        match re {
            Re::Concat(v) => match v.as_slice() {
                [Re::Atom(K::Run(b))] if !b.is_empty() => Some(b.clone()),
                _ => None,
            },
            _ => None,
        }
    }
    match re {
        Re::Alt(branches) => branches.iter().map(one).collect(),
        other => one(other).map(|l| vec![l]),
    }
}

impl Literals {
    fn new(lits: Vec<Vec<u8>>) -> Literals {
        let k = lits.iter().map(Vec::len).min().unwrap_or(1).clamp(1, 8);
        let kmask = if k >= 8 {
            u64::MAX
        } else {
            (1u64 << (8 * k)) - 1
        };
        let lit_bits = 18u32; // 256 Kbit = 32 KB seen-table (L1/L2)
        let mut lit_seen = vec![0u64; (1usize << lit_bits) / 64];
        let mut lit_map: std::collections::HashMap<u64, Vec<usize>> =
            std::collections::HashMap::new();
        for (ai, v) in lits.iter().enumerate() {
            let mut pre = [0u8; 8];
            pre[..k].copy_from_slice(&v[..k]);
            let w = u64::from_le_bytes(pre) & kmask;
            let s = (w.wrapping_mul(C) >> (64 - lit_bits)) as usize;
            lit_seen[s >> 6] |= 1u64 << (s & 63);
            lit_map.entry(w).or_default().push(ai);
        }
        let single_lit = if lits.len() == 1 {
            Some(lits[0].clone())
        } else {
            None
        };
        // a small alternation? → register first-byte compares (no hash table).
        let mut fbs: Vec<u8> = lits.iter().map(|v| v[0]).collect();
        fbs.sort_unstable();
        fbs.dedup();
        let multi_lit = if single_lit.is_none() && fbs.len() <= 8 {
            let mut by_first: Box<[Vec<Vec<u8>>; 256]> =
                Box::new(std::array::from_fn(|_| Vec::new()));
            for v in &lits {
                by_first[v[0] as usize].push(v.clone());
            }
            Some(simd::MultiLit { fbs, by_first })
        } else {
            None
        };
        Literals {
            lits,
            k,
            lit_bits,
            lit_seen,
            lit_map,
            single_lit,
            multi_lit,
        }
    }

    /// The literal that wins at `i` (the first in written order that is there), if any.
    #[inline]
    fn at(&self, hay: &[u8], i: usize, kmask: u64) -> Option<usize> {
        let n = hay.len();
        if i + self.k > n {
            return None;
        }
        let w = window(hay, i, self.k, kmask);
        let s = (w.wrapping_mul(C) >> (64 - self.lit_bits)) as usize;
        if self.lit_seen[s >> 6] & (1u64 << (s & 63)) == 0 {
            return None;
        }
        for &ai in self.lit_map.get(&w)? {
            let lit = &self.lits[ai];
            if i + lit.len() <= n && &hay[i..i + lit.len()] == lit.as_slice() {
                return Some(i + lit.len());
            }
        }
        None
    }

    fn kmask(&self) -> u64 {
        if self.k >= 8 {
            u64::MAX
        } else {
            (1u64 << (8 * self.k)) - 1
        }
    }

    fn find(&self, hay: &[u8]) -> Option<Match> {
        if let Some(lit) = &self.single_lit {
            return simd::find_one(hay, lit);
        }
        let kmask = self.kmask();
        (0..hay.len()).find_map(|i| self.at(hay, i, kmask).map(|end| Match { start: i, end }))
    }

    fn find_iter(&self, hay: &[u8], out: &mut Vec<Match>) {
        if let Some(lit) = &self.single_lit {
            simd::find_literal(hay, lit, out);
        } else if let Some(m) = &self.multi_lit {
            simd::find_multi(hay, m, out);
        } else {
            let kmask = self.kmask();
            let mut i = 0;
            while i < hay.len() {
                match self.at(hay, i, kmask) {
                    Some(end) => {
                        out.push(Match { start: i, end });
                        i = end;
                    }
                    None => i += 1,
                }
            }
        }
    }
}

impl Regex {
    /// Compile a pattern (the `regex::Regex::new` shape). The surface syntax is parsed into the
    /// universal Term (`term::parse`); a pure-literal pattern keeps the membership prefilter, any
    /// other compiles to an NFA (`nfa::compile`). A pattern over a limit ([`MAX_REPEAT`],
    /// [`MAX_PROGRAM`], [`MAX_GROUP_DEPTH`]) is an `Err`.
    pub fn new(pat: &str) -> Result<Regex, String> {
        let re = term::parse(pat)?;
        let prog = nfa::compile(&re)?;
        let (alts, anchored_start, anchored_end) = term::lower(&re);
        let literals = pure_literals(&re).map(Literals::new);
        Ok(Regex {
            literals,
            prog,
            alts,
            anchored_start,
            anchored_end,
            src: pat.to_string(),
        })
    }

    /// Whether the pattern matches anywhere in `hay`.
    #[must_use]
    pub fn is_match(&self, hay: &[u8]) -> bool {
        if let Some(l) = &self.literals {
            return l.find(hay).is_some();
        }
        let mut scratch = nfa::Scratch::new(&self.prog);
        nfa::search(&self.prog, &mut scratch, hay, 0, true).is_some()
    }

    /// Leftmost match (the `regex` default semantics).
    #[must_use]
    pub fn find(&self, hay: &[u8]) -> Option<Match> {
        if let Some(l) = &self.literals {
            return l.find(hay);
        }
        let mut scratch = nfa::Scratch::new(&self.prog);
        nfa::search(&self.prog, &mut scratch, hay, 0, false)
            .map(|(start, end)| Match { start, end })
    }

    /// All non-overlapping leftmost matches, as `regex::find_iter` reports them: after a match the
    /// search resumes at its end, and an empty match right after the previous match is skipped.
    #[must_use]
    pub fn find_iter(&self, hay: &[u8]) -> Vec<Match> {
        let mut out = Vec::new();
        if let Some(l) = &self.literals {
            l.find_iter(hay, &mut out);
            return out;
        }
        let n = hay.len();
        let mut scratch = nfa::Scratch::new(&self.prog);
        let mut at = 0usize;
        let mut last_end: Option<usize> = None;
        while at <= n {
            let Some((mut start, mut end)) = nfa::search(&self.prog, &mut scratch, hay, at, false)
            else {
                break;
            };
            if start == end && Some(end) == last_end {
                // an empty match where the last match ended: look again one byte further
                at += 1;
                if at > n {
                    break;
                }
                match nfa::search(&self.prog, &mut scratch, hay, at, false) {
                    Some(m) => (start, end) = m,
                    None => break,
                }
            }
            out.push(Match { start, end });
            at = end;
            last_end = Some(end);
        }
        out
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.src
    }

    /// true iff the pattern is a pure literal set (routes entirely through the membership prefilter)
    #[must_use]
    pub fn is_membership(&self) -> bool {
        !self.alts.is_empty()
            && self
                .alts
                .iter()
                .all(|segs| matches!(lead_idx(segs).map(|li| &segs[li]), Some(Seg::Lit(_))))
    }

    /// The literals that EVERY match must contain — the index prefilter hook. For a single
    /// (unanchored) alternative, all its `Lit` segments are required by the concat, so a match's
    /// line must contain each. `None` for a top-level alternation (no single literal is guaranteed)
    /// or a pattern with no literal — the caller must then scan rather than narrow.
    #[must_use]
    pub fn required_literals(&self) -> Option<Vec<Vec<u8>>> {
        if self.anchored_start || self.anchored_end || self.alts.len() != 1 {
            return None;
        }
        let lits: Vec<Vec<u8>> = self.alts[0]
            .iter()
            .filter_map(|s| match s {
                Seg::Lit(v) => Some(v.clone()),
                _ => None,
            })
            .filter(|v| v.len() >= 3)
            .collect();
        if lits.is_empty() {
            None
        } else {
            Some(lits)
        }
    }

    /// The regex-as-trigram-query hook (csearch-style). One entry per top-level alternative; each is
    /// that alternative's required literals (≥3 bytes — its concat ⇒ AND). The pattern matches iff
    /// SOME alternative matches (⇒ OR), so the index query is `OR over alts (AND over its literals)`.
    /// An EMPTY inner vec = that alternative has no usable literal (it can match "anything"), so the
    /// caller cannot narrow and must scan. Anchored patterns return one empty entry (scan).
    #[must_use]
    pub fn alt_literals(&self) -> Vec<Vec<Vec<u8>>> {
        if self.anchored_start || self.anchored_end {
            return vec![Vec::new()];
        }
        self.alts
            .iter()
            .map(|alt| {
                alt.iter()
                    .filter_map(|s| match s {
                        Seg::Lit(v) => Some(v.clone()),
                        _ => None,
                    })
                    .filter(|v| v.len() >= 3)
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod regex_group_tests {
    //! `Regex::new`/`is_match` had no test coverage at all before this module (the `parity.rs`
    //! file this module's own doc comment references does not exist in this repo) — these tests
    //! cover both the pre-existing no-group behavior (regression safety: a plain literal/class
    //! pattern must parse and match exactly as it did before `(`/`)` gained real meaning) and the
    //! new mid-sequence/nested group support this commit adds.
    use super::Regex;

    #[test]
    fn plain_literal_and_class_still_work_unaffected_by_group_support() {
        let re = Regex::new("ab.c*d").unwrap();
        assert!(re.is_match(b"xxabXcccdyy"));
        assert!(re.is_match(b"abXd"));
        assert!(!re.is_match(b"abXce"));
    }

    #[test]
    fn top_level_alternation_still_works_unaffected() {
        let re = Regex::new("cat|dog").unwrap();
        assert!(re.is_match(b"I have a cat"));
        assert!(re.is_match(b"I have a dog"));
        assert!(!re.is_match(b"I have a bird"));
    }

    #[test]
    fn whole_pattern_wrapped_in_one_group_still_works() {
        let re = Regex::new("(cat|dog)").unwrap();
        assert!(re.is_match(b"cat"));
        assert!(re.is_match(b"dog"));
        assert!(!re.is_match(b"bird"));
    }

    /// THE bug this commit fixes: a real Stripe-secret-key-shaped pattern
    /// (`prefix(alt1|alt2)_suffix`), the exact shape the parser silently mishandled before (no
    /// `(` handling in `alt()` at all — `(`/`)` fell through to the literal catch-all, so
    /// `sk_(test|live)_...` parsed as two bogus top-level alternatives each requiring a LITERAL
    /// paren character, matching nothing real).
    #[test]
    fn mid_sequence_alternation_group_matches_either_branch() {
        let re = Regex::new("sk_(test|live)_[0-9A-Za-z]{24,}").unwrap();
        // Built at run time: a contiguous `sk_live_<24 chars>` literal looks like a real key to
        // secret scanners, and these are fixtures.
        let body = "abcdefghijklmnopqrstuvwx";
        assert!(re.is_match(format!("sk_{}_{body}", "live").as_bytes()));
        assert!(re.is_match(format!("sk_{}_{body}", "test").as_bytes()));
        assert!(!re.is_match(format!("sk_{}_{body}", "prod").as_bytes()));
        // Too short for the {24,} class must still not match.
        assert!(!re.is_match(b"sk_live_short"));
    }

    #[test]
    fn mid_sequence_group_with_literal_before_and_after() {
        let re = Regex::new("foo(a|b|c)bar").unwrap();
        assert!(re.is_match(b"fooabar"));
        assert!(re.is_match(b"foobbar"));
        assert!(re.is_match(b"foocbar"));
        assert!(!re.is_match(b"foodbar"));
    }

    #[test]
    fn non_capturing_group_syntax_is_accepted_identically() {
        let re = Regex::new("foo(?:a|b)bar").unwrap();
        assert!(re.is_match(b"fooabar"));
        assert!(re.is_match(b"foobbar"));
        assert!(!re.is_match(b"foocbar"));
    }

    #[test]
    fn two_groups_in_one_pattern_cartesian_expand() {
        let re = Regex::new("(a|b)_(x|y)").unwrap();
        assert!(re.is_match(b"a_x"));
        assert!(re.is_match(b"a_y"));
        assert!(re.is_match(b"b_x"));
        assert!(re.is_match(b"b_y"));
        assert!(!re.is_match(b"a_z"));
        assert!(!re.is_match(b"c_x"));
    }

    #[test]
    fn nested_groups_are_supported_recursively() {
        let re = Regex::new("go(od|ne(r|st))").unwrap();
        assert!(re.is_match(b"good")); // go + od
        assert!(re.is_match(b"goner")); // go + ne + r
        assert!(re.is_match(b"gonest")); // go + ne + st
        assert!(!re.is_match(b"gone")); // "ne" alone (no r/st) must not match
    }

    #[test]
    fn a_group_composes_with_a_following_quantified_class() {
        // The exact real-world shape from Sondera's YARA corpus (see hashtrinity-tregex-mid-
        // pattern-alternation-bug.md): a group immediately followed by more literal + a
        // quantified class, not just a group followed by nothing.
        let re = Regex::new(r"\b(echo|printf)\s+(yes|no)\b").unwrap();
        assert!(re.is_match(b"echo yes"));
        assert!(re.is_match(b"printf no"));
        assert!(!re.is_match(b"echoyes"));
    }

    #[test]
    fn case_insensitive_prefix_composes_with_a_mid_sequence_group() {
        // (?i) is applied by callers (e.g. sangsec's hit()) by lowercasing the WHOLE pattern and
        // target before compiling/matching — confirm that still works once a group is involved.
        let re = Regex::new("sk_(test|live)_[0-9a-z]{4,}").unwrap();
        assert!(re.is_match(b"sk_live_abcd"));
    }

    #[test]
    fn an_unterminated_group_fails_to_compile_rather_than_matching_wrong() {
        assert!(Regex::new("foo(a|b").is_err());
    }

    #[test]
    fn a_quantified_group_fails_to_compile_rather_than_being_silently_ignored() {
        assert!(Regex::new("(ab)+").is_err());
        assert!(Regex::new("(a|b){2,3}").is_err());
    }
}
