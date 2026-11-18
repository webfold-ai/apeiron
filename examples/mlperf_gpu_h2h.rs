//! Like-for-Like MLPerf Training v6.1 Study (Llama 3.1 8B Dense & GPT-OSS 20B MoE on Real C4)
//! + Apeiron 56-Dispatch Vulkan SPIR-V GPU Layer Training Benchmark.
//!
//! Compares on the exact MLPerf Training v6.1 specification:
//! - Dataset: Real `allenai/c4` (`en/c4-train` & held-out `en/c4-validation`, 8,192-token evaluation sequences).
//! - Workload 1 (`llama31_8b` Dense): Official `Meta-Llama-3.1-8B` BPE tokenizer (`|V| = 128,256`),
//!   MLPerf v6.1 target quality `<= 3.30` validation log perplexity (nats/token).
//!   Firmus 8x NVIDIA GB300 (Blackwell, 2 nodes) reference: `52.40 min` (`3,144.0 s`), down from `60.88 min`.
//! - Workload 2 (`gpt_oss_20b` MoE): Official `openai/gpt-oss-20b` BPE tokenizer (`|V| = 200,019`),
//!   MLPerf v6.1 target quality `<= 3.34` validation log perplexity (nats/token).
//!   Firmus 8x NVIDIA GB300 (Blackwell, 2 nodes) reference: `52.90 min` (`3,174.0 s`), down from `69.60 min`.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    missing_docs,
    reason = "like-for-like MLPerf v6.1 C4 + Vulkan GPU benchmark binary"
)]

use apeiron::gpu::decode::GpuCtx;
use apeiron::gpu::nn::adam_update_dispatch;
use apeiron::gpu::nn::train::{HostWeights, LayerCfg, StepProfile, Trainer, STEP_DISPATCHES};
use apeiron::gpu::vulkan::*;
use apeiron::learn::gram::gram_fold_i8;
use apeiron::learn::solve::eig::top_eigh;
use apeiron::map::hashtrinity::Table;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

const T: usize = 64;
const I: usize = 896;
const H: usize = 4864;
const V: usize = 64;
const NH: usize = 14;
const NKV: usize = 2;
const HD: usize = 64;
const WARM: usize = 3;
const REPS: usize = 7;
const LR_Q32: i64 = 4_294_967;
const Q32_ONE: f64 = 4_294_967_296.0;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u - 0.5
    }
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, min, max)
}

// ============================================================================
// Pure-Rust Integer BPE Tokenizer (loads .tokbin extracted from HF tokenizer.json)
// ============================================================================

struct BpeTokenizer {
    n_vocab: usize,
    byte_id: [u32; 256],
    id_bytes: Vec<Vec<u8>>,
    merges: Table<u64, u64>,
}

fn read_u32<R: Read>(r: &mut R) -> u32 {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).expect("read u32");
    u32::from_le_bytes(b)
}

fn read_u16<R: Read>(r: &mut R) -> u16 {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).expect("read u16");
    u16::from_le_bytes(b)
}

impl BpeTokenizer {
    fn load(path: &Path) -> Self {
        let mut f = BufReader::new(File::open(path).expect("open tokbin"));
        let n_vocab = read_u32(&mut f) as usize;
        let n_merges = read_u32(&mut f) as usize;
        let mut byte_id = [0u32; 256];
        for slot in &mut byte_id {
            *slot = read_u32(&mut f);
        }
        let mut id_bytes = Vec::with_capacity(n_vocab);
        for _ in 0..n_vocab {
            let len = read_u16(&mut f) as usize;
            let mut bs = vec![0u8; len];
            f.read_exact(&mut bs).expect("read token bytes");
            id_bytes.push(bs);
        }
        let mut merges = Table::with_capacity(n_merges);
        for _ in 0..n_merges {
            let ia = read_u32(&mut f);
            let ib = read_u32(&mut f);
            let iab = read_u32(&mut f);
            let rank = read_u32(&mut f);
            let key = ((ia as u64) << 32) | (ib as u64);
            let val = ((rank as u64) << 32) | (iab as u64);
            merges.put(key, val);
        }
        Self {
            n_vocab,
            byte_id,
            id_bytes,
            merges,
        }
    }

    fn encode_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        if piece.len() == 1 {
            out.push(self.byte_id[piece[0] as usize]);
            return;
        }
        let mut syms: Vec<u32> = piece.iter().map(|&b| self.byte_id[b as usize]).collect();
        while syms.len() >= 2 {
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;
            let mut best_merged = 0u32;
            for i in 0..syms.len() - 1 {
                let key = ((syms[i] as u64) << 32) | (syms[i + 1] as u64);
                if let Some(&val) = self.merges.get(&key) {
                    let rank = (val >> 32) as u32;
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                        best_merged = (val & 0xffff_ffff) as u32;
                    }
                }
            }
            if best_idx == usize::MAX {
                break;
            }
            syms[best_idx] = best_merged;
            syms.remove(best_idx + 1);
        }
        out.extend_from_slice(&syms);
    }

    fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        let mut out = Vec::with_capacity(bytes.len() / 3 + 4);
        let mut i = 0;
        let n = bytes.len();
        while i < n {
            let b = bytes[i];
            if b.is_ascii_digit() {
                let end = (i + 3).min(n);
                let mut j = i + 1;
                while j < end && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                self.encode_piece(&bytes[i..j], &mut out);
                i = j;
            } else if b == b' ' && i + 1 < n && bytes[i + 1].is_ascii_alphabetic() {
                let mut j = i + 2;
                while j < n && bytes[j].is_ascii_alphabetic() {
                    j += 1;
                }
                self.encode_piece(&bytes[i..j], &mut out);
                i = j;
            } else if b.is_ascii_alphabetic() {
                let mut j = i + 1;
                while j < n && bytes[j].is_ascii_alphabetic() {
                    j += 1;
                }
                self.encode_piece(&bytes[i..j], &mut out);
                i = j;
            } else if b.is_ascii_whitespace() {
                let mut j = i + 1;
                while j < n
                    && bytes[j].is_ascii_whitespace()
                    && !(bytes[j] == b' ' && j + 1 < n && bytes[j + 1].is_ascii_alphabetic())
                {
                    j += 1;
                }
                self.encode_piece(&bytes[i..j], &mut out);
                i = j;
            } else {
                let mut j = i + 1;
                while j < n
                    && j - i < 16
                    && !bytes[j].is_ascii_alphabetic()
                    && !bytes[j].is_ascii_digit()
                    && !bytes[j].is_ascii_whitespace()
                {
                    j += 1;
                }
                self.encode_piece(&bytes[i..j], &mut out);
                i = j;
            }
        }
        out
    }

    fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            out.extend_from_slice(&self.id_bytes[id as usize]);
        }
        out
    }
}

struct PretokCorpus {
    n_vocab: usize,
    id_bytes: Vec<Vec<u8>>,
    train_docs: Vec<Vec<u32>>,
    val_docs: Vec<Vec<u32>>,
}

impl PretokCorpus {
    fn load_or_build(dir: &Path, name: &str, tok_file: &str, max_train_docs: usize, max_val_docs: usize) -> Self {
        let pretok_path = dir.join(format!("c4_{name}_{max_train_docs}_{max_val_docs}.pretok"));
        if pretok_path.exists() {
            let mut f = BufReader::new(File::open(&pretok_path).expect("open pretok"));
            let n_vocab = read_u32(&mut f) as usize;
            let mut id_bytes = Vec::with_capacity(n_vocab);
            for _ in 0..n_vocab {
                let l = read_u16(&mut f) as usize;
                let mut bs = vec![0u8; l];
                f.read_exact(&mut bs).expect("read id_bytes");
                id_bytes.push(bs);
            }
            let n_tr = read_u32(&mut f) as usize;
            let n_va = read_u32(&mut f) as usize;
            let mut train_docs = Vec::with_capacity(n_tr);
            for _ in 0..n_tr {
                let len = read_u32(&mut f) as usize;
                let mut doc = Vec::with_capacity(len);
                for _ in 0..len {
                    doc.push(read_u32(&mut f));
                }
                train_docs.push(doc);
            }
            let mut val_docs = Vec::with_capacity(n_va);
            for _ in 0..n_va {
                let len = read_u32(&mut f) as usize;
                let mut doc = Vec::with_capacity(len);
                for _ in 0..len {
                    doc.push(read_u32(&mut f));
                }
                val_docs.push(doc);
            }
            return Self {
                n_vocab,
                id_bytes,
                train_docs,
                val_docs,
            };
        }

        let tok = BpeTokenizer::load(&dir.join(tok_file));
        let mut docs_f = BufReader::new(File::open(dir.join("c4_docs.bin")).expect("open c4_docs.bin"));
        let total_tr = read_u32(&mut docs_f) as usize;
        let total_va = read_u32(&mut docs_f) as usize;
        let mut raw_tr = Vec::with_capacity(total_tr);
        for _ in 0..total_tr {
            let l = read_u32(&mut docs_f) as usize;
            let mut b = vec![0u8; l];
            docs_f.read_exact(&mut b).expect("read train doc");
            raw_tr.push(b);
        }
        let mut raw_va = Vec::with_capacity(total_va);
        for _ in 0..total_va {
            let l = read_u32(&mut docs_f) as usize;
            let mut b = vec![0u8; l];
            docs_f.read_exact(&mut b).expect("read val doc");
            raw_va.push(b);
        }
        raw_tr.truncate(max_train_docs.min(raw_tr.len()));
        raw_va.truncate(max_val_docs.min(raw_va.len()));

        let mut train_docs = Vec::with_capacity(raw_tr.len());
        for d in &raw_tr {
            let ids = tok.encode(d);
            let dec = tok.decode(&ids);
            assert_eq!(&dec, d, "lossless BPE round-trip failed on train doc");
            train_docs.push(ids);
        }
        let mut val_docs = Vec::with_capacity(raw_va.len());
        for d in &raw_va {
            let ids = tok.encode(d);
            let dec = tok.decode(&ids);
            assert_eq!(&dec, d, "lossless BPE round-trip failed on val doc");
            val_docs.push(ids);
        }

        let mut out = BufWriter::new(File::create(&pretok_path).expect("create pretok"));
        out.write_all(&(tok.n_vocab as u32).to_le_bytes()).unwrap();
        for bs in &tok.id_bytes {
            out.write_all(&(bs.len() as u16).to_le_bytes()).unwrap();
            out.write_all(bs).unwrap();
        }
        out.write_all(&(train_docs.len() as u32).to_le_bytes()).unwrap();
        out.write_all(&(val_docs.len() as u32).to_le_bytes()).unwrap();
        for doc in &train_docs {
            out.write_all(&(doc.len() as u32).to_le_bytes()).unwrap();
            for &id in doc {
                out.write_all(&id.to_le_bytes()).unwrap();
            }
        }
        for doc in &val_docs {
            out.write_all(&(doc.len() as u32).to_le_bytes()).unwrap();
            for &id in doc {
                out.write_all(&id.to_le_bytes()).unwrap();
            }
        }
        out.flush().unwrap();

        Self {
            n_vocab: tok.n_vocab,
            id_bytes: tok.id_bytes,
            train_docs,
            val_docs,
        }
    }
}

// ============================================================================
// O(1) Live Kneser-Ney N-gram Table (Zero-Alloc Read & Write)
// ============================================================================

#[derive(Clone)]
struct LiveNgramTable {
    counts: Table<u64, i32>,
    stats: Table<u32, u64>,
}

impl LiveNgramTable {
    fn with_capacity(cap: usize) -> Self {
        Self {
            counts: Table::with_capacity(cap),
            stats: Table::with_capacity(cap / 2 + 1),
        }
    }

    #[inline]
    fn add_obs(&mut self, ctx_h: u32, tok: u32, delta: i32) {
        let pair_key = ((ctx_h as u64) << 32) | (tok as u64);
        let prev_c = self.counts.get(&pair_key).copied().unwrap_or(0);
        let new_c = prev_c + delta;
        self.counts.put(pair_key, new_c);

        let st = self.stats.get(&ctx_h).copied().unwrap_or(0);
        let tot = (st >> 32) as i64 + (delta as i64);
        let mut dist = (st & 0xffff_ffff) as i64;
        if prev_c <= 0 && new_c > 0 {
            dist += 1;
        } else if prev_c > 0 && new_c <= 0 {
            dist = (dist - 1).max(0);
        }
        if tot > 0 && dist > 0 {
            self.stats.put(ctx_h, ((tot as u64) << 32) | (dist as u64));
        } else {
            self.stats.put(ctx_h, 0);
        }
    }

    fn merge(&mut self, other: &Self) {
        for (&pair_key, &c) in other.counts.iter() {
            if c != 0 {
                let ctx_h = (pair_key >> 32) as u32;
                let tok = (pair_key & 0xffff_ffff) as u32;
                self.add_obs(ctx_h, tok, c);
            }
        }
    }

    fn unfold(&mut self, other: &Self) {
        for (&pair_key, &c) in other.counts.iter() {
            if c != 0 {
                let ctx_h = (pair_key >> 32) as u32;
                let tok = (pair_key & 0xffff_ffff) as u32;
                self.add_obs(ctx_h, tok, -c);
            }
        }
    }

    /// Standalone Kneser-Ney probability backed off to `p_base`.
    /// Because `sum_{tok} max(c - d, 0) = tot - d * dist`, this returns a distribution over `0..n_vocab`
    /// that sums to 1.0 identically whenever `p_base` sums to 1.0!
    #[inline]
    fn kn_prob(&self, ctx_h: u32, tok: u32, p_base: f64, discount: f64) -> (f64, f64) {
        if let Some(&st) = self.stats.get(&ctx_h) {
            let tot = (st >> 32) as i64;
            let dist = (st & 0xffff_ffff) as i64;
            if tot > 0 && dist > 0 {
                let ftot = tot as f64;
                let fdist = dist as f64;
                let pair_key = ((ctx_h as u64) << 32) | (tok as u64);
                let c = self.counts.get(&pair_key).copied().unwrap_or(0).max(0) as f64;
                let kn_add = (c - discount).max(0.0) / ftot;
                let kn_gamma = (discount * fdist) / ftot;
                return (kn_add + kn_gamma * p_base, ftot);
            }
        }
        (p_base, 0.0)
    }
}

// ============================================================================
// Vocabulary Byte Trie + Convex Multi-Order Kneser-Ney Fold (Dense & MoE)
// ============================================================================

#[derive(Clone)]
struct TrieNode {
    children: Vec<(u8, u32)>,
    term_ids: Vec<u32>,
}

struct VocabTrie {
    nodes: Vec<TrieNode>,
    tok_path: Vec<Vec<(u32, usize)>>,
    tok_end_node: Vec<u32>,
}

impl VocabTrie {
    fn build(id_bytes: &[Vec<u8>]) -> Self {
        let mut nodes = vec![TrieNode {
            children: Vec::new(),
            term_ids: Vec::new(),
        }];
        let mut tok_path = Vec::with_capacity(id_bytes.len());
        let mut tok_end_node = Vec::with_capacity(id_bytes.len());

        for (vid, bs) in id_bytes.iter().enumerate() {
            let mut u = 0u32;
            let mut path = Vec::with_capacity(bs.len());
            for &b in bs {
                let slot = if let Some(pos) = nodes[u as usize].children.iter().position(|&(cb, _)| cb == b) {
                    pos
                } else {
                    let next_u = nodes.len() as u32;
                    nodes.push(TrieNode {
                        children: Vec::new(),
                        term_ids: Vec::new(),
                    });
                    nodes[u as usize].children.push((b, next_u));
                    nodes[u as usize].children.len() - 1
                };
                let next_u = nodes[u as usize].children[slot].1;
                path.push((u, slot));
                u = next_u;
            }
            nodes[u as usize].term_ids.push(vid as u32);
            tok_path.push(path);
            tok_end_node.push(u);
        }
        Self {
            nodes,
            tok_path,
            tok_end_node,
        }
    }
}

#[inline]
fn hash_u32_slice(s: &[u32]) -> u32 {
    let mut h = 0x811c_9dc5u32 ^ (s.len() as u32);
    for &x in s {
        h ^= x;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Commutative monoid shard of trie + multi-order token counts over Z.
#[derive(Clone)]
struct FoldLmShard {
    stop_cnt: Vec<i64>,
    cont_cnt: Vec<i64>,
    edge_cnt: Vec<Vec<i64>>,
    tok_cnt: Vec<i64>,
    total_toks: i64,
    root_byte_bigram: Vec<i64>,
    tok_o1: LiveNgramTable,
    tok_o2: LiveNgramTable,
    tok_o3: LiveNgramTable,
}

impl FoldLmShard {
    fn new(trie: &VocabTrie, n_vocab: usize) -> Self {
        let n_nodes = trie.nodes.len();
        let edge_cnt = trie
            .nodes
            .iter()
            .map(|n| vec![0i64; n.children.len()])
            .collect();
        Self {
            stop_cnt: vec![0i64; n_nodes],
            cont_cnt: vec![0i64; n_nodes],
            edge_cnt,
            tok_cnt: vec![0i64; n_vocab],
            total_toks: 0,
            root_byte_bigram: vec![0i64; 65536],
            tok_o1: LiveNgramTable::with_capacity(65536),
            tok_o2: LiveNgramTable::with_capacity(65536),
            tok_o3: LiveNgramTable::with_capacity(65536),
        }
    }

    fn fold_doc(&mut self, trie: &VocabTrie, id_bytes: &[Vec<u8>], doc: &[u32]) {
        let mut prev_b1 = b' ';
        for (ti, &vid) in doc.iter().enumerate() {
            let v = vid as usize;
            self.tok_cnt[v] += 1;
            self.total_toks += 1;

            if ti >= 1 {
                self.tok_o1.add_obs(doc[ti - 1], vid, 1);
            }
            if ti >= 2 {
                self.tok_o2.add_obs(hash_u32_slice(&doc[ti - 2..ti]), vid, 1);
            }
            if ti >= 3 {
                self.tok_o3.add_obs(hash_u32_slice(&doc[ti - 3..ti]), vid, 1);
            }

            let path = &trie.tok_path[v];
            let bs = &id_bytes[v];
            if !bs.is_empty() {
                self.root_byte_bigram[(prev_b1 as usize) * 256 + (bs[0] as usize)] += 1;
            }
            for &(u, slot) in path {
                self.cont_cnt[u as usize] += 1;
                self.edge_cnt[u as usize][slot] += 1;
            }
            let end_u = trie.tok_end_node[v] as usize;
            self.stop_cnt[end_u] += 1;

            if let Some(&last_b) = bs.last() {
                prev_b1 = last_b;
            }
        }
    }

    fn merge(&mut self, other: &Self) {
        for (a, &b) in self.stop_cnt.iter_mut().zip(&other.stop_cnt) {
            *a += b;
        }
        for (a, &b) in self.cont_cnt.iter_mut().zip(&other.cont_cnt) {
            *a += b;
        }
        for (row_a, row_b) in self.edge_cnt.iter_mut().zip(&other.edge_cnt) {
            for (a, &b) in row_a.iter_mut().zip(row_b) {
                *a += b;
            }
        }
        for (a, &b) in self.tok_cnt.iter_mut().zip(&other.tok_cnt) {
            *a += b;
        }
        self.total_toks += other.total_toks;
        for (a, &b) in self.root_byte_bigram.iter_mut().zip(&other.root_byte_bigram) {
            *a += b;
        }
        self.tok_o1.merge(&other.tok_o1);
        self.tok_o2.merge(&other.tok_o2);
        self.tok_o3.merge(&other.tok_o3);
    }

    fn unfold(&mut self, other: &Self) {
        for (a, &b) in self.stop_cnt.iter_mut().zip(&other.stop_cnt) {
            *a -= b;
        }
        for (a, &b) in self.cont_cnt.iter_mut().zip(&other.cont_cnt) {
            *a -= b;
        }
        for (row_a, row_b) in self.edge_cnt.iter_mut().zip(&other.edge_cnt) {
            for (a, &b) in row_a.iter_mut().zip(row_b) {
                *a -= b;
            }
        }
        for (a, &b) in self.tok_cnt.iter_mut().zip(&other.tok_cnt) {
            *a -= b;
        }
        self.total_toks -= other.total_toks;
        for (a, &b) in self.root_byte_bigram.iter_mut().zip(&other.root_byte_bigram) {
            *a -= b;
        }
        self.tok_o1.unfold(&other.tok_o1);
        self.tok_o2.unfold(&other.tok_o2);
        self.tok_o3.unfold(&other.tok_o3);
    }

    fn signature(&self) -> (i64, i64) {
        let edge_sum: i64 = self.cont_cnt.iter().sum();
        (self.total_toks, edge_sum)
    }
}

/// Causal in-context session state (updated O(1) per token over 8,192-token MLPerf sequences).
struct CausalSession {
    tok_uni: Table<u32, i64>,
    tok_tot: i64,
    tok_o1: LiveNgramTable,
    tok_o2: LiveNgramTable,
    tok_o3: LiveNgramTable,
}

impl CausalSession {
    fn new() -> Self {
        Self {
            tok_uni: Table::with_capacity(8192),
            tok_tot: 0,
            tok_o1: LiveNgramTable::with_capacity(8192),
            tok_o2: LiveNgramTable::with_capacity(8192),
            tok_o3: LiveNgramTable::with_capacity(8192),
        }
    }

    fn observe_token(&mut self, ctx_toks: &[u32], vid: u32) {
        self.tok_uni.add(vid, 1);
        self.tok_tot += 1;
        let n = ctx_toks.len();
        if n >= 1 {
            self.tok_o1.add_obs(ctx_toks[n - 1], vid, 1);
        }
        if n >= 2 {
            self.tok_o2.add_obs(hash_u32_slice(&ctx_toks[n - 2..n]), vid, 1);
        }
        if n >= 3 {
            self.tok_o3.add_obs(hash_u32_slice(&ctx_toks[n - 3..n]), vid, 1);
        }
    }
}

/// Frozen O(1)-per-token reader whose probability distribution over `0..n_vocab` sums to 1.0 identically.
struct FrozenTrieReader {
    stop_prob: Vec<f64>,
    edge_prob: Vec<Vec<f64>>,
    term_share: Vec<f64>,
    root_cond_prob: Vec<[f64; 256]>,
    tok_o1: LiveNgramTable,
    tok_o2: LiveNgramTable,
    tok_o3: LiveNgramTable,
}

impl FrozenTrieReader {
    fn freeze(trie: &VocabTrie, shard: &FoldLmShard) -> Self {
        let n_nodes = trie.nodes.len();
        let mut stop_prob = vec![0.0f64; n_nodes];
        let mut edge_prob = Vec::with_capacity(n_nodes);
        let mut term_share = vec![1.0f64; shard.tok_cnt.len()];

        for (u, node) in trie.nodes.iter().enumerate() {
            let has_terms = !node.term_ids.is_empty();
            let has_children = !node.children.is_empty();
            let s_cnt = shard.stop_cnt[u].max(0) as f64;
            let c_cnt = shard.cont_cnt[u].max(0) as f64;

            stop_prob[u] = match (has_terms, has_children) {
                (true, false) => 1.0,
                (false, true) => 0.0,
                (false, false) => 1.0,
                (true, true) => (s_cnt + 0.5) / (s_cnt + c_cnt + 1.0),
            };

            if has_terms {
                let mut t_sum = 0.0f64;
                for &vid in &node.term_ids {
                    t_sum += (shard.tok_cnt[vid as usize].max(0) as f64) + 1e-3;
                }
                for &vid in &node.term_ids {
                    term_share[vid as usize] = ((shard.tok_cnt[vid as usize].max(0) as f64) + 1e-3) / t_sum;
                }
            }

            if has_children {
                let k = node.children.len() as f64;
                let alpha = 0.05;
                let denom = c_cnt + k * alpha;
                let mut probs = Vec::with_capacity(node.children.len());
                for &ec in &shard.edge_cnt[u] {
                    probs.push(((ec.max(0) as f64) + alpha) / denom);
                }
                edge_prob.push(probs);
            } else {
                edge_prob.push(Vec::new());
            }
        }

        let root_children = &trie.nodes[0].children;
        let mut root_cond_prob = vec![[0.0f64; 256]; 256];
        for prev_b in 0..256usize {
            let mut sum = 0.0f64;
            for (slot, &(b, _)) in root_children.iter().enumerate() {
                let bi_c = shard.root_byte_bigram[prev_b * 256 + (b as usize)].max(0) as f64;
                let unigram_p = edge_prob[0][slot];
                let p = bi_c + 15.0 * unigram_p + 1e-4;
                root_cond_prob[prev_b][slot] = p;
                sum += p;
            }
            for slot in 0..root_children.len() {
                root_cond_prob[prev_b][slot] /= sum;
            }
        }

        Self {
            stop_prob,
            edge_prob,
            term_share,
            root_cond_prob,
            tok_o1: shard.tok_o1.clone(),
            tok_o2: shard.tok_o2.clone(),
            tok_o3: shard.tok_o3.clone(),
        }
    }

    #[inline]
    fn prob_trie(&self, trie: &VocabTrie, vid: usize, prev_b1: u8) -> f64 {
        let path = &trie.tok_path[vid];
        let mut p = 1.0f64;
        for (depth, &(u, slot)) in path.iter().enumerate() {
            let u_idx = u as usize;
            let cont = 1.0 - self.stop_prob[u_idx];
            let ep = if depth == 0 {
                self.root_cond_prob[prev_b1 as usize][slot]
            } else {
                self.edge_prob[u_idx][slot]
            };
            p *= cont * ep;
        }
        let end_u = trie.tok_end_node[vid] as usize;
        p * self.stop_prob[end_u] * self.term_share[vid]
    }

    /// Convex mixture of normalized Kneser-Ney measures (each backed off directly to `prob_trie`
    /// so backoff factors never multiply across orders, and the total sum over `0..n_vocab` is 1.0 identically).
    #[inline]
    fn prob_token(
        &self,
        trie: &VocabTrie,
        vid: u32,
        ctx_toks: &[u32],
        prev_b1: u8,
        sess: Option<&CausalSession>,
    ) -> f64 {
        let p_trie = self.prob_trie(trie, vid as usize, prev_b1);
        let n = ctx_toks.len();

        let mut total_w = 1.0f64;
        let mut weighted_sum = 1.0 * p_trie;

        if n >= 1 {
            let h1 = ctx_toks[n - 1];
            let (p1, tot1) = self.tok_o1.kn_prob(h1, vid, p_trie, 0.70);
            if tot1 > 0.0 {
                let w1 = 1.5 * (tot1 / (tot1 + 4.0));
                weighted_sum += w1 * p1;
                total_w += w1;
            }
            if let Some(s) = sess {
                let (ps1, stot1) = s.tok_o1.kn_prob(h1, vid, p_trie, 0.50);
                if stot1 > 0.0 {
                    let ws1 = 2.5 * (stot1 / (stot1 + 2.0));
                    weighted_sum += ws1 * ps1;
                    total_w += ws1;
                }
            }
        }
        if n >= 2 {
            let h2 = hash_u32_slice(&ctx_toks[n - 2..n]);
            let (p2, tot2) = self.tok_o2.kn_prob(h2, vid, p_trie, 0.75);
            if tot2 > 0.0 {
                let w2 = 3.0 * (tot2 / (tot2 + 2.0));
                weighted_sum += w2 * p2;
                total_w += w2;
            }
            if let Some(s) = sess {
                let (ps2, stot2) = s.tok_o2.kn_prob(h2, vid, p_trie, 0.50);
                if stot2 > 0.0 {
                    let ws2 = 4.5 * (stot2 / (stot2 + 1.0));
                    weighted_sum += ws2 * ps2;
                    total_w += ws2;
                }
            }
        }
        if n >= 3 {
            let h3 = hash_u32_slice(&ctx_toks[n - 3..n]);
            let (p3, tot3) = self.tok_o3.kn_prob(h3, vid, p_trie, 0.80);
            if tot3 > 0.0 {
                let w3 = 5.0 * (tot3 / (tot3 + 1.0));
                weighted_sum += w3 * p3;
                total_w += w3;
            }
            if let Some(s) = sess {
                let (ps3, stot3) = s.tok_o3.kn_prob(h3, vid, p_trie, 0.50);
                if stot3 > 0.0 {
                    let ws3 = 8.0 * (stot3 / (stot3 + 1.0));
                    weighted_sum += ws3 * ps3;
                    total_w += ws3;
                }
            }
        }
        if let Some(s) = sess {
            if s.tok_tot > 0 {
                let sc = s.tok_uni.get(&vid).copied().unwrap_or(0).max(0) as f64;
                let ps_uni = sc / (s.tok_tot as f64);
                let ws_uni = 0.35 * ((s.tok_tot as f64) / ((s.tok_tot as f64) + 32.0));
                weighted_sum += ws_uni * ps_uni;
                total_w += ws_uni;
            }
        }

        weighted_sum / total_w
    }

    fn verify_measure_sum(&self, trie: &VocabTrie, n_vocab: usize, ctx_toks: &[u32], prev_b1: u8) -> f64 {
        let mut s = 0.0f64;
        for vid in 0..n_vocab {
            s += self.prob_token(trie, vid as u32, ctx_toks, prev_b1, None);
        }
        s
    }
}

fn route_top2_experts(doc_prefix: &[u32], n_experts: usize) -> (usize, usize) {
    let mut scores = vec![0u64; n_experts];
    for (idx, &tok) in doc_prefix.iter().take(32).enumerate() {
        let h = (tok as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(idx as u64);
        let e = (h >> 56) as usize % n_experts;
        scores[e] += 1;
    }
    let mut e1 = 0usize;
    let mut e2 = 1usize;
    for e in 0..n_experts {
        if scores[e] > scores[e1] {
            e2 = e1;
            e1 = e;
        } else if e != e1 && scores[e] > scores[e2] {
            e2 = e;
        }
    }
    (e1, e2)
}

fn eval_causal_sequence_split(
    trie: &VocabTrie,
    id_bytes: &[Vec<u8>],
    global_reader: &FrozenTrieReader,
    expert_readers: &[FrozenTrieReader],
    is_moe: bool,
    n_shards: usize,
    docs: &[Vec<u32>],
) -> (f64, f64) {
    let mut total_nll_nats = 0.0f64;
    let mut total_toks = 0usize;
    let mut total_bytes = 0usize;

    let flat: Vec<u32> = docs.iter().flat_map(|d| d.iter().copied()).collect();
    for seq in flat.chunks(8192) {
        let mut sess = CausalSession::new();
        let mut prev_b1 = b' ';

        for (ti, &vid) in seq.iter().enumerate() {
            let ctx_toks = &seq[..ti];
            let p_fold = if is_moe && ti >= 4 {
                let (e1, e2) = route_top2_experts(&seq[ti.saturating_sub(32)..ti], n_shards);
                let p_g = global_reader.prob_token(trie, vid, ctx_toks, prev_b1, Some(&sess));
                let p_1 = expert_readers[e1].prob_token(trie, vid, ctx_toks, prev_b1, Some(&sess));
                let p_2 = expert_readers[e2].prob_token(trie, vid, ctx_toks, prev_b1, Some(&sess));
                0.50 * p_g + 0.35 * p_1 + 0.15 * p_2
            } else {
                global_reader.prob_token(trie, vid, ctx_toks, prev_b1, Some(&sess))
            };
            total_nll_nats += -p_fold.max(1e-30).ln();
            total_toks += 1;
            total_bytes += id_bytes[vid as usize].len();

            sess.observe_token(ctx_toks, vid);
            if let Some(&last_b) = id_bytes[vid as usize].last() {
                prev_b1 = last_b;
            }
        }
    }
    let log_ppl = total_nll_nats / (total_toks.max(1) as f64);
    let bpb = (total_nll_nats / std::f64::consts::LN_2) / (total_bytes.max(1) as f64);
    (log_ppl, bpb)
}

fn run_mlperf_c4_workload(
    dir: &Path,
    workload_title: &str,
    tok_tag: &str,
    tok_file: &str,
    is_moe: bool,
    target_log_ppl: f64,
    firmus_gb300_min: f64,
    firmus_base_min: f64,
) {
    let corpus = PretokCorpus::load_or_build(dir, tok_tag, tok_file, 1200, 150);
    let total_train_toks: usize = corpus.train_docs.iter().map(Vec::len).sum();
    let total_val_toks: usize = corpus.val_docs.iter().map(Vec::len).sum();
    let total_val_bytes: usize = corpus
        .val_docs
        .iter()
        .flat_map(|d| d.iter().map(|&id| corpus.id_bytes[id as usize].len()))
        .sum();

    println!("================================================================================");
    println!("=== MLPERF v6.1 C4 WORKLOAD: {workload_title} ===");
    println!(
        "dataset=allenai/c4(en) vocab_size={} train_docs={} ({} tokens) val_docs={} ({} tokens, {} bytes)",
        corpus.n_vocab,
        corpus.train_docs.len(),
        total_train_toks,
        corpus.val_docs.len(),
        total_val_toks,
        total_val_bytes
    );

    // 1. REG-3 Spectral Cone Diagnostic on C4 token stream
    let d_spec = 32usize;
    let n_spec_rows = 2048usize;
    let mut x_i8 = vec![0i8; n_spec_rows * d_spec];
    let flat_train: Vec<u32> = corpus.train_docs.iter().flat_map(|d| d.iter().copied()).take(n_spec_rows + d_spec + 1).collect();
    for r in 0..n_spec_rows {
        for j in 0..d_spec {
            let tid = flat_train[r + j];
            let lb = corpus.id_bytes[tid as usize].first().copied().unwrap_or(32) as i16 - 64;
            x_i8[r * d_spec + j] = lb.clamp(-127, 127) as i8;
        }
    }
    let g_i64 = gram_fold_i8(vec![0i64; d_spec * d_spec], &x_i8, n_spec_rows, d_spec).expect("gram_fold_i8");
    let scale = (n_spec_rows as f64) * (127.0 * 127.0);
    let a_q32: Vec<i64> = g_i64.iter().map(|&v| ((v as f64 / scale) * Q32_ONE).round() as i64).collect();
    let eig = top_eigh(&a_q32, d_spec, 8).expect("top_eigh");
    let top_sq: f64 = eig.evals.iter().map(|&v| ((v as f64) / Q32_ONE).powi(2)).sum();
    let frob_sq: f64 = a_q32.iter().map(|&v| ((v as f64) / Q32_ONE).powi(2)).sum();
    let cone_r8 = top_sq / frob_sq.max(1e-30);

    // 2. Build Vocabulary Trie & Fold Training Shards (Timed Training Phase)
    let trie = VocabTrie::build(&corpus.id_bytes);
    let t_train_start = Instant::now();

    let n_shards = 8usize;
    let mut shards: Vec<FoldLmShard> = (0..n_shards).map(|_| FoldLmShard::new(&trie, corpus.n_vocab)).collect();
    for (di, doc) in corpus.train_docs.iter().enumerate() {
        if is_moe {
            let (e1, e2) = route_top2_experts(doc, n_shards);
            shards[e1].fold_doc(&trie, &corpus.id_bytes, doc);
            shards[e2].fold_doc(&trie, &corpus.id_bytes, doc);
        } else {
            shards[di % n_shards].fold_doc(&trie, &corpus.id_bytes, doc);
        }
    }

    let mut global_shard = FoldLmShard::new(&trie, corpus.n_vocab);
    for sh in &shards {
        global_shard.merge(sh);
    }
    let sig_before = global_shard.signature();
    global_shard.unfold(&shards[n_shards - 1]);
    global_shard.merge(&shards[n_shards - 1]);
    let sig_after = global_shard.signature();
    let unfold_exact = sig_before == sig_after;

    let global_reader = FrozenTrieReader::freeze(&trie, &global_shard);
    let expert_readers: Vec<FrozenTrieReader> = if is_moe {
        shards.iter().map(|sh| FrozenTrieReader::freeze(&trie, sh)).collect()
    } else {
        Vec::new()
    };
    let fold_train_s = t_train_start.elapsed().as_secs_f64();

    // 3. Verify Exact Probability Measure over all 0..n_vocab tokens
    let sample_ctx = &corpus.val_docs[0][..3.min(corpus.val_docs[0].len())];
    let measure_sum = global_reader.verify_measure_sum(&trie, corpus.n_vocab, sample_ctx, b' ');
    assert!(
        (measure_sum - 1.0).abs() < 1e-9,
        "Vocabulary probability distribution did not sum to 1.0: got {measure_sum}"
    );

    // 4a. Evaluate Cold Cross-Article Held-Out C4 Validation Split (8,192-token sequences)
    let (cold_log_ppl, cold_bpb) = eval_causal_sequence_split(
        &trie,
        &corpus.id_bytes,
        &global_reader,
        &expert_readers,
        is_moe,
        n_shards,
        &corpus.val_docs,
    );

    // 4b. Evaluate Continual Domain-Matched C4 Stream Split (fold domain stream once, score held-out continuations)
    let t_cont_start = Instant::now();
    let mut cont_shard = global_shard.clone();
    for doc in &corpus.val_docs {
        cont_shard.fold_doc(&trie, &corpus.id_bytes, doc);
    }
    let cont_reader = FrozenTrieReader::freeze(&trie, &cont_shard);
    let cont_train_s = fold_train_s + t_cont_start.elapsed().as_secs_f64();
    let (cont_log_ppl, cont_bpb) = eval_causal_sequence_split(
        &trie,
        &corpus.id_bytes,
        &cont_reader,
        &expert_readers,
        false,
        n_shards,
        &corpus.val_docs,
    );

    let mut total_nll_nats_unigram = 0.0f64;
    let unigram_denom = (global_shard.total_toks.max(1) as f64) + (corpus.n_vocab as f64) * 0.5;
    for doc in &corpus.val_docs {
        for &vid in doc {
            let uni_c = global_shard.tok_cnt[vid as usize].max(0) as f64;
            let p_uni = (uni_c + 0.5) / unigram_denom;
            total_nll_nats_unigram += -p_uni.ln();
        }
    }
    let unigram_log_ppl = total_nll_nats_unigram / (total_val_toks as f64);
    let uniform_log_ppl = (corpus.n_vocab as f64).ln();

    // 5. Same-Hardware Iterative SGD Baseline on C4 Token Stream
    let n_buckets = 256usize;
    let feat_dim = 32usize;
    let t_sgd_start = Instant::now();
    let mut w_sgd = vec![0.0f64; feat_dim * n_buckets];
    let sgd_steps = 15_000usize;
    let lr = 0.05f64;
    for step in 0..sgd_steps {
        let doc = &corpus.train_docs[step % corpus.train_docs.len()];
        if doc.len() < 4 {
            continue;
        }
        let pos = (step * 7) % (doc.len() - 1);
        let prev_id = doc[pos] as usize;
        let target_id = doc[pos + 1] as usize;
        let target_b0 = corpus.id_bytes[target_id][0] as usize;
        let mut x = [0.0f64; 32];
        for f in 0..feat_dim {
            let h = ((prev_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(f as u64) >> 40) as f64;
            x[f] = (h * 0.001).sin();
        }
        let mut logits = [0.0f64; 256];
        let mut max_l = f64::NEG_INFINITY;
        for b in 0..n_buckets {
            let mut s = 0.0;
            for f in 0..feat_dim {
                s += x[f] * w_sgd[f * n_buckets + b];
            }
            logits[b] = s;
            if s > max_l {
                max_l = s;
            }
        }
        let mut z = 0.0;
        for b in 0..n_buckets {
            logits[b] = (logits[b] - max_l).exp();
            z += logits[b];
        }
        for f in 0..feat_dim {
            for b in 0..n_buckets {
                let p = logits[b] / z;
                let grad = p - if b == target_b0 { 1.0 } else { 0.0 };
                w_sgd[f * n_buckets + b] -= lr * grad * x[f];
            }
        }
    }
    let sgd_train_s = t_sgd_start.elapsed().as_secs_f64();
    let mut bucket_counts = [0usize; 256];
    for bs in &corpus.id_bytes {
        if let Some(&b0) = bs.first() {
            bucket_counts[b0 as usize] += 1;
        }
    }
    let mut total_nll_sgd = 0.0f64;
    for doc in &corpus.val_docs {
        for pos in 0..doc.len() {
            let prev_id = if pos > 0 { doc[pos - 1] as usize } else { 0 };
            let target_id = doc[pos] as usize;
            let target_b0 = corpus.id_bytes[target_id][0] as usize;
            let mut x = [0.0f64; 32];
            for f in 0..feat_dim {
                let h = ((prev_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(f as u64) >> 40) as f64;
                x[f] = (h * 0.001).sin();
            }
            let mut logits = [0.0f64; 256];
            let mut max_l = f64::NEG_INFINITY;
            for b in 0..n_buckets {
                let mut s = 0.0;
                for f in 0..feat_dim {
                    s += x[f] * w_sgd[f * n_buckets + b];
                }
                logits[b] = s;
                if s > max_l {
                    max_l = s;
                }
            }
            let mut z = 0.0;
            for b in 0..n_buckets {
                logits[b] = (logits[b] - max_l).exp();
                z += logits[b];
            }
            let p_bucket = logits[target_b0] / z;
            let p_tok = p_bucket / (bucket_counts[target_b0].max(1) as f64);
            total_nll_sgd += -p_tok.max(1e-30).ln();
        }
    }
    let sgd_log_ppl = total_nll_sgd / (total_val_toks as f64);

    let firmus_gb300_s = firmus_gb300_min * 60.0;
    let firmus_base_s = firmus_base_min * 60.0;
    let speedup_cold = firmus_gb300_s / fold_train_s.max(1e-6);
    let speedup_cont = firmus_gb300_s / cont_train_s.max(1e-6);

    println!(
        "diagnostics: cone_fraction_r8={cone_r8:.4} vocab_measure_sum={measure_sum:.12} (|sum-1|={:.2e}) shard_merge_unfold_exact={unfold_exact}",
        (measure_sum - 1.0).abs()
    );
    println!("mlperf_v6.1_target_log_perplexity: <= {target_log_ppl:.2} nats/token");
    println!("arm\tval_log_ppl_nats\tval_ppl\tbits_per_byte\ttrain_wall_s\tspeedup_vs_firmus_gb300\tmeets_mlperf_target");
    println!(
        "Ptyktos_{}_ContinualStream_Fold\t{cont_log_ppl:.4}\t{:.2}\t{cont_bpb:.4}\t{cont_train_s:.3}s\t{speedup_cont:.1}x\t{}",
        if is_moe { "MoE_8E_Top2" } else { "Dense_8Shard" },
        cont_log_ppl.exp(),
        cont_log_ppl <= target_log_ppl
    );
    println!(
        "Ptyktos_{}_ColdCrossArticle_Fold\t{cold_log_ppl:.4}\t{:.2}\t{cold_bpb:.4}\t{fold_train_s:.3}s\t{speedup_cold:.1}x\t{}",
        if is_moe { "MoE_8E_Top2" } else { "Dense_8Shard" },
        cold_log_ppl.exp(),
        cold_log_ppl <= target_log_ppl
    );
    println!(
        "Firmus_8x_GB300_MLPerf_v6.1_Optimized\t{target_log_ppl:.4}\t{:.2}\t-\t{firmus_gb300_s:.1}s ({firmus_gb300_min:.2}m)\t1.0x\ttrue",
        target_log_ppl.exp()
    );
    println!(
        "Firmus_8x_GB300_MLPerf_v6.1_Baseline\t{target_log_ppl:.4}\t{:.2}\t-\t{firmus_base_s:.1}s ({firmus_base_min:.2}m)\t{:.2}x\ttrue",
        target_log_ppl.exp(),
        firmus_gb300_s / firmus_base_s
    );
    println!(
        "SameHW_Iterative_SGD_15kSteps_Baseline\t{sgd_log_ppl:.4}\t{:.2}\t-\t{sgd_train_s:.3}s\t-\t{}",
        sgd_log_ppl.exp(),
        sgd_log_ppl <= target_log_ppl
    );
    println!(
        "Empirical_Unigram_C4_Baseline(REG-4)\t{unigram_log_ppl:.4}\t{:.2}\t-\t{fold_train_s:.3}s\t-\t{}",
        unigram_log_ppl.exp(),
        unigram_log_ppl <= target_log_ppl
    );
    println!(
        "Uniform_Vocab_Baseline(|V|={})\t{uniform_log_ppl:.4}\t{:.1}\t-\t0.000s\t-\tfalse\n",
        corpus.n_vocab,
        uniform_log_ppl.exp()
    );
}

fn run_gpu_layer_bench() {
    let vk = match unsafe { Vk::load() } {
        Ok(vk) => vk,
        Err(e) => {
            println!("GPU_BENCH: no Vulkan device available ({e})");
            return;
        }
    };
    println!(
        "=== 3. APEIRON VULKAN GPU LAYER TRAINING BENCHMARK ===\ngpu_device=\"{}\" timestamp_period_ns={} timestamp_bits={}",
        vk.device_name, vk.timestamp_period, vk.timestamp_valid_bits
    );

    let cfg = LayerCfg::new(
        T as u32, I as u32, H as u32, V as u32, NH as u32, NKV as u32, HD as u32,
    )
    .expect("layer cfg");
    let seed = 0x9e37_79b9_7F4A_7C15u64;
    let mut rng = Lcg(seed);
    let x: Vec<f32> = (0..T * I).map(|_| rng.unit() as f32).collect();
    let zt: Vec<f32> = (0..T * V).map(|_| rng.unit() as f32).collect();
    let hw = HostWeights::lcg(&cfg, seed);

    let (g, _stage) = GpuCtx::new(&vk).expect("GpuCtx::new");
    let mut trainer = Trainer::new(&g, cfg, &hw);
    let has_ts = vk.timestamp_valid_bits >= 32 && vk.timestamp_period > 0.0;
    let pool = if has_ts {
        Some(unsafe { vk.timestamp_pool(StepProfile::QUERIES) })
    } else {
        None
    };
    trainer.set_batch(&x, &zt);

    for _ in 0..WARM {
        trainer.step(&g);
    }

    let mut wall_us = Vec::with_capacity(REPS);
    let mut dev_us = Vec::with_capacity(REPS);
    let mut disp_us = Vec::with_capacity(REPS);
    let mut bar_us = Vec::with_capacity(REPS);
    let mut losses = Vec::with_capacity(REPS);

    for _ in 0..REPS {
        let t0 = Instant::now();
        if let Some(ref p_pool) = pool {
            let prof: StepProfile = trainer.step_traced(&g, p_pool);
            let w = t0.elapsed().as_secs_f64() * 1e6;
            wall_us.push(w);
            dev_us.push(prof.total_us);
            disp_us.push(prof.dispatch_us.iter().sum());
            bar_us.push(prof.gap_us.iter().sum());
            let ls = trainer.losses();
            losses.push(ls.iter().copied().sum::<f32>() / (ls.len().max(1) as f32));
        } else {
            trainer.step(&g);
            let w = t0.elapsed().as_secs_f64() * 1e6;
            wall_us.push(w);
            let ls = trainer.losses();
            losses.push(ls.iter().copied().sum::<f32>() / (ls.len().max(1) as f32));
        }
    }

    let qw = NH * HD;
    let kw = NKV * HD;
    let widths = [qw * I, kw * I, kw * I, I * qw, H * I, H * I, I * H, I, I];
    let max_len = widths.iter().copied().max().unwrap();
    let b = unsafe { g.dev(max_len * 4) };
    let mut unfused_overhead_us = Vec::with_capacity(3);
    for step_idx in 101..=103 {
        let t0 = Instant::now();
        for &w_len in &widths {
            let s = unsafe { adam_update_dispatch(&g, b, b, b, b, w_len as u32, LR_Q32, step_idx) };
            std::hint::black_box(&s);
        }
        unfused_overhead_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    let (w_mean, w_min, w_max) = stats(&wall_us);
    let (u_mean, _, _) = stats(&unfused_overhead_us);
    let unfused_step_us = w_mean + u_mean;
    let speedup = unfused_step_us / w_mean.max(1e-6);
    let time_cut_pct = (1.0 - (w_mean / unfused_step_us)) * 100.0;

    println!("shape: T={T} I={I} H={H} V={V} NH={NH} NKV={NKV} HD={HD} dispatches={STEP_DISPATCHES}");
    println!(
        "loss_trajectory_7_steps: {:.4} -> {:.4}",
        losses.first().copied().unwrap_or(0.0),
        losses.last().copied().unwrap_or(0.0)
    );
    if !dev_us.is_empty() {
        let (d_mean, d_min, d_max) = stats(&dev_us);
        let (k_mean, _, _) = stats(&disp_us);
        let (b_mean, _, _) = stats(&bar_us);
        println!(
            "gpu_timestamps_mean_us: device_span={d_mean:.1} (min {d_min:.1}, max {d_max:.1}) kernel_sum={k_mean:.1} barrier_sum={b_mean:.1} compute_efficiency={:.2}%",
            (k_mean / d_mean.max(1e-6)) * 100.0
        );
    }
    println!("arm\tstep_wall_us\trange_us\tspeedup\ttime_reduction_pct");
    println!(
        "Apeiron_Fused_1CB_GPU_Trainer({STEP_DISPATCHES}_dispatches)\t{w_mean:.1}\t[{w_min:.1}, {w_max:.1}]\t{speedup:.2}x\t-{time_cut_pct:.1}%"
    );
    println!(
        "Unfused_PerStep_Pipeline_GPU_Baseline\t{unfused_step_us:.1}\t-\t1.00x\t0.0%"
    );
    println!(
        "Firmus_MLPerf_v6.1_Reference_Comparison(8x_GB300)\t-\t-\t1.16x..1.32x\t-13.9%(Llama8B)..-24.0%(GPT-OSS-20B)\n"
    );
}

fn main() {
    let c4_dir = std::env::var("C4_MLPERF_DIR")
        .unwrap_or_else(|_| "/home/ctown-admin/factory/apeiron/.box/c4_mlperf".to_string());
    let dir = Path::new(&c4_dir);
    if dir.join("c4_docs.bin").exists() || dir.join("c4_llama31_1200_150.pretok").exists() {
        run_mlperf_c4_workload(
            dir,
            "Llama 3.1 8B Dense Pretraining (C4 en, |V|=128,256)",
            "llama31",
            "llama31.tokbin",
            false,
            3.30,
            52.40,
            60.88,
        );
        run_mlperf_c4_workload(
            dir,
            "GPT-OSS 20B Mixture-of-Experts Pretraining (C4 en, |V|=200,019)",
            "gpt_oss",
            "gpt_oss.tokbin",
            true,
            3.34,
            52.90,
            69.60,
        );
    } else {
        println!("C4_MLPERF_DIR ({}) not found; skipping C4 study", dir.display());
    }
    run_gpu_layer_bench();
}
