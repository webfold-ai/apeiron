//! Does batched prefill agree with token-by-token decode? For prefixes of one prompt, the next
//! token from `prefill_state` over the whole prefix must equal the next token after feeding the
//! prefix one token at a time through `next_token_state`. A disagreement locates a prefill
//! (tiling / attention / RoPE-in-batch) bug by prefix length.
//!
//! `prefill_vs_decode --model <gguf> --ids <file> [--lens 4,16,32,64,128]`

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let model = get("--model").ok_or("--model")?;
    let ids: Vec<u32> = std::fs::read_to_string(get("--ids").ok_or("--ids")?)?
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect();
    let lens: Vec<usize> = get("--lens")
        .unwrap_or_else(|| "4,16,32,64,128".into())
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    let engine = Engine::open(&model)?;
    let mode = KernelMode::Q8Act;
    for n in lens {
        let p = &ids[..n];
        let mut sa = engine.state(n + 2)?;
        let ta = engine.prefill_state(&mut sa, p, mode)?;
        let mut sb = engine.state(n + 2)?;
        let mut tb = engine.prefill_state(&mut sb, &p[..1], mode)?;
        for &t in &p[1..] {
            tb = engine.next_token_state(&mut sb, t, mode)?;
        }
        println!("len {n:4}: prefill next {ta:6}  decode-fed next {tb:6}  {}", if ta == tb { "SAME" } else { "DIFFERENT" });
    }
    Ok(())
}
