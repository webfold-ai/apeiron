//! Generate text from a GGUF model: prompt in, streamed completion out, greedy.
//!
//! Run:
//! `cargo run --release --locked --example generate -- --model <gguf> --tokenizer <tokenizer.json>
//!  [--prompt TEXT | stdin] [--n 64] [--chat] [--system TEXT] [--kernel q8act|exact]`
//!
//! - Without `--chat` the prompt is a raw completion prefix.
//! - With `--chat` the prompt is wrapped as one `ChatML` user turn (the `qwen2` and `qwen35` template:
//!   `<|im_start|>role\n…<|im_end|>\n`) and the model's end-of-turn token stops generation.
//! - `--kernel q8act` (default) is the int8-activation path every throughput receipt measures;
//!   `exact` is the i128 reference path, about an order of magnitude slower.
//! - Sampling is greedy (`argmax_fixed` inside `next_token_state`); there is no temperature.
//! - The completion goes to stdout as it is produced. Prompt ids, throughput and the stop reason go
//!   to stderr, so `generate … > out.txt` captures the text alone.
//!
//! Domain: the prompt and `n` must fit the model's KV capacity (`Engine::state` refuses
//! otherwise); every prompt id must be below the model's vocabulary; an empty prompt is refused.
//!
//! Cost, in engine steps: prefill is one pass over P prompt tokens, decode is at most `n` steps,
//! so P + n steps in time and P + n KV positions in space. Detokenising is O(1) per generated
//! token because only the bytes pending since the last complete UTF-8 character are decoded.

use std::error::Error;
use std::io::{Read, Write};
use std::process::ExitCode;
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;
use apeiron::format::bpe::Bpe;

struct Args {
    model: String,
    tokenizer: String,
    prompt: Option<String>,
    n: usize,
    chat: bool,
    system: String,
    mode: KernelMode,
    dump_ids: Option<String>,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut a = Args {
        model: String::new(),
        tokenizer: String::new(),
        prompt: None,
        n: 64,
        chat: false,
        system: "You are a helpful assistant.".to_string(),
        mode: KernelMode::Q8Act,
        dump_ids: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => a.model = val()?,
            "--tokenizer" => a.tokenizer = val()?,
            "--prompt" => a.prompt = Some(val()?),
            "--n" => a.n = val()?.parse()?,
            "--system" => a.system = val()?,
            "--dump-ids" => a.dump_ids = Some(val()?),
            "--kernel" => {
                a.mode = match val()?.as_str() {
                    "exact" => KernelMode::Exact,
                    "q8act" => KernelMode::Q8Act,
                    other => return Err(format!("--kernel is q8act|exact, got {other:?}").into()),
                }
            }
            "--chat" => a.chat = true,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if a.model.is_empty() || a.tokenizer.is_empty() {
        return Err("--model and --tokenizer are required".into());
    }
    Ok(a)
}

fn run(a: &Args) -> Result<(), Box<dyn Error>> {
    let user = if let Some(p) = &a.prompt {
        p.clone()
    } else {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        s
    };
    let text = if a.chat {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            a.system,
            user.trim_end()
        )
    } else {
        user
    };

    let bpe = Bpe::from_tokenizer_json(&a.tokenizer)?;
    let engine = Engine::open(&a.model)?;
    let ids = bpe.encode(&text)?;
    if ids.is_empty() {
        return Err("the prompt is empty".into());
    }
    let vocab = engine.config.vocab_size;
    if let Some(&bad) = ids.iter().find(|&&id| id as usize >= vocab) {
        return Err(format!("prompt id {bad} >= model vocabulary {vocab}").into());
    }
    // A chat reply ends at the end-of-turn token; a raw completion runs to `n`.
    let stop = if a.chat { bpe.eos_id() } else { None };
    eprintln!("prompt_ids    {} tokens", ids.len());

    let mut st = engine.state(ids.len() + a.n)?;
    let t0 = Instant::now();
    let mut next = engine.prefill_state(&mut st, &ids, a.mode)?;
    let prefill = t0.elapsed().as_secs_f64();

    let out = std::io::stdout();
    let mut out = out.lock();
    // Token bytes can split a UTF-8 character; hold ids until they decode without a replacement.
    let mut pending: Vec<u32> = Vec::new();
    let mut produced_ids: Vec<u32> = Vec::new();
    let mut produced = 0usize;
    let mut reason = "length";
    let t1 = Instant::now();
    while produced < a.n {
        if Some(next) == stop {
            reason = "end-of-turn";
            break;
        }
        pending.push(next);
        produced_ids.push(next);
        produced += 1;
        let piece = bpe.decode(&pending)?;
        if !piece.ends_with('\u{FFFD}') {
            out.write_all(piece.as_bytes())?;
            out.flush()?;
            pending.clear();
        }
        if produced < a.n {
            next = engine.next_token_state(&mut st, next, a.mode)?;
        }
    }
    if !pending.is_empty() {
        out.write_all(bpe.decode(&pending)?.as_bytes())?;
    }
    writeln!(out)?;
    if let Some(path) = &a.dump_ids {
        let mut f = std::fs::File::create(path)?;
        for id in &produced_ids {
            writeln!(f, "{id}")?;
        }
        eprintln!("dumped_ids    {} -> {path}", produced_ids.len());
    }
    let decode = t1.elapsed().as_secs_f64();

    #[allow(clippy::cast_precision_loss, reason = "display ratios")]
    {
        eprintln!(
            "prefill       {prefill:.3} s, {:.2} tok/s",
            ids.len() as f64 / prefill
        );
        eprintln!(
            "decode        {decode:.3} s, {:.2} tok/s ({produced} tokens, stop: {reason})",
            produced as f64 / decode
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let result = parse_args().and_then(|a| run(&a));
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("generate: {e}");
            ExitCode::FAILURE
        }
    }
}
