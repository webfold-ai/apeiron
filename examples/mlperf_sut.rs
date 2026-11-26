//! MLPerf Inference system-under-test: the official LoadGen (C++, via `mlperf/shim.cpp`) drives
//! the sanguine engine on the `llama3.1-8b` CNN/DailyMail task. Tokenised prompts come from the
//! official `download_cnndm.py` output (one sample per line); generation is greedy, max 128 new
//! tokens, stops on a Llama-3.1 end token. Responses are int32 token ids, as the official
//! `evaluation.py` expects.
//!
//! `mlperf_sut --model <gguf> --prompts <ids.txt> --scenario Offline|Server --mode perf|acc
//!   --mlperf-conf <f> --user-conf <f> --logdir <dir> [--min-queries N] [--min-ms N] [--qps X]`
//! `--prompts` holds the first N dataset samples, so LoadGen indices equal dataset indices.

use std::collections::VecDeque;
use std::error::Error;
use std::ffi::{c_char, c_void, CString};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use apeiron::engine::loader::Engine;
use apeiron::engine::KernelMode;

type IssueFn = extern "C" fn(*mut c_void, *const u64, *const u64, usize);

#[link(name = "mlshim", kind = "static")]
#[link(name = "mlperf_loadgen", kind = "static")]
#[link(name = "stdc++")]
extern "C" {
    fn mlperf_complete(id: u64, data: *const c_void, bytes: usize, ntok: i64, first: i32);
    #[allow(clippy::too_many_arguments)]
    fn mlperf_run(
        ctx: *mut c_void,
        f: IssueFn,
        total: usize,
        scenario: *const c_char,
        mode: i32,
        mlperf_conf: *const c_char,
        user_conf: *const c_char,
        model: *const c_char,
        logdir: *const c_char,
        min_queries: u64,
        min_ms: u64,
        qps: f64,
    ) -> i32;
}

const STOP: [u32; 3] = [128001, 128008, 128009];
const MAX_NEW: usize = 128;

struct Queue {
    q: Mutex<(VecDeque<(u64, u64)>, bool)>,
    cv: Condvar,
}

extern "C" fn issue(ctx: *mut c_void, ids: *const u64, idx: *const u64, n: usize) {
    // SAFETY: ctx is the `Arc<Queue>` leaked in main; ids/idx hold n entries for this call.
    let queue = unsafe { &*(ctx as *const Queue) };
    let (ids, idx) = unsafe { (std::slice::from_raw_parts(ids, n), std::slice::from_raw_parts(idx, n)) };
    let mut g = queue.q.lock().unwrap();
    for i in 0..n {
        g.0.push_back((ids[i], idx[i]));
    }
    queue.cv.notify_one();
}

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}

fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<String> = std::env::args().collect();
    let req = |k: &str| arg(&a, k).ok_or_else(|| format!("{k} is required"));
    let prompts: Vec<Vec<u32>> = std::fs::read_to_string(req("--prompts")?)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect())
        .collect();
    let engine = Arc::new(Engine::open(&req("--model")?)?);
    let n = prompts.len();
    eprintln!("mlperf_sut: {n} prompts loaded, model open");
    let queue = Arc::new(Queue { q: Mutex::new((VecDeque::new(), false)), cv: Condvar::new() });
    let server = req("--scenario")? == "Server";
    let worker = {
        let (queue, engine) = (queue.clone(), engine.clone());
        std::thread::spawn(move || loop {
            let job = {
                let mut g = queue.q.lock().unwrap();
                loop {
                    if let Some(j) = g.0.pop_front() {
                        break Some(j);
                    }
                    if g.1 {
                        break None;
                    }
                    g = queue.cv.wait(g).unwrap();
                }
            };
            let Some((id, idx)) = job else { return };
            let p = &prompts[idx as usize];
            let t0 = Instant::now();
            let mut st = engine.state(p.len() + MAX_NEW).expect("state");
            let mut next = engine.prefill_state(&mut st, p, KernelMode::Q8Act).expect("prefill");
            let tp = t0.elapsed().as_secs_f64();
            let mut out: Vec<i32> = Vec::new();
            let mut first_sent = false;
            while out.len() < MAX_NEW {
                if STOP.contains(&next) && !out.is_empty() {
                    break;
                }
                out.push(next as i32);
                if server && !first_sent {
                    let b = [next as i32];
                    unsafe { mlperf_complete(id, b.as_ptr().cast(), 4, 1, 1) };
                    first_sent = true;
                }
                if out.len() < MAX_NEW {
                    next = engine.next_token_state(&mut st, next, KernelMode::Q8Act).expect("step");
                }
            }
            eprintln!(
                "sample {idx}: prompt {} tok, prefill {tp:.1}s, out {} tok, total {:.1}s",
                p.len(),
                out.len(),
                t0.elapsed().as_secs_f64()
            );
            unsafe { mlperf_complete(id, out.as_ptr().cast(), out.len() * 4, out.len() as i64, 0) };
        })
    };
    let c = |k: &str| CString::new(req(k).unwrap()).unwrap();
    let num = |k: &str| arg(&a, k).map_or(0u64, |v| v.parse().unwrap());
    let rc = unsafe {
        mlperf_run(
            Arc::as_ptr(&queue) as *mut c_void,
            issue,
            n,
            c("--scenario").as_ptr(),
            i32::from(req("--mode")? == "acc"),
            c("--mlperf-conf").as_ptr(),
            c("--user-conf").as_ptr(),
            CString::new("llama3_1-8b")?.as_ptr(),
            c("--logdir").as_ptr(),
            num("--min-queries"),
            num("--min-ms"),
            arg(&a, "--qps").map_or(0.0, |v| v.parse().unwrap()),
        )
    };
    queue.q.lock().unwrap().1 = true;
    queue.cv.notify_all();
    worker.join().unwrap();
    if rc != 0 {
        return Err(format!("mlperf_run failed: {rc}").into());
    }
    Ok(())
}
