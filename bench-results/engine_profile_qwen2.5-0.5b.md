# Engine benchmark `Qwen2.5-0.5B-Instruct, HF safetensors -> GGUF, 10.0.63.10`

- model `/home/ctown-admin/models/roundtrip/qwen2.5-0.5b-instruct-q8_0.gguf` (sha256 `1d5e3ec337c3c42c309670532c07d806be2d00583481774f1e032110248b57f4`)
- prompt ids `/home/ctown-admin/factory/apeiron-rt/.box/prompt.ids` (sha256 `3b7fe1e5050ed1433e5d7ec58b3b828c06fe5ae0cd317bc606470b64ce79f9a9`), pp 200, tg 64, 3 rounds
- git `` with 0 dirty paths; src sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- AMD Ryzen 7 5700X 8-Core Processor; governor ; 8 CPUs; rustc 1.96.0 (ac68faa20 2026-05-25)
- read ceiling (1 GiB sum, best of 3): 36.1 / 34.3 / 33.9 GB/s at 8 / 16 / 32 threads
- llama.cpp `af6528e`, threads 8,16,24,32; comparator gpu_info `` backends `CUDA`
- quiet gate busy < 0.05: at least one round timed out (flagged)

| metric (tok/s) | median | min | max |
|---|---:|---:|---:|
| apeiron pp | 32.14 | 29.22 | 32.82 |
| llama.cpp pp (best threads) | 421.62 | 417.02 | 430.44 |
| ratio pp (per round) | 0.07 | 0.07 | 0.08 |
| apeiron tg | 1.42 | 1.42 | 1.45 |
| llama.cpp tg (best threads) | 24.88 | 24.57 | 25.29 |
| ratio tg (per round) | 0.06 | 0.06 | 0.06 |
| apeiron pp busy cores | 6.93 | 6.89 | 6.94 |
| apeiron tg busy cores | 6.96 | 6.95 | 6.96 |
| mean MHz before pp | NaN | inf | -inf |
| mean MHz before tg | NaN | inf | -inf |

### Traced phases

#### pp: 200 tokens, 31.22 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.002 | 0.0% | 1 | - | - |
| norm | 0.749 | 2.4% | 49 | - | - |
| residual | 0.092 | 0.3% | 48 | - | - |
| attn.proj_in | 5.805 | 18.6% | 24 | 0.0 | 4.3 |
| attn.core | 0.818 | 2.6% | 24 | - | - |
| attn.proj_out | 5.443 | 17.4% | 24 | 0.0 | 3.5 |
| ffn | 18.263 | 58.5% | 24 | 0.1 | 17.2 |
| head | 0.044 | 0.1% | 1 | 16.6 | 15.6 |
| untraced in pass | 0.007 | 0.0% | 0 | - | - |
| outside passes | 0.001 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 49.7) | 25.453 | 81.5% | 2404 | - | - |

#### tg: 64 tokens, 697.67 ms/token wall

| phase | ms/token | share of wall | calls | GB/s | GMAC/s |
|---|---:|---:|---:|---:|---:|
| embed | 0.004 | 0.0% | 64 | - | - |
| norm | 0.292 | 0.0% | 3136 | - | - |
| residual | 0.110 | 0.0% | 3072 | - | - |
| attn.proj_in | 150.302 | 21.5% | 1536 | 0.2 | 0.2 |
| attn.core | 157.263 | 22.5% | 1536 | - | - |
| attn.proj_out | 151.285 | 21.7% | 1536 | 0.1 | 0.1 |
| ffn | 227.233 | 32.6% | 1536 | 1.5 | 1.4 |
| head | 11.028 | 1.6% | 64 | 13.1 | 12.3 |
| untraced in pass | 0.150 | 0.0% | 0 | - | - |
| outside passes | 0.002 | 0.0% | 0 | - | - |
| pool.wait (overlay; tasks/dispatch 16.7) | 682.110 | 97.8% | 15552 | - | - |
