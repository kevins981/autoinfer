# T2 sweep, step 3: napkin cost model, calibrated against BASE

Measured: task/baseline/nsys_c16/profile.sqlite (BASE, 64 prompts, c=16), GPU 0, per-step medians over
the `execute_context_*` NVTX ranges of one TP worker (same step definition as scripts/step_anatomy.py;
my own categorisation script, in agents/leader-2/ notes: lm_head = the 970-tile bf16 GEMMs, gdn = GDN
decode/prefill kernels + conv1d, attn = FlashInfer + KV write). ncu isolated Marlin times from
task/baseline/ncu_marlin/marlin_c16_key.csv. Napkin uses spec numbers (1.0 TB/s, 165 TF bf16-MMA,
25 GB/s PCIe per direction, ~8 us per link hop latency).

## Decode step (M ~ 45 rows = 15 req x 3; 501 steps in the profile)
| part | napkin (spec) | measured | ratio meas/napkin | note |
|---|---|---|---|---|
| FP8 body GEMMs (Marlin) | 12.15 GB / 1.0 TB/s = 12.2 ms | 18.48 ms | 1.52 | effective 0.66 TB/s. ncu isolated: in_proj 53.6 us (0.80 TB/s), out_proj 26 us (0.63), gate_up 115 us (0.79), down 60 us (0.77); in-situ (nsys) gate_up 120-141 us |
| all-reduces (130: 128 body + embeddings) | 460 KB each: 18 us transfer + 8 us latency = 26 us -> 3.4 ms | 7.75 ms (~60 us each) | 2.3 | NCCL ring, no P2P (host-memory path) |
| logits all-gathers (target 45 rows, 2 drafts 16 rows) | 19 MB / 25 GB/s = 0.76 ms | 1.46 ms | 1.9 | full logits gathered though only argmax is needed at T=0 |
| lm_heads (target + 2 draft passes, bf16 1.27 GB each) | 3.8 GB -> 3.8 ms | 4.06 ms | 1.07 | target 1.59 ms, each draft 1.37 ms |
| GDN decode (48 layers) | read 1 + write 3 states: 15 req x 1.57 MB x 4 = 94 MB/layer -> 4.5 ms | 2.95 ms (61 us/layer) | **0.65** | **mismatch: faster than DRAM napkin**; either fewer bytes than I think (some destination slots skipped?) or L2 absorbs writes. Needs ncu (dram__bytes_write) |
| full attention (16 layers) + KV write | ~130 MB KV -> 0.13 ms | 0.42 ms | 3.2 | small, latency-bound |
| small GEMMs (GDN a/b bf16, MTP fc) | ~0.1 ms | 0.65 ms | | |
| other (norms, act, sampling, bookkeeping) | ~0.3 ms | 1.10 ms | | |
| GPU idle within step | 0 | 1.03 ms | | CPU-side gaps between draft passes (128 + 250 us gaps seen) |
| **total** | **~25 ms** | **38.2 ms** (period) | 1.53 | |

Calibrated decode model (what the rankings use): T_dec = W_body/0.66 TB/s + n_AR x 60 us + lm_head
bytes/0.94 TB/s + AG + 61 us x 48 (GDN) + ~3.2 ms fixed = 38 ms. The two largest napkin-to-measured
gaps are Marlin (+6.3 ms) and all-reduce (+4.4 ms).

## Prefill-containing step (median 215 prefill + 45 decode rows = M ~260; 50 steps in the profile)
| part | napkin | measured | ratio | note |
|---|---|---|---|---|
| Marlin GEMMs | compute 2 x 260 x 12.15e9 = 6.3 TFLOP / 165 TF = 38 ms (memory 12 ms) | 59.3 ms | 1.55 | compute-bound with bf16 MMA (ridge ~82 rows) |
| all-reduces (128 x 2.66 MB) | 106 us + 8 us -> 14.6 ms | **62.5 ms (~490 us each)** | **4.3** | the worst-calibrated part: NCCL on large messages through host memory |
| GDN prefill (FLA chunk kernels) + decode part | ~1 ms | 7.2 ms | | small chunked kernels |
| lm_head + AG | 1.6 + 0.8 | 4.4 + 1.7 | | |
| other + idle | | 5.8 + 24.5 ms idle | | idle inflated by nsys for eagerly launched parts (PROFILING.md); real value unknown |
| **total** | **~56 ms** | **146.8 ms** | 2.6 | |

## Whole benchmark
~2,500 decode steps x 38.2 ms = 95.5 s + ~310 prefill steps (400 requests x 50/64) x 146.8 ms = 45.5 s
=> 141 s napkin-calibrated vs 143 s measured (91.2K tokens / 635.9 tok/s). Decode ITL median 39.4 ms
official vs 38.2 ms step period here: nsys didn't distort decode much.

**Shares of wall time:** decode steps 67%, prefill-containing steps 32%. All-reduce alone =
2500 x 7.75 + 310 x 62.5 = 19.4 + 19.4 = 38.8 s = **27% of wall time**; Marlin = 46 + 18 = 64 s = 45%.

## What the framework's cost model gets wrong here (observations)
1. It predicts decode GEMMs at the bandwidth roofline; Marlin runs at 0.66 TB/s because at M=45 a
   bf16-MMA-on-dequantised-FP8 kernel needs ~58% of tensor peak *at the same time* as ~100% DRAM;
   the max() in the window formula assumes resources overlap perfectly. A kernel-efficiency factor per
   (op, M) is needed — and it opens a "dtype of the compute" dimension (W8A8) the knobs don't list.
2. The link model (bytes/Link_BW) underestimates NCCL through host memory by 2.3x (460 KB) to 4.3x
   (2.6 MB); protocol and transport matter more than bytes.
3. GDN decode came in under the DRAM napkin: the byte count needs checking before trusting K1 rows.
