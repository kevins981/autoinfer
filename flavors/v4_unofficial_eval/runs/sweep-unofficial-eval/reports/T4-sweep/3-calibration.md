# T4 sweep, step 3: napkin cost model, calibrated against HEAD (T3-004)

Measurement: GPU job 199 (T4), `task/scripts/profile_nsys.sh` on workspace t4-head (= T3-004 + its
serve_args), 64 ShareGPT prompts at c=16, node-level graph trace. Files:
`agents/leader-4/prof/head_node/` (profile.sqlite, breakdown.txt, steps.txt); my scripts
`agents/leader-4/step_kern.py`, `step_detail.py`, `one_step.py`. GPU 0 unless said otherwise.
Caveat: nsys windows per step are attributed by CPU launch (async scheduling), so per-step "mixed"
windows also contain a decode-shaped forward; whole-capture totals are exact.

## Hardware constants used (and corrected)
- DRAM: 1.0 TB/s spec. Streaming GEMMs here reach 0.75-0.83 TB/s (below).
- **Tensor rate: I take FP8 (fp32 acc) = 330 TFLOP/s and bf16 (fp32 acc) = 165 TFLOP/s dense**
  (Ada AD102 whitepaper numbers for the 4090). SPEED_OF_LIGHT.md says 165 for FP8/fp32-acc; the
  measurement contradicts it: a mixed step with 836+42 rows ran its W8A8 GEMMs (21.3 TFLOP of
  body FLOPs) in ~117 ms = **~182 TFLOP/s achieved**, above 165. So FP8 fp32-acc peak >= 182;
  330 is consistent. Ridge points: W8A16 (bf16 math) ~83 rows, W8A8 ~165 rows.
- PCIe (no P2P), host-staged: 25-32 GB/s per direction (to be measured by t4-w1's microbench).

## Whole capture (GPU 0, 19.34 s wall, GPU busy union 18.52 s = 95.8%)
| category | s | share of wall |
|---|---|---|
| W8A16 Triton GEMM (decode M<=48 + draft M=16) | 7.77 | 40.2% |
| W8A8 Triton GEMM + act quant (prefill/mixed, M>=80) | 2.88 | 14.9% |
| NCCL all-reduce (messages > 128 rows) | 2.43 | 12.5% |
| host-memory all-reduce (<= 128 rows) | 2.41 | 12.5% |
| cutlass/cublas (target lm_head bf16, draft head fp8, fc, in_proj_ba) | 1.88 | 9.7% |
| GDN (fused decode kernel, conv, chunked prefill kernels, flush) | 1.15 | 5.9% |
| other small kernels | 0.69 | 3.5% |
| attention (FlashInfer, KV write, rope/norm) | 0.27 | 1.4% |
| NCCL all-gather (argmax / sampling) | 0.04 | 0.2% |
**Communication = 25% of GPU time.**

## Decode step (M = 48 rows: 16 requests x (1 + 2 drafts)); median period 25.25 ms
Napkin per GPU: time = max(bytes/BW, FLOPs/F) per op, AR = message/link.
| part | napkin (ideal) | measured | ratio meas/napkin |
|---|---|---|---|
| body GEMMs, 12.17 GB FP8 weights (W8A16) | 12.2 ms @1 TB/s (compute 1.17 TF @165 = 7.1 ms: memory-bound) | 15.4 ms (gate_up 112.9 us x64, down/out 37 us x128, qkvz 55.7 x48, qkv 49.8 x16) | 1.27 (0.79 TB/s) |
| target lm_head bf16 1.27 GB | 1.27 ms | 1.58 ms | 1.25 |
| 2 MTP draft steps (layer 186 MB + fc ~52 MB + fp8 head 252 MB each) | 0.98 ms | ~1.3 ms (+ 4 small ARs, 2 all-gathers) | 1.3 |
| GDN recurrent state, 48 layers x 16 req x 1.57 MB: read 1.2 GB + ~0.6 GB write (lazy replay) | 1.8 ms | 1.83 ms fused kernel + 0.18 conv + flush | 1.1 |
| all-reduce, 128 + 4 per step, 0.49 MB each | 128 x 0.49 MB / 25 GB/s ~ 2.5 ms if up/down overlap, ~5 ms if serialized (HEAD's one-shot) | 3.1 ms median step (avg 30 us; 42 us in slower windows) | |
| attention (KV ~130 MB) | 0.13 ms | 0.44 ms (latency-bound, 25 us/layer) | 3.4 |
| norms, act, small | ~0.2 ms | ~1 ms | |
| **sum** | **~19.6 ms (AR overlapped) - 22 ms** | **busy ~23.7 ms, period 25.25 ms** | 1.15-1.2 |

Reading: weight streaming at 79% of peak DRAM BW is the biggest residual (3.2 ms), then the
all-reduce (3.1 ms, of which ~0.6-1 ms is the serialized write/read protocol), then ~1.5 ms of
small/latency-bound kernels.

## Mixed step (one ~300-token prompt + 45 decode rows)
Examples (window medians, GPU 0): P=319: period 102.7 ms, W8A8 31 ms, NCCL 29.3 ms (99 calls,
296 us each), plus a decode-shaped forward (~15 ms W8A16 + 5.7 ms host AR) in the same window.
| part | napkin | measured |
|---|---|---|
| W8A8 GEMMs, 2 x 12.17 G x 364 rows = 8.9 TFLOP | 26.8 ms @330 TF | ~31 ms (~250-290 TF achieved incl. small-M tail) |
| 128 all-reduces x 3.7 MB | 128 x 3.7 MB / 25 GB/s = 19 ms (pipelined) | 29-38 ms (NCCL LL, 296-380 us/call = ~10-12 GB/s) |
| GDN chunked prefill + flush | ~2 ms | ~5 ms |
NCCL per-call time vs message: 1.4 MB 173 us, 1.8 MB 222 us, 2.5 MB 242 us, 3.7 MB 296-340 us,
6-9 MB 255-330 us per chunk (2 chunks), i.e. **8-15 GB/s, well below the link**. Same on GPU 1
(median 310 us both GPUs, start skew ~36 us): not a waiting-for-the-peer artifact.

## Whole-benchmark napkin vs official
Decode-only steps: ~91.2K tokens / (16 x 2.33) = ~2,450 steps x ~25 ms = ~61 s. Mixed steps:
400 prompts x (~80-105 ms - 25 ms) extra = ~22-32 s. Total ~83-93 s vs official ~99.5 s
(91.2K / 916 tok/s expected for HEAD). Napkin is ~10% optimistic (tail of the run with < 16
requests, first-token steps, scheduling gaps): good enough to rank, with a 1.1x fudge.
Time shares for ranking: **decode ~62%, mixed/prefill extra ~38%**.

## What the calibration says about the framework's cost model here
- The `max(F, BW, Link)` window model holds per op, but the link is not one resource: the
  host-staged all-reduce uses PCIe up (GPU->host) and PCIe down (host->GPU) as two resources,
  and HEAD's protocol serializes them. The framework's Link_BW should be split into up/down.
- The SoL compute constant for FP8 was off by 2x (see above); ridge points move accordingly
  (W8A16 decode at M=48 is memory-bound, not compute-bound).
