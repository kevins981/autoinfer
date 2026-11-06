# Baseline profile: Qwen3.8-27B-FP8 on 2x RTX 4090 (task serve config, FP8 KV)

What round 1 starts from. Observed facts first, then interpretations (marked). A summary: the
raw data next to this file has every kernel launch and every hardware metric, and
`../PROFILING.md` shows how to query it. Regenerate with `scripts/profile_nsys.sh`.

## Setup and headline numbers
- Server: `harness/serve_args.sh` (TP=2, FP8 weights, FP8 KV cache with a 5 GiB KV budget,
  MTP speculative decoding with 2 draft tokens, max context 128K, max 16 sequences).
- Official BASE (`official_eval`, ShareGPT 400 prompts at concurrency 16, GSM8K-250):
  **630.6 output tok/s**, TTFT p50 234 ms / p99 1355 ms, TPOT p50 23.1 ms / p99 41.2 ms, ITL p50
  39.6 ms, E2E p50 4.7 s, MTP acceptance length 2.34 (per position 0.78 / 0.56), GSM8K 92.8%.
- Memory per GPU: weights 14.4 GiB; KV cache 267,721 tokens in total (2.04x one 128K request);
  CUDA graphs PIECEWISE (18 sizes) + FULL (9 sizes) + spec-decode "speculator" graphs.
- Startup: ~85 s with warm caches (weights in page cache, compile cache populated).

## Model (from config.json)
64 layers, hidden 5120, FFN intermediate 17408, vocab 248320. Hybrid attention: every 4th layer
is full attention (16 layers: 24 query heads, 4 KV heads, head dim 256); the other 48 are
Gated DeltaNet linear attention (16 key heads, 48 value heads, head dim 128; recurrent state,
plus a short causal conv). 1 MTP layer (the draft model). FP8 e4m3 weights with block scales,
dynamic activation scales. With TP=2 each GPU holds half of every layer.

## Hardware
2x RTX 4090 (sm_89, 24 GB, ~1 TB/s DRAM each, FP8 tensor cores). **No GPU peer-to-peer**: TP
all-reduces go through NCCL over PCIe and host memory (vLLM's custom all-reduce is disabled).
48 CPU cores. CUDA 13.0 toolkit, driver 580.

## nsys: where the GPU time goes (`nsys_c16/`)
64 ShareGPT prompts at concurrency 16, 27 s of serving, CUDA graphs traced per kernel, both GPUs.
From earlier runs: the GPUs are kernel-busy ~97% of the wall time, so CPU gaps are not the
bottleneck. Share of kernel time (`nsys_c16/breakdown.txt`):
```
 50.8%  FP8 weight GEMMs (Marlin)
 28.3%  NCCL all-reduce (25.7%, ~88 us each, ~0.5 MB) + all-gather (2.5%)
 10.7%  other GEMMs (cutlass/cuBLAS bf16: lm_head over a 248K vocab, MTP head, small ones)
  6.2%  GDN linear attention (delta-rule update, conv1d)
  2.2%  full attention (FlashInfer, FP8 KV) + KV cache writes
  2.0%  norms, activations, rope, gating, sampling, other
```
Per verify pass (one decode step for the batch, from the earlier profile): 64 gate/up Marlin
GEMMs (~117 us), 192 other Marlin GEMMs (26-60 us), 129 all-reduces (~66-88 us), 48 GDN
updates (~45-75 us).

## ncu: the Marlin GEMMs (`ncu_marlin/`, measured with bf16 KV; the GEMMs are the same)
3 layers x 4 GEMMs, full metric set. FP8 weights run through **Marlin: weight-only
quantization, bf16 compute** (the CUTLASS block-scaled FP8 kernel doesn't support sm_89).
- **DRAM-bandwidth-bound at ~80% of peak**: gate/up reads 91 MB in 117 us = 792 GB/s (80.6%);
  down 46 MB in 58 us; the smallest 16 MB in 26 us (~62%).
- Tensor pipe ~25% busy, SM issue ~9-11%, 2 warps per scheduler, 255 registers per thread,
  128 CTAs (one per SM), shared-load bank conflicts ~5-way (estimated gain < 4%).
- Files: `marlin_c16.ncu-rep` (open with `ncu --import <rep> --page details`, or the GUI),
  `marlin_c16_key.csv` (key metrics per launch), `marlin_c16_src.ncu-rep` (built with
  -lineinfo, source imported: `python scripts/ncu_hotlines.py <rep>` for the hottest lines).
- Interpretation: little left inside these kernels; GEMM gains must come from reading fewer
  bytes, or from amortizing each weight read over more tokens (bigger batches, deeper
  speculation).

## Earlier measurements (qwen38_opt tuning pass, before this project; c=16 ShareGPT, bf16 KV)
Observed (one trial each; noise ~3%):
- MTP draft tokens: 2 beat 3 at c=16 (+12%: each request holds 1 + draft-tokens GDN state blocks,
  so 3 tokens left only ~14 requests running and the rest queued); 3 beat 2 by 2-6% at c=1-8.
- Async scheduling (on by default) is worth ~10%.
- No effect (within noise): prefix caching off, Triton attention, GDN backend flags,
  `--max-num-batched-tokens` 4096 (TTFT +17% worse).
- `--linear-backend triton` (true FP8 tensor-core compute): TTFT -14% but decode ITL +14%, net
  -4%. Interpretation: prefill and decode want different GEMM kernels.
- FP8 KV: forces FlashInfer attention (~1 GiB more non-KV memory); with a 5 GiB KV budget it
  doubles KV capacity. In the official setup it measured 630.6 tok/s vs 656 with bf16 KV.
- Not possible here: `--use-replayssm` (unsupported for this model), turning off chunked
  prefill alone, `--linear-backend cutlass`.
- vLLM lists `FLASHINFER_PCIE_IPC` among its possible all-reduce backends but uses PYNCCL here
  (see `nsys_c16/server.log`). Not investigated.

## Interpretation (to question, not to trust)
Decode at c=16 is a weight-streaming problem plus a PCIe communication problem: half the GPU
time reads FP8 weights near DRAM peak, a quarter is all-reduces over PCIe without P2P, and the
GPUs are almost never idle. Directions that follow from this, none tried yet: fewer or cheaper
all-reduces (or other ways to split the model across two GPUs), overlapping communication with
compute, more tokens per weight read, and the ~11% of non-Marlin GEMMs (e.g. the large lm_head).
