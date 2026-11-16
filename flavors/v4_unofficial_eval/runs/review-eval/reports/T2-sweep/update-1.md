# T2 map, update 1 (resync 1)

## Tested since 6-ranking
| candidate (6-ranking #) | result | evidence |
|---|---|---|
| #1 W1 CUDA graphs for mixed steps | **win, champion**: 689.5 tok/s (+8.4% vs BASE), GSM8K 93.2% | bank/T2-001, jobs 4, 8; review T2-1 |
| #3 argmax before gather (draft: use_local_argmax_reduction; target: batch-sharded sampling, all-to-all) | **win**: 707.0 tok/s (+2.5% vs T2-001), exact | bank/T2-003 (draft; review T2-2 pending), job 14 |
| stack T2-001 + T1-001 (fp8 draft head) + #3 | **OOM at startup** (5 GiB KV fixed; fp8 head copy +0.59 GiB) | bank/T2-002, job 10 |
| #2 W8A8 block-FP8 (Triton, tuned configs) | microbench: tuned Triton beats Marlin at every M (graph replay, cold weights): -3.9% GEMM time at decode M=48, -30..-36% at M>=256 (per prompt token 155 -> 100 us); harness pending (stacked on T2-003, GATES=long,gsm8k) | worker jobs 3, 6, 12; agents/leader-2.w1/mb/ |
| NCCL tuning (P4, now T1's) | worker bench: NCCL already picks Simple from 48 rows; no env variant helps decode sizes; MIN_NCHANNELS=8 ~7% at >= 1024 rows only | job 2 |

## Verdicts changed
- **W1 mechanism (3-calibration B) was overstated** (review T2-1): with breakable graphs, the GDN core
  (`@eager_break_during_capture`) still runs eagerly during replay; the flag captures everything
  around it. The 13 s "eager overhead" was a residual, not a measurement. New candidate below.
- **W8A8 at decode**: in 4-knobs I judged R4' "uncertain, 0-5%"; the cold-L2 graph-replay microbench
  says tuned Triton is ~4% faster than Marlin on decode GEMMs, so one plain-layout kernel can serve
  both regimes (the two-layout memory problem disappears).
- **FP8 fp32-acc peak**: Triton reaches 230 TFLOP/s at M>=1024, so it is above 165 (the open question
  in 3-calibration); the prefill gain is real compute rate, not only Marlin inefficiency.
- **Memory is now a binding constraint** (T2-002): stacking draft-side copies with wider graph pools
  needs either a lower graph cap or KV budget trimming (5 GiB holds 267K tokens; a 128K request needs
  ~half).

## New candidates
- **N1: capture the GDN chunked-prefill core** (or cut its Python launch overhead) in mixed steps.
  Napkin: unknown share of the remaining mixed-step time; profile T2-003 with TRACE=graph first
  (~3 GPU-min) to price it.
- **N2: fp8 draft head (T1-001) on top of T2-003 within memory**: cap prefill graphs at 1024 or trim
  KV to ~4.5 GiB (+128K test). T1 is measuring "fp8 head + cap 1024" (job 16).
- **N3: fuse the activation quant into RMSNorm / SiLU-mul** (if W8A8 wins): ~3 us x 4 GEMMs x 64
  layers = ~0.8 ms per decode step (~2%).

## Current ranking
1. W8A8 tuned Triton on T2-003 (harness running) — napkin +4-5%.
2. N2 fp8 draft head within memory (with T1) — +1-2% on top.
3. N1 GDN prefill core capture — needs a profile to price.
4. N3 fused activation quant — ~+2% if W8A8 lands.
5. Target lm_head kernel choice (ampere 128x64 at 400 GB/s) — +1.4%.
