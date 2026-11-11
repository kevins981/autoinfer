# leader-15 (T15, explore) working notes

## Context (2026-11-09)
Champion T9-005 948.9; bar +5% = 996.3. Prior teams: decode at coupled HBM + bf16-tensor floors (KB69/T10),
comm zero-sum (KB59/62/70/72), speculation ceiling (KB63), prefill compute at the bf16 roof (~20 s of 96 s).
T14 owns mixed-step Marlin M-handling + glue + admission (T14-1).

## Direction candidate D1: a second exact compute pipe (CUDA-core FFMA beside the half-rate bf16 HMMA)
- On GeForce Ada, bf16/fp16 MMA with f32 accumulate is half rate: 165 TF (GPU1) / 146 TF (GPU0, KB69).
  FP32 FFMA is 82.6 TF at 2.52 GHz (73 TF on GPU0 at 2235 MHz): ~50% of the tensor rate (vs ~6% on A100).
- bf16 x e4m3 product = 12 significant bits: exact in fp32; FFMA accumulation is IEEE fp32 (reordering class).
  So an FFMA share of the GEMM is exact-class, unlike FP8 MMA (T10) or fp16-acc MMA.
- Nobody in the run checked whether HMMA and FFMA pipes co-run (grep: no hits).
- Where it pays: compute-bound prefill/mixed GEMMs (~19-20 s/trial at GPU0 roof) and the M=48 decode tensor
  half of the coupled floor (KB69: lifts AR-window prefetch recovery, deeper k).
- Step 1: toy co-issue microbench (agents/leader-15/mb/copipe.cu, job 115).
- Job 115: HMMA and FFMA do NOT co-issue (sum of times). D1 dead. bank/T15-001.
## D2 candidate: PCIe duplex for the decode AR (KB5 predates NUMA fix)
- Jobs 117/118: SM-W+SM-R 25 us (384 KB) vs 17.3/19.0 alone (KB5 holds); CE+CE 19.0; CE-D2H+SM-R 20.8. bank/T15-002.
- Hybrid AR estimate ~31 us vs 41 -> ~0.75 ms/step net, ~+2% e2e. Below bar alone.
- Decode imbalance (T2-006 trace): GPU0 Marlin +1.6% decode, +15% mixed; rank1 AR wait ~3.2 us/AR mean.
- Package sum ~4.3-4.7% payoff: below the bar. Recommendation to manager: close T15, return seed. Write-up agents/leader-15/T15-writeup.md
