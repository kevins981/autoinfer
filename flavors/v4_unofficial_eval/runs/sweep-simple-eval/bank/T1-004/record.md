team:         T1
builds_on:    T1-002 (microbenchmark only, no engine change)
performance:  "not an engine change: microbenchmark negative for W-DBO"
summary:
  hypothesis: W-DBO: in mixed prompt steps (~260 rows, compute-bound GEMMs), split rows into 2 halves and overlap AR(half A) with GEMM(half B) on a second stream; napkin hides most of ~28 ms AR transfer per prompt step (+3..6%).
  change:     none in the engine. 2-process microbenchmark (dbo_bench.py, job.sh here): bf16 [M,5120] x [5120,17408] GEMM (proxy for compute-bound Marlin gate_up) on one stream, NCCL all-reduce of bf16 [M,5120] on another, 4 ops each, median of 12.
  results:    Observed (GPU job 18, 0.2 GPU-min), per op:
                NCCL Simple: M=130 GEMM 246 us, AR 139 us, concurrent 315 us per pair (serial 385): hides 50% of the AR. M=260 GEMM 395, AR 250, concurrent 506 (serial 645): 56%.
                NCCL LL:     M=130 AR 247 us, concurrent hides 47%; M=260 AR 470, hides 41%.
              Projection per layer for a 260-row step: today GEMM(260)+AR(260) = 645 us; DBO with 2 halves = 2 x GEMM(130) + 2 x AR(130) - overlap ~ 492 + 278 - ~140 = ~630 us.
  conclusion: Interpretation: W-DBO gains ~0-3% of a mixed step (well under 1% of the benchmark). Two things kill it: (1) the GEMM at half the rows is 25% less efficient (246 x 2 = 492 vs 395 us: at 130 rows the matmul is not fully compute-bound / tile-efficient), eating the hidden AR time; (2) the overlap is only ~50%: NCCL kernels take SMs and the host path, so the window formula's max(F, Link) is optimistic (framework 07 §2.3, interference). Dropped. Same reasoning weakens W3 (column-chunked GEMM || AR) for decode: chunking also costs GEMM efficiency and overlap is partial.
validity:     n/a (no engine change)
