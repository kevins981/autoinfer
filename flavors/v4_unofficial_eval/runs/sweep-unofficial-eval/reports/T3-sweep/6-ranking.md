# T3 sweep, step 6: ranked candidates (against HEAD; measured on T1-016, HEAD is now T1-017)

Time model: 200-prompt run ~55-57 s; decode-only steps ~63%, mixed steps ~37%; a mixed step = decode step +
~65 ms fixed + ~0.1 ms per prompt token (job 155). Requests finish (and new ones arrive) every ~10 decode steps.
Gains are napkin estimates, not additive where they spend the same slack.

| rank | id | candidate | napkin gain | wins when | riskiest assumption | cheapest test that could show it wrong |
|---|---|---|---|---|---|---|
| 1 | **A1** | **batch prefill admissions**: `--prefill-schedule-interval N` (vLLM's existing generic "prefill cadence": a prefill is admitted at most every N steps; decode keeps running; capacity escape when backed up). Merges ~1+N/10 prompts per mixed step | per prompt: saves F(1-1/k) with F ~65 ms, k ~ 1+N/10; costs ~N/2 steps of one empty slot = N/32 x 28 ms. N=10: +32 - 9 = ~23 ms/prompt -> **~8%**; N=20: ~26 ms but TTFT +~0.5 s | the fixed part of a mixed step is per step (shared by its prompts), and p99 TTFT stays < 1.2 x BASE (1610 ms; HEAD ~1000) | that F is per step, not per prompt (evidence: start-up step with 10 prompts took 37 ms per prompt vs ~94 when alone; few 2-prompt steps observed) | screen N in {8,16} at 200 prompts vs HEAD's cached 833 (gain expected >> noise), then unofficial_eval of the best (~5 GPU-min) |
| 2 | A2 | after A1: tune the mixed-step graph sizes / W8A8 thresholds for the merged sizes (M ~150-600) | 1-2% | merged steps fall into the W8A8 route and captured graph sizes | that merged steps stay <= 512 tokens (graphed) | read the step sizes in an A1 profile |
| 3 | B' | host AR for mid sizes (129-512 rows) instead of NCCL ring (11 GB/s) in graphed mixed steps | NCCL 329 us vs duplex host path ~225 us at 3.6 MB: -13 ms per mid/large mixed step, **~2%** (more after A1: merged steps are bigger) | host AR scales with blocks at large sizes | isolated bandwidth at 1-5 MB with 16-64 blocks | t3-w1 task 1 microbench (running) |
| 4 | R7b | GDN in_proj + attention qkv to the checkpoint layout (W8A8 route at large M; read once at 65-127 rows) | in_proj+qkv = 2.6 of 11.2 GB Marlin weights: -3..-8 ms per mixed step, **~1.5-2%** | as T1-017 | that T1 isn't already doing it | ask leader-1 (T1 is tuning W8A8 configs, T1-018) |
| 5 | SM3'' | chunked GEMM || AR inside graphed mixed steps (<= 512 tokens), side stream in capture | -10..-20 ms per mid mixed step, ~2% (less after B') | AR and GEMM don't slow each other (T2: link and HBM coupled for host AR; NCCL here) | the coupling | microbench Marlin/W8A8 GEMM || NCCL AR on 2 streams |
| 6 | SM2 | decode column-split GEMM || host AR | ~-2 ms/decode step, ~4% | host AR's PCIe traffic doesn't slow a concurrent GEMM | the coupling seen in T2-005..007 (prefetch) | 2-stream microbench (Marlin down GEMM || host AR at M=48) |
| 7 | G3 | decode sync points (NCCL SendRecv 455 us, 5 AllGathers 110 us per step) | <= 0.5 ms/step, ~1% | they wait on something removable | they may just absorb rank skew | read their position in the step timeline from job 153 (free) |
Dropped after 5-compare: decode AR speed (at duplex floor), L2 prefetch (coupled), 3 drafts / tree / lazy K=3
(measured negative), CPU-dispatch dedup (nsys-inflated), AR+norm fusion (tested null).

Plan (80 GPU-min, 75.5 left): (1) A1 screen (~6 GPU-min) -> record + unofficial_eval; (2) B' microbench (running) ->
build if the mid-size floor is ~1.5x below NCCL; (3) coordinate R7b with T1; (4) 2-stream coupling microbench for SM2/SM3''.
