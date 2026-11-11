# Round 13 report: leader-2 (team T1)

## Summary
**New best: T1-040 = T1-037 + RED8 GDN reduction layout. Official 929.1 tok/s** (trials 931.3 / 926.8), +0.7% vs T1-037 (922.8), +46.1% vs BASE. GSM8K 94.0%, 128K pass, latency pass. p99 TPOT is 38.4 ms (T1-037: 30.9), under the 55.3 guard; watch it.
The assigned direction (GDN state compression) failed exactly as designed, but its profiling found the real limiter of the GDN kernel, and fixing that gave T1-040.
GPU: 68.9 / 80 used (official evals 21.6, w23 26.7, w24 20.5).

## Experiments
| record | builds on | what | performance |
|---|---|---|---|
| T1-038 | T1-037 | lm_head codec v2 (3-bit offsets + escape pool, ratio 0.705 vs 0.781), L2 prefetch off | official 922.2 = T1-037: no e2e gain despite a -9.4% kernel (ITL flat) |
| T1-039 | T1-037 | lossless GDN fp32 state codec (0.885, flag bit 7, NaN-tag guarded mode 3) | killed: exact, but +15.8 us/layer slower; decode-only with 11.5% fewer bytes = 0 gain |
| T1-040 | T1-037 | RED8: GDN row dots reduced over 8 lanes (3 shuffles per 4 rows instead of 5 per row), one routine on all state paths | **official 929.1** (ABBA +0.94%, ITL -0.47 ms) |
Also: argmax fusion into the lm_head GEMM (exact, +3-8 us; killed; in T1-038 notes). GDN latency hiding (register prefetch, 3 CTAs/SM): bitwise, no gain (T1-040 notes).

## Validity notes
- **T1-040:** fp32 throughout; the same per-element ops; only the summation order of the per-row dots changes (allowed).
  - One shared routine on replay, new tokens, flush and mixed paths, so cap 8/6/0 are bitwise self-consistent (27k checks).
  - vs stock: state 2.4e-7 rel, output 1 bf16 ulp on 0.01%.
  - Greedy c=1: 1/10 identical (expected for a reorder); gsm8k50 = BASE.
- **T1-038:** same lossless rule as T1-037 (bitwise verify with the GEMM's decode); outputs bitwise == T1-037.

## Proposed SHARED_KB
1. **CORRECTION to KB48:** the GDN replay kernel was not HBM-bound. ncu (T1-037 kernel, N=16): DRAM 0.66 TB/s (~65%); the LSU/MIO path 66-72% busy, ~60% of it warp shuffles from 32-lane row-dot reductions; occupancy 27%.
   - That is why 3 CTAs/SM (T1-036) and state compression (T1-039) gave nothing.
   - RED8 cut LSU to 31% and instructions by 35%. The kernel is now latency-bound (long_scoreboard 42%, DRAM 0.68 TB/s): -4 us/layer microbench, -0.47 ms/step in-server (T1-040).
2. GDN fp32 state entropy: mantissa 22.8 of 23 bits are noise; order-0 bound 0.859; best fixed-rate 0.885 (T1-039).
3. T1-037's lm_head is no longer an e2e lever: a -9.4% kernel (-0.1 ms) gave no tok/s or ITL change; the lm_head is likely overlapped or off the critical path now (T1-038).
4. **Process:** a concurrent `ws build` (CPU load) slowed a whole benchmark job by ~5% (job 156), likely via the CPU-sensitive host-staged AR. No CPU-heavy work while GPU jobs run.

## Left open
- GDN latency (w23's proposals, ws leader-2.red8/out/NOTES.md):
  - P1a: speculative chunk-0 loads (4 -> 2 dependent hops).
  - P1b: warp-private cp.async rings with no per-chunk CTA barrier (-2..-4 us/layer).
  - P2: one CTA per (request, key head) streaming its 3 value heads (-5..-6 us/layer at N=16).
  - Premise check first: a clock64 timing build (~1 GPU-min).
- Re-test the GDN state codec only if the kernel ever becomes DRAM-bound.

## Next direction (one)
**GDN replay kernel latency hiding (P1b + P1a, then P2 if the premise holds).**
- Evidence: after RED8 the kernel takes 39.9 us vs ~30 us of streaming at ~0.9 TB/s, and long_scoreboard is 42%. The in-server gain was ~2x the microbench for RED8.
- Estimate: 0.1-0.3 ms/step (P1), up to ~0.5 ms/step with P2 (0.5-2%).
- All exact (same RED8 ops and order, scheduling only).
- Start with the ~1 GPU-min clock64 premise check.
