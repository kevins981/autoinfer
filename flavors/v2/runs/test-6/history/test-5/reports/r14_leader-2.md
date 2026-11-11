# Round 14 report: leader-2 (team T1)

## Summary
**New best: T1-042 = T1-041 + DMA AR sub-chunk pipelining. Official 942.2 tok/s** (trials 941.4 / 942.9), +1.4% vs the round's starting champion T1-040 (929.1), +48.2% vs BASE. GSM8K 93.2%, 128K pass, latency pass.
Two official gains this round, both exact (bitwise):
- **T1-041 (GDN scheduling, mode 5): 934.1.**
- **T1-042 (AR copy pipelining): 942.2.**
The assigned direction (GDN latency hiding) mostly failed its premise: the hop chain is < 1 us, so only ~1.6 us/layer was hideable. That was banked as T1-041. A step anatomy then pointed to exposed AR communication in mixed steps, which gave T1-042.
GPU: 58.8 / 80 (official evals 21.3, w23 37.4).

## Experiments
| record | builds on | what | performance |
|---|---|---|---|
| T1-041 | T1-040 | GDN replay mode 5: batched prologue + warp-private cp.async staging (no per-chunk CTA barrier); bitwise == T1-040 | **official 934.1** (+0.5%); microbench -1.65 us/layer |
| T1-042 | T1-041 | DMA AR lane pipelined in 4 pieces (per-piece doorbells, aux stream, graph-capturable); bitwise | **official 942.2** (+0.9%); ABBA +1.3%; 3.93 MB AR 338 -> 255 us |
| T1-043 | T1-042 | residual add + gemma RMSNorm fused into the host-AR epilogue | killed: lean ABBA -0.6%; only -0.7 us/AR at M=48 |
Not filed, closed on evidence:
- P1a speculative chunk-0 loads and P2 one-CTA-per-key-head: the premise failed (hop chain 0.75 us).
- q/k-norm dedupe: est < 0.3 us/layer.

## Validity notes
- **T1-041:** scheduling only. Each warp stages exactly the rows it reads, so dropping the CTA barrier is race-free. Bitwise over 27k checks; negative control fails.
- **T1-042:** the same bytes and the same add kernel (bitwise vs 1 piece at M = 1-1200, incl. graphs).
  - Per-piece controls with the delay before each piece's D2H: skipping any single piece's ready fails (10.5-12.0M diffs).
  - Ack-gated reset protocol rules out stale doorbells.
  - Test hooks are inert in production (dbg = 0).
- **T1-043:** res' bitwise, y <= 1 bf16 ulp (fp reorder of the sum of squares). Also found and fixed a prefetch-stride bug in that branch only (it never affected champion code).

## The p99 TPOT question (manager)
Official p99 TPOT is noisy: T1-023 33.9, T1-025 41.4, T1-026 34.3, T1-028 40.1, T1-030 32.1, T1-035 39.7, T1-037 30.9, T1-038 35.3, T1-040 38.4. T1-040's 38.4 sits inside the 31-41 band; T1-037's 30.9 was the low draw. T1-041/042's ABBA arms show the same spread (36-44 in both A and B). No cause found because none is needed: it is noise from the ~4 worst requests.

## Proposed SHARED_KB
1. **GDN replay after RED8:** per-CTA time is prologue 8.6 us (first-load latency under DRAM contention), hop chain 0.75, chunk waits 1.6 (5.7 with GEMM), compute 7.9 us. Scheduling can hide only ~1.6 us/layer (T1-041); more CTAs, deeper rings and dedupe do nothing.
2. **DMA AR copies pipeline well on the copy engines**, unlike the SM path in KB5: 3.93 MB AR 338 -> 275 (2 pieces) -> 255 us (4 pieces); 6 MB 501 -> 365. Next to a hostar AR on another stream: 431 -> 367. Official +0.9% (T1-042).
3. **Step anatomy of T1-041** (job 171, agents/leader-2.w23/anat/REPORT.md):
   - decode 25.5 ms: Marlin 13.5 / AR 6.2 / GDN 2.1 / lm_heads 1.6 / attention 0.87 / norms 0.47 / small bf16 GEMMs 0.35 / idle 0.22;
   - mixed steps ~39% of wall time, of which exposed AR communication ~11%.
4. **AR + RMSNorm fusion doesn't pay** here: the norm is 2.6-3.6 us but the fused epilogue costs ~1.5 us inside the AR tail; net -0.7 us/AR at M=48 and -0.6% e2e (T1-043).
5. **Negative controls for cross-rank readiness** must delay the producer BEFORE its copy. A delay between the copy and the doorbell leaves the data already in place, so skipped readies pass by accident (T1-042, job 175).

## Left open
- Small-M Marlin 2-slab cliff in mixed steps (M 105-180, ~0.22 s of the 14.1 s window): needs a Marlin restructure, too big for one round.
- Attention: 0.87 ms/decode step vs ~0.5 floor, 6 small kernels per layer.

## Next direction for round 15 (one round, finishable)
**Replace the GDN ba-projection's cuBLAS bf16 GEMM + splitK reduce with one skinny exact kernel** (and the draft fc if trivial).
- Evidence: anatomy row "small bf16 GEMM 0.35 ms/decode step vs ~0.1 floor". The ba projection is 7.7 us x 48 layers for 0.5 MB of weights per layer, i.e. far from its ~1 us bandwidth floor, and is 2 launches (GEMM + splitK).
- Estimate: -0.2..-0.25 ms/step, ~+0.8-1%.
- Exact up to fp reordering (fp32 accumulation, one bf16 rounding).
- Small and self-contained: microbench at M = 3-64 and ~300 -> drop-in -> ABBA -> official, within one round.
