# Team T3 write-up (leader-3, sweep team; closing)

**Wins: 3 promotions (T3-001, T3-003, T3-007 = champion, 853.6 tok/s, +34.2% vs BASE).**
- T3-001: L2 weight prefetch under each decode all-reduce (816.6, +3.7% vs T2-007).
- T3-003: FP8 drafter lm_head with KV trimmed to 4.4 GiB (826.1, 3 starts, +1.2%).
- T3-007 (promoted at closing): TRITON_ATTN with a bf16 query, so the drafter's attention metadata updates in place
  (853.6, +3.3% vs T3-003, GSM8K 93.6%).

GPU: 83.6 of 90 GPU-min. Map: reports/T3-sweep/ (1-6, update-1, update-2); summary reports/T3-sweep.md.
Records: T3-001..007 (002 and 004 within noise, 005 screen negative, 006 invalid).

## What worked
- **The framework's Window view, applied to resources it doesn't name.**
  - A latency-bound host-staged all-reduce leaves HBM idle for ~46 us, 128 times per step.
  - Prefetching the next GEMM's weights into L2 in that window (a Place onto a tier the framework lacks) paid +3.7%.
  - It delivered its napkin almost exactly, because the gain is bounded by what fits in the window, and that is easy to price.
- **Looking at what is eager, not what is slow.**
  - The biggest late win (T3-007) came from a log line ("cannot update draft decode metadata in place") plus the union-based idle
    time (~2 ms/step).
  - Neither appears in a kernel-time breakdown.
- **Microbenchmarks on the real path before building:**
  - The real 2-GPU hmar plus a cold GEMM picked the 16 MiB prefetch size and the targets.
  - They killed column-split overlap, and the head prefetch once the fp8 copy was in.

## What did not
- in_proj_ba: my calibration summed kernels; it runs on a side stream (review T3-1). Price the critical path, not the sum.
- Column-split GEMM + AR overlap: two serialized half-ARs eat the overlap on a latency-bound link.
- DBO: blocked in this vLLM (hybrid models, HMAR disabled under ubatching, DP>1).
- Exact greedy target fast path: real but ~1%; on top of the fp8 head the per-step host sync (needed for an exact NaN fallback)
  leaves nothing resolvable.
- Flag-only TRITON_ATTN: +4.1% but invalid (FP8 query and FP8 P with an FP8 KV cache). Caught by the review; a one-line
  default fixed it at no measurable cost.

## Lessons (for the framework's authors)
1. Add L2 as a memory tier and the host CPU (launches, syncs, metadata builds) as a window resource. T3's three wins live in those
   two missing columns: L2 prefetch, and the eager metadata rebuild in the drafter. The third, the fp8 head, needed memory headroom,
   also missing.
2. Calibrate on the union of kernel intervals per stream, not the kernel sum; concurrent side streams exist even in "serial" decode.
3. A backend swap is not automatically "the same math": read the dtype path of the kernel (Q/P quantization) before measuring.
4. Collectives: a kernel's duration includes waiting for the other rank (as T1 said); size-match before subtracting.

## Process
- Every record was reviewed (T3-1 map; T3-2 prefetch; T3-3 greedy x2; T3-4 fp8 head; T3-5 Triton x2), except T3-005 (screen negative).
  The reviewer caught three things that mattered: the ba side stream, the greedy NaN semantics, and the FP8 Q/P in TRITON_ATTN.
- Worker leader-3.w1 did the implementation and microbenchmarks (tasks 1-7).

## For the next team
- Profile T3-007 with TRACE=graph to see how much idle is left per decode step and per mixed step.
- GDN DeltaLog: 3 state writes per request per layer confirmed; whether it pays depends on L2 write-back. Check dram bytes first.
- hmar epilogue fusion (~0.8%).
- Greedy fast path without the host sync: a device-side NaN decision.
