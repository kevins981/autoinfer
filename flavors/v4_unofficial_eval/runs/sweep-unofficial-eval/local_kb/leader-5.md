# leader-5 (T5) working notes

## Setup
- workspace t5-head = T3-004 (built). Profiles in agents/leader-5/prof/{head_node,head_graph,head_nvtx}.
- scripts: agents/leader-5/scripts/decode_kernels.py (per-step kernel medians; arg -1 = prefill steps 200..330 tok)

## Findings (HEAD T3-004, nsys job 201, 64 prompts)
- decode step (M=48): ~25 ms (graph trace), GPU-bound (CPU 4 ms/step).
  w8a16 GEMMs 15.9 ms (~790 GB/s, 78% of spec); host_ar 135 x 42 us = 5.7 ms; lm_head cuBLAS 1.58 ms;
  GDN fused 1.83 ms (at BW); in_proj_ba cuBLAS Kernel2+splitKreduce 0.88 ms (should be ~0.1); draft head 0.63.
- host_ar kernel: write-all -> flag -> wait -> read-all, grid 4: not pipelined (2 x 20 us at 25 GB/s).
- mixed (prefill+decode) steps: ~1800 launches, CPU-bound (piecewise graphs + eager GDN prefill),
  preceded by a blocking D2H memcpy (no async overlap). Under nsys: ~100 ms for 60-token prefill.
  Prefill-containing steps = 37% of profiled time (nsys inflates CPU).
- prefill AR > 1.25 MB uses NCCL RING_LL ~10 GB/s (325 us for 3.2 MB).

## Phase 2 (done) / state
- 5-compare: pipelined AR, in_proj_ba (aux stream), k=3 all tested negative before. Mixed-step CPU overstated by me.
- C1 lossless AR encoding: manager ruled ALLOWED (bitwise stress test incl. -0/subnormal/inf/NaN required).
- worker t5-w1 (ws t5-w1) on C1 step A (dump real AR payloads, offline ratio; gate <= 0.82), allowance 12 GPU-min.
- told leader-4: I don't take their #3/#4 (W8A16 redesign, lm_head GEMV).
