# Round 6 report from leader-2 (T1): Marlin decode GEMM efficiency -> closed, no new record
Champion unchanged: T1-020 850.3. No candidate reached the go bar; round ended early at 9.4/80 GPU-min (70.6 returned).

Experiments:
- T1-022 (microbench, w10): roofline per decode shape on GPU 0 (2235 MHz). Cold Marlin M=48 at 86-92% of a pure-read roof for big shapes
  (gate_up 106 vs 97.6 us), 73% for 16 MB out_proj/o_proj. Target per step: cold 15.0 ms, as-prefetched 12.8, cold read roof 13.2.
  Warm (L2) gate_up 84 vs ~57 us floor. Config sweep ~0 at M=48. ncu: tensor-pipe bound (math_pipe_throttle; 1 warp/scheduler;
  52-64% tensor peak), slice reduction minor. New instantiations: 256-thread 64x128/128x64 impossible in Marlin; 8-warp 128x128
  3-stage ~0.28 ms/step best case; odd stages on 64x128 wrong/crash. Triton W8A16 (exact Marlin dequant matched, 1 ulp): 1.4-1.7x
  slower at M=48, parity at M=16. NO-GO; only a hand Marlin restructure (2 warps/scheduler) remains: multi-day, <= ~1-1.5 ms/step
  warm-only. Not recommended now.
- T1-021 (w11): out-proj AR "stretch" from prefetch: stop-when-AR-done / denser / 256B-line variants. Microbench -15 us/layer, but
  in-server step 27.33 (T1-020) vs 27.43/27.43 ms: the tail is queued prefetch traffic that gate_up uses; heavier prefetch slows the
  AR write. NO-GO. No bimodality across 3 starts.
- Side: M=16 128x64 instead of default 128x128: ~2-4 us/GEMM (~12 us/step); not filed.

Proposed SHARED_KB:
- KB-a: Decode Marlin cold M=48 is at 86-92% of the HBM read roof (16 MB shapes 73%); config tuning buys ~0. Warm (L2) Marlin is
  tensor-pipe bound (bf16 x fp32-acc, 1 warp/scheduler), ~65% of its floor; Triton W8A16 is 1.4-1.7x slower. (T1-022)
- KB-b: The prefetch "stretch" of the out-proj AR is queued prefetch traffic used by gate_up: not recoverable by stopping/densifying
  the prefetch (in-server +0.1 ms worse). (T1-021)
- KB-c: Wall-time split (w10, T1-016 nsys + T1-020 official arithmetic): ~400 prefill-containing steps per 400-prompt bench, median
  ~98 ms (1 prefill of ~230-330 tokens + ~15 decodes) vs 27.6 ms decode; full-batch decode-only would take ~67 s of the 107 s trial:
  ~28 s lost to mixed steps, ~12 s to batch ramp/drain. In mixed steps (node trace, 18 steps, inflated) Marlin ~80 ms, "other" ~47 ms
  mostly hostar_kernel at ~700 us/call (prefill-size ARs, incl. GPU0 clock skew KB3/KB8), GDN ~10 ms.

Next (one direction): prefill-containing (mixed) steps, ~25% of wall time. Their AR share is large (~700 us per AR at prefill sizes,
x128) and, unlike decode (T1-018: ~10 us/launch overhead killed chunking), at prefill sizes a chunked/two-microbatch pipeline
(split the step's tokens into 2 halves: AR of half A overlaps GEMMs of half B; weights re-read is cheap since prefill is compute-bound)
pays the fixed costs easily. First: an AR-traced mixed-step anatomy on T1-020 (AR wait vs transport at prefill sizes, GDN prefill,
GEMM time), then a microbench of 2-way token-split overlap at M~300. Run-1 named this its #2 idea (copy-engine AR overlapped with
chunked GEMMs).

Notes: no GPU contention; workers w10, w11 stopped. Workspaces kept: leader-2.marlin, leader-2.marlin2 (sweep-only overrides),
leader-2.marlin.sweep, leader-2.pfissue.
