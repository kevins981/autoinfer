# T2 map proposal 1 (leader-2, after T2's phase-1/2 sweep: reports/T2-sweep/1..6)

## Changes
### A. §1 "Mixed step parts": add a caveat line after "Mixed step parts: ..."
> **Caveat (T2):** about half of the mixed-step "all-reduce" time is one rank waiting for the other. In
> BASE's profile, the two GPUs' AR kernels in prefill steps start a median **150 us** apart. Once the
> later rank arrives, an AR takes ~133 us (GPU0's AR kernels show a 305 us median, GPU1's 182 us).
> Decode steps are in lockstep (skew 10 us, ~60 us after both arrive). So mixed steps are partly
> CPU/launch-bound or rank-imbalanced (24.5 ms GPU idle under nsys; real size not yet measured
> without nsys). A faster AR kernel alone will not remove this part.

### B. §1 "Dimensions / knob sites": add three bullets
- **Rank lockstep / host CPU in mixed steps:** eager parts (GDN prefill FLA kernels, attention metadata) can desynchronize the ranks; the cost shows up inside the AR kernels as waiting. The FLOPs/bytes model has no CPU resource for it.
- **L2 (72 MB) as a memory tier filled during link-bound phases:** while an AR runs, DRAM is idle. A concurrent prefetch of the next GEMM's weights (in_proj 42 MB; first ~40 MB of gate_up's 89 MB) into L2 could hide up to the AR time per AR.
- **Compute dtype at decode, not only mixed steps:** Marlin at M~45 does bf16 MMA on dequantized FP8. That is ~96 flop/byte, against a ~165 bf16 ridge, and it must run at the same time as full DRAM bandwidth. An FP8xFP8 block-scaled kernel (2x MMA rate, no dequant) could move decode GEMMs toward the bandwidth roofline.

### C. §2 table: add rows
| part | status | evidence |
|---|---|---|
| NCCL protocols/channels at 16-2048 rows + logits AG | **tested, no gain** (corroborates T1): default best or tied; LL/LL128 1.5-2x worse at >=48 rows; 4-8 channels +3-7% only at >=512 rows, -10-20% at 48-64 rows; AG [45x124160] 891 us, [16x124160] 324 us (~12 GB/s) | T2 job 2 (agents/leader-2.w1/out/) |
| mixed-step rank skew (lockstep) | **untried, newly found**: make mixed steps GPU-bound (CUDA-graph more of the mixed path, fewer launches); napkin 0-10% overall, size unknown until measured without nsys | T2 4-knobs W4b, BASE profile analysis |
| L2 prefetch of next GEMM's weights during AR | **untried, new**: napkin up to ~5 ms/decode step (~9%) at today's AR length, ~3 ms if AR gets ~2x faster; risk: L2 residency vs Marlin's cache hints | T2 4-knobs W2 |
| W8A8 FP8 GEMMs at decode (M~45-60) | **untried**: napkin decode Marlin 18.5 -> ~14 ms if 0.85 TB/s (~7% overall), on top of the mixed-step gain | T2 4-knobs R2 |
| replicate embedding (drop 3 embedding ARs/step) | not worth doing: 0.18 ms/step (0.3%); costs 1.27 GB/GPU | T2 4-knobs |
| whole draft on one GPU (no draft comm) | not worth doing: dominated by reduced-vocab draft lm_head at TP=2 (same saving, no idle GPU) | T2 4-knobs |
| merging more prefills per step | not worth doing: mixed steps already compute-bound at M~260 | T2 4-knobs |

### D. §2 table: correct the ReplaySSM row (T1's entry; correction with evidence)
Replace "ReplaySSM (GDN state write every L steps) | untried — ~+2%, high effort" with:
> ReplaySSM / DeltaLog for GDN | untried. Today the MTP decode kernel **writes the state after every row** (3 slots/request/layer/step: csrc/gdn/fused_gdn_decode_kernel.cu, store_state4 inside the per-token loop). The simplest variant writes only the verified row's state plus the drafts' tiny inputs, and the next step replays the accepted drafts from them (same fp32 ops, so bitwise identical). It saves 2 of 3 writes. Napkin up to 2.3 ms/step, but measured GDN is only ~2.95 ms/step (61 us/layer, below its own byte napkin of ~94 MB/layer), so ~1-2.6%. Check dram bytes with ncu first. | T2 4-knobs K1a, 3-calibration

### E. New §3 "Method notes" (framework vs this machine)
- Calibrate before ranking: the spec-rate napkin gives ~25 ms/decode step against 38 ms measured. The big gaps are Marlin efficiency (+6.3 ms) and NCCL (+4.4 ms); the window max() assumes a perfect overlap that a kernel near the ridge doesn't achieve.
- Link bytes/BW underestimates NCCL through host memory by 2.3x (460 KB) to 4.3x (2.6 MB, in-server).
- No CPU resource in the framework; mixed steps here need one.

## Why
All numbers are from reports/T2-sweep/3-calibration.md and 4-knobs.md (BASE profile analysis plus T2 GPU job 2). The skew analysis matches each GPU0 AR kernel in the BASE profile's prefill steps to GPU1's AR kernel with the nearest end time (20 steps, 2,164 matched ARs).
