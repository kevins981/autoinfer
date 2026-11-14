# map.md
Changelog: 2026-11-14 proposals/T2-map-1.md (T2): mixed-step rank-skew caveat, 3 knob sites, 7 rows, ReplaySSM row refined, method notes. | 2026-11-14 proposals/T1-map-1.md (T1): initial skeleton (time split, pricing rules, knob sites, status table).

# Map of the search space (Qwen3.8-27B-FP8, 2x RTX 4090, TP=2, c=16)

## 1. What the space is
**Time split (BASE):** decode steps 72.5% of benchmark time (38.5 ms each, 16 req x 3 rows), mixed
(prefill + decode) steps 27.5% (~148 ms each, ~330 prompt + 45 decode rows). Overall gain of a change ≈
0.725·ΔD/38.5 + 0.275·ΔP/148. Tokens per decode step = 16 x acceptance (2.345 at 2 drafts).
**Decode step parts (per GPU, BASE):** Marlin FP8 GEMMs 18.6 ms (12.2 GB at 0.66 of 1 TB/s) · all-reduce
7.7 ms (135 x ~58 us, NCCL via host SHM) · lm_head bf16 1.27 GB read 3x/step (target + 2 drafts) 4.2 ms ·
logits all-gathers 1.4 ms · GDN state update 3.0 ms (fp32 state r+w 2.1 GB) · MTP layer x2 ~0.9 ms ·
small kernels ~1.5 ms · idle 1.3 ms. Napkin at spec rates ~24 ms.
**Mixed step parts:** Marlin 81 ms (compute-bound, ~113 TFLOP/s bf16) · all-reduce 39 ms (294 us each at
375 rows) · GDN ~10 ms · rest ~18 ms.
**Caveat (T2):** about half of the mixed-step "all-reduce" time is one rank waiting for the other. In
BASE's profile, the two GPUs' AR kernels in prefill steps start a median **150 us** apart. Once the
later rank arrives, an AR takes ~133 us (GPU0's AR kernels show a 305 us median, GPU1's 182 us).
Decode steps are in lockstep (skew 10 us, ~60 us after both arrive). So mixed steps are partly
CPU/launch-bound or rank-imbalanced (24.5 ms GPU idle under nsys; real size not yet measured
without nsys). A faster AR kernel alone will not remove this part.
**Pricing rules (calibrated):** NCCL AR(n rows of 5120 bf16) ≈ 53 us at n=42, +0.72 us/row (~7.3 GB/s payload);
Marlin ≈ bytes / 0.66 TB/s at M≤64 in-situ (0.8 isolated per ncu); bf16 GEMV lm_head ≈ 0.9 of BW; launch
floor ~3 us/kernel. Link: no GPU P2P; all traffic via host memory over PCIe 4.0.
**Dimensions / knob sites:**
- Combine nodes (128 all-reduces/forward + drafter's): implementation, overlap with GEMMs (split d), payload.
- Body GEMMs: kernel efficiency at M=42-64 (decode, BW-bound); FP8 W8A8 for mixed steps (compute-bound).
- Speculation: draft depth, drafter cost (draft lm_head precision/vocab, logits gather), tree vs chain.
- lm_head / sampling: logits all-gather vs local argmax (greedy), target and draft side.
- GDN state (fp32, in-place): kernel efficiency, write-back frequency (ReplaySSM).
- CPU/launch: idle between graphs, eager mixed steps, draft metadata rebuild.
- Parallelism: TP vs PP vs hybrid (see verdicts).
- **Rank lockstep / host CPU in mixed steps:** eager parts (GDN prefill FLA kernels, attention metadata) can desynchronize the ranks; the cost shows up inside the AR kernels as waiting. The FLOPs/bytes model has no CPU resource for it.
- **L2 (72 MB) as a memory tier filled during link-bound phases:** while an AR runs, DRAM is idle. A concurrent prefetch of the next GEMM's weights (in_proj 42 MB; first ~40 MB of gate_up's 89 MB) into L2 could hide up to the AR time per AR.
- **Compute dtype at decode, not only mixed steps:** Marlin at M~45 does bf16 MMA on dequantized FP8. That is ~96 flop/byte, against a ~165 bf16 ridge, and it must run at the same time as full DRAM bandwidth. An FP8xFP8 block-scaled kernel (2x MMA rate, no dequant) could move decode GEMMs toward the bandwidth roofline.

## 2. What we know, per part
| part | status | evidence |
|---|---|---|
| NCCL env tuning (PROTO Simple/LL/LL128, SHM_USE_CUDA_MEMCPY, NTHREADS/MIN_NCHANNELS) | **tested, no gain**: default already Simple at n≥42; LL/LL128 20-100% worse; others within noise | T1 job 1 (microbench) |
| FlashInfer PCIe-IPC all-reduce (env VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC) | **impossible here**: needs CUDA peer access ("peer access is not supported") | T1 job 1 |
| vLLM custom all-reduce | **impossible here** (needs P2P; disabled in BASE log) | BASE server log |
| zero-copy AR through mapped pinned host memory | untried (T1 testing) — napkin ~+10% | T1 6-ranking #1 |
| AR/GEMM overlap (split row-parallel output along d; RMSNorm scale commutes past next GEMM) | untried — napkin ~+9% | T1 4-knobs W1 |
| FP8 draft lm_head (`draft_lm_head_quantization: fp8`, ruling 1) | untried (T1 testing) | engine option |
| local argmax for draft tokens (`use_local_argmax_reduction`) | untried (T1 testing) | engine option |
| local argmax for target verification (greedy, generic fallback) | untried — ~+2% | T1 4-knobs R1 |
| reduced-vocab draft lm_head (ruling 2) | untried — ~+5% with fp8 | T1 4-knobs R3b |
| draft depth 3-4 (chain) | untried — +4% now, +9-11% with a cheap drafter | T1 4-knobs S1 |
| tree speculation | not now: GDN state must fork per tree node (needs a tree-aware GDN verify kernel) | T1 4-knobs |
| W8A8 FP8 GEMMs in mixed steps | untried — ~+6.5%; kernel speed on sm_89 unknown | T1 4-knobs R5 |
| Marlin decode efficiency (0.66 in-situ vs 0.8 isolated) | untried — ~+6% | T1 3-calibration |
| PP=2 / asymmetric PP | **not worth doing**: at decode each GPU must stream its 12.2 GB serially (~48 ms/step vs 38.5) and microbatches re-read weights; AR saved (7.7 ms) < cost | T1 4-knobs (napkin) |
| sequence parallel (RS+AG) | not worth doing: same bytes, 2x latency terms | napkin |
| lm_head replication | not worth doing: doubles lm_head bytes per GPU, costs KV budget | napkin |
| draft-in-verify bubble / decode microbatch split | not worth doing at decode: Shared-Read Rule doubles weight reads at N=48 ≪ ridge | napkin |
| mixed-step 2-microbatch AR overlap | untried — up to +7%, shrinks if AR gets faster | T1 4-knobs |
| bf16 GDN state, quantized comms | **forbidden** (lossy) | task_definition |
| ReplaySSM / DeltaLog for GDN | untried. Today the MTP decode kernel **writes the state after every row** (3 slots/request/layer/step: csrc/gdn/fused_gdn_decode_kernel.cu, store_state4 inside the per-token loop). The simplest variant writes only the verified row's state plus the drafts' tiny inputs, and the next step replays the accepted drafts from them (same fp32 ops, so bitwise identical). It saves 2 of 3 writes. Napkin up to 2.3 ms/step, but measured GDN is only ~2.95 ms/step (61 us/layer, below its own byte napkin of ~94 MB/layer), so ~1-2.6%. Check dram bytes with ncu first. | T2 4-knobs K1a, 3-calibration |
| NCCL protocols/channels at 16-2048 rows + logits AG | **tested, no gain** (corroborates T1): default best or tied; LL/LL128 1.5-2x worse at >=48 rows; 4-8 channels +3-7% only at >=512 rows, -10-20% at 48-64 rows; AG [45x124160] 891 us, [16x124160] 324 us (~12 GB/s) | T2 job 2 (agents/leader-2.w1/out/) |
| mixed-step rank skew (lockstep) | **untried, newly found**: make mixed steps GPU-bound (CUDA-graph more of the mixed path, fewer launches); napkin 0-10% overall, size unknown until measured without nsys | T2 4-knobs W4b, BASE profile analysis |
| L2 prefetch of next GEMM's weights during AR | **untried, new**: napkin up to ~5 ms/decode step (~9%) at today's AR length, ~3 ms if AR gets ~2x faster; risk: L2 residency vs Marlin's cache hints | T2 4-knobs W2 |
| W8A8 FP8 GEMMs at decode (M~45-60) | **untried**: napkin decode Marlin 18.5 -> ~14 ms if 0.85 TB/s (~7% overall), on top of the mixed-step gain | T2 4-knobs R2 |
| replicate embedding (drop 3 embedding ARs/step) | not worth doing: 0.18 ms/step (0.3%); costs 1.27 GB/GPU | T2 4-knobs |
| whole draft on one GPU (no draft comm) | not worth doing: dominated by reduced-vocab draft lm_head at TP=2 (same saving, no idle GPU) | T2 4-knobs |
| merging more prefills per step | not worth doing: mixed steps already compute-bound at M~260 | T2 4-knobs |


## 3. Method notes (framework vs this machine)
- Calibrate before ranking: the spec-rate napkin gives ~25 ms/decode step against 38 ms measured. The big gaps are Marlin efficiency (+6.3 ms) and NCCL (+4.4 ms); the window max() assumes a perfect overlap that a kernel near the ridge doesn't achieve.
- Link bytes/BW underestimates NCCL through host memory by 2.3x (460 KB) to 4.3x (2.6 MB, in-server).
- No CPU resource in the framework; mixed steps here need one.
