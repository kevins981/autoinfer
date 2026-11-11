# T2 write-up (leader-2, explore). Closed voluntarily at resync 1

GPU: 44.6 of 100 used. Records: bank/T2-001..T2-006. KB: KB59, KB61. Worker: leader-2.w1.

## What we tried: a spatial prefill lane (T2-1, payoff if it works +20%)
- Idea: run new requests' prefills on a green-context SM partition (f SMs), concurrently with decode steps, instead of in serialized ~94 ms mixed steps.
- Premise, measured:
  - Prefill GEMMs at M=256-320 run at 125-140 TF, against a bf16 dense peak of 144 (GPU 0, locked clock) / 158 (GPU 1) (job 1).
  - Green contexts work on sm_89 and partition linearly (job 1).
- Microbench, pure FP8 Marlin chain (T2-001, job 6):
  - Decode M=48 slows only x1.31 at f=48 while prefill keeps ~45 TF.
  - Required configuration: decode on the plain stream with a Marlin grid sized 128-f, prefill in the green ctx.
  - Configurations that fail: decode in a green ctx (1.8-5x slower), the default Marlin grid (+45% from 2 waves), stream priorities (graph replays serialize).
- In-server probe (T2-002, ABBA): a synthetic lane of real per-rank linear modules at M=256 plus copy-engine AR traffic.
  - Decode ITL slows x1.46 / x1.65 / x1.85 at f=32/48/64, with the lane at 22 / 36 / 45 TF. Kill gate failed.
- Ablation (T2-003, T2-004): not the PCIe copies (+4 ms only), not cross-rank amplification (lane on one rank: -2.4 ms), not the Marlin grid, not CPU/GIL (a CUDA-graph lane gives the same ITL, 41.9 vs 25.3 OFF, same job).
- **KB59:** the full champion decode step slows as ~128/(128-f), unlike a pure Marlin chain. Prefill is also ~1 TF/SM, so spatial SM sharing is zero-sum here (f=48: decode at 0.61 of full speed plus lane at 0.29 of the full prefill rate = 0.90 < 1).
  - This rules out the whole family of NanoFlow/Bullet-style intra-GPU overlap on this stack, unless something makes decode stop needing every SM.

## Tooling pitfall (KB61)
`ws create --from <record>` keeps the prebuilt BASE `_C` lib until `ws build`. An unbuilt T1-042 workspace silently runs without GDN replay or the csrc changes. Our first anatomy (T2-005) was invalid for this reason. Always build, and check kern_sum for the expected kernels.

## The champion's step anatomy (T2-006, job 22, critical path on rank 0, median ms)
| family | decode | mixed pt<150 | pt 150-399 | pt>=400 |
|---|---|---|---|---|
| period | 23.8 | 76.0 | 91.1 | 144.2 |
| Marlin | 13.6 | 33.0 | 84.3 | 157.7 (2 streams) |
| host-AR wait | 5.9 | 10.2 | 11.1 | 29.9 |
| AR memcpy exposed | 0 | 0 | 7.5 | 12.8 |
| GDN | 2.0 | 4.4 | 5.9 | 8.5 |
| norm/act/elementwise | 0.4 | 3.0 | 3.6 | 5.5 |

Steps per trial (scaled from the 64-prompt capture): decode ~3150, mixed ~150 / 50 / 119.
- The GPU never idles between steps. The ~18 ms "host gap" in small mixed steps is host run-ahead (~1790 launches per mixed step); it becomes a lever only if GPU work gets much faster.
- Host AR is at the transport floor at every size: 0.61 us/token + 15.5 us per call on rank 0. Rank 1 waits 20-40% longer for the slower-clocked GPU 0.
- Prefill Marlin runs at ~85% of its microbench rate.
- **Decode floor estimate:** ~12 ms weights + 3.8 AR bytes + 2.1 AR latency (137 x 15.5 us) + 2.0 GDN + ~2.3 other ≈ 22 ms, vs 23.8 measured.

## Small exact mixed-step items (not built; together +1-2% at most, each below noise)
| item | s per trial | confidence |
|---|---|---|
| DMA-AR combine `add_kernel` (dma_staged_ar.cu:61) fused into the following FusedAddRMSNorm | 0.3-0.5 | medium |
| index_select/gather glue in GDN (qwen_gdn_linear_attn.py:1878-1881, 1954-1955) replaced by slices (decode rows are contiguous) | 0.15-0.3, plus ~1000 fewer launches/step | medium |
| silu*mul fused into the gate_up Marlin epilogue | 0.3-0.5 | low-medium |
| DMA pipeline tail after each chunk's last GEMM (7.5 / 12.8 ms per step) | 0.3-0.7 | low |
| in-span idle of mixed steps (eager-piece launch gaps) | <= 1.3 | low |
| mixed-step GDN replay vs decode-graph speed | <= 0.2 | low |

## Where the headroom is (opinion)
- Every component of the decode step is within ~10% of a measured floor. Prefill compute is at the bf16 tensor roof, and FP8/INT8 compute is forbidden or (exact Ozaki) needs ~6 int8 passes, which is no faster.
- The remaining large lever is **tokens per decode step** (T3's area: KB60 shows fine-tuning the MTP head doesn't move acceptance; T3-2 budgets verification rows).
- A second, smaller one is **AR latency**: 2.1 ms/step is fixed per-call cost (fence/doorbell), not bytes. Fewer calls or a cheaper sync would help, but data dependencies allow little.
- The mixed-step items above are worth stacking onto someone's official run, not running on their own.
