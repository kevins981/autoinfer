# T1 sweep, step 6: ranked candidates

Gains are napkin estimates against BASE's calibrated 143.2 s (3-calibration.md), corrected by 5-compare.md. They are not additive
where they spend the same slack (noted).

| rank | id | candidate | napkin gain | wins when | riskiest assumption | cheapest test that could show it wrong |
|---|---|---|---|---|---|---|
| 1 | X1 | faster all-reduce: NCCL proto/channel tuning, else a custom one-shot AR through pinned host memory mapped in both GPUs (graph-capturable, flag-synchronized) | **~15%** (decode 60->~25 us: 7%; prefill ~520->~200 us: 9%) | host path sustains >= ~20 GB/s per direction with ~5 us sync latency | that the PCIe/host path is much faster than NCCL LL achieves (the 3x gap is NCCL's, not the bus's) | AR microbenchmark across sizes (running: t1-w1, <= 8 GPU-min) |
| 2 | W1/W3 | overlap prefill-step ARs with GEMMs (half-batch stagger or column chunks) | ~13% at BASE AR speed; ~4% after X1 | comm kernel/copies don't steal much SM/HBM from Marlin; prefill GEMMs stay compute/re-read bound | contention: NCCL/custom AR + Marlin on the same SMs | microbenchmark: Marlin GEMM [437x5120x...] on stream A || AR on stream B, time vs serial |
| 3 | L2 | prefetch next GEMM's weights into L2 during an AR (T2's site) | up to ~9% at BASE AR speed; shrinks with X1 | 72 MB L2 keeps ~40 MB of prefetched weights until the GEMM runs | L2 residency on sm_89 without eviction by the AR/norm kernels | microbench: prefetch kernel (or cudaAccessPolicyWindow persisting) + Marlin, measure GEMM time drop |
| 4 | R2a | read-once large-M GEMM for prefill rows (Marlin re-reads weights per 64 rows) | ~5% | a kernel reading each weight once at M~437 runs near 165 TF (bf16 or W8A8/fp32 acc) | that such a kernel exists/tunes on sm_89 (triton block-FP8 W8A8; or dequant-to-bf16 + cuBLAS per layer is too costly) | microbench vLLM's triton w8a8_block_fp8_matmul and Marlin at M=437 on the 4 layer shapes |
| 5 | R3 | better decode GEMM at M=48 (Marlin 0.67 of BW overall) | ~5.7% | per-shape tuning/fusion lifts BW to ~85% | that the loss is in the kernel (not in launch tails / small shapes) | per-shape ncu/nsys table at M=48 (from the baseline profile: free) |
| 6 | R6 | draft lm_head FP8 (flag) / reduced draft vocab (generic frequency source) | 2.0% / 3.6% | acceptance stays ~2.34 | FP8 head: MRv2 path works with MTP here; reduced vocab: acceptance loss small | flag A/B via unofficial_eval (~5 GPU-min) reading acceptance length |
| 7 | R1 | local argmax instead of logits all-gather (draft: flag; target: code for all-greedy batches) | 2.6% (draft part 0.9%) | all rows greedy and no logprobs | flag works with MTP + TP=2 | same flag A/B as R6 (bundle) |
| 8 | S1 | 3 MTP drafts | ~0% now, ~5% after R1/R6/X1 | p3 >= ~0.38 and draft step cheap | p3 (3rd-position acceptance) | k=3 bench of 100 prompts, read per-position acceptance (~3 GPU-min) |
| 9 | R2b | FP8 W8A8 prefill GEMMs with fp16 in-block accumulation (330 TF) | further ~10% on top of R2a | no fp16 overflow inside a 128-K block; GSM8K A/B within 2 points | e4m3 x e4m3 products up to 2e5 > fp16 max | numeric microbench on real activations: count overflow/inf in fp16 block sums |
| 10 | R5/W4 | torch.compile on / async scheduling | ~1-2% each | supported with MTP + hybrid model here | why compile mode is NONE (maybe unsupported) | server start + 100-prompt bench with the flag |
| 11 | K2 | ReplaySSM GDN state (fewer state write-backs) | ~1.8% | - | big kernel + state plumbing effort | defer |
| 12 | R4 | AR + residual + RMSnorm fusion inside a custom AR | ~0.7% | X1 built | - | add to X1 |

Plan with 80 GPU-min: (1) X1 microbench now (t1-w1), then build the winner as a record; (2) flag bundle R6a+R1-draft as one record
through unofficial_eval; (3) S1 acceptance probe; (4) W1/L2 microbench only after X1 is measured (they compete for the same slack).
