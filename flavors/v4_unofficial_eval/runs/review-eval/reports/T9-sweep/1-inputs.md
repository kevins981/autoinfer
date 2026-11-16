# T9 sweep, 1: inputs (leader-9; compressed: the team opened at 07:08 UTC, the run ends 08:15)
- Model: Qwen3.8-27B-FP8: 64 layers = 48 GDN (linear attention, fp32 state 48 v-heads x128x128 per req/layer, TP-halved) + 16 full attention (every 4th), hidden 5120, MLP 17408, vocab 248,320 (untied bf16 lm_head, ~1.27 GB/GPU), 1 MTP layer as drafter (task/SPEED_OF_LIGHT.md inputs).
- Hardware: 2x RTX 4090, 1 TB/s DRAM, ~165 TFLOP/s FP8 (fp32 acc), no P2P (PCIe via host).
- Workload/objective: ShareGPT 400 prompts, ~260 in / ~228 out, c=16 closed loop, output tok/s; gates TTFT/TPOT p99 <= 1.2x BASE, GSM8K-250 >= 89.6%, 128K request.
- Contract: task_definition.md (exact math up to fp reordering; W8A8 dyn-128 allowed; exact spec decode; no benchmark fitting; prefix caching on).
- Starting point = champion T5-001 (895.4 tok/s): TP=2; MTP k=2 drafts (fp8 drafter lm_head, local argmax); W8A8 Triton GEMMs with tuned tiles + fused act quant; host-mapped all-reduce (HMAR, <=96 rows) + L2 weight prefetch during AR; TRITON_ATTN with bf16 Q over FP8 KV; GDN DeltaLog (one state write per step, log rows replayed; kMaxLogRows=3, so supports up to 4 verify rows); prefill cudagraphs to 2048 tokens; KV 4.4 GiB; batch-sharded sampling.
- Measured: acceptance length 2.343, per-position acceptance 0.782 / 0.567; TPOT p50 16.2 ms; decode-step wall ~29.0 ms (job 118 node trace).
