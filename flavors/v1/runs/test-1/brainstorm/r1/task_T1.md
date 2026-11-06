# Round 1 task for leader T1
Direction: **Attack the idle-DRAM window and cost of the FFN down_proj all-reduce.** (bottleneck: 64 FFN all-reduces ~88us each ≈ 5.6 ms of a ~39.6 ms pass, DRAM idle during each; gate_up Marlin reads 91 MB at ~80% DRAM peak)
Steps, cheapest first:
 1. NCCL tuning (env vars via serve_args-style env if possible: NCCL_PROTO/ALGO/BUFFSIZE/NTHREADS) and the FLASHINFER_PCIE_IPC / other vLLM all-reduce backends; microbench 0.5 MB 2-GPU AR latency first. Must stay full bf16/fp32 reduction (no low-precision comm).
 2. L2 weight prefetch: during the all-reduce, a side-stream kernel (CUDA-graph capturable) prefetches/loads the next layer's gate_up weights into the 72MB L2 (evict_last / prefetch.global.L2); Marlin then hits L2. Microbench Marlin gate_up (real shapes, M=48) with/without pre-touch and a concurrent 0.5MB NCCL AR before wiring into vLLM.
 3. If time: custom 2-rank AR fused with residual+RMSNorm; or N-split overlap of down_proj and its AR.
Ideas in brainstorm/r1/*.md (napkin: 2-3 ms/pass possible from each of AR speedup and prefetch).
Budget: 80 GPU-minutes this round (gpu/budget.json). Keep ~14 for official evals.
Champion/reference: BASE, 630.6 tok/s, TTFT p99 1355 ms, TPOT p99 41.2 ms, GSM8K 92.8%.
Rules: see task_definition.md (lossless, no low-precision all-reduce, FP8 weights/KV, etc.).
