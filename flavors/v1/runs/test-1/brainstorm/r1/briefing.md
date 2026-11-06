# Round 1 briefing
Task: serve Qwen3.8-27B-FP8 faster on 2x RTX 4090 (TP=2, no P2P). Read $AUTOINFER_RUN/task/TASK.md, task_definition.md (correctness/loopholes), focus.md, and task/baseline/PROFILE.md.
**Focus this run: FFN speedups only** (gate/up/down projections, hidden 5120, intermediate 17408, 64 layers, TP=2) and anything around them: Marlin FP8 GEMM kernel/layout, gate-up+activation fusion, bytes read, the down_proj all-reduce, comm/compute overlap.
Facts (BASE, 630.6 tok/s at c=16): Marlin FP8 GEMMs 50.8% of kernel time (FFN gate/up ~117us each x64/pass, down ~58us), DRAM-bound at ~80% peak, tensor pipe 25% busy, 128 CTAs, 255 regs/thread; NCCL all-reduce 25.7% (~88us each, ~0.5MB, 129/pass); MTP acceptance length 2.34 with 2 draft tokens. Nothing tried yet this run.
Constraints: lossless math, FP8 weights/KV only, no fp8/low-precision all-reduce, max context>=128K; max-num-seqs may change. Propose ideas with napkin-math upside and how to test cheaply.
