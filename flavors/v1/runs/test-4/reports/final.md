# Final report (run test-4, task qwen38-27b-fp8-2x4090)

## Champion: T1-003, 691.3 tok/s official at c=16 (+8.7% vs BASE 635.9)
Trials 696.1 / 686.4. TTFT p50/p99 226/1303 ms, TPOT p50/p99 21.3/46.1 ms, GSM8K 92.8%, MTP acceptance 2.346, 128K request pass.
- Code (bank/T1-003/change.diff): a host-staged 2-rank all-reduce through a /dev/shm segment registered as mapped pinned memory
  on both GPUs (works without P2P), used for bf16 tensors <= 2 MiB with NCCL above that. Bitwise equal to NCCL. Kill switch
  VLLM_USE_HOST_AR=0. JIT-built via torch cpp_extension.
- Flags (bank/T1-003/serve_args.txt): MTP spec-config with use_local_argmax_reduction + draft_lm_head_quantization=fp8 (drafter
  only), plus --enable-batch-sharded-sampling.
- Path: BASE -> T1-001 (host AR, +5.7%) + T2-003 (flags, +2.8%) -> T1-003 (+8.7%). The two stack nearly additively.

## What was learned (shared_kb.md, 15 entries)
- No P2P: NCCL RING_LL pays ~35 us fixed + half link rate on small ARs; a one-shot host-staged AR cuts T=48 from 68 to 45 us,
  within ~10 us of the PCIe floor (the link is not full-duplex for SM zero-copy). NCCL env knobs: no gain at any size.
- Drafter-side cuts (FP8 head, local argmax) keep acceptance; target-logits exchange is no longer a lever (sharded sampling and
  a custom exact greedy argmax tie).
- Draft depth 3 raises acceptance (2.35 -> 2.76) but is flat or negative end-to-end and fails TTFT p99.
- Prefill-step AR overlap works mechanically (needs Marlin to reserve SMs) but is unmeasurable at c=16.
- Measurement: one-server-per-arm A/Bs drift ~2%; use ABAB. Node-level nsys inflates graph-launch gaps.

## Tree: tree.md. Untried ideas: ideas.md.

## Left open
- One-graph capture of the spec-decode glue (drafter passes + sampling + attention-metadata builds): ~1.6 ms/step of real CPU
  idle, est. 3-4% (KB12). The largest measured remaining lever on decode.
- Marlin (~51% of span): in-kernel efficiency near its floor at M=48 per measurements (ideas.md R1-C). Untried.
- Fusing residual-add + RMSNorm into the host AR kernel (~1.6 ms/step of norms etc.). Untried.
- Cause of the depth-3 TTFT p99 jump.
- T2-006 (exact greedy fast path) ties the champion: an equivalent alternative if batch-sharded sampling is ever unwanted.
- Hardening: host AR assumes both ranks construct it (if one fails, the backends disagree); T1-004 code is hack-grade.

## Budget
Round 1: T1 50.8, T2 29.6 GPU-min (+19.8 brainstormer profiling). Round 2: T1 26.9, T2 37.4. 
Official evals: T2-003, T1-001, T1-003, T2-006.
