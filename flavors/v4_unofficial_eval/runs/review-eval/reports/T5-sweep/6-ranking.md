# T5 sweep, step 6: ranking (vs champion T3-007, 853.6 tok/s; 106.6 s per trial)

Split with T4 (leader-4, agreed 2026-11-16 ~05:30): T5 takes GDN DeltaLog; T4 keeps AR+norm+quant fusion and k=3.

| # | Candidate | Napkin gain | Wins when | Riskiest assumption | Cheapest test that could show it wrong |
|---|---|---|---|---|---|
| 1 | **GDN DeltaLog, row-0 checkpoint layout**: slot 0 <- S after verify row 0 (always accepted); slot 1 <- log of rows 1..2 inputs (k after l2norm, v, decay, beta; ~1 KB/row/head); next step reads slot 0 and replays (accepted-1) logged rows in-kernel before the new rows. Decode-only steps (fused CUDA kernel) first; mixed steps route spec rows through the same kernel or a matching Triton change; align-mode temporal copies materialize the exact state (replay) | GDN 3.69 ms/step -> ~1.8-2.4 (1 write + ~1.33 replayed rows of FMA): -1.3 to -1.9 ms x 2,368 decode steps = 3.1-4.5 s: **+3-4%** | the in-server kernel time is driven by state writes reaching DRAM | that the 3 writes cost ~2/3 of the kernel in the server (L2 write-back could absorb them: in-server 72 us vs 120 us cold) | ncu of the live server, --cache-control none: DRAM write bytes of gdn_decode_post_conv_mtp (worker task 3, ~3 GPU-min); then a microbench of 1 vs 3 writes in a warm loop |
| 2 | **Mixed-step GEMM->AR row-split overlap** (out_proj/o_proj -> AR and down -> AR split in 2 row chunks, AR of chunk 1 on a side stream overlapping GEMM of chunk 2; only when rows > ~2x ridge) | hides up to min(GEMM, AR) per site: <= 13 ms of ~120 ms per mixed step; realistic ~5 ms x 371 = 1.9 s: **+1.5-2%** (<= +4.5%) | NCCL AR and the Triton GEMM co-run without slowing each other (SM contention: NCCL uses few SMs; the GEMM grid at M~150 is large) | co-running slowdown and graph-capture of the 2-stream pattern in piecewise graphs | microbench on 2 GPUs: GEMM(M=150, down shape) concurrent with NCCL AR of 1.3 MB vs serial |
| 3 | k=3 drafts (T4's) after #1 frees capacity | +2-3% (T4) | p3 >= ~0.3, memory | capacity: needs the log outside the slots or a 4th slot | T4's |
| 4 | Fused GDN mixed glue (R2) | ~1% (after T4-001's correction) | small mixed steps | host stall is only 5-14 ms | — (dropped for the time left) |

Dropped (with reasons in 4-knobs/5-compare): PP, asymmetric splits, drafter TP=1, decode micro-batching, tree drafts,
reduced-vocab draft head (T1-002), larger L2 prefetch (T3), hmar > 96 rows (T1), recurrent prefill (T4-001), prefill
admission windows (policy, validity).

Plan: #1 gated on the ncu result (job pending); if the writes don't reach DRAM, switch to #2's microbench.

## Review (reviews/T5-1) and my answers
1. Row-0 checkpoint: sound for continuation ("an irrevocably committed checkpoint"): ADOPTED wording.
2. State-access surface is larger (CUDA verify, mixed Triton spec kernel, Triton fallback, non-spec decode/prefill, prefill checkpoint
   export, CPU-planned and GPU temporal copies in align mode incl. boundary-specific accept_token_bias, count lifecycle): ADOPTED as
   the design checklist; leaving slot 2 unwritten frees no capacity (allocator reserves num_speculative_blocks): ADOPTED (k=3 capacity
   claim withdrawn).
3. Lifecycle cases (first decode after prefill, align pre-copy/postprocess, preemption, padding/NULL slots, mixed steps): ADOPTED.
4. Bitwise replay possible if fp32 normalized k, decay and the computed delta are logged and the FMA order is kept: ADOPTED (log delta).
5. #2: pipeline cost is G1 + max(AR1, G2) + AR2 vs G + AR; only ~half the mixed steps have >= 192 rows: ADOPTED -> #2 napkin
   ~+0.5-1%, microbenchmark-first.
6. Retract 3-calibration C/D host floor (nsys-inflated; 39.8 s mixed time is a residual): ADOPTED (see update file when written).
   Recurrent short prefill "parked, not refuted"; reduced-vocab FP8 head not refuted by T1-002; PP "needs a schedule argument":
   noted, all stay below #1 for the time left.
New fact from the review: vLLM already ships a delta-rule DeltaLog for Kimi KDA ("RecoverSSM": verify writes corrections/kg logs
outside the KV cache, a commit kernel advances the state by the accepted rows; vllm/models/kimi_k3/nvidia/ops/recoverssm.py) and a
Mamba2 ReplaySSM (--use-replayssm); neither supports Qwen GDN, and RecoverSSM with prefix caching needs the V2 model runner.
