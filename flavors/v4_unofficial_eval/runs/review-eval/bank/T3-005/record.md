team:         T3
builds_on:    T3-001 (+ T3-003's flags in the bench)
performance:  "screen negative (no harness): quick bench job 64, 1 start each, fp8 draft head + KV 4.4 in both arms: head prefetch on 825.6 vs off 824.4 (+0.15%, below noise)"
summary:
  hypothesis: extend T3-001's L2 prefetch to the hmar all-reduce right before the drafter's lm_head (MTP final norm); worker microbench
              with the bf16 head at M=16 showed AR+head 1510 -> 1376 us with a 16 MiB prefetch.
  change:     by worker leader-3.w1 (workspace t3-w4): models/qwen3_5/nvidia/mtp.py sets the prefetch target to the drafter's current
              lm_head weight before fused_allreduce_gemma_rms_norm; env VLLM_T3_HEAD_PREFETCH. change.diff = agents/leader-3.w1/t3-w4-headpf.diff.
  results:    Observed: with the fp8 copy (the champion's head since T3-003) the microbench gets worse: AR+head 794.6 us without vs
              883-930 us with 8-64 MiB prefetch (agents/leader-3.w1/hmar_pf_fp8head.log). Quick bench +0.15% (job 64). c=1 text identical.
  conclusion: Interpretation: the fp8 head GEMM at M=16 is purely bandwidth-bound and its column-major copy is not read in storage order,
              so the prefetch competes instead of helping. Dropped. (Target-head prefetch after the last layer: ~15 us, not built.)
validity:     reads only (as T3-001). Not a candidate. Reviewed by leader-3 (diff read); not sent to the reviewer (screen negative,
              no harness, no further work planned).
