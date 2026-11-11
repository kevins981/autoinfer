team:         T1
builds_on:    T1-016 (+ T1-017's test-only AR trace, default off)
performance:  "official #1: 837.0 tok/s (+3.9% vs T1-016 805.7, +31.6% vs BASE), GSM8K 93.6%, TTFT p99 1272, TPOT p99 39.1, accept 2.339, all gates + 128K pass"
summary:
  hypothesis: during each decode host AR (~42 us, PCIe-bound, 8 SMs) HBM is idle; extra blocks of the AR kernel can issue
              prefetch.global.L2::evict_last over the consumer GEMM's FP8 weights so the following Marlin GEMM reads part of them from
              L2 (T1-017 measured the AR is pure transport; T1-018 showed the AR kernel co-runs with Marlin).
  change:     host_staged_ar.cu: Params.G (AR blocks per role) + optional prefetch range; blocks >= 2G only issue L2 prefetches and join
              the done counter (last finisher advances the epoch). host_staged_ar.py: allreduce_pf for messages <= 1 MiB when given a
              weight; env VLLM_HOSTAR_L2PF (default 1; 0 opts out), _BLOCKS 8, _MAX_MB 88, _STRIDE 32 B, _MSG_MAX 1 MiB.
              New custom op vllm.all_reduce_prefetch + GroupCoordinator.all_reduce_prefetch (host-AR only; otherwise plain AR).
              fused_allreduce_gemma_rms_norm(prefetch=) and Qwen3_5RMSNorm.forward_fused pass the consumer linear's weight (the model
              already defers each row-parallel AR into the next norm): down AR -> next layer's in_proj/qkv; out-proj AR -> gate_up.
              Worker leader-2.w9, ws leader-2.arsplit.
  results:    Observed: microbench (agents/leader-2.w9/pf, jobs 60-61): per layer 405 -> 368 us (gate_up 135 -> 95.5, in_proj 56 -> 43.5,
              each AR +8). In-server (agents/leader-2.w9/integ, jobs 62-65): decode step median 29.90 -> 27.71 ms; target fwd
              26.11 -> 23.97 ms; AR kernel 41 -> ~49-52 us; B=4 prefetch outlasts AR (no gain), B=16 slightly worse than 8.
              ABBA job 65: A 757.6, B 793.1, B 794.1, A 760.6 (+4.5%); TPOT p50 18.7 -> 17.8 ms; acceptance 2.339 vs 2.330 (noise).
              128K request pass (job 62). Estimate (microbench) up to 2.3 ms/step, in-server measured 2.2 ms/step.
  conclusion: Interpretation: the AR's idle-HBM window is usable for warming L2; ~7% of the decode step saved.
validity:     leader-2 review of the raw diff: only cache hints (prefetch instructions), no data or math change -> bitwise identical
              outputs by construction. Prefetch blocks never touch flags or wait; grid <= 16 co-resident blocks; epoch advanced once by
              the last finisher, which re-reads state[0] (unchanged during the launch) -> monotonic wait logic unchanged, no new hang
              path. Prefill (> 1 MiB) and NCCL paths untouched. Pointers baked at graph capture (weights static). Defaults generic
              (tuned on decode step time, not benchmark-specific). Test-only trace code from T1-017 remains, default off. OK.

official #1 (2026-11-08T03:55): 837.0 tok/s at c=16 (mean of 2 trials: 839.1 / 834.8, CV 0.4%) (+31.6% vs BASE), GSM8K 93.6%, MTP acceptance 2.339, TTFT p50 190 ms, TPOT p50 17.2 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
