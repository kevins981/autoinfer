team:         T1
builds_on:    T1-020
performance:  "unofficial: no gain. In-server trace: step 27.33 (T1-020) vs 27.43 / 27.43 ms (variants); NO-GO"
summary:
  hypothesis: the out-proj AR kernel ends ~12 us after its AR work (40.2 vs 52.2 us) because the 89 MB gate_up prefetch blocks
              outlast it (~0.7 ms/step); stopping prefetch when the AR is done, or issuing denser/wider prefetches, recovers it.
  change:     VLLM_HOSTAR_L2PF_MODE (default 0 = T1-020 behaviour): bit0 "stop" (AR blocks count done in state[2], prefetch blocks
              poll and stop); instruction variants (prefetch / ld.L2::256B evict_last / cp.async L2::256B); stride env. AR outputs
              bitwise identical. Worker leader-2.w11, ws leader-2.pfissue, agents/leader-2.w11/ (NOTES.md, mb/, integ/t1).
  results:    Observed: microbench (M=48, 3-layer emulation): current 372-375 us/layer, stop 16x64 357, stop 16x128 355; 128 B
              stride keeps the in_proj gain (one prefetch = one 128 B line), 256 B loses some; side-stream prefetch during gate_up
              worse (gate_up 119-145 vs 95). Microbench exaggerates the stretch (AR1 ~70 vs 52 in-server).
              In-server trace job 75 (3 fresh servers, 100 prompts c=16, rank 0): step median A 27.33, B (stop 16x128) 27.43,
              C (stop 16x64) 27.43 ms; out-proj AR end 52.2 / 51.2 / 47.0 us; down AR 41.2 / 44.8 / 45.0; AR write phase 21.0 ->
              23.4-23.8 us; tok/s 783.3 / 776.8 / 777.2. No bimodality across the 3 starts.
  conclusion: Interpretation: the tail is queued prefetch traffic that gate_up then uses; cutting it or densifying it only moves
              cost (and heavier prefetch slows the AR write). The ~0.7 ms "stretch" is not recoverable this way. NO-GO.
validity:     default-off code, exact (cache hints only). Not for official eval.
