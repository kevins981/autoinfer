team:         T22
builds_on:    T14-004
performance:  "unofficial: -0.9% (ABBA NP=300, every B below every A)"
summary:
  hypothesis: down_proj runs cold in situ (56.7 vs 45.0 warm, T22-001); prefetching its first 16 MB (evict_last) from the AR before gate_up recovers most of the gap (~0.65 ms/step).
  change:     env VLLM_HOSTAR_L2PF2_DOWN_MB (default 0 = champion): the decode-size host AR before gate_up also prefetches the first X MB of the layer's down_proj weight as pf2, with its own byte count, pf2 loop issued first (host_staged_ar.cu). Cache hints only. Mixed-step token-split path untouched. serve: champion args + env X=16 for B.
  results:    Observed: job 153 (ABBA, 300 prompts, c=16, fresh starts): A(X=0) 929.8, 931.4; B(X=16) 918.2, 926.4 tok/s; acceptance 2.327-2.342 in all. Smoke job 152 (X=16, 40 prompts) ok. Per-kernel times not logged.
  conclusion: Interpretation: no gain, likely slight harm: the down prefix competes with gate_up's AR-window prefetch for L2 capacity / extends the AR kernel (KB78 coupling), cancelling the down gain. Down headroom is not reachable through the AR window.
validity:     exact (cache hints only; default off). Not for official eval.
