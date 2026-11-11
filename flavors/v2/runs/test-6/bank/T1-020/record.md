team:         T1
builds_on:    T1-019
performance:  "official #1: 850.3 tok/s (850.7/849.9; +1.6% vs T1-019 837.0 cross-job, same-job pooled +0.8%), +33.7% vs BASE, GSM8K 93.6%, TTFT p99 1373, TPOT p99 38.1, accept 2.336, all gates + 128K pass"
summary:
  hypothesis: the down AR's 42 MB in_proj prefetch ends inside the AR, leaving slack: also prefetch the layer's out_proj/o_proj
              (15.7 MB, evict_last survives the GDN state traffic), and give the final-norm ARs (target + drafter) a prefetch of
              their lm_head's first 88 MB.
  change:     second prefetch range in host_staged_ar (pf2) plumbed through all_reduce_prefetch(tensor, prefetch, prefetch2?) custom
              op, cuda_communicator, fused_allreduce_gemma_rms_norm; decoder layer passes its own out_proj weight as prefetch2 to
              input_layernorm; final norms (model.py, mtp.py) prefetch lm_head weight via a non-submodule owner ref. JIT ext build
              dir named by source hash. Worker leader-2.w9, ws leader-2.pf2.
  results:    Observed: microbench (job 67): out_proj 29.4 -> 22.6 us (25 MB GDN writes) / 32.5 -> 23.8 (75 MB), layer -4.5/-6.2 us,
              AR2 +0.7 us. M=16 gate_up cap not worth it. ABBA job 68 (200 prompts c=16): tok/s A 797.2, B 808.1, B 797.6,
              A 799.1; TPOT p50 A 17.71/17.52, B 17.36/17.29; acceptance same. 128K pass. Estimate ~0.4 ms/step (1.4% decode).
  conclusion: Interpretation: small real gain likely (~0.5-1%); B shows per-server-start variance (2 of 4 at A level), unexplained. 2nd ABBA (job 69, B A A B): 811.2, 798.0, 796.3, 798.7; TPOT 17.41/17.54/17.67/17.86.
validity:     leader-2 review: cache hints only (exact); same safety as T1-019 (prefetch blocks never wait/touch flags); lm_head owner
              ref is a plain list, no parameter registration change; prefill/NCCL untouched. OK.

official #1 (2026-11-08T04:30): 850.3 tok/s at c=16 (mean of 2 trials: 850.7 / 849.9, CV 0.1%) (+33.7% vs BASE), GSM8K 93.6%, MTP acceptance 2.336, TTFT p50 189 ms, TPOT p50 17.0 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
