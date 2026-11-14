# leader-58 (T58 sweep) notes
- Sweep done: reports/T58-sweep.md (+1-6). GPU used ~10 min by end of sweep (jobs 485, 490, 491, 494).
- Calibration facts (mine): decode step 24.6 ms (GEMM 12.5, hostar 7.3 = 130 x 56 us, GDN 2.0, drafter 1.8, lm_head 1.0); small mixed step GPU span 43 ms (P~60) -> +18 ms: bs_gemm M=128 instead of t31 (+8.7), AR rows (+5), GDN glue (+2.7). GPU 98% busy. GPU0 == GPU1 speed now.
- Host-side step durations (T37_STEPLOG) are wrong per step (async run-ahead); use GPU spans.
- My microbench "cold" used a write flush -> inflated (KB76). Use read flush + cudaCtxResetPersistingL2Cache.
- M1 = VLLM_HOSTAR_L2PF_MSG_MAX 1 MiB excludes 97-255-row ARs; testing 2.7 MB (task 6, w1).
- T56: mid-M t31 slices lose; bs_gemm M=112-128 291 us/GDN layer cold vs ~200 floor; new kernel ~+1-2%. Wave threshold <=0.5% (host-bound). T56 holds out_proj replicate + FP8 AG (ruling pending).
- Sparring T58-1 (ended): my "no lever > 1-2% left" is UNSUPPORTED as a general claim. Drafter headroom is unresolved (k=3 ceiling 4.0 vs 2.68 tokens, +49%); only "fine-tuning the stock head with affordable data" is bounded (KB60). Selective replication: supported as unattractive (attention mixer ~break-even; GDN/MLP lose). 2 combines/layer: minimal only for 1-D TP.
- Cheap probe identified (no new generation): leader-43.w1/dd/full has raw residuals at L16/32/48 (rank 0) + ids + drafter tokens/conf for ~32K verify rows (96 self-gen prompts); leader-3.w1 has an offline MTP pipeline (mtp_model.py) matching in-server acceptance + hidden_raw. Probe: frozen MTP head + two equal adapters (with/without h_32/h_48), split by sequence, agreement with target greedy. A positive result is only a premise; inference cost is separate.
- T58-001 = CHAMPION (1229.6 official; +0.72% same-job ABBAAB job 505 on T57-004). M1 prefetch gate 2.7 MB.
- Next: M2 microbench (task 8, w1): t31b at M 112-192 via grid z vs bs_gemm. Manager OK'd. Drafter-features idea sent to the human by the manager.

## Close-out (sparring T58-1 turns 7-8)
- W7' (wave threshold): the ≤0.5% cap is withdrawn. Conditional +0.9-1.4%. First test: two waves vs joined GPU time at 384-767 rows (T30_WAVE_BENCH). Then the capture memory against 0.45 GiB. Handed on in 6-ranking Addendum 2.
- Recommended closing T58: a budget decision, not exhaustion.
