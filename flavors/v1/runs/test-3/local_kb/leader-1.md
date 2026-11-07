# LOCAL_KB leader-1 (T1)
## Round 1 (R1-A prefill graphs, R1-B host-staged AR). Budget 80: w1 20, w2 35, evals ~15, slack ~10.
- ws leader-1.prefillgraph (w1): flag-only test, bench400 + gsm8k50 + 128K request.
- ws leader-1.hostar (w2): microbench -> integrate -> server bench.
- T1-001 (R1-A): prefill graph cap 2048 -> +2.2% unofficial (noise). Prefill steps GPU-bound, not launch-bound. 128K passes. ~6 GPU-min used.
- R1-B microbench v1 (w2): SM kernel via /dev/shm mapped pinned host, bitwise == NCCL. M=1 10us, 16 26.5, 48 52 (NCCL 63-70), 128 ~105 (133), 450 322 (345), 2048 1367 (1487). Write phase ~24us + read ~25us at M=48, serial; ~25 GB/s each. Misses 40us gate; told w2 to try fence/CE/chunk overlap, then integrate anyway if <=52 (est +4-5%).
- R1-B integrated (w2, ws leader-1.hostar): bench400 665.0 tok/s (+5.3% unoff), TTFT p99 1382, TPOT p99 39.9, gsm8k50 88%. Microbench M=48 48.6 vs NCCL 67.
  Review found a hang hazard: reader waits flag != epoch -> peer can overwrite with e+1 first. Asked for a >= fix + race test before the official eval.
  Notes: kernel spins and needs all blocks resident (like NCCL). JIT via torch cpp_extension (cache VLLM_CACHE_ROOT/torch_extensions).
- T1-002 OFFICIAL: 662.8 tok/s +5.0%, TTFT p99 1357, TPOT p99 49.0, GSM8K 94.0%. gpu used ~20/80.
- T1-003 nsys of T1-002: decode AR 49.6 vs 60.8 us (gain all in decode, ~1.5 ms/step). Prefill AR ~620 us, dominated by rank skew (~250-316 us), no gain. AG 1.14 ms/step unchanged. M=48 overlap: link full-duplex only partly (41 us concurrent vs 27 ideal), chunk fence ~5 us -> chain near floor.
- T1-004 (AG): ~0. T1-005 (T1-004 + prefill graph flag): 699.0 unoff (+10.7%); flag removes skew (316->214 us prefill, 30->10 us decode). Official eval launched.
- T1-005 OFFICIAL 699.2 (+10.7%), TTFT p99 1309, TPOT p99 50.6, GSM8K 93.2%. 128K re-test running.
- T1-005 128K re-test passed (graph mem 1.33 GiB vs 2.07 with NCCL). Round done ~46.6/80 GPU-min.
## Round 2 (R2-A logits diet on T1-005). Budget 80: w3 flags+3drafts 35, w4 greedy verify 25, evals ~15.
- KB8: GPU0 clock locked 2235 MHz -> prefill skew; don't chase.
- Note: a 2nd --speculative-config must carry full JSON (method, num tokens).
- T1-006 local_argmax +1.1%; T1-007 +fp8 draft head 718.0 vs 696.9 same-job (+3.0%), accept 2.343, 128K pass, +0.59 GiB/rank.
- T1-008 k=3: -6.9%, TTFT p50 210->792 ms (cause unknown), accept 2.715. Dropped. w3 stopped. T1-007 rerun 719.7.
- RULING (human, r2): correctness = math + numeric op checks on same inputs + GSM8K/acceptance; text equality only for bitwise-exact changes at c=1. Relayed to w4; w4 base now T1-007.
- T1-009 = T1-007 + greedy verify fast path: 732.6 vs 715.3 same-job (+2.4%); 25511 real rows 0 mismatches (467 ties). Official eval launched. w3,w4 stopped. Budget 54.8/80 before eval.
- T1-009 OFFICIAL 730.9 (+4.5% vs T1-005). Round 2 done, 64.9/80.
## Round 3 (on T1-009 730.9). R3-A k=3 + bigger KV (w5, 25 min); R3-B GDN replay (w6, 35 min); evals ~15.
- Lesson: I missed "Running: 14, Waiting: 2" in T1-008's log -> always check scheduler Running/Waiting for capacity-changing configs.
- R3-B ceiling (w6): stock GDN 122.8 us/layer alone, 267 with 2 GEMMs; last-store-only 61.8/213.7; replay R=2 65.6/217.7 -> ~45-50 us/layer, ~2.1-2.4 ms/step. GO. Kernel = csrc gdn_decode_post_conv_mtp_kernel (needs ws build). Gave design options (a) next-step fused vs (b) end-of-step replay kernel.
- T1-010: k=3 + KV 5.5-6 GiB OOMs in graph capture; KV ceiling 5.0-5.5 GiB with 2048 prefill graphs. Trying cap 1024 @5.6 GiB (then 512).
- T1-011: cap1024 + KV 5.6 GiB starts but runtime OOM. R3-A stopped; k=3 needs R3-B v2 (smaller spec pages). w5 stopped (7.2 GPU-min).
- R3-B v1 (w6): design (a) fused next-step replay + flush kernels before align copies/non-fused paths. Unit bitwise 0 mismatches (negative control 106). 200-prompt same-job A/B 712.4 vs 697.3 (+2.2%), accept 2.359. Asked: Q1 unconditional per-layer flush?, Q2 flags capacity decided once, flush counters, v2 feasibility (fewer spec blocks for k=3).
- T1-012 filed (replay v1, Q2 fix + counters). Official eval launched. v2 = 1-2 days, not this round.
- T1-012 OFFICIAL 749.7 (+2.6% vs T1-009; below 3% bar but consistent with +2.2% same-job). Round 3 used 35.9/80. w6 stopped.
