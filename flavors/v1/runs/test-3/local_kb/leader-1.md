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
