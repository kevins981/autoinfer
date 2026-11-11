team:         T5
builds_on:    T5-001 (same workspace t5-k3, built; no code change)
performance:  n/a (profile). k=3 forward delta +3.89 ms = Marlin +3.00, hostar +0.61, rest +0.28.
summary:
  hypothesis: the k=3 forward delta not explained by the cold Marlin microbench (T5-002) is GDN/attention at 4 tokens/request;
              the drafter step can be made much cheaper.
  change:     none. nsys node traces, GPU 0: job 30 (k=2 champion flags, 40 prompts, 78 decode steps at 16 req/48 tok; 3.8 GPU-min,
              k=3 part timed out) and job 32 (k=3 cap1024 KV 5.75 GiB, 28 prompts, 31 steps at 16 req/64 tok; 2.1 GPU-min).
              Worker leader-5.w1; agents/leader-5.w1/prof/{k2,k3}/ (seg.txt per-family tables), scripts prof/seg.py, seq.py.
  results:    Observed (median us per decode step, target forward incl. lm_head):
              | family          | k=2 M=48 | k=3 M=64 | delta |
              | Marlin          | 13279    | 16282    | +3003 |
              | hostar AR       | 5973     | 6587     | +614  |
              | GDN replay/flush| 1845     | 2002     | +157  |
              | lm_head FP8     | 1020     | 1045     | +25   |
              | norms/attn/bf16/elementwise/act | 1689 | 1755 | +66 |
              | total (wall)    | 23520    | 27413    | +3893 |
              Drafter per step (wall): k=2 882/897; k=3 940/788/897 => extra draft step +0.85 ms. Draft step content: draft lm_head
              ~310 us (252 MB FP8/rank, ~roof), MTP Marlin 200-263, fc ~80, AR 110-183; wall - kernel-sum 20-50 us (no launch gaps).
              Sample segment 120 us; the timers' "sample" 1.12 ms = compute_logits (lm_head 1.02) + argmax/reject/housekeeping (code read).
              nsys inflation vs CUDA-event timers: ~1% (k=2 25.42 vs 25.15; k=3 30.16 vs 29.99).
  conclusion: Interpretation: Marlin is the whole k=3 cliff, larger in situ (+3.0 ms) than cold (+2.1): in situ M=48 Marlin (13.28) runs
              below its cold sum (15.56) thanks to L2 prefetch, and the 4-m-block kernels seem to keep less of that benefit (untested).
              GDN/attention/norms at 4 tokens are ~0.2 ms: not a target. The drafter is at its byte roof; only a smaller draft vocab
              (~-0.2 ms/step, costs acceptance) would cut it. With M=64 Marlin at the M=48 cost, k=3 step would be ~27.5 ms (+8%) for
              E +16% => ~+7.7% decode.
validity:     profile only.
