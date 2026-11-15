team:         T1
builds_on:    T2-006
performance:  "unofficial: +1.14% at c=16 (ABBA 200 prompts, all B > all A: B 783.9/793.9 vs A 780.3/779.6), worker job 55"
summary:
  hypothesis: In mixed (prefill) steps the 133 all-reduces at ~300+ rows go to NCCL (~290 us each, ~38 ms per ~105 ms mixed step).
              Worker microbench (job 50) showed NCCL ARs DO run concurrently with the champion's W8A8 Triton GEMMs (87-253 us
              hidden per pair), unlike Marlin + HOSTMEM (T1-007). Splitting only the row-independent sublayers (GEMMs, norms,
              act, ARs) into 2 row halves, with each half's AR on a side stream while the other half computes, hides ARs
              without any attention/GDN metadata change. Napkin: ~25 ms hidden - ~8 ms extra weight reads per mixed step -> +3-4%.
  change:     models/qwen3_5/nvidia/rowsplit.py (side stream, async_all_reduce with events + record_stream); Qwen3_5Model.forward
              takes _forward_rowsplit when N >= VLLM_ROWSPLIT_MIN_TOKENS (256) on dense TP without PP/SP; per layer: in_proj
              per half, token-mixer core on all rows (precomputed proj/qkv kwargs added to GDN forward_cuda and
              Qwen3NextAttention.forward), out_proj per half + side AR, per-half add+norm+MLP + side AR, next layer's h1 in_proj
              overlapping AR(h2). VLLM_ROWSPLIT=0 disables. Decode steps and MTP layers unchanged.
  results:    Observed: A 780.3/779.6, B 783.9/793.9 tok/s; acceptance unchanged (B 2.340/2.344, A 2.346/2.340); TTFT p99 B 951/892
              vs A 969/959 ms; TPOT p99 B 39.4/39.1 vs A 32.6/36.7 ms (gate 55.3). Log: "Row-split TP all-reduce pipelining
              active for steps with >= 256 tokens". gsm8k 50 on B: 90.0%.
  conclusion: Interpretation: real but ~1/3 of napkin: the dependency chain leaves less to overlap than independent kernels,
              and the 2x weight reads cost. Threshold / split tuning untested.
validity:     Exact up to fp reordering (GEMM tile choice by M); AR sums unchanged (bf16 a+b); HOSTMEM serialization preserved:
              all split-region ARs on one side stream in a fixed order, all joined before later main-stream ARs (worker code read).
              Generic: keyed to token count only.
