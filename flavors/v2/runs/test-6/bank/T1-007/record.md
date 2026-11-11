team:         T1
performance:  "unofficial: 718.0 vs same-job T1-005 696.9 tok/s (+3.0%)"
summary:
  hypothesis: + FP8 draft lm_head halves the 2 x 1.37 ms draft heads without hurting acceptance.
  change:     T1-006 + draft_lm_head_quantization="fp8" (drafter's copy of lm_head only; ruling 1). serve_args.txt.
              Worker leader-1.w3, ws leader-1.logitsflags (out/C2).
  results:    Observed (job 22, same job as C0/C1): 718.0 tok/s, TTFT p50/p99 229/1272, TPOT p50/p99 20.4/50.6, acceptance 2.343
              (C0 2.340), gsm8k50 88%. Memory: model loading 14.99 vs 14.40 GiB (+0.59 GiB/rank), graph capture 1.33 GiB, KV fixed
              5 GiB. 128K test PASS (128000-token prompt HTTP 200 in 65 s, follow-up request + /health OK). Napkin ~+5%.
  conclusion: Interpretation: no acceptance loss from the fp8 draft head; +3.0% at the noise edge (single run). Candidate to stack
              with the target greedy fast path before an official eval.
validity:     leader-1: ruling 1 allows FP8 draft lm_head in the drafter; target verification at baseline bf16 head -> output
              identical. Memory change (+0.59 GiB weights) -> 128K test passed on this exact config. OK.
