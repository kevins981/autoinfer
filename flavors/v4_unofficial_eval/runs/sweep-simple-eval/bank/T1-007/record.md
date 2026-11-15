team:         T1
builds_on:    T2-005 + T1-006
performance:  "not measured: identical code and serve args to T2-006 (promoted, 738.8 / 747.8); eval cancelled after 0.6 min to avoid duplicate GPU use"
summary:
  hypothesis: T1-006's target-side distributed argmax (removes the ~0.93 ms/decode-step logits all-gather and the full-logits sampling kernels) is orthogonal to T2-005's W8A8 GEMMs and should stack: ~+2-2.5%.
  change:     T2-005 (humming W8A8, reduced-vocab draft head, local draft argmax, prefill graphs to 1024, max-num-batched-tokens 1024) + the 3 files of T1-006 (logits_processor.py, model_runner.py, rejection_sampler.py), applied unchanged (git apply clean). Serve args = T2-005's.
  results:    pending
  conclusion: pending
validity:     same as T1-006 (exact for greedy: argmax-merge with lowest-index ties = torch.argmax over gathered logits; full-gather fallback for any non-greedy / logprobs / processed request) and T2-005 (reviewed by leader-2; W8A8 checkpoint scheme, GSM8K 93.2%, 128K pass). The two changes touch disjoint code (GEMM backend vs sampling path); no memory change from T1-006.

unofficial #1 (2026-11-15T17:04): FAILED at exit 130 (bank/T1-007/unofficial/1/)
