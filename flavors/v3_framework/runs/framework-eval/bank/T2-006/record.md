team:         T2 (combination, credit T2 + T1)
builds_on:    T2-004 + T1-009
performance:  "official #1: 819.9 tok/s (+4.3% vs champion T1-009 786.4, +28.9% vs BASE), all gates pass"
summary:
  hypothesis: T2-004 (W8A8 Triton GEMMs, +4.73% unofficial on T2-002, mostly in graphed mixed steps) and T1-009
              (--attention-backend TRITON_ATTN on T2-002: +3.8% official, removes decode-step host syncs) act on different
              parts (GEMM kernels vs attention metadata/syncs), so they should roughly add (~+4% over T1-009).
  change:     T2-004's code (T1-004 HOSTMEM AR + W8A8 Triton default on sm89 + tuned configs) + T1-009's serve flags.
  results:    Observed: official #1 819.9 (821.6/818.3, CV 0.3%), GSM8K 93.2%, acceptance 2.327, TTFT p50/p99 168/970 ms,
              TPOT p50/p99 18.0/31.3 ms, 128K pass.
  conclusion: Interpretation: W8A8 and Triton attention add (+4.1% and +3.8% separately on T2-002 -> +8.2% combined).
validity:     Union of T2-004's review (checkpoint W8A8 scheme, GSM8K-250 A/B 93.6 vs 92.4 within 2.0, job 32) and T1-009's
              (attention kernel change, exact up to fp reordering, fp8 KV). The W8A8 GSM8K A/B was run with FlashInfer
              attention; the attention backend does not touch the GEMMs, and the official eval re-runs GSM8K-250 and 128K.

official #1 (2026-11-14T18:59): 819.9 tok/s at c=16 (mean of 2 trials: 821.6 / 818.3, CV 0.3%) (+28.9% vs BASE), GSM8K 93.2%, MTP acceptance 2.327, TTFT p50 168 ms, TPOT p50 18.0 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
