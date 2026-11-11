team:         T38
builds_on:    T37-002 (champion code, run with speculation OFF; + a one-line guard in gdn/replay.py: min(..., default=0),
              see replay_guard.diff; without it the champion crashes at PP=2: job 391)
performance:  measurement only (T38-3 stage 1). Job 392 (6.6 min): nsys-profiled serving, 64 ShareGPT prompts at c=16
              (19,755 prompt tokens, 13,942 output tokens), PP=2 arm then TP=2 arm, same code. Job 391 failed (2.0 min).
summary:
  hypothesis: in mixed serving, PP=2 (no all-reduce) cuts the marginal cost of prompt tokens to <= 0.70x TP's.
  change:     none to the served math. serve args = T37-002's + --speculative-config None [+ --tensor-parallel-size 1
              --pipeline-parallel-size 2]. Script agents/leader-38/s1/run_s1.sh; outputs agents/leader-38/s1/{pp,tp}/
              (steps.csv GPU0, steps_g1.csv GPU1, bench.log, breakdown.txt, profile.sqlite).
  results:    Observed (under nsys; throughput numbers are not official-grade):
              - bench: TP 31.87 s, 439.5 output tok/s, TPOT 24.8 ms, TTFT 843 ms; PP 48.95 s, 286.1 tok/s, TPOT 40.3, TTFT 321.
              - TP: decode step period 19.96 ms (14 seqs); 51 mixed steps; wall-time extra of mixed steps over the decode
                median = 4.05 s = 205 us per prompt token (period fit 10.2 ms + 231 us/token).
              - PP (GPU work excluding NCCL wait kernels): decode microbatch (median 5-6 seqs) GPU0 15.2 ms, GPU1 17.0 ms
                (GPU1 holds the full-vocab lm_head); prefill extra GPU work per GPU = 2.06 s (GPU0) / 2.08 s (GPU1) for
                19,755 tokens = ~105 us per prompt token per GPU; per-stage fit 98-99 us/token. Busy: GPU0 38.4 s, GPU1
                43.1 s of a ~49 s run (GPU1 88% busy; the rest is pipeline bubble/imbalance in stock vLLM PP).
              - Model check: T38-003 predicted a 32-layer stage's GEMMs at M=16 = 13.4 ms; measured stage compute
                15.2 / 17.0 ms -> non-GEMM ~1.8 ms (GPU0) and ~2.2 ms + lm_head 1.4 (GPU1): ~4 ms per microbatch over
                both stages vs 3 ms assumed.
  conclusion: Interpretation: PP's marginal prompt-token cost is ~105 us of GPU time per GPU, running concurrently on both,
              vs 205 us of wall per token in TP (the TP figure includes the copy-engine all-reduce, which shows as kernel
              idle): ~0.51x, PASSING the stage-1 rule (<= 0.70x) IF the pipeline stays full. The stock vLLM PP pipeline does
              not stay full: ~12% bubble at spec-off decode (GPU1 heavier by ~1.8 ms/microbatch, plus scheduling). Stage-time
              model holds within ~4% (non-GEMM +1 ms per microbatch over the assumption). Decode without trees is ~0.65x TP,
              as the byte arithmetic says. So the direction stands or falls on decode recovery (r >= ~0.92-0.94) and on a
              bubble-free pipeline.
validity:     measurement only; the guard only avoids min() on an empty sequence when no GDN replay layer registers.
