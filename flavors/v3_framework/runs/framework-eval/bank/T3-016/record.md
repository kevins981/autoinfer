team:         T3
builds_on:    T3-009
performance:  unofficial: mixed, -1.45% (ABBA job 143; wide spread in both arms) -> inconclusive, leaning negative
summary:
  hypothesis: T3-014's microbench showed row-splitting loses at 256 rows (+41%/layer) and wins at 512 (-6.5%); raising
              VLLM_ROWSPLIT_MIN_TOKENS from 256 to 448 should stop the loss in 256-447-row mixed steps.
  change:     env only: B = VLLM_ROWSPLIT_MIN_TOKENS=448 (same workspace t3-rs11 both arms).
  results:    Observed: A 887.8 / 910.7 vs B 896.4 / 876.1 tok/s; verdict mixed -1.45%; acceptance 2.331-2.355;
              TTFT p99 856-966 ms both arms.
  conclusion: Interpretation: no sign of the gain the microbench suggested; the real layer has more compute per AR than
              the microbench's MLP-only layer, so the server's crossover is lower than the bench's. Not repeated (run
              paused; budget kept for k=3).
validity:     env-only, exact.
