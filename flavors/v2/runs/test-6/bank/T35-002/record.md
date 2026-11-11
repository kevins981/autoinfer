team: T35
builds_on: T31-003
performance: "unofficial screen: MTP k=3 1041.3 tok/s vs k=2 1069.0 (cross-job 355 vs 353, 1 start each, 200 prompts): -2.6%"
summary:
  hypothesis: decode steps are mostly fixed cost (M=3 19.0 ms vs M=48 23.3 ms, job 352), so verifying more draft tokens per step should pay.
  change: serve flag only: num_speculative_tokens 3 (rest of the champion's speculative-config unchanged). k=4 failed at engine init (not investigated).
  results: Observed (job 355, agents/leader-35/k_sweep/k3): 1041.3 tok/s, acceptance 2.687 (per position cumulative 0.765/0.536/0.386), TPOT p50 11.7 ms, ITL p50 25.1 ms; 1459 steps vs 1527.
  conclusion: Interpretation: the 3rd draft adds more step time (drafter pass + verify of 16 more tokens) than its ~0.39 x 16 accepted tokens save. Cross-job single pair: screen negative, not a measured fact.
validity: flag only, exact speculative decoding. Not for official eval.
