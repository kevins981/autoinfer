team: T35
builds_on: T31-003
performance: "unofficial screen: --max-num-batched-tokens 128 -19%, 192 -2.6% vs 2048 (same job 353, 1 start each, 200 prompts)"
summary:
  hypothesis: decode steps (M=48) leave tensor cores idle; dribbling prefill in small chunks into decode steps would ride on spare compute.
  change: serve flag only: champion + --max-num-batched-tokens {128,192} (reference 2048 = champion's effective budget).
  results: Observed (job 353, agents/leader-35/budget_sweep): 2048 -> 1069.0 tok/s (TTFT p99 800 ms); 192 -> 1041.1 (1330); 128 -> 870.5 (1624). Steps 1527/1566/1625. Prefill marginal cost from step counters: ~0.124 ms/token at 2048 vs ~0.26 ms/token at 128.
  conclusion: Interpretation: mixed steps carry a large fixed cost (GDN chunked-prefill path ~25 kernels/layer, ping-pong chunks re-read the weights, GEMM path change above M=96), so small chunks cost more per token. Prefill is ~16% of wall time, so the idea is capped at that anyway. Screen negative (1 start each), not a measured fact about a purpose-built mixed step.
validity: flag only, exact (scheduling). Not for official eval.
