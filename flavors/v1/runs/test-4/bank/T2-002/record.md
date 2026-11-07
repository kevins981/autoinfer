team:         T2
performance:  "unofficial: -3.9% vs T2-001 (584.4 vs 608.0 tok/s); TTFT p99 2225 ms FAILS the 1609 gate"
summary:
  hypothesis: with a cheaper drafter, 3 draft tokens beat 2 (est. +4-8%).
  change:     T2-001 flags with num_speculative_tokens=3. Empty diff. Builds on T2-001.
  results:    Observed (same job as T2-001, 150 prompts): 584.4 tok/s, acceptance 2.757 (.779/.565/.413), TPOT med/p99 21.0/46.2,
              TTFT p99 2225 ms. No crash/OOM under bench load (128K not tested).
  conclusion: Interpretation: +16.5% tokens/step but step cost rises more (step 64 verify tokens + extra draft pass); TTFT p99
              blows up (more tokens per step → prefill chunks share steps with bigger decode?). Dropped. Position-3 acceptance 0.41
              (matches the 0.40 guess). Depth 3 is not worth it at this cost structure.
validity:     leader-2: flags only, exact speculation. Not pursued.
