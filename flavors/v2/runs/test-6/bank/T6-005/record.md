team:         T6
builds_on:    T6-004
performance:  "unofficial (200 prompts, job 54): policy B 888.5/888.3 vs stock A 895.0 (n=1) = -0.7%, within noise; C (policy + --max-num-batched-tokens 96 piggyback) 625.5/616.1 = -30%."
summary:
  hypothesis: (B) the Marlin policy speeds mixed steps at k=2; (C) small chunked-prefill budget lets prefill ride decode slack.
  change:     none beyond T6-004; C adds serve flag --max-num-batched-tokens 96 (vLLM applies max_num_scheduled_tokens=96).
  results:    Observed (job 54, ws t6-pol, KB61 check ok, order A B C C B; final A skipped by budget guard):
    A 895.0 tok/s TTFT mean 258 p99 1332 TPOT mean 16.68 p99 41.4 acc 2.349
    B 888.5 / 888.3; TTFT mean 253/250; TPOT mean 16.01/16.39; acc 2.328/2.341
    C 625.5 / 616.1; TTFT mean 839/874 p99 3222/2802; TPOT mean 21.22/21.47; Running 13-14 with 1-3 waiting.
    Policy active in B/C (AI_LOG tmb2/tmb3 configs at M=56-96), off in A.
  conclusion: Interpretation: at the champion's 2048 budget few steps land in M=49-190, so the GEMM gain is ~0 end to end (needs more A runs to bound +-1%). Piggyback with budget 96 loses: mixed steps cost far more than decode steps of the same M (TPOT +5 ms) and TTFT rises 3.3x -> fewer running requests. Mixed-step fixed overhead (graph mode/GDN prefill path/host) is the blocker, not the GEMM.
validity:     B/C math as T6-004 (reorder only). C changes scheduling only (chunked prefill, exact).
