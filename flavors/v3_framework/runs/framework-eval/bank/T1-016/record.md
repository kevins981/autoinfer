team:         T1
builds_on:    T1-014
performance:  "screen negative (1 pair, -9.16%)"
summary:
  hypothesis: Spar T1-1: "k=2 is best" was only extrapolated from the 2->3 loss. k=1 wins only if the step gets >24% cheaper
              (acceptance 2.34 -> ~1.78). Napkin from the measured 2->3 cost: step -17..-20%, so k=1 loses ~5%.
  change:     serve flag only: num_speculative_tokens 1 (code = T1-014).
  results:    Observed (job 67, agents/leader-1/jobs/k1): A (k=2) 816.3 tok/s, acc 2.355; B (k=1) 741.6 tok/s, acc 1.797;
              TTFT/TPOT p99 B 799/35.6 vs A 932/31.4 ms. Implied step time ratio B/A = (1.797/2.355)/(741.6/816.3) = 0.84.
  conclusion: Interpretation: step -16% (napkin -17..-20%) against tokens/step -24%: k=1 loses, as napkin'd. With T1-010 (k=3
              -5.8%), k=2 is the optimum for this drafter on both sides (one-pair screens each).
validity:     flag only; exact spec decode.
