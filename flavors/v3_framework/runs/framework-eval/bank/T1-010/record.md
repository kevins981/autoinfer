team:         T1
builds_on:    T1-009
performance:  "screen negative (1 pair, -5.82%)", T1 GPU job 37, agents/leader-1/jobs/k3b/
summary:
  hypothesis: T1-003's k=3 loss might have come from eager mixed steps and CPU prep idle, both fixed in T1-009; with those
              gone, a 3rd draft (+0.37 acceptance) could pay off.
  change:     num_speculative_tokens 2 -> 3 on the champion T1-009.
  results:    Observed: A 745.9 tok/s acc 2.341 (TTFT/TPOT p99 1348/48.1); B 702.5 acc 2.707 (1695/41.3). Stopped after 1 pair.
  conclusion: Interpretation: step cost grows ~23% (2.707/2.341 x 745.9/702.5) for +33% verify rows (48->64) and +1 draft
              step: the extra verify rows are NOT free here. Weight-only Marlin computes in bf16 (ridge ~80 rows), so at 48-64
              rows the body GEMMs are already near the compute ridge, GDN/attention/AR scale with rows, and the draft step
              adds ~1.5 ms. The framework's "free verification slack below N=333" (04 sec.4) does not hold on this hardware/kernel
              mix: the ridge is ~80, not 333. Revisit only with W8A8 GEMMs (ridge ~165, T2-004) and check TTFT p99.
validity:     exact (draft count only). Not a candidate.
