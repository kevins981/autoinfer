team:         T3
builds_on:    T3-005
performance:  n/a (offline simulation; no serving change)
summary:
  hypothesis: Fixed verify width M=48 (B=32) with on-device current-step ranking keeps most of the adaptive-verification gain without a CPU budget decision; a calibrated ranker adds more.
  change:     agents/leader-3.w1/sim_budget2.py; isotonic / logistic calibrators fit on train sequences, evaluated on held-out only.
  results:    Observed (CPU; held-out 40 seqs, 14,058 positions; KB step-time model): fixed k=2 E 2.348. K_MAX=3 B=32 1.058; K_MAX=4 B=32 1.064 (raw), 1.069 (iso). Rule (d) K_MAX=4: current-B 1.067, stale-B 1.062 (raw); iso 1.077 / 1.072. Admitted slots per request at K_MAX=4 B=32 (raw): 0:22% 1:22% 2:17% 3:11% 4:28%.
  conclusion: Interpretation: no CPU budget needed; fixed M=48 is as good as rule (d) with stale B. Calibration +0.5 pt (suggestive; 40 sequences). Superseded on costs by T3-007 (measured drafter/verify costs).
note:         the confidence dump (job conf5) ran in the unbuilt t3-distill.dump workspace (KB61); the MTP/target math is the same (replay is bitwise), so acceptance/confidence data are unaffected; only step-time inputs from T3-007 are in question.
validity:     No serving change; calibrators trained on self-generated data only (provenance as T3-001).
