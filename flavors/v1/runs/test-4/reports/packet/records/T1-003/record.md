team:         T1
performance:  "official #1: 691.3 tok/s (696.1/686.4), +8.7% vs BASE, +5.7% vs T2-003; all gates pass"
summary:
  hypothesis: T1-001 (all-reduce) and T2-003 (drafter head flags) touch different parts of the step; gains should add.
  change:     T1-001's change.diff (unchanged) + T2-003's serve_args.txt (local-argmax drafting, FP8 draft
              lm_head, --enable-batch-sharded-sampling). Builds on T1-001 and T2-003.
  results:    Observed (job 15, one job, bench 100 c=16, tok/s / TPOT med): BASE 591.5 / 24.36; host AR 625.9 / 23.36;
              T2 flags 613.9 / 23.48 and 622.5 / 23.22 (repeat); host AR + T2 flags 657.1 / 22.00 and 647.8 / 22.00.
              MTP AL 2.35-2.37 all legs. Logs: agents/leader-1.w1/ab4/.
  conclusion: Interpretation: the gains stack (~additive). Leg-to-leg noise ~9 tok/s at bench 100.
validity:     leader-1: diff = T1-001 (reviewed OK, official pass). serve_args = T2-003's (reviewed by leader-2,
              official pass; FP8 draft head and local argmax in the drafter are explicitly allowed by
              task_definition). Combination changes no math beyond its parts.

official #1 (2026-11-07T05:43): 691.3 tok/s at c=16 (mean of 2 trials: 696.1 / 686.4, CV 1.0%) (+8.7% vs BASE), GSM8K 92.8%, MTP acceptance 2.346, TTFT p50 226 ms, TPOT p50 21.3 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
  profile: bank/T1-003/profiles/anatomy.md (nsys job 17; raw: agents/leader-1.w1/prof_t1003/)
