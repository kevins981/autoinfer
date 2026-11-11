team:         T3
builds_on:    T3-003
performance:  n/a (offline training experiment; no serving change)
summary:
  hypothesis: Soft-label KL distillation (teacher = target lm_head on stored hidden states, 98,304-id slice) gives the generalizing signal hard labels lacked; a data-fraction sweep gives the data-scaling slope.
  change:     agents/leader-3.w1/train3.py: KL-C (single head, full FT) lr 2e-6/1e-5; KL-A (dedicated step-2 layer) lr 1e-5; KL-C on 25%/50%/100% of the 360 train prompts; train-set fit check.
  results:    Observed (job t3-kl4, 3.6 GPU-min): held-out a1/a2cum stock .7820/.5617; KL-C lr1e-5 best .7832/.5630 (bootstrap a2 +0.13 pt, CI [-0.25, +0.48]); KL-A .7820/.5617. Held-out KL to teacher falls 11% (step 1) / 17% (masked step 2) while argmax agreement stays flat. Train set: stock .763/.539 -> KL-C lr1e-5 .815/.658. Data slope (lr1e-5, a2 vs stock): 25% -0.55, 50% -0.14, 100% +0.13 pt.
  conclusion: Interpretation: the head fits the training prompts strongly but does not generalize at 130K tokens; the held-out slope is ~+0.3-0.4 pt a2 per doubling, so ~15x more data extrapolates to ~+1.5 pt a2 (~+0.6% throughput), far below the proposal's +12 pt "works" case. The stock jointly-trained MTP head is close to what this layer can learn from affordable data. Direction killed (T3-1); see proposals/T3-2.md for the pivot.
validity:     No serving change; data provenance as T3-001.
