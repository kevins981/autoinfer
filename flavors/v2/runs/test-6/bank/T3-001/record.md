team:         T3
builds_on:    T1-042
performance:  n/a (data/measurement record: no serving change intended)
summary:
  hypothesis: The stock MTP head's acceptance can be reproduced offline from dumped target hidden states, giving a cheap test bed for a self-distilled draft head (proposals/T3-1.md).
  change:     Env-var + file-flag-gated hidden-state dump hook in both model runners (workspaces/t3-distill.dump; off by default, not for serving). Self-generated data pipeline: agents/leader-3.w1/ (gen_driver_phaseA.py with the topic list, gen_driver.py, dump_driver.py, assemble.py, mtp_model.py, eval_stock.py); provenance agents/leader-3.w1/data/PROVENANCE.md.
  results:    Observed: GPU jobs 2 (10 min, prompts; cold start 409 s), 5 (4.3 min, responses), 6 (2.1 min, dump). 400 target-generated prompts (387 self-instruct from a hand-written generic topic list, 13 Magpie), T=0 responses max 384 tokens, 200 thinking on / 200 off: 144,489 response + 43,579 prompt tokens, all with post-final-norm bf16 hidden states. Split data/split.json (seed 20261108, 40 held-out / 360 train), fixed before eval. Stock head offline (teacher-forced, bf16, 98,304-id draft vocab), held-out 14,138 positions: a1 0.781, cum a2 0.561 (thinking on 0.761/0.533, off 0.803/0.591). In-server benchmark (T1-042 official logs): 0.77/0.55.
  conclusion: Interpretation: the offline re-implementation reproduces in-server acceptance closely (indirect check; no per-position argmax comparison with vLLM's drafter yet). Gap to the proposal's "works" target (0.83/0.68): +5 pts a1, +12 pts a2.
validity:     No serving change. Data provenance follows the manager's clarification (task_definition.md end): target-generated prompts only, no external/benchmark data, held-out-only selection.
