[from the human, via the designer; the manager is offline] You lead team T33, an **external** team, in run test-6 (task qwen38-27b-fp8-2x4090). It is the only team open right now.

Current CHAMPION: T31-003, 1088.9 tok/s official at c=16 (+2.5% over T29-006 1062.1, +71.2% vs BASE 631.4; trials 1091.3/1086.6; TTFT p50/p99 142/839 ms, TPOT p50/p99 13.4/29.9 ms, GSM8K-250 93.6%, MTP acceptance 2.33, 128K pass). Details: champion.md, bank/T31-003/. A performance reference, not a required starting point.

Your direction will come from a deck of external technique cards. A scout is writing it now. It will land in scout/T33-1/ (read index.md first), and you will get an inbox message when it is ready (usually 10-30 min). `scout --status --team T33` shows its progress. Until then, get to know the system.

Grant status: 20 GPU-min seed (`gpu status`). There is no start grant yet. To get one, write a proposal (proposals/T33-<n>.md) and then end your turn: the manager is offline for now, so do not message it; the human reads the file. It should name the card(s) it builds on and make a transfer argument (a seed measurement of the condition the technique needs here). It also needs a roadmap, a forecast (the payoff if it works vs the current champion, broken into measured and assumed parts, plus your central estimate and odds) and the GPU time you are asking for. The bar: the payoff if it works must be >= 5% over 1088.9 (1143.3). Rules: funding_policy.md. Up to 1 worker at a time.

Rules to know (task_definition.md, as amended):
- FP8 ACTIVATIONS: the checkpoint's W8A8 scheme is allowed (dynamic per-token-group-128 e4m3 activations vs 128x128 block-scaled FP8 weights; nothing coarser, static or lower; fp32 across K-blocks). A record using new FP8-activation code runs the GSM8K-250 subset (gsm8k.py default) in the same job as a bf16-activation reference and must be within 2.0 points. The clarification and addenda at the end of task_definition.md say when an existing A/B may be reused.
- Generic fine-grained prefix caching for the hybrid model is allowed (no tuning to the benchmark's prompts).
- Tooling: task/scripts/abba.sh (same-job A/B, stops early on a clear loss); run `gpu run` as a background command (KB90).

Where things are:
- shared_kb.md (KB59-KB100 are this run's findings).
- reports/: each closed team's write-up says what it tried and what is still open. The most recent: T31 (register-direct e4m3 decode GEMM; KB97-99), T30 (layer-pipelined mixed steps; KB100), T29 (W8A8 for mixed steps), T28 (small-M FP8 GEMM, k=3 cost decomposition), T27 (screened draft head), plus earlier ones.
- graph.md, bank/ (any record can be a starting point), announcements.md, history/test-5/.
- task/ (TASK.md, harness, baseline profiles + PROFILING.md), task_definition.md, assumption_ledger.md, focus.md.
Earlier conclusions are evidence, not limits: challenge them with measurements.

