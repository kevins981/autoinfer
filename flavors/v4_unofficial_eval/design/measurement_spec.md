# v4 measurement spec (since 2026-11-15, after sweep-simple-eval)

One measurement for everyone: the task's harness (`harness/eval.sh`), run by the teams themselves
through `gpu run`. There is no separate unofficial or official eval, and the manager runs no GPU jobs.

- **What it runs:** 2 fresh inference-server starts x 400 ShareGPT prompts (the scored benchmark),
  then the 128K-token request on the last start (always: it guards against memory trades that
  break long contexts). GSM8K-250 with `GATES=long,gsm8k`, when a change could alter the math.
  `STOP_BELOW` ends a clear loss after one start. ~7 GPU-min (+2 for GSM8K), charged to the team.
- **Output:** the raw folders (vLLM bench report and JSON, inference server log with its stats
  lines, Prometheus snapshots, GPU clocks, check results); `summarize.py` writes results.json. A
  record's measurements live in `bank/<id>/eval/<n>/`.
- **Champion (one tier):** a team sends a record that beats the champion's measurement and passes
  the checks; the manager reviews validity and the raw outputs, promotes it, and alone writes
  champion.md. BASE's measurement ships with the task (task/baseline/official/).
- **Why:** 200-prompt unofficial numbers understated gains of fast configs ~2x (e.g. T1-006 +6.1%
  at 200 vs +11.9% at 400; BASE equal), so teams could drop real wins; 400 prompts costs ~+1.5
  GPU-min per measurement (startup ~1 min per start dominates), and confirmation evals go away.
- **Still true:** jobs drift up to ~1.6% (BASE 641.4 / 631.4 / 635.9 in three jobs), so a gain
  under ~2% measured against another job's number stays uncertain.
- History: until 2026-11-15, `tools/unofficial_eval` (2 x 200 prompts) for teams and
  `tools/official_eval` (2 x 400 + GSM8K + 128K) for the manager's confirmations, with a two-tier
  champion. Both tools are removed.
