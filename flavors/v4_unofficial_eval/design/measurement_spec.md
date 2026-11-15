# v4 measurement spec: expected behavior

How a change gets measured and becomes champion in v4. Tool: `tools/official_eval`; GPU harness: the task's `harness/eval.sh` (one script for both kinds
of measurement; agents run it themselves through `gpu run` for unofficial numbers). If the tools disagree with this file, that is a bug.

## Two kinds of measurement

| | Unofficial eval | Official eval |
|---|---|---|
| Who runs it | anyone (the harness via `gpu run`, TRIALS=2 PROMPTS=300) | the manager only (`official_eval <record>`) |
| GPU harness | `eval.sh` with `TRIALS=2 PROMPTS=300`, optional GSM8K/128K, stop-loss | `eval.sh` defaults: 2 trials x 400 prompts, GSM8K-250 + 128K request |
| What runs | the code asked for, nothing else | the record |
| Cost | ~5 GPU-min (~3 if stopped early), charged to the caller's team | ~8-11 GPU-min, charged to MGR |
| Result | the code's tok/s per start (+ p99s, MTP acceptance, checks) | an official tok/s number |

Every start is a freshly started inference server; the bench client runs the baseline code.

## Unofficial eval
1. Any agent runs the harness itself: `gpu run -- env TRIALS=2 PROMPTS=300 GATES=... STOP_BELOW=...
   $AUTOINFER_RUN/task/harness/eval.sh <workspace> <out> [serve args file]`. A record is measured on
   a workspace made from it (`ws create --from <id>`), output in `bank/<id>/unofficial/<n>/`.
2. The output is the harness's raw folders (vLLM's bench report and JSON, the inference server log,
   Prometheus snapshots, GPU clocks). No wrapper, no summary line: agents read and compare.
3. The manager measures BASE at run start, so teams have a reference.
4. (Until 2026-11-15 a wrapper, `tools/unofficial_eval`, did this and printed one summary line;
   dropped because the line hid what the raw files show, e.g. the inference server's
   running/waiting requests.)

## Champion (two tiers; the manager keeps champion.md, no tool)
- **Unofficial champion**: the manager promotes a valid record whose unofficial starts
  beat the unofficial champion's (it need not be built on the unofficial champion).
- **Official champion** (an official eval backs it). The manager runs
  `official_eval <unofficial champion>` when the unofficial champion is ~5% above the official champion or 4 records were promoted since, and at
  the end; if the official number bears out the unofficial one, the unofficial champion becomes the official champion, otherwise
  the manager finds out why and reverts what doesn't hold up. Judgment, not a formula.
- Funding: a promotion counts as a win; a later revert counts against the team.

## Not in v4
- No `abba.sh` and no same-job A/B protocol in the run's task copy (new_run replaces TASK.md's
  protocol with the noise facts): agents measure any code with the harness and compare.
- No `--ignore-eos`: tested, fixed output lengths don't reduce start-to-start noise.
- No startup overlap: shelved.
