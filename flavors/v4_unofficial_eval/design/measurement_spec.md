# v4 measurement spec: expected behavior

How a change gets measured and becomes champion in v4. Tools: `tools/unofficial_eval`,
`tools/official_eval`; GPU harness: the task's `harness/eval.sh` (one script for both kinds of
measurement). If the tools disagree with this file, that is a bug.

## Two kinds of measurement

| | Unofficial eval | Official eval |
|---|---|---|
| Who runs it | anyone (`unofficial_eval <record>` or `--ws <workspace>`) | the manager only (`official_eval <record>`) |
| GPU harness | `eval.sh` with `TRIALS=2 PROMPTS=200`, optional GSM8K/128K, stop-loss | `eval.sh` defaults: 2 trials x 400 prompts, GSM8K-250 + 128K request |
| What runs | the code asked for, nothing else | the record |
| Cost | ~5 GPU-min (~3 if stopped early), charged to the caller's team | ~8-11 GPU-min, charged to MGR |
| Result | the code's tok/s per start (+ p99s, MTP acceptance, checks) | an official tok/s number |

Every start is a freshly started inference server; the bench client runs the baseline code.

## Unofficial eval
1. Measures what it's asked: a bank record (its diff + serve args, in a fresh workspace built from
   the bank, CPU, not charged) or a workspace as it is (`--ws`, `--label`, optional `--args FILE`,
   `--env K=V`). It knows nothing about champions or references.
2. 2 trials of 200 prompts; `--gsm8k` / `--long` add the checks; `--stop-below X`: skip trial 2
   if trial 1 is below X tok/s (the agent picks X, usually from the unofficial champion's numbers).
3. No verdict, no cache: it writes the numbers (`bank/<id>/unofficial/<n>/` + a line in record.md,
   or `unofficial/ws/<label>/<n>/`) and a line in `unofficial/log.md`. Agents compare numbers
   themselves; records don't change once filed, so a record's numbers stay valid.
4. The manager measures BASE at run start (`unofficial_eval BASE`), so teams have a reference.

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
  protocol with the noise facts): `unofficial_eval` measures and compares any code.
- No `--ignore-eos`: tested, fixed output lengths don't reduce start-to-start noise.
- No startup overlap: shelved.
