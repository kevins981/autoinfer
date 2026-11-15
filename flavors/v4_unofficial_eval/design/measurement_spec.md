# v4 measurement spec: expected behavior

How a change gets measured and becomes champion in v4. Tools: `tools/unofficial_eval`, `tools/champion`,
`tools/official_eval`; GPU harness: the task's `harness/eval.sh` (one script for both kinds of
measurement). If the tools disagree with this file, that is a bug.

## Two kinds of measurement

| | Unofficial eval | Official eval |
|---|---|---|
| Who runs it | a team (`unofficial_eval <record>`) | the manager only (`official_eval <record>`) |
| GPU harness | `eval.sh` with `TRIALS=2 PROMPTS=200`, optional gates, stop-loss | `eval.sh` defaults: 2 trials x 400 prompts, GSM8K-250 + 128K request |
| What runs | only the candidate (B); the champion's numbers come from a cache | the record |
| Cost | ~5 GPU-min (~3 if stopped early), charged to the team | ~8-11 GPU-min, charged to MGR |
| Result | a verdict vs HEAD | an official tok/s number |

Every start is a freshly started inference server; the bench client runs the baseline code.

## Unofficial eval
1. Refuses (exit 2, nothing run) unless the record's `builds_on:` names the current HEAD.
2. Builds a fresh workspace from the bank diff (CPU, not charged).
3. Reference = HEAD's cached starts (`unofficial/cache.json`). Cache key = diff hash + serve args +
   prompts + eval.sh version, so the same code under the same harness never re-runs. If fewer
   than 2 cached starts exist (e.g. BASE at run start), the same GPU job first runs the missing
   ones (`eval.sh` on HEAD's workspace, no gates) and caches them. Normally this never happens:
   the manager runs `unofficial_eval BASE` at run start (reference starts only, charged to MGR),
   and every promoted champion's own 2 starts are its reference.
4. Runs B: 2 trials. Stop-loss: if trial 1 is more than 1.5% below the reference mean, trial 2
   is skipped (only when the reference was fully cached).
5. Verdict: **PASS** = both B starts above the reference mean and every gate run passed (latency
   guard always; GSM8K with `--gsm8k`; 128K with `--long`). Otherwise mixed / negative /
   negative (1 start) / FAIL (gates) / failed.
6. Writes `bank/<id>/unofficial/<n>/unofficial.json` and appends one `unofficial #n` line to record.md.
   B's starts are cached, so if the record becomes HEAD they are its reference.

## Champion (two tiers)
- **HEAD** = the unofficial champion. `champion promote <id>` (manager) needs the record's
  latest unofficial eval to be a PASS against the current HEAD. Teams build on and run unofficial_eval against HEAD.
- **CONFIRMED** = the official champion (an official eval backs it).
- The chain = records promoted since CONFIRMED, each with its unofficial gain.
- **Confirmation is due** when the summed unofficial gains reach 5% or the chain holds 4 records
  (fixed, not a judgment call). The manager then runs `official_eval HEAD` and
  `champion confirm HEAD`.
- **MATCH** if all hold: gates pass; the official number beats CONFIRMED's; and it is no lower
  than expected - max(1.5%, 2 sigma), where expected = CONFIRMED x the chain's gains and
  sigma = sqrt(k x 0.75^2 + 0.8^2) % for k records. Then the record becomes CONFIRMED.
- **MISMATCH**: bisect the chain with official evals (the tool names the next record). The first
  record that mismatches is reverted with everything promoted after it (`champion revert`);
  HEAD falls back.
- Funding: a promotion counts as a win; a later revert counts against the team.

## Not in v4
- No `abba.sh` (removed from the run's task copy): `unofficial_eval` is the only comparison path.
- No `--ignore-eos`: tested, fixed output lengths don't reduce start-to-start noise.
- No startup overlap: shelved.
