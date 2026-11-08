#!/bin/bash
# seed_v2_run.sh <run name>: a v2 run seeded with test-5's state (bank, champion T1-042, task definition,
# assumption ledger, focus, shared KB, graph) and test-5's history (journal, ideas, reports).
set -e
RUN=$1
V2=/scratch/kevinsong/learning/autoinfer/flavors/v2
S=/scratch/kevinsong/learning/autoinfer/flavors/v1/runs/test-5
R=$V2/runs/$RUN
$V2/tools/new_run $RUN --task qwen38-27b-fp8-2x4090 --no-start | tail -1
rm -rf $R/bank && cp -r $S/bank $R/bank
cp $S/task_definition.md $S/assumption_ledger.md $S/focus.md $S/shared_kb.md $S/graph.md $S/champion.md $R/
cat >> $R/task_definition.md <<'MD'

## Ruling (human, 2026-11-08, for v2 runs): trained draft heads
Retraining or distilling the MTP draft head (same role: it only proposes tokens; verification
stays exact, so outputs are unchanged) is allowed. Its training or tuning data must not be the
benchmark's data: not ShareGPT, not the benchmark's prompts, not anything selected because it
resembles them. It must generalize to other workloads. The validity review and the manager check
where any trained or tuned artifact's data came from.
MD
mkdir -p $R/history/test-5 && cp -r $S/journal.md $S/ideas.md $S/reports $R/history/test-5/
cat > $R/history/README.md <<'MD'
# History: run test-5 (an earlier run on this task)

This run continues from an earlier run, test-5, made with a different version of AutoInfer:
one exploit-style team at a time, 15 rounds of short-horizon directions, 631 -> 942 tok/s (+49%).
Its returns had flattened by the end (about +1%/h over its last rounds).

What carried over into this run, as this run's own state:
- bank/: all 46 test-5 records (BASE, T1-001 ... T1-042), with diffs and official results.
  Any of them can be a starting point (`ws create --from <record>`).
- champion.md: T1-042, 942.2 tok/s official.
- task_definition.md, assumption_ledger.md, focus.md: agreed with the human in test-3/test-5;
  they stand.
- shared_kb.md and graph.md: test-5's understanding of the system and what it tried.

What is here, for reference:
- test-5/journal.md: its decisions and why.
- test-5/ideas.md: ideas proposed there but never tried.
- test-5/reports/: per-round reports and its final report (final.md).

These are test-5's conclusions, made under short-horizon pressure. They can be challenged with
measurements.
MD
cat >> $R/journal.md <<'MD'
## Run start: seeded from test-5
This run starts from test-5's state (see history/README.md): bank with 46 records, champion T1-042
(942.2 tok/s official), test-5's task definition, assumption ledger, focus, shared KB and graph.
initialize() is skipped: the task definition agreed with the human stands. Team ids start at T2
(T1 is test-5's team, whose records are in the bank).
MD
echo "seeded $R"
