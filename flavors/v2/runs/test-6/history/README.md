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
