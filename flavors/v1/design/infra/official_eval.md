# Official eval (spec)

The only way a number becomes official. Builds a record from scratch (`baseline + change.diff`),
runs the task's frozen harness on exclusive GPUs, and appends the results to the record. Shared
by all flavors; the v1 rules (who may request it, what counts) are in
`flavors/v1/design/evaluation.md`.

## Command
```
official_eval <record id>
```
Requested by a leader, after its validity review. It blocks until done (run it as a background
Bash command). Three steps:
1. **Prepare (CPU, not charged)**: `ws create eval-<id>-<n> --from <id>` (a fresh workspace from
   the baseline + the record's diff only), then `ws build` if the diff touches compiled code.
2. **Measure (GPU)**: one job, `gpu run --label official:<id> --timeout 15 -- <task harness>`,
   charged to the leader's team. The harness, from the task folder:
   start the server with the task's serve command → wait until ready → bench (qwen: ShareGPT, 400
   prompts, c=16) → GSM8K-250 → stop the server.
3. **Record**: write the results to `bank/<id>/official/<n>/` and append one line to the record's
   `performance` field, e.g. `official #1: 689 tok/s at c=16 (+5.0% vs BASE), GSM8K 93.6%`.

## What is saved (`bank/<id>/official/<n>/`)
- `results.json`: output tok/s, TTFT p50/p99, TPOT p50/p99, ITL p50, MTP acceptance length,
  GSM8K accuracy (+ number without an answer), and the comparison with BASE.
- `server_cmd.txt`, `server.log`, the bench and GSM8K raw outputs, GPU clocks during the bench.
- Provenance: record id, the hash of `change.diff`, the baseline commit, the harness version
  (its git hash in this repo), start/end times.

## Rules
- **Built only from `baseline + change.diff`.** Nothing else from the worker's workspace is used.
- **The harness is frozen**: it lives in `tasks/<task>/harness/`, agents must not modify it
  (trusted in v1), and its git hash is saved with every result so a change would show.
- **Serve flags**: the task's serve command, plus optional extra flags from a `serve_args.txt` in
  the record (for changes that need a flag, e.g. a different way to split the model across the
  GPUs). The leader's validity review checks them against the task definition (e.g. changing the
  number of MTP draft tokens or the max context is forbidden).
- **Append-only.** A record can be evaluated more than once (`official/1`, `official/2`, ...);
  nothing is overwritten. A fixed version of a change is a new record, not a re-eval.
- **Failures are results too**: a failed build, a server that doesn't start, or a crash mid-bench
  is saved as `official #n: failed at <stage>` with the logs.
- The eval doesn't decide anything: the manager compares results with the champion between rounds.

## Implementation (`tools/official_eval`, `tasks/<task>/harness/`)
- Harness: `eval.sh <workspace> <out> [serve_args.txt]` (server from the workspace on port 8100,
  ShareGPT c=16 x 400, GSM8K-250), `serve_args.sh` (the task's serve flags), `gsm8k.py`,
  `summarize.py` (→ `results.json`, with `vs_baseline_pct` against BASE's latest successful eval).
- The **clients** (bench, GSM8K) run from a clean baseline tree prepared by `ws setup-task`, not
  from the record's workspace.
- Exit codes: 0 measured, 1 failed (recorded), 2/75 GPU job not run (nothing recorded).
- Tested: BASE eval in 6.2 min (630.6 tok/s, GSM8K 92.8%); a diff that doesn't apply is recorded
  as a failure; the build check flags `csrc/`, `cmake/`, `CMakeLists.txt`.

## Not in v1
Several trials per eval (noise is ~0.5% for the qwen task; one trial is enough), the full GSM8K,
other concurrency levels.
