# autoinfer baseline: a single-agent research loop

This is an experiment to have the LLM do its own research: make an LLM inference server faster.
(Adapted from karpathy/autoresearch `program.md`: one agent, one metric, keep or discard. The
task, harness and tools are AutoInfer's.)

## Setup

`new_run` has already prepared the run; you don't need the human to start. Check it:

1. **The run folder** is `$AUTOINFER_RUN`: `config.md` (this run's limits), `task/` (the task),
   `results.tsv` (header only), `workspaces/main/` (your copy of the engine).
2. **The branch**: the code tree `$AUTOINFER_RUN/workspaces/main/vllm` is a git worktree on
   branch `autoresearch/<run>`, at the task's baseline commit plus one setup commit that adds an
   empty `serve_args.txt`.
3. **Read the in-scope files** for full context:
   - `task/TASK.md`: the objective, metric, gates, what is fixed and what is open, correctness
     rules. It is the rulebook: read it fully.
   - `task/PROFILING.md` and `task/baseline/`: profiles of the baseline (where the time goes).
   - `task/harness/`: the frozen benchmark (`eval.sh`, `serve_args.sh`, `summarize.py`). Read,
     never modify.
   - `task/scripts/`: helpers for your own GPU jobs (start a server, bench, gsm8k, profile).
   - The engine source in the code tree: whatever you need, as you need it.
4. **Confirm the limits** in `config.md` (GPU hours, experiments and/or wall-clock hours), then start.

## Experimentation

Each experiment measures the current commit with the task's official harness (2 trials on fresh
servers, ShareGPT 400 prompts at concurrency 16; GSM8K-250; one 128K-token request); ~11 min with
a warm cache, more if it must build. You launch it simply as: `experiment` (as a background
command: it takes longer than the Bash limit).

**What you CAN do:**
- Modify anything in the code tree: Python, kernels (`csrc/`, rebuilt automatically by
  `experiment`; or JIT-compiled at server start), scheduler, memory, communication, compilation.
- `serve_args.txt` at the root of the code tree: extra `vllm serve` flags on top of the task's
  (`task/harness/serve_args.sh`), one line, shell syntax. It is committed with the code.
- Run your own GPU jobs for profiling, debugging or quick checks with `gpujob` (see Tools).

**What you CANNOT do:**
- Modify `tasks/` (the task, harness, benchmark, dataset, gates), or anything outside your run
  folder.
- Break the task's fixed list or correctness rules (`TASK.md`): same model and weights, FP8
  weights and KV cache, exact sampling and speculative decoding, nothing specific to the
  benchmark, full answers. A faster number that breaks them doesn't count.
- Install packages system-wide or use sudo. Never write to `$HOME`.

**The goal is simple: get the highest `tok_s` (output tokens/s at concurrency 16) with all gates
passing** (GSM8K, latency p99s, the 128K request). BASE scores 635.9 tok/s.

**Noise.** Repeat measurements of the same code on fresh servers differ by up to ~1.6%. A gain
smaller than that may be noise: if a change looks like a small win, re-run `experiment` on it (or
on the previous best) before you build on it. `TASK.md` explains the noise in detail.

**Simplicity criterion**: all else being equal, simpler is better. A small improvement that adds
ugly complexity is not worth it. Removing something and getting equal or better results is a win.

**The first run**: your very first run should always establish the baseline: run `experiment` on
the setup commit as is.

## Output format

When `experiment` finishes it prints a summary like this:

```
---
tok_s:             635.9
tok_s_trials:      641.2 / 630.6
vs_base_pct:       +0.00
gates:             pass (gsm8k pass 94.4%, latency pass, 128K pass)
ttft_p99_ms:       1341
tpot_p99_ms:       46.1
mtp_acceptance:    2.33
results:           .../experiments/001-1a2b3c4/results.json
limits:            1 experiments done; 0.18 GPU-hours used of max 4; 0.25 h elapsed
```

Everything else (server logs, bench JSONs, GSM8K answers, the build log) is in that experiment's
folder. If it prints `status: FAILED at <stage>`, the logs are there too.

## Logging results

When an experiment is done, log it to `$AUTOINFER_RUN/results.tsv` (tab-separated, NOT
comma-separated: commas break in descriptions). It is outside the code tree; never commit it.

The TSV has a header row and 6 columns:

```
commit	tok_s	gates	status	experiment	description
```

1. git commit hash (short, 7 chars)
2. tok_s achieved (e.g. 652.3); 0.0 for crashes
3. `pass` or `FAIL` (which gate, e.g. `FAIL:gsm8k`); `-` for crashes
4. status: `keep`, `discard`, or `crash`
5. the experiment folder name (e.g. `003-1a2b3c4`)
6. short text description of what this experiment tried and why

Example:

```
commit	tok_s	gates	status	experiment	description
1a2b3c4	635.9	pass	keep	001-1a2b3c4	baseline
b2c3d4e	652.3	pass	keep	002-b2c3d4e	fuse residual add into rmsnorm (fewer launches)
c3d4e5f	661.0	FAIL:latency	discard	003-c3d4e5f	max-num-batched-tokens 8192
d4e5f6a	0.0	-	crash	004-d4e5f6a	custom all-reduce kernel (illegal memory access)
```

## The experiment loop

The experiment runs on the branch `autoresearch/<run>` in the code tree.

LOOP until the run's limit is reached:

1. Look at the git state: the current branch/commit we're on.
2. Change the engine (or `serve_args.txt`) with an experimental idea by directly hacking the code.
   Profile or test first with `gpujob` when that is the fastest way to know where to go.
3. git commit.
4. Run the experiment: `experiment` (background; you'll be notified when it ends).
5. Read out the results: the summary block it prints.
6. If it failed, read the logs in the experiment folder (`eval.log`, `trial*/server.log`,
   `build.log`) and attempt a fix. If you can't get things to work after more than a few
   attempts, give up on that idea.
7. Record the results in the tsv.
8. If tok_s improved over the best kept result and all gates pass, you "advance" the branch,
   keeping the git commit.
9. If tok_s is equal or worse, or a gate fails, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep.
If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like
you're getting stuck in some way, you can rewind but you should probably do this very very
sparingly (if ever).

**Timeout**: `experiment` kills an eval after 45 min and reports a timeout; treat it as a failure
(discard and revert). Server startup must stay under 15 min.

**Crashes**: if a run crashes (OOM, a bug, a build error), use your judgment: if it's something
dumb and easy to fix (a typo, a missing import), fix it and re-run. If the idea itself is
fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**GPUs are shared**: `experiment` and `gpujob` refuse to start (exit 75) if someone else is using
the GPUs. Then wait (e.g. 10 min) and retry; never compete for the GPUs, never start a server or
other GPU work outside these two tools, and never run two at once.

**DON'T STOP EARLY**: once the experiment loop has begun, do NOT pause to ask the human if you
should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human
might be asleep, or gone from a computer and expects you to continue working until the run's limit
is reached. You are autonomous. If you run out of ideas, think harder: re-read the profiles and
the in-scope files for new angles, profile the current best, read the engine code on the hot path,
try combining previous near-misses, try more radical changes.

## If the human asks you to pause
This is the one exception to "don't stop early". Let a running `experiment` or `gpujob` finish if
it ends within ~15 min, otherwise stop it (TaskStop) and note it in results.tsv as `crash` with
"interrupted by pause"; make sure nothing of yours is left on the GPUs (`nvidia-smi`); commit or
note any work in progress; then reply with where things stand and wait. The human will resume this
same session later: then check the state of what you had started, and continue the loop.

## When the run ends

When `experiment` (or `gpujob`) says the run's limit is reached (exit 76), stop experimenting and
write `$AUTOINFER_RUN/final.md`: the best kept commit and its result, the branch's kept changes in
order (what each does and what it gained), what you tried that didn't work and why you think it
didn't, and what you would try next. Then make sure nothing of yours is still running on the GPUs,
and stop.

## Tools

On your PATH; `<tool> --help` explains each.
- `experiment [--timeout MIN]`: the official measurement of the current commit (above).
- `gpujob --timeout MIN [--label L] -- <command>`: one ad-hoc GPU job (e.g. a script that starts
  a server from your workspace, benches or profiles it, stops it). Background command; it kills
  everything it started when it ends. Logs in `$AUTOINFER_RUN/gpu/jobs/`. Use
  `task/scripts/lib.sh` (`start_server`, `bench`, `gsm8k`, `long_request`, `stop_server`) and the
  profiling scripts.
- `ws build main` (background): rebuild compiled code by hand (`experiment` does it for you when
  the commit changes `csrc/`). `source $AUTOINFER_RUN/workspaces/main/env.sh` before running
  anything from the workspace yourself: it sets the environment and makes Python load your code.
- `ws diff main`: your branch's change vs the baseline.
Keep scratch files, logs and scripts outside the code tree (e.g. `$AUTOINFER_RUN/scratch/`), or
`git add -A` will commit them.
