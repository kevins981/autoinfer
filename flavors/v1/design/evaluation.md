# v1: Evaluation

How v1 decides that a candidate is valid and how fast it is. The pseudocode in `DESIGN.md`
uses this; details live here.

## Three kinds of check
| Check | What it answers | Who/what runs it |
|---|---|---|
| **Validity review** | Does the change respect the task's hard constraints and correctness criteria (same math; FP8 weights + FP8 KV; no benchmark-specific tricks)? Done by reading the raw diff, not the author's description. | **The leader** (an agent, no GPU) |
| **Accuracy (GSM8K)** | Did the change break the model's outputs? Catches gross breakage only: n=250 has SE ±1.6, so drops under ~4 points are inconclusive. | GPU job |
| **Benchmark** | How fast is it? This is the objective itself (qwen task: output tok/s on ShareGPT, 400 prompts at concurrency 16), not just a "does it run" check. It also gives MTP acceptance length for free, which is a sensitive correctness signal: if the math drifts, the draft model's acceptance drops. | GPU job |

## The leader is responsible for validity
The leader is ultimately responsible for every experiment it files into the bank and reports to
the manager being valid: it reviews the raw diff against `TASK.md`'s hard constraints and
correctness criteria before requesting an official eval, and its record says it did. A worker's
"it's correct" is not enough. (Workers are the arms; the leader is the brain and signs off.)
There is no separate independent judge in v1: the leader's review is enough.

## Official vs unofficial runs
Everything that touches the GPU goes through the GPU job queue (shared infrastructure). The
difference is not *who* may use the GPU but *what the result counts for*:
- **Unofficial** (any agent, including workers, within a GPU budget): whatever it needs to keep
  optimizing: smoke tests, microbenchmarks, benches, nsys/ncu profiles. Used to develop and to decide what is worth an official
  eval. Never counts.
- **Official** (requested by the leader only, after its validity review): clean build of the
  task baseline + the record's diff, frozen harness the agents can't modify, exclusive GPUs, c=16 bench + GSM8K-250 on the
  same server. Recorded append-only in the bank. Only official numbers count (champion, reports).

## GPU budget
Workers need the GPU while implementing (does it build, does it serve, is it faster), so they get
it within a budget: manager → leader (per round) → workers. See
[budget_and_stopping.md](budget_and_stopping.md).

## Cost of each kind of test (qwen task, approximate)
Unit: **GPU-minutes of exclusive use of both GPUs** (the model needs both 24 GB GPUs, so any job
that loads it blocks everything else). Costs are proportional to wall time. Only GPU time is
counted: CPU work (builds, napkin math, reading code) is plentiful and not budgeted.
| Test | GPU-minutes |
|---|---|
| Kernel microbenchmark with real shapes (1 GPU, no model) | 0.5 |
| Python-only change: restart server | 1.5 (warm) |
| Smoke test: start server + a few prompts | 2 |
| GSM8K-250 (server up) | 2 |
| Bench c=16, 400 prompts (server up) | 3 |
| nsys profile at c=16 | 3 + startup |
| **Official eval** (start + bench c=16 + GSM8K-250) | **7** |

Takeaways:
- GSM8K-250 is cheaper than the benchmark; server startup is a big share of every job. So GSM8K
  doesn't need to be rare; batching several tests on one server start matters more.
- **With 2 GPUs, only one model-loading job runs at a time across the whole run.** At ~7 min per
  official eval that is at most ~8 per hour for all leaders together, before any unofficial work.
  The GPU, not agent tokens, is the scarce resource for this task, so the cost table and the
  validation funnel matter a lot.

TODO: the cost table belongs to the task (move it to `tasks/<task>/` when the harness is
ported), measure the estimated rows, and let the GPU queue use it for scheduling and budgets.

## Example: one experiment end to end
Direction (leader): "overlap the TP all-reduce with the next GEMM." The leader decomposes it; one
experiment for worker W: "implement the overlap for the down-proj all-reduce; budget 30
GPU-minutes; done when you have a working change with c=16 numbers and a profile."
1. W does napkin math (all-reduce is ~27% of kernel time; how much can overlap?) - free.
2. W implements; microbenchmarks the overlapped pair with real shapes - < 1 min.
3. W starts a server, smoke tests, checks T=0 outputs on a few prompts match the baseline - ~2 min.
4. W runs a short unofficial bench + nsys profile - ~6 min. Repeats 2-4 while iterating.
5. W reports: numbers, profile, Observed/Interpretation; the change is in its workspace.
6. The leader files the record, reviews the raw diff for validity, and if it is promising,
   requests an official eval (~7 min). The official numbers and GSM8K are appended to the record.
