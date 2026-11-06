# GPU queue (spec)

All GPU work in a run goes through one command, `gpu`. It makes sure only one job uses the GPUs at
a time, charges each team's budget, and kills jobs that run too long. Shared by all flavors; the
v1 budget rules are in `flavors/v1/design/budget_and_stopping.md`.

## Model: batch jobs (like a tiny Slurm)
A job is **one command** (usually a script the agent writes) that gets both GPUs exclusively until
it exits or hits its timeout. E.g. "start the server, smoke test, bench c=16, nsys, stop the
server" is one job, which also batches several tests on one server start (startup is a big share
of every job). No interactive leases in v1: an agent can't hold the GPUs across several of its own
Bash calls.

## Commands
| Command | What it does |
|---|---|
| `gpu run --timeout <min> [--label L] -- <cmd...>` | Wait for a turn (FIFO), check the budget, run `cmd` with both GPUs, print where its output is, and return its exit code. **Blocks** until the job ends; agents run it as a background Bash command and get notified when it exits (a job can wait and run longer than the 10-minute Bash limit). |
| `gpu status` | The running job, the waiting jobs, and per team this round: budget, used, remaining, with a per-agent breakdown. |
| `gpu cancel <job id>` | Cancel a waiting job, or kill a running one (charged for the time used). |

The team and agent are read from the environment (`AUTOINFER_TEAM`, `AUTOINFER_AGENT`, set by
`agent start`), not passed as arguments.

## Rules
- **One job at a time** on the GPUs, for everyone in the run (a model-loading job needs both
  24 GB GPUs). Even a 1-GPU microbenchmark takes a whole turn in v1; they are short.
- **FIFO.** No priorities.
- **Admission**: a job starts only if the team's remaining budget for this round covers its
  timeout. Otherwise it is rejected with "budget exhausted: <remaining> GPU-min left". Checked
  twice: when submitted (fail fast) and again when the job reaches the front of the queue (the
  team may have spent budget while it waited). Since only one job runs at a time, the check at
  the front is exact.
- **Charging**: actual wall time from start to exit, in GPU-minutes. Time spent waiting in the
  queue is free.
- **Timeout**: at the timeout the whole job is killed (its process group, then any leftover GPU
  processes of ours). This also catches hung servers.
- **Cleanup after every job**: the queue checks that none of our processes still hold GPU memory
  before starting the next job, so a crashed job can't poison the next one.
- **Unbudgeted** agents (manager, brainstormers; no `AUTOINFER_TEAM`) still queue, and their time
  is recorded but not charged.
- **Other users** of the shared GPUs: before starting a job, the queue checks `nvidia-smi` for
  processes that aren't ours; if there are any, it waits (not charged) and says so in `status`.

## Budgets
- The manager writes each round's budgets at the start of the round:
  `$AUTOINFER_RUN/gpu/budget.json` = `{"round": 3, "teams": {"T1": 40, "T2": 40, "T3": 60}}`.
  Unused budget doesn't carry over.
- **Remaining** = `budget.json[team]` − the GPU-minutes of that team's jobs in `ledger.jsonl`
  for the current round. The team comes from the job's environment (`AUTOINFER_TEAM`, set by
  `agent start`), and the queue measures each job's time itself (start and end timestamps), so
  agents never report their own usage.
- Spending is an append-only log, `$AUTOINFER_RUN/gpu/ledger.jsonl`: one line per finished job
  (job id, round, team, agent, label, start, end, GPU-minutes, exit code, killed by timeout?).
  `status` is computed from it.

## Files
```
$AUTOINFER_RUN/gpu/
  budget.json       # this round's budgets (manager)
  ledger.jsonl      # one line per finished job (append-only)
  queue/            # tickets of waiting jobs (FIFO order)
  lock              # held by the running job (flock)
  jobs/<id>/        # cmd, stdout+stderr, meta (team, agent, label, timeout, times)
```
No daemon: `gpu run` takes a ticket, waits for its turn on a `flock`, runs the job, appends to the
ledger and releases the lock.

## Official evals
An official eval is a normal job, submitted by the leader with `--label official:<record id>`
and running the official eval script (spec: TBD, subsystem 4). It is charged to the team like
any other job.

## Not in v1
Priorities, preemption, interactive leases, running two 1-GPU jobs side by side, moving budget
between teams mid-round, enforcing that agents use the queue at all (they are trusted).
