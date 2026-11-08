# Run tools (for run agents)

Shared by all flavors; on every agent's PATH. `$AUTOINFER_RUN` is the run folder;
`$AUTOINFER_AGENT` / `$AUTOINFER_TEAM` say who you are (set when you were started).
Agents get the essentials in `agents/common.md`; every tool's `--help` (and `<tool> <command>
--help`) explains each command, with examples.

## `agent`: who is who
- `agent list`: the run's agents (name, role, team, parent, busy/idle, current task, last activity).
  Use it to find who to talk to; then talk with `SendMessage` to the run-scoped name in its SENDMESSAGE TO column (e.g. `test-4-leader-1`).
- `agent start <name> --role R [--team T] [--cwd DIR] [--persona P] [--task-file F]`: start a new agent
  (manager and leaders). It gets the flavor's `agents/common.md` + `agents/<role>.md` as its
  instructions (override with `--prompt-file`), and the task file as its first message.
  Model and effort come from the role's line in the run's `config.md` (override with
  `--model`/`--effort`); context auto-compacts at 400K tokens (`--autocompact`). `agent stop <name>` when it is no longer needed: agents never
  exit on their own.
- Start every `SendMessage` with your name and role, e.g. `[leader-1, leader of team T1]`.

## `ws`: workspaces (your own copy of the engine)
- `ws create <name> [--from <record id>]` (leaders): a full, independent copy at the baseline +
  that record's diff, in seconds. `source $AUTOINFER_RUN/workspaces/<name>/env.sh` before running
  anything: it sets the environment and makes Python load this workspace's code.
- `ws create <name> --from-ws <workspace>` (anyone, e.g. workers): a copy of a workspace's
  current change, uncommitted edits included. Use it to keep editing while a GPU job uses the
  original: inference servers load a workspace's code when they start, so don't edit one mid-job.
- Python-only changes: nothing to build; restart the server.
- C++/CUDA changes: `ws build <name>`. **Run it as a background command**: the first build in a
  workspace takes ~1 min when the shared compile cache has its files, more per changed kernel file
  (can pass the 10-minute Bash limit); later builds are incremental.
- `ws diff <name>`: the change vs the baseline; this is what a record's `change.diff` is. Keep logs,
  results and scratch files **outside the code tree** (e.g. in the workspace folder), or they end
  up in the diff.
- Workspaces are kept after an experiment; delete one (`ws rm`) only if it is broken or useless.

## `gpu`: the GPU queue (all GPU work goes through it)
- `gpu run --timeout <min> [--label L] -- <command>`: runs your command with exclusive use of the
  GPUs, one job at a time for the whole run, first come first served. **Run it as a background
  command** (you are notified when it ends; waiting in the queue can take a while). Your team is
  charged the actual minutes, if its remaining budget covers `--timeout`.
- **One job = one script**: e.g. start the server, smoke test, bench, profile, stop the server.
  Everything the job started is killed when it ends or times out, so batch several tests per
  server start, and don't detach processes from the job (`setsid`, daemons).
- Exit codes: the command's own; 124 = timed out; 76 = your team's budget is exhausted (do GPU-free
  work); **75 = the GPUs are used by someone outside this run: stop GPU work and report it to the
  manager** (other people use these GPUs; we must not compete).
- `gpu status`: the running job, the queue, and each team's budget, used and remaining this round.
- `gpu cancel <job id>`: cancel a waiting job or kill a running one.
- Startup of a model server is a big share of every job (~1.5-4 min); a full job output log is
  in `$AUTOINFER_RUN/gpu/jobs/<id>/output.log`.
