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
- **Guard:** a workspace's `env.sh` refuses (exit 1, with the reason) when its C++/CUDA code
  (`csrc/`, `CMakeLists.txt`, `cmake/`) differs from the baseline and was never built, or changed
  after the last build. Otherwise the workspace would silently run the baseline's compiled kernels
  (found in test-6, KB61). `ws create` says when a copy needs a build, and `ws list` shows
  `BUILT: NEEDED`. Because a sourced script can't stop its caller, any Python started from that
  shell then exits at once with the same message, even without `set -e`. Override only on
  purpose: `AUTOINFER_ALLOW_UNBUILT=1`.
- `ws diff <name>`: the change vs the baseline; this is what a record's `change.diff` is. Keep logs,
  results and scratch files **outside the code tree** (e.g. in the workspace folder), or they end
  up in the diff.
- Workspaces are kept after an experiment; delete one (`ws rm`) only if it is broken or useless.

## `inspire`: inspiration cards
- `inspire`: prints a set of cards for your team (general principles, and facts about this
  machine's hardware and software) and saves the same text to
  `$AUTOINFER_RUN/inspiration/<team>-<n>.md`. For inspiration only; the cards say how to use them.
- `inspire --list` / `inspire --show <file>`: all titles / one card again.
- Manager: `inspire --team <team>` when a team opens (attach the saved file to the first task,
  unchanged), and optionally at a last chance.
- How many cards of each kind: the `inspiration` lines of the run's `config.md`.

## `gpu`: the GPU queue (all GPU work goes through it)
- `gpu run --timeout <min> [--label L] -- <command>`: runs your command with exclusive use of the
  GPUs, one job at a time for the whole run, first come first served. **Run it as a background
  command** (you are notified when it ends; waiting in the queue can take a while). Your team's account
  is charged the actual minutes (at most `--timeout`), if its balance covers `--timeout`. The
  balance is printed when the job starts and ends.
- **One job = one script**: e.g. start the server, smoke test, bench, profile, stop the server.
  Everything the job started is killed when it ends or times out, so batch several tests per
  server start, and don't detach processes from the job (`setsid`, daemons).
- Exit codes: the command's own; 124 = timed out; 76 = not admitted: your team's balance can't cover
  `--timeout`, or the team is closed (do GPU-free work; leaders may request a resync); **75 = the GPUs are used by someone outside this run: stop GPU work and report it to the
  manager** (other people use these GPUs; we must not compete).
- `gpu status`: the running job, the queue, the run's pool, and every team's account (granted,
  used, balance). `gpu history <team>`: a team's grants and jobs over time.
- Manager only: `gpu team open/close`, `gpu grant`. Human only: `gpu pool init/add`.
- `gpu cancel <job id>`: cancel a waiting job or kill a running one.
- Startup of a model server is a big share of every job (~1.5-4 min); a full job output log is
  in `$AUTOINFER_RUN/gpu/jobs/<id>/output.log`.
