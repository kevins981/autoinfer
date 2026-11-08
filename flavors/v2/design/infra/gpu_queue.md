# v2 GPU queue: grants (spec)

**Status:** agreed with the human on 2026-11-08. Implemented in `flavors/v2/tools/gpu`; see
"Tests" below.

This is a delta from v1's queue ([../../../v1/design/infra/gpu_queue.md](../../../v1/design/infra/gpu_queue.md)).
Everything not mentioned here stays as in v1:
- one exclusive job at a time, first in, first out, with no priorities;
- batch jobs: `gpu run --timeout MIN [--label L] -- <cmd>`, plus `gpu wait`;
- timeout kills and GPU cleanup after every job;
- "other users come first" (exit 75);
- the team and agent are read from the environment (`AUTOINFER_TEAM`, `AUTOINFER_AGENT`,
  `AUTOINFER_ROLE`);
- an append-only `ledger.jsonl` of finished jobs;
- agents are trusted to send every GPU job through the queue.

What changes is the **budget model**. v1 gave per-round budgets in `budget.json`. v2 has a fixed
run pool and per-team grants, with no rounds.

## Concepts
- **GPU-minute:** one minute of the GPU pair held exclusively, as in v1. Queue time is free.
- **Pool:** the run's total GPU-minutes. It is set once when the run starts and can be raised later
  by the human. Every grant comes out of it.
- **Account:** one per team, opened when the team is created and closed when it is shut down.
  - Team ids are never reused: a replacement team gets a new id.
  - Each account has a type: `explore`, `exploit`, or `manager`.
  - The manager's own account, `MGR` (type `manager`), is opened automatically with the pool.
- **Grant:** GPU-minutes moved from the pool to an open account. Kinds:
  - `seed`: given automatically when an account opens. Its size is set with the pool.
  - `start`, `extension`, `last-chance`: given by the manager. Any of these kinds can also fund
    `MGR`.
- **Account balance** = the sum of the account's grants − the GPU-minutes of its finished jobs.
- **Pool free** = pool total − the grants of open accounts − the GPU-minutes used by closed
  accounts. So a closed account's unspent time goes back to the pool, and minutes a closed
  account's last job uses are still counted.
- **Grant sizes other than the seed live only in the manager's instructions and config.** Leader
  prompts and the tool don't hard-code them.

## Commands
| Command | Who | What it does |
|---|---|---|
| `gpu pool init <minutes> --seed <minutes>` | human / `new_run` | Creates the pool and the `MGR` account. Refused if a pool already exists |
| `gpu pool add <minutes> --note "<why>"` | human | Raises the pool total |
| `gpu team open <team> --type explore\|exploit` | manager | Opens the account and grants the seed. Refused if the id was ever used, or if the pool can't cover the seed |
| `gpu grant <team> <minutes> --kind start\|extension\|last-chance --note "<why>"` | manager | Refused if the account isn't open or the pool can't cover it |
| `gpu team close <team> --note "<why>"` | manager | Closes the account and cancels the team's waiting and running jobs. A running job is charged for the time it used |
| `gpu run ...` | anyone | As in v1, with admission checked against the account balance (see Rules) |
| `gpu status [--json]` | anyone | The running job, the queue, the pool (total, granted to open accounts, spent, free), then each account: type, open or closed, granted, used, balance, use per agent |
| `gpu history <team> [--json]` | anyone | The account's events (open, grants with notes, jobs, close) in time order, with the running balance. Raw material for the team's track record |
| `gpu wait`, `gpu cancel` | anyone | As in v1 |

"Who" is checked through `AUTOINFER_ROLE`, on trust as in v1:
- manager commands need `AUTOINFER_ROLE=manager`, or no role at all (a human or designer shell);
- `pool init` and `pool add` need no role (the human, or `new_run` before the manager starts).

## Which account a job is charged to
| Environment | Account |
|---|---|
| `AUTOINFER_TEAM=<team>` | that team |
| no team, `AUTOINFER_ROLE=manager` | `MGR` |
| no team, no role (a human or designer shell) | none. The job is recorded but not charged and not admission-checked |
| no team, any other role | refused (exit 76, "no GPU account") |

## Rules
- **Admission:** a job starts only if its account is open and the balance covers its timeout.
  Otherwise it exits 76 with the reason ("budget exhausted: X GPU-min left", "team closed", "no
  account").
  - It is checked at submission and again at the front of the queue, as in v1. Only one job runs
    at a time, so the check at the front is exact.
  - **Charge** = min(actual run time, timeout). The kill grace and GPU cleanup after a timeout
    or cancel are never charged, so a balance can't go negative. The ledger also records the real
    `wall_minutes`. A job cancelled by `team close` is charged too.
  - **A job refused at the front of the queue** gets `state: "refused"` and a reason in its
    `job.json`, so `gpu wait` returns for it.
  - **Signals can't split the bookkeeping.** During ledger writes and process start, signals are
    held back. A second cancel while a job is being killed is ignored. Either way the job is
    charged exactly once and its processes are killed.
- **The balance on every job:** when a job starts and when it ends, `gpu run` prints the account's
  balance. Example: `team T2 (explore): 34.5 GPU-min left of 100 granted`.
- **Low-balance notice:** when a job ends and the balance is under 25% of the account's latest
  grant (the seed counts as a grant), the output adds: `low balance: consider requesting a resync
  from the manager`. This is not shown for `MGR` or for closed accounts.
- **Closed accounts:** a job cancelled by `team close` prints `cancelled: team T closed`. The
  balance lines then say the team is closed instead of showing minutes that went back to the
  pool. `team close` waits (up to 2 min) until the cancelled jobs are charged, then prints the
  pool's free minutes.
- **Pool free while a closed team's job is still finishing** holds back that job's timeout until
  the job is charged, so a grant made in the meantime can't overdraw the pool.
- **Official evals are normal jobs** charged to the requesting team, as in v1. That includes the
  re-measurement after porting onto a new champion.
- **Validation:**
  - team ids are 1–32 letters, digits, `-` or `_`, and not `MGR`;
  - `MGR` can't be closed;
  - `--note` is required for `grant`, `team close` and `pool add`;
  - `pool init` needs 0 ≤ seed ≤ total;
  - `pool init` is refused if a pool or any account event already exists;
  - a `pool.json` that exists but can't be read is an error, never a reason to re-create the
    pool;
  - malformed lines in the logs are skipped.
- **"Spent"** in `status` = the charged GPU-minutes of all accounts. Uncharged (human) jobs are
  listed separately.
- **Trust:** as in v1, nothing stops an agent from setting another team's `AUTOINFER_TEAM` or
  cancelling another team's job. Now that accounts have consequences (closure), the manager's
  spot-checks and `gpu history` are the safeguard.
- **Grants and opens are atomic:** checking the pool and appending the event happen under one
  lock, so two concurrent grants can't overdraw the pool.

## Files (changes from v1)
```
$AUTOINFER_RUN/gpu/
  pool.json         # {"total": 1500, "seed": 20, "created": ...}  (pool init / pool add)
  accounts.jsonl    # append-only events, one JSON object per line:
                    #   {"t", "event": "open",  "team", "type", "by"}
                    #   {"t", "event": "grant", "team", "kind", "minutes", "by", "note"}
                    #   {"t", "event": "close", "team", "by", "note"}
                    #   {"t", "event": "pool_add", "minutes", "by", "note"}
  accounts.lock     # flock for open / grant / close / pool_add
  ledger.jsonl      # as in v1: job, team (the account charged, or null), agent, label, start,
                    # end, gpu_minutes (charged), wall_minutes, exit_code, timed_out,
                    # cancelled, charged. No "round".
  (budget.json is gone)
```
Balances are always computed from `accounts.jsonl` and `ledger.jsonl`, never stored, so they
can't drift.

## Decided against (for now)
- **Two 1-GPU jobs side by side.** They share PCIe, host memory and CPU, so each could distort the
  other's measurements. Accuracy comes first.
- **Priorities.** First in, first out. Revisit if one team floods the queue.
- **An automatic starting grant for exploit teams in the tool.** The manager grants it right
  after opening the account; the tool doesn't know grant sizes.

## Tests
`flavors/v2/tools/tests/test_gpu.sh` uses fake jobs (`sleep`), runs under `.cache/gpu2test/` and
needs no GPU memory. It still calls `nvidia-smi` for the "outside this run" check.

**First version** (2026-11-08): 38 checks covering:
- **Roles:** only the human sets up the pool, and only the manager opens, grants and closes.
- **Accounts:** the seed is automatic, team ids are never reused, and grants are refused when the
  pool can't cover them or the account is unknown or closed.
- **Admission and charging:**
  - which account a job is charged to;
  - the admission checks and the re-check at the front of the queue;
  - timeouts are charged.
- **Output:** the balance is printed at the start and end of every job, and the low-balance
  notice appears.
- **Closing a team:** its running and waiting jobs are cancelled and charged, and its unspent
  minutes return to the pool.
- **`history`**, and concurrent grants not overdrawing the pool.

**A review subagent then checked the implementation against this spec.** Its findings were fixed
and 20 checks were added:
- a job that ignores SIGTERM: the charge is capped at the timeout;
- a double cancel: charged once, no traceback, processes gone;
- the refused state at the front of the queue, and `gpu wait` on such a job;
- `history MGR`, and structured `history --json`;
- team id validation and required notes;
- the cancel reason and closed-account output after `team close`;
- `pool init` refused after a lost `pool.json`;
- an odd line in `accounts.jsonl`.

All 58 checks pass.
