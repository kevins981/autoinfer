# v2 GPU queue: grants (spec draft)

**Status:** draft, under discussion with the human. Not implemented.

This is a delta from v1's queue ([../../../v1/design/infra/gpu_queue.md](../../../v1/design/infra/gpu_queue.md)).
Everything not mentioned here stays as in v1:
- one exclusive job at a time;
- FIFO;
- batch jobs (`gpu run --timeout`);
- timeout kills and cleanup;
- "other users come first" (exit 75);
- the team and agent come from the environment;
- an append-only `ledger.jsonl`.

What changes is the **budget model**. v1 gave per-round budgets; v2 has a fixed run pool and
per-team grants, with no rounds.

## Concepts
- **GPU-minute:** one minute of the GPU pair held exclusively, as in v1. Queue time is free.
- **Pool:** the run's total GPU-minutes, set by the human when the run starts. Every grant comes
  out of it.
- **Team account:** opened when a team is created, closed when it is shut down. Team ids are
  never reused: a replacement team gets a new id.
- **Grant:** GPU-minutes moved from the pool to a team account. Kinds:
  - `seed`: 20, automatic when the account opens;
  - `start`: 80;
  - `extension`;
  - `last-chance`.

  Only the manager makes grants, other than the seed.
- **Team balance** = the sum of the team's grants − the GPU-minutes of its finished jobs.
- **Pool balance** = pool total − the sum of all grants + the unspent balances of closed teams.
  This means a closed team's unspent time goes back to the pool.

## Commands
| Command | Who | What it does |
|---|---|---|
| `gpu pool init <minutes>` | `new_run` | Sets the pool. Done once. The human can raise it later with `gpu pool add <minutes>` |
| `gpu team open <team> --type explore\|exploit` | manager (via `agent start` of a leader) | Opens the account and grants the 20-min seed |
| `gpu grant <team> <minutes> --kind start\|extension\|last-chance --note "<why>"` | manager | Refused if the pool can't cover it |
| `gpu team close <team> --note "<why>"` | manager | Closes the account. Waiting jobs are cancelled and the unspent balance goes back to the pool |
| `gpu run ...` | anyone | As in v1. Admission is checked against the **team balance** |
| `gpu status` | anyone | Pool (total, granted, spent, free), then each team: type, open or closed, grants, used, balance, per-agent use; the running job; the queue |
| `gpu history <team>` | anyone | The team's grants and jobs over time: raw material for its track record |

## Rules
- **Admission:** a job starts only if the team's balance covers its timeout (exit 76 otherwise).
  It is checked at submission and again at the front of the queue, as in v1. A job is charged its
  actual time, never more than its timeout, so a balance can't go negative.
- **The balance on every job:** when a job starts and when it ends, `gpu run` prints the team's
  balance. Example: `team T2 (explore): 34.5 of 100 GPU-min left`. It is the built-in reminder.
- **Low-balance notice:** when a job ends with the balance under 25% of the team's latest grant,
  the output adds: `low balance: consider requesting a resync from the manager`.
- **Closed teams can't run jobs** (exit 76, `team closed`).
- **The manager's own jobs** (profiling a new champion, spot-checking cited jobs) are charged to a
  manager account, `MGR`. The manager funds it from the pool like any other grant, so its use is
  visible in `gpu status`.
- **Official evals are charged to the team that requests them,** as in v1. That includes the
  re-measurement after porting onto a new champion.

## Files (changes from v1)
```
$AUTOINFER_RUN/gpu/
  pool.json         # {"total": 1500}  (`gpu pool init` / `gpu pool add`)
  accounts.jsonl    # append-only: open / grant / close events
                    #   {"t":..., "event":"grant", "team":"T3", "kind":"extension",
                    #    "minutes":80, "by":"manager", "note":"..."}
  ledger.jsonl      # as in v1, without "round"
  (budget.json is gone)
```
Balances are always computed from `accounts.jsonl` + `ledger.jsonl`, never stored, so they can't
drift.

## Open questions
1. **Exploit teams' starting grant.** Is it automatic when the team opens (seed + 80 straight
   away, with no proposal), or does an exploit team also send a short proposal first?
   Suggestion: automatic. Exploit work is near-term, and a proposal step adds little.
2. **1-GPU jobs side by side.** Two microbenchmarks could run at once, one per GPU, which would
   roughly double throughput for explore teams' early work. But the two jobs share PCIe, host
   memory and CPU, so each could perturb the other's numbers.
   Suggestion: not in the first version, because measurement accuracy comes first.
3. **Who may grant.** v1 trusts agents; the queue only checks `AUTOINFER_ROLE=manager`.
   Suggestion: keep that trust model.
4. **Queue order.** FIFO stays. With no priorities, a team with a large grant can't jump the
   queue, and a team with a small grant isn't starved. Long queues just mean slow wall-clock for
   everyone.
   Suggestion: FIFO for the first version. Revisit if one team floods the queue.
