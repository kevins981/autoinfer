# v1: Budget and stopping

## Budget
- **Unit: GPU-minutes** (costs per kind of test: [evaluation.md](evaluation.md)). CPU time is not
  budgeted. Tokens and wall time are not budgeted in v1, only reported.
- **Per team, per round.** The human sets `CONFIG.gpu_minutes_per_team_per_round` for each run.
  The manager may adjust it per assignment (more for a promising direction, less for a long shot).
- **Passed as text** in the leader's task ("budget: 40 GPU-minutes"), so the interface stays
  `run(task: str)`. The leader divides it among its workers (in their task text) and its own
  official evals, and monitors it.
- **The manager's own GPU work** (initial profile, re-profiling the champion, integration) is
  unbudgeted for now, but still goes through the queue.
- A round takes at least (teams × budget) of GPU wall time, since only one model-loading job runs
  at a time (e.g. 3 teams × 40 min ≥ 2 h per round). The manager tells the human this when the
  run starts.

## Enforcement: in the GPU queue
The GPU queue is a wrapper we write (shared infrastructure). All GPU work goes through it; agents
don't use the GPU directly. **v1 trusts agents to use the queue** (nothing technically stops them).
- Every job carries a team id and a timeout. The queue starts it only if the team's remaining
  budget covers the timeout, and charges the actual time when it finishes. (The timeout also kills
  hung servers, and a team can never overspend.)
- `gpu_queue status` shows each team's remaining budget and per-worker spending. Leaders use it to
  monitor; they don't keep their own accounts.
- Only the team total is enforced; the leader's per-worker split is guidance.
- Out of budget: new jobs are rejected with "budget exhausted".
- FIFO. No priorities, and no moving budget between teams mid-round, in v1.

## Stopping
- **Leader** (returns its report, ending `leader.run` for this round) whenever it judges it has
  reached a useful checkpoint: the direction is explored enough, worked, or is dead. It doesn't
  have to finish the whole direction in one go: its report can propose continuing (in "what I'd
  do next"), and the manager may assign it again next round. Also stops when the manager says so.
  Running out of budget doesn't force a stop: the leader can still do work that doesn't need the
  GPU (analysis, code, reviewing, writing the report), just no new GPU jobs. Unused budget doesn't
  carry over.
- **Round**: ends when every leader has reported (barrier).
- **Run** (the whole invocation): ends at `CONFIG.max_rounds`, when the manager judges there's
  nothing worth running (and tells the human why), or when the human stops it. Total GPU use is
  already bounded by max rounds × teams × budget, so there is no separate total budget.
