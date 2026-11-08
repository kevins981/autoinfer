# v2: open items

These are not agreed yet. Where the designer has a suggestion it is labelled; none of it is
decided.

## Incentives and funding
- **Manager reserve.** Suggestion: about 10% of the pool for the manager's own work (profiling a
  new champion, spot-checking cited jobs).
- **Pool size for the first v2 run.** Suggestion: about 1,500 GPU-min. Test-5 used 562 GPU-min.
- **Reports must cite evidence** (GPU job ids, records). Funding now depends on reports, so the
  pull to oversell is stronger. Should the manager spot-check by re-running cited jobs?
- **Where the track record lives.** Suggestion: a per-team file the manager reads, because
  leaders get restarted and replaced.
- **Should forecast accuracy affect trust in a team's later forecasts?**
- **Thinking time costs nothing in GPU.** Should there also be a wall-clock reporting interval?
- **Credit for enabling another team's win.**

## Asynchronous mechanics
- **GPU contention.** There is one GPU pair, and benchmarks need both. Could microbenchmarks run on
  one GPU each? The `gpu` tool currently allocates both.

## Next (human, 2026-11-08)
1. ~~**Broadcast contents.**~~ Done: `agent announce` and the formats in agents/manager.md.
   Original note: The manager sends these to every agent in real time, with no reply
   expected. Each one points to a file or folder with the details. Every agent can read every
   other agent's results and implementations; nothing is hidden.
   - **Funding:** "team X received N GPU-min to work on <direction>; details: <proposal file>".
   - **New champion:** "new champion <record> from team X (+Y%); how: <record folder>".
2. ~~**The reminder system, using hooks.**~~ Done: DESIGN section 12 and infra/announcements_and_reminders.md.
   Original note: It periodically re-states each role's key message to
   long-running agents, above all the explore leaders' persona: the long-horizon incentive and
   autonomy.

## GPU queue changes
- Charge every job to a team. Enforce the grant. Print the remaining balance on every job, which
  also serves as a built-in reminder.

## Agent settings
- Done: WebSearch and WebFetch are in the v2 allow list. v1's deny rule on fetching the autoinfer
  repository stays (in `confine()`).

## Other
- **Reminder prompts via hooks**, so long-running agents keep their instructions.
- **Judging v2:**
  - GPU-minutes per win by team type;
  - whether explore teams found anything outside the answer key;
  - how forecasts compared with outcomes.
- **A sceptic reviewer** for forecasts, instead of the manager.
