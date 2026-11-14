# v2 announcements and reminders (spec)

**Status:** agreed with the human on 2026-11-08. Implemented in `flavors/v3_framework/tools/agent`
(`announce`, `remind`) and `flavors/v3_framework/tools/_remind_hook.py`; see "Tests".

Both are built on v1's agent inbox ([../../../v1/design/infra/agent_runtime.md](../../../v1/design/infra/agent_runtime.md)).
`agent send` appends a user message to a headless agent's `inbox.jsonl`, and the agent sees it
as its next turn.

## Announcements: one message to every agent, in real time
**Why:** every team should know right away when a team is funded and when there is a new
champion, both for inspiration and because the bar moves. Nothing is hidden: every agent can read
the whole run folder (bank, workspaces, proposals, reports).

**`agent announce (TEXT | -f FILE)`** (manager or human):
- Delivers the text to every live headless agent of the run except the sender. Each copy gets the
  header `[announcement from <sender>; no reply needed]`.
- Appends it to `$AUTOINFER_RUN/announcements.md` as `## <timestamp> (<sender>)` followed by the
  text, so agents started later can catch up. The leader prompts point to this file.
- Prints who received it and who was skipped: agents not running, collapsed into one line, and
  interactive sessions, which have no inbox.
- **Workers receive announcements too.** The human asked for "every agent". Each announcement
  starts a turn for every idle agent, so announcements should stay rare: grants and champions.
- **Announcements and reminders are not tasks.** They don't change the agent's "last task" that
  `agent list` shows.

**Content:** the manager's prompt gives two formats, each pointing to the details:
- **Funding:** `team T3 (explore) received 80 GPU-min to work on <direction, one line>;
  proposal: <path>`. The same format covers extensions.
- **New champion:** `new champion T2-014 from team T2: 1010.3 tok/s (+7.2% over T1-042); how:
  bank/T2-014/`.

The manager may announce other things too, e.g. a team being shut down and its write-up.

## Reminders: keeping the explore persona alive
**Why:** agents run for hours to tens of hours, and instructions given only at the start fade,
above all after context compaction. The explore leaders' persona is what matters: the
long-horizon incentive and their autonomy.

**Only explore leaders get reminders** (human decision, 2026-11-08). Exploit leaders, workers and
the manager don't. There is one file, `flavors/v3_framework/agents/reminders/leader_explore.md`: a few lines
restating the role's core. A role without a file gets nothing.

There are two ways a reminder reaches an agent.
1. **After every resync, from the manager.** When the manager finishes a resync with an explore
   team, it runs `agent remind <leader>`.
   - The leader gets the reminder as an inbox message headed `[reminder; no reply needed]`, with a
     body starting `<system message>:`.
   - There is no time-based schedule. (`agent remind` also accepts `--role R`, or no argument for
     everyone with a reminder file.)
2. **After every context compaction, through a hook.** A `SessionStart` hook with matcher `compact`
   runs `_remind_hook.py`.
   - It reads `AUTOINFER_ROLE` and prints the role's reminder as `additionalContext`, so the
     agent sees it right after its context is summarized.
   - `new_run` adds the hook to both settings files. It prints nothing for roles without a
     reminder file.

## Robustness
- `meta.json` writes use a per-process temp file, so concurrent writers can't collide.
- A delivery failure to one agent is reported as skipped; the rest still get the message.
- Odd agent folders (no `meta.json`, corrupt, or no name) are ignored.
- `-f` and text together are refused.
- A missing `-f` file is a clean error.
- Naming yourself in `remind` is reported ("that is you").
- `remind --role X` with no matching agent says so.

**Trust:** as in v1, the role check relies on `AUTOINFER_ROLE`. An agent could unset it and pass
as the human.

## Tests
`flavors/v3_framework/tools/tests/test_announce_remind.sh` uses fake agents (a `meta.json` plus a live
`sleep` as their process), so no Claude sessions are started. On 2026-11-08: 36 of 36 checks
passed. Nine of them were added after a review subagent's findings: the stopped-agent report,
`last_task` left alone, odd `meta.json`, reminding yourself, `-f` with text, a missing `-f` file,
an unknown role, and concurrent announces with sends (all logged and delivered). Covered:
- **Who may use them:** only the manager or the human can announce or remind.
- **Announcements:**
  - delivery to live headless agents only: not the sender, not interactive sessions, not stopped
    agents;
  - the header, the body, the `announcements.md` log, `-f`, and an empty announcement refused.
- **Reminders:** all agents, `--role`, by name; roles without a file are skipped; the header and
  the `<system message>:` body.
- **The hook:** it outputs the role's reminder as `additionalContext`, and prints nothing for
  roles without a file (worker, manager) or without a role.
- **`new_run` settings:** they carry the compact hook next to the message guard.

**Not yet tested end to end:** a real Claude session compacting and receiving the hook's text.
