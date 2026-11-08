# v2 announcements and reminders (spec)

**Status:** agreed with the human on 2026-11-08. Implemented in `flavors/v2/tools/agent`
(`announce`, `remind`) and `flavors/v2/tools/_remind_hook.py`; see "Tests".

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
- Appends it to `$AUTOINFER_RUN/announcements.md` with a timestamp, so agents started later can
  catch up. The leader prompts point to this file.
- Prints who received it and who was skipped (stopped agents, and interactive sessions, which
  have no inbox).

**Content:** the manager's prompt gives two formats, each pointing to the details:
- **Funding:** `team T3 (explore) received 80 GPU-min to work on <direction, one line>;
  proposal: <path>`. The same format covers extensions.
- **New champion:** `new champion T2-014 from team T2: 1010.3 tok/s (+7.2% over T1-042); how:
  bank/T2-014/`.

The manager may announce other things too, e.g. a team being shut down and its write-up.

## Reminders: keeping each role's key message alive
**Why:** agents run for hours to tens of hours, and instructions given only at the start fade,
above all after context compaction. The explore leaders' persona is what matters most: the
long-horizon incentive and their autonomy.

**Reminder files:** `flavors/v2/agents/reminders/<role>.md`, only for the leaders
(`leader_explore`, `leader_exploit`). Each is a few lines restating the role's core. Workers and
the manager don't need reminders (human decision, 2026-11-08), so they have no file, and a role
without a file gets none.

There are two ways a reminder reaches an agent.
1. **Periodically, sent by the manager:** `agent remind [--role R] [NAME...]` (manager or human).
   - It sends every live headless agent (or the named ones, or one role) its role's reminder as an
     inbox message.
   - The header is `[reminder; no reply needed]`, and the body starts with `<system message>:`.
   - The manager decides when. The run's config gives a default interval (`reminder_interval`,
     e.g. every 2 hours), placed in the manager's prompt.
   - Not a PostToolUse hook: that would be too frequent and intrusive.
2. **After every context compaction, through a hook.** A `SessionStart` hook with matcher `compact`
   runs `_remind_hook.py`.
   - It reads `AUTOINFER_ROLE` and prints the role's reminder as `additionalContext`, so the
     agent sees it right after its context is summarized.
   - `new_run` adds the hook to both `agent_settings.json` and `manager_settings.json`.

## Tests
`flavors/v2/tools/tests/test_announce_remind.sh` uses fake agents (a `meta.json` plus a live
`sleep` as their process), so no Claude sessions are started. On 2026-11-08: 27 of 27 checks
passed. Covered:
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
