# Run test-6: how to read it

AutoInfer v2 on one task: make vLLM serve Qwen3.8-27B-FP8 (TP=2, 2x RTX 4090) faster, measured as
output tok/s on ShareGPT at concurrency 16 (`tasks/qwen38-27b-fp8-2x4090/` in this repo). A manager
agent funds teams with GPU time; each team has a leader agent and worker agents. Every experiment
is a record in `bank/`.

- **Result:** 631.4 tok/s (BASE, stock vLLM) -> **1212.4 tok/s (T52-005), +92%**.
- **Seeded from test-5:** this run started from test-5's state (942.2 tok/s, champion T1-042). Records
  `T1-*` are test-5's; this run's teams are T2-T55. See `history/README.md`.
- **Dates:** 2026-11-08 to 2026-11-12. GPU pool 2000 GPU-min (raised from 1500 for T52/T53); see `gpu/ledger.jsonl`.

## Start here
1. `champion.md`: the current best record, its official numbers, why it is valid, and the champion path.
2. `reports/final.md`: the run's summary (written before T50-T53; `champion.md` and `journal.md` are current).
3. `graph.md`: every direction and record as a tree (★ on the champion's path, ✗ failed, ○ null).
4. `shared_kb.md`: what the run learned about the system, as numbered facts (KB1-KB131), each
   citing its records. Later entries may correct earlier ones.
5. `journal.md`: the manager's decisions in order (openings, grants, rulings, promotions, closures).

## Champion path
BASE 631.4 -> (test-5) T1-042 942.2 -> T9-005 948.9 -> T14-004 956.6 -> T27-002 967.3 ->
T29-004 1045.8 -> T29-006 1062.1 -> T31-003 1088.9 -> T37-002 1142.2 -> T45-005 1173.4 ->
T50-002 1181.3 -> T52-005 1212.4

## Teams
One team = one leader (+ workers) with its own GPU account. Team types (prompts in
`flavors/v2/agents/leader_<type>.md`; all types: `flavors/v2/design/team_types.md`; versions below):

| Teams | Type | What it is |
|---|---|---|
| T2-T31, T34 | explore (now called exploit-long) | propose one long-horizon direction with a forecast, funded in stages |
| T32-T33 | external | a direction drawn from a deck of published work (`scout/`) |
| T35-T38 | moonshot | no run history; large-upside bet funded like a venture |
| T39-T49 | research | goal: novel insights for future teams, speedup a bonus (T39-T41 closed before working; T42-T45 and T46-T49 are two sets of four prompt framings) |
| T50 | exploit | make the champion faster now, no proposal |
| T51 | exploit-long | as explore |
| T52-T53 | exploit-long | same, with a direction set by the human ("megakernels"), Opus leader and worker |
| T54-T55 | research-handoff | built from an outside research memo (`research_handoff_1113.md`); both closed after triage, no GPU time |

## Folders
| Path | What |
|---|---|
| `teams/T<n>.md` | one page per team: type, grants, results, why it closed |
| `proposals/T<n>-<k>.md` | a team's k-th proposal (idea, forecast, roadmap, GPU asked) |
| `reports/T<n>-writeup.md` | a team's closing write-up; `T<n>-<k>.md` = its k-th resync report |
| `bank/<record>/record.md` | one experiment: `builds_on`, `performance`, hypothesis, change, results, conclusion, validity |
| `bank/<record>/change.diff` | the full code change vs vLLM commit 4527515cd7 (cumulative, not vs the parent) |
| `bank/<record>/serve_args.txt` | extra `vllm serve` flags |
| `bank/<record>/official/<n>/results.json` | an official eval (throughput, latency, GSM8K, 128K gate) |
| `bank/<record>/notes/` | extra measurements and scripts a team kept |
| `local_kb/leader-<n>.md` | a leader's own working notes |
| `sparring/` | conversations with the sparring partner (a skeptical outside model) |
| `inspiration/T<n>-<k>.md` | inspiration cards drawn for a team |
| `scout/` | decks of published work for external teams |
| `gpu/ledger.jsonl` | every grant and GPU charge; `gpu/jobs/<id>/job.json` one GPU job (team, command, time) |
| `history/` | test-5's journal, ideas and reports |
| `task_definition.md`, `assumption_ledger.md`, `focus.md` | the rules agreed with the human (what counts as exact, allowed changes) |
| `config.md`, `funding_policy.md` | the human's run settings (team counts, models, grants) |
| `announcements.md` | messages the manager broadcast to all agents |
| `kickoff.md` | the manager's first message |

## Prompt and tool versions
The design (`flavors/v2/agents/`, `flavors/v2/tools/`) changed during the run. Repo commits in
effect at each manager start or resume:

| From | Commit | Change it brought |
|---|---|---|
| 2026-11-08 | a7db733 | start: explore teams, payoff bar 5% |
| 2026-11-09 | 5915cea | per-team model slots |
| 2026-11-10 03:37 | deff909 | external teams (scout decks) |
| 2026-11-10 15:39 | f776d0d | sparring partner; reopening closed teams |
| 2026-11-10 21:25 | 9c962cc | moonshot teams; speed-of-light page |

Later changes went in mid-run without a manager restart: research teams (2026-11-11), the rename
of explore to exploit-long and message delivery to a stopped manager (2026-11-11). `git log
flavors/v2` has the rest.

Not in git: workspaces (vLLM copies), agent transcripts, raw per-request benchmark outputs and
per-question GSM8K outputs.

## Reading a record
`performance` gives the headline number and how it was measured. "Official" numbers come from the
harness on a fresh inference server (two trials) and are the only ones that can make a champion.
"ABBA" or "same-job" means A/B runs interleaved in one GPU job, the run's test for small gains.
`builds_on` names the parent record; follow it to trace a lineage. Job ids (e.g. "job 431") are in
`gpu/jobs/`.
