# Modules: composable ways to shape a team's search

Team types and what each means: `team_types.md`.

Several mechanisms change how a team finds ideas. They are designed (or being moved) to be
**independent and composable**: any team type can get any combination, chosen per run in
`config.md`. That lets runs compare them one at a time.

| Module | What it does | Tool / file | Config | Today used by |
|---|---|---|---|---|
| Inspiration cards | general principles of where systems lose time + hardware/software facts, drawn at random | `inspire`, `inspiration/` | `inspiration cards per draw` | exploit, exploit-long, moonshot, research |
| External deck | a scout (no project knowledge) searches the web and writes ~15 cards of recent exact techniques | `scout`, `scout/<team>-<n>/` | `scout` | external |
| Sparring partner | a skeptical outside researcher (Codex, reads no files) that questions the beliefs behind a leader's reasoning, one question per turn; offered again at every closing decision | `spar`, `sparring/` | `sparring` | all leaders |
| History-free | the team doesn't read the run's history (KB, graph, reports, other teams); after a proposal it gets pointers to related records only | `leader_moonshot.md`, `agent announce` skips it | team type | moonshot |
| Venture funding | judged on the upside if it fully works and on the riskiest assumption's cheapest test, not on odds; staged grants | `funding_policy.md` "Moonshot teams" | team type | moonshot, research-handoff |
| Research memo | a memo of candidate optimizations written outside the run; replaces the cards; the team triages every idea first | `research handoff memo` config line | `research handoff memo` | research-handoff |
| Speed of light | a task-specific page of physical lower bounds (bytes, FLOPs, link traffic per step; no implementation history) | `tasks/<task>/SPEED_OF_LIGHT.md` -> `task/` | `speed of light` | moonshot, research, research-handoff |

Today "moonshot" bundles three modules (history-free, venture funding, speed of light) as a team
type. As runs compare modules, they should become per-team switches (e.g. an explore team with
speed of light, a history-free team with explore funding) rather than new team types. Rules for
that: each module is one prompt section or one tool, says nothing about the others, and is
switched on in config without editing role prompts.
