# Team types (v2)

One team = one leader (+ workers) with its own GPU account. The type sets the leader's prompt
(`agents/leader_<role>.md`), how the manager funds and judges it, and its defaults. The run's
`config.md` says how many teams of each type to open; per-run team ids are listed in each run's
`README.md`.

| Type (`gpu --type`) | Prompt | Goal | Where the direction comes from | Run history | Funding / judged on | Since |
|---|---|---|---|---|---|---|
| exploit | `leader_exploit.md` | make the champion faster now | the team's own choice; no proposal | full | seed + start grant; judged on wins (progress = a win) | v2 start |
| exploit-long | `leader_exploit_long.md` | longer-horizon directions, judged on the speedup they deliver | the team proposes one (forecast, roadmap); the human may set one in config (e.g. "megakernels") | full | proposal must forecast >= 5% payoff; staged grants | v2 start (named "explore" until 2026-11-11) |
| external | `leader_external.md` | like exploit-long | a scout's deck of ~15 published techniques (`scout/`) | full | as exploit-long | 2026-11-10 |
| moonshot | `leader_moonshot.md` | much faster by an approach the current design doesn't take | the team's own choice, from first principles | **none** (pointers to related records only after its proposal) | venture: upside if it fully works >= 5%, riskiest assumption's cheapest test first; odds not gated; stages up to ~200 GPU-min | 2026-11-10 |
| research | `leader_research.md` | novel insights that help future teams; speedup a bonus | the team's own choice | full | judged on novelty and what it teaches | 2026-11-11 |
| sweep | `leader_sweep.md` | find the remaining opportunities systematically, then win | a framework (config `sweep framework`): the team analyzes the champion's design with it, compares with the history, ranks the untested opportunities | full | exploit-style: seed + start grant, judged on wins (small gains count); the analysis report comes first | 2026-11-13 |
| research-handoff | `leader_research_handoff.md` | speedups built from a research memo written outside the run | the memo named in config (`research handoff memo`); ideas are hypotheses and inspiration; stays within the memo's ideas | full | like moonshot, without the history check; triage of every memo idea first | 2026-11-13 |

Defaults per type (config can change them): inspiration cards for all but external,
research-handoff and sweep (deck / memo / framework instead); sparring for all; speed-of-light page for moonshot, research,
research-handoff and sweep (test-6 also gave it to exploit and exploit-long). Personas (`agents/personas/`)
and model slots are per-team options, not types. See `modules.md` for the modules.
