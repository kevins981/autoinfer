# AutoInfer baseline flavor

A single-agent research loop, ported from karpathy/autoresearch: one agent edits the engine,
measures each commit with the task's official harness (`tools/experiment`), keeps it if it's
faster with all gates passing, and otherwise resets. No manager, teams, GPU queue or messaging.
It is the control arm for the multi-agent flavors (see `EVAL.md`).

A session started in this folder is the run's researcher: its instructions are
[program.md](program.md); read it now and follow it. (`tools/new_run` starts it with program.md
already in its system prompt; this file covers a session started by hand, e.g. to resume.) If
`$AUTOINFER_RUN` is not set, ask the human which run (`runs/<run>/`).

- `program.md`: the agent's instructions (Karpathy's, adapted to the task).
- `config_default.md`: run limits (max experiments, max hours) and model; copied to each run.
- `tools/`: `new_run` (create a run + start the agent), `experiment` (official eval of the
  current commit), `gpujob` (ad-hoc GPU job), `ws` (workspaces: copied from v1), `_gpu.py`.
- Run folder: `runs/<run>/`: `task` link, `config.md`, `run.json`, `results.tsv`, `final.md`,
  `workspaces/main/` (code tree on branch `autoresearch/<run>`), `experiments/<n>-<commit>/`,
  `gpu/ledger.jsonl` + `gpu/jobs/`, `agents/researcher/meta.json` (for `resources.py`).
