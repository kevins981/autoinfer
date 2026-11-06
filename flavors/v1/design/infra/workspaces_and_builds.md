# Workspaces and builds (spec)

**Implemented 2026-11-06: `tools/ws`** (+ `ws setup-task` and `ws list`), with the task's
`tasks/<task>/workspace.json` (repo, baseline commit, build targets) and `tasks/<task>/env.sh`.
The run folder links its task: `$AUTOINFER_RUN/task -> tasks/<task>`. Tested: create in 1.6 s
(~200 MB), `import vllm` loads the workspace's code, workspaces are isolated, `diff` includes new
files, `create --from` applies a record's diff, cleanup when creation fails.

How a leader gives each worker its own copy of the engine, how a change gets built, and how the
same mechanism produces a clean build for an official eval. Shared by all flavors. Written for
the qwen task (vLLM); a task supplies the specifics (see "What a task provides").

## Goals
- A workspace is cheap to create (seconds, not the ~1 h full vLLM build).
- Python-only changes need no build at all; C++/CUDA changes rebuild only what changed.
- Workspaces never interfere with each other or with `qwen38_opt/`.
- An official eval can rebuild any record from `baseline + change.diff` alone.

## Facts (qwen task, measured 2026-11-06)
- `qwen38_opt/vllm` is an editable install. The compiled extensions are `.so` files inside
  `vllm/vllm/` (the main one, `_C_stable_libtorch`, holds the CUDA kernels). Source: ~9K tracked
  files. A cmake+ninja build of `_C_stable_libtorch` alone takes ~9.5 min cold; its build folder is
  ~130 MB.
- **`PYTHONPATH=<workspace>` overrides the editable install** (tested): `import vllm` loads the
  workspace's code, so workspaces need no `pip install`.

## Worktrees are full, independent copies
A git worktree is a separate folder with its own **complete checkout of every file** and its own
branch. Edits in one worktree never touch another, so three workers get three worktrees and work
in parallel without interfering. The only shared thing is the git history database (`vllm.git`),
which is read-only for us. Cost per worktree: ~150 MB of source, plus ~130 MB of build folder
once it has been built.

## Layout
```
.cache/tasks/<task>/              # materialized once per task, shared by all runs (not in git)
  vllm.git/                       # our own clone of vLLM at the baseline commit (read-only)
  prebuilt/                       # the baseline's compiled .so files
  ccache/                         # shared compiler cache
runs/<run>/workspaces/<name>/     # one per worker (and per official eval)
  vllm/                           # git worktree of vllm.git at the baseline commit
                                  #   + the start record's diff applied
  build/                          # this workspace's incremental cmake/ninja folder (created on
                                  #   the first C++ build)
  env.sh                          # source this: shared env + PYTHONPATH=<workspace>/vllm + caches
  BRIEF.md ...                    # whatever the leader copies in
```

## Commands (one script, `ws`)
| Command | What it does |
|---|---|
| `ws create <name> [--from <record id>]` | `git worktree add` at the baseline commit, apply the record's `change.diff` (default: none = the baseline), symlink the prebuilt `.so` files into it, write `env.sh`. Seconds. |
| `ws build <name>` | Rebuild the compiled extensions from the workspace's source: cmake+ninja in `build/`, shared ccache, then point the workspace's `.so` files at the new build. Only needed after C++/CUDA changes. Cold ~10 min the first time in a workspace (less with a warm ccache), then incremental. |
| `ws diff <name>` | The workspace's change against the baseline commit (tracked + new files, no build outputs): this becomes the record's `change.diff`. |
| `ws rm <name>` | Remove the worktree and its build folder. Rarely used (see below). |

**Workspaces are kept.** They stay on disk after the experiment, as part of the record (the
exact tree that was run, build outputs, logs, scratch files). The leader deletes one only when
it is known to be broken or contributed nothing. This goes into the leader's role prompt. Disk is
not a concern: ~300 MB each on a 1.8 TB disk.

The leader runs `create`/`diff`/`rm`; workers run `build` (and edit, run, test) inside their own
workspace. Servers started from a workspace use its `env.sh`, so they run that workspace's code.

## Official eval
`ws create eval-<record> --from <record>`, `ws build` if the diff touches compiled code, then the
frozen harness. Because the workspace is rebuilt from `baseline + diff`, anything the worker did
outside the diff (stray files, manual tweaks, a different `.so`) can't leak into the official
number.

## Environment
- One shared Python env for every workspace, used **read-only**: v1 reuses `qwen38_opt/env`
  (nothing is installed into it; the editable vLLM there is overridden by `PYTHONPATH`).
- Model weights: read from `qwen38_opt/.cache` with `HF_HUB_OFFLINE=1` (no download, no writes).
- Every other cache (ccache, triton, torch inductor, vLLM compile cache, TMPDIR) goes to
  `autoinfer/.cache`, never to `qwen38_opt/.cache` or `$HOME`.
- If sharing the env causes trouble, fall back to a copy of it under `autoinfer/.cache`.

## What a task provides
The engine repo and baseline commit, how to build (the cmake targets, flags), which compiled files
to prebuild, and the env and weights to use. For the qwen task: vLLM `4527515cd7`, target
`_C_stable_libtorch` (+ the others in `vllm/vllm/*.so`), `TORCH_CUDA_ARCH_LIST=8.9`.

## Not in v1
Sharing build folders between workspaces, per-workspace Python envs, containers or sandboxes
(workers are trusted to stay in their workspace).

## To check while implementing
- ccache hit rate across worktrees at different paths (`CCACHE_BASEDIR`, relative paths).
- That a `ws build` of only `_C_stable_libtorch` is enough for the changes agents make, and how
  to detect that a diff needs other targets.
- Whether the vLLM compile cache must be keyed per workspace (code changes must invalidate it).
