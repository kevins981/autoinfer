# Profiling data: what exists and how to query it

The baseline's profiles are in `baseline/` (summary: `baseline/PROFILE.md`). The summaries are
a starting point, not all there is: the raw files hold every kernel launch and every hardware
metric, and you can query them however you like (Python is easiest). Make new profiles of any
workspace with `scripts/profile_nsys.sh` and `scripts/profile_ncu.sh` (GPU jobs); they produce
the same kinds of files. `source <workspace>/env.sh` (or the task's `env.sh`) first.

## nsys (Nsight Systems): the timeline. `baseline/nsys_c16/`
- `profile.nsys-rep`: the report (64 ShareGPT prompts at c=16, ~27 s, both GPUs, CUDA graphs
  traced per kernel). `profile.sqlite`: the same data as a SQLite database (`nsys export --type
  sqlite`; `profile_nsys.sh` writes it too). `kern_sum.csv`, `breakdown.txt`: per-kernel totals
  and time by category (`scripts/kernel_breakdown.py`).
- Main tables: `CUPTI_ACTIVITY_KIND_KERNEL` (one row per kernel launch: start, end [ns],
  deviceId, streamId, shortName/demangledName → ids in `StringIds`, grid/block, registers,
  shared memory, graphNodeId), `CUPTI_ACTIVITY_KIND_RUNTIME` (CUDA API calls on the CPU),
  `CUPTI_ACTIVITY_KIND_MEMCPY`, `NVTX_EVENTS`, `TARGET_INFO_GPU`.
- Example (Python, stdlib):
  ```python
  import sqlite3
  c = sqlite3.connect("baseline/nsys_c16/profile.sqlite")
  # time per kernel name and GPU
  c.execute("""SELECT k.deviceId, s.value, COUNT(*), SUM(k.end-k.start)/1e6 AS ms
               FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
               GROUP BY k.deviceId, s.value ORDER BY ms DESC LIMIT 20""").fetchall()
  # the kernel sequence of GPU 0 in a time window (e.g. one decode step), gaps between kernels
  c.execute("""SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON s.id = k.shortName WHERE k.deviceId = 0
               AND k.start BETWEEN ? AND ? ORDER BY k.start""", (t0, t1)).fetchall()
  ```
  Use `demangledName` for full template names (e.g. to tell Marlin GEMM shapes apart by grid
  size or template arguments). Baseline facts from this: each GPU is ~93-95% kernel-busy;
  per GPU, Marlin ~13 s and all-reduce ~6.5 s of the 27 s window.
- CLI: `nsys stats --report <name> profile.nsys-rep` (`nsys stats --help-reports` lists
  reports, e.g. cuda_gpu_kern_sum, cuda_api_sum, cuda_gpu_trace); `nsys analyze`.
  nsys is `/usr/local/cuda-12.6/bin/nsys` (2024.5).

## ncu (Nsight Compute): per-kernel hardware metrics. `baseline/ncu_marlin/`
- `marlin_c16.ncu-rep`: 12 Marlin launches (3 layers x 4 FP8 GEMMs) during c=16 decode, full
  metric set (~1700 metrics per kernel: DRAM/L2/L1 traffic and throughput, tensor-pipe use,
  stall reasons, occupancy, ...). `marlin_c16_key.csv`: the key metrics. `marlin_c16_src.ncu-rep`:
  built with -lineinfo, source imported (per-source-line and per-SASS stall samples).
- Python: `ncu_report` (Nsight Compute's API):
  ```python
  import sys, os; sys.path.insert(0, os.environ["NCU_PYTHON"])   # set by env.sh
  import ncu_report
  ctx = ncu_report.load_report("baseline/ncu_marlin/marlin_c16.ncu-rep")
  r = ctx.range_by_idx(0)
  for i in range(r.num_actions()):
      a = r.action_by_idx(i)                      # one profiled kernel launch
      a.metric_by_name("gpu__time_duration.sum").as_double()     # ns
      a.metric_by_name("dram__bytes_read.sum").as_double()       # bytes
      # a.metric_names(): every metric available
  ```
- CLI: `ncu --import <rep> --page details` (the GUI's sections as text), `--page raw --csv`
  (every metric, one row per kernel), `--page source --csv --print-source cuda,sass` (per line,
  needs a -lineinfo report). `python scripts/ncu_hotlines.py <rep> [launch] [top]`: hottest source
  lines. ncu is `$ENV_PREFIX/bin/ncu` (2025.3, works without root; counters may only work on
  GPU 0: `--devices 0`).

## Making new profiles (GPU jobs, through `gpu run`)
- `scripts/profile_nsys.sh <ws> <out> [num_prompts] [serve args]` (~3 min): nsys + SQLite +
  kernel summary + breakdown of that workspace's server.
- `scripts/profile_ncu.sh <ws> <out> <kernel regex> [skip] [count] [ncu args]` (~5 min for a few
  kernels): ncu on chosen kernels during c=16 decode. Use `--devices 0`.
- Profiling slows the server down: use profiles for where the time goes, not for throughput.
