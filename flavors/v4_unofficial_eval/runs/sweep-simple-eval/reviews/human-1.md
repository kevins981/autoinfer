# Review human-1

by human (team human); reviewer: codex gpt-6.1-sol / high; started 2026-11-15T22:16:16.985516+00:00

## human (22:16:16)

Please review record T1-003 (bank/T1-003/) before it is filed. It concludes that the host-memory all-reduce is not worth pursuing (at best a tie with NCCL). Check whether the conclusion follows from the evidence: the measurements, what they were compared against, and the code.

## reviewer's commands (22:19:07)

- `/bin/bash -lc "pwd; rg --files -g 'AGENTS.md' -g 'TASK.md' -g 'PROFILING.md' -g 'record.md' -g 'change.diff' bank/T1-003 task . | head -100"`
- `/bin/bash -lc 'cat task/TASK.md task/PROFILING.md bank/T1-003/record.md'`
- `/bin/bash -lc 'cat bank/T1-003/change.diff'`
- `/bin/bash -lc "rg --files bank/T1-003; rg --files -g AGENTS.md -g '*T1-003*' -g '*host*' -g '*allreduce*' ."`
- `/bin/bash -lc 'rg --files workspaces/leader-1.hostar; ls gpu/jobs/13 gpu/jobs/17; cat gpu/jobs/13/output.log gpu/jobs/17/output.log'`
- `/bin/bash -lc "ls -la; cat task_definition.md assumption_ledger.md bank/T1-003/serve_args.txt bank/T1-002/record.md bank/T2-001/record.md; rg -n 'hostar|host.all.reduce|T1-003|43 us|52 / 92' gpu/jobs shared_kb.md reports local_kb workspaces --glob '*.md' --glob '*.log' --glob '*.sh' --glob '*.py' --`
- `/bin/bash -lc 'cat gpu/jobs/13/job.json gpu/jobs/17/job.json; ls -l workspaces/leader-1.hostar/bench; rg --files agents/leader-2.w1 agents/leader-1.w1 | head -100; cat agents/leader-2.w1/allreduce_results.md'`
- `/bin/bash -lc "rg --files workspaces/leader-1.hostar -g '"'!vllm/**'"' | rg -v '/vllm/' | head -100; git -C workspaces/leader-1.hostar/vllm diff --stat; rg -n 'hostar|graph|NCCL|nccl|time|barrier|Event' workspaces/leader-1.hostar/bench --glob '*.py' --glob '*.sh' --glob '*.log'"`
- `/bin/bash -lc 'nl -ba workspaces/leader-1.hostar/bench/test_hostar.py; cat workspaces/leader-1.hostar/bench/job.sh workspaces/leader-1.hostar/bench/job2.sh; nl -ba agents/leader-2.w1/bench.py; nl -ba agents/leader-2.w1/hostar.py'`
- `/bin/bash -lc "nl -ba workspaces/leader-1.hostar/vllm/vllm/distributed/device_communicators/host_all_reduce.py; sed -n '30,65p' workspaces/leader-1.hostar/bench/srv/server.log; cat workspaces/leader-1.hostar/env.sh; cat workspaces/leader-1.hostar/bench/job_output.txt workspaces/leader-1.hostar/bench`
- `/bin/bash -lc "nl -ba gpu/jobs/13/output.log; nl -ba gpu/jobs/17/output.log; nl -ba bank/T1-003/record.md; nl -ba workspaces/leader-1.hostar/vllm/vllm/distributed/device_communicators/cuda_communicator.py | sed -n '80,145p;380,455p;565,645p'; rg -n 'def destroy|host_ar|def close' workspaces/leader-1`
- `/bin/bash -lc "rg --files agents/leader-1 agents/leader-2 | rg 'prof.*(sqlite|csv|txt)"'$|steps.csv$'"'; nl -ba workspaces/leader-1.hostar/vllm/vllm/distributed/device_communicators/pynccl.py | sed -n '1,28p;150,230p'; rg -n 'tensor_model_parallel_all_reduce|all_reduce\\(' workspaces/leader-1.hostar`
- `/bin/bash -lc "ls -l workspaces/leader-1.ar/bench; rg -n 'host|NCCL|nccl|48|graph' workspaces/leader-1.ar/bench --glob '*.md' --glob '*.txt' --glob '*.json' --glob '*.jsonl' | head -100; sed -n '450,480p;835,875p' workspaces/leader-1.hostar/vllm/vllm/distributed/device_communicators/cuda_communicato`
- `/bin/bash -lc "ls -lh agents/leader-1/prof_t1002 agents/leader-2/prof_cg2048 task/baseline/nsys_c16; head -3 agents/leader-1/prof_t1002/steps.csv agents/leader-2/prof_cg2048/steps.csv; nl -ba agents/leader-2.w1/nccl.jsonl | head -6; nl -ba agents/leader-2.w1/custom.jsonl | head -4; rg -n 'HostAllRed`
- `/bin/bash -lc "python -B - <<'PY'
import json
from pathlib import Path
rows=[json.loads(l) for l in Path('agents/leader-2.w1/nccl.jsonl').read_text().splitlines()]
simple=next(r['res'] for r in rows if r['label']=='Simple')
ref=simple['48'][1]
print('NCCL Simple graph us:', {m:simple[m][1] for m in `

## reviewer (22:19:07)

I would file T1-003 as **inconclusive**, not as evidence that host-memory all-reduce is “at best a tie” or unworthy of further work. Parking it is a prioritization decision; the measurements do not establish a negative result.

1. **There is no matched NCCL control.** The 56.6 µs reference comes from [T2’s separate measurement](agents/leader-2.w1/nccl.jsonl:4). Its [harness](agents/leader-2.w1/bench.py:12) captures 30 operations per graph using in-place `torch.distributed.all_reduce`; [T1’s harness](workspaces/leader-1.hostar/bench/test_hostar.py:22) captures 10 out-of-place operations, synchronizes after each timed replay, and reports only rank 0. These differences leave launch skew and harness effects unresolved. The engine itself uses out-of-place PyNCCL. Neither job 13 nor 17 measures NCCL alongside hostar, and neither measures server throughput.

2. **The conclusion omits favorable results from its own logs.** [Job 17](gpu/jobs/17/output.log:13) reports **47.4 µs at M=48 with 16×512**, approximately 16% below the cited NCCL reference. Even the default configuration’s [51.8 µs](gpu/jobs/13/output.log:5) is approximately 8.5% below it. These unpaired comparisons do **not** prove a win, but they contradict “at best a tie” as an established finding. Conversely, default M=48 results reach 95 µs, demonstrating unresolved variability. “Loss at M≥96” also overgeneralizes: the measurements cover 96 and 128, the cited NCCL benchmark has no M=128 result, and this implementation falls back to NCCL above 128 rows.

3. **The code has a buffer-reuse race when sizes change.** In [the kernel](workspaces/leader-1.hostar/vllm/vllm/distributed/device_communicators/host_all_reduce.py:82), each block’s memory range depends on the current tensor size, but it waits only for the peer’s **same block number** to finish the previous call. With eight blocks, changing from 16 to 48 rows makes new block 0 overwrite regions previously owned by blocks 0, 1, and 2. Waiting for peer block 0 does not establish that peer blocks 1 and 2 have finished reading. This is a correctness issue identified by inspection, not a reproduced GPU failure. The numerical tests synchronize between sizes and replay constant-size graphs, so they do not cover it. The arithmetic is appropriate; the blanket “exact” validity claim is premature.

4. **Some supporting claims need correction.** The [server log](workspaces/leader-1.hostar/bench/srv/server.log:41) confirms hostar enabled on both ranks, so the three-chat smoke test is legitimate integration evidence. It establishes neither benchmark performance nor comprehensive correctness. Graph numerical checks cover only M={16,48,96,128}; the other listed sizes are eager-only. The integrated default host mapping is **2,654,208 bytes, about 2.53 MiB**, rather than the prototype’s roughly 40 MB.

Before drawing a performance conclusion, resolve the size-transition race, then compare hostar with out-of-place PyNCCL using the same harness and both ranks’ timings. An in-server on/off comparison should hold the NCCL setting and serving flags constant, since this patch also includes the Simple default.

Suggested filing language:

> Standalone timings are variable and lack a matched NCCL control. Some configurations suggest a small-tensor advantage, but server throughput remains unmeasured. Correctness testing passes the exercised cases; a size-transition synchronization issue remains unresolved. Parked pending further investigation.

I used read-only inspection and CPU calculations; no GPU work or file changes.
