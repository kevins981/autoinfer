# LOCAL_KB leader-2 (T2, explore)
## Facts measured
- job 1 (agents/leader-2/mb1/peak.py): bf16 8192^3 GPU0 ~144 TF (locked clock), GPU1 158; int8 538/569 TOPS; fp8 284/301.
  cuBLAS bf16 model shapes: M=48 ~41 TF (memory-bound); M=256-320 125-140 TF; M=1024 139-151.
  Green contexts (torch.cuda.green_contexts) WORK on sm_89/driver 580; gate_up M=4096 scales linearly 32/64/96 SMs (35.8/71.5/106.5 TF).
  copy(80 SM green) + gemm(48 SM green) concurrently: 163 ms vs serial 183, ideal 136 -> interference exists.
## Napkin
- per-rank GEMM FLOPs/token 24.3 GF (12.15 GB fp8 weights). Mixed step ~94 ms; ~300 rows GEMM ~58 ms at ~126 TF; rest ~36 ms (GDN prefill, attn, exposed AR).
- Decode Marlin M=48 13.5 ms: memory floor ~12-14, compute floor 8.2 ms -> tensor ~60% busy in Marlin, idle in AR (6.2 ms).
- Prefill marginal per trial ~400 x (94-25.5) = 27 s of ~97 s. Decode work ~69 s.
- Direction: prefill on a green-context SM partition concurrently with decode steps (NanoFlow/Bullet-like spatial overlap). Ideal T=max(27/f, 69/s(1-f)).
## Team
- T3 does self-distilled MTP draft head (agreed split, stacks with ours).
## Progress
- T2-001 filed (jobs 1, 6). Proposal T2-1 funded: 80 start (98.2 left). Gates: TTFT p99 <= 1609 ms, TPOT p99 <= 55.3 ms. Manager wants M1 ABBA before the gate.
- job 6: decode on plain stream sms=128-f + prefill green f: f=32 x1.28, f=48 x1.31 (vs 128-f grid), pre 32/45/59 TF at f=32/48/64. green/green bad; priorities serialize graphs.
- w1 task 2 = M1 shadow-lane probe (agents/leader-2/task_w1_2.md), budget 18.
- M2 design sketch: scheduler "lane-pending" state; new request prefill dispatched to worker lane (async, green ctx stream, own buffers/AR instance); rank-0 reports lane completions (+ first token, drafts) in a later step's output; scheduler then adds it to decode; decode stream waits on lane-done event on each rank. Batch queue (core.py step_with_batch_queue) is FIFO; avoid touching it by reporting lane completions inside decode-step outputs.
- job 10 smoke: shadow lane works in-server (green ctx + graphs + hostar), 26-49 TF/rank at f=48, +190 MiB/rank.
- T2-002 M1 FAILED gate: f=48 ITL x1.65 (+16.4 ms/step), lane 36 TF. f=32 x1.46/22 TF, f=64 x1.85/45 TF. No payoff on these numbers. Ablation (task 3) running to find the cause.
- T2-003 ablation (agents/leader-2.w1/abl): gemm-only ITL 42.9 (lane 42.8 TF) = full; copy-only 29.7 (+4.4); rank0-only 39.4; decode sms=128 43.3. => not PCIe, not cross-rank amplification, not grid. Suspect GIL/CPU (lane thread Python launches). Task 4: graph-captured lane.
- T2-004: graph lane (CPU-free) ITL still 41.9 -> GPU-side. ITL ~ 25.3*128/(128-f): decode step SM-proportional in-server; spatial sharing zero-sum (0.61+0.29=0.90). T2-1 CLOSED. ~34 GPU-min used.
- T2-005 anatomy (job 19): mixed steps: Marlin near roof; levers = AR exposure 17-44 ms/step, host gap ~17 ms in pt<150 steps, elementwise 3-6 ms, mixed GDN decode update 4.0 vs 2.3. Each a few %. Task 6 (CPU-only) sizing them.
- T3 closed draft head (no fine-tune gain, KB60); on last chance.
- Pivot thinking: no exploit team in this run -> an exact "mixed-step diet" package with an official eval has real value even if ~+3-6%.
- WARNING: ws create --from <record with csrc changes> symlinks the prebuilt BASE _C lib until ws build: T2-005 profile was not the champion. Always ws build before measuring.
- T2-006 (job 22, built): mixed levers sum +1-2% only; AR at transport floor (0.61us/tok+15.5us); decode 23.8 vs floor ~22. Resync requested: recommended closing T2 (b) or small package (a) stacked on T3-2. Used 44.6/100.
