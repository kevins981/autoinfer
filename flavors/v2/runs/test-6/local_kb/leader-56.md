# LOCAL_KB leader-56 (T56, sweep, framework/)

## Setup
- workspace t56-champ = T52-005, built (ws build ok).
- job 483: nsys of T52-005, 160 prompts -> agents/leader-56/prof_champ/
- agents/leader-56/step_dump.py: dumps one step's GPU0 kernel sequence from an nsys sqlite.

## Calibration notes (T37-era trace agents/leader-38/prof_champ_node, k=3, M=64)
- GDN layer decode ~367 us: qkvz t31b 36 | conv+misc 8 | gdn replay 38 | out t31b 19 | hostar 52 | norm 3 | gate_up 95-103 (now bs_gemm ~82) | silu-quant 3 | down 54 | hostar 53 | norm 3
- drafter per step ~1.97 ms: level1 ~690 us (64 rows), level2/3 ~535 us; per level: GEMMs 268 us (fc bf16 68, qkv 48, o 22, gate_up 78, down 52) vs 238 MB bytes; 4 hostar ARs ~103 us; PCA screen ~107 us; small ops ~60 us.
- target lm_head 1028 us (T37; certified plane in T50+ ~0.75 ms).

## Sweep report written: reports/T56-sweep.md (ranked: 1 spatial lane, 2 mid-M GEMM, 3 drafter fc/embed table, 4 wave threshold via freed memory, 5 out_proj replication + FP8 AG (ruling))
## job 483 calibration: decode 24.7 ms (GEMM 12.3 / AR 6.66 / GDN 2.0 / small+attn 1.3 / lm_head 0.75 / drafter 2.06); mixed excess 24% wall, ~0.19 ms/prompt token under nsys (~0.165 official); small mixed step M=120: bs_gemm 20.4 ms, hostar 11.6 ms.
## spar T56-1 (lane): synthetic lane r is GEMM-only (no dependent ARs, no GDN/attn) -> go threshold for probe: (1-D)/r <= ~0.11 ms/token; cross-prompt pipelining not available (126/138 mixed steps carry 1 prompt); in-prompt waves could cover lane ARs (untested at ~260 rows); decode recovers all SMs when lane idle (primary ctx); resized permanent decode grids: to test separately. A failed probe kills only the tested arrangement.
## worker leader-56.w1: task 1 = lane probe (agents/leader-56/task_w1_1.md), allowance 15 GPU-min, record T56-001.
## split with T57 (agreed): T56 = in-server synthetic lane on today's grids (T56-001); T57 = microbench branch (lane vs streaming read / real decode kernels by family at 128-f SMs; resized grids must hold at 128). T57-001 (job 484): mixed steps GPU-bound, host wait <=1.5 s/41 s.
## T57-002 microbench lane (jobs 486/488): f=32/48/64 decode x1.20/1.40/1.62, lane 57/85/115 TF -> my metric (1-D)/r = 0.071/0.082/0.081 ms/token (passes 0.11 on GEMMs only; T57 used a different gate and dropped it). Drafter (incl. my fc/embed table) handed to T57.
## T57 point (accepted): e2e conversion of T56-001 must charge the slot waiting while its prompt sits in the slower lane (~0.6-1.0 of 16 slots at lane duty 0.4-0.6), i.e. longer TTFT dead time vs a ~70 ms joined mixed step.
## T56-001 lane: NO-GO (GEMM-only 0.137-0.156 ms/token vs joined 0.167; real lane worse). Next: ask manager ruling on out_proj/o_proj replication + FP8 AG; consider mid-M GEMM.
## waiting: manager ruling on cand 5 (msg sent). cand 2 needs new kernel on Marlin layout or 2nd weight copy (T33 ov4 is M=48 only) -> heavy.
## manager: cand 5 sent to the human; don't build. spar T56-1 turns 6-7: 2 ARs/layer necessary under TP; FP8 bytes floor; remaining = mixed steps. Task 2 (w1): sliced-t31b mid-M screen, T56-002, 6 GPU-min, gate >=15%/GDN layer at M=112-160.
## T56-002: sliced t31b NO-GO (+66-84%); bs at M=128 291 us/GDN layer vs floor ~200 -> new-kernel ceiling ~+1-2%.
## cand 5 ALLOWED by human (2026-11-13). Task 3 to w1 (agents/leader-56/task_w1_3.md): S1 premise microbench (AR vs AG M=64-512 + extra GEMM), gate >=40 us/layer at M=128-256; S2 build (VLLM_T56_REPL); S3 128K + ABBA. Records T56-003/004.
## T56-003 S1: bitwise ok; net -10/+10/+55/+109 us/layer at M=64/128/256/512 -> ~+0.3-0.8% e2e: NO-GO S2.
## spar T56-2: close is a budget judgment, not exhaustion. New mid-M GEMM ~1.5-2.2% conditional (weighting supported, odds assumed); replication ~0.3-0.8% resolvable by ABBA (~25-35 GPU-min after build); 2 ARs/layer necessary under the sharding; PP measured only for decode.
## CLOSED: reports/T56-writeup.md written; worker stopped.
