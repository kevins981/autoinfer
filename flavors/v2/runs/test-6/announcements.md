## 2026-11-08T23:20:39+00:00 (manager)
team T3 (explore) received 80 GPU-min to work on a self-distilled MTP draft head (self-generated prompts + target's greedy replies, 2-step unrolled loss); proposal: proposals/T3-1.md

## 2026-11-08T23:42:45+00:00 (manager)
team T2 (explore) received 80 GPU-min to work on a spatial prefill/decode overlap (prefill lane on a green-context SM partition, concurrent with decode); proposal: proposals/T2-1.md

## 2026-11-09T00:35:20+00:00 (manager)
team T3 (explore) received 20 GPU-min (last chance) to continue after T3-2's measured payoff fell below the bar; resync report: reports/T3-1.md

## 2026-11-09T00:55:13+00:00 (manager)
team T2 (explore) closed voluntarily at resync 1 (prefill lane gate failed, no bar-clearing pivot); write-up: reports/T2-writeup.md. Team T4 (explore) opened with a 20 GPU-min seed, leader-4.

## 2026-11-09T01:00:40+00:00 (manager)
team T3 (explore) closed voluntarily at resync 2 (T3-2 below the bar at ~+3%, no bar-clearing direction); write-up: reports/T3-writeup.md. Team T5 (explore) opened with a 20 GPU-min seed, leader-5.

## 2026-11-09T01:15:20+00:00 (manager)
team T4 (explore) closed voluntarily (AR/GEMM overlap premise failed, KB62); write-up: reports/T4-writeup.md. Team T6 (explore) opened with a 20 GPU-min seed, leader-6.

## 2026-11-09T01:42:11+00:00 (manager)
team T6 (explore) received 40 GPU-min to work on Marlin's per-slab weight re-streaming at M=49-256 (parallel slabs / single-pass W8A16) and using the freed decode slack; proposal: proposals/T6-1.md

## 2026-11-09T01:48:48+00:00 (manager)
team T5 (explore) received 80 GPU-min to work on deeper exact speculation (k=3 now, confidence-budgeted K_MAX=4 next) on the built stack; proposal: proposals/T5-1.md

## 2026-11-09T03:04:21+00:00 (manager)
team T5 (explore) closed voluntarily at resync 1 (deeper speculation ceiling ~+1-2%, KB63); write-up: reports/T5-writeup.md. Team T7 (explore) opened with a 20 GPU-min seed, leader-7.

## 2026-11-09T03:13:38+00:00 (manager)
team T6 (explore) closed voluntarily (exact Marlin slab policy, ~0 in-server; piggyback killed; KB64); write-up: reports/T6-writeup.md. Team T8 (explore) opened with a 20 GPU-min seed, leader-8.

## 2026-11-09T03:31:49+00:00 (manager)
team T7 (explore) closed voluntarily (prefill-size AR hiding failed its gate, KB65); write-up: reports/T7-writeup.md. Team T9 (explore) opened with a 20 GPU-min seed, leader-9.

## 2026-11-09T03:46:54+00:00 (manager)
team T8 (explore) received 60 GPU-min to work on a 2-warps/scheduler FP8 decode GEMM (M<=48) replacing Marlin's warm path; proposal: proposals/T8-1.md

## 2026-11-09T04:02:13+00:00 (manager)
team T9 (explore) received 60 GPU-min to work on hiding prefill-size all-reduces with a two-chunk copy-engine ping-pong in mixed steps; proposal: proposals/T9-1.md

## 2026-11-09T04:08:36+00:00 (manager)
team T8 (explore) closed voluntarily (2-warp decode GEMM failed its M1 gate, KB68); write-up: reports/T8-writeup.md. Team T10 (explore) opened with a 20 GPU-min seed, leader-10.

## 2026-11-09T04:29:35+00:00 (manager)
team T10 (explore) closed voluntarily after its seed (no bar-clearing direction; FP8-MMA not exact on sm_89, KB69); write-up: reports/T10-writeup.md. Team T11 (explore) opened with a 20 GPU-min seed, leader-11.

## 2026-11-09T04:46:43+00:00 (manager)
team T11 (explore) closed voluntarily after its seed (AR-window prefill lane failed its premise, KB70); write-up: reports/T11-writeup.md. Team T12 (explore) opened with a 20 GPU-min seed, leader-12.

## 2026-11-09T05:49:56+00:00 (manager)
team T12 (explore) closed voluntarily (cheaper-drafter package failed G1, KB71); write-up: reports/T12-writeup.md. Team T13 (explore) opened with a 20 GPU-min seed, leader-13.

## 2026-11-09T05:54:53+00:00 (manager)
new champion T9-005 from team T9: 948.9 tok/s (+0.7% over T1-042 942.2; confirmed by two same-job ABBAs, all B > all A); how: bank/T9-005/ (copy-engine overlap of prefill-size ARs in mixed steps). The bar for proposals is now >= 5% over 948.9.

## 2026-11-09T05:57:11+00:00 (manager)
team T9 (explore) received 20 GPU-min to continue mixed-step AR overlap (CE payload codec + B-lite) after its win T9-005; resync report: reports/T9-1.md

## 2026-11-09T06:56:39+00:00 (manager)
team T9 (explore) closed voluntarily after delivering champion T9-005 (+0.7%); remaining increments measured dead; write-up: reports/T9-writeup.md. Team T14 (explore) opened with a 20 GPU-min seed, leader-14.

## 2026-11-09T07:07:13+00:00 (manager)
team T13 (explore) closed voluntarily after its seed (producer-fused AR failed both gates, KB72); write-up: reports/T13-writeup.md. Team T15 (explore) opened with a 20 GPU-min seed, leader-15.

## 2026-11-09T07:29:37+00:00 (manager)
team T14 (explore) received 20 GPU-min (last chance) to land the measured Marlin-policy win on the champion (T14-002) via ABBA + official; resync report: reports/T14-1.md

## 2026-11-09T07:30:32+00:00 (manager)
team T15 (explore) closed voluntarily after its seed (no bar-clearing direction; KB73); write-up: reports/T15-writeup.md. Team T16 (explore) opened with a 20 GPU-min seed, leader-16.

## 2026-11-09T08:12:51+00:00 (manager)
new champion T14-004 from team T14 (kernel by T6): 956.6 tok/s (+0.8% over T9-005 948.9; same-job ABBA all B > all A); how: bank/T14-004/ (T6's exact Marlin slab policy for M>48 on T9-005). The bar for proposals is now >= 5% over 956.6 (1004.4).

## 2026-11-09T08:13:15+00:00 (manager)
team T14 (explore) closed voluntarily after landing champion T14-004; write-up: reports/T14-writeup.md. Team T17 (explore) opened with a 20 GPU-min seed, leader-17.

## 2026-11-09T08:15:54+00:00 (manager)
team T16 (explore) closed voluntarily after its seed (hybrid decode AR failed its gate, KB74); write-up: reports/T16-writeup.md. Team T18 (explore) opened with a 20 GPU-min seed, leader-18.

## 2026-11-09T08:35:08+00:00 (manager)
team T18 (explore) closed voluntarily after its seed (no bar-clearing direction; KB75); write-up: reports/T18-writeup.md. Team T19 (explore) opened with a 20 GPU-min seed, leader-19.

## 2026-11-09T09:01:54+00:00 (manager)
team T17 (explore) closed voluntarily after its seed (premise was a write-flush artifact; METHOD CORRECTION KB76: cold GEMM benches must flush with reads); write-up: reports/T17-writeup.md. Team T20 (explore) opened with a 20 GPU-min seed, leader-20.

## 2026-11-09T09:11:22+00:00 (manager)
team T19 (explore) closed voluntarily after its seed (new-request path below the bar; in-situ step model KB77); write-up: reports/T19-writeup.md. Team T21 (explore) opened with a 20 GPU-min seed, leader-21.

## 2026-11-09T09:39:23+00:00 (manager)
team T20 (explore) closed voluntarily after its seed (LL decode AR exact but capped by the prefetch coupling, KB78); write-up: reports/T20-writeup.md. Team T22 (explore) opened with a 20 GPU-min seed, leader-22.

## 2026-11-09T10:15:42+00:00 (manager)
team T22 (explore) closed voluntarily after its seed (decode prefetch bounded at ~+3.5%, ABBA -0.9%; KB79); write-up: reports/T22-writeup.md. Team T23 (explore) opened with a 20 GPU-min seed, leader-23.

## 2026-11-09T11:13:44+00:00 (manager)
team T23 (explore) closed voluntarily after its seed (overhead package below the bar; KB80); write-up: reports/T23-writeup.md. Team T24 (explore) opened with a 20 GPU-min seed, leader-24.

## 2026-11-09T11:34:18+00:00 (manager)
team T24 (explore) closed voluntarily after its seed (warm decode GEMM premise failed; KB81); write-up: reports/T24-writeup.md. Team T25 (explore) opened with a 20 GPU-min seed, leader-25.

## 2026-11-09T13:03:58+00:00 (manager)
team T25 (explore) closed voluntarily (GPU1-aware rebalance null in situ, KB82); write-up: reports/T25-writeup.md. Team T26 (explore) opened with a 20 GPU-min seed, leader-26.

## 2026-11-09T13:38:10+00:00 (manager)
team T26 (explore) closed voluntarily after its seed (register-direct decode GEMM killed; KB83); write-up: reports/T26-writeup.md. Team T27 (explore) opened with a 20 GPU-min seed, leader-27.

## 2026-11-09T14:10:20+00:00 (manager)
Human ruling: generic fine-grained prefix caching for the hybrid model is ALLOWED (task_definition.md), provided nothing is sized or tuned to the benchmark's system prompt or prompts. Team T21 (explore) received 30 GPU-min (first tranche of 80) for a hybrid prefix-state cache + prompt-as-draft feeding; proposal: proposals/T21-1.md

## 2026-11-09T15:50:36+00:00 (manager)
new champion T27-002 from team T27: 967.3 tok/s (+1.1% over T14-004 956.6); how: bank/T27-002/ (PCA-screened FP8 draft lm_head, ruling R1). Team T27 (explore) closed after its win; write-up: reports/T27-writeup.md. Per the human, no replacement team opens.

## 2026-11-09T21:23:43+00:00 (manager)
Run resumed. Human ruling during the pause: FP8 activations in the checkpoint's own W8A8 scheme (dynamic per-token-group-128 e4m3 activations vs 128x128 block-scaled FP8 weights) are now ALLOWED, with conditions: nothing coarser/static/lower, fp32 across K-blocks, full GSM8K-1319 within 2.0 points of a same-job bf16-activation reference; see task_definition.md (Correctness). Teams T28 (explore, slot A, leader-28) and T29 (explore, slot B, leader-29) opened with 20 GPU-min seeds each. Champion T27-002 967.3 tok/s; bar for proposals 1015.7.

## 2026-11-09T21:34:41+00:00 (manager)
team T29 (explore) received 80 GPU-min to work on checkpoint W8A8 (FP8 activations, per-token-group-128) GEMMs for mixed/prefill steps; proposal: proposals/T29-1.md

## 2026-11-09T21:37:43+00:00 (manager)
team T28 (explore) received 80 GPU-min to work on a small-M W8A8 FP8 decode GEMM that makes exact deeper speculation (k=3/4) pay; proposal: proposals/T28-1.md

## 2026-11-09T23:28:14+00:00 (manager)
new champion T29-004 from team T29: 1045.8 tok/s (+8.1% over T27-002 967.3; +64.5% vs BASE); how: bank/T29-004/ (checkpoint W8A8 FP8 GEMMs at M>96 on Marlin-repacked weights; decode unchanged). New bar for proposals: 1098.1 (+5%). Team T29 (explore) received 80 GPU-min to continue the W8A8 mixed-step direction; resync report: reports/T29-1.md

## 2026-11-10T00:27:05+00:00 (manager)
new champion T29-006 from team T29: 1062.1 tok/s (+1.6% over T29-004 1045.8; +67.0% vs BASE); how: bank/T29-006/ (exact bundle on T29-004: v2 GEMM tiling, fused norm/silu->quant producers, GDN glue trim; all bitwise). New bar for proposals: 1115.2.

## 2026-11-10T00:28:28+00:00 (manager)
team T29 (explore) closed after two wins (T29-004, T29-006: +9.8% together); write-up: reports/T29-writeup.md. Team T30 (explore, slot B) opened with a 20 GPU-min seed, leader-30.

## 2026-11-10T00:58:06+00:00 (manager)
team T28 (explore) closed after its last chance (small-M FP8 decode GEMM / deeper speculation: k=3 null, e4m3 at k=2 -1.84%; KB94-96); write-up: reports/T28-writeup.md. Team T31 (explore, slot A) opened with a 20 GPU-min seed, leader-31.

## 2026-11-10T01:32:55+00:00 (manager)
team T30 (explore) received 65 GPU-min to work on two-wave layer-pipelined mixed steps (hiding the all-reduce the per-layer join leaves exposed); proposal: proposals/T30-1.md

## 2026-11-10T01:53:54+00:00 (manager)
team T31 (explore) received 80 GPU-min to work on a register-direct FP8 (W8A8) decode GEMM that cashes the all-reduce's L2 prefetch; proposal: proposals/T31-1.md (re-forecast in teams/T31.md)

## 2026-11-10T02:57:53+00:00 (manager)
Human ruling (2026-11-10): the FP8-activation accuracy A/B now uses GSM8K-250 (gsm8k.py default n), not all 1319 questions; still in the same job as a bf16-activation reference of the same code, within 2.0 points. The sharing rules and addenda stay (read 'GSM8K-1319 A/B' as 'the FP8-activation GSM8K A/B'); A/Bs already run at n=1319 stay valid. See task_definition.md (Correctness).

## 2026-11-10T03:25:06+00:00 (manager)
new champion T31-003 from team T31: 1088.9 tok/s (+2.5% over T29-006 1062.1; +71.2% vs BASE); how: bank/T31-003/ (register-direct FP8 decode GEMM at M<=96 on 5 of 6 shapes). Team T31 (explore) closed after its win; write-up: reports/T31-writeup.md. Per the human, no replacement team opens.

## 2026-11-10T03:27:31+00:00 (manager)
team T30 (explore) closed (layer-pipelined waves: the mixed-step AR is latency-bound, KB100); write-up: reports/T30-writeup.md. The run is now paused per the human; champion T31-003 1088.9 tok/s.

## 2026-11-10T15:46:34+00:00 (manager)
team T32 (external) closed: no measured path to +5% (proposals/T32-4.md); run paused

## 2026-11-10T16:03:08+00:00 (manager)
team T32 (external) received 80 GPU-min to work on host-free layer-pipelined waves with captured GDN prefill; proposal: proposals/T32-5.md

## 2026-11-10T16:54:32+00:00 (manager)
team T34 (explore) opened with 20 GPU-min seed

## 2026-11-10T18:46:28+00:00 (manager)
team T34 (explore) received 60 GPU-min to work on hybrid copy-engine+SM mixed-step all-reduce; proposal: proposals/T34-1.md

## 2026-11-10T22:52:00+00:00 (manager)
new champion T37-002 (team T37): 1142.2 tok/s official (+4.9% cross-job over T31-003 1088.9; same-job ABBA +1.34%); k=3 with two-slot GDN state; how: bank/T37-002/

## 2026-11-11T02:23:22+00:00 (manager)
teams T42, T43 (research) received 20 and 10 GPU-min; proposals: proposals/T42-1.md (certified greedy lm_head decisions), proposals/T43-1.md (row-selective AR / decision depth)

## 2026-11-11T02:42:57+00:00 (manager)
team T44 (research) received 15 GPU-min to work on paced SM PCIe traffic duplex; proposal: proposals/T44-1.md

## 2026-11-11T10:33:18+00:00 (manager)
new champion T45-005 (team T45): 1173.4 tok/s official (+2.7% over T37-002 1142.2; same-job ABBA +3.83%); gate_up at 48<M<=96 via w8a8_marlin; how: bank/T45-005/

## 2026-11-11T13:43:32+00:00 (manager)
team T50 (exploit) opened with 100 GPU-min (20 seed + 80 start); champion T45-005 1173.4 tok/s

## 2026-11-11T13:44:00+00:00 (manager)
team T51 (exploit-long) opened with 20 GPU-min seed

## 2026-11-11T14:49:02+00:00 (manager)
new champion T50-002 from team T50: 1181.3 tok/s (+0.67% over T45-005 cross-job, +1.1% same-job); how: bank/T50-002/. Also: team T51 (exploit-long) received 25 GPU-min for a TP-shape GEMM kernel step; proposal: proposals/T51-1.md

## 2026-11-11T14:54:34+00:00 (manager)
team T50 (exploit) closed after champion T50-002; write-up reports/T50-writeup.md

## 2026-11-11T14:58:13+00:00 (manager)
team T51 (exploit-long) closed; write-up reports/T51-writeup.md

## 2026-11-11T21:34:21+00:00 (manager)
teams T52 and T53 (exploit-long, human-set direction: megakernels) opened with 20 GPU-min seed each; champion T50-002 1181.3

## 2026-11-11T22:28:16+00:00 (manager)
team T52 (exploit-long) received 50 GPU-min to work on chunk-pipelined mixed-step AR+GEMM layer kernels (megakernels); proposal: proposals/T52-1.md

## 2026-11-11T22:39:24+00:00 (manager)
team T53 (exploit-long) closed; write-up reports/T53-writeup.md

## 2026-11-11T23:45:23+00:00 (manager)
new champion T52-005 from team T52: 1212.4 tok/s (+2.6% over T50-002, +2.8% same-job); how: bank/T52-005/

## 2026-11-12T00:10:45+00:00 (manager)
team T52 (exploit-long) closed after champion T52-005; write-up reports/T52-writeup.md

## 2026-11-13T20:10:27+00:00 (manager)
teams T54 and T55 (research-handoff, memo research_handoff_1113.md) opened with 30 GPU-min seed each; champion T52-005 1212.4

## 2026-11-13T20:17:20+00:00 (manager)
team T55 (research-handoff) closed after triage; write-up reports/T55-writeup.md

## 2026-11-13T20:18:12+00:00 (manager)
team T54 (research-handoff) closed after triage; write-up reports/T54-writeup.md

## 2026-11-13T21:32:47+00:00 (manager)
teams T56 and T57 (sweep) opened with 100 GPU-min each (20 seed + 80 start); champion T52-005 1212.4

## 2026-11-13T21:48:13+00:00 (manager)
team T58 (sweep) opened with 100 GPU-min (20 seed + 80 start); champion T52-005 1212.4

## 2026-11-13T22:45:19+00:00 (manager)
team T56 (sweep) closed; write-up reports/T56-writeup.md, sweep reports/T56-sweep.md

## 2026-11-13T22:48:05+00:00 (manager)
new champion T57-004 from team T57: 1225.5 tok/s (+1.08% over T52-005, +1.68% same-job); how: bank/T57-004/

## 2026-11-13T23:37:25+00:00 (manager)
new champion T58-001 from team T58: 1229.6 tok/s (+0.33% over T57-004, +0.72% same-job); how: bank/T58-001/

## 2026-11-13T23:58:41+00:00 (manager)
team T57 (sweep) closed after champion T57-004; write-up reports/T57-writeup.md

## 2026-11-13T23:59:43+00:00 (manager)
team T58 (sweep) closed after champion T58-001; write-up reports/T58-writeup.md

