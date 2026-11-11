# Final report: run test-6 (qwen38-27b-fp8-2x4090)
(Updated 2026-11-11 after moonshot T38 and research teams T42-T49. Sections below cover T2-T49; the champion is T45-005.)

## Result
- **Champion: T45-005, 1173.4 tok/s official at c=16** (trials 1174.7/1172.1, CV 0.2%), +85.9% vs BASE 631.4, +24.5% vs the test-5 seed champion T1-042 (942.2). Same-job ABBA vs T37-002: +3.83%, all B > all A (job 415); cross-job +2.7%. TTFT p50/p99 149/831 ms, TPOT p50/p99 12.4/25.9 ms, GSM8K-250 94.8% (FP8 A/B vs bf16 activations +1.6 pt, job 416), MTP acceptance 2.678, 128K pass. bank/T45-005/, champion.md. Previous champion T37-002 1142.2.
- Path: T1-042 942.2 -> T9-005 948.9 -> T14-004 956.6 -> T27-002 967.3 -> T29-004 1045.8 -> T29-006 1062.1 -> T31-003 1088.9 -> T37-002 1142.2 -> T45-005 1173.4.
- GPU pool: 1500 GPU-min, 992.0 spent, 508.0 unspent (returned by closed teams). All accounts closed; nothing is granted.
- Bar for a +5% payoff over the final champion: 1232.1. Since T31-003, wins were T37-002 (moonshot) and T45-005 (research); no team found a +5% step.

## What produced the gains (all exact / validity-reviewed; see shared_kb.md, bank/)
1. T9-005 / T14-004: copy-engine all-reduce overlap in mixed steps + Marlin policy for mixed M (T9, T14).
2. T27-002: PCA-screened FP8 draft lm_head (KB84).
3. T29-004/006 (+8.1%, +1.6%): checkpoint W8A8 (dynamic per-token-group-128 e4m3 activations, per-128-K fp32 promotion) for M>96 steps on Marlin-layout weights, plus exact fused glue (KB87-93). Human rulings on FP8 activations and the GSM8K A/B (1319 -> 250 subset).
4. T31-003 (+2.5%): register-direct e4m3 decode GEMMs at M<=96 on the raw Marlin words, cashing the AR L2 prefetch; gate_up and the drafter stay Marlin (KB97-99).

## Moonshot teams (T35-T37; history-free leaders)
- T35 (run-ahead L2 weight streamer): refuted at milestone 1; GEMMs already stream at 0.8-0.92 TB/s (KB108). Screen negatives: small prefill token budgets, k=3 cross-job.
- T36 (confidence/load-chosen speculation depth): withdrawn before its milestone when past records answered the riskiest assumption; decode AR anatomy shows bandwidth-bound transport (KB109).
- T37 (static row-budgeted step -> k=3 depth unlock): **T37-002 champion 1142.2** (two-slot GDN state + k=3 with all 16 requests running, +1.34% same-job). Also found and fixed a correctness defect in the older two-slot code (garbage past a 1600-token page boundary; KB110, KB111); records using two-slot (T28-003/005) carry it. Scope of its stop: reports/T37-writeup.md (all-reduce volume floor, k=4, static mixed step are assumed, not established).

## Moonshot T38 and research teams T42-T49 (2026-11-11)
- T38 (moonshot, PP=2 engine + draft trees): PP prefill 0.51x TP but decode ratio 0.91-0.92 < 0.93 needed; no win (KB112, reports/T38-writeup.md).
- T39-T41 (research): closed unused (0 GPU-min).
- T42: certified exact target-lm_head (7-bit plane + rigorous enclosure); exact but ~0.8% of decode (KB121; task_definition ruling 2026-11-11). 21.9 GPU-min.
- T43: row-selective AR / decision depth; little rejection signal before the final layers (KB114). 4.3 GPU-min.
- T44: SM-driven PCIe traffic is not inherently half duplex when paced (KB115), but no variant helps the champion (KB116); a CE-AR reorder claim was withdrawn (KB117 superseded, KB119). 1.6 GPU-min.
- **T45 (WIN): row-pricing rule, a decode GEMM's cost above 64 rows is a step (KB118); gate_up band bench (KB120) -> T45-005 routes gate_up at 48<M<=96 through the existing W8A8 Marlin path: +3.83% same-job, champion 1173.4. Lesson: re-check GEMM dispatch whenever decode width changes (KB122).** 24.3 GPU-min.
- T46: host memory as a tier; exact host embedding frees 1.27 GB/GPU, link adds ~2.4% weight bandwidth, ceiling ~1% (KB125). 0.5 GPU-min.
- T47: asymmetric waves failed their gate (KB128); DRAM saturates from 16-32 SMs, so KB59's zero-sum is a property of today's kernel grid sizing (KB129). 5.7 GPU-min.
- T48: hardware compression comes in ratio steps 1x/2x/3.4x/8.6x and the real checkpoint has nothing compressible (KB126); oracle bound recomputed, k=4 would need ~0.115 ms/row above 64 (KB127). 1.3 GPU-min.
- T49: PP with one decode microbatch does not beat TP in simulation (KB123); compressible memory works on the 4090 (KB124). 0.4 GPU-min.
Write-ups: reports/T42..T49-writeup.md.

## Teams (T2-T34; grants/usage in teams/, `gpu history`)
- Wins: T9, T14, T27, T29 (two), T31. All other explore teams closed on measured limits or premises that failed (T2-T8, T10-T13, T15-T26, T28, T30; write-ups in reports/).
- Since the last pause: T32/T33 (external teams, slot C) and T34 (explore): no wins.
  - T32: deck assessment; sparring revised the verdict to "no measured path"; built and measured the captured GDN prefill core (exact, neutral alone), then found wave-eligible steps are only 14-17% of GPU time -> closed (reports/T32-1.md, KB107).
  - T33: lossless FP8 weight coding inside the decode GEMM and the overlapped weight-stream/MMA kernel: ~+1-1.5% overlap-only, coding loses once weights are L2-resident -> closed (reports/T33-1.md, KB101-103).
  - T34: hybrid copy-engine + SM all-reduce for big mixed steps reaches the PCIe duplex floor but saves only 3-39 us/AR; the champion transport is within ~11% of the floor -> closed (reports/T34-writeup.md, KB104-106).
- Incident: T34 job 345 (flag-polling hybrid AR microbench) crashed and left unkillable driver-teardown processes; a server reboot was needed (KB106, journal.md).

## What was learned (headline KB: shared_kb.md KB59-KB129)
- Decode GEMMs are at their L2/HBM floors with the current weight bytes; AR transport is within ~11% of the PCIe duplex floor in both decode and mixed steps; comm/compute co-run is zero-sum for SM-driven traffic with today's kernel grids (KB129: DRAM saturates from 16-32 SMs, so smaller-grid GEMMs might change this); deeper speculation (k=3) is AR/drafter-bound (~+2.4% cap); wave pipelining is capped by AR dependency latency and by the small share (14-17%) of time in big mixed steps.

## What is open (none individually clears +5%)
- Bytes: lossless codec floor 543 us vs hostar 620 us at 1100 rows (~+1%); denser codes near the 0.821 entropy bound are untested.
- hostar vs the DMA lane in exposed mixed-step ARs (-12% at 1100 rows in isolation; slower in situ in T30 arm C, unproven in server).
- Layer-pipelined waves: ~2% achieved; the 116 -> 90 ms replay gap is unexplained.
- Overlapped weight-stream/MMA decode GEMM stack (~+1-1.5%, agents/leader-33.w1/ASSET.md); captured GDN core asset T32_GDN_CAP (bank/T32-003).
- Fewer bytes per AR message for mixed steps (KB100); quant fusion at M<=96 and drafter in e4m3 (<=0.5% each, KB99); FP8-on-the-link all-gather (needs a human ruling).

## Open after the moonshots
- Tensor partition / collective schedule derived from minimum cross-GPU information (untested, T37 scope note); k=4 (~+1% est.); static mixed step (2-4% est.); small-op fusion (~1 ms/step gross, KB108); plus the earlier open leads above.

## Open after the research teams
- k=4: oracle +10.2% accepted tokens (KB127) but breaks even only at ~0.115 ms/row above 64; T45-005's W8A8 gate_up may bring the row price near that, untested.
- Decode GEMMs hand-sized for 32-64 SMs to free SMs for comm/prefill lanes (KB129); re-test spatial lanes on them.
- Other decode shapes with a bf16-Marlin step at M=64 (KB118/KB120/KB122); host tier (~1%), certified lm_head (~0.8%) as small exact stacking gains.

## Where things are
champion.md, bank/, graph.md, shared_kb.md, journal.md, teams/, reports/, announcements.md, proposals/, sparring/, task_definition.md.
