## Run start: seeded from test-5
This run starts from test-5's state (see history/README.md): bank with 46 records, champion T1-042
(942.2 tok/s official), test-5's task definition, assumption ledger, focus, shared KB and graph.
initialize() is skipped: the task definition agreed with the human stands. Team ids start at T2
(T1 is test-5's team, whose records are in the bank).

## 2026-11-08: teams opened
- Opened T2 and T3 (explore, per config.md: 2 explore, 0 exploit), 20 GPU-min seed each; pool free 1460.
- Leaders leader-2 (T2) and leader-3 (T3) started with champion T1-042 (942.2) as reference, no direction given; start grants (80) on a funded proposal.
- T3-1 funded (start 80): self-distilled MTP head; checks pass (parts marked measured/assumed, payoff ~+10% > 5% bar, 80 within reason). Ruled self-generated prompt data acceptable under the 2026-11-08 ruling; added a clarification to task_definition.md (human may override).
- T2-1 funded (start 80): spatial prefill/decode overlap via green contexts. Checks pass (GEMM co-run measured in T2-001; payoff +20% > 5%; 80 within reason with an M1 kill gate). Flagged the TTFT p99 gate (1609 ms) as a risk to the forecast. T2 and T3 directions are distinct and should stack.
- 2026-11-09 T2 progress: M1 kill gate failed (T2-002: decode x1.65 at f=48, lane 36 TF). T2 running one ablation before closing/pivoting; no action needed.
- 2026-11-09 T3 progress: closed T3-1 on its kill criterion (self-distillation gives ~0 held-out gain, T3-001..004). Pivot scoping confidence-budgeted verification on the remaining balance; a pivot doesn't reset the grant.
- 2026-11-09 T3-2 accepted (pivot, no new grant): confidence-budgeted MTP verification; payoff ~+6% narrowly clears the bar, E gain measured offline (T3-005), drafter cost assumed and measured first.
- 2026-11-09 T2 closed T2-1 (prefill lane) on measured evidence (T2-003/004). Added KB59 (spatial sharing zero-sum) and KB60 (stock MTP head near its learnable limit; T3-001..005) to shared_kb.md.
- 2026-11-09 Correction: I pointed T2 at test-5's anatomy files (agents/leader-2.w23/anat/), which are in another run's folder and off-limits; withdrawn. T2's own mixed-step anatomy (<=6 GPU-min) proceeds.
- 2026-11-09 Resync T3 #1: last chance, 20 GPU-min (T3-2 payoff fell to ~+4% on measured costs, T3-007). Leader may spend its balance on building T3-2 as a small exact win or on a new direction; a second last chance closes the team. reports/T3-1.md.
- 2026-11-09 T3 last-chance plan: build T3-2 as an exact ~+3% win (K_MAX=3, B=2*num_decode_reqs, on-GPU ranking); ~45 GPU-min incl. official; kill if ragged path needs a rewrite or ABBA < +1.5%.
- 2026-11-09 T2 found a ws tooling hazard (unbuilt ws from a record runs base kernels; KB61). T2-005 invalidated, rerun. Asked T3 to confirm its workspaces were built.
- 2026-11-09 T3-007 confirmed unbuilt (KB61); T3 resync #1's payoff rested on invalid costs. Will re-classify the last chance as an extension if re-measured payoff clears the bar.
- 2026-11-09 Resync T2 #1: CLOSE (voluntary, team's own recommendation) and replace with a new explore team (T4). No bar-clearing pivot after T2-1's gate failed; T2-006 sized remaining mixed-step items at +1-2%. Waiting for reports/T2-writeup.md before gpu team close.
- 2026-11-09 T2 closed (gpu team close; 55.4 returned, pool free 1335.4); leader-2 + w1 stopped. T4 (explore) opened, seed 20, leader-4 started (pointed at T2 write-up, KB59-61). Pool free 1315.4.
- 2026-11-09 Resync T3 #2: built re-measure keeps T3-2 below the bar (~+3.3%, K_MAX=3 no memory window, T3-008); last chance stands. T3 asks to close voluntarily; accepted; replace with T5 after write-up. Unharvested exact package (T3-2 + T2-006 items, ~+3-5%) flagged to the human: config has no exploit team.
- 2026-11-09 T3 closed (80.5 returned, pool 1395.9); leader-3 + w1 stopped. T5 (explore, config unchanged) opened, seed 20, leader-5 started. Pool free 1375.9.
- 2026-11-09 Ruling for T4: moving RMSNorm's per-row 1/rms past the next linear (fp32 epilogue) is allowed fp reordering, with conditions; added to task_definition.md (human may override).
- 2026-11-09 T4: premise microbench running (<=8 seed GPU-min): AR/GEMM overlap ceiling + first-chunk latency; proposal T4-1 follows, or kill if ceiling < ~15 us/AR.
- 2026-11-09 T4 premise failed (T4-001: AR/GEMM overlap saves <=5-8 us/AR vs 15 go bar; KB62). T4 asks to close voluntarily; accepted pending write-up; will replace with T6 (config unchanged).
- 2026-11-09 T5: probing fixed k=3 on the built stack (prefill-graph grid capped at 1024, KV 5.75 GiB) vs k=2 same job + 128K, ~14 seed GPU-min; challenges T3's unbuilt verify cliff and KB20.
- 2026-11-09 T4 closed (18.5 returned); leader-4 + w1 stopped. T6 (explore) opened, seed 20, leader-6 started. Pool free 1374.6. T5 probe amended (adds k=2 trimmed-grid control C; worker allowance 16).
- 2026-11-09 T6: direction = W8A16 GEMM that streams weights once for M=49-256 (+ chunked-prefill piggyback at the ridge); seed probe <=4 GPU-min (Marlin vs roofs per M). Coordinated with T5 (T6 GEMM side, T5 speculation). Pointed to KB14/28/33/42/44/46 (prior Marlin work).
- 2026-11-09 T6-1 funded 40 (as asked): Marlin slab re-streaming fix + decode-slack use. Payoff +5-6% just at the bar, low odds, partly shared with T5's k=3; funded in a small step with an early G1 gate.
- 2026-11-09 T5-1 funded (start 80): deeper exact speculation at M=64 (k=3 measured +3.1% at equal memory; payoff +7% T5 alone, central +3.5%, ~30%). Credit split with T6 agreed (T6 kernel at k=2, T5 speculation).
- 2026-11-09 T5 progress: fixed k=3 is a loss (T5-004, 0.965); T5-alone payoff now ~+2.5% (< bar), ~+8% only stacked on T6's kernel. T5 is holding GPU spend pending T6. Note for resync: T5's bar-clearing case now depends on T6.
- 2026-11-09 T6 progress: kernel milestone in microbench (M=72-120 -21..-35%, T6-003/004) but null at k=2 in-server and piggyback -30% (T6-005). T6-own payoff ~0-1%; value is enabling T5 at M=80.
- 2026-11-09 Resync T5 #1: CLOSE (voluntary): deeper speculation ceiling ~+1-2% (T5-004..006, KB63). Added KB63, KB64 (T6 slab policy). Will replace with T7 per config after write-up.
- 2026-11-09 T5 closed (38.9 returned; actual use 61.1, leader had reported ~41). leader-5 + w1 stopped. T7 (explore, config unchanged, no human reply on the exploit question) opened, seed 20, leader-7 started. Pool free 1273.6.
- 2026-11-09 Resync T6 #1: CLOSE (voluntary): kernel exact but ~0 in-server; piggyback killed (T6-007). 31.2 returned. KB64 extended, KB61 addendum. T8 (explore, config unchanged) opened, leader-8 started. Pool free 1284.8. Five teams closed without a win; human's answer on team type still pending.
- 2026-11-09 Human (relayed by the designer, session autoinfer2): explore only, keep going; config unchanged (2 explore, 0 exploit); replace closed teams with explore teams as usual. Question closed.
- 2026-11-09 T7 seed gate failed (T7-001, KB65); closes voluntarily before proposing. Will replace with T9 (explore) per human's instruction.
- 2026-11-09 T7 closed (15.2 returned); leader-7 + w1 stopped. T9 (explore) opened, seed 20, leader-9 started. Pool free 1280.0.
- 2026-11-09 T8-1 funded 60 (as asked): 2-warps/scheduler decode GEMM; premise measured (T8-002), payoff +5.6% just above the bar, gated by M1. Added KB66.
- 2026-11-09 T9-1 funded 60 (as asked): CE ping-pong of prefill ARs; premise measured (T9-001), payoff +5.6%. Added KB67 (partial correction of KB65: likely far-node artifact).
- 2026-11-09 Resync T8 #1: CLOSE (voluntary, M1 gate failed, T8-003, KB68); 78.1 returned. T10 (explore) opened, leader-10 started. Pool free 1218.1.
- 2026-11-09 T9 self-corrected forecast: payoff ~+4% (not +5.6%; R>=256 comm-only 15-23% of span per KB66/own bench), central +2%. Plan: M1, M2 ABBA; add-on (a) decode CE AR (KB68 leftover) after M2; (b) rank rebalance only with a ruling.
- 2026-11-09 T10 seed report: no bar-clearing proposal; T10-001 measured kill of FP8-MMA/Ozaki (KB69). Closes voluntarily; replace with T11 (explore) per human.
- 2026-11-09 T10 closed (20 returned); leader-10 stopped. T11 (explore) opened, leader-11 started. Pool free 1218.1.
- 2026-11-09 T11 closed after seed (T11-001 failed, KB70; 18.4 returned); write-up copied to reports/T11-writeup.md. T12 (explore) opened, leader-12 started. Pool free 1216.4.
- 2026-11-09 T12 seed report: no single bar-clearing direction (T12-001 CPU kill: free n-gram drafting E +0.7..2.7%). Leader suggested a stacked package of exact items (+6-9% if it works) needs an exploit owner; I clarified the policy allows an explore team to propose a stacked package (bar on payoff-if-it-works) and left the choice to leader-12.
- 2026-11-09 T12-1 approved (stacked package, +6-6.5% if it works, central +2%, ~15%); G1 on seed; start grant released at G1 per the team's gating. Ruled FP8 MTP fc inside drafter allowed (task_definition.md).
- T9-005 (S3c, job 100): same-job ABBA +1.2% (B 875.0/871.2/881.0 vs A 867.6/864.2/863.9; all B > all A); p99 TTFT 1178 vs 1264, TPOT 40.5 vs 41.5; acceptance equal; gsm8k-50 0.88; 128K pass, KV unchanged; +130 MB/rank -> prefill-graph cap 2048->1536 (T5-004: neutral). official_eval T9-005 launched.
- 2026-11-09 T9-005 official 948.9 (947.9/949.8) vs T1-042 942.2 (941.4/942.9): +0.7%, both trials above both; gates pass (GSM8K 92.8, acc 2.327, TTFT p99 1035, TPOT p99 35.2, 128K). Validity reviewed (scheduling-only, ba ULP reorder, cap 2048->1536 tested). Promotion pending a repeated same-job ABBA (400 prompts); granted T9 12 GPU-min for it.
- 2026-11-09 T12 G1 interim (job 105): draft step 0.573 ms (gate 0.65) with 32K vocab + replicated FP8 fc; acceptance 2.248 (first-N-ids subset covers only 82%; champion's 98K 93%) -> frequency-ranked subset from self-generated + repo text being built. A2: 4 host collectives per draft step; excess = embed AR skew + fc gather. Per-channel FP8 fc accepted (ruling amended). Granted +12 to finish G1.
- 2026-11-09 T12 closed (G1 failed, KB71; 21.5 returned). T13 (explore) opened, leader-13 started. Pool free 1193.9.
- T9-005 PROMOTED to champion (2026-11-09): official 948.9 (+0.7%), ABBA #2 (job 104, 400 prompts) +0.7% all B > all A, ABBA #1 +1.2%. First win of test-6.
- Resync 1 (2026-11-09, reports/T9-1.md): EXTEND 20 (beat the champion). Harvest codec + B-lite, ABBA-gated.
- 2026-11-09 leader-9, leader-9.w1, leader-13 stopped mid-turn ~05:58-06:00Z on the account's 5-hour usage limit (reset 06:30Z; reported by the designer). Resumed all three with agent send.
- 2026-11-09 T9 recommends voluntary close: codec killed (T9-007), B-lite <=~1% for 3-4 days (design review), rank rebalance ~600 MB/rank vs 360 MiB free, ~+2% payoff. Accepted; write-up requested; then close and open T14.
- 2026-11-09 T9 closed after its win (~42 returned); leader-9 + w1 stopped. T14 (explore) opened, leader-14 started (briefed with champion T9-005). Pool free 1196.0.
- T13 direction: producer-fused AR (posted host writes from row-parallel producer GEMM tiles); premise T13-001 <=6 min, gates: writes-only GEMM slowdown <=5% at >=18 GB/s, >=50% tiles done by 70% of kernel.
- 2026-11-09 T13 closed after seed (T13-001, KB72; ~19 returned). T15 (explore) opened, leader-15 started. Pool free 1195.2.
- 2026-11-09 T14-1 approved (stacked mixed-step package, payoff +5.5% just over 996.3, central +2%, ~20%); start 80 released on G1 pass (in-situ Marlin critical-path A/B on T9-005).
- 2026-11-09 Resync T14 #1: G1 failed (package ~+3.5% < bar); start 80 not released; LAST CHANCE 20 to harvest the measured exact Marlin policy on T9-005 (ABBA then official).
- 2026-11-09 T15 closed after seed (19.9 returned; KB73). T16 (explore) opened, leader-16 started. Pool free 1175.1.
- T14-003 ABBA (job 119, 14.0 min) incomplete: 4th start killed by timeout (cold compile on start 1). 1A 949.2, 2B 951.1, 3B 953.6 (+0.2/+0.5%); not a separation. Decision: run a complete ABBA on the remaining 13.8 with warm caches; top-up for official only if all B > all A; else close.
- T14-004 ABBA (job 120, 300 prompts): A 928.9/926.1, B 938.7/932.7 (+0.9%, all B > all A); acceptance equal. Granted +9 for official_eval T14-004.
- official_eval T14-004 refused (20-min reservation > 12.5 balance); granted +8.
- T14-004 PROMOTED to champion (2026-11-09): official 956.6 (+0.8% vs T9-005), ABBA job 120 +0.9% all B > all A; gates pass (GSM8K 94.0). Credit shared with T6 (kernel). Bar now 1004.4.
- 2026-11-09 T14 closed after its win (10.2 returned); leader-14 + w1 stopped. T17 (explore) opened, leader-17 started (briefed with champion T14-004, bar 1004.4). Pool free 1148.3.
- 2026-11-09 T16 closed after seed (15 returned; KB74). T18 (explore) opened, leader-18 started. Pool free 1143.3.
- 2026-11-09 T17-1 approved (roofline decode GEMM M<=64 + k=3; payoff +6.3% vs bar 1004.4, central +1.5%, ~15%); grants per milestone from M0 pass.
- 2026-11-09 T18 closed after seed (0 GPU used; KB75 TTFT/slot anatomy). T19 (explore) opened, leader-19 started. Pool free 1143.3.
- 2026-11-09 T17 closed after seed (M0 killed; KB76 method correction: write-flush inflated cold GEMM numbers in T5-002/T6-001; 19.3 returned). T20 (explore) opened, leader-20 started. Pool free 1142.6.
- 2026-11-09 T19 closed after seed (KB77; 5.0 returned; 12 min lost to a worker hook bug in job 132). T21 (explore) opened, leader-21 started. Pool free 1127.6.
- 2026-11-09 T20-1 approved (LL host AR + rebalance + small items; +5.9% vs bar 1004.4, central +2.7%, ~15%); start 80 on G1 pass; atomicity proof required before served use.
- 2026-11-09 T21 ruling request: enable generic non-default fine-grained prefix caching for the hybrid model (--prefix-match-unit, --enable-mamba-shared-prefix-checkpoint) so the chat template's shared 45-token default system prompt hits (17% of prompt tokens; ~+3.5%). task_definition (loophole: 'prefix caching tuned to the benchmark; generic default prefix caching is fine') doesn't settle it -> ESCALATED TO HUMAN. T21 may measure only (record marked PENDING RULING); must confirm the system text is the template default.
- 2026-11-09 T21-1 received, conditional on the prefix-caching ruling (on hold). Template check: system text is chat_template.jinja's default (reasoning_effort default 'xhigh'); client sends only a user message. Stock flags hit 144/104,718 tokens (T21-001).
- 2026-11-09 T20 closed after seed (KB78; 16.2 returned). T22 (explore) opened, leader-22 started. Pool free 1123.8.
- E1 (jobs 148-151, 1.0 min): in-situ decode GEMMs at L2-warm floor except down (+11.7 us) and qkvz (+3.6); cold-warm gap mostly first ~16 MB (latency); GDN warm floor row unreliable. Ceiling <=~1.6 ms/step (~+3.5%) < bar: no proposal. E2: in-situ ABBA (<=13 seed min) of prefetching the first 16 MB of down_proj from the AR before gate_up (exact, env-gated), expected +0.5..1.5%.
- 2026-11-09 T22 closed after seed (KB79; 5.4 returned). T23 (explore) opened, leader-23 started. Pool free 1109.3.
- 2026-11-09 T23-1 approved (stacked exact overhead items, +5.9% vs bar, central +1.9%, ~15%); start 80 in tranches from the first gate pass; items harvested one by one.
- 2026-11-09 Stall (from the human's monitor): leader-23.w1 idle since 10:37 with no task (task_w1_1.md 0 bytes, heredoc failed). Told leader-23 to rewrite and re-send.
- M0 (CPU, agents/leader-23.w1/W1-1/REPORT.md): M1 drafter near floor (1.61 ms/step; MTP GEMMs ~roof; lm_head 1.16x; rank-1 wait = GPU0 clock skew on ~35 tiny sampling kernels) -> M1 downgraded to <=0.3%; payoff-if-it-works now ~+5.0% (at bar). Small mixed steps (pt<256): 67 vs 23.5 ms; Marlin +20.5, comm +16, glue +2.4, chunked GDN +2.3, idle +3.1. Marlin staircase ~210 us per 64-row slab; tail padding <=3.2-4.6 ms per small step. Next W1-2: exact dead-m-block skip in last slab (<=6 seed min); G2a kill if <1.5 ms/small mixed step.
- G2a FAILED (CPU estimate from in-situ data, 0 GPU; W1-2/REPORT.md): dead m-blocks avg 0.74/small mixed step under T6 policy, <=1.0 ms/step (gate 1.5). M2a dropped; payoff-if-it-works ~+4.1% < bar. Start 80 not released. Remaining lead W1-3 (<=5 seed min): stock Marlin remainder launch re-streams weights for down/qkv at M>128 (in situ down M=133-155: 106+33 us); gate 0.6 s/trial. Closure condition unless >=+5% restored with measured parts; may harvest a single item that passes ABBA.
- 2026-11-09 T23 closed after seed (KB80; 19.4 returned). T24 (explore) opened, leader-24 started. Pool free 1108.7. Note: 9 consecutive explore teams (T15-T23) closed at seed in test-6; ~390 min of 1500 spent.
- T24-001 (job 155, 0.2 min): premise FAILED. gate_up M=48: Marlin warm 85.8; 3-m-tile kernel warm 114.8; no-B-loads 88.7 (rule <=75); B-loads-only 103.2 (rule <=70). Fixed per-fragment cost dominates; Marlin near realistic warm floor (<=~+2% e2e). No proposal; closure confirmed pending write-up.
- 2026-11-09 T24 closed after seed (KB81; 19.8 returned). T25 (explore) opened, leader-25 started.
- 2026-11-09 T25-1 approved (stacked package led by GPU1-aware MLP rebalance; +7.2% payoff, central +2.7%); start 80 in tranches (30 on G1, 50 on G2); memory-change rules apply.
- G1 (T25-001, ~0.4 min): marginal pass, +1.2% wall predicted (0.8-1.6); S1 central ~+0.8-1%. Released tranche 1: start 30. Package payoff-if-it-works now ~+6.3% with S4, ~+4.5% without (below bar w/o S4). Plan: (a) uneven split, ABBA; (b) duplicated slice.
- S1a T25-002 built (GPU0 8064 / GPU1 9344 MLP cols; two-slot GDN; KV 4.3 GiB = 237,658 tokens vs champion 267,721; 128K pass; 64-prompt load OK; offline <=2 ulp). Jobs 157/158 wasted (wrong model file; OOM at KV 5). Validity: split ratio as code default tuned to the fixed hardware's GPU0 clock lock -> allowed under ruling 3 (manager); must be a documented hardware constant at tp=2. ABBA next (<=20 min), must show no preemption / Running<16 stalls.
- G2 FAILED for S1a (job 160, 10.2 min ABBA): A 939.0/930.9, B 929.6/932.4 (-0.4%, noise); 0 preemptions; TPOT p50 higher in B (decode pays for f=1.073: forecast had costed f=1.05). Next S1b (decode keeps even split via duplicated slice; mixed-only gain ~+1%) on remaining 21.9 min: smoke + ABBA; falsifier = B TPOT back to A's.
- 2026-11-09 T25 closed (KB82; 6.3 returned). T26 (explore) opened, leader-26 started.
- 2026-11-09 T26 closed after seed (KB83; 19.5 returned). T27 (explore) opened, leader-27 started.
- 2026-11-09 T27-1 approved (+6.8% payoff, central +2.5%). R1 screened draft lm_head ruled ALLOWED (draft-only, self-generated data, task_definition.md). Tranche 1 (30) granted now: seed alone too thin for porting + M1.
- 2026-11-09 Human ruling (via designer): fine-grained prefix caching for the hybrid model APPROVED, no tuning to the 45-token prompt/benchmark; added to task_definition.md. T21-1 released: tranche 1 = 30 granted; 50 on M1 gate.
- 2026-11-09 Human instruction (via designer): let T21 and T27 run to completion under normal funding rules; do NOT open replacement teams. When both have closed: pause (no agents or GPU jobs running), status note here, report to the human. Pause, not finalize (no reports/final.md).
- 2026-11-09 T27 resync #1: G1 failed (budget -2.6%, unexplained penalty persists; payoff now ~+3.6% < bar). LAST CHANCE, no new grant (<=20 of remaining 35.8): fused screened lm_head at k=2 alone -> ABBA -> official if it wins; one split run to localise the penalty; then write-up. Tranche 2 not released. reports/T27-1.md
- Last chance done: T27-002 screened draft head (exact; R1 documented; 115 vs 336 us/draft step) ABBA job 183: A 939.0/931.5, B 937.5/947.8 (+0.8% pooled, not all-B>all-A). T27-003 n=1: adaptive plumbing penalty -8.5% outside the decode step. Manager: one repeat same-job ABBA (~11 min, own balance) to settle T27-002; official (top-up) only if all-B > all-A; else close.
- T27-002 repeat ABBA job 186 (B A A B): B 946.2/940.1 vs A 937.1/935.6, all-B > all-A; pooled 8 starts +0.76%. Validity skimmed: exact (verification unchanged), R1 pipeline documented, PCA artifact loaded from bank/T27-002 by default path, 128K pass. Granted 15 (top-up) for official_eval T27-002.
- 2026-11-09 PROMOTED T27-002 to champion: official 967.3 (966.3/968.3) vs T14-004 955.8/957.5; ABBAs jobs 183/186 pooled +0.76%; gates pass; artifact data = T3 self-generated (R1). T27 closed after win; no replacement (human instruction). Open teams: T21 only.
- 2026-11-09 T21 resync: M1 gate failed (+0.5% n=1 vs +2.5%); payoff ~+2.5% < bar. Decision: close with write-up; +0.5% harvest not funded. reports/T21-1.md. Run pauses after T21 closes.

## 2026-11-09 RUN PAUSED (human instruction; not finalized)
- T21 closed (KB86). T21 and T27 were the last live teams; no replacements opened (human instruction).
- State: no agents running except the manager; no GPU jobs running or waiting; all team accounts closed.
- Champion: T27-002, 967.3 tok/s official (+1.1% vs T14-004, +53.2% vs BASE 631.4). test-6 path: T1-042 942.2 -> T9-005 948.9 -> T14-004 956.6 -> T27-002 967.3.
- Pool: 1500 total, 497.8 spent (all of test-6's teams T2-T27), 1002.2 free.
- test-6 tally: 26 explore teams (T2-T27); 3 wins (T9, T14 with T6's kernel, T27); the rest closed with write-ups in reports/. KB59-KB86 added in test-6.
- Open leads (exploit-sized, ~1% each): mixed-step excess (~6% above tensor roof, KB83/77/80); the adaptive-verify plumbing penalty (KB85) blocking budgeted K=3; prompt-as-draft feeding (T21 M2, KB86); slot lifecycle <=1% (KB77).
- To resume: start a new manager session on this run folder; it rebuilds from journal.md, champion.md, teams/, graph.md, gpu status. No reports/final.md written.

## 2026-11-09 Human ruling during the pause (written by the designer; the manager session had ended)
- FP8 activations: the checkpoint's own W8A8 scheme (dynamic per-token-group-128 e4m3 activations
  against the 128x128 block-scaled weights) is now ALLOWED, with conditions (no coarser/static
  scales, fp32 across K-blocks, full GSM8K-1319 within 2.0 points of a same-job bf16 reference).
  Edited in place: task_definition.md (Correctness) and assumption_ledger.md (Frozen).
- The run stays PAUSED. A resuming manager should announce this ruling to teams when work resumes;
  don't resume work because of it.

## 2026-11-09 Policy change during the pause (human decision; written by the designer)
- Explore persistence: funding_policy.md replaced with the updated v2 default (commit 969cd6b;
  the run's copy predated the voluntary-closure row, otherwise unchanged). A failed attempt isn't
  a dead direction: when a team recommends closing after one failed attempt with a fixable cause
  (not a hard limit), encourage a variant first (a second seed if its seed is spent); close if it
  still wants to. After a failed test, encourage teams with budget and a fixable cause. Pivots
  after the first funded proposal stay related to the direction. No penalty for closing; the 5%
  bar stays. Kickoffs frame earlier write-ups as tried/open, not as dead ends. The v2 role prompts
  (manager.md, leader_explore.md) carry the same change.
- Applies when the run resumes. The run stays PAUSED.

## 2026-11-09 RUN RESUMED (new manager session)
- Resume per human: 2 explore teams, no exploit; config.md slots A/B, up to 2 workers per leader. Funding policy as updated during the pause (re-read).
- Opened T28 (explore, slot A, leader-28) and T29 (explore, slot B, leader-29), 20-min seeds each; inspiration cards inspiration/T28-1.md, T29-1.md attached unchanged. First tasks carry the FP8-activation ruling, prefix-caching ruling, abba.sh, 2-worker limit; also announced. Champion T27-002 967.3; bar 1015.7. Pool free 962.2.
- 2026-11-09 T29-1 approved (W8A8 FP8 GEMMs for mixed/prefill; +12.5% if it works, central +5%, ~30%): start 80. Split with T28 (decode sizes) is similar-but-different, not a duplicate.
- 2026-11-09 T28-1 approved (small-M W8A8 decode GEMM -> exact k=3/4; +8.5% if it works, central +3.5%, ~25%): start 80. Complements T29 (dispatch by M), not a duplicate.
- 2026-11-09 T28 progress: T28-003 k=3 on W8A8 path, step +2.6 ms (vs +4.3 Marlin), E 2.671, unpaired +1.8%; ABBA next. Noted, no action.
- 2026-11-09 Clarification (manager, task_definition.md): one GSM8K-1319 FP8 A/B may serve records with identical FP8 code; exact changes (k, two-slot, scheduling) may differ; FP8 code changes need their own A/B. Asked by T28 (k=3 on T29's Triton W8A8 path). T28-003 same-job ABBA k=3 vs k=2 on W8A8 +2.93% all B>A; T28-004 Marlin-e4m3 M=64 warm 13.39 vs Marlin M=48 13.75 ms.
- 2026-11-09 T29 on mode 2 (Marlin layout, W8A8 only M>96; T29-003 ABBA queued). Teams converging on a combined stack sharing one GSM8K-1319 A/B (consistent with the clarification). Noted.
- 2026-11-09 T29-003 milestone: +9.2% same-job ABBA vs champion (job 207, all B > all A), W8A8 at M>96 only. Spot-checked. Official path: GSM8K-1319 A/B + 128K + official_eval; balance ~62 covers it.
- 2026-11-09 PROMOTED T29-004 to champion: official 1045.8 (1043.9/1047.7) vs T27-002 966.3/968.3 (+8.1%); ABBA job 207 +9.2% all B>A; gates pass; FP8 condition met (GSM8K-1319 job 214 +0.61 pt). Validity checked: per-128-K-block fp32 promotion in bs_gemm.cu, dynamic per-token-group quant, decode bitwise champion, serve_args unchanged, configs from shape microbench. New bar 1098.1.
- Resync T29 #1: EXTEND 80 (beat the champion). reports/T29-1.md. Pool free 722.2.
- 2026-11-09 KB87-KB90 added from T29 (KB89's host-launch-bound part marked unconfirmed).
- 2026-11-09 KB89 corrected (T29 job 219 untraced probe: small mixed steps GPU-bound; ~4 ms backlog margin before host floor).
- 2026-11-09 T29 re-forecast: bundle +1-2% then likely close/merge into T28's combined stack. Noted; harvest within existing balance is fine.
- 2026-11-09 Addendum to GSM8K-1319 sharing clarification: bitwise-equal fused quant producers and tiling/split-K-only config changes count as identical FP8 code (bitwise test on real served activations required). Asked by T28 for T29's bundle on T28-005.
- 2026-11-09 Resync T28 #1: payoff now ~+2.4-3.8% < bar (k=3 fixed costs: AR bytes, drafter); LAST CHANCE from existing 51.6 balance (no grant): fixed single-pass kernel + one ABBA; promotable if it wins reliably; else close. reports/T28-1.md
- 2026-11-10 T28 k=3 closed (structural register limit at 64-row slab, T28-007); last test e4m3 at k=2. Told: a reliable ~+1% win is worth GSM8K-1319 + official; top-up from the pool if needed.
- 2026-11-10 Config change (human, via the designer): slot B workers now claude-sonnet-5-5/medium (leader stays Opus 5.5); applies to T29's new workers and any slot-B replacement team; T29's current workers (w1, w2) keep Opus. config.md updated by the designer; agent start applies it.
- 2026-11-10 PROMOTED T29-006: official 1062.1 (1062.3/1061.8) vs T29-004 1043.9/1047.7 (+1.6%); ABBA job 231 +1.09% all B>A; gates pass; FP8 accuracy via bitwise-reuse of job 214 (147 served dumps 0 diffs; config table 0 diffs; glue 192/192). Bar now 1115.2.
- 2026-11-10 T29 recommends close after 2 wins (mixed-step floors measured: hard limit). Accepted per policy; write-up requested; then close and open T30 (explore, slot B).
- 2026-11-10 KB91-KB93 added (T29). T29 closed (two wins; mixed-step floors measured; reports/T29-writeup.md); leader-29 + w1/w2 stopped; unspent returned. T30 (explore, slot B, leader-30) opened with seed 20; inspiration/T30-1.md attached unchanged. Pool free 786.0.
- 2026-11-10 T28 closed after its last chance (T28-008 -1.84% loss; k=3 null; spill-fix variant T28-007 tried) per policy (variant aimed at cause already tried). KB94-KB96 added (KB94 corrects KB63's cliff cause). T31 (explore, slot A, leader-31) opened with seed 20; inspiration/T31-1.md. Pool free 804.6.
- 2026-11-10 T31-1 approved (register-direct e4m3 decode GEMM cashing AR L2 prefetch; +12% if it works, central +5%, null ~40%). Start 80 released on S1 seed measurements (kernel warm rate, L2 read rate at 40 MB) not falsifying the premise.
- 2026-11-10 T30-1 approved (two-wave layer-pipelined mixed steps; +8% if it works vs bar 1115.2, central +3%, ~35%): start 65 as requested. Pool free 739.6.
- 2026-11-10 T31 re-forecast +6.7..11% (was +12%), still clears bar; start 80 held until W1-2 meets its own lines (qkvz warm <=26 us, cold within 5% of Marlin).
- 2026-11-10 T31 S1 met (kernel v2 -15% warm per forward; gate_up cold +6% with visible cause): start 80 granted.
- 2026-11-10 Resync T31 #1 (reports/T31-1.md): T31-003 ABBA +2.25% all B>A (job 285); gate_up prefetch part dead by measurement; payoff now ~+3.75% < bar. Decision: harvest T31-003 to official on existing balance (~69); no new grant; then last-chance-style decision on remaining items.
- 2026-11-10 Addendum 2 (task_definition.md): strict-subset reuse of a GSM8K-1319 A/B allowed (rest falls back to validated paths). T31 testing T31_MAX_M=48 (TTFT +15 ms hypothesis: host cost in small mixed steps).
- 2026-11-10 Human ruling (via the designer): the FP8-activation accuracy A/B now uses the GSM8K-250 subset (gsm8k.py default) instead of all 1319 questions; same-job bf16 reference, within 2.0 points. Sharing rules and addenda unchanged; past 1319 A/Bs stay valid. task_definition.md + task/TASK.md updated.
- 2026-11-10 Config change (human, via the designer): slot A leader now claude-opus-5-5/medium (workers stay Sonnet 5.5); applies to the next slot-A leader (T31's replacement). leader-31 keeps Fable 5.1. config.md updated; agent start applies it.
- 2026-11-10 Human instruction (via the designer): let T30 and T31 run to completion under normal funding rules; do NOT open replacement teams. When both have closed: pause (no agents or GPU jobs running; not finalized).
- 2026-11-10 PROMOTED T31-003: official 1088.9 (1091.3/1086.6) vs T29-006 1062.3/1061.8 (+2.5%); ABBA job 290 +2.75% all B>A; gates pass; FP8 A/B job 288 (+0.30, n=1319; dispatch rewrite after it is host-only, configs identical); serve_args unchanged. Bar 1143.3.
- 2026-11-10 T31 closed after its win (KB97-KB99; 44.1 returned); leader-31 + w1/w2 stopped. No replacement (human instruction). Open: T30 only.
- 2026-11-10 T30 closed (k-wave variant aimed at the cause failed; AR latency-bound, KB100; 15.6 returned). leader-30 + workers stopped. No replacement (human instruction).

## 2026-11-10 RUN PAUSED (human instruction; not finalized)
- T30 and T31 were the last live teams; both closed; no replacements opened (human instruction).
- State: no agents running except the manager; no GPU jobs running or waiting; all team accounts closed.
- Champion: T31-003, 1088.9 tok/s official (+2.5% vs T29-006, +71.2% vs BASE 631.4). Path since resume: T27-002 967.3 -> T29-004 1045.8 -> T29-006 1062.1 -> T31-003 1088.9 (+12.6% over T27-002).
- Since resume: 4 explore teams (T28-T31); 3 wins (T29 x2, T31); T28 and T30 closed with write-ups. KB87-KB100 added. Rulings: FP8-activation A/B sharing clarification + addenda 1-2; human: A/B at GSM8K-250.
- Open leads: fewer bytes per AR message for mixed steps (KB100); lean host path for the eager GDN core (~1.24 -> ~0.21 ms/call costed, KB100); quant fusion at M<=96 and drafter in e4m3 (<=0.5% each, KB99); gate_up is an L2 wall (KB98); deeper speculation AR/drafter-bound (KB95).
- Pool: 1500 total, 780.7 spent, 719.3 free.
- To resume: start a new manager session on this run folder; it rebuilds from journal.md, champion.md, teams/, graph.md, gpu status. No reports/final.md written.
- 2026-11-10 Human (via the designer): resume with ONE external team (T32, slot C: Opus leader + Sonnet worker, max 1 worker); no other teams; pause again when it closes. New team type: design/external_teams.md; tools/scout. A fresh manager session starts (kickoff_resume2.md).

## 2026-11-10 RESUME 3 (new manager session; human instruction: one external team, slot C)
- Rebuilt state from files: champion T31-003 1088.9; all teams closed; pool free 719.3; no GPU jobs.
- T32 (external, slot C, leader-32) opened with seed 20; scout deck T32-1 started (scout/T32-1/). Pool free 699.3. Proposal bar 1143.3. When T32 closes: pause again (human).
- 2026-11-10 PAUSED by the human (via the designer) after leader-32's assessment of scout deck T32-1 (proposals/T32-1.md: no card clears +5%; best sub-bar idea: all-gather the FP8 out_proj input + duplicated out_proj, from card 10, ~+2-3%, needs a comm ruling). T32 left open (not closed); leader-32 and the manager stopped; no GPU jobs. Seed unused. Not finalized.
- 2026-11-10 leader-32 did its own web research at the human's request (proposals/T32-2.md, CPU only): no single candidate reaches +5%; best package = FP8 all-gather out_proj (card 10) + lossless FP8 weight coding for gate_up + batch-level draft budgeting, ~+4.5-7% if all work, central ~+2.5%. leader-32 stopped again; run PAUSED.
- 2026-11-10 Human (via the designer): T32 closed; T33 (external, slot C) opened with a fresh scout deck T33-1 for a second data point; manager stays offline; pause after leader-33's reaction to the deck.
- 2026-11-10 leader-33 wrote proposals/T33-1.md (reaction to deck T33-1); leader-33 stopped; run PAUSED (manager offline).
- 2026-11-10 Human (via the designer): leader-32 resumed (CPU only, team closed) to try the new sparring partner (spar; Codex gpt-6.1-sol / high) on its T32-1/T32-2 conclusions; result to proposals/T32-3.md; then pause.
- 2026-11-10 leader-32 sparred (sparring/T32-1.md, 6 turns, Codex) and wrote proposals/T32-3.md; stopped; run PAUSED.
- 2026-11-10 Human (via the designer): T33 closed; T32 reopened (seed 20 restored) to continue from its sparring conclusions (T32-3); new manager session on Sonnet 5.5 (kickoff_resume3.md).
- 2026-11-10 RESUME 4 (new manager, Sonnet 5.5): T32 reopened (seed 20, open); leader-32 restarted with task to continue from proposals/T32-3.md. Pool free 699.0. Pause again when T32 closes.
- 2026-11-10 T32 closed on its leader's recommendation (proposals/T32-4.md, bank/T32-001): CPU tests show waves capped by AR dependency latency (131.0->116.2 ms vs 88-93 floor), GEMM headroom ~+1.2-2.4% e2e, RS+AG FP8-input test ~8% slower (inconclusive harness). Only stack clearing the bar (waves+byte cut+GEMM tuning, +6.8-8.1% if all work) has central +1.5-2%, odds >=5% ~5-10%, needs 150-200 GPU-min over three builds and revives T30's closed direction. Declined the optional 20-min staged bet (leader itself recommended closing). 19.8 returned; pool free 718.8. leader-32 stopped.

## 2026-11-10 RUN PAUSED (T32 closed; not finalized)
- No agents except manager; no GPU jobs; all accounts closed. Champion T31-003 1088.9. Pool 1500 total, 781.2 spent, 718.8 free.
- 2026-11-10 Human (via the designer): spar can reopen ended conversations; manager asked to resume leader-32 (CPU only) and tell it.
- 2026-11-10 Human (via the designer): leader-32 resumed, CPU only, T32 stays closed, to be told that `spar send` now reopens an ended sparring conversation and `spar guide` has a section on coming back with results. Pause after.
- 2026-11-10 Human (via the designer): correction: T32 reopened (balance 19.8); not CPU-only; normal funding rules.
- 2026-11-10 Human correction (via the designer): T32 reopened (gpu team reopen), balance 19.8; not CPU-only; normal funding rules. leader-32 told.
- 2026-11-10 T32-5 approved (external; host-free 3-wave layer-pipelined mixed steps via CUDA-graph GDN prefill core capture; payoff if it works A+C +4.3-5.5%, A+B+C +6.1-7.3%; central ~+1.8%, ~10% for >=+5%). Forecast parts marked measured (A replay job 294) / assumed. Low odds, small steps: Stage 0 on the 19.8 balance (<=15); start grant 80 released when Stage 0 gates pass (real-sequence states bitwise equal; host <=0.15 ms/call; padding <=+10% GPU). B (FP8 all-gather on the link) needs a human ruling: not ruled, not needed for Stage 0/1.
- 2026-11-10 Human (via the designer): sparring partner now asks one question per message; T33 reopened; manager asked to resume leader-33.
- 2026-11-10 Human (via the designer): T33 reopened (gpu team reopen, balance 19.8); leader-33 restarted with spar available; normal funding rules. Pause when both T32 and T33 have closed.
- 2026-11-10 T32 Stage 0 passed (bank/T32-002: bitwise 15/15, host 0.51->0.003+0.03 ms/layer, GPU +0-3%). Start 80 granted. Stage 1 gates: in-server big-step span within ~5 ms of replay; ABBA >=+2% else stop.
- 2026-11-10 T32 Stage 1 plan: 1a (<=20 GPU-min) captured GDN prefill in-server, flag off by default; 1b waves. Risk noted by leader: non-core per-wave pieces also need graphs (T30-004 eager raised host enqueue ~70->135 ms). Noted, no action.
- 2026-11-10 T32 milestone (bank/T32-003): captured GDN prefill in-server, bitwise tokens; host enqueue -33 ms/mixed step, GPU span unchanged; ABBA job 326 +0.29% (neutral, expected). 33.7/100 used. Next 1b (3-wave). Not a champion candidate. Noted.
- 2026-11-10 Resync T32 #1 (reports/T32-1.md = proposals/T32-6.md): wave-eligible steps only ~14-17% of GPU time (not 28%); A ceiling +1.6-2.0%; A+C +2.8-4.4% < bar; central ~+1%. Hard measured ceiling -> closed on leader's recommendation; declined optional 5-min RS+AG probe (would need a human ruling on B). 69.0 returned. Assets: T32_GDN_CAP (bank/T32-003, exact, host -33 ms/mixed step, neutral alone), Stage 0 bench. leader-32 + w1 stopped. No replacement (human: pause when T32 and T33 closed). Open: T33.
- 2026-11-10 Human (via the designer): add 1 explore team (config: explore 1). New manager rule: offer sparring once when a team wants to close or finds no direction.
- 2026-11-10 Human (via the designer): config explore: 1. Opened T34 (explore, slot A, leader-34), seed 20, inspiration/T34-1.md unchanged. New manager rule: when a team recommends closing or finds nothing worth proposing and hasn't sparred, offer spar once (one line). Pause when all teams closed. T32 closed earlier (reports/T32-1.md). Pool free 648.2.
- 2026-11-10 T34 status: candidate = mixed-step AR transport (hybrid CE-write + SM-read-add at prefill sizes, + 0.78 codec); stage-0 microbench <=6 of seed running; proposal T34-1 to follow. Leader already spared (sparring/T34-1). Noted.
- 2026-11-10 T33 closed on leader's recommendation (reports/T33-1.md = proposals/T33-2.md, bank/T33-001..007): overlap-only stack ~+1-1.5% e2e (exploit-sized), card 13 lossless coding loses when weights are L2-resident, variant aimed at cause tried (T33-006), ceiling ~+4.4-4.8% even with free decode. Already sparred (11 turns): no spar offer needed. KB101-103 added. Assets: agents/leader-33.w1/ASSET.md. Open: T34.
- 2026-11-10 GPU BLOCKER: T34 job 345 crashed (illegal address in graph replay), orphans PIDs 1015428 (R, 100% CPU) and 1015460 (defunct) hold GPU contexts (606 MiB each), unkillable; gpu run refuses jobs as 'outside this run'. Needs human (GPU reset/root); no team to attempt. T34 told to hold GPU work. Reported to human.
- 2026-11-10 17:2x designer (human OK): two orphaned python processes from T34 job 345 (pids 1015428/1015460, ~600 MiB on each GPU) stuck in driver teardown; kill -9 had no effect (in-kernel). GPUs otherwise free (~23 GiB each).

## 2026-11-10 RUN PAUSED (human, via the designer; GPU issue)
- Stuck processes from job 345 (PIDs 1015428, 1015460) hold GPU contexts; human is sorting it out. No GPU jobs running/waiting. leader-34 and leader-34.w1 stopped (state kept, can resume). T34 NOT closed (seed 20, no start grant; proposal T34-1 pending, stage-0 AR microbench blocked). Champion T31-003 1088.9. Pool free 666.4. Wait for the human before resuming; T33 and T32 closed.
- 2026-11-10 ~17:58 Run PAUSED for a server reboot (human). Manager session 0b2fbb25-d642-4d87-9d76-1de48c16357e stopped (resumable); leader-34/.w1 stopped earlier; T34 open (draft T34-1). Champion bundle exported to .cache/exports/.
- 2026-11-10 Server rebooted (human); GPUs clean. Manager resumed (same session) and asked to resume leader-34.
- 2026-11-10 RUN RESUMED (human): server rebooted, GPUs clean. leader-34 (and w1) restarted; T34 open. Pause when all teams closed.
- 2026-11-10 Passed the human's facts on the job-345 GPU incident (driver-teardown hang of flag-polling kernels; reboot needed) to leader-34; prevention is its call.
- 2026-11-10 leader-34 safety plan before next GPU job: code review, 1-min eager smoke with syncs, single checked graph replay, bounded 20 ms polls, os._exit on error, suspect arms dropped. Acceptable; noted.
- 2026-11-10 T34-1 approved (hybrid CE+SM mixed-step AR at PCIe duplex floor: 639 us vs 631 floor at 1100 rows, bitwise; payoff if all of H+C+F+W works ~+5%, right at the bar; central ~+2%, odds 15-20%). Parts marked measured/assumed. Low odds, funded in small steps: start 60 (stages 1-2); stage 1 gate ABBA all-B>all-A and >=+1.0% else close; waves/official at a resync. Waves component is limited by T32's finding that wave-eligible steps are ~14-17% of GPU time; leader to account for it.
- 2026-11-10 Resync T34 #1: stage 1 (T34-002, job s7) falsified the premise: champion mixed-step transport within ~11% of raw PCIe duplex floor (H1 660 vs DMA 699 vs hostar 620 us at R=1100); H saves 3-39 us/AR, <=~0.8% wall < gate. Leader recommends close (hard measured limit; sparred). Accepted; write-up requested, then close.
- 2026-11-10 T34 closed (reports/T34-writeup.md; bank/T34-001..002): hybrid AR reaches floor but saves <=0.8% wall; KB104-106 added. Used 5.5, 74.5 returned. All teams closed; no replacements.

## 2026-11-10 RUN PAUSED (all teams closed; not finalized)
- T32, T33, T34 closed (no wins). Champion T31-003 1088.9 tok/s official (+71.2% vs BASE 631.4); bar 1143.3.
- No agents running except the manager; no GPU jobs. Pool 1500 total, free: see `gpu status`. KB101-106 added since resume 4.
- Open leads: bytes (codec floor 543 vs hostar 620, ~+1%); hostar vs DMA lane in exposed ARs (unproven in server); waves (~2% achieved, 116->90 ms replay gap unexplained); overlap-only decode GEMM stack (~+1-1.5%); T32_GDN_CAP asset (bank/T32-003).
- To resume: new manager session on this run folder. No reports/final.md written.
- 2026-11-10 leader-34 caveat (floor and champion DMA from different jobs; skew explanation inferred) added to KB104. T34 already closed, w1 already stopped.

## 2026-11-10 RUN FINALIZED (human instruction via the designer)
- graph.md, shared_kb.md (KB101-107), teams/ updated; reports/final.md written. Champion T31-003 1088.9. Pool: ~819 spent, ~681 free. No agents or GPU jobs running.
- 2026-11-10 Paused by the human after finalization; designer stopped the manager session (5e2defa2) and the dashboard.
- 2026-11-10 Designer resumed the manager (feb84da5) for one task from the human: back-fill '(related: KBn)' pointers in shared_kb.md (kickoff_resume4.md). Run stays paused; no new teams.
- 2026-11-10 Maintenance (human): applied the '(related: KBn)' pointer rule to KB1-KB107 in one pass (pointers only to later entries on the same topic; no content changed). Run stays paused.
- 2026-11-10 Maintenance (human): applied the '(related: KBn)' pointer rule to KB1-KB107 in one pass (pointers only to later entries on the same topic; no content changed; backup agents/manager/shared_kb.pre-related.bak). Run stays paused.

## 2026-11-10 RUN RESUMED (human): one moonshot team
- New team type `moonshot` (flavor ffcc7b3; agents/leader_moonshot.md, manager.md moonshot sections, funding_policy.md "Moonshot teams" section added to this run's copy). config.md: moonshot 1 in slot A (Opus leader / Sonnet worker); explore/external 0.

## 2026-11-10 RUN RESUMED (human): one moonshot team
- New team type `moonshot` (flavor commit in flavor_commit.txt; agents/leader_moonshot.md, manager.md moonshot sections; "Moonshot teams" section added to this run's funding_policy.md). config.md: moonshot 1 in slot A (Opus leader / Sonnet worker); explore/external 0.
- Designer resumed the manager (76381902) with kickoff_resume5.md.

## 2026-11-10 RUN RESUMED (human): one moonshot team
- Opened T35 (moonshot, slot A, leader-35): seed 20 + 10 top-up = 30; history-free first task (champion T31-003 only) + inspiration/T35-1.md unchanged. No other teams. Pool free before: 681.0. Do not tell it about earlier teams, KB or conclusions except the one history check after its proposal.
- 2026-11-10 Human rule change (designer relayed to the manager): moonshot history check gives pointers only (no results/verdict); past work never a reason to reject or shrink a moonshot grant. Flavor 6316701; this run's funding_policy.md updated.
- 2026-11-10 Rule change (human, via the designer): moonshot history check = only pointers to related past records ('related work: records X, Y (team T); read them if useful, the decision is yours') or 'nothing related found'. No results, verdict or advice; past work is never a reason to reject or shrink a moonshot grant. Applies to T35.
- 2026-11-10 T35-1 approved (moonshot: streaming-bound decode via run-ahead L2 streamer; rejects 'decode step = chain of dependent kernels each streaming its own weights'; upside bound ~+35%, central +8%, ~25% odds; riskiest assumption = idle-DRAM windows >=5 ms/step fillable; stage 1 = ncu DRAM accounting + microbench, kill <~290 us/layer). Stage 1 grant 15. History check sent as pointers only (T22-001/002, T1-019, T31-001/002, T1-034, T33-001..007), no results/verdict.
- 2026-11-10 Human: second moonshot team in a new slot D (leader Fable 5.2 / medium, worker Sonnet 5.5 / medium); config.md updated.
- 2026-11-10 Human: second moonshot team. Opened T36 (moonshot, slot D, leader-36), seed 20 + 10 = 30; history-free first task + inspiration/T36-1.md unchanged. T35 stage 1 running (milestone 1 microbench; leader narrowed hypothesis from the pointers to streaming the next GEMM's weights concurrently with the current one). Pool free 606.0.
- 2026-11-10 T35 closed (moonshot, per policy: riskiest assumption shown false). T35-003: run-ahead L2 streamer made every GEMM slower (GEMMs already DRAM-bound at 0.8-0.9 TB/s); best variant 340.8 vs champion-like 317.5 us/layer; remaining gap to DRAM floor is AR latency (39/50 us in situ vs 12 us harness). Screen negatives T35-001 (small token budgets -19%/-2.6%), T35-002 (k=3 -2.6% cross-job). Already sparred (sparring/T35-1). Used 20.1 of 45; ~24.9 returned. Write-up reports/T35-writeup.md (= bank/T35-003/record.md). Open: T36.
- 2026-11-10 Human rule change (designer relayed): offer sparring once per closing decision (any team type) unless those closing reasons were already discussed; human asked the manager to offer leader-35 such a session now (guidance only). Flavor ce7f450.
- 2026-11-10 New rule (human): offer a sparring session once per closing decision unless the team's spar already discussed the closing reasons. T35's spar predated its streamer test; offered to leader-35 (restarted, T35 stays closed, CPU only).
- 2026-11-10 T35 sparring session taken (spar T35-1 turns 4-6): closing stands; corrections: remaining gap not all communication (small ops ~15 us/layer, AR delta ~2.05 ms/step). T35 stays closed, leader-35 stopped. KB108 added.
- 2026-11-10 T36-1 approved (moonshot: 'fill every weight pass': confidence/load/tile-chosen speculation depth + GDN state memory decoupled from depth; upside if works ~+8-9%, central +4-5%, ~35% odds; riskiest assumption = drafter confidence separates accepted/rejected rows; milestone 1 fits in seed). Stage 2 (+20) on milestone 1 passing. Ruling added to task_definition.md: load-adaptive depth from live state allowed if verification exact and nothing keyed to benchmark. History check sent as pointers only. Noted: T36 saw the tail of job 357 (T35 output) via gpu wait on a wrong id; unused, dropped overlapping direction.
- 2026-11-10 T36-1 withdrawn by leader before milestone 1 (pointed records T5-006/T27-001/T28-003/T28-005 + own job 363 c=14 K3 vs K2 +2.1% show 'more accepted tokens per pass' tops out ~+2-4% < bar). bank/T36-001 null. Used 17.9 of 30. Leader proposed one more GPU-free search (new inspiration set + sparring) before closing: agreed, no new grant; T36 stays open on its remaining 12.1 seed. Measured facts: decode pass ~80% row-independent (18.6 ms at M<=12, 23.4 at M=48); mixed step 30.8 + 0.160 ms x prompt tokens.
- 2026-11-10 T36 closed on leader's recommendation (T36-1 withdrawn; GPU-free search + sparring added T36-002: decode AR is bandwidth-bound, cut-through chunking screen negative). Used 23.0 of 30; 7.0 returned. KB109 added. Write-up reports/T36-writeup.md (= proposals/T36-1.md) + local_kb/leader-36.md. Leader named one untested idea: copy-engine transfer beside the kernel path at decode size (not funded: team closed).

## 2026-11-10 RUN PAUSED (all teams closed; not finalized again)
- T35 and T36 (moonshots) closed without wins (T35 streamer refuted; T36 speculation-depth idea answered by past records, AR shown bandwidth-bound). KB108-109 added. Champion T31-003 1088.9. No agents or GPU jobs.

## 2026-11-10 RUN RESUMED (human): third moonshot team, speed-of-light module
- task/SPEED_OF_LIGHT.md added (flavor 9c962cc; tasks/qwen38-27b-fp8-2x4090/). config.md: moonshot 3 (new team in slot D, Fable 5.2 leader / Sonnet 5.5 worker), 'speed of light: moonshot'. Moonshot first tasks: no 'tuned by N teams'. Relayed to the manager by the designer.

## 2026-11-10 RUN RESUMED (human): third moonshot
- Opened T37 (moonshot, slot D, leader-37), seed 20 + 10 = 30; history-free first task; task/SPEED_OF_LIGHT.md listed in the task's 'where things are' line without comment; no mention of effort spent on the champion. Inspiration/T37-1.md unchanged. Pool free 607.9. Other teams closed.
- 2026-11-10 T37-1 approved (moonshot: one static R-row graph per step, rows spent by expected value: deeper spec rows w/ state memory independent of k + prompt tokens as rows; upside ~+18% if works, central +7%, ~35% odds; riskiest assumption: 0.10 ms/row slope holds to 64-96 rows). Stage 1 grant 31 (total 61 granted; roadmap ~200). History check as pointers only (spec depth, state memory, prefill graphs/host launch, prefill in decode steps), no results. Note overlap with T36-1 (withdrawn): not a reason to reject per rule.
- 2026-11-10 T37 re-forecast after reading pointers: upside ~+6%, central +2-3%, ~10% odds; continuing stage 1 as redefined (T31-003 + two-slot GDN state + k=3 at 16 running, ABBA, 128K, official only if every B>A). Grant left as is (past work not a reason to shrink a moonshot grant).
- 2026-11-10 T37 stage 1: T31-003 + two-slot GDN (fixed) + k=3 with 16 running, ABBA job 375 +1.34% all B>A; 128K pass (job 376). Found correctness defect in the two-slot code (garbage past a 1600-token page boundary, job 377; fixed in job 378); KB110 added; records relying on two-slot (T28-003/005) carry it. Champion unaffected. T37 to run final exact-config check then request official eval (balance 17.8).
- 2026-11-10 T37-002 official 1142.2 (1138.4/1146.1) vs T31-003 1091.3/1086.6; ABBA job 375 +1.34% all B>A; gates pass (GSM8K-250 94.0, 128K). Candidate NOT yet promoted: FP8-activation condition open (job 288 not shown to have run at M>48; no bitwise test on served activations; no k=3 GSM8K-250 A/B). Asked for the GSM8K-250 same-job A/B vs bf16 reference at k=3 (~6 GPU-min); +6 granted.
- 2026-11-10 PROMOTED T37-002: official 1142.2 (1138.4/1146.1) vs T31-003 1091.3/1086.6 (cross-job +4.9%; same-job ABBA job 375 +1.34% all B>A; true gain ~+1.3-2%); gates pass (GSM8K-250 94.0, TTFT p99 780, 128K); FP8 A/B job 381 +0.8 pt; boundary defect fixed (KB110); serve_args differ only in num_speculative_tokens 3. Validity checked: diff touches GDN state storage + boundary copy fix, FP8 code unchanged. New bar 1199.3.
- 2026-11-10 T37 closed (leader recommendation after the win and a sparring session; scope of the stop recorded in reports/T37-writeup.md: all-reduce volume floor, k=4, static mixed step are assumed, not established). Used ~56 of 67; ~10.7 returned. All teams closed.

## 2026-11-10 RUN PAUSED (all teams closed; not finalized again)
- T37 closed after promoting T37-002. Champion T37-002 1142.2 tok/s official (same-job +1.34% vs T31-003; +81% vs BASE 631.4); bar 1199.3. Moonshots: T35, T36 no wins, T37 one win. No agents or GPU jobs. reports/final.md predates T35-T37.
- 2026-11-10 Human: update the final report for T35-T37, then open a fourth moonshot team in new slot E (leader Opus 5.5 / high, worker Opus 5.5 / medium); config.md updated.
- 2026-11-10 Records brought up to date for T35-T37 and champion T37-002 (KB108-111, graph, teams/, reports/final.md updated).

## 2026-11-10 RUN RESUMED (human): fourth moonshot
- Opened T38 (moonshot, slot E, leader-38), seed 20 + 10 = 30; history-free first task (champion T37-002); inspiration/T38-1.md unchanged. Pool free: see gpu status.
- 2026-11-10 T38 ruling question: may the FP8 GEMM use the f16 MMA accumulator within a 128-K block (convert to fp32 per block, fp32 across blocks; exact overflow guard + f32 recompute; GSM8K-250 A/B)? Ambiguous precision decision: passed to the human via the designer; T38 told to keep measuring and not build on it until answered.
- 2026-11-10 T38 withdrew the f16-accumulator question: job 386 shows f16 K-block partials overflow on essentially every output with checkpoint-style scaling, and the kernel gain was only 1.1-1.5x. No ruling needed; question dropped.
- 2026-11-10 T38-1 approved (moonshot: decode AR stops being a barrier: chunk-wise delivery + next weight-only FP8 GEMM runs concurrently, waits per K-chunk; AR = 7 ms of ~26 ms step; upside +8-12%, central +6%, ~35% odds). Milestone 1 (~10 GPU-min) inside seed, no new grant yet; next stage on milestone pass. Pointers sent (AR/GEMM overlap, producer-fused AR, hybrid AR, prefetch).
- 2026-11-10 T38-1 closed by the leader before building (0 GPU-min; pointer T4-001's ceiling and own job 388). New proposal T38-2 approved (self-distilled recursively trained MTP draft head so k=3 accepts more / k=4 pays; upside +6% (k=3) to +9% (k=4), central +4%, ~30% odds; kill if fine-tune adds <+4% expected accepted length). Stage 1 grant 15 (balance ~41, covers milestone 1 ~35). Pointers only: T3 self-distillation records, T27-002, T1-006/007, T21.
- 2026-11-10 T38-2 closed pre-GPU by the leader (T3 records already ran its milestone 1). Leader asked whether T37-1 / mid-M prefill GEMM / short-prefill GDN were covered: answered with pointers only (per moonshot rule). Unspent 15 grant stays in account, returns at close. Leader considering a static mixed-step proposal (ceiling ~7.5%) or closing; spar offered once for closing.
- 2026-11-11 T38-3 approved (moonshot: replace TP=2 with a pipeline-parallel engine, no all-reduce, + dynamic draft trees; upside if works ~+11%, central 0..+2%, odds 10-15%; parallelism may change per task_definition). Stage 1 (~12 GPU-min, spec off PP=2 vs TP=2 nsys profile; pass PP<=0.70x TP marginal prompt-token cost, else close) covered by balance ~40.8; no new grant. Pointers only: write-ups mentioning PP (T4, T11, T18, T23, T26, T36).
- 2026-11-11 T38-3 stage 1 passed (bank/T38-004, job 392): PP prefill 0.51x TP per prompt token (105 vs 205 us) if pipeline stays full (stock PP GPU1 88% busy); PP decode ~0.65x TP; needs decode r>=0.92-0.94 for +5%; today r~0.85. Stage 2 (narrow-K W8A8 configs M=96-160, cheaper draft head; ~12-15 GPU-min) on own balance 32.2; no new grant; pass if measured parts predict r>=0.93 else close.
- 2026-11-11 T38 closed (leader recommendation; sparring supported). T38-1 (non-barrier AR) and T38-2 (self-distilled head) withdrawn pre-build after pointers; T38-3 (PP=2 + trees): stage 1 passed (prefill 0.51x), stage 2 decode r=0.91-0.92 < 0.93 rule. Used 13.7 of 45; 31.3 returned. Write-up reports/T38-writeup.md; KB112 added. All teams closed.

## 2026-11-11 RUN PAUSED (all teams closed; not finalized again)
- T38 closed without a win. Champion T37-002 1142.2 tok/s official (bar 1199.3). No agents or GPU jobs. reports/final.md covers through T37 (T38 summarized in graph.md and reports/T38-writeup.md).

## 2026-11-11 RUN RESUMED (human): three research teams (framing test)
- Opened T39 (no persona), T40 (persona research_ownership), T41 (persona research_reviewer), all research, slot F, seed 20 each, inspiration cards unchanged, first tasks with history (explore-style), speed-of-light pointer from the config (listed in the task's 'where things are'). Rule: each stops after its first proposal; don't fund or judge; record, report to the human, stop its agents. Personas not mentioned in tasks.
- 2026-11-11 Human: research prompt changed, fourth persona. Closed T39-T41 before any work (no write-ups); opened T42 (no persona), T43 (research_ownership), T44 (research_reviewer), T45 (research_radical), all research, slot F, seed 20, same first-task format. Rule: each stops after its first proposal; don't fund or judge; record, report to the human, stop its agents.
- 2026-11-11 Hardware-state finding (leader-43, verified from gpu_clocks.csv): GPU0's 2235 MHz clock lock cleared after the job-345 reboot. KB113 added; champion.md note: T37-002 official partly reflects clock; same-job comparisons only. To report to the human.
- 2026-11-11 T43 (research, research_ownership) sent its first proposal (proposals/T43-1.md); recorded without funding or judging per the human's rule; team closed, leader stopped, 20 GPU-min returned. To report to the human.
- 2026-11-11 T42 (research, no persona) sent its first proposal (proposals/T42-1.md); recorded without funding or judging per the human's rule; team closed, leader stopped, 20 GPU-min returned. Open validity question in it (screening plane memory) left for the human.
- 2026-11-11 Human: research teams continue. Reopened T42 and T43, restarted leader-42/43. T42-1 funded (+20; ruling on certified target lm_head screening added to task_definition.md); T43-1 funded (+10).
- 2026-11-11 T42 stage 1 started (worker w1, workspace t42-cs, record T42-002, cap 8 GPU-min). Deployment: 7-bit plane (~0.57 GB/rank) replaces the 0.993 GB encoded bf16 shard in HBM; candidate rows read exactly from a pinned host copy; fallback full exact lm_head; written bound; gate whole path <=0.65 ms at M=64 vs 1.07 ms. Consistent with the ruling (exact copy on host). Noted.
- 2026-11-11 T43 stage 1 done (bank/T43-001): decision depth late (KB114); gate failed; leader recommends closing (4.3 GPU-min used, 25.7 left). Offered a GPU-free pass with spar (once) before closing; its call.
- 2026-11-11 T43 closed (leader recommendation after the GPU-free pass and sparring T43-2; reasoning rests on the oracle bound ~+1.8% e2e, not on 'late decision depth'). KB114 reworded as asked. 25.7 returned.
- 2026-11-11 T44-1 funded as a research proposal (+15; novelty: duplex finding corrects KB5/73/109). KB115 added.
- 2026-11-11 T44 correction: champion's >=256-token mixed-step ARs are copy-engine-only (already duplex); revised forecast <=~0.5% e2e, odds of >=+1% ~10%; stage 1 narrowed (gate >=10% off exposed hostar AR at R>=128, cap 5 GPU-min); KB115 caveat updated; unused grant returned if killed.
- 2026-11-11 T44 stage 1 killed (bank/T44-002, job 401): no variant >=10% off exposed hostar AR; KB116. Leader recommends closing (balance 33.9). Spar offered once before closing.
- 2026-11-11 T44: sparring changed the close decision. KB104's 631 us floor is for the hybrid transport; CE+CE pipelined bound ~540 us at 1100 rows / ~170 at 300 vs measured 688/247 (~80 us/AR, up to ~4 s/trial if recoverable). Next: 1 GPU-min timeline decomposition of the two-lane AR, then close or a short T44-2 proposal. Caveat added to KB116.
- 2026-11-11 T44-2 accepted as a continuing research proposal: split-step two-lane CE AR ~30% above its correctness-required path at 300 rows (bank/T44-003, job 403); fix = per-lane doorbell stream, reordered resets/join; forecast +1.5-2.5% e2e, ceiling +3.5%; plan ~4 GPU-min bench (Go >=15% at 300 rows) then ~20 GPU-min ABBA; on balance 33.8, no new grant. KB117 added.
- 2026-11-11 T45-1 judged as a research proposal: the chunked-prefill mechanism is known (said so itself), so it is not novel as a technique; the novel/valuable part is the unifying rule (rows priced by the PCIe link; a row pays only if it yields >=~0.14-0.2 tokens; prompt rows are the only certain yield) and the falsifiable claim that KB64/KB108 are implementation limits (eager steps, GEMMs at 1.1-1.5x floor at M=64-128), not physics. Stage 1 (<=6 of seed GPU-min; kill if marginal GEMM cost >40 us/row) funded from the seed, no grant; stage 2 would be exploit-sized and a hand-off.
- 2026-11-11 T44-2 step 1 killed (bank/T44-004): reordered CE AR bitwise identical but slower (x1.05-1.12 at 192-300 rows); T44-003 attribution wrong; KB117 marked superseded/corrected, KB116 updated. Leader recommends closing (balance ~33.2). Spar offered once more for this new closing decision.
- 2026-11-11 T44 closed on leader's recommendation (declined a second spar; reasons come from direct interventions); reports/T44-writeup.md; ~1.8 of 35 used, balance returned. T45-1 stage 1 hit its kill rule (bank/T45-003, KB118); leader recommends closing unless the one e4m3 gate_up bench at M=96-128 is wanted. KB118-119 added.
- 2026-11-11 Human: second set of 3 research teams. Opened T46 (no persona), T47 (research_ownership), T48 (research_reviewer), slot F, seed 20 each, same first-task format (champion T37-002, history included, SPEED_OF_LIGHT pointer). They continue after their first proposal.
- 2026-11-11 Human: second research set now 4 teams. Opened T49 (research_radical), slot F, seed 20, same first-task format.
- 2026-11-11 T42 interim: certified lm_head exact in-server (stage 1: 0 mismatches on 188K real + 1.7K adversarial; stage 2 bank/T42-003: HBM -0.34 GiB, c=1 10/10 identical, 128K pass, fallback 0/26.5K rows); ABBA job 410 +0.53% mixed (not all B>A); expected ~0.42 ms/step, observed ~0.23. Stage 2b (cap 13 GPU-min): nsys profile, fuse/graph-capture sample path, re-ABBA; official only if all B>A; may ask for more for an official eval. Noted.
- 2026-11-11 T45 bench (bank/T45-004): piggyback bet stays dead; found: champion's gate_up at M=64 (k=3 decode width) sits on Marlin's step; routing 48<M<=96 gate_up through w8a8_marlin estimated +1.5-2.5% e2e (not measured in server). KB120. Granted T45 +25 (extension) to run it itself: same-job ABBA, FP8 reuse/A-B, official if all B>A; no exploit team is open.
- 2026-11-11 T42 stage 2b (bank/T42-004, job 414): certified lm_head exact but small (lm_head section 1065->796 us, decode period 25.14->24.94 ms, ~0.8% decode); under the leader's stop line; leader recommends closing (balance 18.1) or (b) +20 for an evict_last prefetch fix + ABBA. KB121 added. Spar offered once for this closing decision.
- 2026-11-11 T42 closed (leader recommendation after sparring supported it). Certified target lm_head exact but ~0.8% of decode (KB121); freed 0.34 GiB/GPU unverified as useful. Write-up reports/T42-writeup.md; bank/T42-001..004. ~18 returned.
- 2026-11-11 T45-005 (T45_GU_BS: gate_up at 48<M<=96 via w8a8_marlin): ABBA job 415 +3.83% all B>A (A 1107.5/1116.5, B 1150.4/1158.7); FP8 A/B job 416 +1.6 pt (<2.0); 128K pass; boundary gate not re-run (GDN state/page code untouched, T37-002 fix carried). official_eval T45-005 started; promotion check on result. KB122 added.
- 2026-11-11 T46-1 judged as a research proposal ('host as a memory tier': embedding table in host memory, exact, frees ~1.27 GB/GPU and removes 4 ARs/step; feeding a fraction of decode GEMM FP8 weights over the link; forecast +1-2% at best, ~50% odds zero-sum; value = freed memory + a measured answer whether the link adds bandwidth). Novel enough (link-as-memory-tier unmeasured in the run); informative either way. Stage 1 microbench <=6 GPU-min from the seed with a kill rule; no new grant; stage 2 (~15) on a pass.
- 2026-11-11 T47-1 judged as a research proposal (asymmetric waves: decode rows as a full decode CUDA graph on a high-priority side stream cover the prefill rows' copy-engine AR; differs from T30's symmetric row-waves by cover size; upper bound +5-7%, central +2-3%, ~35% odds of >=+2%; predicts a crossover prompt length ~30-40 tokens). Novel enough, a testable rule either way. Stage 1 (<=8 GPU-min, gate: >=8 ms saved at P~256, better than joined at P~128) from the seed; no new grant; ~25 GPU-min at resync on a pass. Side findings CPU-only (draft level 0.54 ms; pool ~2x oversized; recurrent-state branches).
- 2026-11-11 PROMOTED T45-005: official 1173.4 (1174.7/1172.1) vs T37-002 1138.4/1146.1 (+2.7% cross-job, same hardware state); ABBA job 415 +3.83% all B>A; gates pass (GSM8K-250 94.8, TTFT p99 831, 128K); FP8 A/B job 416 +1.6 pt; serve_args same; diff = one dispatch condition in an already-modified file. Validity checked. New bar 1232.1. KB122 official line to update.
- 2026-11-11 T45 closed after its win (leader ready; reports/T45-writeup.md). ~20.7 returned.
- 2026-11-11 T49 first report (no proposal, no grant asked; 19.6 of seed unspent): T49-001 (PP with one decode microbatch, closed-loop sim: PP stays closed for latency reasons; KB123), T49-002 (compressible memory useless for model data; KB124). Leader has no idea that clears the bar; offered seed return/close; spar offered once if beliefs spar didn't cover the closing reasons.
- 2026-11-11 T49 closed (leader; T49-1 sparring covered the reasons; no proposal; KB123-124). 19.6 returned.
- 2026-11-11 T46-1 stage 1 done (bank/T46-001/002, 0.5 GPU-min): embedding in host memory exact (saves ~0.2% e2e; frees 1.27 GB/GPU); link adds bandwidth but caps ~2.4% of weight stream (~1% e2e ceiling); no stage 2 asked. KB125. Leader offers to close; spar offered once.
- 2026-11-11 T48 seed report (proposals/T48-1.md, bank/T48-001; 1.3 GPU-min): compression-ratio rule (KB126), five other directions killed at 0 GPU-min; leader recommends closing (~18.7 returned); optional k=6 per-position acceptance log (~3 GPU-min). Spar offered once.
- 2026-11-11 T46 closed (leader; sparring T46-2). Used 0.5 of 20; 19.5 returned. Open premises recorded in reports/T46-writeup.md.
- 2026-11-11 T48 closed (leader; spar on closing reasons used). Pointer recorded as KB127: k=4 on T45-005 plausibly break-even to ~+1% (unmeasured). 18.7 returned.
- 2026-11-11 T47-1 stage 1 failed its gate (bank/T47-001): split waves ~even/worse (J-P||D -4.9/-4.1/+0.65/+2.7 ms at P=64/128/256/512; gate >=+8 at 256); decode rows cost 10-16 ms inside a joined step (not a full pass). KB128. Leader recommends closing (14.5 GPU-min left) or a GPU-free pass with spar; spar offered once.
- 2026-11-11 T47 asks a ~2 GPU-min measurement (T47-002) from its own balance before closing: DRAM bandwidth vs SM count and the champion's decode GEMMs at reduced grids (is KB59's zero-sum a hardware limit or a property of today's kernels?). Agreed; no grant; spar not needed (covered).
- 2026-11-11 T47 closed (leader; T47-001 split waves even/worse KB128; T47-002 DRAM saturates from 16-32 SMs, KB59's zero-sum is kernel grid sizing, KB129). 14.2 returned. All teams closed.

## 2026-11-11 RUN PAUSED (all teams closed; not finalized)
- Research sets T42-T49 closed. Champion T45-005 1173.4 tok/s official (bar 1232.1). Pool: see gpu status. No agents or GPU jobs. reports/final.md predates T38-T49 (champion line outdated).

## 2026-11-11 records update (human request)
shared_kb.md (KB1-KB129) and graph.md already covered T42-T49; fixed graph's stale "T46-T49 running" line. Rewrote reports/final.md to cover T38 and T42-T49 with champion T45-005 1173.4 (bar 1232.1), pool 992.0 spent / 508.0 free. No new teams; run remains paused, not finalized beyond this report.

## 2026-11-11 config change (human)
"explore" renamed exploit-long (role leader_exploit_long, gpu type exploit-long). Opened exploit team T50 (slot A, leader-50, 100 GPU-min, first task with SPEED_OF_LIGHT pointer + inspiration T50-1). Pool free 408.0. Run resumed for T50; pause again when it closes.

## 2026-11-11 (human)
Opened exploit-long team T51 (slot B, leader-51, seed 20; SPEED_OF_LIGHT pointer; inspiration T51-1). Pool free 388.0.

## 2026-11-11 PROMOTION T50-002 + T51 funding
T50-002 promoted: 1181.3 official (T45-005 +0.67%), ABBA +1.04/+1.25%, gates pass. Validity flag accepted: T42's proof assumes a tensor-core error bound (2^8 margin over measured); reported to the human, who may override (VLLM_LMH_CERT=0). Bar now 1240.4. Previous champion saved agents/manager/champion.T45-005.md.
T51-1 funded: 25 GPU-min (start) for a bounded kernel-design step (+8% if it works, central +2%, odds of >=5% ~15%, measured/assumed split given, M1 failed on existing kernels). Funded under "when in doubt, fund".

## 2026-11-11 T50 closed
Win T50-002; remaining items <=0.4% e2e each; leader asked to close; no extra sparring offered (T50-1 covered). Returned 72.2 GPU-min; pool free 435.2. KB130 added. T51 still open.

## 2026-11-11 T51 closed
Kernel gate failed on measured evidence (ncu: tensor-issue limited at ~74% of roof); leader recommended closing, sparring T51-1 agreed so no offer. KB131. Not replaced: config has exploit-long 1 and no replacement instruction. ALL TEAMS CLOSED: RUN PAUSED (not finalized). Champion T50-002 1181.3 (bar 1240.4).

## 2026-11-11 (human)
Pool raised to 2000. Opened T52, T53 (exploit-long, slot F, Opus/Opus, direction megakernels word for word in first task, SoL, cards T52-1/T53-1). Pool free 926.5. Run resumed; pause when all teams close.

## 2026-11-11 ruling for T53
1/rms past W8A8 quant allowed with conditions (task_definition.md); human may override.

## 2026-11-11 T52-1 funded 50 GPU-min (mixed-step chunk pipelining; payoff +9.1% clears bar; parts marked measured/assumed). T53 keeps decode. KB100 (mixed AR latency-bound) is relevant; team to decide.

## 2026-11-11 T53 closed
Decode megakernel ceiling <1.5 ms/step; leader recommended; had a sparring session (T53-1), no extra offer. KB132. T52 still open (50+ GPU-min).

## 2026-11-11 PROMOTION T52-005
1212.4 official (+2.6% vs T50-002), ABBA job 477 +2.82% all B>all A, gates pass, diff reviewed (Python-only scheduling, same math up to fp reordering). Bar now 1273.0. Previous champion saved agents/manager/champion.T50-002.md.

## 2026-11-11 T52 resync
T52-006 rank-coupling fix negative; T52-007 host cost of eager waves 117 ms/step. Leader recommended closing (remaining ~+1-1.5
## 2026-11-11 T52 resync
T52-006 rank-coupling fix negative; T52-007 host cost of eager waves 117 ms/step. Leader recommended closing (remaining ~+1-1.5%); manager chose to continue: fund the lean wave loop (threshold 768 -> 384, est ~+1%) on the existing balance 31.9, no new grant, because T52 delivered a win, the pool has 887 free and no team competes. Not a closing decision, so no sparring offer.

## 2026-11-12 T52 closing request
T52-008 (host cost of wave steps 117 -> 85-100 ms/step, bitwise, 128K pass) too small; leader recommends closing (41.5 of 70 used, write-up reports/T52-writeup.md). Sparring T52-1 predated these reasons, so one sparring session offered; close pending its answer.

## 2026-11-12 T52 not closing yet
After sparring T52-2 the leader withdrew its closing recommendation: waves leave 186-353 us/layer above their no-comm floor, the link is not saturated, only one alternative schedule tested; conditional headroom ~6% e2e. Approved a <=0.5 GPU-min chain screen (3 waves vs 2, copy pieces at producer wave releases, M=384/736) on its balance 28.5; follow-up proposal if it halves the residual, else close.

## 2026-11-12 T52 closed; RUN PAUSED
T52-009 screen: 3rd wave nets -9% at M=736, 0 at M=384; threshold work +0.2-0.5%; leader recommended closing, sparring T52-2 covered the reasons. KB133. All teams closed: RUN PAUSED (not finalized). Champion T52-005 1212.4 (bar 1273.0). reports/final.md predates T50-T53.

## 2026-11-13 (human) research-handoff teams
Opened T54, T55 (research-handoff, slot A, seed 30 each = 20 + 10 moonshot top-up; memo research_handoff_1113.md in place of cards; SoL pointer; history included). Run resumed; pause when all teams close. Pool free 855.6 of 2000.

## 2026-11-13 T55 closed
Triage only (ideas 1 and 5 of the memo): already in champion or wrong workload; 0 GPU-min used, 30 returned. Sparring T55-1 had been held, no further offer. KB134. T54 still open (ideas 2, 6, 7).

## 2026-11-13 T54 closed; RUN PAUSED
Triage only (ideas 2/6/7): none clears +5% (premises wrong for this system); 0 GPU-min used. Sparring T54-1 held, no extra offer. KB135. All teams closed: RUN PAUSED (not finalized). Champion T52-005 1212.4 (bar 1273.0). reports/final.md predates T50-T55. Pool 2000, free 915.6.
