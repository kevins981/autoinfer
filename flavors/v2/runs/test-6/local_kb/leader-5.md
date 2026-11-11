# LOCAL_KB leader-5 (T5, explore)

## Orientation (2026-11-09)
- Rows model of the trial (T1-042, 96.3 s, 90.7K out, 104.5K in): prefill rows ~104K (~47% of rows, at bf16 roof),
  decode verify rows ~118K (23% wasted: 48 - 16x2.325 = 10.8/step). Everything is near a measured roof except AR
  (~16-20% of wall exposed, decode + mixed). T4 (leader-4) owns AR overlap (consumer-side deferred-norm dataflow).
- Official T1-042 trial1: mean TTFT 220 ms vs mean e2e 3766 ms => ~5.8% slot dead time; GPU0 2235 MHz/254 W, GPU1 2655 MHz/370 W.
- Built k=2 champion timers (T3-008 job 23): fwd M=3 17.44, 18 18.20, 27 20.54, 33 20.91, 36 21.72, 42 21.91,
  45 22.34, 48 22.40 ms; sample 1.12; propose 0.84/draft step. Base step 25.2 ms.
- KEY OPENING: T3 killed deeper speculation on (a) memory (k=3 13 GDN blocks/req, only 14 of 16 fit at 5 GiB) and
  (b) a "verify cliff above M=48" measured ONLY on an unbuilt workspace (KB61). KB20's k=3 null is from T1-015
  (FlashInfer, old stack). On the built stack, extra rows at fixed 16 requests should cost ~0.06 ms/row
  (Marlin ~0.036 + AR bytes ~0.04 at 0.78 codec) => k=3 step ~+2 ms (+8%) vs E +17.6% (acceptance .77/.56/.41
  in-server, T3-007). Estimate +5-9% decode throughput. UNMEASURED.
- Direction candidate: "deeper speculation on the built stack": k=3 at 16 req (memory via graph trim or GDN slot
  rework), then confidence-budgeted K_MAX=4 at M<=64 (T3-2 design), cheaper draft steps.

## Probe 1 (leader-5.w1, agents/leader-5/w1_task1.md)
k=3 + trimmed prefill graphs (cap 1024) + KV 5.75 GiB vs champion k=2, same job B,A,B, 200 prompts. Pending.
- T4 closed (KB62): comm vs memory-bound GEMM is zero-sum; removing an AR worth only 24-28 us (prefetch recovers ~18).
  T4's cost model agrees: remaining lever = fewer verify rows per output token (speculation). Supports T5's direction.
- T5-001 (job 28): k=3+cap1024 -1% vs champion; cap1024 alone -4%; k=3 fwd +3.97 ms at M=64; step jump already at M=52 (13 req)
  => cliff at M>48, not per-row. Task 2 (worker): microbench Marlin M=48..64 per shape + tile overrides for 4 m-blocks; GDN/attn 3 vs 4 tok.
  If cliff fixable to ~+1 ms: k=3 step ~27.2 (+8%) vs E +16% => ~+7% decode; memory must come via GDN slots (route B), not graph cap.
- T5-002 (job 29): Marlin cliff at M=49 (+2.1 ms/forward), best tile override recovers 0.76. T6 takes GEMM side (agreed). Non-Marlin k=3 delta ~1.9 ms: T5-003 nsys pending.
- T5-003 k=2 nsys (job 30, 3.8 min; agents/leader-5.w1/prof/k2/): step 25.42 = fwd 23.52 (Marlin 13.28, hostar 5.97, GDN 1.85,
  lmh 1.02, norms .47, attn .41, bf16 .37, act .20) + sample .12 + draft 0.88 x2. Draft step at bytes roof: draft lm_head 310 us
  (252 MB), MTP Marlin 200-225, fc ~80, AR 110-170; ~no gaps. => "cheap drafter" only via smaller draft vocab (~-0.2 ms/step, costs E).
- Napkin: budgeted K_MAX=3/4 at M=48 with realistic drafter => ~+3-4% e2e (below bar). Only route above bar: budget at M=64
  AFTER T6 fixes the cliff, if the non-Marlin k=3 delta (~1.9 ms) is mostly per-token-of-k fixable => ~+6% if works. k=3 profile running.
- T5-003 (jobs 30,32): k=3 fwd delta = Marlin +3.0 (in situ > cold 2.1), hostar +0.6, rest +0.3. Drafter at byte roof. Balance 3.0.
- Proposal proposals/T5-1.md sent (payoff +7% T5 alone / +11% w/ T6 kernel; central +3.5%; 80 GPU-min). Credit split with T6 agreed:
  T6 = kernel gain at k=2; T5 = k=3/budget gains citing T6 as dependency. Worker Task 4 (GPU-free): route B code read + M=64 budget sim.
- FUNDED 80 GPU-min (manager notes: repeat cap penalty; memory-change proofs incl. route B bitwise at 1600-token boundaries;
  cite T3-005..008 for K_MAX=4; override max-abs diff). Task 5 job: A, B'(k=3+cap1792/1536+override, KV5.75), C'(k=2 same cap), B', A + 128K + override numerics. Allowance 15.
  Resync after step 2's ABBA.
- Task 4: route B feasible at 2 slots/group (S0 committed + S1 log): 7 blocks/req for any k (k=3 13->7: frees 2.6 GiB at 16 req) => k=3 fits at KV 5.0 with full grid.
  Sim (sim_m64.txt): vs k=2: fixed k=3 0.987 (cliff +3.0) / 1.081 (+0.4); budget Kmax4 M=64 1.020 / 1.113; budget Kmax4 M=48 1.032 (no cliff).
  => without T6, the M=48 budget (+3.2%) is T5's best decode-step lever; M=64 needs T6. Task 6: implement route B in ws t5-rb.
- T5-004 (job 36): k=3+override+cap1536 = -3.5% pooled vs champion; cap1536 alone nil (+0.4%); override -0.4 ms in situ (half of cold);
  not bitwise (ulp reorder). Fixed k=3 dead without T6. T5-alone path left: budget Kmax4 at M=48 (sim +3.2% decode => ~+2% e2e) needs
  route B (Kmax4: 16 blocks stock). M=64 budget (+11% decode) needs T6. Balance 67.7.
- Manager: resync will judge route B exactness + ragged-verify prototype + whether stacked payoff becomes real with T6's kernel
  (K_MAX=4@M=64 counts as T5's lever if T6 lands). Route B exactness job approved (8 GPU-min); Task 7 budget design next.
- T5-005 (job 39): route B bitwise-exact (4 gens across 1600/3200 boundaries + prefix hit); k=3 16 req at KV 5.0 full grid; ~1.00x. Budget design approved, K_MAX=3@M=48 first (T3 sim 1.046). Balance 59.4.
- T6-003: 2-slab single-read Marlin: M=80 19.0 cold vs 15.45 at M=48 (+3.5); M=64 est ~15.9 via 2x32 mix (unmeasured). In situ cliff was 1.4x cold — ask for warm/in-situ numbers.
- Task 7: P2 fused argmax+sumexp conf kernel free (4 us in graph vs 16 us old path); P3 upstream assign_draft_token_budget OK. Sim K_MAX=3 M=48 iso 1.041 > K_MAX=4 1.037. Next: wire P2 + forced-pattern ragged exactness at c=4 (4 GPU-min). Balance 59.3.
- Job 48: ragged adaptive path (upstream adaptive_verification + MTP + local argmax + two-slot + TRITON_ATTN) runs, varlen graphs captured, 32 n_i patterns at M=12. 3/4 reqs diverge vs c=1 k=2 ref, all at bf16 ties/1-quantum gaps. Next: champion c=4 control + VLLM_TEST_GREEDY_IDS_CHECK on ragged (3.5 GPU-min). Balance 55.2.
- T6-004 (ws t6-pol, AI_POLICY): M=49-64 only -3% (cold 17.13->16.61 vs 15.45@48); M=80 25.3->17.8 cold. Est M=64 budget K_MAX=4 ~+7% decode w/ in-situ factor. Re-sim when T6 in-situ A/B lands.
- Job 53: ragged generic-path verify internally consistent (1853 rows, 0 mismatches); greedy-ids fast path untested (client used logprobs). Folded into Task 8 job run 0. Balance 50.5.
- Task 8 job approved (13 GPU-min): run0 greedy-path ragged check; k2_on; bud3 (THE candidate: k=3 budget 2/req M=48 real conf); k3_on; k3_off (T6 policy in-situ).
- T6: M=64 still +1.0 cold/+1.6 warm over M=48 with policy; M=80 +2.2 cold/+6 warm. Napkin: K_MAX=4@M=80 loses; budget K_MAX=4@M=64 ~+6% decode if in-situ ~+2 ms. T6 chunk-piggyback -30% (no contention for rows).
- T5-006 (job 55): bud3 (K_MAX=3, M=48, real conf) -4.8% single run; timer model +1.7% (E +5.6% vs sim +8.1%; step +3.8%);
  mixed steps slower under adaptive path (mean ITL +9.5%, TTFT +17%). T6 policy in situ only -0.3 ms at M=64 (cliff +3.4).
  => Direction ceiling ~+1-2% e2e. Below bar. Recommend resync -> voluntary close + write-up. Balance 38.9.
- Write-up drafted: agents/leader-5/T5-writeup.md. Waiting on manager's resync decision.
