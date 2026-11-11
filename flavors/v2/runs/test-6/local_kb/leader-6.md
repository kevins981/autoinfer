# LOCAL_KB leader-6 (T6, explore)

## Orientation (2026-11-09)
- Champion T1-042 942.2. Built in-server decode fwd (T5 job 28 A / T3-008): M=3 17.46, 24 18.62, 36 21.58, 48 22.35 ms
  (~0.109 ms/row below 48); sample 1.13; propose 1.68 (2 draft steps). k=3 (T5 job 28 B): M=52 25.76, M=64 26.32
  => cliff ~3.4 ms just above 48 (Marlin 3->4 m-blocks?).
- Mixed steps (T3-008 timers, PIECEWISE): M 101-128 ~51 ms flat, 135-190 ~64, 197-244 ~80, then ~0.25 ms/row:
  steps of ~13-16 ms per 64-row slab = Marlin re-streams the weights per slab (hypothesis).
- Physical roofs: weights 12.15 GB/rank (~13 ms); bf16 tensor ~146 TF on GPU0 => 0.166 ms/row; ridge M~75-88.

## Dead ends checked (CPU only)
- N-gram / suffix (prompt + cross-request) drafting with MTP on T3's self-generated data: E +2-3% at +10-25% rows.
  (agents/leader-6/sim/ngram_sim.py). Not worth it.
- Chunked-prefill piggyback sim (agents/leader-6/sim/loop_sim*.py): pays only if the GEMM reaches >=85-95% tensor
  efficiency at M 64-128 and the mixed-step overhead is <=1-3 ms; kernel alone (whole prefill) +2-5% at eff .85-.95;
  piggyback adds +2-4% on top. With current Marlin: negative.

## Direction (candidate): single-pass W8A16 GEMM for M=49-256 + prefill piggyback
Task 1 (leader-6.w1, ws t6-gemm): Marlin vs roof microbench per M; config sweep at M=49-64; CUTLASS mixed-input search.
Coordinated with leader-5 (T5 owns spec policy; T6 GEMM side).
- T5-002 (job 29, leader-5.w1/mb): Marlin cold sum/forward M=48 15.56 ms, M=49-64 17.65 (+2.1 step); best configs 16.91 at
  M=64 => ~1.35 ms needs a kernel change (ours). Non-Marlin part of k=3's +3.97 ms ~1.9 ms (GDN/attn; T5 measuring).
  Agreement: T6 = GEMM side; T5 = spec policy, GDN-slot memory, drafter cost. T5 not patching marlin.cu.
- T6-001 (job 31): Marlin vs roof per M (table in record). 1.45-1.93x roof at M=72-112 (per-64-row launches each stream
  all weights); M=128 two slabs in one launch share L2 (25.9 < M=72's 25.4..112's 29.7). Large-M Marlin ~120-133 TF = roof.
- Re-sim with measured roofs: piggyback +5% only if eff>=0.95 of 133TF, rowc 0.08, eps 1; otherwise <=+2% or negative.
  Kernel alone on champion mixed steps ~+1-1.5%. Proposal T6-1 sent: payoff +5-6%, central +1.5%, odds 15-20%, 40 GPU-min.
- Task 2 (w1): parallel-slab host split in marlin_mm (ws t6-slab), gate G1: M=72-112 <=1.25x roof, M=64 <=16.0 ms.
- Funded: T6-1 got 40 GPU-min (start grant).
- Manager: resync credit = T6's own measured kernel gain + freed slack in same-job A/B; don't double-claim k=3 with T5; record needs per-M exactness check + KB61 build check.
- T5-003 (nsys in situ, M48->64): Marlin 13.28->16.28 (+3.0, bigger than cold +2.1; warm/prefetch effect), AR +38 us/row,
  rest +14 us/row. Re-sim with rowc 0.055: piggyback +2.7/+4.9/+7.0% at eff .85/.9/.95 (eps 1), ~+1-4% at eps 3.
  (whole-prefill+kernel column optimistic: phys mixed model drops measured mixed overhead.)
- Credit split agreed with leader-5: T6 = kernel gain at k=2 (+ piggyback); T5 = k=3/budgeted gains citing T6 kernel.
- T6-002 (jobs 33-34): parallel slabs in one launch: -3..-8.6 ms at M=72-120/176, M=49-64 unchanged, M=144 +1.4. G1 FAIL.
  gate_up 2-slab ~190 us at any M (even 2x32) = ~2x DRAM roof (93 us): slabs not time-aligned on weights for >L2 shapes.
  Task 3: slab-inner stripe ordering (2nd slab reads hit L2). Exactness: 1-ulp (reduction order) for re-tiled M.
- job 35 warm: M48 12.32, M64 14.67 (+2.35 warm step); AI_SLAB gains persist warm at 72-120 (-1.7..-7 ms).
- job 37 ncu gate_up: stock M64 119us 91.5MB tensor 38.6%; new M96 2x48 186us 135MB; M128 196us 119MB DRAM 66% tensor 42%, 255 regs 1 CTA/SM => latency/issue-bound; extra DRAM explains only 30-45 of +70us. Stripes already roughly time-aligned. Slab-inner ceiling maybe ~150us.
- T6-003 (jobs 40-43): slab-inner: M=96 27.6->19.0 cold (DRAM = 1 stream), 72 25->18.8, 144 33->26.7; per-shape policy needed. Next: option 1 (2 CTA/SM for 128-thr cfgs, may help M<=48 decode too), option 2 (256-thr tmb2-4), then deployable policy + in-server A/B.
- T5-005: GDN route B two-slot exact; k=3 fits 16 req at 5 GiB full grid; k=4 memory no blocker. T5 sim: decode +2% -> +11% across M=64 cost range. They want a stackable ws/record of our per-shape build; M=49-64 in situ is their key number.
- T6-004 (job 51): deployable policy ws t6-pol; occupancy probes null; stage-3 k64 kernels incorrect. Task 5: in-server A(stock) B(policy) C(policy+mnbt 96) ABCCBA 200 prompts.
- T5 will run in-situ timers k=3/k=4 on t5-rb + T6-004 diff, AI_POLICY on/off, results to us (cited T6-004).
- T6-005 (job 54): B policy ~0 (-0.7%, n=1 A); C mnbt96 -30% (TPOT 16->21, TTFT x3.3). Piggyback blocked by mixed-step overhead. Next: timers on B vs C to decompose mixed-step cost (host vs GPU).
- Task 6 refocused: per-step wall vs GPU (fwd+sample+propose) by bucket; T2-005 small-mixed anatomy was invalid; T3 fwd 51-64 vs T2 period 76 => 12-20 ms unexplained. Told T5 explicitly M=64 cliff remains (~+1.5-2 in situ); M=80 is where T6-004 pays.
- T5 CLOSED (KB63): in-situ M=64 +3.4 ms over M=48 even with T6-004 (-0.30 ms); deeper spec ceiling +1-2%. So T6-004's k=3 consumer is gone; T6 value now rests on mixed-step overhead (task 6). Plan: after task 6, resync; likely voluntary close + write-up unless task 6 shows a big host-bound pocket.
- T6-006 INVALID: t6-pol.t unbuilt (baseline _C). Next: nsys on BUILT t6-pol, C arm, ~4 GPU-min.
- T6-007 (job 57 nsys, built): mixed M=96 kernel floor ~42 ms vs decode 24.8 (Marlin in situ +9.1 ms for 48 rows); piggyback dead. Small mixed steps partly launch-bound (2583 vs 1041 kernels): exploit-scale. => recommend voluntary close.
- Resync requested: recommend voluntary close, return 31.2 GPU-min; write-up reports/T6-writeup.md; KB64 proposed.
