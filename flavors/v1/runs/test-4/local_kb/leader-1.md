# leader-1 LOCAL_KB (team T1)
## Round 1: R1-A host-staged AR (no P2P)
- ws leader-1.hostar, worker leader-1.w1 (30 GPU-min allowance; ~22 kept for official eval(s)).
- Facts carried in: NCCL graph T=16/48/96 = 41.6/65.5/101.7 us; link 25/23 GB/s; max prefill chunk 2048 tok.
- Budget raised to 80 GPU-min (manager). T2-003 (flags) 653.9 official; orthogonal, may test on top later.
- w1 microbench (jobs 5,8,10): host-AR graph T=16/48/96 = 19.6-24.9 / 45-47 / 80-84 us vs NCCL 43/68/100. Ties NCCL at T>=256 -> dispatch T<=128.
  Bitwise = NCCL (42 cases, 1500 alternating-size calls, graph replays). NCCL_PROTO=Simple at prefill sizes: no gain (eager).
  Link is not full-duplex: concurrent write+read of 491KB = 29-33us (vs 18.7 / 21.5 alone). Floor ~34us at T=48.
  Each fence.sys + flag costs ~2.5us. GO given; worker allowance raised to 45.
- job 11 A/B (pre-review-fix code; bench 100 c=16): OFF 601.3 tok/s, TPOT med 24.04 / ON 626.0, 23.28 (+4.1%); AL 2.355 vs 2.350.
  gsm8k50 44/50; long_request 128K ok. Dispatch: bytes <= 2MiB, nb=64 nt=256 S=1. Second pair (ab2) pending.
- job 12 pair 2 (final code): OFF 600.9 / ON 632.3 (+5.2%). Filed T1-001; official eval launched.
- w1 next: ws leader-1.hostar2 (from T1-001): unified write+read blocks / per-warp flags; 4-config A/B incl. T2-003 flags (stacking).
- T1-001 OFFICIAL: 672.5 (669.1/675.8) +5.7% vs BASE, GSM8K 93.6, TPOT p99 41.9, TTFT p99 1381, 128K pass.
- T1-002 (null): unified blocks -2us; per-warp flags worse (PCIe polling contention). AR kernel ~10us off floor at T=48.
- job 15 stacking: BASE 591.5, AR 625.9, T2 613.9/622.5, AR+T2 657.1/647.8. Filed T1-003 (=T1-001 diff + T2-003 flags).
- T1-003 OFFICIAL: 691.3 (696.1/686.4) +8.7% vs BASE; GSM8K 92.8, TPOT p99 46.1, TTFT p99 1303, 128K pass.
- T1-003 nsys (job 17): decode span 39.5->33.0 ms, AR 8.10->5.88 ms/step (44.5 us/AR avg). Step period only 39.8->37.0:
  ~4 ms/step now between decode sub-steps (small clusters of lm_head Kernel2 + allgather + drafter), idle 5.3%->10.8% (under nsys; unverified without it).
  Prefill steps unchanged: NCCL AR ~31% of big prefill steps (181 of 590 ms); NCCL prefill AR = 8% of span; host AR 12.4%; Marlin 51%.
## Round 2: overlap prefill-step AR with GEMMs (champion T1-003 691.3)
- Engine: compilation mode NONE, graphs <=96 tokens; prefill steps eager. AR happens in the norm (use_fused_allreduce_norm fallback), not RowParallelLinear.
- ws leader-1.overlap (from T1-003), worker leader-1.w2, 30 GPU-min allowance; budget 55.
- w2 mb (job 18): T=1024 GEMM K=3072 265us, K=8704 676us, NCCL AR 757/1485us (T=1024/2048). Pipelined: -13..-21% per layer (<30% bar).
  Independent GEMM||AR barely overlap at T=1024 (1400 vs 1428): Marlin persistent grid = all SMs blocks NCCL CTAs; high-prio stream no help.
  Chunked GEMM bitwise 99.94-100%, max 1 ulp. Next: Marlin sms reserve (needs C++ build) -> retest.
- Marlin SM reserve (env VLLM_MARLIN_SM_RESERVE, getenv per call) fixes overlap: conc K8704 T1024 1404->751us. Pipeline n=4 R=2: -23% (T1024) / -25% (T2048) per layer; Marlin@126 SMs +2%.
- job 23 A/B (bench 100, T1-003 args): OFF 646.3/643.2, ON 649.4/654.2 (+1.1%); TTFT p50 ~250 vs 254-272; TPOT p50 22.2 vs 23.0. gsm8k50 44/50, 128K ok.
- Confirm job: 400-prompt OFF/ON/OFF/ON requested (~24 min). Then official if positive. Diff consumers marker-aware (norms + mtp); aux-hidden path (eagle3) would double AR - inactive here.
- job 25 bench-400 OFF 682.4/692.5 ON 686.1/688.2 => 0.0%. TTFT p50 +6% ON, p99 better. Null. Mismatch vs napkin ~3%: profiling.
- T1-004 null filed; nsys: big prefill steps -10%, -3.8% of profile, cancelled by step-mix variation. Only down_proj hides AR; o_proj chunks too short. w2 stopped.
