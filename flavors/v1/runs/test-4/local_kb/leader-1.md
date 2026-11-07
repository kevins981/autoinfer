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
