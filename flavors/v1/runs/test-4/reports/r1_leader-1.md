# Round 1 report, leader-1 (T1, R1-A host-staged all-reduce) — received 2026-11-07
Best: T1-003 = T1-001 diff (host-staged 2-rank AR, bitwise = NCCL) + T2-003 serve_args. OFFICIAL 691.3 (696.1/686.4), +8.7% vs BASE,
+5.7% vs T2-003; GSM8K 92.8, TPOT p99 46.1, TTFT p99 1303, 128K pass, AL 2.346.
T1-001 host AR alone: OFFICIAL 672.5 (669.1/675.8), +5.7%; same-job A/Bs +4.1/+5.2/+5.8%.
T1-002 lower-overhead AR variants: null (microbench only; unified blocks -2 us; per-warp flags worse).
Same-job job 15 (bench 100): BASE 591.5; AR 625.9; T2 flags 613.9/622.5; both 657.1/647.8.
KB proposals: AR microbench table; PCIe not full-duplex for SM zero-copy (T=48 floor ~34 us, kernel within ~10 us);
NCCL_PROTO=Simple no gain at prefill sizes; T1-003 anatomy (decode span 39.5->33.0 ms but step period 39.8->37.0 ms, ~4 ms/step
inter-cluster idle, GPU idle 5.3->10.8% under nsys; prefill AR still NCCL, 31% of big prefill steps, 8% of span; Marlin 51%).
Left open: prefill ARs on NCCL; the ~4 ms/step idle; AR+residual+RMSNorm fusion; one-rank construction-failure robustness.
Next: overlap prefill-step AR with compute (2 micro-batches; copy-engine AR on side stream), est +3-5% (assumed).
Alternative: verify the 4 ms idle without nsys (possibly ~10%, in T2's drafter/sampling path).
GPU: ~51/80 used.
