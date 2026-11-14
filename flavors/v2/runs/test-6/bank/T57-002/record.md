team:         T57
builds_on:    T52-005 kernels (microbench only, no engine change)
performance:  "microbench: spatial prefill lane fails its gate (no f gives lane >=80 TF and decode <=1.3x together)"
summary:
  hypothesis: (sweep #1) the zero-sum of KB59 is a kernel-sizing property (KB129); a W8A8 prefill lane on f SMs and DRAM-bound decode on 128-f SMs can co-run with both near their rates. Gate (closed-loop napkin + sparring T57-1): lane >=80 TF AND decode <=1.3x simultaneously.
  change:     none to the engine; microbench agents/leader-57.w1/task2/mb/corun.py, GPU0: lane = quant + bs_gemm_marlin M=512 on 4 TP shapes in a green ctx of f SMs; decode = (a) T47 cp.async streaming read, (b) champion M=64 kernels (t31 qkvz/gdn_out/down, gate_up path, a norm).
  results:    Observed (jobs 486 smoke, 488 full, 1.1 GPU-min; agents/leader-57.w1/task2/REPORT.md; single run, 1.5 s windows):
              lane alone 74/106/144 TF at f=32/48/64. Decode alone: (a) 958 GB/s; (b) per layer 232 us (qkvz 48.5, norm 4.8, gdn_out 22.2, gate_up 101.9, down 54.4).
              Co-run, decode on plain streams: f=32 decode 1.20x / lane 57 TF; f=48 1.40x / 85 TF; f=64 1.62x / 115 TF. At f=48 qkvz 1.62x, gate_up 1.50x, down 1.15x, gdn_out 1.14x.
              Decode also in a green ctx: 1.52x/1.86x/6.8x (worse). Ideal stream (a) keeps 881-914 GB/s co-run but the lane collapses to 12-39 TF.
              Grid cap with lane idle: (a) at 128-f SMs 955-956 vs 958 GB/s (no cost). Real t31 grids are fixed by shape; each GEMM alone in 128-f SMs: qkvz +45% at every f; gate_up 104 (f=32) / 128-130 (f=48,64); down, gdn_out flat to f=48.
  conclusion: Interpretation: contention is DRAM/L2 arbitration, not SM count: a pure stream keeps its bandwidth and starves the lane, so T2's zero-sum is not only kernel sizing. At f=48: lane 0.80 of solo + decode 0.71 = 1.5 GPU-equivalents (some overlap), but below the gate. Closed-loop napkin at f=48 (lane 85 TF -> ~90 ms/prompt, rho~0.54, ~0.86 slot idle, decode s=0.40 during lane): ~+2% e2e before lane ARs, GDN/attention cores and in-server interference (T2 found in-server ~3x the microbench). Candidate #1 closed for T57 unless T56-001's in-server probe disagrees.
              Caveats: no repeat; continuous decode loop (upper bound on slowdown); lane without AR; GDN replay and attention not driven.
validity:     microbench only.
note (leader-56, T56): on T56's metric, the cost per lane prompt token (1-D)/r against ~0.165 ms in the joined mixed step (go at <=0.11), these co-run numbers pass on GEMMs alone: 0.071 / 0.082 / 0.081 ms/token at f=32/48/64. The two metrics differ because my closed-loop napkin also charges the slot that sits idle while its prompt waits in the slower lane (queueing). T56-001's in-server probe decides; T57 is not building the lane meanwhile.
