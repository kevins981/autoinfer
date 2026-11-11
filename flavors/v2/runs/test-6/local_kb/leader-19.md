# leader-19 (T19, explore) notes
## Orientation (CPU, 2026-11-09)
- Read KB1-75, T2/T4/T7/T9/T11/T14/T18 write-ups, T17-1. Consensus: decode on coupled HBM/tensor/PCIe floors;
  T17 owns the M<=64 decode GEMM + k=3.
- Candidate direction: the "new-request path": slot dead time (KB75: server TTFT 204 ms mean vs ~95 ms mixed
  step; refill at N+3) + small-mixed-step excess (T2-006: pt<150 steps 58 ms GPU span vs 24 decode; T7: ~30 ms
  excess/step x 215 steps) + coalescing (T7 napkin +1%).
- Worker leader-19.w1 started on L1 (task agents/leader-19/w1_task1.md): in-situ lifecycle + per-step timing
  trace on T14-004, ws t19-champ, allowance 8 GPU-min.
- Job 132 (L1 try 1): 12 GPU-min burned, no data (engine-core hook bug -> bench hung to timeout). Rerun at --timeout 8 uses the rest of the seed. Lesson: CPU-test hooks + engine-death watchdog before GPU.
- T17 closed (KB76: cold GEMM benches must flush with reads; their M=64 headroom was an artifact). Decode GEMM lane now unowned but likely dead. T20 opened.
- L1 done (T19-001, job 134, 3.0 min): decode 25.3 ms (86% of steps); mixed = 38.5 + 0.229 ms/ptok (R2 .98), 40% of wall;
  <64-tok steps 49.7 ms; refill at N+3.23 but dead slot-time only 1.1% -> lifecycle dead (<=1%). Drain 2.6%.
  p99 TTFT = ramp burst (first 16 arrivals); steady p99 0.41 s. Coalescing: saves ~13 ms/merge vs ~10 ms wait cost -> ~0.
- T14's B1 nsys (champion): ar_un in mixed steps large (R>=600: 77 ms union, ~37 overlapped w/ Marlin -> ~40 exposed of 254).
  Napkin: exposed prefill-size AR ~8% of wall (small 3.3 s, mid 1.2, large 3.8). L2 (CPU) = proper priority attribution.
  Caveat: T9 says per-chunk cores <=1%; more chunks hides ~half the remaining -> ~2-3%.
- L2 (CPU, B1): mixed excess over decode 33.7 s = Marlin 22.0 / comm 6.8 / gdn 1.6 / norm 1.4 / idle 1.5; decode comm overlap 0. Recommended voluntary close; write-up agents/leader-19/T19-writeup.md; 5.0 GPU-min returned.
