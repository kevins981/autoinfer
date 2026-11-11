team:         T5
builds_on:    T5-001 (t5-k3 + env-gated Marlin config override AI_M4_CFG for thread_m_blocks==4; microbench only)
performance:  n/a (microbench). Best per-shape tile at M=64 recovers 0.76 ms of a +2.1 ms Marlin step.
summary:
  hypothesis: the k=3 forward jump at M>48 (T5-001) is Marlin going from 3 to 4 m-blocks, and a tile override fixes it.
  change:     microbench only (agents/leader-5.w1/mb/mb_marlin.py), job 29, 0.3 GPU-min, worker leader-5.w1.
  results:    Observed: cold (256 MB L2 flush, graph of 8 flush+GEMM minus flush-only, median 25), per-rank FP8 shapes, us:
              | shape (layers)            | M=48  | 49    | 52    | 56    | 64    |
              | qkvz_in 8192x5120 (48)    | 51.6  | 55.7  | 55.8  | 55.9  | 56.2  |
              | gdn_out 5120x3072 (48)    | 22.8  | 27.8  | 27.9  | 27.7  | 28.0  |
              | attn_qkv 7168x5120 (16)   | 48.6  | 54.8  | 54.6  | 54.8  | 54.9  |
              | attn_o 5120x3072 (16)     | 23.1  | 27.8  | 28.2  | 28.0  | 27.9  |
              | gate_up 17408x5120 (64)   | 111.2 | 126.4 | 126.9 | 126.9 | 126.3 |
              | down 5120x8704 (64)       | 58.2  | 65.7  | 65.6  | 65.6  | 65.9  |
              Sum per forward: 15.56 ms (M=48) -> 17.63-17.67 (M=49-64, default). Overrides at M=64: {64,128,128} all 16.93-17.01;
              {128,64,128} 17.14-17.23; {64,256,256} 19.9; {128,128,256} 17.67; best per shape ({64,128,128} for gate_up/gdn_out/attn_o) 16.91.
  conclusion: Interpretation: the cliff is a step at M=49 (4 m-blocks), flat to 64, not a per-row cost. Configs recover 0.76 ms; ~1.35 ms
              remains (needs a kernel change; T6 / leader-6 is taking the GEMM side). Marlin explains ~2.1 of k=3's +3.97 ms forward
              delta; ~1.9 ms is elsewhere (nsys split in T5-003). Even with a perfect M=64 Marlin, plain k=3 gives only ~+4.5% decode
              before the memory fix => budgeted verification at M=48 (no cliff) is the better structure.
validity:     microbench; the override is env-gated and off by default; not a serving candidate.
