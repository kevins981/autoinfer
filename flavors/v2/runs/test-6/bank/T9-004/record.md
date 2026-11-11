team:         T9
builds_on:    T9-002/T9-003 harness (bench4.py = bench3.py + arms S3, S3c)
performance:  "microbench (job 86, 1.5 GPU-min): extending the champion's chunk chain with the next layer's qkvz GEMM per chunk (cores stay on full rows, join per layer, i.e. compatible with breakable graph segments) saves 72/84/209 us/layer (S3, champion ARs) and 130/149/272 us/layer (S3c, both chunks on copy engines) vs S2 at R=384/768/1024 (rank 0); full ping-pong P saves 242/348/459."
summary:
  hypothesis: mixed steps replay as breakable CUDA graphs (single-stream capture, GDN/attention core ops eager at replay), so a no-join cross-layer ping-pong cannot live inside the capture; part of its gain should be available inside one graph segment by letting each chunk continue into the next layer's qkvz before the join.
  change: none to the engine. S3: per layer, core on full rows (eager break in the engine), then per chunk on its own stream: out_proj -> AR (A: DmaStagedAllReduce 4 pieces, B: HostStagedAllReduce, as the champion) -> add+norm -> gate_up/silu/down -> AR -> add+norm -> NEXT layer's qkvz GEMM on the chunk rows into a full buffer; join; next core on full rows. S3c: same with both ARs on my copy-engine send/recv (2 lanes, 2 pieces).
  results: |
    Observed, rank 0 (rank 1 within ~10 us), us/layer, real core (6 GDN + 2 attention layers of 8), medians of 5 reps x 20 replays:
      R=384:  N 1245  S2 1653  S3 1581  S3c 1522  P 1411
      R=768:  N 2389  S2 3054  S3 2971  S3c 2905  P 2706
      R=1024: N 3148  S2 4094  S3 3885  S3c 3821  P 3635
    Saved vs S2: S3 72/84/209, S3c 130/149/272, P 242/348/459 us/layer. Hidden of (S2-N): S3 18/13/22%, S3c 32/22/29%, P 59/52/49%.
  conclusion: |
    Interpretation. The in-segment extension keeps 43-59% of the full ping-pong's gain over S2 without any cross-layer stream state, GDN state handoff or metadata slicing (cores unchanged, bitwise). Moving the second AR onto copy engines (S3c vs S3) adds ~60 us/layer. Full P needs two per-chunk capture lists replayed on two streams (not supported by the single-stream breakable capture) and doubles the eager core launches; its extra ~110-190 us/layer over S3c is an upper bound before that launch cost.
validity: measurement only. GPU 1.5 min.
