team:         T52
builds_on:    T50-002 (chain microbench; engine flag VLLM_T52_LINKPRI added in workspaces/t52-w1, default 0, not yet server-tested)
performance:  "chain microbench (M=384/736, NL=4-8): link priority -63..-70 us/layer; waves+FIFO link priority -174 (384) / -230 (736) us/layer vs joined+LINKPRI (~-240 vs champion-like) (jobs 469-473, ~1.6 GPU-min)"
summary:
  hypothesis: mixed-step ARs (copy-engine DMA lanes, token-split 2 chunks) are exposed because of scheduling, not transport; fixing link order and
              removing the per-layer join hides most of it.
  change:     agents/leader-52.w1/m2/ (m2.py/m2.sh harness: 2-rank chain of GDN-layer tails with the champion kernels, token split 192/192 or 368/368,
              DMA AR lanes; tl.py trace parser). Engine: VLLM_T52_LINKPRI in _qwen3_5_split_chain (vllm/models/qwen3_5/nvidia/model.py): chunk 1 starts
              after chunk 0's out GEMM; chunk 1's AR1/AR2 wait for chunk 0's (events only).
  results:    Observed (us/layer, median of 7): M=384 NL=4: E 1198, G (concurrent halves = server) 1083 (no-comm 630, exposed 453), S 1073, P (link
              priority) 1020 (exposed 312), Q 1041. ONE D2H copy engine: the two lanes' pieces interleave so both ARs end together 230-290 us after the
              out GEMMs with nothing beside them (= the in-server window, agents/leader-38/prof_champ_node). Pieces 0.49 MB in 21-23 us (~21 GB/s),
              7-8 us gap; gap fixes P2 1020 / P3 1010 / event-doorbells 1121 (no gain). Job 473 NL=8 with a 60-70 us stand-in core: M=384 J 1158,
              W 1219, WP 984 (floors Jn 833, Wn 765); M=736 J 1894, W 2026, WP 1664 (Jn 1369, Wn 1306). WP remains 220/359 above its floor; much
              of it is the peer's D2H lag (~100 us, rank skew). J bitwise eager; W/WP differ by M-dependent rounding of the stand-in.
  conclusion: Interpretation: in mixed steps the AR exposure comes from copy-engine interleaving + the per-layer join, not the link. Waves without
              link priority lose (reproduces T30/KB128); with FIFO link priority they gain 12-15% of the layer. If the host enqueue of wave steps is
              hidden (T32 captured core port), est. +3.5-4.5% e2e on the >=384-row steps (21.7% of wall). Next: port waves to T50-002 (T52_WAVE).
validity:     microbench; LINKPRI is stream ordering only (bitwise by construction, to be server-checked). Not for official eval yet.
