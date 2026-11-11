# Round 1 report from leader-1 (T1), as received (lightly formatted)
Best: bank/T1-005, official 699.2 tok/s (+10.7% vs BASE; 3 trials, CV 0.4%), GSM8K 93.2%, TTFT p99 1309 ms, TPOT p99 50.6 ms, gates pass, 128K re-tested on exact config.
Experiments: T1-001 (prefill-graph flag only, unofficial +2.2%, noise); T1-002 (host-staged AR, official 662.8, +5.0%); T1-003 (diagnosis, nsys + overlap microbench); T1-004 (+ host-staged all_gather, unofficial ~0); T1-005 (T1-004 + prefill-graph flag, official 699.2).
Proposed insights: see shared_kb.md (all accepted, entries KB1-KB7).
Left open: large-M AR at ~12-14 GB/s vs ~20 GB/s ceiling (copy-engine path untried); prefill-step rank skew ~214 us/AR unexplained (GEMM/attention imbalance or piecewise-graph eager parts; full mixed-batch graphs untested); prefill steps stall decodes (ITL p99 ~290 ms); logits diet, L2 prefetch, Marlin retune untried.
Notes: first host-AR version had a latent hang (fixed by reasoning; the race test couldn't reproduce it). JIT .cu compiles at server start (~30-60 s cold). Spinning kernel needs all 8-64 blocks resident; both ranks must issue the same collective sequence.
Budget: 46.6/80 GPU-min used. Workspaces: leader-1.prefillgraph, leader-1.hostar (T1-002), leader-1.hostar2 (T1-004/005).
