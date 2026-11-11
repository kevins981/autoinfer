# Round 5 report from leader-2 (T1): decode all-reduce
Best: bank/T1-020 (T1-019 + 2nd prefetch range + lm_head prefixes): official 850.3 tok/s (850.7/849.9), +5.5% vs T1-016 805.7,
+33.7% vs BASE; GSM8K 93.6%, TTFT p99 1373, TPOT p99 38.1, acceptance 2.336, all gates + 128K pass. Its increment over T1-019 is
borderline (same-job pooled +0.8%, cross-job official +1.6%); T1-019 alone is solid: official 837.0 (+3.9% vs T1-016; ABBA +4.5%).
Propose T1-020 as champion (exact, both official trials above both T1-019 trials, same-job pooled B mean above every A).

Experiments:
- T1-017 (T1-016) measurement: in-server %globaltimer split of each decode host AR. Rank 0 (pacing): 42 us = 21 write + ~0 wait + 21
  read/add, at the PCIe floor (~23.6 GB/s each way); rank 0 arrives last on 98% of ARs but skew only ~3 us; rank 1's ~7 us wait is
  slack. KB19's 53.6 us included nsys overhead. Wait/skew is NOT a lever.
- T1-018 (microbench): N-chunked row-parallel GEMM + AR pipeline on 2 streams: overlap works (AR co-runs with Marlin, recovers 70-75%
  of overlappable time) but each extra Marlin launch costs ~10 us and each extra AR ~9-12 us: best ~0.35 ms/step. NO-GO.
- T1-019 (T1-016): L2 prefetch of the next GEMM's FP8 weights (prefetch.global.L2::evict_last) by 8 extra blocks of the decode AR kernel:
  out-proj AR -> gate_up (89 MB), down AR -> next in_proj/qkv. Cache hints only (exact). Microbench layer 405 -> 368 us; in-server step
  29.9 -> 27.7 ms; ABBA 759.1 -> 793.6 (+4.5%); official 837.0.
- T1-020 (T1-019): + out_proj as a 2nd range in the down AR (evict_last lines survive GDN traffic: out_proj 29 -> 23 us microbench) +
  final-norm AR -> lm_head first 88 MB (target + drafter). 2 ABBAs pooled +0.8% (B bimodal: 808/798/811/799 vs A 796-799), TPOT -0.7%;
  official 850.3.

Proposed SHARED_KB:
- KB-a: Decode host AR on the pacing rank (GPU 0) = ~21 us write + ~21 us read at the PCIe floor, ~0 wait; arrival skew ~3 us.
  Skew/wait and transport are not levers; KB19's 53.6 us/call was nsys-inflated (real 42 us rank 0, 48 us rank 1). (T1-017)
- KB-b: The 8-block host AR kernel co-runs with Marlin GEMMs, but splitting GEMMs/ARs into chunks costs ~10 us per extra launch:
  chunked pipelines don't pay at decode sizes. (T1-018)
- KB-c: HBM idles during each decode AR; prefetching the consumer GEMM's weights into L2 (evict_last) from extra AR blocks saves
  ~2.2 ms of a 29.9 ms decode step (+4.5% same-job, official 837.0); evict_last lines survive the GDN state traffic. (T1-019, T1-020)
- KB-d: Decode step now 27.6 ms (rank 0): target layers ~24.0 (ARs ~6.0 incl. ~11 us/out-proj AR stretched by prefetch issue,
  compute ~18, mostly Marlin), lm_head ~1.3, drafter ~1.1, inter-step ~0.48. Cold gate_up streams at ~660 GB/s. (w9 traces)

Left open: T1-020's B bimodality (unexplained per-server variance); prefetch issue stretches the out-proj AR +11 us (x64 ~0.7 ms;
faster issue/split ~0.4-0.6 ms); drafter embed ARs/fc prefetch (~0.06 ms); M=16 cap (~2 us).

Next (one direction): Marlin decode GEMM efficiency, now ~18 ms of the 27.6 ms step (65%). Cold gate_up streams at ~660 GB/s vs the
4090's ~1 TB/s; L2 prefetch already showed bytes-from-L2 matter. A per-GEMM roofline at M=48 (bytes/time per shape, cold vs
L2-warm) tells the headroom; levers would be Marlin config/stages/split-k per shape, or a decode-specialized FP8 kernel (same math up
to fp32 reorder). Evidence: w9 traces + T1-016 profile; KB22 showed config choice moves gate_up 6 us.

Notes: GPU 60.0/80 (w9 41.7, evals 18.3); 20 unused. No GPU contention. w9 found/fixed a shared-JIT-cache hazard (all workspaces share
TORCH_EXTENSIONS_DIR; host AR ext build dir now named by source hash). Worker w9 stopped. Workspaces kept: leader-2.arsplit (T1-019),
leader-2.pf2 (T1-020).
