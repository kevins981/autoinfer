team:         T9
builds_on:    T9-002 (bench3.py = bench2.py + stamped P timeline, PD, PU arms; same ce.cu/job.sh pattern)
performance:  "diagnostic microbench (job 84, 1.2 GPU-min). The core dependency is NOT what limits P: P-nodep (PD) is no faster than P (R=768 2730 vs 2716 us/layer; R=384 1500 vs 1416). Uneven split (A=1/3) is worse (1585 / 2859). In-P copies run at 8-14 GB/s (lone 20-24), each AR takes 300-500 us at R=768 and the chunks' stagger is only ~130-270 us."
summary:
  hypothesis: P hides only ~50% (T9-002) instead of T9-001's 86% (core=0) because (a) B's core waits for A's core each layer so both chunks' ARs fall in phase, or (b) the real GDN kernels contend for HBM with the copies / GEMMs.
  change: none to the engine. bench3.py. Real core as in T9-002 (6 GDN + 2 attention layers of 8). Arms (interleaved, 5 reps x 20 replays, rank 0 shown): N, S2, P (as T9-002), PD (P without the B-waits-A-core event: math wrong, ceiling only), PU (A = 1/3 rows, 64-aligned: 128/256 rows at R=384/768), plus a stamped P graph (globaltimer stamp kernels on the compute and copy streams, layers 1 and 2, perturbs slightly: layer time ~3.1 ms vs 2.72 ms unstamped at R=768).
  results: |
    Observed, us/layer rank 0 (medians; first two reps of the R=384 block were slower for every arm, medians unaffected):
      R=384: N 1245  S2 1652  P 1416  PD 1500  PU(A=128) 1585   -> hidden (S2-P)/(S2-N): P 58%, PD 37%, PU 16%
      R=768: N 2389  S2 3051  P 2717  PD 2730  PU(A=256) 2859   -> hidden: P 50%, PD 49%, PU 29%
    Stamped P timeline R=768, layer 1, us from the first stamp (chunk A | chunk B, 384 rows each):
      qkvz        0-250 | 272-599          (a 384-row qkvz GEMM runs ~250-330 us while the other chunk's work overlaps it)
      core        250-593 (343 us) | 599-792 (193 us)     (GDN alone for 384 rows is ~70-90 us per T9-002: 2-4x slower when overlapped)
      out_proj end 705 | 894
      AR1 D2H     710-1019 (309 us = 12.7 GB/s) | 897-1402 (505 us = 7.8 GB/s)
      AR1 H2D     784(after doorbell)-1061 (13.5 GB/s) | 1056-1403 (11 GB/s)
      gate_up end 1939 | 2402 ; down end 2541 | 2669
      AR2 D2H     2544-2844 (13 GB/s) | 2672-3108 (9 GB/s); H2D 2547-2843 | 2703-3108
      layer end   2866 | 3140
    R=384 layer 1: AR1 D2H 137 us (A, 1.97 MB = 14 GB/s), 210 us (B, 9.4 GB/s); AR2 192 / 266 us (10 / 7.4 GB/s); chunk B trails chunk A by ~150 us all layer.
    Within a layer both chunks' down GEMM ends 128 us apart (A 2541, B 2669), but each AR takes 300-440 us, so after the down GEMM there is a window (A: 2541-2867, B: 2669-3140 at R=768) in which that chunk waits for its AR while the other chunk has only 0-270 us of independent work left; the idle GPU time is what P leaves exposed.
  conclusion: |
    Interpretation. (1) The dependency is not the explanation: removing it (PD) gives no speedup at R=768 and makes R=384 worse (the chunks then run even more in phase, B starts as soon as its qkvz ends). (2) An uneven split makes it worse (PU), so a bigger chunk A does not buy a better phase. (3) What the timeline shows is that the copies inside P run at 8-14 GB/s, not the 20-24 GB/s of a lone copy, so each AR of a 384-row chunk takes 300-500 us, longer than the other chunk's independent segment; the stagger between chunks is only ~130-270 us. This is consistent with contention for HBM/SMs between the copy engines (their reads/writes of device memory) and the concurrently running Marlin / GDN kernels, plus A's and B's copies in the same direction sharing one link (e.g. R=768 AR2: both D2H spans overlap and total 7.9 MB in 564 us = 14 GB/s). I did NOT directly isolate "HBM contention from the real GDN": that would need P with the real core vs P with no core in the same job (T9-001 core=0 had similar copy rates in P, 7-20 GB/s, yet hid 86% at R=768; the T9-002 core is only ~+100 us/layer in N but costs +340 us/layer in P vs T9-001 core=0, so the real core's kernels interact with the schedule more than their own duration suggests). So: dependency ruled out; "copies too slow relative to the short stagger, with the real core's kernels making it worse" is supported; the exact split between GDN HBM traffic and Marlin traffic is untested.
    Napkin: link-bound floor per layer at R=768: 4 ARs x 3.9 MB = 15.7 MB each direction at 20 GB/s = 785 us (33% of the 2.39 ms layer), so hiding is not link-limited, it is schedule/latency limited.
validity: measurement only. GPU use 1.2 GPU-min (balance 72.1 -> 70.9).
