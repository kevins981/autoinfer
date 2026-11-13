team:         T53
builds_on:    T53-002
performance:  "measurement only: coupled 2-GPU, best (C2 codec) saves 9.7-12.3 us per down boundary vs champion hostar+codec+prefetch; gate C (>=15 us) FAILS"
summary:
  hypothesis: With chunked codec posting plus in-kernel peer readers, the down -> all-reduced-output time drops by
              >=15 us per boundary versus the champion (t31b + hostar with codec and L2 prefetch), including the
              next GEMM's benefit from the prefetch.
  change:     benchC.py / jobC.sh: one process drives both GPUs (bound to node-0 CPUs, host buffers verified on
              N0). Per rank: a host-flag barrier, then the chain, then stamps.
              Baseline: t31b cfg 292 + the champion's hostar (built from the champion source; grid 16, K 4, 512
              threads, mode 20 codec, 8 hint-prefetch blocks over the next layer's in_proj (qkvz 42 MB) + out_proj
              (15.7 MB)), then the qkvz GEMM (t31b cfg 292, the champion's warm pick).
              Fused: one t31c kernel (80 GEMM CTAs, 8 codec posters, 8 readers decoding + adding the peer chunk,
              8 prefetch CTAs issuing hostar's hint loop after the GEMM), then the same qkvz GEMM.
              The chain is down -> AR -> qkvz, not gate_up: in the model the AR after down prefetches in_proj and
              out_proj.
              Kernel fixes along the way:
              - Warm up all kernels on both GPUs first (job 450 died: CUDA lazy loading blocked a first-time
                launch behind the spinning barrier; 1.0 GPU-min).
              - Job 452: with 8 reader CTAs polling host flags, the GEMM ran ~3x slower. Fixed with a single host
                poller with __nanosleep backoff, relayed through a device word.
              Final: job 454 (1.0 min), 12 reps, cold.
  results:    Observed (job 454, medians r0/r1; baselines vary between jobs: champion down->AR was 108.5 in job
              452 and 122.9 here, so compare within the job only):
              - champion: down->AR 122.9/124.9 us; full chain (+qkvz) 159.2/160.8; qkvz 36 us (prefetched).
              - hostar codec without prefetch: 123.4/123.9; full 174.6; qkvz 51 us.
              - fused C2 codec + pf: down->AR 113.2/112.6 (saves 9.7/12.3); full 148.0/148.5 (saves 11.3/12.3).
                GEMM done 72/71 (single-GPU with posters: 62.5).
              - fused C4 codec + pf: 120.3/118.8 (saves 2.6/6.1); full saves 3.6/6.7. GEMM done 87.0/78.8
                (single-GPU: 66.6). Chunk-0 posting tail ~12 us.
              - fused C4, readers start polling only after the own GEMM is done: GEMM done 72.7/71.7, but
                down->AR 138.2/134.7 (-15.4/-9.7) because reading no longer overlaps.
              - fused C4 without prefetch: down->AR saves 10.2/6.1, but qkvz then runs 52-54 us (full -7.7/-10.2).
              - Numerics: every AR result is bitwise equal to the fp32 sum of the two bf16 partials, and the ranks
                are equal. C2 (kseg 1) is bitwise equal to the champion's AR. C4's kseg2 differs by max-abs 1.2e-4
                (summation order).
  conclusion: Interpretation: gate C fails. The best (C2) saves ~10-12 us per down boundary end to end (64 down
              boundaries per forward x ~11 us = ~0.7 ms of a 25 ms step; out_proj boundaries cannot gain, T53-001). The GEMM slows more on two GPUs (incoming peer reads + own posting:
              C4 1.18-1.31x its single-GPU time, C2 ~1.15x). Readers kept off the host restore the GEMM but lose the
              overlap: PCIe traffic is the floor, and posting moves it rather than removing it. The prefetch
              placed after the GEMM is as good as hostar's (qkvz 35 us in both).
validity:     measurement only (no engine change).

leader review (leader-53, 2026-11-11): microbenchmark only, no engine/serving change; nothing to check against TASK.definition. Numerics reported (bitwise vs stock where reduction order matches; AR results bitwise vs fp32 partial sums). Accepted as measurement.
