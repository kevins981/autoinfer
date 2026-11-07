team:         T1
performance:  "unofficial: microbench worse than T1-001 (no-go, not taken to the server)"
summary:
  hypothesis: Lower per-AR overhead than T1-001 at T=48 via finer-grained pipelining: (a) unified
              write-then-read blocks, (b) per-warp flags without __syncthreads (+ nanosleep backoff).
  change:     builds on T1-001. Variant kernel in ws leader-1.hostar2 (inactive by default); (a) was
              measured with the worker's first kernel (job 5).
  results:    Observed (CUDA graph, 128 ARs/graph, us per AR, T=16 / T=48):
              T1-001 split nb=64 S=1: 19.6 / 45.1.
              (a) unified nb=16 nt=128: 17.8 / 43.1; nb=64 nt=512: 18.3 / 43.8 (T=96 ~81).
              (b) per-warp best nb=32 nt=128: 23.1 / 48.8; nb=16 nt=256: 23.9 / 52.1; nb=64 nt=128:
                  32.4 / 55.5; bigger grids 60-190 us; nanosleep 256 ns nb=32 nt=256: 29.4 / 64.1.
              Earlier: S>=2 sub-chunks +3-5 us per sub-chunk. Concurrent write+read floor 29-33 us (491 KB).
  conclusion: Interpretation: flag polling over PCIe is expensive and contends with data traffic; many
              pollers hurt. Unified blocks give ~2 us; below the >=4 us bar for a new server run. The kernel
              is within ~10 us of the link's practical floor at T=48; further AR-kernel tuning has small upside.
validity:     not reviewed for an official eval (not taken forward); same math as T1-001.
