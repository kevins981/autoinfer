team:         T1
builds_on:    T1-035
performance:  "unofficial: within noise / no gain (microbench 49.7 -> 50.4 us/layer at 16 reqs, cap 8)"
summary:
  hypothesis: The lazy GDN replay kernel's grid (384 blocks against 256 slots at 2 blocks/SM, i.e. 1.5 waves) wastes a partial wave. Fitting 3 blocks/SM (1 wave) would cut its per-layer time.
  change:     ws leader-2.lazygdn.c3 (w21): removed shared-memory state staging (register prefetch one chunk ahead), __launch_bounds__(256,3), ~12 KB smem -> 3 blocks/SM.
  results:    Observed:
              - nsys (job 144, T1-035 server, c=16, GPU 0): lazy kernel median 49.7 us at 16 requests (p10 47.6, p90 51.8); 32.5 / 31.2 / 18.7 us at 11 / 8 / 4 requests; flush kernel 3.7 us per call.
              - (a), job 146, bitwise vs current over 40 seeded steps. us/layer alone / with GEMM:
                cap 8: current 49.7 / 150.8, (a) 50.4 / 151.6
                cap 0: current 65.2 / 165.2, (a) 62.8 / 164.5
              - Microbench at 8 / 11 / 16 requests: current 26.6 / 38.9 / 49.7, (a) 26.6 / 39.3 / 50.4.
  conclusion: Interpretation: the kernel is bandwidth-bound, with time linear in the number of blocks; the uneven last wave is not what limits it. GDN is at its floor for exact changes. Untested: the step from ~32 us at 11 requests to 49.7 us at 16 looks like an L2 effect. Not pursued.
validity:     bitwise identical (w21's test); not proposed for official eval. No precision or serving change.
