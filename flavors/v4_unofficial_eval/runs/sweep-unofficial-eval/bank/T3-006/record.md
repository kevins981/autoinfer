team:         T3
builds_on:    T3-004
performance:  "microbench only: no-go (job 200)"
summary:
  hypothesis: mixed steps cost ~28 ms fixed + ~0.17 ms/prompt token (unprofiled, job 198); the serialized host AR is ~0.10 ms/token.
              A full-duplex AR via copy engines (D2H || H2D, chunked) could approach ~180 us at 3.6 MB vs ~270-284.
  change:     none built; microbench agents/t3-w1/ce/ (2 procs, both GPUs copying at once, torch pinned buffers).
  results:    Observed: D2H alone 10-16 GB/s, H2D 12-15 GB/s per direction; D2H||H2D does not overlap (3.6 MB: 468 us vs 274+234);
              chunked eager pipeline worse (10-15 us host overhead per chunk); graphed n=4: 493 us at 3.6 MB. Est. AR ~510 us
              vs host AR 284 / NCCL 269; 0.5 MB: ~90 vs 43 us.
  conclusion: Interpretation: copy engines get about half the SM-driven zero-copy rate here and combined traffic is capped at
              ~13-15 GB/s; no duplex headroom via copy engines. The /dev/shm-registered buffer variant was not tested.
validity:     not proposed.
