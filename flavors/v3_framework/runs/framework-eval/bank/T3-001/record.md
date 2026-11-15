team:         T3
builds_on:    T2-008
performance:  "screen negative (1 pair, -6.57%): B 796.6 vs A 852.7 tok/s (200 prompts), T3 GPU job 79"
summary:
  hypothesis: In mixed steps the row-split halves (> 1.31 MB, i.e. > 128 rows) go to NCCL, which in the server ran ~650 us
              per 2.7 MB AR (T1-014 nsys, job 72) vs 242 us isolated (job 74, ar_bench.py). The hostmem kernel does
              218-254 us isolated at that size; routing every AR size through it might cut the exposed AR in big mixed
              steps (49-130 ms per 480-890-token step).
  change:     no code: env VLLM_HOSTMEM_AR_MAX_BYTES=25165824 (24 MB) on the B arm (champion default is 1,310,720).
              change.diff = T2-008's (same workspace for both arms).
  results:    Observed: ABBA (abba.sh, 200 prompts) stopped after 1 pair: A 852.7 tok/s (acc 2.348, TTFT/TPOT p99 934/29.2 ms),
              B 796.6 (acc 2.345, p99 1225/39.2 ms). Log: agents/leader-3/jobs/abba_hm/.
  conclusion: Interpretation: a large loss (~10 sigma of a pair), not noise. The spinning hostmem AR on the row-split side
              stream takes SMs from the overlapped GEMMs (T1's update-2 observation), and for large serial ARs it is no
              faster than NCCL. So the in-server slowness of NCCL at these sizes is not NCCL's protocol; likely contention
              with the concurrent GEMMs and rank skew (T2's finding). The cap stays at 1.31 MB.
validity:     n/a (env var only, exact; not pursued).
