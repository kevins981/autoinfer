team:         T3
builds_on:    T2-008 (uses T3-002's microbench finding)
performance:  unofficial: -1.56% (ABBA job 86, all B < all A: B 837.1/827.6 vs A 847.4/843.7 tok/s)
summary:
  hypothesis: T3-002 showed a copy-engine AR overlaps with GEMMs (+3-14 us over GEMM alone vs +27-115 for NCCL,
              +87-333 for hostmem). Using it for the row-split side stream should cut mixed-step exposed AR time.
  change:     ce_ar.cu + ce_all_reduce.py (graph-safe: device epoch counter; per call: 1-warp kernel waits for the
              peer's ack of the previous call, D2H memcpy into my mapped-shm slot, 1-warp kernel writes my flag and
              waits for the peer's, H2D memcpy of the peer slot, add kernel (rank0+rank1 order) that writes my ack).
              cuda_communicator creates it (VLLM_CE_AR_MAX_BYTES, default 20 MB); rowsplit.async_all_reduce uses it
              for every side-stream AR. ROWSPLIT_MIN_TOKENS unchanged (256).
  results:    Observed: 2-rank test (job 86 prelude): 0/30 eager and 0/120 CUDA-graph-replay mismatches vs the hostmem
              AR (bitwise). ABBA (A ws t3-c8 = T2-008, B ws t3-ce, 200 prompts): -1.56%, all B < all A; acceptance same.
  conclusion: Interpretation: the microbench overlap gain doesn't carry over. Unexplained so far. Candidate causes:
              (a) for 128-262-row halves CE replaces hostmem and is ~15 us slower alone; (b) mixed-step exposure is
              mostly rank skew/waiting, which CE can't hide; (c) copies captured in CUDA graphs or queued behind other
              copy-engine traffic (input H2D, sampled-token D2H) behave differently from eager. An nsys profile of B
              would tell; not run yet.
validity:     exact (bitwise equal to the hostmem AR, tested in eager and graph replay); +42 MB pinned host memory, no GPU
              memory change except a 16-byte counter. Not proposed for official eval.
