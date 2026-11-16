team:         T6
builds_on:    T3-007
performance:  "screen only (harness job 117 cancelled during start 1 when T5-001 was promoted; eval/1 holds only the partial server log) (no harness): quick bench 400 prompts c=16, 1 start: 843.4 tok/s (job 105, OFF arm of the split A/B, workspace t6-arbig.b with VLLM_T6_SPLIT=0 = this change). Cross-job vs the champion's 853.6: -1.2%, within cross-job noise. Not a measured loss."
summary:
  hypothesis: in mixed steps (M > 96 rows) the TP all-reduce goes through NCCL (RING_LL/Simple via host SHM). The champion's host-mapped
              all-reduce (HMAR) is capped at 96 rows / 1 MB. Raising the cap could cut the per-call cost at 128-768 rows.
  change:     hmar.cu REGION_BYTES 64 KB -> 512 KB (32 MiB pinned host buffer); host_mapped_all_reduce.py MAX_ROWS 96 -> 768 (by leader-6.w1).
  results:    Observed: microbench job 101 (isolated, 2 ranks, us/call, 5120-wide bf16 rows):
                rows   96  128  192  256  384  512  1024  2048
                NCCL  138  418  452  545  665  751  1123  1617
                HMAR  126  133  205  269  396  530  1016  1904
              HMAR output bitwise-equal to NCCL at every size. In the server: 843.4 tok/s, 1 start (job 105).
  conclusion: Interpretation: the isolated 1.7x per-call gain at 192-512 rows did not show up end to end in one start. T1's lesson applies:
              mixed-step AR time is dominated by peer skew, which a faster transport doesn't remove. This contradicts T1's microbench claim
              (bare NCCL ~205 us at 2.66 MB): here NCCL takes 545 us at 256 rows. Open: a harness measurement was never made.
validity:     exact (bitwise-equal sums, communication at bf16 as before). Pinned memory +31 MiB host only.
