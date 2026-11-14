team:         T57
builds_on:    T57-004
performance:  "unofficial: decode step -0.05 ms (probe, single run); not A/B'd; value is memory (+1.17 GiB free per GPU after capture)"
summary:
  hypothesis: the target's vocab-parallel embedding (lookup + AR) can be served exactly from host memory (T46/KB125), saving the embed AR and freeing the GPU embedding shard.
  change:     T57_HOSTEMB (default on): full [248320,5120] bf16 embedding table per rank in registered host pages, built with the stock lookup itself; compile-safe custom op does a zero-copy row gather; the drafter's shared embedding swapped too. Also tested (NOT in this diff): screen-kernel variants for _chunk_topk/_merge_exact (16-way threshold search, warps/BK sweeps).
  results:    Observed (job 501, 200 prompts c=16, step probe; agents/leader-57.w1/task5/REPORT.md): in-server bitwise self-check passed both ranks (192 ids); greedy c=1 5x200 tokens identical to T57-004; GPU allocated 18.053 -> 16.784 GB/rank; free after graph + T32 capture 1.62 vs 0.45 GiB; host +2.54 GB pinned/rank; load +4.6 s/start; decode step median 24.47 (on) vs 24.52 (off) vs 24.68 (T57-004 flags off) ms; acceptance 2.672 vs 2.669.
              Screen variants: bitwise but slower (chunk_topk 35-39 vs 12.3 us; merge_exact 20.7-26 vs 17.5 us); BK=512/4 warps already best; _scores at HBM floor (58 vs 52.6 us).
  conclusion: Interpretation: HOSTEMB is exact and worth ~0.2% in time; its value is ~1.17 GiB/GPU of headroom for memory-blocked items (e.g. more captured graphs). Screen kernels need a different top-64 algorithm (single-pass select fused into _scores, est. <=50 us/step), not tuning.
validity:     bitwise for the target (same rows the stock lookup returns); no precision change; GPU memory falls; host pinned memory +2.54 GB/rank (on top of FCTAB's 1.27). Not proposed for an official eval alone.
