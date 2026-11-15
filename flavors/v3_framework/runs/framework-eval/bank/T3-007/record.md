team:         T3
builds_on:    T3-005 / T2-009 (microbenches only)
performance:  microbench negative: DeltaLog 2-step checkpoint ceiling ~0.33 ms/step; evict_first/.cs state stores: no gain
summary:
  hypothesis: (a) DeltaLog with a checkpoint every 2 steps halves the remaining GDN state writes (~2% napkin).
              (b) writing the row-0 state with evict_first / streaming stores reduces the L2 drain cost before gate_up.
  change:     worker leader-3.w1, microbench dl_bench.py (in-situ order: GDN kernel -> out_proj -> 48 us AR-window spin
              with 40 MB gate_up prefetch -> gate_up; 16 req x 3 rows, 24 value heads, states rotated so not L2-resident).
              (a) ws t3-dl: "mode 2" = T2's log kernel with the row-0 write skipped (best case of a non-checkpoint step).
              (b) ws t3-ef: row-0 store with createpolicy evict_first + st.global.L2::cache_hint (mode 3) or st.global.cs (4).
  results:    Observed: (a) job 109, prefetch on: sequence 240.7 us (T2 log) -> 226.8 (row-0 write skipped); kernel 59.8
              -> 33.8. Ceiling for L=2 = 0.5 x 13.9 us x 48 = 0.33 ms/step before the extra replay (4.35 -> ~5.5 rows/step).
              (b) job 114, two runs: mode 1 240.6/240.2, mode 3 240.2/237.2, mode 4 240.4/237.5 us; bitwise equal state/out.
  conclusion: Interpretation: (a) net ~0.1-0.25 ms/step (+0.3-0.8%) for a risky rewrite of T2's materialize hooks: not
              built. (b) the state is re-read a full step later (from HBM either way); the drain cost is the number of
              dirty bytes, which priority hints don't change. Both dropped. Design note: dl2.md.
validity:     microbenches only; no served change.
