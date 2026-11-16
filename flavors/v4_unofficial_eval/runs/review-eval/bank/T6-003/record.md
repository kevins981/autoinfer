team:         T6
builds_on:    T3-007
performance:  "+0.28% vs the champion (T3-007, 853.6): 856.0 tok/s, starts 859.8 / 852.2 (eval/1, GPU job 113). The second start is below 853.6, so it does not beat the champion under the promotion rule; within noise."
summary:
  hypothesis: same as T6-002. In mixed steps with M >= 192 rows, each GEMM->AR site runs in 2 row halves, and half 1's NCCL AR on a side stream
              overlaps half 2's GEMM. Napkin +0.5-2% (review T6-1: G1 + max(AR1, G2) + AR2 vs G + AR; ~60% of mixed time qualifies).
  change:     T6-002's code plus the fix for its double all-reduce: the target's final norm (model.py ~666), the MTP final norm (mtp.py ~168) and
              the aux hidden-state capture (model.py ~614) skip their all-reduce when the input is the split's already-reduced output.
              change.diff = ws diff t6-split (the measured workspace). Env: VLLM_T6_SPLIT (default 1), VLLM_T6_SPLIT_MIN_ROWS (default 192).
  results:    Observed: 856.0 (859.8, 852.2); acceptance 2.335 / 2.331 (champion 2.340 / 2.335, restored from T6-002's 2.294); TTFT p99 902 / 979 ms;
              TPOT p99 33.6 / 36.0 ms; gate_latency pass; 128K pass; GSM8K-250 93.2% (pass). "[t6_split_ar] first split call" appears in both starts.
              Mechanism (worker quick A/B, jobs 105/107, on T6-002's code, which had the bug but the same split timing): median mixed-step elapsed
              -8% at 192-383 rows and -10% at 384-767 rows; flat at >= 768 rows. Those bins are ~20 s of ~107 s.
  conclusion: Interpretation: the step-bin gains (-8/-10%) come from the buggy T6-002 code with HMAR768, in separate jobs, with the ON log including a GSM8K run; they are not established for this code. In any case the whole-run effect (+0.3%, one start of two below
              the champion's mean) is within noise and below the ~+2% the step bins imply. Possible reasons, untested: the AR2 tail and the extra
              launch and weight re-read cost; the quick-bench bins came from different code and jobs. Not a champion candidate. It is exact; stacking with other mixed-step work is untested but it may be worth re-measuring on a future champion together with other mixed-step work.
numeric check: job 116 (leader-6.w1, agents/leader-6.w1/numchk.py): the real RowParallelLinear on TP=2, W8A8 Triton, K=8704/N=5120, bf16 and
              QuantizedActivation inputs, M in {192, 260, 1000}: split vs unsplit reduced output is BITWISE identical (max diff 0, 6/6 cases), and so are the
              rank-local partials. Linear layer only (the final-norm fix is checked by review T6-2, not by this test).
validity:     Review reviews/T6-2 (code): intermediate paths reduce exactly once; dyn128 scales (M, K/128) are row-major and sliced with the codes;
              bias only on rank 0 (and these projections have none); graph capture is safe for serialized forwards (fork after GEMM1, join after
              AR2); not safe under concurrent ubatching (not used here). The three final/aux paths it flagged are fixed in this record.
              Math: row chunks of the same GEMM plus an elementwise bf16 sum of 2 partials (the same as NCCL's); the only difference is the
              Triton tile config chosen per M bucket (K accumulation order), i.e. fp reordering. GSM8K 93.2% and acceptance corroborate it. Review T6-2 turn 2: fix complete for serialized forwards; record wording softened per its points.
