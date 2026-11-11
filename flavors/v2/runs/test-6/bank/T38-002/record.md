team:         T38
builds_on:    BASE (offline measurement; no engine change). Data and offline MTP from team T3 (agents/leader-3.w1, read-only, pointed to by the manager)
performance:  n/a (offline acceptance measurement; 0 GPU-min: ran on CPU)
summary:
  hypothesis: With wide draft TREES from the stock MTP head, expected tokens per verify step could reach >= 3.9
              (break-even for a 2-stage pipeline-parallel engine, where verify rows carry no all-reduce) and >= 4.1 (>5%).
  change:     none. leader-38.w1 computed exact tree acceptance offline: T3's data are the target's own greedy outputs
              with dumped target hidden states, and T3's vLLM-faithful offline MTP reproduces in-server acceptance.
              A tree node on the true path sees exactly the teacher-forced inputs, so acceptance is exact (up to bf16
              near-ties: 131/54,709 true-path nodes, 0.24%, differ from the teacher-forced argmax).
              Code and outputs: agents/leader-38.w1/tree/ (tree_eval.py, analyze.py, robust.py, full_cpu_analysis.txt,
              robust.txt, NOTES.md, full_cpu.npz).
  results:    Observed (held-out 40 seqs, 13,578 response rows; draft vocab 98,304 ids; 4.0% of true tokens are outside it;
              tokens/step = accepted drafts + 1; 95% CI = bootstrap over sequences):
              calibration: a1 .7810, cum a2 .5607 (T3 .781/.561 exact).
              N   chain k=N        best static (shape, cross-fit equal)   dynamic best-first (N sequential MTP steps)
              3   2.731            2.731 (chain)                           2.775 [2.66,2.89]
              7   3.486            3.486 (chain)                           3.728 [3.48,3.99]
              11  (k=8 3.580)      3.827 (2,2,1x9)                         4.211 [3.86,4.59]
              15  (k=12 3.817)     4.054 (2,3,2,1x8)                       4.525 [4.09,4.99]
              19  (k=16 3.947)     4.218 (2,3,3,2,1x9)                     4.743 [4.25,5.28]
              31  -                4.554                                   5.154 [4.55,5.83]
              63  -                4.944                                   not run
              dynamic - chain k=3 at N=19: +2.01 [+1.62,+2.45]; dynamic - best static at N=19: +0.53 [+0.41,+0.65].
              dynamic N=19: median 4 tokens/step; mean tree max depth 9.5 (deep, narrow); 26.6% of accepted drafts come
              from rows accepting >= 10; capped at 8 accepted drafts, N=11/15/19 = 4.03/4.18/4.29; without the two
              repetitive sequences (ids 7, 396), N=19 = 4.51 (chain k=3 2.70).
  conclusion: Interpretation: draft acceptance is NOT the limit for a tree-based design: dynamic trees reach 4.2 tokens/step
              at 11 nodes and 4.7 at 19 (vs 2.73 today). Whether this pays depends on the cost of verify rows and of deep
              drafting, which is the subject of proposal T38-3 (pipeline parallel + dynamic trees). In the TP engine, rows
              cost ~0.1 ms each in all-reduce/GEMM, so trees don't pay there (napkin: N=7 in TP ~ break-even).
              Caveat: dynamic numbers assume one MTP step per added node; level-wise drafting (realistic) is being measured.
validity:     n/a (no engine change)
