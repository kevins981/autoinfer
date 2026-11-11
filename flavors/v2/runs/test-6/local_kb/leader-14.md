# leader-14 (T14, explore) working notes
## Direction: T14-1 mixed steps at compute roof (package): proposals/T14-1.md
- CPU analysis of T2-006 trace (agents/leader-14/marlin_steps.py): summed Marlin excess vs max(0.183 ms/row, 12.7 ms) = 1.8 s (R<256) + 1.6 (256-600) + 3.5 (>=600) per trial; padded 112/160 -> 2 launches/GEMM; odd-32 large steps 716 vs 456 launches (~+20 ms).
- Decode step vs #reqs (T2-006): 1 req 20.3 ms, 8 req 21.5, 16 req 25.5 -> ~80% of decode step is per-step fixed.
- Long "decode" periods after big prefills are attribution artifacts (span ~25 ms); GPU busy 97% of the 64-prompt capture.
## Worker
- leader-14.w1 in ws leader-14.m1 (from T9-005): W1-1 = port T6-004 policy + microbench + in-situ nsys A/B (G1 gate >=1.5 s/trial).
## G1 result (T14-001/002, jobs 114/116): FAILED gate
- Policy on T9-005: Marlin cp -0.59 s/trial [0.43,0.75]; step span -1.28 s [0.41,2.27]; decode unchanged; exact (bitwise M<=48, <=1 ulp).
- Re-derived cp-based Marlin excess on T9-005: 3.9 s/trial (not 6.9; summed time over-counts token-split overlap). Large M is ~at roof (0.197 ms/row cp); stock M>=192 at 64-multiples is 0.172-0.188 cold.
- Package payoff now ~+3.5% < bar.
- T14-004 promoted to champion (956.6). Recommended close; write-up reports/T14-writeup.md.
