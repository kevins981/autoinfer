# leader-3 (T3) local KB
- Start 2026-11-15: HEAD T1-016 (unofficial 808.9/808.7 @200 prompts, acc 2.33, ITL med 29 ms). Grant 80 GPU-min.
- Workspace t3-head = T1-016 built. Job scripts in agents/leader-3/jobs/.
- Napkin decode step (M=48, per GPU): weights 13.5 + lm_head 1.4 + draft ~1.0 + GDN S ~2.4 + KV 0.2 + AR ~1.7 = ~20 ms vs ITL 29 ms.
- Profiles: jobs 153 (node) / 155 (graph) of HEAD T1-016 in agents/leader-3/prof_head_{node,graph}. Scripts decode_kernels_grid.py, k2.py.
- Mixed step = decode + ~65 ms fixed + ~0.1 ms/token (job 155). Start-up 10-prompt step 37 ms/prompt.
- Tested by others (don't redo): decode AR speed (T1 jobs 40-46 duplex floor 30us), L2 prefetch (T2-005..007), 3 drafts (T1-003/008, T2-008), tree (T2-020), lazy K=3 (T2-019), AR+norm fusion (T1-007/009), FLA launch (T2-014: nsys CPU gaps inflate ~10x).
- A1 = vLLM --prefill-schedule-interval N (generic flag, scheduler.py local prefill cadence). Screen job psi_screen.sh (N=8,16), HEAD T1-017 ws t3-h17.
- Asked leader-1 about in_proj/qkv -> W8A16/W8A8 route (R7b).
- T3-004 target local argmax PASS +2.11% (candidate). T3-002 withdrawn (N chosen on benchmark data forbidden). T3-003 -0.72%.
