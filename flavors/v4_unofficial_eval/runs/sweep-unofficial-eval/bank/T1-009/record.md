team:         T1
builds_on:    T2-012
performance:  screen: within noise (+0.28% mean, 1 of 2 pairs above; job 63); not unofficial-evaluated
summary:
  hypothesis: T1-007's fused host-AR + add + gemma RMSNorm lost (-1.6%) for lack of parallelism (fixed 4-block grid); with one constant 16-block grid for all host-AR calls (AR time is flat in block count, t1-w1 microbench) the fusion should save a launch + an activation pass per AR (~1%).
  change:     host AR fixed grid 16 (kHostArGrid/_GRID) for plain and fused calls; fused kernel host_ar_add_rmsnorm as in T1-007 (row-aligned, 512 threads); env VLLM_HOST_AR_NORM=0 disables. On T2-012 code.
  results:    Observed (job 63, ABBA fresh starts, 100 prompts, HEAD args): A 716.2 / 721.9, B 721.4 / 720.6 tok/s (pair 1 +0.73%, pair 2 -0.17%, means +0.28%); B median TPOT 19.6-19.7 vs A 19.9-20.1 ms; acceptance equal. Offline: residual bitwise, normed max 1 bf16 ulp (~4.8e-6 of elements).
  conclusion: Interpretation: grid 16 removed T1-007's loss; the remaining gain (if any) is below this test's resolution (~0.3%). Not submitted alone; could ride along with another change. Note: fp-reordering only (norm reduction order).
validity:     exact up to fp reordering (1-ulp norm differences); same parity safety (one constant grid).
