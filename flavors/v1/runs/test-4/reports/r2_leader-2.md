# Round 2 report, leader-2 (T2) — received 2026-11-07
Re-scoped: depth 3 is dead on the champion. Best: T2-006 = T1-003 diff + T2-004 diff (exact greedy vocab-parallel target fast path),
T1-003 spec-config WITHOUT --enable-batch-sharded-sampling. OFFICIAL 699.5 (697.7/701.3), +1.2% vs T1-003, +10.0% vs BASE; GSM8K 92.8,
AL 2.335, TTFT p99 1343, TPOT p99 48.8, 128K pass. Same-job +2.1/+1.9% (1 server per arm, order not controlled). nsys: window between
target graphs 6.40 -> 5.90 ms (matches the -0.5 ms estimate).
T2-005: depth 3 on champion: step +12.6% for +16.7% tokens; end-to-end flat (prefill steps 35-38% of wall); TTFT p99 2185 fail. Closed.
  Node-level nsys inflates cudaGraphLaunch to ~1.2 ms (artifact); graph-level ~120 us.
T2-007: analysis only: real inter-graph idle ~1.6 ms/step (4.5%), 1.24 ms CPU Python around FlashInfer metadata builds. Only a
  one-graph capture of spec-decode glue would remove most (est 3-4%, invasive). Not pursued.
Left open: one-graph capture of the spec-decode glue; depth-3 TTFT p99 cause; prefill steps now the bigger share.
GPU 26.9/55. Manager asked for an ABAB same-job A/B vs T1-003 before deciding.
  ABAB (job 26): +0.2% vs T1-003, within noise -> not champion (manager).
