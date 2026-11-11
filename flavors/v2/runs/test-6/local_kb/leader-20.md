# leader-20 (T20 explore) notes
- Proposal T20-1 (proposals/T20-1.md) approved; 80 start released when G1 passes. G1 = LL tagged hostar microbench, gate >= 6 us/AR in chain both ranks; manager conditions: 1e5+ AR stress with skew, negative control, A_pf w/ prefetch, NUMA verified.
- Ws t20 = T14-004, built. Worker leader-20.w1 (8 GPU-min allowance). Task: agents/leader-20/w1_task.md.
- CPU check: no hidden decode idle in T14 nsys (agents/leader-20/idle.py); big gaps = mixed-step CE AR waits.
- Key numbers: champion hostar 38.5 us standalone, 22 us/AR net in chain (T16-001); decode grid 8, K=1 (serial write->fence->flag->read).
- Rebalance (G3): MLP uneven split + lm_head vocab shifted to GPU0; memory moves, no duplication -> needs 128K test.
- Manager: before any served candidate, stress at in-server scale (~3e5 ARs/trial); record AR count AND words-checked count per skew arm in the record.
- G1 FAILED (T20-001: chain +2-3 us/AR; prefetch window sets in-chain AR duration). Closed voluntarily; write-up reports/T20-writeup.md.
