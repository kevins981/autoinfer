# T6 map update 2 (closing, leader-6, ~07:05 UTC)

## Tested since update-1
| candidate | result | evidence |
|---|---|---|
| T6-001 HMAR768 harness | cancelled during start 1 when T5-001 was promoted (a T3-007-based record can't be promoted over T5-001; the HMAR768 screen remains incomplete) | job 117 |
| T6-004 row-split overlap stacked on T5-001 (DeltaLog) | starts 895.6 / 905.1 / 894.7, mean 898.5 (+0.34%); third start below 895.4: no win | jobs 119, 121 |
| split numeric check | split vs unsplit RowParallelLinear, TP=2, W8A8 Triton: bitwise identical | job 116 |

## Verdict
Row-split GEMM->AR overlap: exact for the tested serialized configuration (not under concurrent ubatching). It gave +0.28% on T3-007 and +0.34% on T5-001, the same sign on two codes but within
noise each time. No consistent end-to-end gain was demonstrated from overlap or a faster transport; rank skew as the cause remains a hypothesis (no controlled split on/off timing on the fixed code).

## Ranking left for others
1. Decode GEMM efficiency (W8A8 at ~76% of DRAM peak, ~3.9 ms/step): untried, hard.
2. k=3 drafts: memory-blocked (DeltaLog keeps the slot count); an allocator change that frees the 2 spec slots per request would unblock it.
3. Row-split overlap + HMAR768 for the < 192-row mixed steps: untested combination on correct code.
4. A model of rank skew in the framework's cost model (what makes a collective's kernel time removable).
