# T22 write-up (leader-22, explore). Recommending a voluntary close after the seed

GPU: 14.6 of 20 used (E1 jobs 148-151: 1.0; E2 smoke 152: 3.5; E2 ABBA 153: 10.1). ~5.4 returned.
Records: bank/T22-001 (warm floors, measurement), bank/T22-002 (down_proj prefetch, null/-0.9%). Worker leader-22.w1 (stopped). Notes: local_kb/leader-22.md.

## Direction examined: "HBM-continuous decode"
Observed (CPU only, agents/leader-22/onestep.py over T14's in-situ nsys B1, rank 0, 283 decode steps with the modal 1034 kernels; kernel sum = span, no gaps):
per GDN layer, us: in_proj 45.2, splitK 3.6, conv 4.0, gdn_replay 37.5, out_proj 21.3, AR2 (prefetches gate_up) 52, norm 3.6, gate_up 84.3, act 3.1, down 56.7, AR1 (prefetches in_proj+out_proj) 39, norm 3.6; ~360 us total.
HBM bytes per layer (191.6 MB weights + ~25 MB GDN state) need only ~230 us at 0.96 TB/s, so HBM is idle ~35% of the decode step.
Hypothesis: kernels that run cold could be warmed by prefetch placed in those idle slots.

## E1: warm floors (T22-001, jobs 148-151)
Marlin at M=48, read flush, us:
| GEMM | cold | warm | in situ |
|---|---|---|---|
| qkvz | 55.3 | 41.6 | 45.2 |
| gdn_out | 29.3 | 21.0 | 21.3 |
| attn_qkv | 52.3 | 38.8 | 38.9 |
| attn_o | 29.7 | 21.0 | 21.3 |
| down | 61.6 | 45.0 | 56.7 |
- gate_up's microbench is unrepresentative (in-situ 84.3 is below it).
- The first 16 MB of prefetch recovers 50-85% of each cold-to-warm gap: it's a latency effect of the first tiles, not bandwidth.
- GDN replay: fully warm 23.4 vs 37.5 in situ, but state prefetch did not reproduce the warm time (unreliable).

Interpretation:
- The champion's AR-window prefetch already brings every decode GEMM except down (+11.7 us) and qkvz (+3.6 us) to its L2-warm floor.
- The HBM idle time is real, but there are almost no cold bytes left on the critical path to spend it on.
- Ceiling of the direction: ~1.6 ms per decode step (down 0.75 + qkvz 0.17 + GDN <= 0.67 if its warm floor were reachable), ~+3.5% of wall, below the 5% bar. No proposal.

## E2: down_proj prefix prefetch in situ (T22-002, job 153)
Change: the AR before gate_up also prefetches the first 16 MB of down_proj (evict_last, its own byte count, issued first).
Result (ABBA, 300 prompts, 2+2 fresh starts, tok/s):
- A (champion): 929.8, 931.4.
- B (with the prefetch): 918.2, 926.4.
- -0.9%; every B below every A. Acceptance unchanged.

Interpretation:
- The down prefix competes with gate_up's prefetch for L2 and AR-kernel time (the KB78 coupling).
- Down's headroom isn't reachable through the AR window.
- Reaching it from inside gate_up would cost issue slots in a warm Marlin that has none spare (KB39/KB49). I did not test that.

## KB candidates
1. In situ, the champion's decode GEMMs sit at their L2-warm floors, except down_proj (56.7 vs 45.0 us) and qkvz (45.2 vs 41.6).
   - The cold-to-warm gap is mostly the first ~16 MB of each weight (latency, not bandwidth).
   - HBM idles ~35% of the decode step, but there are no cold bytes left to fill it with; the ceiling of warming everything is ~+3.5%. (T22-001)
2. Prefetching down_proj's first 16 MB from the AR before gate_up is -0.9% in an ABBA test (every B below every A): it competes with gate_up's prefetch. (T22-002)

## Opinion
The decode step is bounded by Marlin's tensor-and-HBM coupling plus the serialized PCIe AR. The L2-prefetch layer is already near what it can buy. I agree with T18/T19 that the remaining items are ~1-2% exploit work. I don't have a >= 5% direction to propose with honest numbers.
