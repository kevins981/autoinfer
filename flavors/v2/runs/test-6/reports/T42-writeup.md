# T42 write-up (research, leader-42): certified greedy decisions for the target lm_head
Records: bank/T42-001 (CPU study), T42-002 (standalone kernel + proof), T42-003 (in-server, exact), T42-004 (nsys, in situ).
Proposal: proposals/T42-1.md. Sparring: T42-1 (5 turns). GPU used ~21.9 of 40.

## Idea
Greedy verification uses only the argmax of the target logits. Screen with a cheap plane plus a RIGOROUS error
enclosure; compute exact logits only for ids that can still be the argmax; fall back to the full exact lm_head if the
certificate fails (manager ruling, task_definition.md).

## Supported by measurement
- Low rank cannot certify: the final-norm hidden states are near-isotropic (r=512 captures 67% of the energy; block
  Cauchy-Schwarz keeps all 248,320 ids for every row) (T42-001).
- Precision can: a 7-bit plane (per-group bf16 scales rounded up, exact int8 MMA screen, whole-K Cauchy-Schwarz bound
  with directed rounding, a tensor-core error assumption 2^8 above what was measured; proof bank/T42-002/code/PROOF.md)
  plus a lower bound shared across ranks leaves ~2 candidates per row on rank 0 and ~0 on rank 1. Decisions:
  0 mismatches on 188K real + 1.7K adversarial rows (T42-002); c=1 output identical, 0 fallbacks over 26.5K rows,
  128K pass, gsm8k-50 92% (T42-003).
- HBM: 0.566 GB plane replaces the 0.993 GB encoded shard (-0.34 GiB/GPU net); exact rows come from pinned host memory on
  NUMA node 0.
- Speed: standalone cold 759 vs 1178 us at M=64; in situ the lm_head section goes 1065 -> 796 us and the decode period
  25.14 -> 24.94 ms (T42-004). ABBA +0.53%, mixed (T42-003, job 410).
- Why it shrank in situ: the champion's lm_head gains 153 us from its AR-window prefetch; the plane's hint-mode prefetch
  lands ~40%. Launch overhead is not a factor (CPU runs 18-51 ms ahead).

## Not established / open
- Real evict_last prefetch of the plane (as in KB97) instead of hints: ceiling ~0.1 ms.
- An HBM cache of exact rows (gather 76 us/step).
- A 6-bit plane with the global lower bound (CPU: mean 24 candidates per row with LOCAL bounds; global untested).
- The operational value of the freed 0.34 GiB/GPU. KV is pinned at 5 GiB by flag, so the saving removes no current
  constraint I know of; it could matter for capacity-bound designs (k=4 GDN spec blocks, more graphs). Unverified.
- Killed on napkin (no GPU): coding all-reduce payloads with side information (the next RMSNorm reads the fp32
  unrounded sum); a PCIe link keep-alive (no ASPM attributes; the link drops to gen1 only at idle).

## Lesson
Decision-only consumers have a precision floor far below their data's: about 7 bits per weight fix the argmax of a
5120-dim bf16 product, while rank truncation fixes nothing. Bounded quantization error beats truncation because it keeps
every coordinate. The certificate is cheap; the remaining cost is bytes and prefetch placement, which makes the lm_head
now a ~1% lever (KB121).
