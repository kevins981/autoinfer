# leader-33 (T33, external) notes
## Orientation (2026-11-10)
- Champion T31-003 1088.9; bar 1143.3. Seed 20 GPU-min, 1 worker max. Manager offline: write proposal, end turn.
- Time: decode ~60% wall (step ~24 ms: GEMM ~12, host AR+L2 prefetch ~6.9, GDN 1.7, lm_head ~1, small kernels 1.5); mixed ~40% (AR fully exposed in big steps ~1/3; GEMM at cuBLASLt-minus-promotion).
- Measured floors: decode GEMMs (KB79/98, gate_up L2 wall), AR at PCIe floor + 0.781 codec (KB43), comm/GEMM overlap zero-sum (KB59/62/70), k=3 capped ~+2.4% (KB95), MTP head near learnable (KB60), host hidden (KB89).
- T32 (prev external) concluded no card clears bar; best package FP8-AG out_proj + gate_up weight coding + TETRIS budget, +4.5-7% if all work, central +2.5% (proposals/T32-2.md).
- Waiting on scout deck scout/T33-1/.
## Deck T33-1 assessment (2026-11-10)
- Card 13 measured (T33-001, job 295, 0.2 GPU-min): FP8 exponent entropy 2.6/4 bits; practical code 0.883, entropy bound 0.821. Chain f=0.88 -17.8/-19.8 us/layer vs champion-like (271.8/279.4) => ceiling ~+3% e2e (0.82: ~+4.3%). Hard limit.
- Other cards: costed from KB (table in proposals/T33-1.md). No card clears bar. Optional 13+08-adapted package ~+5-6% if both work, central +1.5%, not recommended.
- Proposal written: proposals/T33-1.md (no grant requested). Seed left 19.8.
## Reopened (manager msg): spar T33-1
- Spar point: 12 ms "HBM floor" + AR are coupled; combined GEMM+AR ~19 ms for 12.15 GB (~640 GB/s avg) => ~6 ms not bandwidth-saturated.
- gate_up ablation (T31): weights-only 56-63, MMA-only 53, full 85 => ~sum not max. Same regs (220 vs 212, cuobjdump), 1 CTA/SM. Not a hardware wall, a kernel-pipeline property (hypothesis).
- New direction candidate: overlapped stream/MMA decode GEMM (gate_up, down) + card 13 code. Napkin +5-7% e2e if works.
- W1-1 (leader-33.w1): synthetic partition microbench, <=8 GPU-min. Kill: MMA-only partition > ~55 us or concurrent ~sum.
- T32 funded on mixed-step waves (T32-5): don't touch mixed steps.
- W1-1 DONE (T33-002, jobs 303-305, <1 GPU-min): synthetic B16 4 stream+12 MMA: both ~= max (cold 99 vs sum 135; pf48 58.2 vs stream 57.8). MMA-only 30-36 us.
- Spar turn 6-8: charge residency to the real mechanism. Chain data (T31 W23): hint-48 saves Marlin gu 23.6 us (111.5->87.9), no AR stretch; LD-32 loses (pair 130.3 vs 125.5). => effective ~25 MB resident. Re-based per layer: overlap-only -20 us (-1.3 ms/step, +3.6% e2e); +code -37 us (-2.4 ms, ~+7%). Need ~1.75 ms/step for +5% at 65% decode share (assumed).
- W1-3 acceptance (spar): code (0.889-0.897 incl offsets) must add >= ~7 us/layer net, i.e. gate_up at ~25 MB-resident regime >= ~7 us faster with decode than raw handoff (ideal ~9.7 us => >=70% of ideal).
- W1-2 (T33-003): real handoff: cold 100.6 (hidden), pf48 70.7 GPU0 best (stream 62); consumer side 44-55. Marlin flat ~88 for 30-58 MB resident => hint residency can't be inferred from Marlin. W1-3: levers + in-chain plugin (stage-major layout so hint prefetch covers each CTA's leading K).
- W1-3 (T33-004, job 325): in chain, ov4 gu 81.6 vs Marlin 88.0 (pf48 hint): overlap alone -6 us/layer. All four at 0.875x (free decode): -30/-27 us/layer vs champion-like 272.2/274.2 => ~-1.8 ms/step ~+5% e2e (65% decode). At real code ~0.895 scale bytes part: ~-26 us/layer -> ~+4.5%. Bar needs better code (~0.86) or hidden decode + down overlap. W1-4: real decode, real weights.
- W1-4 (T33-005, jobs 338-340): real 0.8925 code bit-exact; decode+MMA floor 85.4 vs MMA 62.8 => loses at pf24/48 in chain (gu 90.0 vs raw 82.3). Free-decode ceiling at 0.8925 ~+4.4-4.8%: below bar alone.
- W1-5: escape-free main (7-window + code7=zero mask) + exact correction (validity Q for manager: products unsplit, grouping inside 128-K block); + overlapped down kernel. Kill: decode+MMA floor > ~70 us GPU0.
- W1-5A (T33-006): exact escape correction floor ~118-121 us => killed. W1-5B (T33-007): overlapped down -1.5..-2.5 us (noise); overlap-only stack ~-8 us/layer (~+1.5%). Recommending close: proposals/T33-2.md.
- Spar T33-1 ended: open/assumed: decode share ~65% (unmeasured), hint residency amount (inferred), ~3.2 GB during AR (estimate). Rejected: stream/MMA can't overlap; tested exact codecs pay in prefetched regime. Not established: universal ~11% data limit.
- Worker stopped. proposals/T33-2.md = close recommendation. 18.2 GPU-min unused.
