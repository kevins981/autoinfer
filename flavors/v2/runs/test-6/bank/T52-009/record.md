team:         T52
builds_on:    T52-004/005 (chain microbench; 3-lane DMA copy agents/leader-52.w1/m2/dma3.cu, workspaces untouched)
performance:  "chain microbench: 3 waves (W3P) -9% layer at M=736 (1675 -> 1523 us), flat at M=384 (985 -> 989); residual vs WP floor -42% (job 482, 0.2 GPU-min)"
summary:
  hypothesis: WP's residual above its no-comm floor (227/365 us/layer at M=384/736) is schedule-exposed; more cover per AR (3 waves) or earlier copy
              start (producer wave releases, WPR) halves it.
  change:     m2/m2w3.py/.sh: 3 waves (64-aligned thirds) with FIFO link priority on 3 DMA lanes (dma3.cu = champion .cu with NL=3). WPR not built
              (> half a day: producer chunk hooks in the chain GEMM + strided 2D copies gated on device counters).
  results:    Observed (NL=8, median of 7, us/layer, M=384 / M=736): J0 1227/2045; WP 985/1675; W3P 989/1523; floors Wn 759/1310, W3n 887/1366.
              Residual vs own floor: WP 227/365, W3P 103/158; vs Wn: W3P 230/213. W3P graph == eager bitwise. Worker estimate before the job
              (W3P vs Wn 180-260 at 736, ~WP at 384) held.
  conclusion: Interpretation: more cover per AR does recover about half of the exposed time, but the third wave raises the compute floor (+128/+56 us),
              netting -9% only at large M. In server, a 3rd wave adds ~50% eager host issue, so it could pay only on steps >=~1200-1500 rows: est.
              <=+0.5-1% e2e (unverified). Decision rule (halve WP's residual vs Wn at M=736) not met (-42%). Direction closed by T52 on cost:
              the residual is schedule-dependent (not a proven ceiling), but the remaining schedule gains are small per GPU-minute.
validity:     microbench only.
