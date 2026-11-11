# LOCAL_KB leader-44 (T44, research)

## Inspiration cards (first set)
- "Links that slow down when idle" + "How the host learns the GPU is done" made me look at the PCIe link itself: the run
  treats ~22-29 GB/s SM host traffic as a floor, but nobody examined TLP/ordering/snoop/link-state effects. NUMA placement
  of host slots (KB37) was a hidden host-side effect once already. Imbalance card: upstream link saturated by SM writes
  while downstream idles. Others (per-item, timing, unused structure) -> already mined by the run (lm_head, rows).

## Reading notes (2026-11-11)
- Champion T37-002 1142 tok/s. Decode step 26.2 ms (k=3, 64 rows). ~72% fixed. Rows cost ~0.10 ms, ~60% of it AR bytes.
- AR transport ~18-20% of trial (KB41). Mixed AR at "PCIe floor" (KB104); decode AR capped by prefetch coupling (KB78).
- T15-002: SM-W+SM-R 384KB 25 us vs 17.3/19.0 alone; CE+CE 19.0 (duplex); CE-W+SM-R 20.7 (near duplex);
  SM-W+CE-H2D 24.9 (as bad as SM+SM) => SM posted WRITES break duplex for any inbound reads. Mechanism never explained.
- My hypothesis H-TLP: SM sysmem writes leave the GPU as small TLPs (32/64B) that use the upstream link's raw capacity
  (64B TLP payload efficiency ~70% -> 22 GB/s goodput = saturated raw link), starving read requests (also upstream).
  CE writes use 256B TLPs (~90%) and leave room. Fix candidates: store pattern/width/cache-op, fewer write CTAs, pacing.
- Other teams' proposals: T42-1 certified greedy lm_head; T43-1 row-selective AR + decision depth. Avoid overlap.
- Directions considered and dropped (napkin): trees in TP (T38 owns, ridge), DP/compressed drafter (draft step is only
  0.55-0.62 ms; replication doubles weight bytes), FP8-at-quantizer AG (needs ruling, ~2%), GDN fp32 memory horizon
  (exactness only probabilistic), IMMA exponent windows (4 passes = no gain), prefill FP8 at ~85% of cuBLASLt peak at
  large M (no 2x headroom: leader-29.w1/mb/summary.md: cuBLASLt 264 TF at M=1024 on GPU0).
- Machine: Xeon Silver 4410Y x2 (SPR), both GPUs on node 0 different root ports, no IOMMU, BAR1 256 MB, ASPM policy
  default, link reads 2.5 GT/s idle. Power under load ~325 W of 450 (not power-capped).
- Spar T44-1: attacked trees-cross-ridge (answered with cost model), AR partition (min cut ~ norms force 2/layer), drafter
  capacity (T3 evidence; not pursued).

## Results (2026-11-11)
- T44-001 (jobs 397/398, w1): SM duplex collapse = writer oversubscription (pacing <=22-23 GB/s or 1:2 writer:reader CTAs
  -> CE-level duplex at >=1.5 MB; 12MB 678->568). Nothing at 384 KB. Memory type/WC/THP/LLC/link-power: no effect.
- Champion hostar uses 1:1 writer:reader (8/16/32 per role) -> oversubscribed region; token-split runs DMA lane + hostar
  lane concurrently -> SM writers starve DMA-lane reads (predicted).
- Proposal T44-1 written (rate-matched mixed-step AR). Forecast +1.2-2.5% e2e, finding stands regardless.
- CORRECTION (w1 code read): T37-002 token-split mixed steps (>=256 tok) use DMA on BOTH lanes (VLLM_T9_PINGPONG=1 default,
  model.py:149-158, 746-760) -> CE-only, already duplex. Hostar serves only decode + mixed <256 tokens (<=~2 MB). My
  proposal's main application was void; forecast cut to <=0.5%. Lesson: trace the actual default path (env defaults) before
  forecasting. Stage 1 narrowed: gate >=10% exposed hostar AR at R>=128, cap 5 GPU-min. Manager told.
- Stage 1 KILL (T44-002, job 401): hostar at R=48-255 is chain-latency-bound at ~10 GB/s; pacing/ratio no help (best -6.5%
  at 255 with worse p90). DMA two-lane at 700-1100 rows runs 16.4 GB/s vs 21.5 CE duplex standalone: ~25% gap, unexplained.
- Spar T44-2 (close check): the 631 us "floor" (KB104) is the CE-D2H+SM-read hybrid, not CE+CE. The CE+CE pipelined bound is
  ~540 us at 1100 rows and ~170 at 300 rows, vs 688/247 measured. ~400 split steps x 128 ARs x ~80 us ~ 4 s/trial (4-5%) if
  recoverable. "Latency is structural" is unestablished -> one timeline decomposition job before closing.
- T44-003 (job 403): DMA two-lane AR gap is code-imposed: doorbell write-value on the D2H stream blocks the next D2H (~60%),
  reset before H2D (~15%), ack before join (~5%). Required-path estimate ~-30% at 300 rows. Proposal T44-2 sent
  (doorbell stream + reorder; bitwise). Forecast +1.5-2.5%. Waiting on the manager before step 1.
- Task 4 (job 406): the reordered DMA AR is bitwise and passes the negative control, but is SLOWER (+8% at 300 rows, +12% at
  192, ~0 at 1100). Likely misread: the D2H end event fires at write issue, not landing; the 13-60 us "doorbell wait" was the
  data draining. T44-003's attribution is probably wrong. Job 409 tests (b)+(c) alone.
- T44-004 KILL (jobs 406/409): reorder 1.05-1.10x slower at small R; (b)+(c) null. T44-003 attribution superseded: D2H-end
  event = issue, not landing. Recommending close. Lesson: event timelines around posted writes mislead; intervene cheaply first.
