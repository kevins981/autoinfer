# T52 write-up (exploit-long, leader-52; direction set by the human: megakernels)
Records bank/T52-001..008. Worker leader-52.w1 (agents/leader-52.w1/NOTES.md, ASSETS.md). Proposal proposals/T52-1.md. Sparring T52-1.
GPU: 41.5 of 70 used (seed 20 + start grant 50); 28.5 unspent.
**Result: CHAMPION T52-005, 1212.4 tok/s official (+2.6% vs T50-002; same-job ABBA +2.82%, all B > all A).**

## Path
1. **Decode: consumer-side AR+GEMM fusion. Closed on measured limits.**
   - T52-001: whole-row fused AR+norm+quant+qkvz persistent kernel. Bitwise, but 103.5 us vs 88.5-90.4.
     - The champion's hint prefetch already brings qkvz to its warm floor (34 vs 32.7 us).
     - Streaming 42 MB during the AR stretches the AR by ~10 us.
   - T52-002: K-chunk cut-through (1/rms deferred, per the 2026-11-11 ruling). Parity: 89-91 vs 88-92 us.
     - A chunk is usable only ~20 us after it is published, and the first publish takes ~18-25 us.
   - T53 found the same tail limit from the producer side (reports/T53-writeup.md).
2. **Pivot to mixed steps** (35% of wall). The trace showed each layer's two ARs as ~260-280 us of copy-engine pieces with no compute running
   (agents/leader-52/mix2.py on agents/leader-38/prof_champ_node).
   - T52-003: chunk-ordered GEMM hooks. Producer side free and bitwise; consumer side +4-5% after fixes. Chunk release is wave-quantized.
3. **Cause found (T52-004): scheduling, not the link.**
   - The two token-split lanes interleave on the single D2H copy engine, so both ARs end together.
   - Link priority (chunk 1's ARs wait for chunk 0's) gives -63 us/layer.
   - Layer-pipelined waves (no per-layer join; T30's idea) with FIFO link priority give -12..-15% of layer time vs joined.
   - Without priority, waves lose. That reproduces T30/KB128 and explains why T30's waves failed.
4. **T52-005 (champion).** T50-002 + link priority + T30's waves (ported) at >=768 rows + T32's captured GDN prefill core (ported, so wave
   steps aren't host-bound).
   - Evidence: ABBA job 477; official #1 1212.4 (1210.1/1214.8), GSM8K 93.2%, p99 TTFT/TPOT 667/26.6 ms, 128K pass.
   - All Python. Compiled files are byte-identical to T50-002.
5. **Follow-ups: negative or too small.**
   - T52-006: ordering only the copies (no completion waits) is worse.
     - The per-AR peer lag is a symptom; the completion waits were spacing copies under compute.
     - The limit is cover per AR.
   - T52-007: wave-step host cost is 117 ms/step: ~1/3 GDN core glue, ~1/3 linear wrappers, ~1/5 AR issue.
   - T52-008: a lean wave loop (bitwise) cut it only to 85-100 ms, so MIN stays ~650-700.
     - The rest is the cores' Triton/Python launch paths.
     - Lowering MIN is worth ~+0.2-0.5% (estimate), below same-job noise, so I stopped.

## Forecast vs outcome
- Proposal: +9.1% if it works, central +3%.
- Outcome: +2.6% official.
- What the proposal assumed that didn't hold:
  - that most of the ~550 us/layer of exposed mixed-step AR was hideable. The chain shows a floor of ~compute + fill/drain, and cover per AR
    limits it;
  - that waves pay at >=384 rows. Host cost restricts them to >=768.

## For later teams
- What limits the mixed step now, measured in the chain (T52-004/006):
  - WP sits 186/353 us/layer above its no-comm floor at M=384/736;
  - steps of 256-767 rows run joined with link priority only;
  - steps under 256 rows (10% of wall) use SM hostar with no overlap at all.
- Host cost of eager wave steps lives in the GDN/attention core launch paths (~40-50 ms/step). Capturing the cores per wave size is the lever
  that could move MIN to ~384 (est. ~+1%). It is a memory question: 0.45 GiB free after T32's capture.
- Decode megakernel: both teams closed it with today's host-staged transport. Chunk latency (~20-40 us) is about the length of a decode GEMM,
  and launch boundaries already cost ~0 under graphs (T53-006).
- Assets: agents/leader-52.w1/ASSETS.md. It covers:
  - the 2-rank chain harnesses (m2/m2w with stand-in core, chain52/53);
  - the fused/cut-through decode kernels;
  - the wave and T32-cap ports;
  - host probes (T52_WTIME, cProfile);
  - the lean loop (t52-w2).
- Process notes:
  - Pass env-flag stacks to official evals as code defaults (the eval passes serve_args only).
  - Bound every device-side wait: one bench spin cost 3.3 GPU-min.
  - Cancel and restart an ABBA early when a better parameter shows up (job 476 at 1.2 min).

## Addendum (2026-11-12): sparring T52-2 + last chain screen (T52-009)
- Sparring T52-2 withdrew my "nothing left" ceiling. WP's residual is schedule-exposed: per layer at 384 rows the link needs ~375 us of copies
  against ~700 us of compute. Conditional mixed-step headroom is ~6% e2e, if the no-comm floors were reached.
- Screen (job 482, 0.2 GPU-min): 3 waves cut the residual vs their own floor by half (365 -> 158 us/layer at M=736). But the third wave raises
  the floor, so the net is -9% layer time at M=736 and 0 at M=384.
  - In server, the host cost (+50% eager issue) limits it to steps of >=~1200-1500 rows: est. <=+0.5-1% e2e (unverified).
  - The decision rule (halve the residual vs WP's floor) was not met (-42%). Close.
- Untested lever for the record: copies starting at the producer GEMM's wave releases (WPR). Ceiling ~110 us/layer at M=736; more than half a
  day to build.
- Remaining ideas, untested:
  - WPR;
  - a 3-wave split only for steps of >=1200 rows;
  - core capture per wave size, to lower the wave threshold;
  - any schedule that adds cover without raising the floor, e.g. splitting work other than token rows.
