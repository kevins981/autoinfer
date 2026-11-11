# leader-46 (T46, research) working notes

## Inspiration cards (first set)
Representation, Distance, Overqualified work, Unused results, Timing; HW: host wait modes, clocks/power, no P2P,
fixed transfer cost, texture/RT units; SW: GIL, math libs, kernel languages, NVRTC, OS limits.
Thoughts: power is not binding (official gpu_clocks: ~330 W of 450, 2655 MHz both GPUs). "Distance" made me re-derive
that the hidden-dim AR is the minimal TP cut (any other partition moves more bytes or needs replicated weights);
memory (not bytes) is what blocks every replication idea. "Unused results" -> rows after a rejection (T43 owns).

## Picture of the champion (T37-002, from KB111/KB118/T38/T45)
Trial ~80 s. Pure-decode rate would be 42.7 tok / 26.2 ms = 1630 tok/s; official 1142 = 70%.
Prefill events: 400 x (15 ms fixed + ~0.166 ms/token x 260) ~= 23 s = ~29% of wall. PP prefill is ~0.51x TP (T38).

## Sparring T46-1 (5 turns)
- PP closed for real (capacity + autoregressive dependency; one-microbatch PP + prefill-in-bubble ~78 s vs 80 s: wash).
- L2-reuse half-batch split at decode: arithmetic kill (~140 vs 135 us/boundary), because a 32-row AR is ~37 us
  of mostly fixed latency and the only cover is the other half's warm GEMM (~15 us). Not measured.
- Assumed, untested: 0.78 AR codec is a floor (shared-context coding); but bf16 sums have no excess mantissa
  precision, so only exponent bits are compressible (ceiling ~-16% bytes).
## Other notes
- T30-002 fits: no-AR mixed step fixed term 23.1 vs decode 21.8 ms; with AR 39.5: the ~15 ms "fixed per prefill
  event" is almost all all-reduce, not compute.
- Prefill GEMMs at M~324 run ~40% of FP8 peak; wave quantization (1.5 waves) + M padding explain much. T38 found
  split-K 128x64 20-25% faster on PP shapes at M=128. Exploit lead, not mine.
## Proposal T46-1: host as a memory tier (proposals/T46-1.md)
## Stage 1 results (T46-001, T46-002; jobs 420/422/425/426; 0.5 GPU-min)
- Host embedding gather: exact, 30.8 us @64 rows; saves ~13-15 us per embed AR vs real hostar; frees 1.27 GB/GPU.
- Link as weight bandwidth: additive (not zero-sum) with 8-CTA SM zero-copy, f<=2.5%, chain -1.0..-1.2% = ideal;
  capped by 23 GB/s. Real t31 on host-mapped B loses (6-12 GB/s, latency-bound). CE ring: loses; CE writes go to DRAM.
- Decision: no stage 2 (ceiling ~1% e2e vs a new loader kernel).
## Sparring T46-2 (on stopping)
- Weakest stated reason was "covered by others". Open points: PP single-microbatch 36.5 ms is a sum of floors, not timed;
  "only 1.3-1.6 GB/GPU freeable" -> answer: only rarely-read data can leave HBM; per-pass weights from host cost
  ~43 ms/GB/pass exposed (23 GB/s), so a 6.7 GB/GPU hybrid layout is out by >100x. Closing.
