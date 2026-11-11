# leader-36 (T36, moonshot) working notes

## Measurements of the champion T31-003 (my own)
Job 356 (nsys, 64 prompts c=16; node + graph traces in agents/leader-36/prof/):
- Observed: GPU 0 busy 97-98% of wall time in the graph trace (15.23 s of 15.67 s). step_anatomy's
  "idle in window" for prefill steps is an artifact (async GPU work spills into the next step window).
  Interpretation: the engine is GPU-bound; no host-side slack worth chasing.
- Observed: decode-only step period vs rows M (graph trace medians): M<=12: 18.6 ms; 15-24: 19.5-20.3;
  30-36: 21.5-22.5; 42-48: 23.3-23.5. ~ +1.6 ms per 16-row tile; ~80% of a decode step is M-independent.
- Observed: mixed (prefill) step GPU busy = 30.8 + 0.160*P ms (fit over 46 steps), vs 23.1 for a full decode
  step. Time split of the profile: decode-only steps 72.5%, prefill steps 27.5% (prefill extra = 20%:
  17.7% token-proportional, 2.4% fixed).
- Observed (node trace, one decode step ~24.3 ms under nsys): 64-layer body 21.6 ms; target lm_head
  (lmh_gemm_kernel) 1.02-1.08 ms; verification ~0.1; MTP draft step 1: 0.62 ms, step 2: 0.55 ms; next-step
  prep 0.35 ms. hostar_kernel (AR) avg 42.9 us x 128/step = 5.5 ms. Marlin gate_up 87 us; t31b e4m3 kernels
  ~34 us each; gdn_decode_replay 26.8 us.
- Napkin: weights per GPU 12.2 GB (191 MB/GDN layer: 89.1 gate_up, 44.6 down, 41.9 qkvz, 15.7 out).
  At 1.0 TB/s -> 12.2 ms/step DRAM floor; GEMM kernel time sum ~12.1 ms/step.
- Official champion: acceptance 0.778 / 0.559 (cond. 0.72), 2.33 tok/step/seq.

## Directions considered (napkin, before proposal)
- PP instead of TP: reads weights twice per seq-step -> worse (31 ms vs 23.4). Dead by arithmetic.
- Lossless weight compression in the body: entropy ~6.3 bit/weight, decode cost > saving. Dead (assumed).
- Row-split leapfrog at decode (hide AR under other half's GEMM): ~+5% at best after L2 re-read and doubled
  small kernels (assumed numbers); champion already does it for >=256-token steps.
- Deeper / confidence-adaptive drafting: value depends on p3, p4 of the MTP head -> K sweep job (below).

## Inspiration cards (first set)
"Per-item overhead" + "Idleness" made me check what is fixed per decode step (80% is M-independent) and
whether the GPU/DRAM is idle (GPU busy 98%; DRAM ~52% utilized vs the 12.2 ms floor). "Fixed choices" ->
per-sequence draft depth instead of K=2 for all. Hardware cards: nothing used yet.

## K sweep and proposal T36-1 (record T36-001)
- Observed: K=3 acceptance .770/.542/.388 (jobs 361, 363); K=3 at c=16 only runs 14 requests (state memory), more KV
  OOMs (job 362); at c=14 K=3 is +2.1% vs K=2 (one pair, job 363), step +10.1%, tokens +15.9%.
- Estimate was +5% overall; measured +2.1%. My cost model overstated the overall gain ~2x (decode share, 2.3 ms real
  step growth).
- After the proposal the manager pointed to T1-008/015, T3-005..008, T5-004/006, T27-001/003, T28-003/005/007/008.
  They had measured the same things: budgeted verification in-server raised accepted tokens only +5.0-5.6% at M=48
  (T5-006, T27-001), two-slot state + K=3 gave +2.9% (T28-003) then null on a later champion (T28-005).
  Interpretation: T36-1's riskiest assumption was already answered; upside < 5%. Withdrawn, milestone 1 not run.
- Sparring T36-1: weight-pump estimate (16-17 ms) withdrawn as unjustified; usable DRAM-idle windows outside ARs are
  ~40 us/layer. Supported: 80% fixed pass cost, draft-step cost. Assumed, never measured: lossless weight compression
  too expensive (single GPU with compressed weights loses on bytes anyway: 21 GB through one GPU's DRAM > 12.2 ms).
- Lesson: `gpu wait` only on my own job ids; never name a shell variable CC in job scripts.

## Inspiration cards (second set, after withdrawing T36-1)
"Predictability" -> draft the next step during the current target pass (assume full acceptance): valid for only ~40%
of sequence-steps and the batched drafter's cost is fixed per pass, so nothing is saved. "Rework" -> rejected rows are
~22% of rows, ~1 ms/step, not detectable before the lm_head. Hardware cards (concurrent kernels, four one-way lanes):
only relevant to the AR/prefetch area I left alone. Nothing that argues to 5%.

## All-reduce anatomy (record T36-002; sparring T36-2)
- Observed (job 364, champion's own trace): decode AR = write own packed payload 17.4 us + flag/peer wait 2-3 us +
  read peer and add 16.5 us + ~6-8 us tail (prefetch blocks end). Peer wait is the same on both ranks: the clock
  asymmetry is not what the AR waits for.
- Observed (job 365, one pair): 4-chunk cut-through: write 24.6 us, reader end 41.0 (K=1: 17.4 / 36.9); 1029.6 vs
  1042.2 tok/s. Interpretation: ~22 GB/s total per GPU for kernel access to host-mapped memory, directions not
  independent; the AR is at the wire time of its bytes for this transport.
- Supported: transfer breakdown; ~22 GB/s for the kernel path. Assumed: the payload is minimal for a hidden-dimension
  split (shape argument); that no >=5% structural direction exists (not established, only none found).
- Open cheap test nobody on my team ran: copy-engine (DMA) transfer running beside the kernel path at decode size.
- Closing: T36-1 withdrawn, no T36-2. GPU used 23.0 of 30.
