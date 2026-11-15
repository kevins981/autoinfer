# T2 sweep, step 5: my map vs the other maps and the history

History available at this point (2026-11-15, ~15:35): only BASE (official 635.9; unofficial
634.5 / 643.4, bank/BASE/unofficial/1). No other records, no shared_kb entries, no sparring files.
Other map: T1 (reports/T1-sweep/1-4, phase 1, same day).

## Status of my candidates
Nothing has been built or tested by anyone yet (bank holds only BASE). In flight:
- W1 (prefill CUDA graphs to 2048 tokens): T2 unofficial eval queued (job 4, label t2-cg2048, --long).
- W4 / I-AR (all-reduce transport): microbenchmarks by **both** teams (T2 worker job 2 + 5, T1 worker job 3): duplicated; to split from now on.

## Where the maps agree
Same dominant decode terms with nearly identical measurements (Marlin ~18.6 ms at ~67% BW, AR ~8 ms,
lm_heads + gathers ~4.3 ms, GDN ~3 ms); same dead groups (DP: capacity; PP for decode: weight re-read;
DBO in decode: Shared-Read rule; bf16 GDN state: contract; continuous-tensor speculation: Gate 2);
same candidates R-ARGMAX = my R1, R-DLM8 = my R4, reduced-vocab = my R5, MTP k=3 = my S1, Row B GDN
= my K1, I-MARLIN = my R8, I-AR = my W4.

## Disagreement 1: what the mixed-step time is (biggest one)
- T1: mixed steps are slow mainly because NCCL is slow on 2.66 MB messages (5.7 GB/s); fixes are a
  faster AR (I-AR, ~+13%) and DBO in mixed steps (W-DBO); CUDA graphs/compile for >96 tokens ranked
  low (~1%).
- T2: regressing mixed-step period on rows (3-calibration C) gives **88 ms fixed + 0.244 ms/row**;
  the NCCL part is **40 ms fixed + 0.060 ms/row**. The per-row NCCL rate (10 KB x 135 per row in
  0.060 ms = 23 GB/s) is at link bandwidth; the fixed 40 ms is not bandwidth: a 57-row prompt step
  still takes 111-120 ms (vs 39.9 decode). GPU idle in mixed steps is 20 ms. Reading: eager launches
  of ~1,500 kernels per rank skew the ranks and every all-reduce waits for the slower rank. So the
  fix is to graph-capture mixed steps (W1, ~12%), and I-AR only helps mixed steps by the bandwidth
  part. Evidence that would settle it: the t2-cg2048 eval (if mixed steps shrink to ~40 + 0.24/row
  the skew reading holds) and T1's AR microbenchmark at 2.6 MB (if NCCL alone takes ~460 us there,
  T1's reading holds).
- Both readings can be partly right: the decision variable is the measured result of job 4.

## Disagreement 2: FP8 MMA rate with fp32 accumulation
T1 (following SOL) takes 165 TFLOP/s for FP8 with fp32 accumulate (= bf16), so W8A8 only helps via
fp16 accumulation inside 128-blocks (with an overflow risk they note). I took 330 (Ada whitepaper,
dense). This decides R6's value (6% vs ~0 at fp32 accumulate). Cheapest test: a torch._scaled_mm /
triton FP8 GEMM microbenchmark at M=256..2048 on one 4090 (seconds of GPU).

## What T1 has that my sweep missed
- **W-DBO in mixed steps** (split r into halves, AR(half A) || GEMM(half B)): I only priced DBO for
  decode and dropped it; the framework's own condition says it wins when the step is compute-bound
  (N > ridge), which mixed steps are. After W1, mixed-step AR is ~15 ms (bandwidth part), so W-DBO
  would be worth ~10 ms x 400 = ~4 s (~3%), not T1's 14%. Lesson about my use of the framework: I
  priced the Window knob per step type only for the dominant step type.
- PP=2 as a conditional candidate for mixed steps: correct under T1's NCCL reading; under mine
  (after W1) PP loses in mixed steps too (mixed AR ~15 ms vs PP's doubled decode weight reads).

## What my sweep has that T1's lacks
- The **Host resource** (CPU launch and cross-rank skew) as a dimension: it turns T1's "I-COMPILE,
  ~1%, low priority" into my top candidate W1 (~12%).
- **S5** (async host prep) is in both as a low item; I price it at 2.2%.
- K1 detail: the BASE GDN MTP kernel writes S for all 3 positions (verified in
  csrc/gdn/fused_gdn_decode_kernel.cu), and the measured GDN time is *below* the napkin of that
  design: an open mismatch neither map explains.
- R5 with a data-free vocabulary choice (the tokenizer's id order ~ BPE merge order) that avoids the
  "generic token-frequency data" problem T1 lists as the blocker.
- R8/R9 smaller items (merge in_proj_qkvz + in_proj_ba).

## Framework notes (both maps hit them)
- No Host/launch resource (my 3-calibration C).
- Link_BW depends on message size and transport (T1 calibration point 1).
- The vocab index is not in the standard template (07 §4.3), yet lm_heads + gathers are 11% of decode.
