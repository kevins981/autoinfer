# T2 sweep, step 5: my map vs T1's map and the history

Read at 2026-11-16 ~00:07 UTC: reports/T1-sweep/1-6, T1-sweep.md, shared_kb.md (empty),
announcements.md, bank/ (BASE only), reviews/ (none). **Disclosure:** leader-1 sent me a split
proposal containing T1's top candidates (AR transport, draft lm_head fp8/reduced vocab, k=3; T2 to
take W8A8 for mixed GEMMs) and then retracted it; both reached my context right after my write of
4-knobs.md returned (mtime 00:05:00). Files 1-4 were not changed afterwards.

## Status of my candidates in the history
Every candidate is **untried** (bank holds only BASE; no reviews; shared_kb empty). T1 has a
worker running an all-reduce microbenchmark (GPU job 1); my worker runs AR (job 2) and FP8 GEMM
(job 3) microbenchmarks; my harness of W1 is queued.

## Agreements (both maps found them independently)
| candidate | T1 | T2 | note |
|---|---|---|---|
| faster all-reduce transport (NCCL proto/channels; T1 adds FlashInfer PCIe-IPC AR) | rank 1, up to +14% | P4, <=9% | T1 prices the mixed-step AR at 65 ms/step (all of it transport); I attribute part of it to waiting on the other rank's eager launches (see below), so my number is lower |
| draft lm_head FP8 / reduced vocab | rank 2 | R2b / R3 | same |
| W8A8 or better GEMM for mixed steps | rank 3, +8.7% | R4, 0-7% | same uncertainty: FP8 fp32-acc peak 165 vs 330 decides; T1 assumes 165 (gain = Marlin inefficiency only) |
| k=3 after cheaper drafts | rank 5 | S1 | same napkin (~+3% now, ~+7-9% later) |
| argmax before gather | rank 6, +1.7% | R1, +2.8% | I count the target gather (0.95 ms) at M=48; T1 used M~42 means |
| AR/compute overlap on mixed steps only (DBO) | rank 4 | P6 | same, and the same reason to exclude decode (shared-read rule) |
| PP=2 loses at decode, DP infeasible, bf16 GDN state illegal, tree spec dropped, DeltaLog low | yes | yes | same verdicts |

## What my sweep found that T1's map lacks
1. **Mixed steps are CPU/launch-bound when they exceed the largest captured CUDA graph (96 tokens)
   (W1).** T1's calibration attributes the 147 ms mixed step to Marlin (92 ms) + AR (65 ms). The
   per-GPU split contradicts this for small mixed steps: at 102-115 tokens, GPU 1 is *idle* 54-65 ms
   of a ~125 ms step while GPU 0's NCCL kernels sit at 58-71 ms (waiting for GPU 1 to arrive);
   GPU 1's own NCCL time is 15 ms. The GDN chunked-prefill path is launched kernel by kernel from
   Python (gaps of 20-170 us between 1-10 us kernels). So a big part of "prefill AR time" is
   rank skew from eager launching, not link time. The missing dimension is **CPU launch time as a
   resource** (the framework's window cost has F, BW, Link_BW but no host/launch term).
   Candidate: CUDA graphs for mixed steps (`max_prefill_cudagraph_capture_tokens` up to 2048; vLLM
   uses 8192 by default on SM100). Napkin ~13 s (9%). Flag-only; harness queued.
   Consequence for T1's #1: the AR transport gain on mixed steps will be smaller than +14% if the
   step is not link-bound; conversely, W1 may expose the real AR cost and make #1 matter more.
2. **The target lm_head's kernel** (ampere 128x64 at 400 GB/s, 1.59 ms) and the draft lm_head's
   8-CTA grid as kernel-choice problems separate from precision (R2', R2a): exact, no ruling needed.
3. Calibration correction both maps needed: the bf16 peak is 165 (not 82.6) TFLOP/s; T1 has it right
   from the start, I corrected it in step 3.

## What T1's map has that mine lacks
1. **FlashInfer PCIe-IPC all-reduce** (an existing opt-in path in this vLLM) as a transport option:
   I only had NCCL tuning and a hand-written host-mapped all-reduce (P5). Finding about my use of
   the framework: I swept "Place" over memory tiers for the reduction buffers but didn't search the
   engine for existing implementations of a point I had priced.
2. **Split GEMM_out along d to overlap the AR** (exact, works within a decode step without
   re-reading weights since the split is along the output dim of one GEMM, not across row
   micro-batches). My sweep only split rows (r, i), where decode loses by the shared-read rule; I
   missed that splitting d of the output has no shared-read penalty. A genuine miss of a split site.
3. A replicated reduced-vocab draft head (Place + Rewrite combined: no gather at all).

## Disagreements
- T1's mixed-step bucket (312 x 147 ms, AR 65 ms) vs mine (370 mixed steps, E ~35 ms eager overhead):
  the GPU-1 idle data above is my evidence. The W1 harness run will settle it: if W1 gives little,
  T1's reading (link-bound) is right.
- T1 drops async scheduling as "check if on": the server log shows GPU idle 1.3 ms per decode step
  with 120-250 us host gaps before each draft graph, so it is not fully hidden; low value either way.
