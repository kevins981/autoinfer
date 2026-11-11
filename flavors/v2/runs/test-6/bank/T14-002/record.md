team:         T14
builds_on:    T14-001
performance:  "unofficial, nsys (profiling slows steps): policy B vs champion A: Marlin critical path -0.59 s per 400-prompt trial (95% CI 0.43-0.75), GPU span of mixed steps -0.9..-1.3 s (CI 0.35-2.3), almost all at R<256"
summary:
  hypothesis: in the T9-005 champion, mixed-step Marlin runs far above the 0.183 ms/row roof; T6-004's policy recovers part of it in-situ.
  change:     none beyond T14-001 (change.diff identical). A = AI_POLICY=0 (champion), B = policy on. ONE job (116, ~11.5 GPU-min): fresh servers A1, B1, A2 (planned A B B A; the time guard dropped B2), T9-005 serve_args, profile_nsys.sh TRACE=node, 64 prompts c=16, rank 0 (GPU0). Workspace built; gdn_replay_op_available True, _C = workspace's (job116.log).
  results:    Observed (steps14.py -> steps_*.json; pair14.py -> pair.txt; gap.txt). Mixed steps paired across arms by prefill-token count (44-45 pairs). Savings = sum over pairs (A-B) x 6.25 (400/64). 'Marlin cp' = union of Marlin kernel intervals (both streams) in the step; 'summed' = sum of kernel durations; 'span' = first kernel start to last kernel end.
    Noise baseline A1 vs A2 (same config, 45 pairs): Marlin cp +0.01 s [-0.02,0.05]; span -0.10 s [-0.87,0.71]; period -1.08 s [-2.8,0.6].
    A1-B1 / A2-B1 (s per 400-prompt trial, 95% bootstrap CI over steps):
      Marlin cp, R<256:    0.44 [0.34,0.54] / 0.41 [0.30,0.51]
      Marlin cp, 256-600:  0.08 [0.02,0.16] / 0.08 [0.02,0.17]
      Marlin cp, >=600:    0.07 [0.02,0.13] / 0.08 [0.03,0.15]
      Marlin cp, all:      0.59 [0.43,0.75] / 0.57 [0.41,0.75]
      Marlin summed, all:  0.86 [0.62,1.11] / 0.82 [0.49,1.15]
      step GPU span, R<256: 1.32 [0.57,2.15] / 0.95 [0.46,1.55];  256-600: -0.06 / -0.08;  >=600: 0.02 / 0.02;  all 1.28 [0.41,2.27] / 0.89 [0.35,1.59]
      step period (incl. host), all: 1.06 [-1.0,3.3] / 1.85 [-0.5,4.2] (noisy; R<256 +0.5/+0.9, 256-600 +1.1/+1.8, >=600 -0.6/-0.8)
    Mean per mixed step A -> B (ms, A1 vs B1): Marlin cp R<256 35.6 -> 32.7; 256-600 91.6 -> 90.0; >=600 157.5 -> 156.5. Marlin launches per step 459 -> 291 (R<256), 714 -> 601, 627 -> 573.
    Marlin cp excess vs roof (max(0.183 R, 12.7)), champion arms A1/A2, per trial: R<256 1.84/1.75 s; 256-600 0.80/0.81 s; >=600 1.32/1.29 s; total ~3.9 s (not 6.9 s: T2-006's trace was older and counted summed time). B: 1.41 / 0.85 / 1.15 = 3.4 s. Per row (cp): R<256 0.275 (B 0.252); 256-600 0.217 (B 0.213); >=600 0.197 (B 0.196).
    Summed vs cp: at 256-600 summed/row 0.250 vs cp/row 0.217 (+15%), >=600 0.219 vs 0.197 (+11%), R<256 equal: the token split's two streams overlap Marlin kernels, and summed time inflates (each kernel is slower while sharing SMs with its twin).
    Decode steps (Marlin must be unchanged): median Marlin 13.725 / 13.704 / 13.726 ms (A1/B1/A2), 264 launches each, GPU span 25.47/25.48/25.51 ms: identical.
  conclusion: Interpretation: the policy gives a real but small in-situ gain: Marlin cp -0.6 s/trial (CI excludes 0), GPU span -0.9..-1.3 s in R<256 steps; step-period gain is within noise. It does not reach 1.5 s/trial. The gain is confined to R<256 (padded graph sizes 80-160 where launch pairs vanish); at R>=256 the policy is a no-op by design (T14-001) and Marlin cp is 0.197-0.217 ms/row. The 10-15% in-situ vs cold-microbench gap at large M (microbench 0.172-0.188) is largely (a) the summed-vs-cp inflation from the two concurrent chunk streams (kernels slower while overlapped), (b) non-64-multiple padded sizes/remainders (224, 288, 352...), (c) clocks/L2-warmth not separable from this trace. Remaining excess over the 0.183 roof is ~0.8 s (256-600) + ~1.3 s (>=600) per trial at cp, i.e. 0.014/0.02 ms/row above roof at >=600 - small. Caveats: 1 B arm only (B2 dropped), nsys-perturbed timings, pairing by prefill tokens (steps with same pt may differ in decode context), 64-prompt sample scaled x6.25.
validity:     as T14-001. Profiling-only; no throughput claim.
