team:         T51
builds_on:    T51-002 (chain harness) + new bs_gemm configs 132/133 (2,2,32,32,{2,3},64) in workspace leader-51.pig
performance:  "microbench screen negative (job 438, 1.7 GPU-min): 4-CTA/SM gate_up config and down split-K do not lower the M=128-192 GEMM band (65-99 us/row vs 69-90 reference)"
summary:
  hypothesis: TP-shape GEMM band is wave quantization (544/816 CTAs on 384 slots, 71% util); a 4-CTA/SM config (512 slots) and split-K down should cut gate_up at M=160/192 from ~163 to ~150 us.
  change:     configs 132/133 (126/124 regs, no spills, 4 CTAs/SM) for gate_up; down cfg 127 with split-K 2/3. Same chain as T51-002.
  results:    Observed: gate_up M=128 117.4 (ref) / 134.7 (132) / 118.7 (133); M=160 159.1/158.8/157.8; M=192 162.8/175.4/167.1 us.
              down M=128 72.9 / 83.1 (sp3) / 64.9 (sp2); M=160-192 worse with split-K (+9..+23 us). Band per added row: ref 70/90/72, b133 65/95/87 us at M=128/160/192.
  conclusion: Interpretation: times did not follow occupancy either way; the wave-quantization explanation for TP vs full-width is not confirmed (not refuted). The limit is something else (B re-read per m-tile, per-CTA fixed cost, issue limits), unidentified. Config-level tuning of bs_gemm is exhausted; the band needs a different kernel design (e.g. stream-K / larger-BM with balanced grid), unproven. Only crumb: down split-K 2 at M=128 -8 us/layer. Standalone split-K wins do not transfer to the chain.
validity:     microbench only.
