team:         T1
builds_on:    T1-026 (standalone prototype; no engine change)
performance:  "microbench gate: NO-GO (warm gate_up 129 / in_proj 60 us vs Marlin 84 / 43; gate needed <= 72 / 36.5)"
summary:
  hypothesis: Marlin at M=48 is tensor-pipe bound when L2-warm with 1 warp/scheduler (KB28); a decode kernel with 2 warps/scheduler
              could be >= 15% faster warm at equal cold time.
  change:     standalone JIT kernel agents/leader-2.w16/k/w8dec2.cu: swap-AB mma.m16n8k16 bf16/fp32, reads Marlin's repacked qweight
              and bias-folded scales as-is, bit-identical dequant (shift + HFMA2.BF16), 8 warps/CTA, 5 tile configs (512/256/128
              cols, 4-6 stages, 2 CTA/SM), persistent stripes with fixed-order cross-CTA fp32 reduction. Worker leader-2.w16
              (NOTES.md, k/r1..r5; ~2 GPU-min).
  results:    Observed: correct (max diff vs Marlin 1 bf16 ulp, 0-0.1% elements; fp32-ref error same as Marlin; deterministic).
              M=48 cold/warm us: Marlin gate_up 108/84, in_proj 51/43; mine full 156/129, 66/60; no epilogue 112/99, 56/50; loads only
              (no MMA) 103/89, 48/42. Every config within a few %. M=16 warm: Marlin 72.8/31.7 vs my loads-only 86/41. A pure stream
              of prefetched in_proj reaches only 1.56 TB/s.
  conclusion: Interpretation: Marlin's warm data path is 20-30% better than a straightforward load pipeline; without explaining that
              (ncu: L2 hit rate, sector use), 2 warps/scheduler has no warm headroom. Epilogue partials (~25 MB) also costly. NO-GO.
validity:     prototype only.
