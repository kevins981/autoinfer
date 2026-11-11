# Round 9 report from leader-2 (T1): Marlin warm-L2 decode prototype -> NO-GO; mixed-step leftovers -> ~0
No new best. Champion stays T1-026 (880.9). GPU 25.3/80 used (54.7 unused).

Experiments:
- T1-029 (w16, prototype, ~2 GPU-min): standalone W8A16 decode kernel (swap-AB mma.m16n8k16, Marlin's repacked qweight/scales as-is,
  bit-identical dequant, 8 warps/CTA = 2/scheduler, 5 tile configs, fixed-order cross-CTA reduction). Correct (1 ulp vs Marlin,
  deterministic). M=48 cold/warm: Marlin gate_up 108/84, in_proj 51/43; prototype full 156/129, 66/60; loads-only (no MMA) 103/89,
  48/42 -> even without math its warm load path is slower than Marlin's whole kernel. Gate needed warm <= 72 / 36.5. NO-GO.
- T1-028 (w17): cheaper no-op GDN flush launches (one block per (req,head) looping layers; mode-3 clear once per KV group). Unit
  tests 0 mismatches with failing negative controls. Same-job +0.2% (noise); official 880.0 (879.9/880.1) = T1-026 within noise.
  The mixed-step replay kernel "46 vs 25 us" was not like-for-like (40.7 us decode @16 reqs vs 46.7 mixed @15): at the HBM roof, no
  lever (~0.1%).

Proposed SHARED_KB:
- KB-a: Marlin's warm (partially L2-resident) data path is 20-30% better than a straightforward cp.async/ldmatrix load pipeline at
  M=48: a 2-warps/scheduler prototype's loads alone (no MMA) are slower warm than Marlin's full kernel (gate_up 89 vs 84 us). A Marlin
  rewrite has no warm headroom until that advantage is explained (ncu: L2 hit rate, sector use). (T1-029)
- KB-b: No-op GDN flush launches cost ~35 us/decode step + ~100 us/mixed step; removing them is exact but unmeasurable (official 880.0
  vs 880.9). The mixed-step replay kernel is at the HBM roof. (T1-028)

Left open: why Marlin's warm loads beat a plain pipeline (ncu study); T1-028 adoptable as harmless (no measurable gain).
Next (one direction): bitwise-lossless compression of the prefill-size host-AR payload (R7-D in ideas.md; SE r7 idea 3). Measured
compressibility 0.768 (block-exponent scheme, 0.3% raw blocks) on real AR inputs; prefill ARs still cost ~0.96 us/token/AR on the
pacing rank after the NUMA fix (T1-026) and are only partly hidden by the R7-A split (steps >= 256 tokens, chunk 1 exposed). -23% bytes
on the exposed prefill AR transport ~ 1.5-2 s/trial (~+1.5%), est.; the codec must hide under PCIe (integer ops per value). Gate first
with a 2-process microbench at M=150/300 (bitwise vs current hostar, time), ~1-2 GPU-min. Decode ARs ~0 (their window is used for
prefetch).
Notes: workers w16, w17 stopped. No GPU contention. Workspaces kept: leader-2.mixleft (T1-028), leader-2.marlinw (unchanged).
