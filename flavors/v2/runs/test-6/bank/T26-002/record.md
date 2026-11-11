team:         T26
builds_on:    T26-001 (standalone microbench, no engine change)
performance:  "microbench: full swap-AB register-direct kernel v1 SLOWER than Marlin: gate_up chain 106.8 vs 94.4 us (P=32), qkvz 64 vs 43.8"
summary:
  hypothesis: K0's 93%-of-peak loop + K1's 0.95 TB/s stream combine into a decode GEMM >=10% below Marlin in the chain (AR-window) arm.
  change:     agents/leader-26.w1/W1-2/swab.cu: weights LDG.128 -> A frag, Marlin dequant+scale (bit-identical), tokens cp.async+ldmatrix, 128 CTAs, split-K 2 for down/gdn_out/attn_qkv, deterministic.
  results:    Observed (jobs 166/167, GPU0, read flush, mine/Marlin us): gate_up chain P=0/16/32/48 106.8/106.6/106.8/106.8 vs 95.3/93.4/94.4/94.1; cold 108.6/104.9; warm 106.5/94.8. qkvz warm/chain 64/43.8. down chain 54/48.5. gdn_out 23.1/21.3. attn_qkv 41/40.4.
              Numerics: max-abs 2.4-4.9e-4, 0.03-0.10% elements differ, deterministic. Ablations gate_up: loop-only 75 us (K0 63), loads-only 96 cold / 88.7 warm; qkvz warm loads 0.78 TB/s (K1 1.29).
              Regs: gate_up 20 warps x 96 regs (84 B spill); others 8-16 warps, 128-145 regs.
  conclusion: Interpretation: v1 lost on implementation: spills, a shallow register prefetch that exposed L2 latency (so the L2 prefetch gave nothing), tile imbalance, and poor load/MMA overlap. One bounded tuned retry (W1-3, T26-003) decides the direction.
validity:     microbench only; numerics fp32 reorder only.
