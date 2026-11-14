team:         T58
builds_on:    T58-001 (microbench only; no server change)
performance:  "microbench: hybrid (t31b for out/down at 97-144 rows) -17..-22 us per GDN layer vs bs_gemm; napkin ~+0.2-0.3% e2e; not built"
summary:
  hypothesis: (sweep candidate M2) small mixed steps (97-255 rows) run every GEMM on bs_gemm; the decode t31b kernel tiled over M in ONE launch
              (blockIdx.z) might beat it at M=112-192.
  change:     none in the engine. Scratch copy of t31_gemm_e4m3.cu with the launcher's M<=96 check removed (agents/leader-58.w1/m2/).
  results:    Observed (job 512, GPU0, cold = read flush + persisting-L2 reset; warm = same GEMM just before), bs_gemm / t31x us at M=112-128:
              qkvz 51-53/78-79 cold, 45-48/57-58 warm (t31 loses); out 29.7/26.6 cold, 28.6/23.7-24.5 warm (t31 wins); gate_up 121-123/280-283
              (t31 2.3x slower); down 75.8/61.4-62.5 cold, 73.7/56-58 warm (t31 wins); attn_qkv 49/71 (loses). M=160/192: t31 loses every shape
              (MT capped at 3 -> z>=4). Best t31 configs un4 ks2 kseg1 mt3. Numerics vs bs_gemm: max-abs 6e-5..2.4e-4 (~1 bf16 ulp, K-split order).
              Per GDN layer at M=128: 279.6 -> 262.1 us cold (-17.4), 283.7 -> 264.3 warm (-19.4); M=112 -16.4/-22.4; M>=160 0.
              Not the same test as T56-002 (separate <=96-row launches).
  conclusion: Interpretation: ~0.8-1.2 ms per small mixed step (<=144 rows, ~150 steps/trial) = ~+0.2-0.3% e2e before in-situ effects, below
              one ABBA's noise; widening t31's M range also needs its own FP8 GSM8K A/B (task_definition Addendum 2). Not pursued. A real
              mid-M gain needs a new kernel (T56: bs_gemm at M=112-128 is 291 us/GDN layer cold vs ~200 floor).
validity:     not applicable (no served change).
