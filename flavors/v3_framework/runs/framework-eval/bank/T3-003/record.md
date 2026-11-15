team:         T3
builds_on:    T2-008
performance:  official #1: 897.4 tok/s (+1.97% vs T2-008 880.1; GSM8K 93.6%, all gates pass). Unofficial ABBA: +2.41% (job 85)
summary:
  hypothesis: During each decode-size hostmem all-reduce (~48 us, PCIe-bound) HBM is idle. Extra CTAs in the AR
              kernel can prefetch the next GEMM's weights into L2 (evict_last); this also makes the L2 write back
              the GDN state's dirty lines during the AR instead of during gate_up. Napkin 1.5-3 ms/decode step.
  change:     (worker leader-3.w1, ws t3-pf) hostmem_ar.cu: ar_kernel gets nblocks + 64 extra blocks that issue
              `prefetch.global.L2::evict_last` over [pf_ptr, pf_ptr+pf_bytes); the AR uses nblocks, not gridDim.x.
              hostmem_all_reduce.py: module-level hint (set_prefetch_hint), consumed by the next hostmem AR with
              <= 128 rows. model.py (Qwen3_5DecoderLayer.forward): before attention/GDN the hint = mlp.gate_up_proj
              .weight (first 40 MB); before the MLP the hint = the next layer's in_proj_qkvz / qkv_proj weight;
              cleared after the MLP. Env: VLLM_AR_PREFETCH_MB (40, 0 = off), _BLOCKS (64), _MODE (0), _MAX_ROWS (128).
  results:    Observed: microbench (worker job 82, M=48, [25 MB state write -> out_proj -> AR -> gate_up]): AR+gate_up
              201.5 us off -> 168.2 us with 40 MB/64 blocks (AR alone not slower: 49.5 vs 52.3). 
              ABBA same workspace, 200 prompts, T2-008 serve args: A VLLM_AR_PREFETCH_MB=0, B default 40:
              job 85 B 870.5 / 860.2 vs A 848.3 / 841.6 tok/s (+2.41%, all B > all A), acceptance 2.335-2.356.
              (Job 83, with the hint placed in forward_fused, which is not on this config's path: -0.73%, hint never
              consumed: a null test.)
  conclusion: Interpretation: about 0.7 ms/decode step saved (~33 us per AR+GEMM pair in the microbench, x ~2 per
              layer, part of it lost in situ). Below the 1.5-3 ms napkin: probably part of the prefetch is evicted
              by the norm/quant kernels or doesn't finish in the AR window; MB/blocks/target choice is untuned.
validity:     (leader-3) Prefetch only: the extra CTAs read weight bytes into L2 and never write anything (mode 0 has no
              loads into registers). No change in math, precision, memory footprint or scheduling; outputs bitwise
              identical by construction. Weight pointers are static, so the hint is safe in CUDA graphs; the hint is
              global but cleared after each layer's MLP and only consumed by hostmem ARs (<=128 rows), so drafter /
              mixed steps are unaffected. AR blocks have no inter-block dependence, so extra CTAs can't deadlock.
              No benchmark-specific logic. No memory change, so no 128K test needed.

official #1 (2026-11-14T22:25): 897.4 tok/s at c=16 (mean of 2 trials: 895.7 / 899.2, CV 0.3%) (+41.1% vs BASE), GSM8K 93.6%, MTP acceptance 2.337, TTFT p50 159 ms, TPOT p50 16.5 ms; gates: GSM8K pass, latency pass, 128K request pass (official/1/results.json)
