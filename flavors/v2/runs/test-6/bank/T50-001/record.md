team:         T50
builds_on:    T45-005 (no engine change: microbenchmark only)
performance:  "n/a (microbench, job 430, 0.9 GPU-min)"
summary:
  hypothesis: mixed-step W8A8 GEMMs (bs_gemm_marlin, T29) run at ~200-250 TF in situ vs a ~330 TF e4m3 roof; the
              tile table (configs.json) was tuned in job 209 before the reboot (GPU0 clock-locked at 2235 MHz,
              KB113), so a retune might leave >10% on the table at token-split chunk sizes.
  change:     none; bsweep.py: all 43 Marlin-layout configs, 5 shapes, M=128..2048, cold (512 MB read flush),
              GPU0 at 2670 MHz, plus torch._scaled_mm per-tensor FP8 (cuBLASLt, no per-128-K promotion).
  results:    Observed (bsweep.json):
    - table cfg is the best or within 1% at all M for 4 of 5 shapes; gate_up 17408x5120 loses 3-7% at M=192-320
      (table 165.9/207.9/249.8 us vs cfg 103 160.8/194.6/231.4) and 1-2% at M>=768 (cfg 128 vs 103).
    - Efficiency: 136-217 TF at M=128-192, 190-256 TF at M=256-384, 230-271 at 512, 280-296 TF at M>=1024.
    - cuBLASLt per-tensor is 0-15% faster (typically 8-12% at M>=512) and slightly above my 329 TF "roof" at
      M>=1536 (the roof at 2670 MHz is ~340 TF).
  conclusion: Interpretation: no cheap retune beyond gate_up (-> cfg 103 for M buckets 192..1024, used in T50-002).
              bs_gemm is ~88-92% of cuBLASLt (the gap is the mandatory promotion, as KB87/KB91 said). The real slack is
              mid-M (128-512) efficiency (wave quantization / tile shape): 190-250 TF vs ~295 at large M. That needs
              a kernel change (stream-K or larger tiles under the promotion register budget), not a config.
validity:     measurement only.
