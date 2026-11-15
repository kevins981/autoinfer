team:         T1
builds_on:    T2-015
performance:  "unofficial cancelled (job 106): HEAD T2-015 was reverted mid-eval; superseded by T1-014 (same change on T1-012)"
summary:
  hypothesis: The target lm_head (bf16, N=124160/rank, K=5120) at M=48 runs a split-K cuBLAS kernel
              (s1688 128x64 sliced, splitK 5) at 1589 us = 0.80 TB/s. cuBLAS's addmm/bias path picks
              s16816 128x64 at ~1440 us (0.88 TB/s). Napkin: ~0.15 ms per decode step (~0.5% of the step,
              ~0.3% total). Near the noise floor.
  change:     vocab_parallel_embedding.py: for a bf16, bias-free ParallelLMHead, a zero bias is allocated at
              load. apply() uses torch.addmm(zero_bias, x, W^T) when rows <= 128
              (VLLM_LM_HEAD_ADDMM_ROWS, 0 = off). The fp8 draft head (a quantized method) is untouched.
              Worker t1-w1, workspace t1-w8.
  results:    Observed: microbench job 99 (M=48, cold, 4 weight copies): F.linear 1589 us; addmm 1439;
              transposed 1440; cublasLt 1437; Triton 1388. Job 105: on random data, top-1 agreement 992/1008,
              all flips near-ties (margin <= 0.25, 6 exact ties); max abs diff 0.25.
  conclusion: (pending)
validity:     Same bf16 products and fp32 accumulation; only the split-K order differs (allowed: fp reordering).
              Adding a zero bias is exact. Near-tie argmax flips are inherent to any reordering. The GSM8K gate
              is run. Not fitted to the benchmark. Reviewed by leader-1.

unofficial #1 (2026-11-15T09:39): FAILED at exit 130 (unofficial/1/)
