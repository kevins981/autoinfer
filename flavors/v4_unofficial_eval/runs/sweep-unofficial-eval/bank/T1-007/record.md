team:         T1
builds_on:    T2-011
performance:  screen negative: -1.6% (1 start each, jobs 53 vs 54); not unofficial-evaluated
summary:
  hypothesis: fusing the residual add + gemma RMSNorm that follow each decode all-reduce (Qwen3.5 forward_fused) into the host-pinned AR kernel saves a launch and an HBM pass per AR: napkin 3-6 us x ~128 = 0.5-0.8 ms per decode step (1.5-2%).
  change:     new host_ar_add_rmsnorm CUDA kernel (fixed grid 4 x 512 threads, row-aligned split, same seq/parity scheme as the plain host AR), custom op vllm::host_ar_add_rmsnorm, dispatched in fused_allreduce_gemma_rms_norm for bf16 <=128 rows; env VLLM_HOST_AR_NORM=0 disables. Built on T2-011 code.
  results:    Observed: offline (job 52, M=1..128 x 3 seeds): residual bitwise equal; normed output max abs diff 1 bf16 ulp in ~4.5e-6 of elements (reduction order / rsqrtf). Server, HEAD args, 100 prompts c=16, separate jobs: fused 703.4 tok/s (job 53) vs non-fused 714.5 (job 54), acceptance 2.338 vs 2.342, gsm8k-50 88% both.
  conclusion: Interpretation: slower, probably because 4 blocks x 512 threads process only 16 rows at once (2 serial row passes at 48 rows) vs flashinfer's norm spread over many SMs; the fixed 4-block grid (needed for parity safety) caps parallelism. Screen negative (1 start each, not in the same job); dropped.
validity:     exact up to fp reordering (norm reduction order); not proposed for promotion.
