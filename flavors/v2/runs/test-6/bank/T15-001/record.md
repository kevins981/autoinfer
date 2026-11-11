team:         T15
builds_on:    BASE (microbench only, no served change)
performance:  "microbench: HMMA (bf16, f32 acc) and FFMA do NOT co-issue on sm_89: no second exact compute pipe"
summary:
  hypothesis: On GeForce Ada, bf16 MMA with f32 accumulate is half rate (146/165 TF, KB69) while FP32 FFMA is
              ~73/83 TF. A bf16 x e4m3 product is exact in fp32 and FFMA accumulation is IEEE fp32, so an FFMA share
              of each GEMM would be exact-class. If the two pipes co-run, compute-bound GEMMs (prefill ~20 s/trial,
              the tensor half of the M=48 decode floor) could get up to +50% compute.
  change:     copipe.cu (this folder): 1 block/SM; nt warps loop on independent mma.sync m16n8k16 bf16/f32 chains,
              nf warps loop on independent FFMA chains, fixed work per warp; time alone vs together. Job 115, both GPUs.
  results:    Observed (GPU0 | GPU1):
              tensor alone (nt=4/8): 145 | 164 TF.  FFMA alone (nf=8-16): 69 | 78 TF.
              nt=4 + nf=8: 0.708 ms vs 0.463 (tensor alone) + 0.243 (FFMA alone) = 0.706 ms: fully serial.
              nt=4 + nf=12: 0.795 vs 0.463 + 0.364 = 0.827.  nt=8 + nf=8: 1.164 vs 0.922 + 0.243 = 1.165.
              Combined throughput never exceeds tensor alone (best mixed 137 TF on GPU0 < 145).
  conclusion: Interpretation: on sm_89 the f32-accumulate HMMA and FFMA share issue/datapath resources; their work
              adds up in time. An FFMA-assisted exact GEMM cannot beat the bf16 tensor roof. Together with KB69
              (FP8 MMA inexact) and T2/T10/T12 (INT8 slicing needs >= 4-9 passes), the exact compute roof is bf16 MMA.
validity:     measurement only.
