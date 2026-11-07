# T1-003 profile anatomy (T1-001 host AR + T2-003 flags), nsys, 64 ShareGPT prompts c=16, GPU 0

Profile: prof_t1003/profile.sqlite (job 17). Script: agents/leader-1.w1/anatomy.py (validated on the BASE profile:
it reproduces BASE decode 39.5 ms span, Marlin 18.58, NCCL AR 8.10, GDN 3.88, logits AG 1.62, large-AR skew p50 183 us).
Steps = clusters of GPU-0 kernels separated by gaps >250 us. "Steady decode" = decode clusters (900-1100 kernels, span<60 ms)
with AR time >= 0.85 x p90 (i.e. full c=16 batches). nsys adds overhead: shares/anatomy, not throughput.

## (1) Steady decode step (249 full-batch steps), vs BASE (124 steps)
| category | BASE ms/step | T1-003 ms/step | calls (T1-003) |
|---|---|---|---|
| Marlin FP8 GEMM | 18.58 | 18.46 | 260 |
| all-reduce | 8.10 (NCCL, 132) | **5.88 (host ar_kernel, 132)** | 132 |
| GDN / linear attn | 3.88 | 3.84 | 97 |
| lm_head + other GEMM (Kernel2 + ampere gemm) | 4.9 (3.29 + 1.59) | 2.64 in-graph (1.59 gemm + 1.05) ; the rest of lm_head runs in a small cluster after the step (see below) | |
| logits AllGather / NCCL other | 1.62 | 0.53 (incl. SendRecv 0.46) | |
| norms/act/sampling/copies | 3.94 | 1.59 | |
| attention + KV write | 0.87 | 0.87 | |
| **main-cluster span** | 39.47 | **32.99** | |
| **step period (start to next decode step start)** | 39.83 | **36.95** | |

- Per-AR: 5.88 ms / 132 = 44.5 us avg in the profile (mix of T=48 verify and T=16 drafter ARs) vs NCCL 61.4 us. AR saved 2.2 ms per step.
- The main-cluster span dropped 6.5 ms (more than the AR saving: T2 flags move lm_head/argmax/logits AG out of the graph step). The step period only drops 2.9 ms because ~4 ms now sits between decode clusters: small clusters that hold Kernel2 (lm_head cutlass, 629 launches, 271 ms total ~0.43 ms each), reduce_kernel, a 641-launch NCCL AllGather (9 ms), Marlin (drafter) and ar_kernel (42 ms). Sum of inter-cluster gaps (>250 us): 1008 ms of 26.2 s (BASE: 180 ms). So the T2-flag path adds CPU-side/launch gaps between decode sub-steps: that is the new idle to attack in round 2.

## (2) Steps with a prefill chunk
- Prefill-like clusters (span>60 ms): 52 (BASE 112). Several of the large ones:
| span ms | Marlin | NCCL AR | GDN | other | attn |
|---|---|---|---|---|---|
| 590 | 360.7 | 181.4 | 16.6 | 19.7 | 6.9 |
| 581 | 354.1 | 179.5 | 17.0 | 17.4 | 7.7 |
| 307 | 182.5 | 91.8 | 12.4 | 8.7 | 5.3 |
| 295 | 171.9 | 92.3 | 12.1 | 8.4 | 4.8 |
| 250 | 148.3 | 72.5 | 11.3 | 7.5 | 4.1 |
  (BASE equivalents: 582 ms: Marlin 353.5, NCCL 180.3; 291 ms: Marlin 171.6, NCCL 86.5.) Prefill steps are unchanged: ARs there are > 2 MiB (NCCL), so the AR share is still ~31% of a prefill step (181/590).
- Start skew between the GPUs at those (large, >100 us) NCCL ARs, matched by order, n=3535: **p50 196 us, p90 638 us** (BASE p50 183 us, p90 1103 us). Skew ~ unchanged.
- Note: the AR in the 590 ms step takes 181 ms; the prefill chunk ARs are ~8x larger per call than the NCCL decode ones and cost ~1.4 ms each at T=2048.

## (3) Whole profile, GPU 0 (span 26.2 s, kernel-busy union 89.2% -> idle 10.8%; BASE 94.7% busy, idle 5.3%)
| category | seconds | % of kernel time | % of span | BASE % of kernel |
|---|---|---|---|---|
| Marlin FP8 GEMM | 13.26 | 55.8 | 50.6 | 51.7 |
| host AR kernel | 3.24 | 13.6 | 12.4 | – |
| norms/act/sampling/copies | 2.19 | 9.2 | 8.3 | 10.8 |
| NCCL AllReduce (prefill-sized) | 2.09 | 8.8 | 8.0 | 24.6 (all ARs) |
| GDN | 1.64 | 6.9 | 6.3 | 6.3 |
| other GEMM (lm_head etc) | 0.57 | 2.4 | 2.2 | 2.0 |
| attention + KV | 0.55 | 2.3 | 2.1 | 2.2 |
| NCCL AllGather/other | 0.23 | 1.0 | 0.9 | 2.5 |
- All-reduce total (host + NCCL) = 5.33 s = 20.4% of span vs 23.4% in BASE. Idle doubled (5.3% -> 10.8%) because of the extra decode sub-step gaps from the T2-flag path.

## Raw output (anatomy.py)
```
clusters 1827, big 540, decode-like 488, prefill-like 52
decode cluster kernel counts: [(960, 160), (1010, 94), (961, 85), (962, 37)]

== (1) steady decode step: 249 steps with 959-1051 kernels (GPU0) ==
main-cluster span median 32.99 ms
step period (start of step to start of next decode step, incl. sampling/prep gaps) median 36.95 ms
category                                      ms/step  calls
Marlin FP8 GEMM                                 18.46  260
host AR kernel                                   5.88  132
GDN / linear attn                                3.84  97
GEMM other (lm_head etc)                         1.75  50
other (norms, act, sampling, copies)             1.59  250
attention (full)+KV write                        0.87  167
NCCL AllGather/other                             0.53  4

top kernels in a decode step (ms, calls, name):
   18.46   260  Marlin
    5.86   132  ar_kernel
    3.64    48  gdn_decode_post_conv_mtp_kernel
    1.59     1  ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn
    1.05    50  Kernel2
    0.48   130  kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_a
    0.46     1  ncclDevKernel_SendRecv
    0.32    17  BatchPrefillWithPagedKVCacheKernel
    0.20    65  act_and_mul_kernel
    0.19    48  _causal_conv1d_update_kernel
    0.16    49  splitKreduce_kernel
    0.08     3  ncclDevKernel_AllGather_RING_LL
    0.05    51  vectorized_elementwise_kernel
    0.05    17  reshape_and_cache_flash_kernel

small clusters (<300 kernels, >0.5ms): 380 total 759 ms

== (2) steps with prefill chunk: 52 clusters (span>60ms) ==
  span   590.0 ms n= 2933 | Marlin 360.7 | NCCL 181.4 | other 19.7 | GDN 16.6 | attention 6.9
  span   580.6 ms n= 2986 | Marlin 354.1 | NCCL 179.5 | other 17.4 | GDN 17.0 | attention 7.7
  span   307.4 ms n= 2725 | Marlin 182.5 | NCCL 91.8 | GDN 12.4 | other 8.7 | attention 5.3
  span   295.1 ms n= 2725 | Marlin 171.9 | NCCL 92.3 | GDN 12.1 | other 8.4 | attention 4.8
  span   293.2 ms n= 2725 | Marlin 173.9 | NCCL 88.1 | GDN 11.7 | other 8.5 | attention 4.9
  span   291.0 ms n= 2725 | Marlin 171.8 | NCCL 88.3 | GDN 12.1 | other 8.3 | attention 5.0
  span   290.0 ms n= 2725 | Marlin 171.8 | NCCL 86.4 | GDN 12.1 | other 8.3 | attention 4.8
  span   272.6 ms n= 2724 | Marlin 158.9 | NCCL 81.5 | GDN 11.7 | other 8.0 | attention 4.6
  span   262.8 ms n= 2725 | Marlin 152.6 | NCCL 81.4 | GDN 11.4 | other 7.8 | attention 4.3
  span   255.6 ms n= 2725 | Marlin 149.8 | NCCL 75.6 | GDN 11.4 | other 7.7 | attention 4.4
  span   252.8 ms n= 2725 | Marlin 142.2 | NCCL 82.0 | GDN 11.0 | other 7.5 | attention 4.0
  span   249.8 ms n= 2725 | Marlin 148.3 | NCCL 72.5 | GDN 11.3 | other 7.5 | attention 4.1
NCCL AR kernels in prefill steps: GPU0 time per step (ms,n0,n1) [(0.0, 0, 0), (181.4, 131, 131), (179.5, 132, 132), (114.2, 132, 132), (0.0, 0, 0), (0.0, 0, 0)]
  large ARs only (>100us, n=3535): skew p50 196 p90 638
start skew between GPUs at those ARs (us): p50 196 p90 638 mean 341

== (3) whole profile, GPU0: span 26.2 s, kernel-busy (union) 89.2% -> idle 10.8% ==
  Marlin FP8 GEMM                                13.26 s   55.8% of kernel time   50.6% of span
  host AR kernel                                  3.24 s   13.6% of kernel time   12.4% of span
  other (norms, act, sampling, copies)            2.19 s    9.2% of kernel time    8.3% of span
  NCCL AllReduce                                  2.09 s    8.8% of kernel time    8.0% of span
  GDN / linear attn                               1.64 s    6.9% of kernel time    6.3% of span
  GEMM other (lm_head etc)                        0.57 s    2.4% of kernel time    2.2% of span
  attention (full)+KV write                       0.55 s    2.3% of kernel time    2.1% of span
  NCCL AllGather/other                            0.23 s    1.0% of kernel time    0.9% of span

decode-like clusters: (n, span, marlin, ar ms) most common
(960, 33, 6) 88
(962, 24, 1) 36
(1010, 35, 6) 33
(1010, 24, 2) 33
(960, 34, 7) 22
(1010, 25, 2) 21
(961, 32, 6) 19
(961, 29, 4) 19
(960, 23, 2) 19
(960, 22, 2) 17
(961, 33, 6) 15
(961, 25, 3) 12
(912, 22, 1) 10
(1011, 27, 3) 9
(959, 33, 6) 8

```
