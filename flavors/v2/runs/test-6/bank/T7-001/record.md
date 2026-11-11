team: T7
builds_on: T1-042
performance: "microbenchmark only: split-half RS/AG pipeline on copy engines (arm H) hides only ~10-16% of the AR at R=256/384 and is SLOWER than stock host AR at R=128 (H 674 vs S 641 us/layer). Gate (>=60% hidden at R=128 and 256) FAILED."
summary:
  hypothesis: split-half reduce-scatter/all-gather on copy engines, overlapped with half-GEMMs, hides most of the prefill-size AR (leader estimate ~145 of ~190 us/layer at R=128).
  change: none to the engine. Microbench (bench.py, half_ar.cu in this dir): 2 processes, real FP8 Marlin, per-rank shapes, 8-layer chain, CUDA graphs, interleaved 5 reps x 25 replays. Arms: S = host AR (mode 20 compression, grid/k from _config, no L2 prefetch since msg > 1 MB), N = copy instead of AR, H = the schedule (N-split producer GEMMs peer-half first, D2H+doorbell, rsn kernel = bf16 add + residual add + bf16(h*g), AG of y half, K-half consumer A then B, bf16 add / fused add+silu_mul), HG = H's launches and kernels with no copies, H32 = H plus an fp32 carry of A (bf16->fp32 write + fp32 read; upper bound on the carry cost).
  results: |
    Observed, us/layer (median of 5 reps) rank0 / rank1:
      R=128: S 641/640  N 465/403  H 674/675  HG 510/447  H32 679/679
      R=256: S 1151/1156 N 822/717  H 1108/1111 HG 856/724 H32 1134/1140
      R=384: S 1646/1651 N 1181/1001 H 1571/1570 HG 1237/1052 H32 1605/1608
    AR cost S-N: R=128 176/237, R=256 330/439, R=384 465/650. Hidden fraction (S-H)/(S-N): R=128 -19%/-15%, R=256 13%/10%, R=384 16%/12%.
    Cost of splitting the GEMMs alone (HG-N): R=128 +45/+44, R=256 +34/+7, R=384 +57/+51 us/layer. fp32 carry (H32-H): +5/+4 (128), +25/+29 (256), +34/+38 (384).
    Timeline R=128, layer 1, boundary 1 (out_proj -> gate_up), us from start, rank0 (rank1 similar): prodA(peer half) done 29; D2H of 655 KB done 77 (48 us, ~14 GB/s); prodB(own half) done 57; peer partial H2D'd by 125 (waiting for peer's D2H + 48 us H2D); rsn 130; AG D2H done 199 (69 us for 655 KB, ~9.5 GB/s while consumer A runs); consA done 236 (106 us for a 44 MB-weight half GEMM; ~45-50 us expected); AG H2D done 244; consB done 354 (107 us). Stamps are kernels on the streams, so timeline is slightly perturbed (HT arm 685 vs H 674).
  conclusion: |
    Interpretation: the RS phase cannot hide: a 655 KB half partial needs ~48 us D2H + ~48 us H2D, but the own-half producer GEMM is only ~28 us (out_proj) / ~25-30 us (down), so ~100 us/boundary is exposed. The AG phase is partly hidden under consumer A but A runs ~2x slower than expected while copies are in flight (106 us) and the copies themselves slow to ~10 GB/s, so little is gained. Splitting the GEMMs costs +35-57 us/layer by itself, which eats the rest. Net: ~0 at R=128, ~10-16% at R>=256. Direction dead as specified (the 60% gate fails by a wide margin). Napkin estimate (leader): 145 us saved at R=128; measured: -33 us. My own pre-run napkin (D2H+H2D ~66 us per transfer vs 10-50 us of GEMM to hide under) predicted ~15% hidden: closer to the result.
    Not measured: champion token-split path (S2 arm in bench.py, WITH_S2=1): it did not complete in the smoke job (suspect deadlock host-AR-on-aux + DMA AR in my reimplementation, not investigated), so no comparison vs the champion's R>=256 path. S above is host AR only (champion path at R=128, where the split is off). Residual-norm 1/rms sidecar (per-row sumsq) not modeled.
validity: measurement only (no engine change; random weights, bf16 outputs for A+B add).
