# systematic_engineer ideas (round 1)
1. L2 weight prefetch during down_proj all-reduce (top bet; 5-8%, med-low conf). Microbench first (~15 GPU-min).
2. Cheaper small all-reduce: (a) NCCL env tuning (PROTO/ALGO/BUFFSIZE/NTHREADS), (b) FLASHINFER_PCIE_IPC backend, (c) custom 2-rank pinned-host AR fused with residual+RMSNorm. 6% FFN-only, 12% all. 
3. Marlin tuning toward DRAM roofline (80->90%) + SiLU*mul epilogue fusion: ~3-4%.
4. Batch-shape/padding audit; max-num-seqs and k=3 spec: 3-6%, low conf, not FFN.
5. down_proj N-split with overlapped chunked all-reduce: <=1 ms realistic, low conf.
Suggested order: 2(a/b) cheap first, then 1, then 3.
