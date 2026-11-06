# adjacent_field ideas (round 1) — received as message
1. L2 weight prefetch during the down_proj all-reduce (side-stream kernel warming next layer's gate_up weights into 72MB L2). Est 8-10%, medium confidence. Test: microbench Marlin + pre-touch + concurrent NCCL.
2. Custom 2-rank all-reduce via pinned host memory / FLASHINFER_PCIE_IPC, fused with residual+RMSNorm. Est 10-13% (129 AR x ~40us saved). Test: microbench 0.5MB exchange vs NCCL.
3. PP=2 with 2 interleaved micro-batches instead of TP=2. Ceiling ~25%, risky (vLLM PP+MTP support). Test: --pipeline-parallel-size 2 no MTP.
4. Spec depth k=3 with GDN state pool sized for 16x4 blocks. 5-10%. (not FFN-focused)
5. Gate/up+SiLU epilogue fusion and Marlin pipeline tuning (more cp.async stages, tile sweep). 3-5%, high confidence.
Bets by proposer: 2 and 1 best and stack; 3 highest ceiling riskiest; 4 cheap; 5 last.
