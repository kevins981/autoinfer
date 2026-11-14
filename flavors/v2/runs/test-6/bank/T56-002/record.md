team:         T56
builds_on:    T52-005
performance:  "microbench only, screen negative: t31b row slices are 1.7-1.85x SLOWER than today's quant_perm+bs_gemm per GDN layer at M=112-192 (gate was >= 15% saving)"
summary:
  hypothesis: for 96 < M <= 192, running t31b (register-direct e4m3) in <=96-row slices on the same weights (later slices L2-warm) beats the bs_gemm dispatch.
  change:     none in the engine. bench.py (this dir; job 495, 0.3 GPU-min, GPU0, champion kernels from t56-champ, synthetic randn->e4m3 weights, scales 0.01, bf16 randn activations). Every variant is captured in a CUDA graph and replayed after a 512 MB READ flush + cudaCtxResetPersistingL2Cache (KB76/KB103); median of 15. A = quant_perm + bs_gemm_marlin at M (T29.pick cfg, split 1). B = quant_perm once at M, then t31_gemm_e4m3 on row slices xq[i:i+m]: s64 (64-row slices + remainder), s96 (96 + remainder), even (ceil(M/96) equal slices); regime cold for all slices (cc) or cold first slice + warm later ones (cw). Per-GEMM numbers are one GEMM per graph (cold); "chain" = the 4 GEMMs of a layer in one graph with one flush before it (no AR/norm/GDN between them, so later GEMMs are no colder than in a real layer; L2 reuse across GEMMs is not modeled as the weights are distinct).
  results:    Observed (us, cold, quant included; A vs best B per M, B = s96_cw except M=192 even_cc):
    M     A: qkvz out  aqkv o    down gate_up | GDN layer sum/chain | ATT layer sum/chain | best B GDN sum/chain
    112   55.3  33.9 54.3 33.8 78.9 123.9 | 292.0/291.8 | 290.9/290.8 | 488.2/487.1 (+67%)
    128   53.3  32.8 51.4 32.8 79.8 124.9 | 290.8/293.8 | 288.9/292.9 | 487.4/487.4 (+66%)
    160   75.8  32.9 71.8 32.9 81.2 163.8 | 353.8/344.9 | 349.7/338.7 | 632.4/629.8 (+83%)
    192   82.9  33.5 74.8 33.8 82.9 165.9 | 365.2/359.7 | 357.4/350.1 | 662.9/662.4 (+84%)
    Per-GEMM best B at M=128 (s96_cw): qkvz 75.8 (A 53.3), out 41.0 (32.8), aqkv 70.7 (51.4), o 41.0 (32.8), down 86.0 (79.8), gate_up 284.7 (124.9, expected loss). Every shape at every M loses; 64-row slices are worse than 96-row (s64 GDN 618 us at M=128). Warm-regime config for slices 2+ helps 1-14 us (down most) but never closes the gap. Not even qkvz/out/aqkv/o alone win: B/A = 1.1-1.5x (qkvz M=128), e.g. out 41.0 vs 32.8.
    Correctness (Observed): t31b vs bs_gemm on the same words: max|diff| <= 0.0156 on outputs of max ~3-4.5 (rel Frobenius 0.6-3e-5, i.e. promotion-order level), no M/slice dependence beyond that. t31b row results vs slicing: qkvz bitwise identical for 96/64/48/32-row slices and for cold vs warm cfg; down differs by up to 0.0078 (1 bf16 ulp at |y|~2-4) between slice sizes/cfgs (split-K/reduction order depends on M cfg). So slicing is not bitwise to t31b-at-M but it is within the kernel's own M-dependent rounding.
  conclusion: Interpretation: the premise (t31b's 48-row speed repeating per slice, later slices L2-warm) does not hold: t31b at 96 rows costs ~70-85 us on the 42 MB qkvz (A's bs_gemm at M=112-128 is 55 us, ~ t31b's M=48 cold 55.3 per KB79), i.e. t31b at 96 rows is already mma-bound rather than weight-stream-bound, so the second slice re-pays most of the MMA (and re-streams, because the 40-90 MB tensors do not stay resident between slices under graph replay: warm cfg gains only 1-14 us). bs_gemm at M=112-128 is flat (qkvz 55, gate_up 124) up to the next m-tile, so today's dispatch is already near the weight-stream floor there. Dead: no in-server build. (M=160/192, where A steps up, is also worse for B: +83%.)
validity:     microbench only, no engine change, never a candidate.

leader review (leader-56): NO-GO. Sliced t31b loses 66-84% per layer; the weights do not stay L2-resident between slices.
Side fact for the mid-M GEMM candidate: bs_gemm at M=112-128 costs 291 us per GDN layer (cold). That compares with:
- a DRAM floor of 200 us (192 MB at 957 GB/s);
- an e4m3 MMA floor of ~170 us (49 GFLOP at 290 TF).
A new kernel reaching ~1.1x the floor would save ~25% of mid-M GEMM time, ~5 ms per small mixed step, i.e. ~+1-2% e2e.
That needs a new stream+MMA kernel on the Marlin layout (or a second weight copy), so it is a GEMM-kernel project.
