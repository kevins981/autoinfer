team:         T1
builds_on:    T2-018
performance:  "unofficial: PASS +0.34% vs T2-018 (GSM8K 93.2%)"; quick ABBA job 144: +1.03% (B 780.3/788.1 vs A 773.6/778.7, 100 prompts)
summary:
  hypothesis: Marlin's out_proj/o_proj (5120x3072 per rank) runs at 25 us at M=48 vs a 15.6 us HBM floor. A Triton W8A16
              kernel on the checkpoint layout (fp8 [N,K] + 128x128 block scales, bf16 activations, fp32 accumulate)
              does it in 20 us (graph-timed, job 141/143). Napkin: ~5 us x 64 layers = ~0.3 ms per decode step (~1%).
              This is also step 1 of a larger path. Without the Marlin repack, one weight copy can serve W8A16 for decode
              and W8A8 block-fp8 for prefill (0.70x Marlin at M>=768, job 138). See sparring/ T1-2.
  change:     new vllm/model_executor/kernels/linear/scaled_mm/w8a16_triton.py: Triton kernel, static config table
              per (N,K,M bucket), custom op with a fake impl (compile/graph safe), warm-up compile at load. In
              scaled_mm/marlin.py: block-quant fp8 layers whose prefix ends in out_proj/o_proj (VLLM_W8A16_TRITON_PROJ,
              default "out_proj", empty = off) and whose (N,K) is in the table keep the checkpoint layout (no Marlin
              repack, same bytes) and use the Triton kernel for all M. GDN linear_attn.out_proj and attention o_proj
              both match. Worker t1-w1, workspace t1-w11.
  results:    Observed: offline (job 143, M=1..1024): max abs diff vs Marlin 0.0039 (1 bf16 ulp); graph-timed 20.1 vs 25.0 us at
              M=48, 0.88-0.91 at M<=32, 1.00 at M=64, 1.22-1.45 at M=65-128 and 256 (untuned mid-M; ~0.1% of wall),
              0.99 at M=1024. Server ABBA (job 144): A 773.6/778.7, B 780.3/788.1 (+1.03%); median TPOT -2.6%; gsm8k-8 8/8.
              unofficial #1: B 808.9 / 808.7 vs HEAD mean 806.1 (+0.34%); p99 TTFT 1215 / TPOT 48.7 ms; acceptance 2.332.
  conclusion: Interpretation: a small, consistent gain (both starts above HEAD), below the quick ABBA's +1.0%. The untuned
              mid-M configs (65-512 rows, 1.2-1.45x Marlin) cost back part of the decode gain in mixed steps and may explain the higher
              p99 TPOT. Task 29 tunes them.
validity:     Same math as Marlin W8A16: fp8 weights x block scales, bf16 activations, fp32 accumulation. The scale is applied
              to each 128-K partial in fp32 rather than folded into bf16 weights, so differences are rounding/reordering
              only (1 ulp). Weight precision unchanged, no act quant, no benchmark fitting (configs keyed on shapes and the
              hardware). Reviewed by leader-1: BN<=128 and BK<=128, so each tile stays within one 128x128 scale
              block; K divides BK. GSM8K gate run.

unofficial #1 (2026-11-15T11:16): PASS: +0.34% vs HEAD T2-018 (B 808.9 / 808.7 vs reference mean 806.1 over 2 starts, 0 run in this job; 200 prompts); GSM8K 93.2%, p99 TTFT 1215 ms / TPOT 48.7 ms, MTP acceptance 2.332 (unofficial/1/unofficial.json)
