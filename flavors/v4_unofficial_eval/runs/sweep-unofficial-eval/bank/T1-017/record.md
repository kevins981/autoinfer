team:         T1
builds_on:    T1-016
performance:  "unofficial: PASS +3.01% vs T1-016 (GSM8K 93.6%, 128K pass)"; quick same-job runs, job 151 (order A B C C): C 820.3/813.2 vs A 779.7 (~+5%), B (W8A16 only) 766.6
summary:
  hypothesis: Marlin repacks weights, so the faster block-fp8 W8A8 Triton GEMM (0.67-0.79x Marlin at M>=384, job 138/149)
              could not be used for prefill without a second weight copy. T1-016 showed that a checkpoint-layout W8A16
              Triton kernel matches Marlin in decode (graph-timed: gate_up 1.01x at M=48, down 0.93-0.96x, out 0.80x;
              jobs 141/147). Moving gate_up, down and out_proj to the checkpoint layout lets one weight copy serve W8A16
              (small M) and W8A8 (large M). Napkin: -14 ms at a 512-row step, -38 ms at 1024, -68 ms at 2048 for these 3
              GEMMs (job 149), i.e. several % of wall, since prefill/mixed steps are ~35% of wall. From sparring T1-2.
  change:     w8a16_triton.py (on T1-016): VLLM_W8A16_TRITON_PROJ defaults to "out_proj,gate_up,down" (all 64 layers'
              out_proj/o_proj, gate_up_proj and down_proj keep the checkpoint layout, with no Marlin repack). Static M-bucket
              configs come from the job 147 sweep. When M >= a per-shape threshold (gate_up 128, down 160, out 160; job 149),
              the route is vLLM's per_token_group_quant_fp8 (128) + w8a8_triton_block_scaled_mm on the same fp8 weights and
              scales. VLLM_W8A8_PREFILL=0 disables that route. Warm-up compiles all configs at load. qkv/in_proj stay on
              Marlin. Worker t1-w1, workspace t1-w12.
  results:    Observed:
              - offline (job 150): W8A16 max diff vs Marlin 0.4-0.6% of max|out|; W8A8 2.3-2.9% (fp8 activation rounding).
              - memory: weights 15.09 vs 15.21 GiB per GPU; free memory after start about the same (676 MiB).
              - gsm8k-8 8/8 twice.
              - job 151 (100 prompts, c=16): A 779.7 (TTFT p99 1439), B 766.6, C 820.3 / 813.2 (TTFT p99 1019/909, mean TPOT -7%).
                The run order confounds drift with arm.
              unofficial #1: B 831.0 / 835.3 vs HEAD mean 808.8 (+3.01%); GSM8K 93.6% (HEAD 93.2%); 128K pass;
              p99 TTFT 1003 ms (HEAD 1215), TPOT 45.4 (HEAD 48.7); acceptance 2.333.
  conclusion: Interpretation: the W8A8 prefill route on checkpoint-layout weights gives most of the gain (TTFT p99 -17%),
              with accuracy unchanged. W8A16-only on all three shapes is not a gain by itself (job 151 arm B).
validity:     W8A16 part: same math as Marlin up to fp reordering. W8A8 part: activations quantized to fp8 per-token-group-128
              against the checkpoint's 128x128 block weights. This is the checkpoint's own scheme, which task_definition.md
              allows with a GSM8K A/B (the unofficial eval runs --gsm8k against HEAD's 93.x%). Communication stays bf16. No
              benchmark fitting: thresholds and configs are keyed on shapes and hardware. 128K request checked with --long.
              Reviewed by leader-1.

unofficial #1 (2026-11-15T11:38): PASS: +3.01% vs HEAD T1-016 (B 831.0 / 835.3 vs reference mean 808.8 over 2 starts, 0 run in this job; 200 prompts); GSM8K 93.6%, 128K pass, p99 TTFT 1003 ms / TPOT 45.4 ms, MTP acceptance 2.333 (unofficial/1/unofficial.json)

FP8-activation GSM8K A/B (task_definition: Activations > Accuracy), job 157, same job, same code (workspace t1-w12 = this
diff; script notes/job.sh): bf16-activation reference (VLLM_W8A8_PREFILL=0) GSM8K-250 = 93.2% (no-answer 4); W8A8 route on
(default) GSM8K-250 = 93.6% (no-answer 5). Difference +0.4 points, within 2.0. PASS. Results: notes/ref_gsm8k.json, notes/w8a8_gsm8k.json.
