team:         T2
builds_on:    T3-001
performance:  "change not exercised in the measured config: 1 start 804.2 tok/s (GPU job 57; T3-001 816.6, starts 811.3/821.8), stopped by STOP_BELOW. No performance evidence for or against the fusion."
summary:
  hypothesis: GDN out_proj's dyn128 activation quant runs as a standalone launch (~3 us x 48 layers ≈ 0.15 ms per decode
              step, napkin +0.5%); fuse it into the producer of out_proj's input.
  change:     change.diff (authoritative delta = 2 files on top of T3-001; og.patch also contains inherited norm hunks):
              ops/fused_gdn_gated_norm.py gains a QUANT_DYN mode (group = one 128-wide head; scale = amax/448), and
              model.py's _fused_gdn_gated_rms_norm_quant returns a QuantizedActivation for dyn128 consumers.
  results:    Observed:
              - Unit test (GPU job 56; M = 0..2048, 3 amplitudes, strided z): scales identical everywhere; FP8 codes
                identical for M <= 320; at M >= 1024, 0-3 codes in ~6M differ. Debug job 54: 6-19 bf16 y elements per
                run differ from the unfused kernel.
              - Harness, job 57: 1 start at 804.2 tok/s, then stopped. Latency gate passed; 128K and GSM8K not run.
              - reviews/T2-7, confirmed by the worker from the code: the server log shows "GDN spec decode kernel: cuda".
                With MTP that enables the fused spec-decode branch in qwen_gdn_linear_attn.py forward_cuda (~l.1327),
                and the branch has no token-count condition. It runs attention plus gated norm in one custom op
                (qwen_gdn_attention_core_fused_norm_packed), writes bf16, calls out_proj and returns. It never reaches
                the modified callback, so the QUANT_DYN code did not run in job 57.
  conclusion: Interpretation: T2-008 is a null result for an untested change, not a measured loss. The standalone out_proj
              quant does exist, but on the packed CUDA path. Fusing it means changing the epilogue of the
              qwen_gdn_attention_core_fused_norm_packed CUDA op, which was left for a later team (napkin +0.5%, plus
              o_proj ~+0.15%; measure 3+ starts). The large-M y differences are consistent with fp reordering or
              contraction differences between compiled kernel variants; the cause was not isolated.
              Lesson: confirm that the live path is taken (log line, counter or profile) before measuring.
validity:     leader-2: same W8A8 scheme; dormant code in the measured config. Not a candidate.
review:       reviews/T2-7: bypass finding, wording of the y-difference cause, and use of change.diff over og.patch;
              all agreed by the worker (leader-2.w1) and adopted.
