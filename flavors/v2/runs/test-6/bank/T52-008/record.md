team:         T52
builds_on:    T52-005 (workspace t52-w2; Python only; flags T52_LEAN_STREAM / T52_LEAN_LIN / T52_LEAN_GDN default on; test-only T52_LEAN_CTL/CHECK, T52_WTIME, T52_CPROF)
performance:  "measured: wave-loop host 117 -> ~85-100 ms/step (bitwise, 128K pass); lowest MIN with host < GPU ~650-700 (job 481, 3.4 GPU-min); no ABBA (expected +0.2-0.5%, below noise)"
summary:
  hypothesis: removing host glue in the eager wave loop (stream handles, direct T29 linear calls, direct GDN core call) cuts ~117 -> ~55-65 ms/step,
              letting T30_WAVE_MIN drop 768 -> 384-512 (~+1% e2e).
  change:     vllm/t52_lean.py + wave loop in model.py: set_stream once per pass and AR priority events on streams[c]; direct norm_quant/gemm_q/
              w8a8_marlin/silu_mul_quant (fallback to module path where ineligible); direct _forward_core_fused_norm_packed. Not done: C++ AR events,
              cached Triton launchers, buffer preallocation (allocator already reuses). Files agents/leader-52.w1/srv/s5/, NOTES.md "Route 1".
  results:    Observed: T52LEANCHECK 32/32 bitwise; sequential greedy lean/old/lean/old identical (old-old control clean); 128K pass (lean on);
              free memory after capture 0.45 GiB unchanged. Wave loop host (ms/step, lean vs old, same server): n=461 97.5/89.9 vs 108.1/115.3;
              n=989 86.4/84.2 vs 100.3/101.8; n=1600 ~88 vs ~102; bench n=750-900 ~100 (was 115-119). Per wave-pass at n=989: attention core 450 us,
              GDN core 250 (~350 mixed), mlp 76, ar1 68, in_proj 58, ar2 55, out_proj 52, norms 34. Hostprobe: host-bound at n<=542, GPU-bound n>=749.
  conclusion: Interpretation: saved 15-20 ms/step vs estimate 52-62: the remaining host cost is in the GDN/attention cores' Triton/Python launch
              paths (~40-50 ms/step), not in the loop glue; cProfile over-weighted deep Python stacks. Best further case (cached launchers + C++ AR)
              ~70-75 ms -> MIN ~512-576, worth ~+0.2-0.5% e2e (estimate), below same-job noise. Not pursued; not for official eval.
validity:     bitwise to T52-005 on tested paths (same kernels/args/order); not submitted.
