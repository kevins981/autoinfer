team:         T2
builds_on:    T2-004
performance:  "screen: -9.4% vs same-job T2-004, 1 start each (667.7 vs 737.1 tok/s), job 21 (quick bench 400 prompts c=16; not a harness run)"
summary:
  hypothesis: BASE-family runs compilation mode NONE, so vLLM's fusion passes (fuse_norm_quant: RMSNorm + per-token-group
              fp8 quant; fuse_act_quant: SiLU-mul + quant) never run, and every body linear does its own separate
              activation-quant op (~3 us/GEMM in graph, 4 GEMMs x 64 layers ~ 0.8 ms of a ~20 ms decode step).
              Turning on torch.compile (VLLM_COMPILE) with the fusion passes might remove it.
  change:     flags only (no code): VLLM_USE_BREAKABLE_CUDAGRAPH=0 +
              --compilation-config '{"max_prefill_cudagraph_capture_tokens": 2048, "custom_ops": ["all","+quant_fp8"],
              "pass_config": {"fuse_norm_quant": true, "fuse_act_quant": true}}' on T2-004's other flags
              (workspace t2-w8a8, T2-004 config). Script: agents/leader-2.w1/fuse/screen.sh.
  results:    Observed (job 21, same job, 1 start each, quick bench 400 prompts c=16): compile 667.7 tok/s (TPOT p50
              21.6 ms, TTFT p50 204 ms, acceptance 2.340) vs T2-004 config 737.1 (19.9 ms, 194 ms, 2.335): -9.4%.
              Server logs: "`torch.compile` is turned on, but the model ... does not support it"; "Cudagraph mode
              FULL_AND_PIECEWISE requires piecewise capture, but the loaded model provides neither a compiled
              submodule nor breakable CUDA graphs. Overriding to FULL_DECODE_ONLY". No fusion pass runs
              ("Enabled custom fusions: norm_quant, act_quant" is only the config flag, also printed in T2-004).
  conclusion: Interpretation: (1) with default breakable CUDA graphs (this architecture's default)
              config/vllm.py:950 forces mode=NONE, so no flag can enable the fusion passes there; (2) opting out
              leaves a model that does not support torch.compile and no piecewise/prefill graphs (loses the
              T2-001 prefill-graph gain), consistent with losing the prefill graphs (the comparison does not isolate that as the entire cause): -9.4%. Dead end for the flag route. A real fused quant needs model
              code under the breakable-graph path (QuantizedActivation handoff from norm/SiLU-mul to the linear;
              layers/fusion/quant_activation.py; only minimax_m3 uses it today) plus a fused RMSNorm+group-quant
              kernel; gain ceiling estimated ~+1% (revised in agents/leader-2.w1/fuse/plan.md; the earlier +2-3% used a wrong step time). Weak negative: 1 start each; the gap is far above noise but this is a screen.
validity:     n/a (negative result, flags only; no change to math)
review:       reviews/T2-4: mechanism supported (vllm.py:950 forces NONE with breakable graphs; no flag path runs the
              passes with breakable graphs). Corrections: the -9.4% is "consistent with" losing prefill graphs, not
              isolated; the implementation note is wrong that only minimax_m3 uses QuantizedActivation: Qwen already
              uses it, its MLP calls maybe_fused_act_quant (a dynamic-group-128 SiLU producer exists); Qwen's norm
              fusion accepts static FP8/NVFP4 only, not this dynamic-group scheme. Work = extending existing
              producer/consumer support; the +2-3% ceiling is an estimate.
              worker (leader-2.w1) answer: agrees with all review points as applied; leader-2 judgment: adopted.
