team:         T2
builds_on:    T2-006
performance:  "screen negative (1 pair, -7.52%): B 743.2 vs A 803.7 tok/s, T2 GPU job 46"
summary:
  hypothesis: breakable CUDA graphs (auto-enabled for Qwen3.5) force CompilationMode.NONE, so vLLM's torch.compile
              fusions (norm_quant, act_quant: RMSNorm + FP8 group quant) never apply; with W8A8 these would remove the
              separate activation-quant kernels (~0.5 ms/step).
  change:     env VLLM_USE_BREAKABLE_CUDAGRAPH=0 (B arm only), T2-006 code and flags.
  results:    Observed: B log: mode VLLM_COMPILE, but "torch.compile is turned on, but the model ... " and "Cudagraph mode
              FULL_AND_PIECEWISE requires piecewise capture, but the loaded model provides neither a compiled submodule
              nor break..."; graph capture 0.31 GiB (vs 1.42 GiB in A): no piecewise graphs for mixed steps.
  conclusion: Interpretation: this model has no torch.compile path here; turning off breakable graphs just loses the
              mixed-step graphs (back to eager, cf. T2-001) -> large loss. The norm+quant fusion would need a hand-written
              fused RMSNorm+per-token-group-quant kernel instead. Not a measurement of the fusion itself.
validity:     n/a (not pursued).
