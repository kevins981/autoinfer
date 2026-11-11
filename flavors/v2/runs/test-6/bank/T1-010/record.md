team:         T1
performance:  "server failed to start (CUDA OOM in graph capture) at KV 6.0 / 5.625 / 5.5 GiB"
summary:
  hypothesis: k=3 on T1-009 with --kv-cache-memory raised so 16 requests fit (13 blocks/req x 16 = 208 blocks; ~38 blocks/GiB
              -> >= ~5.45 GiB).
  change:     T1-009 + num_speculative_tokens=3 + --kv-cache-memory 6442450944 / 6039797760 / 5905580032. Worker leader-1.w5,
              ws leader-1.k3kv (out/k3kv_<bytes>/server.log), jobs 32, 34 (5.9 GPU-min).
  results:    Observed: all three OOM during CUDA graph capture (at 5.5 GiB: in "Capturing prefill CUDA graphs (PIECEWISE)" after
              the speculator capture). At 5 GiB (T1-008) start OK, graphs 1.34 GiB. Initial free 22.89 GiB, model 14.99 GiB ->
              ~7.9 GiB for KV + graphs + speculator + comm + non-torch.
  conclusion: Interpretation: with the 2048-token prefill graph set, the KV ceiling is between 5.0 and 5.5 GiB, below the ~5.45
              GiB k=3 needs. Next: smaller prefill graph cap (T1-011), or shrink GDN spec pages (R3-B v2).
validity:     no run; nothing to evaluate.
