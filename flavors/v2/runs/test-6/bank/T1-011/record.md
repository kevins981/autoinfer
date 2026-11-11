team:         T1
performance:  "runtime CUDA OOM seconds into the bench; no numbers"
summary:
  hypothesis: capping prefill graphs at 1024 tokens frees enough memory for KV 5.6 GiB so k=3 holds 16 requests.
  change:     T1-009 + k=3 + --kv-cache-memory 6012954214 + max_prefill_cudagraph_capture_tokens 1024. Worker leader-1.w5,
              ws leader-1.k3kv (out/k3kv_6012954214_cap1024/server.log, out/job3.out), job 37 (1.3 GPU-min).
  results:    Observed: startup OK, KV 290,520 tokens, graphs 1.08 GiB (vs 1.34 at cap 2048). First bench requests served, then
              both workers OOM ("Tried to allocate 28 MiB ... 25.6 MiB free ... 22.18 GiB allocated by PyTorch"); engine died.
              nvidia-smi peak 24,080 / 24,564 MiB.
  conclusion: Interpretation: the binding limit is runtime activation/workspace memory, not graph memory (cap 1024 saved only
              0.26 GiB). KV ~5.0 GiB is the practical ceiling with these args; k=3 at c=16 needs ~0.6 GiB more -> must come from
              smaller GDN spec pages (R3-B v2). R3-A stopped.
validity:     crashed; not a candidate.
