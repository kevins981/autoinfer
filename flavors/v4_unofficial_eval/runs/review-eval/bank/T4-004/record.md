team:         T4
builds_on:    T3-007
performance:  "build failed at server start: CUDA_ERROR_OUT_OF_MEMORY in cuGraphInstantiateWithFlags during graph capture (eval/1, GPU job 94, 1.6 GPU-min)."
summary:
  hypothesis: (review T4-5, closing review) T3-007 captures mixed-step graphs on a coarse token grid (..., 256, 320, 384, 448, 512, 640,
              ...); padded rows are processed by GEMMs and the 132 all-reduces. A grid every 8 tokens up to 1024 (generic, keeps all
              existing buckets) removes ~8.7K of 10.3K padded rows over the 400 prompts (from T4-002's OFF iteration log): est. ~+1%.
  change:     flag only (change.diff = T3-007's): --compilation-config {"max_prefill_cudagraph_capture_tokens":2048,
              "prefill_cudagraph_capture_token_sizes":[1,2,4]+range(8,1025,8)+[1280,1536,1792,2048]} (135 captures vs 31). serve_args.txt.
  results:    Observed: server start failed: CUDA out of memory instantiating a captured graph (eval/1/trial1/server.log).
  conclusion: Interpretation: memory headroom (T3-003 trimmed KV to 4.4 GiB to fit the fp8 draft head) does not cover ~100 extra
              piecewise graphs. A coarser grid (e.g. every 16) or freeing memory elsewhere (GDN spec state slots, see T4 update-2)
              would be needed; untested for lack of time/budget.
validity:     Flag only; no math change except GEMM tile-config selection for new M (would need GSM8K). Not run.
