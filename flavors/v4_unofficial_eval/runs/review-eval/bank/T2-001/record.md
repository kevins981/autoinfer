team:         T2
builds_on:    BASE
performance:  "+8.4% vs the champion (BASE 635.9): 689.5 tok/s (starts 689.7, 689.3), eval/1 (GPU job 4)"
summary:
  hypothesis: Mixed (prefill + decode) steps above 96 tokens exceed BASE's largest captured CUDA graph
              and run eagerly; the 48 GDN layers' chunked-prefill path is then launch-bound (nsys of
              BASE: at ~110-token mixed steps GPU 1 idles 54-65 ms of a ~125 ms step while GPU 0's NCCL
              waits for it; reports/T2-sweep/3-calibration.md B, 5-compare.md). Capturing prefill/mixed
              graphs up to the batch budget (2048 tokens) removes most of that overhead. Napkin: up to
              ~13 s of 143 s (+9%).
  change:     serve flag only (serve_args.txt):
              --compilation-config '{"max_prefill_cudagraph_capture_tokens": 2048}'
              (vLLM's documented option; its default on SM100 is 8192). Prefill graph sizes become
              [1..96, 112, 128, 160, ..., 1792, 2048]. No code change (change.diff empty).
  results:    Observed (eval/1, harness, 2 fresh starts x 400 prompts, GPU job 4):
              689.7 / 689.3 tok/s (mean 689.5, +8.4% vs 635.9; both starts above).
              TTFT p50/p99 203/1186 ms (BASE 235/1341); TPOT p50/p99 21.3/47.5 ms (BASE 23.0/46.1;
              gate 55.3); MTP acceptance 2.337 (BASE 2.345). 128K request: pass (57.3 s; server answers after).
              Graph capture memory 2.07 GiB per GPU (BASE 1.26); KV cache size unchanged (267,721 tokens).
              Server startup: 260 s first start (capturing 31 prefill sizes + compile caches), 58 s second.
              GSM8K-250: 93.2% (5 no-answer), eval/gsm8k_1 (GPU job 8, same flag; BASE 94.4%, gate 89.6%).
  conclusion: Interpretation: the gain matches the napkin (9%) for removing eager-launch overhead of
              mixed steps; median TTFT drops (mixed steps are faster) and median TPOT drops (fewer long
              mixed-step stalls). Mechanism not yet profiled directly (a TRACE=graph nsys of the
              mixed steps would confirm); the size of the gain at a flag that touches only graph capture
              is itself strong evidence.
validity:     leader-2 review: serve flag of vLLM's compilation config; no change to model math,
              precision, sampling, parsers, prefix caching (on), max-num-seqs (16), KV budget (5 GiB
              unchanged). Memory-affecting (graph pool +0.8 GiB): 128K request tested and passed, no
              OOM under the benchmark load. Graphs replay the same kernels on padded batch shapes:
              per-row math is unchanged up to kernel/tile choice for the padded M (fp reordering at
              most). Not tuned to the benchmark (a generic size grid up to the batch budget).
              GSM8K-250 93.2% (gate 89.6%): pass. Latency gate (summarize.py re-run): pass.
review:       reviews/T2-1 (codex, 2026-11-16): measurements, gates, validity confirmed (recomputed 689.459,
              both starts above BASE; TPOT p99 per start 41.8/53.3 <= 55.3; GSM8K 233/250). Corrections
              accepted by leader-2:
              - The grid is not "[1..96] + larger": the new grid also drops some sizes below 96 (3, 6, 12,
                72, 88). Resolved grid: [1,2,4,8,16,24,32,40,48,56,64,80,96,112,128,160,...,2048].
              - Mechanism overstated: cudagraph mode is FULL_AND_PIECEWISE with breakable graphs, and the
                GDN core (qwen_gdn_linear_attn.py, @eager_break_during_capture) still runs eagerly during
                replay. The flag captures the projections, norms, MLP and other eligible work around it,
                not the GDN chunked-prefill kernels. The 13 s "eager overhead" in 3-calibration B is a model
                residual, not a measured removable cost; the +8.4% shows a benefit of wider capture coverage
                without isolating its source. => the GDN prefill core is still eager: a follow-up candidate.
              - Padding safety: attention metadata uses actual token counts, GDN slices to num_actual_tokens,
                padded KV slots get PAD_SLOT_ID; wider padded GEMMs only reorder fp (allowed).
