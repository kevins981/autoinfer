team:         T5
builds_on:    T3-007
performance:  +4.9% vs the champion (T3-007, 853.6): 895.4 tok/s, starts 898.7 / 892.1 (eval/1, GPU job 115); both starts above 853.6;
              gain > 1%, so no third start required.
summary:
  hypothesis: in every decode-only step the fused GDN MTP kernel (csrc/gdn/fused_gdn_decode_kernel.cu, gdn_decode_post_conv_mtp_kernel)
              reads 1 fp32 state and WRITES 3 (after each verify row) per request, v-head and layer: ~77 MB written to L2 per layer per GPU at
              c=16. For ordinary continuation only the state at the last accepted row is read again (align-mode boundary copies can read intermediate states; DL is off near boundaries). Row 0 of each verify window is the token sampled last step
              (committed), so a "DeltaLog" (framework 02 §4.3 Row B) can store just the state after row 0 plus a small log of rows 1..T-1
              (normalized k, delta, decay) and replay the a-1 accepted rows at the start of the next step. ncu of the live champion (job 85,
              --cache-control none): 78.7 us/launch, DRAM read 26 MB, write 37.6 MB. Napkin: 1-write kernel 40-64 us -> -1.0..1.4 ms per
              ~28 ms decode step, ~+2.5-3.5% (reports/T5-sweep/6-ranking.md #1, update-1.md).
  change:     by worker leader-5.w1 (workspace t5-dl; change.diff = ws diff t5-dl, md5 3a07bc41, identical to the snapshot taken before job
              115). T3-007's code + new CUDA ops fused_gdn_decode_post_conv_mtp_dl and gdn_dl_materialize (csrc/gdn/fused_gdn_decode_kernel.cu,
              ops.h, torch_bindings.cpp, _custom_ops.py; rebuilt with ws build); qwen_gdn_linear_attn.py: decode-only spec steps use the DL op;
              a per-(slot, head) int32 fmt flag per layer marks "slot 0 = checkpoint after row 0, slot 1 = log"; before every other consumer
              (_forward_core: mixed-step Triton spec path, non-spec decode, prefill, checkpoint export; and the old-kernel fallback) a
              materialize op rewrites the full accepted state into slot[a-1] (old format) using the layer's own per-KV-cache-group
              spec_state_indices[:num_spec_decodes]. gdn_attn.py: a persistent seq_lens copy; DL is used only when the request's
              num_computed % mamba_block_size(1600) is in [64, 1536), so align-mode prefix-cache copies (which fire only at 1600-token
              boundaries) always read canonical full states. VLLM_T5_GDN_DELTALOG=0 restores the old op exactly. Contains env-guarded debug
              code (VLLM_T5_DL_VERIFY, VLLM_T5_DL_NOSTORE, one-time "T5DL" info logs), inert with defaults (review T5-2 turn 2, point 4).
              Serve args identical to T3-007's.
  results:    Observed: eval/1 (job 115, GATES=long,gsm8k): 895.4 tok/s (898.7, 892.1), TTFT p99 1038/1018 ms, TPOT p99 35.6/38.9 ms
              (gate_latency pass), TPOT p50 16.22/16.31 ms (T3-007: 17.28/17.19), acceptance 2.343 (T3-007 2.337), GSM8K-250 93.2% (gate pass),
              128K pass, 0 preemptions; server.log: "T5 GDN DeltaLog enabled", DL op used.
              Exactness: replay preserves the original fp32 recurrence; tested cases were bitwise equal and the fixed integration passes
              accuracy. Bitwise unit test (job 90): outputs and reconstructed accepted states bitwise equal to the old op over 12 steps with
              random accept counts, simulated mixed steps, boundary requests, NULL and zero-length rows (tested cases only). c=1 greedy, DL on
              vs off identical returned TEXT incl. decode across 1,600 tokens and a prefix-cache hit (job 92, on the PRE-fix build; the fix
              changed the mixed-step materialize, the old-kernel fallback and fmt allocation; not re-run). In-graph check (job 108, pre-fix
              build): an old-op reference on a materialized clone of the current state matched the DL output in sampled counter snapshots
              (local recurrence equivalence, not end-to-end); that pre-fix server scored GSM8K-30 53.3% (the cross-group bug).
              Development history (filed for the record): the first c=16 build corrupted outputs (job 91: GSM8K-50 30%, acceptance 2.21);
              cause found by review T5-2 (P1): the materialize used one cache group's block ids for all 4 KV-cache groups. Fixed build: job 111,
              GSM8K-50 88%, acceptance 2.35, quick bench 841.6 vs 825.5 tok/s with DL off (job 92; screens).
              Cold kernel microbench (job 90): 120.4 -> 61.8 us/call.
  conclusion: Interpretation: removing 2 of 3 state writes per GDN layer is the intended mechanism. Median TPOT fell 0.97 ms per output
              token (not the same as per decode step). The gain exceeds the napkin (+2.5-3.5%); a plausible but unproven explanation is that
              deferred L2 write-back no longer slows the following GEMMs; job-to-job drift may also contribute (magnitude and sign unknown).
              The excess is unexplained until the profile below. Debug code is numerically inactive with defaults but adds small
              checks/logging overhead. Mechanism profile: see notes below (filled when job lands).
validity:     leader-5: same function, bitwise: the replay applies exactly the same fp32 ops (decay multiply, rank-1 FMA with the logged delta,
              same order) as the original token loop; the bitwise test (job 90) and the c=1 identical-text test (job 92) confirm. No precision,
              sampling, or verification change; KV cache, weights, activations untouched; no benchmark keying (generic 1600-token boundary
              margin from the engine's own mamba block size). GSM8K-250 run anyway: 93.2%. Reviews: reviews/T5-2 turn 1 (P1 bug found -> fixed;
              P1b fixed), turn 2 (fix verified, no remaining cross-group or stale-row hazard in the champion's V2 serving path, debug code
              inert). Known limitation (P2, not in this config): without an explicit --kv-cache-memory, a KV-cache rebind after the profiling
              pass could leave the fmt tensor wrongly shaped on the materialize path (fails a shape check, i.e. a startup error, not silent
              corruption); the champion's serve args set --kv-cache-memory, which skips that path.
              Review of this record: reviews/T5-2 turn 3 (wording narrowed accordingly; no promotion blocker).
notes (mechanism profile, job 118, nsys TRACE=node, 64 prompts, champion serve args, GPU 0, agents/leader-5.w1/prof_node_dl,
       cat_node_dl.txt; vs job 78 of T3-007, same setup; full-batch decode steps D>=15, 250 vs 246 steps):
  - GDN decode kernel per call 72.7 -> 35.6 us (-51%); per decode step (48 layers) 3,505 -> 1,719 us (-1.79 ms).
  - Following kernels per decode step: gate_up 7,245 -> 7,077, down 3,467 -> 3,317, out/o_proj 1,594 -> 1,320, hmar 6,198 -> 5,962,
    qkvz 1,956 -> 1,937 us: ~-0.8 ms total, consistent with less deferred L2 write-back (one profile per version: part may be noise).
  - Decode-step wall (node trace) 31.61 -> 29.00 ms (-8.3%). Materialize: 48 calls per mixed step, ~1.5 ms per mixed step (~1.4%).
  - Napkin check: -8.3% on ~65% of time (decode) minus ~0.5% (materialize on mixed) ~ +5%, matching the harness's +4.9%. The excess
    over the original +2.5-3.5% napkin is the next-kernel effect the napkin did not count (written bytes deferred past the kernel).
