team:         T4
builds_on:    T3-003
performance:  "screen negative (no harness): quick benches, 1 start each, 200 prompts c=16: job 74 ON 791.5 vs OFF 801.9 (-1.3%); job rp-iter (worker task 4) ON 784.2 vs OFF 795.6 (-1.4%). Both within start-to-start noise, neither shows a gain."
summary:
  hypothesis: nsys of T3-001 (job 66) showed mixed steps (a prompt + ~15 decoding requests) with ~60 prompt rows costing 72-122 ms vs
              32.7 ms for decode, with ~1730 eager launches per step, mostly the FLA chunked GDN prefill pipeline (~12 Triton launches
              + gather/scatter per GDN layer) -> "mixed steps are CPU-launch-bound" (T4 map, 6-ranking #1). Replace the chunk pipeline
              for short prompts by ONE recurrent Triton kernel per layer (framework 03 §4.2 dual form, exact identity).
  change:     by worker leader-4.w1 (workspace t4-rp): new vllm/model_executor/layers/mamba/gdn/recurrent_prefill.py (one program per
              (V-block of 16, seq x HV); l2norm, gating, decay -> delta -> output in fp32; initial state from ssm_state or zeros; final
              state stored once if idx > 0); hook in qwen_gdn_linear_attn.py _forward_core (spec decodes present or not; skipped with
              checkpoints/flashinfer backend) when max per-sequence prefill length <= VLLM_T4_GDN_RECURRENT_PREFILL_MAX (default 512;
              0 = off); gdn_attn.py metadata gains max_prefill_seq_len (CPU value). Serve args = T3-003's.
  results:    Observed: microbench (agents/leader-4.w1/rp_results.md): CPU per call ~390 us (old) -> ~22 us (new); GPU ~0.7 us/token
              (T=256: 192 vs 57 us old). Numerics: vs old path rel. err 4.4e-3 (outputs), 3e-3 (state); vs an fp64 token-by-token
              reference new 1.7e-3 / 1e-7, old 4.0e-3 / 3.0e-3 (old stores q,k in bf16; g is fp32, review T4-2). Server (job 74): path fires on every
              prefill-containing step (spec=True, prompt lengths 58-487); GSM8K-50 88.0% (= BASE's 44/50). Per-iteration timing without
              nsys (eval/iter_quickbench/table.md, --enable-logging-iteration-details): mixed p<128 rows median 55.9 (ON) vs 65.5 ms (OFF);
              128-256: 80.1 vs 77.4; 256-512: 130.4 vs 119.4; >=512: 209.8 vs 194.3; decode D=16 31.6 vs 31.0; total mixed time 20.0 s in
              both.
  conclusion: Interpretation (revised after review T4-2): nsys likely exaggerated the launch overhead (task/PROFILING.md warns of
              it): unprofiled, a small mixed step costs 56-66 ms, not 72-122. The replacement improves small mixed steps (matched shapes,
              one 58-64-row prompt + 15 decoders: 54.2 ms ON vs 65.3 OFF, 20/18 observations) and loses on prompts above ~256 rows
              (its GPU cost ~0.7 us/token/layer). The unprofiled CPU-vs-GPU split is NOT resolved (elapsed_ms is a future wait that
              includes worker launch delays; the ~52 ms GPU floor is a T3-001 nsys estimate). A ~128-row threshold might give ~1% (an
              estimate, not a measurement): not pursued for lack of time.
validity:     leader-4: same function up to fp reordering (token-by-token form of the same recurrence; fp32 throughout, more accurate
              than the old path vs fp64). No benchmark keying (generic size threshold). Not a candidate, so GSM8K-250 not run (GSM8K-50 ok).
              Review reviews/T4-2: two code bugs to fix before any revival: (a) the kernel loads state_idx_ptr + i_n, but without spec
              rows the indices are block_table[:, 0] with stride 1+num_spec_blocks -> with several prefills in one step it reads/writes the
              wrong slots (pass the stride); (b) idx <= 0 guards the store but not the initial-state load. Not hit visibly in the screens
              (mostly 1 prompt per step, spec rows present), but the screen numbers are from this code. Tests needed on revival: strided
              multi-sequence indices, mixed fresh/resumed states, invalid slots, both output routes.
