team:         T1
builds_on:    BASE
performance:  +6.7% vs the champion (BASE 635.9): 678.4 tok/s, starts 677.0 / 679.7 (eval/1, GPU job 5)
summary:
  hypothesis: the MTP drafter reads the full bf16 lm_head (124,160 x 5120 per rank) in each of its 2 draft
              steps; an FP8 copy for the drafter only halves those bytes. Napkin (reports/T1-sweep/4-knobs.md):
              -1.4 ms/decode step, ~+2%.
  change:     serve flag only (serve_args.txt): speculative-config {"method":"mtp","num_speculative_tokens":2,
              "draft_lm_head_quantization":"fp8"} (vLLM's built-in option). Empty change.diff.
  results:    Observed: harness eval/1 (job 5): 678.4 tok/s (677.0, 679.7); MTP acceptance 2.346
              (per position 0.781/0.565, 0.780/0.566) vs BASE 2.345 (0.783/0.568, 0.779/0.561);
              TTFT p99 1157 ms (gate 1609), TPOT p99 42.1 ms (gate 55.3); 128K request pass.
              server.log: "Quantized a fp8 copy of the lm_head (124160 x 5120 per rank)" on both ranks.
              GSM8K not run (see validity).
  conclusion: Interpretation: real gain, every start above BASE's mean, acceptance unchanged. The gain is
              ~3x the napkin: the draft head's bf16 GEMM was probably running below the BW napkin
              (small-M kernel choice), and the fp8 path uses a different kernel. Mechanism profile still to do
              (not yet measured: per-step draft-head kernel time).
validity:     Allowed explicitly by ruling 1 (task_definition.md: FP8 draft lm_head, drafter only).
              The target lm_head, target activations and verification are unchanged; greedy verification
              against the target means the output is the target's, so GSM8K is unaffected by construction
              (skipped to save GPU time; can run if the manager wants it). Memory: +0.63 GB per rank for the
              fp8 copy; KV budget unchanged (5 GiB); 128K request passed.
              Reviewed by leader-1.

review (reviews/T1-1, codex, after filing): numbers, draft-only use of the fp8 copy and ruling 1 confirmed
(the fp8 path is per-row weight scales + dynamic per-token activation FP8 via CUTLASS, drafter only).
Open points and my answers:
- Model snapshot differs from BASE's run (017b9c7 vs c3a462c, both in the shared HF cache). Checked
  (leader-1): all weight blobs and config.json are identical; only chat_template.jinja and tokenizer_config.json
  differ (the newer template wraps assistant turns in {% generation %} markers and adds a response_template;
  the rendered text is the same). Benchmark input tokens are identical (104,526 in both). Every run in this
  run folder uses c3a462c. Interpretation: the BASE comparison holds.
- Mechanism: the reviewer re-queried BASE's profile: the main bf16 head GEMM streams at ~0.94 TB/s, so
  "inefficient small-M bf16 kernel" is NOT established. The cause of the ~3x-napkin gain stays a hypothesis
  (candidates: other head kernels at 0.4-0.8 TB/s, mixed-step savings, tail effects) until matched head timings.
