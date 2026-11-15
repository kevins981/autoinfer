team:         T2
builds_on:    T1-005
performance:  server start failed: CUDA OOM (unofficial #1)
summary:
  hypothesis: the BASE server log says "FLASHINFER cannot update draft decode metadata in place; falling back to rebuilding attention metadata eagerly": the 2 MTP draft steps rebuild attention metadata on the CPU between graph replays, leaving ~1 ms of GPU idle per decode step (graph-level profile, GPU job 2: 0.99 ms idle/step). TRITON_ATTN supports in-place draft metadata update (supports_draft_decode_metadata_update=True), so using it for the drafter only should let the draft refresh be recorded in the draft CUDA graph. Napkin -0.5..1 ms/step, ~1-2%.
  change:     serve args: HEAD's speculative config + "attention_backend":"TRITON_ATTN" (drafter's attention backend only; target stays FlashInfer).
  results:    Observed: server start OOM on both GPUs during startup with the drafter on TRITON_ATTN (target FLASHINFER): 23.5 GiB used (unofficial/1/b/trial1/server.log).
  conclusion: Interpretation: a second attention backend for the drafter needs extra memory (separate metadata/workspace, possibly KV layout) that doesn't fit next to HEAD's 5 GiB KV budget, FP8 draft head copy and prefill graphs. Not pursued (would need a smaller KV budget, tested against the 128K gate).
validity:     Draft side only: verification against the target is unchanged (exact); the drafter's attention computes the same softmax attention with a different kernel (fp reordering inside the draft; can only change acceptance). Memory: drafter metadata buffers only -> run with --long anyway (graph changes).

unofficial #1 (2026-11-15T06:58): FAILED at server_start (trial 1) (unofficial/1/)
