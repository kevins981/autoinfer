# Assumption ledger (run test-4). Agreed with the human (via proxy), 2026-11-07.
## Frozen
Model + FP8 weights; FP8 KV cache; bf16 activations; baseline-precision communication (lossless
compression OK); sampling as requested; benchmark/harness/gates; 2x RTX 4090; reasoning + tool
parsers on; max-num-seqs >= 16; exact spec decode only; one 128K request must still fit.
## Open
Engine code (kernels, scheduler, memory, comms, compilation); parallelism (TP/PP/hybrid); serving
flags via serve_args.txt; drafter (method, draft count, FP8 / reduced-vocab head from non-benchmark
text); max context and KV budget (within the 128K rule).
## Assumptions
- Latency gates: fixed numbers from BASE's shipped official result +20% (TTFT p99 1609, TPOT p99 55.3 ms).
- Runs are independent: nothing from other runs' folders is used.
- Process: no blocking human pauses. The manager decides brainstorm selection and round ends,
  and sends a short summary to the proxy qwen38-opt-12 at each round end.
- Config: 2 leaders x 1 worker, 2 rounds, 55 GPU-min per team per round; brainstormers
  systematic_engineer + adjacent_field.
