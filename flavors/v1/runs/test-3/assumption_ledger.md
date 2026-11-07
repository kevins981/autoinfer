# Assumption ledger (run test-3)
## Frozen
Model + FP8 weights; FP8 KV cache; bf16 activations (no FP8 activations / W8A8); baseline-precision
communication; sampling as requested; benchmark/harness/gates; 2x RTX 4090; reasoning + tool parsers
on; max-num-seqs >= 16; exact spec decode only; one 128K request must still fit.
## Open
Engine code (kernels, scheduler, memory, comms, compilation); parallelism scheme (TP/PP/hybrid);
serving flags via serve_args.txt; spec-decode method/draft count (exact only); max context and KV
budget (within the 128K rule).
## Assumptions
- Latency gates are relative to this run's BASE official numbers.
- Noise ~0.5% CV; < 3% treated as noise.
- Config defaults: 1 leader, up to 2 workers, 5 rounds, 80 GPU-min per team per round,
  brainstormers systematic_engineer + adjacent_field.
