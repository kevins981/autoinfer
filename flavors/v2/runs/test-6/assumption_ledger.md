# Assumption ledger (run test-5, resumed from test-3)
## Frozen
Model + FP8 weights; FP8 KV cache; bf16 activations, or the checkpoint's own W8A8 scheme (dynamic
per-token-group-128 FP8 activations; allowed by the human 2026-11-09, see task_definition.md); baseline-precision
communication; sampling as requested; benchmark/harness/gates; 2x RTX 4090; reasoning + tool parsers
on; max-num-seqs >= 16; exact spec decode only; one 128K request must still fit.
## Open
Engine code (kernels, scheduler, memory, comms, compilation); parallelism scheme (TP/PP/hybrid);
serving flags via serve_args.txt; spec-decode method/draft count (exact only); max context and KV
budget (within the 128K rule).
## Assumptions
- Latency gates are relative to this run's BASE official numbers.
- Noise: ~0.5% CV within one server, up to ~1.6% across restarts and jobs (task/TASK.md). No minimum
  gain: small gains count when same-job A/Bs show them consistently (the human, at the resume, replaced
  test-3's "< 3% treated as noise").
- Config: 1 leader, up to 2 workers, 15 rounds (1-3 in test-3; raised from 8 by the human during round 5), 80 GPU-min per team per round,
  brainstormers systematic_engineer + adjacent_field.
## Accepted environment facts (human, round 2)
- GPU 0's SM clock is locked at 2235 MHz (GPU 1 boosts to ~2400-2670 MHz): a fixed fact of this
  machine. No planned change. GPU 0 is ~15% slower on compute-bound work (KB8).
- Another user's long-running nsys agent and their idle MPS control daemons are on this server
  (why profiler counters work only on GPU 0). Neither counts as "someone else using the GPUs".
## How correctness is judged (human, round 2; TASK.md's Correctness section says the same)
Identical output text is NOT a correctness test: even at temperature 0, results depend on batch
composition and kernel choice, so the same code at c=16 can produce different text run to run, and
a valid fp reordering can flip a near-tie. Judge correctness by:
- the math: exact by construction, or a bounded-error reordering;
- numeric checks of the changed op against the old op on the same inputs;
- GSM8K and MTP acceptance.
Token-for-token output comparison is meaningful only for bitwise-exact changes, run one request
at a time (no batching effects).
