# Round 3 task for leader-1 (team T1)

## Direction (on top of the champion T1-009). Two parts, in this order:

### R3-A: k=3 that actually holds 16 requests (cheap, do first)
Both brainstormers found it in T1-008's own server.log: k=3 failed on admission, not on compute.
"Running: 14, Waiting: 2, KV usage 95.3%" for the whole run. The GDN spec-state pages share the
5 GiB block pool: 13 blocks per request at k=3 vs 10 at k=2 (KB13).
- Re-run T1-009 + num_speculative_tokens=3 with --kv-cache-memory raised to the smallest value
  that holds 16 requests plus margin: try ~5.75 GiB, then ~6 GiB.
  - The free memory measured at start leaves ~1.5 GiB headroom before activation peaks. Not
    certain.
  - Fallback if memory is short: trim the CUDA-graph capture list (41 sizes, 1.34 GiB/rank).
- Check: the log shows `Running: 16`, Waiting 0. Acceptance as expected (~2.7).
- Memory change: needs the 128K test plus the full benchmark without OOM.
- Expected: +2-5% alone (T1-008's per-request decode was +6% even with 14 running).
- Pre-check (~3 GPU-min): the server starts with the bigger budget and logs enough blocks.

### R3-B: GDN spec-decode state checkpoint + replay (main line)
Today the GDN spec-decode kernel stores the full fp32 state after every one of the k+1 tokens.
Instead:
- keep one committed state plus a small per-request log of the last step's per-token GDN inputs;
- at the next step, replay the accepted tokens, write the one committed state, and run the new
  tokens without storing their states.
- Do the conv state the same way.

Why it pays:
- GDN decode costs ~5.3 ms of a decode step: 3.65 ms in-kernel + ~1.7 ms measured spill into
  the next Marlin GEMMs (KB9, KB14).
- Traffic per layer goes from ~100 MB to ~50 MB at k=2. The gain grows at k=3.
- A second version can shrink num_speculative_blocks for GDN, freeing ~60% of the per-request
  pool use. That makes R3-A fit without extra KV.
- Est. +4.5-6% at k=2.

Details, code pointers and ceiling tests (~1 GPU-min: drop the per-token stores, timing only;
go if >= ~30-40 us saved per layer, drop if < 20 us):
- brainstorm/r3/systematic_engineer.md (Idea 1);
- brainstorm/r3/adjacent_field.md (Idea 3);
- brainstorm/r2/adjacent_field.md (D2, Idea 1).
Design reference: vLLM's ReplaySSM for Mamba2/KDA (config/cache.py:322-335), not enabled for
Qwen3.5.

Main integration risks:
- prefix-cache "align" checkpoints (vllm/v1/worker/mamba_utils.py ~1557-1610);
- preemption;
- chunked prefill continuation.

Correctness (human's rule, assumption_ledger.md): bitwise exact by construction. Show it with a
unit test: states bitwise-equal to today's kernel under random accept patterns. Also GSM8K and
acceptance.

Order: run R3-B's ceiling microbench early (1 GPU-min), in parallel with R3-A. If the ceiling
kills R3-B, put the rest of the round into making R3-A robust.

## Budget
T1: 80 GPU-minutes. Up to 2 workers. Keep ~15 for official evals. If R3-B's kernel is real
progress and needs more, ask me.

## Reference to beat
CHAMPION = T1-009: 730.9 tok/s (TTFT p99 1438 ms, TPOT p99 41.3 ms, GSM8K 93.6%, acceptance
2.344). Gates vs BASE: TTFT p99 <= 1784 ms, TPOT p99 <= 58.1 ms, GSM8K >= 89.6%. A win must beat
730.9 by more than ~3%. A stacked R3-A + R3-B is the likely official candidate.

## Not this round (ideas.md)
Lossless AR compression (R3-C), L2 prefetch (R3-D). Ruled down: Marlin retune, k=4,
reduced-vocab draft head.
