# Failure cases (raw material for a taxonomy of agent ineffectiveness)

One entry per observed case where agents missed, dropped or under-realized something worth having.
Concrete and cited; no categories fixed yet (EVAL.md: the taxonomy comes from enough cases).
Issue types seen so far (human, 2026-11-15): **measurement methodology** (the experiment itself was
flawed: noise, wrong setup, too little data) and **interpretation** (the numbers were fine but read
wrongly: wrong baseline, wrong floor, irrelevant comparison). A case can have both.
Possible axes so far (human, 2026-11-15), to test against cases, not to force them:
never thought of the direction; investigated but stopped too early; ran the experiment wrong;
collected too little data; misread the number.

## 2026-11-15, v4_unofficial_eval/sweep-simple-eval: host-memory all-reduce dropped on a proxy
- What: T1-003 (exact, working pinned-host all-reduce) was parked untested in the inference server.
  The same change gave +3.4% then +5.1% in sweep-unofficial-eval (T1-002/T1-005), and was part of
  framework-eval's first +11.3%.
- Why (several at once):
  - Proxy as verdict: a standalone microbenchmark decided it; the in-server test was one
    unofficial_eval (~5 GPU-min) against a napkin of +3-4%, with 40 GPU-min left.
  - Wrong baseline: compared with NCCL Simple (57 us), though the team had just shown the engine
    runs NCCL's default LL (~64-81 us); against LL even its worst-but-one reading wins.
  - Flawed instrument, known and ignored: rank-0-only timing without per-sample sync, 52 vs 92 us at
    the same size; the record itself says "rank skew ... undiagnosed".
  - Irrelevant loss counted: "loses at M>=96", though the design routes only small all-reduces.
  - No second look: no sparring session before dropping it (the prompt asks for one).
- Types: methodology (rank-0-only timing, no per-sample sync: 52 vs 92 us for the same kernel) AND
  interpretation (called NCCL Simple's 57 us "the floor" though its own 13 GB/s figure gives ~40 us;
  compared with standalone Simple although its own profile showed in-engine NCCL at ~72 us/AR =
  9.66 ms per decode step / ~135 calls, reports/T1-sweep/update-2.md:8,16,20; vs 72 us even 52 us
  saves ~2.7 ms of a ~39 ms step, ~7%).
- Evidence: bank/T1-003/record.md; workspaces/leader-1.hostar/bench/test_hostar.py:34-38,
  job_output.txt, job2_output.txt; agents/leader-2.w1/allreduce_results.md; reports/T1-sweep/update-1.md.
- Idea it prompted (human): a fresh-context verification worker that a leader can call before
  concluding or claiming something (like a code-review agent). To discuss.

## 2026-11-15, sweep-simple-eval: FP8 draft head dropped for memory; fix stayed local (path dependence)
- What: T2-002 (FP8 draft head, +3% in 3 other runs) lost: with 2048-token prefill graphs (T2-001) and
  a 4 GiB KV only 15 of 16 requests run (KV quantized by the 1,600-token hybrid page). Diagnosis right
  (T2-003 profile). The fix tried smaller heads (T2-003, T2-004 bf16 65K subset), never revisiting the
  2048 graph cap that held the memory (sue used 512 and kept the FP8 head).
- Type: neither methodology nor interpretation: a search/decision issue (didn't revisit an earlier
  choice once it became the constraint).
- Evidence: bank/T2-001..T2-004/record.md.

## 2026-11-15, sweep-simple-eval: attention host planning priced as kernel time
- What: T1's knob table put attention at "1% of a step, nothing to gain" (reports/T1-sweep/4-knobs.md:10).
  The cost is FlashInfer's per-step host planning (GPU idle ~1-1.4 ms/step), worth +3.8-6.8% in other
  runs via TRITON_ATTN. T1 later learned to price GPU idle from a graph-level trace and found the
  drafter's half (T1-008, +2.3%), but called the target side "exact-only by design" and left it.
- Type: interpretation (priced the wrong cost: kernel time instead of the idle it causes).

## 2026-11-15, sweep-simple-eval: noise promoted as a win
- What: T1-002 (NCCL_PROTO=Simple) promoted at +1.0% unofficial, inside the ~1.6-point repeat spread;
  reverted later (no in-engine effect). The first revert reason (kernel names) was itself wrong.
- Type: interpretation (noise threshold) + wrong mechanism evidence.
