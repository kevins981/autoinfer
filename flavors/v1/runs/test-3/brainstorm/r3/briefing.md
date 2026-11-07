# Round 3 briefing (run test-3)

## The problem
Same as before (brainstorm/r1/briefing.md, task/TASK.md): output tok/s at c=16, Qwen3.8-27B-FP8,
2x RTX 4090 without P2P. Rules: task_definition.md (incl. the human's rulings) and
assumption_ledger.md. New since round 2:
- **How correctness is judged:** by the math, by op-level numeric checks and by GSM8K/acceptance.
  Output-text equality counts only for bitwise-exact changes run one request at a time.
- **Accepted environment facts:** GPU 0's clock is locked at 2235 MHz. Another user's nsys agent
  and idle MPS daemons are on this server.

## Facts
- CHAMPION = **T1-009**: 730.9 tok/s official (+15.8% vs BASE 631.4), TTFT p50 204 / p99 1438 ms,
  TPOT p50 20.0 / p99 41.3 ms, GSM8K 93.6%, MTP acceptance 2.344. What it is: champion.md and
  bank/T1-009/ (diff, serve_args.txt, official results).
- Gates (vs BASE): TTFT p99 <= 1784 ms, TPOT p99 <= 58.1 ms, GSM8K >= 89.6%.
- Profiles:
  - **No profile of T1-009 exists yet.** Make one if you need it (task/scripts/profile_nsys.sh on
    `ws create <you>.T1-009 --from T1-009`; remember its serve_args.txt). Save small outputs to
    bank/T1-009/profiles/; keep big .sqlite/.nsys-rep files in your workspace and point to them.
  - Older ones: BASE in task/baseline/; T1-005 (= T1-009 minus the logits diet) in
    workspaces/leader-1.hostar2/out/nsysB/.
- Confirmed insights: shared_kb.md, KB1-KB12. For example: the decode AR is near its PCIe floor;
  GPU 0 is ~15% slower on compute; FP8 draft head costs no acceptance; 3 draft tokens hurt TTFT.

## What was tried
- Round 1 (reports/r1.md): host-staged AR/AG + prefill graphs, +10.7%.
- Round 2 (reports/r2.md): logits diet, +4.5% more.
- What was tried: tree.md. Scored but untried ideas: ideas.md. Untried so far:
  - GDN spec-state checkpoint + replay (R2-B);
  - L2 prefetch during the AR (R2-C);
  - Marlin M=48 retune (R2-D);
  - copy-engine AR for large M (R2-E);
  - reduced-vocab draft head.
- Undiagnosed: why 3 draft tokens blew up TTFT (T1-008, KB11).
- leader-1 (idle) ran both rounds: ask them.

## Where things are
- Engine code: /scratch/kevinsong/learning/autoinfer/.cache/tasks/qwen38-27b-fp8-2x4090/baseline/vllm
- Champion code: workspace leader-1.greedyverify, or `ws create ... --from T1-009`.
- Helper scripts: task/scripts/.
