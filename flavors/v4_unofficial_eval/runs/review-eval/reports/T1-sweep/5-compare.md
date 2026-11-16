# T1 sweep, step 5: comparison with other maps and the history

## State of the run when this was written (2026-11-16)
- Bank: only BASE. shared_kb.md empty. journal/announcements: setup, team openings, the promotion
  rule (human, 2026-11-16). No reviews, no other records.
- T2's map: `reports/T2-sweep/` holds 1-inputs.md and 2-graph.md only (no knob sweep yet). **I could
  not read their contents:** the session's permission check denied the read of another team's
  files (twice). I did not work around it. So this comparison is against the history only; a
  map-vs-map comparison with T2 is pending (to do once access is sorted out or via messages after
  my 6-ranking).

## My candidates vs the history
Every candidate in 4-knobs.md is **untried** in this run (no records besides BASE). Nothing in
the history contradicts a verdict.

## What I can say about differences
- Dimensions my sweep added beyond the framework's template, which another map may lack:
  1. the **transport quality of the all-reduce** (protocol/channels/IPC path): the framework prices
     Link_bytes / Link_BW only, while the measured link efficiency here is 22% for 2.66 MB
     messages. The biggest napkin-vs-measured gap in my calibration lives in a parameter the
     framework treats as a hardware constant.
  2. **kernel efficiency per M regime** (Marlin at 46% of peak for M~260 vs 80% BW at M~42): the
     framework's (F, BW) model has no per-kernel efficiency. It is the Implementation layer in
     01 §3, but here it decides a ~9% candidate.
  3. the **vocab index v as a split / reduce site** (argmax-before-gather, reduced-vocab draft head):
     07 §4 item 3 lists "include vocabulary V in the template" as open; it matters here (lm_head ~10%
     of a decode step).
- Framework findings: the ridge point on this GPU is ~82 rows (weight-only FP8, bf16 compute), not
  333, which changes the tree-speculation and DBO verdicts (little free compute slack at M=48;
  the mixed step at M~260 is compute-bound, so DBO is worth considering there).
