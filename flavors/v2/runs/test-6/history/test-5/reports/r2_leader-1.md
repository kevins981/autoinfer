# Round 2 report from leader-1 (T1), as received (condensed)
Best: bank/T1-009, official 730.9 tok/s (+4.5% vs T1-005, +15.8% vs BASE; 2 trials CV 0.2%), GSM8K 93.6%, TTFT p99 1438 ms, TPOT p99 41.3 ms, acceptance 2.344, 128K pass.
T1-009 = T1-005 code + greedy target-verification fast path (code); serve_args: prefill-graph flag + spec config {mtp, 2, use_local_argmax_reduction, draft_lm_head_quantization=fp8}.
Experiments: T1-006 local-argmax drafts (unofficial +1.1%); T1-007 + FP8 draft head (unofficial +3.0% same-job, acceptance unchanged, +0.59 GiB/rank, 128K pass); T1-008 3 draft tokens (unofficial -6.9%: acceptance 2.715 but TTFT p50 210->792 ms, p99 2133 ms fails gate; cause undiagnosed; dropped); T1-009 + greedy verification fast path (unofficial +2.4% same-job; official 730.9).
Correctness of T1-009 (per human ruling r2): exact by construction (same pick incl. tie-break); in-server dual path on 25,511 real rows (467 ties) 0 mismatches; 4,764 adversarial unit checks 0 mismatches; GSM8K/acceptance unchanged. Eligibility from per-request sampling params only.
Left open: TTFT blow-up with k=3 (no profile; k=3 worth ~+6% on decode if fixable); reduced-vocab draft head (ruling 2) untried; cleanup: custom-logits-processor check is len(...)==3.
Notes: w3 lost 2.2 GPU-min to an env error; w4 ran ~5 min over its allowance (approved). Budget 64.9/80 used.
