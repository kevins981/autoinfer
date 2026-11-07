# Round 1 report, leader-2 (T2, R1-B head diet) — received 2026-11-07
Best: T2-003 (flags only: MTP use_local_argmax_reduction + FP8 W8A8 draft lm_head + --enable-batch-sharded-sampling),
OFFICIAL #1 653.9 tok/s (652.1/655.6), +2.8% vs BASE; GSM8K 93.2%, acceptance 2.348, TTFT p99 1341, TPOT p99 48.2, 128K pass.
Records: T2-001 (drafter flags, unofficial +2.4% same-job), T2-002 (depth 3: acceptance 2.757 but -3.9% tok/s, TTFT p99 2225 fail),
T2-003 (above), T2-004 (custom greedy vocab-parallel argmax: exact 0/9600 mismatches, no gain over T2-003).
Proposed KB: drafter flags keep acceptance; batch-sharded sampling exact with MTP on V2; custom greedy argmax adds nothing
on top; depth 3 costs ~+21% step for +16.5% tokens; short-run p99 TPOT swings 42-57 ms.
Left open: why depth 3 costs +21% (est +5-6%) and its TTFT p99 jump.
Next: profile the depth-3 verify step and remove what makes it expensive; start from T2-003.
Issue: worker edited code during a job (arm C of job 1); re-measured cleanly in job 2.
GPU: 29.6/55 used.
