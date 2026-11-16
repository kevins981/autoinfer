# T6 sweep, step 5: my map vs the other teams' maps and the history (read after files 1-4)

Sources: reports/T1..T5-sweep*, T*-writeup.md, bank records, shared_kb.md.

| my candidate | status in the run | by whom / on what code | still open? |
|---|---|---|---|
| 1 large-message AR through host-mapped memory | **tested, closed**: bare NCCL at 2.66 MB ~205 us = 0.79 us/row, the same per-row cost as HMAR; NCCL env knobs ~nothing (MIN_NCHANNELS=8 +7% only >= 1024 rows) | T1 update-2, T2 job 2, on T1-007/BASE code (transport unchanged since) | no. **My 4x napkin gap is peer skew/waiting inside the NCCL kernel, not transport** (T1 lesson, T4 5-compare) -> my verdict changed |
| 2 ReplaySSM / DeltaLog for GDN spec states | in progress | T5 (row-0 checkpoint layout, jobs 90-102) | T5 owns it |
| 3 prefill dual-batch overlap | full DBO blocked (V2 ubatching needs DP>1, hybrid unsupported: T3, review T3-1); the **per-site row-split GEMM->AR overlap** is T5's #2, napkin +0.5-2% after review T5-1 | T3 code reading; T5 not started | **yes**: I propose T6 takes it (message to leader-5) |
| 4 k=3 drafts | T1-004 (KV-capped), T1-009 stopped after 1 start -0.9% on T1-007 code (bf16 draft head, eager draft metadata); memory: the fp8 draft-head copy forced KV to 4.4 GiB and k=3 needs ~5.5 GiB (T3-003 OOM at 5 GiB) | T1 | not on today's code, but capacity-blocked until DeltaLog frees slots |
| 5 reduced-vocab draft head | T1-002: acceptance -4.6%, lost to the FP8 head | T1 on BASE family | low (T4: ~1.3% left after FP8 head) |
| 6 mixed-step GPU idle | nsys artifact (T4 writeup: node trace inflates eager launches; unprofiled mixed steps 56-66 ms for ~60 rows) | T4 | closed; my 3-calibration "idle 28 ms" is the same artifact |
| 7 decode GEMM efficiency 76% | T2 tuned tiles (random data); T3 L2 prefetch | T2, T3 | untried beyond tiles; hard |

## What my sweep has that others lack
- N_ridge for W8A8 with fp32 accumulation is ~82 rows, not ~330 (framework example uses the fp16-accumulate rate). It matters for k=3 (M=64 stays BW-bound) and for row-split overlap (a 130-row chunk is still compute-bound, so re-reading weights per chunk is ~free).
- Column-split (N) pipelining of out_proj/down -> AR at decode (weights read once); dropped by me because the L2 prefetch already fills the AR window with DRAM work.

## What others have that I missed
- Peer skew as a cost term: a collective's kernel time includes waiting for the slower rank (T1). The framework prices a collective at link bytes and doesn't model skew; I priced the kernel time and got the wrong verdict for candidate 1.
- CPU edges (scheduler, input prep, eager launches) as graph nodes (T3, T4), and per-iteration unprofiled timings (`--enable-logging-iteration-details`, T4) for calibrating mixed steps.
- AR+norm fusion (T4-002, neutral), recurrent short prefill (T4-001/003, neutral).

## Disagreements
- None strong. T5's review cut the row-split overlap to +0.5-1% assuming the pipeline saves only min(AR1, G2). With the skew counted, part of the ~216 us/call waiting also overlaps the second chunk's GEMM, so I put it at +1-3%. Unproven.
