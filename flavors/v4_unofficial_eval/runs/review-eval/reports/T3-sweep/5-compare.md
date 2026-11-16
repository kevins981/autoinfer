# T3 sweep, step 5: my map against T1's and T2's maps and the run's history

Read: reports/T1-sweep/ (1-6, update-1, update-2), reports/T1-writeup.md, reports/T2-sweep/ (1-6, update-1), reports/T1-1, T2-1,
shared_kb.md, announcements.md, journal.md, bank/T1-001..009, T2-001..007 (records; T2-007 has no record.md yet: eval/1 = 787.6 tok/s,
T1-007 + T2-006's fused activation quant, apparently a pending candidate).

## 1. Status of my candidates
| my id | candidate | status | by whom, on what code | does the result still hold for the champion? |
|---|---|---|---|---|
| K4-a | faster large (mixed-step) all-reduce | **tested, mostly closed**: NCCL env knobs gave nothing at decode sizes, MIN_NCHANNELS=8 ~7% only >=1024 rows (T2 job 2); bare NCCL 2.66 MB = ~205 us = 0.79 us/row, the same per-row cost as hmar (T1 update-2) | T1, T2 on BASE-family code | yes (transport facts). **My calibration was wrong here**: I priced the mixed-step AR at its kernel time (363 us mean over 422-row mean steps). Corrected (review T3-1): size-matched, rank-0 ARs average 218-248 us at 244-276 rows, close to the bare transfer; skew is real (93 us mean start offset) but not ~160 us. Remaining: a full-duplex pipelined copy could bring 205 -> ~110-130 us (link floor) for >=2.66 MB: ~-10 ms per mixed step, ~-2.5% total, but T1's pipelined hmar was slower at small M (flag cost) |
| K4-c | DBO on mixed steps | untried (T1 #4, T2 P6 ranked it; nobody read whether vLLM's ubatching overlaps TP all-reduce) | — | open |
| K7-a | decode GEMM efficiency | partly: T2-004 tuned Triton configs (5 JSONs), T2-006 extended to 31 M keys; still 0.69-0.79 TB/s in my profile of T1-007 | T2 | open beyond tuning (split-K/stream-K for gate_up) |
| K2-c | reduced-vocab draft head | **tested, lost**: T1-002 (K=32768 prefix ids, bf16, on BASE): step -8.5% but acceptance -4.6% -> loses to the fp8 head | T1 on BASE | likely still holds; an fp8 variant written, untested |
| K4-b | column-split GEMM + overlapped AR | untried (T1 #7) | — | open; T1 found hmar's write/read overlap costs ordering (job 42) — a different overlap |
| K5-a | L2 weight prefetch during the AR window | **not in any map** | — | open (new dimension: memory tier L2 as a Place target during link windows) |
| K3-a | k=3 drafts | **tested, neutral**: T1-009 stopped after 1 start, 764.2 vs 771.3 (KV 5.5 GiB) | T1 on T1-007 code | holds; would only change if the draft step gets much cheaper |
| K1-a | GDN DeltaLog | untried (both ranked low: +1.6% T1, +2.2% T2) | — | open; my calibration (kernel faster than its write napkin) says the ceiling is likely lower still: test before building |
| K2-b | fp8 drafter lm_head | **tested**: +6.7% on BASE (T1-001); stacked on T2-003 neutral (+0.75%, T1-003) and needed KV trim to 4.25 GiB (OOM with prefill graphs at 5 GiB; 1 preemption/start) | T1 | holds; memory headroom is binding (<0.6 GiB free) |
| K2-a | fused `in_proj_ba` GEMV (cuBLAS split-K 27 + splitKreduce, 25 us/layer) | **not in any map** | — | open |
| K2-d | AR + residual + norm (+quant) fusion | partly built by T2: T2-006/T2-007 fuse the group quant into add-RMSNorm and SiLU-mul (3 of 4 quants per layer); the AR itself stays separate | T2 on T2-004 / T1-007 | remaining: fold add+RMSNorm(+quant) into the hmar epilogue (~0.4 ms/step) |
| K5-cpu | CPU gaps / async scheduling | async scheduling already on (T1 update-2); GDN chunked-prefill core runs eagerly in mixed steps (T2 update-1, N1, untried) | T1, T2 | open (mixed steps) |
| SendRecv | 454 us ncclSendRecv per decode step | not in any map | — | open (likely batch-sharded sampling's exchange, absorbing rank skew) |

## 2. What my sweep found that the other maps lack
1. **`in_proj_ba` overhead** (calibration A): a 0.5 MB bf16 GEMM costs 25 us/layer (split-K 27 + an 18 us reduce kernel), 1.2 ms per decode step,
   ~3% of decode. Neither map lists the bf16 small GEMMs by site. Missing because both priced GEMMs by weight bytes only; a tiny GEMM's cost is
   its kernel structure, not its bytes. Framework note: the napkin needs a per-launch floor, and should be checked per node, not per category.
2. **HBM is idle during every all-reduce** (6 ms per decode step): the Place knob can target the L2 tier for the *next* GEMM's weights during the
   link window (K5-a). Neither map has L2 as a memory tier; the framework lists HBM / peer / CPU tiers only.
3. **The GDN kernel runs faster than its own write traffic napkin** (writes 3 state slots per request per layer): a calibration flag against
   both maps' DeltaLog estimates (theirs assume the writes are fully paid).
4. **The target and draft lm_heads are already at 0.80-0.93 TB/s**: T2's #4 "lm_head kernel choice (1.59 ms at 400 GB/s -> 0.8 ms)" priced the
   per-GPU head at 0.635 GB; it is 1.27 GB bf16 (248,320/2 x 5120 x 2 B). **I disagree with that verdict**: the kernel-choice gain is <= ~0.3 ms
   (target) and ~0.1 ms (draft), not 0.8 + 1.2 ms. Only fewer bytes (fp8 copy, smaller vocab) can cut it.

## 3. What they have that my sweep missed
1. **Collective time includes waiting for the other rank** (T1 lesson 1). I priced the mixed-step AR at kernel time; the bare transfer is ~205 us,
   not 363 us. My K4-a (-7.3%) shrinks to ~-2.5% at most. Framework finding (agree with T1): price collectives from a bare microbenchmark and
   check whether they sit on the critical path; my calibration should have compared both ranks' AR start times.
2. **Memory headroom is a binding dimension** (T1 lesson 2, T2-002 OOM): the fp8 draft copy, k=3's KV and prefill graphs collide at <0.6 GiB free.
   My knob sweep had capacity only for DP; I should have checked headroom for K2-b and K3-a.
3. **Mixed-step graph capture** (T2-001) and **argmax before gather** (T2-003) were already in the champion, so my sweep correctly started after
   them; I had no row for "what is still eager in mixed steps" (T2 N1: the GDN chunk core under `@eager_break_during_capture`) — it is the likely
   source of the launch skew that inflates the mixed-step AR.
4. T2's S3 (speculate the target verdict to start drafting early) is a Speculate site I did not list; I agree with its low ranking.

## 4. Disagreements
- T2 #4 lm_head kernel choice: see 2.4 (bytes wrong by 2x).
- T1/T2 DeltaLog (+1.6/+2.2%): plausible upper bounds, but my calibration says the kernel's writes are partly hidden; I'd price it below +1.5%
  until a write-skip microbenchmark says otherwise.
- T1 k=3 "neutral": agreed as a screen (1 start); not a measured fact (TASK noise rule), but it needs a cheaper draft step to win.
