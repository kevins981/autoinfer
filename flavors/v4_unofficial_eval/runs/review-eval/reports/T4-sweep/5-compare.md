# T4 sweep, step 5: my map against T1/T2/T3 and the history

Read: reports/T1-sweep/*, T2-sweep/*, T3-sweep/* (incl. updates), T1/T2 write-ups, resync reports,
shared_kb.md, bank records T1-001..T3-005 (record.md). Champion now T3-003 (826.1 = T3-001 + FP8 draft
lm_head, KV 4.4 GiB).

## 1. Status of my candidates
| my site | status | by whom, on what code | does the result still hold? |
|---|---|---|---|
| K2-a FP8 draft lm_head | **built**: T1-001 (+6.7% on BASE), T1-003 neutral on T2-003 (KV 4.25, preemptions), **T3-003 champion** (+1.2% on T3-001) | T1, T3 | yes, it is the champion |
| K2-b reduced-vocab draft head | tested: T1-002, step cheaper but acceptance -4.6%, loses to FP8 head | T1 on BASE-family | yes, and on top of the FP8 head the remaining saving (~0.7 ms/step, ~1.3%) is smaller than a 4.6% acceptance loss: **dropped** |
| K3-a 3 drafts | tested: T1-004 (confounded by small KV), T1-009 stopped after 1 start at 764.2 vs 771.3 (T1-007 code, bf16 draft head, 5.5 GiB KV) | T1 on T1-007 code | **not settled on today's code**: the draft head is now FP8 (3rd draft step ~0.8 ms instead of ~1.4), but memory headroom is tighter (KV already cut to 4.4 GiB for the FP8 copy); 1 start is a screen, not a fact |
| K45-c faster AR for mixed steps | NCCL env knobs: nothing at decode sizes, MIN_NCHANNELS=8 ~7% only >= 1024 rows (T2); bare NCCL 2.66 MB ~205 us = same per-row cost as hmar (T1 update-2); T1 lesson: mixed-step AR kernel time is mostly peer skew | T1, T2 on BASE/T1-007 code | mostly yes: my K45-c napkin used the 293 us kernel time, which includes skew; the transfer itself is ~205 us at 2.66 MB. **Re-priced: -10 ms/mixed step at best (+2.7%), only if skew is removed too** |
| K45-d pipelined hmar (decode) | tested: T1 "pipelined: slower; geometry sweep within 1 us; ordered overlap slower; relaxed order fast but wrong" | T1 (workspaces t1-car5) | yes: **dropped** (I don't have a better design than theirs) |
| K6-b DBO in mixed steps | blocked: V2 ubatching needs DP>1, hybrid unsupported, mamba asserts ubatch 0, HMAR off under ubatching (T3, review T3-1) | T3, code reading | yes: **dropped** (building ubatching for hybrid TP is too large for the time left) |
| K1-a GDN DeltaLog | open: T3 confirmed all 3 slots written; cold microbench 120/89/60 us for 3/2/1 writes vs 70-73 us in the real server; "needs a dram-bytes check" | T3 (on T3-001 code) | open; my napkin (-1.4 ms/step, +2.6%) assumed 72.7 -> ~42 us; T3's cold numbers suggest 3->1 writes saves ~50% of the cold time, but the in-server kernel is already 40% faster than cold, so the real saving is uncertain (could be ~0.6-1.4 ms) |
| K6-c mixed-step fixed overhead | partly: T2-001 (prefill graphs to 2048, +8.4%) removed the eager launch-bound part; T2 noted the GDN chunked-prefill core still runs eagerly (`eager_break_during_capture`, untried, "<=2%"); T3 ranked "mixed-step skew" <=2% | T2, T3 | **my measurement disagrees with "<=2%"**: see §2 |
| K6-a chunked prefill into decode steps (smaller token budget) | T1 "prefill chunk size: no napkin gain, drop"; T3 "unpriced" | — | **untried**; T1's drop assumed mixed-step cost scales with FLOPs; it doesn't (§2) |
| K6-f L2 prefetch | built (T3-001); per-target tuning and mixed-step prefetch on T3's list | T3 | T3's lane, not mine |
| K2-e greedy fast path for target logits exchange | T3-002 +0.9% screen, unresolved | T3 | T3's lane |
| K45-a PP, K45-b DP, K3-c trees | pruned by all teams for the same reasons | — | agree |

## 2. What my sweep found that the other maps lack
1. **Mixed-step cost vs prompt length is flat at the low end.** In the champion's graph trace a mixed step
   with 57-91 prompt rows costs 72-122 ms (bimodal ~78 / ~118 ms), the same as one with ~320-400 rows
   (106-122 ms); slope ~0.22 ms per extra row above ~300 (prof_graph/steps.csv, 49 steps). A pure decode
   step is 32.7 ms. So a mixed step carries a **fixed excess of ~40-85 ms independent of P**, on top of
   the compute that the prompt really needs. Over ~300 mixed steps that is ~12-25 s of a 112 s run
   (10-20%). All three maps priced mixed steps as "compute-bound + AR"; T2 and T3 saw "~7.5-10 ms GPU idle
   per mixed step" but nobody plotted cost vs P. The idle alone doesn't explain it: busy time is also
   50-100 ms in small-P steps. Cause under analysis (agents/leader-4.w1/mixed.md).
2. **Chunked prefill as a Window/merge choice (06 §2.1) is not in any map as a priced row.** It wins only if
   a step with ~50 prompt rows can cost close to a decode step (~37 ms). Today it can't (finding 1), so it
   is conditional on fixing the fixed excess. With the fix, it is the largest single lever in my map
   (~+14% napkin), because prompt rows would ride decode steps' unused compute (M = 48 is below the ridge).
3. **A per-P breakdown is the missing dimension**: the framework's window cost treats a step as
   FLOPs/bytes/link; the measured step has a per-step fixed term that depends on the step's kind (pure
   decode FULL graph vs mixed piecewise + eager GDN/attention + metadata). T2's lesson 1 (add a
   launch/latency term) says the same for kernels; here it is per step kind and much bigger.

## 3. What they have that my sweep missed
- **Collectives include peer wait** (T1 lesson 1). My K45-c used kernel time; corrected above.
- **Memory headroom as a binding dimension** (T1 lesson 2; T3-003 had to trim KV to fit the FP8 head copy).
  My K3-a napkin ignored it: k=3 needs more KV per running request and bigger graphs. Added as its
  riskiest assumption.
- **L2 as a memory tier** (T3): the prefetch under latency-bound ARs. My map had it only as K6-f.
- **The logits all-to-all (SendRecv ~0.45 ms)**: I listed it as bookkeeping; T3 priced and built it (T3-002).
- **Engine options first** (T2 lesson 2): several wins were existing flags. For K6-a the knob is an
  existing flag (`--max-num-batched-tokens`, `long_prefill_token_threshold`), cheap to test once the
  fixed cost is understood.

## 4. Where I disagree
- T1's "prefill chunk size: no napkin gain, drop": the drop assumes step cost ~ FLOPs; measured cost is
  dominated by a per-mixed-step fixed term, so chunk size and the *number* of mixed steps matter
  (either fewer, larger mixed steps or cheaper small ones).
- T3's "mixed-step skew <= 2%": measured excess is ~40-85 ms per mixed step, not ~7 ms. Even if part of
  it is real work I haven't identified, it is the largest unexplained block in the run.
- T1's k=3 "neutral": one start on older code with a bf16 draft head; on today's code it is untested.
