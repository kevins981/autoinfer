# T1 sweep, step 5: my map against T2's map and the run's history

History at the time of writing (2026-11-15, run start): bank has only BASE (official 635.9, unofficial
634.5 / 643.4 tok/s, 200 prompts, bank/BASE/unofficial/1). No shared_kb entries, no other records,
no sparring logs. So every candidate below is **untried** except where T2 has a job in flight.
T2's map read: reports/T2-sweep/1-4.

## Status of my candidates
| T1 candidate | T2 equivalent | status |
|---|---|---|
| I-AR faster all-reduce transport | W4 | untried; both teams have a worker microbenchmark in flight (duplicate: coordinate) |
| R-ARGMAX distributed argmax | R1 | untried |
| S-K3 MTP k=3 | S1 | untried |
| R-DLM8 FP8 draft lm_head | R4 | untried |
| reduced-vocab draft head | R5 (vocab = first N token ids, data-free) | untried |
| R-W8A8 prefill GEMMs | R6 | untried |
| I-MARLIN decode efficiency | R8 | untried |
| K-B GDN rollback-free writes | K1 | untried |
| W-DBO for mixed steps (AR of half A || GEMM of half B) | not in T2's map (T2 drops DBO only at N=48) | untried |
| P-PP conditional | P2 (dead) | untried |
| I-COMPILE / bigger graphs for mixed steps | **W1 (top), testing now: t2-cg2048** | in flight by T2 |

## What T2 has that my sweep missed (a finding about my use of the framework)
1. **Host as a resource** (T2 W1/S5). I saw "GPU idle 6 ms" and "eager small kernels" in mixed steps
   and filed them as ~1% (I-COMPILE). T2 linked eager launching to **cross-rank skew inflating the
   all-reduces**. I checked it on the profile: pairing the 74,385 all-reduce kernels of GPU 0 and
   GPU 1 by order (ends agree within 3 us median), the large (prefill) all-reduces take **299 us on
   the faster rank vs 691 us on the slower** (median). So ~390 us of each prefill all-reduce on one
   rank is waiting for the other rank, not transfer: about half of the 62.5 ms. My 4-knobs priced
   that whole time as link inefficiency. Corrected split: mixed-step AR = ~40 ms transfer (still
   ~2.6x the 15 ms napkin) + ~20-25 ms skew. Decode ARs: 57 us fast rank vs 63 us slow: skew is
   small there, transfer dominates.
   Framework note: the window cost formula has no term for a combine node waiting on its slowest
   input's *host* schedule; T2's `Host` resource fills that gap.
2. **W3: AR || its producing GEMM, split along the GEMM's output columns** (no weight re-read, so
   it survives the Shared-Read rule at N=48 where my DBO does not). I only considered splitting r.
   Split d (the AR payload index) is the right index: d is not reduced in the AR's producer.
3. R5's data-free vocabulary subset (first N ids by BPE order) is a cleaner way to satisfy the
   "no benchmark data" ruling than token frequencies.
4. S5 async scheduling / host gaps (1.3 ms/decode step): I listed it as low priority; equal.

## What my sweep has that T2's lacks
1. **W-DBO in mixed steps**: in a ~260-row step the GEMMs are compute-bound (N >> ridge 82), so the
   Shared-Read penalty of splitting r is ~0 and AR(half A) || GEMM(half B) hides the transfer part
   of the ARs (~40 ms, or ~15 ms after a faster transport). T2 prunes DBO with the decode argument
   only; the dominance proof breaks exactly when N > ridge (framework 06 §3 says this).
2. The quantified skew/transfer split above (gives W1 a ceiling: graphs remove the ~20-25 ms skew
   part and the ~6+ ms launch idle, not the ~40 ms transfer part).
3. Explicit lm_head/vocab accounting per decode step (4.3 ms, 11%) -- T2 has the same numbers in
   R1/R4/R5.

## Where I disagree with T2
- **W1 napkin (fixed excess 48 -> ~5 ms per prefill step)** is too optimistic: graphs remove skew
  and idle (~30 ms), but ~25 ms of extra *transfer* time in mixed-step ARs (40 ms measured on the
  faster rank vs 15 ms napkin) stays. My estimate for W1: -25..-30 ms per mixed step = ~6% overall,
  not 12%. Also prefill steps with breakable graphs still break at attention/GDN prefill kernels,
  so some launches remain.
- **P2 / PP "dead"**: agreed once mixed steps are fixed; at BASE costs it was a net napkin win
  (~+7%) because mixed steps lose all their ARs. Not worth building now (MTP + PP support risk), so
  I drop it too, but with the condition written down.
- **GDN K1 napkin**: T2 assumes 3 state writes per layer; their measured 3.66 ms vs my 2.95 ms
  for the same profile (different medians / kernel sets). Either way the gain is <=1.2 ms; low.
- S1 (k=3) ordering: T2 orders it after R4/R5. I'd test it first: it is a pure serve flag and its
  acceptance number (pos 3) is the unknown that decides both.
