# T5 write-up (leader-5, explore). Recommending a voluntary close at resync 1

GPU: ~41 of 100 (20 seed + 80 grant) used. Records: bank/T5-001..T5-006. Worker: leader-5.w1 (WRAPUP.md in its folder).
Champion reference throughout: T1-042, 942.2 tok/s.

## Direction: deeper exact speculation on the built stack (proposals/T5-1.md)
- Premise: everything else is near a roof (T2, T4/KB62). So the lever is more accepted tokens per weight pass.
- T3 had dropped deeper speculation over two problems:
  - a "cliff above M=48", measured unbuilt;
  - GDN state memory.
- T5 re-measured both on a built engine, then built confidence-budgeted verification.
- Payoff if it works was forecast at +7%. The measured ceiling is ~+1-2%.

## What was measured
- **The M>48 cliff is real, and it is all Marlin.** Three m-blocks become four at M=49, a flat step up to M=64.
  - Cold: +2.1 ms per forward (T5-002).
  - In situ: +3.0-3.4 ms. GDN, attention and norms at 4 tokens per request add only ~0.2 ms; AR adds +0.6 (T5-003, T5-006).
  - Tile configs recover 0.76 ms cold but only 0.3-0.4 ms in situ (T5-002/004). Not bitwise: an ulp-level K-reorder.
  - T6-004's per-shape policy also recovers only 0.30 ms in situ at M=64 (T5-006). T6 fixed M=72-96, not 49-64.
- **Fixed k=3 loses:** -1% (T5-001) and -3.5% (T5-004) vs the champion, same job.
  - Step +17-19% for E +15-16% (2.33 -> 2.69-2.73).
  - The "-4% graph-cap penalty" in T5-001 was single-run noise: cap 1536 = 1.004 (T5-004).
- **The drafter is at its byte roof:** 0.8-0.9 ms per draft step (T5-003).
  - Draft lm_head ~310 us; MTP Marlin 200-260; fc ~80; AR 110-180.
  - There are no launch gaps.
- **Two-slot GDN state (route B) is bitwise-exact** (T5-005).
  - Layout: S[0] committed + S[1] log per group, for any k.
  - Tested across 1600/3200-token boundaries and with a prefix-cache hit.
  - k=3 then fits 16 requests at KV 5.0 GiB with the full prefill-graph grid. The 128K request passes.
- **Budgeted verification works end to end, but gains little** (T5-006).
  - Pieces built:
    - an in-graph draft confidence (fused argmax + sumexp over the sharded draft vocab; 4 us per call, cheaper than the old path);
    - upstream's adaptive_verification with a fixed budget, enabled for MTP + local argmax + two-slot GDN + TRITON_ATTN.
  - Exactness: a per-row check found 0 mismatches over 1853 ragged rows on the generic sampler path. The greedy-ids ragged path is untested.
  - Results for K_MAX=3, M=48:
    - E rose +5.6% (2.357 -> 2.489); the offline sim had said +8.1%.
    - The step got +3.8% longer (the 3rd draft step is +0.80 ms).
    - The timer model predicts +1.7%. A single run measured -4.8%.
    - Mixed steps run slower on the adaptive path (mean ITL +9.5% vs median +3.6%; TTFT +17%). This is not isolated; see the worker's WRAPUP.md for code-read suspects.

## Forecast vs outcome
| part | forecast | measured |
|---|---|---|
| k=3 vs k=2, equal memory | +3.1% | +3.1% (T5-001), then -3.5% vs champion with override (T5-004): noise ~+-1.5% per pair |
| memory without the graph cap | +0..1% | route B is exact; the cap penalty was ~0 anyway |
| tile override in situ | +1.5-2% | -0.4 ms fwd, net negative together with k=3 |
| budgeted K_MAX=4 at M=64 | +1-2% (on top) | dead: the in-situ cliff stays +3.4 ms |
| budgeted K_MAX=3 at M=48 | (sim +3.2-4%) | timer model +1.7%, single run -4.8% |

My main error: I estimated extra verify rows at ~0.06 ms/row from per-row arithmetic. I didn't check the m-block step. The data to see it was already in T3-008's built timers (M=48 22.40 ms with nothing beyond 48).

## Reusable
- **Two-slot GDN** (ws t5-rb, VLLM_GDN_TWO_SLOT): bitwise; cuts GDN blocks per request for k>=3 (13 -> 7 inferred; the pool-token gain is smaller). Prerequisite for any K>=3 work.
- **In-graph draft confidence:** greedy_argmax_with_sumexp, v1/worker/gpu/sample/greedy_argmax.py in t5-rb.
- **Adaptive-verify plumbing for MTP:** AUTOINFER_BUDGET_PER_REQ, test-only AUTOINFER_BUDGET_FORCED, budget log.
- **In-situ step costs by M at 16 requests** (T5-003/006): k=2 25.25 ms; budget M=48 26.21; k=3 M=64 29.5-29.8.
- **Pitfall:** a client sending logprobs, min_tokens or ignore_eos disables the greedy-ids fast path. Use max_tokens-only clients when testing it.

## Opinion
- On this stack, tokens per weight pass is capped by the drafter's quality, which KB60 says fine-tuning can't move. More draft depth costs a full byte-roof draft step (0.8 ms) per +4-6% E, and rows above 48 cost a Marlin m-block step.
- A bar-clearing speculation gain would need two things at once:
  - a single Marlin pass at M<=64 with no step from 3 to 4 m-blocks (a kernel, not a config);
  - a drafter producing several tokens per weight pass (a trained parallel-draft head, which is data-limited per KB60).
