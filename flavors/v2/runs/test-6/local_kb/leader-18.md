# T18 write-up (leader-18, explore). Recommending a voluntary close at the seed

GPU: 0.0 of 20 used. No records, no worker. Notes: local_kb/leader-18.md.

## What I did
I read every write-up (T2-T16), KB1-KB74, the champion diff (hostar kernel) and T14-004's official artifacts, and costed each candidate from first principles. I kept T17's direction (decode GEMM at M<=64 + k=3) out of scope.

## New observation (CPU only, from bank/T14-004/official/2/trial1: metrics_after.txt, bench_c16.json)
Request lifecycle at c=16 on the champion:
- Server TTFT: mean 204 ms (82.4 s / 403 requests). It splits as:
  - API-server preprocessing: ~17 ms (204 - 24 - 163);
  - engine queue (arrival to scheduled): 24 ms;
  - "prefill" phase (scheduled to first token): 163 ms, against a mean mixed step of ~95 ms.
- Client TTFT: mean 219 ms, so ~15 ms more on the client/HTTP side. Only 1 of 403 requests has TTFT < 80 ms.
- Slot accounting: sum of E2E latencies 1486.7 s vs 16 x 95.27 = 1524.3 slot-s. That gives 2.5% idle outside requests (ramp/drain), and TTFT is 5.7% of slot time.
- Mechanism (code read: vllm/v1/engine/core.py step_with_batch_queue, batch queue of 2):
  - when a request finishes in step N, step N+1 is already running;
  - the engine schedules N+2 as soon as it has processed N's outputs, before the replacement request can arrive (~20-35 ms of API + client round trip);
  - so the new prefill lands in N+3. The freed slot is dead for ~2 steps plus the prefill step itself.
- Interpretation: at most about one step (~30 ms) plus part of the ~32 ms API/client path is recoverable. Two possible fixes, neither built:
  - late scheduling of N+2 (just before N+1 ends);
  - a faster API path.
  - Ceiling about 400 x ~40-60 ms / 16 slots ≈ 1.0-1.6% of the trial. That is exploit-sized, and the late-scheduling margin is tight, because a decode step (~25 ms) is shorter than the arrival latency.

## Candidates costed and rejected (none has an honest payoff >= +5%)
| candidate | why not |
|---|---|
| Pipelined (duplex) SM hostar | already in the champion: writer and reader CTAs are separate, with K chunks and per-chunk flags (host_staged_ar.cu). It sits at the T15 duplex floor (25 us) + fence + chunk tail. |
| LL-style flag-in-data AR (no fence) | a 16B tag costs 33% bytes, which is a wash. 64B-line atomicity over PCIe isn't guaranteed, so it is a correctness risk. Ceiling ~+3%. |
| Pre-armed copy-engine AR (wait-value chain) | T9-006 and T16-001 already used stream wait/write-value. Only the first hop would change, worth ~5 us. Dead per KB74. |
| PP=2 + deeper k | redone with current numbers: stage ~16.7 ms at M=40 (k=4, 8 req), 16 x ~2.95 tokens / 33.4 ms = 1.41 tok/ms, vs TP 1.52. Worse. |
| DP mixers / replicated out_proj / hidden-sharded residual | the same bytes or more (T4, T11). RS+AG gives the same bytes as AR with more latency. |
| Exact INT8 IMMA slicing | 3 weight pieces x 3 activation pieces at 4x rate = 2.25x slower (agrees with T2, T11). |
| Request lifecycle (above) | ~1-1.6%. |
| GPU1 compute rebalance in mixed steps | ideal -7% of mixed-step Marlin, about +1.8%. It needs ~1.1-2 GB of duplicated MLP on GPU1. It could be funded by the ~1.3 GB/rank of KV that T5-005's two-slot GDN frees. Not done; exploit-sized. |
| Small-mixed-step fast path (full graph + fused short GDN prefill) | Marlin and AR are at their floors per row (T6-007's +9.1 ms for 48 rows = 0.19 ms/row = roof), so it buys ~6 ms/step x 215 ≈ 1.4%. |
| Prefill rows absorbed into decode GEMM slack | needs a GEMM above Marlin's 68% tensor efficiency at M=48-64. That is the same slack T17 is spending on k=3; with a perfect GEMM, ~17 free rows/step, net ~3% after the per-row AR. It is a follow-on for T17 if their GEMM lands, not a separate direction. |

## Opinion
I agree with T2-T16: decode sits on coupled HBM, bf16-tensor and PCIe floors, and prefill on the bf16 tensor roof. What remains (lifecycle ~1-1.6%, rebalance ~1.8%, small-step fusions ~1.4%, lm_head screen ~0.8%) is a set of exploit items. A package of them would reach ~+5-6% only if every item hit its ceiling, which is the T14-1 pattern (failed G1 at +3.5%). I would rather not re-propose it as an explore direction. The cheapest next measurement, if anyone takes the lifecycle item, is one bench run that logs engine step boundaries and per-request arrival/scheduled/first-token step indices (~4 GPU-min).
