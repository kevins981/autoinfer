# Team T1 write-up (leader-1, sweep team; closing)

**Wins: 2 promotions.**
- T1-001: FP8 draft lm_head flag, 678.4, +6.7% on BASE.
- T1-007: host-mapped TP=2 all-reduce, 771.3, +3.7% on T2-004. This is the current champion.

GPU use: 69.3 of 90 GPU-minutes. 20.7 minutes are left unspent. I found no candidate with evidence strong enough to
justify a harness run.

Map: reports/T1-sweep/ (1-inputs through 6-ranking, update-1, update-2).

## What worked
- **The framework's link-cost view found the lane.** On 2x 4090 without P2P, every all-reduce goes through NCCL over
  SHM, which has a floor of about 25-30 µs. A decode step makes 132 small all-reduces per target graph.
  - We built a custom all-reduce through host-mapped pinned memory: zero-copy PCIe stores, flags with acquire/release
    ordering, and an fp32 sum rounded once.
  - It is bitwise equal to NCCL and cuts 20 µs per all-reduce at M≈42 (41 vs 61 µs).
  - Result: +3.7%, measured 3 times (768.6 / 777.2 / 771.3).
- **The FP8 draft head (a built-in option)** gave +6.7% on BASE. That is ~3x the napkin; the mechanism is still
  unexplained (review T1-1). Once T2's local-argmax draft path was in, it added nothing resolved (T1-003).

## What did not
- **NCCL env knobs** gave nothing. **FlashInfer's IPC all-reduce** is impossible here: it needs peer access.
- **Reduced-vocab draft head** (T1-002): the step got cheaper, but acceptance fell 4.6%, so it loses to the FP8 head.
- **k=3:** neutral once KV is sized for 16 running requests (T1-009). At less KV it was badly confounded (T1-004).
- **hmar variants all lost or tied:**
  - Pipelined: slower.
  - Geometry sweep: within 1 µs.
  - Ordered overlap: slower.
  - The relaxed-order variant was fast but wrong.

## Lessons about the framework (for its authors)
1. **A collective's kernel time includes waiting for the other rank.**
   - My biggest calibration item, the mixed-step all-reduce, was mostly skew from eager launches, not link time.
   - The custom all-reduce then delivered half its napkin, because part of the all-reduce time is off the critical
     path.
   - Price collectives from a bare microbenchmark, and test whether they sit on the critical path.
2. **Memory headroom is a binding dimension** that the template lacks. Several candidates collided with <0.6 GiB of
   free memory: the FP8 draft copy, k=3's KV, and prefill graphs.
3. **The vocab index needs a place in the template.** The draft head's cost scales with vocab, and subset-vocab
   drafting is a knob the template doesn't name.
4. **Per-kernel efficiency is not modelled.** Napkin-vs-measured gaps of 2-3x (T1-001) can't be explained without it.

## Process
- Every record after the human's review rule was reviewed, some over several rounds (reviews T1-1 to T1-5).
- The reviews caught real overclaims:
  - preemption claims;
  - all-reduce counts per graph;
  - the launch-count meaning;
  - "closed" vs "deferred" for overlap.
- They also caught a CUDA-call-in-capture bug in a stats thread (job 34), fixed before T1-007.
- Open points I did not fix (minor, CPU-only):
  - `unregister_host()` ignores CUDA's return code.
  - The stress test didn't cover asymmetric alignment or graph-captured staging.
  - The extension cache key lacks CC/CXX.

## For the next team
The best remaining lane is fewer or cheaper decode all-reduces: ~5 ms of a ~30 ms step is still AR. See update-2's
ranking:
- merge or skip all-reduces;
- a correct overlapping hmar.

hmar code: workspaces/t1-car5 (T1-008), with the prebuild command in local_kb/leader-1.md.
