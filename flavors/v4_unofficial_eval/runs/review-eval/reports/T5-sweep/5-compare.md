# T5 sweep, step 5: my map against the other maps and the history

Read: reports/T1..T4-sweep (+ updates, writeups), bank/ (all records), shared_kb.md.

## A. Status of each of my candidates
| My candidate (4-knobs) | Status | By whom / on what code | Holds on the champion? |
|---|---|---|---|
| Host family R1 (recurrent GDN prefill for short prompts) | **tested, screen negative** (-1.3%, -1.4%, 1 start each) | T4-001 on T3-003 | yes: same GDN path in T3-007 |
| Host family premise (mixed steps host-bound, ~110 ms floor) | **refuted outside nsys**: per-iteration timer (no nsys): small mixed steps 56-66 ms vs ~52 ms GPU work | T4-001 eval/iter_quickbench | yes; my 3-calibration C used nsys numbers -> overstated (see C) |
| Host family R2 (fuse the GDN mixed-path glue) | untried as such; its premise is the same as R1's, so its value is now ~5-14 ms per small mixed step (T4), ~1% total | — | low value now |
| K1 DeltaLog GDN state | **untried**; ranked by T1 (#9, +1.6%), T2 (#8, +2%), T3 (#4: "all 3 slots really written; cold microbench 120/89/60 us for 3/2/1 writes vs ~70 us in the server; needs a DRAM-bytes check"), T4 (#2: "slot contract too large for the time left") | — | yes (same kernel) |
| S1 3 drafts | tested on older code: T1-004 (1 start, -13.8%, KV-capped at 12 requests, TTFT fail), T1-009 (1 start, -0.9% vs 771.3 on T1-008); T4: blocked by GDN state capacity (a 4th slot ~1.2 GB/GPU) | T1 on T2-004/T1-008 (before fp8 draft head, Triton attention) | partly stale; capacity is the binding constraint now |
| R3 reduced-vocab draft head | tested: T1-002 acceptance -4.6%, -2.4% vs fp8-head champion | T1 on BASE | likely still negative (fp8 head already halved the bytes) |
| W1 L2 prefetch volume | tested: T3 microbench picked 16 MiB; 32 MiB+ net loss | T3-001 (jobs 47-49) | yes -> drop |
| P1 hmar for > 96 rows | T1: hmar's per-row cost ~ NCCL's at MB sizes; T4 #5 dropped | T1 microbench | yes -> drop |
| DBO / micro-batch AR-compute overlap on mixed steps (my 4-knobs dropped only the decode version, didn't price the mixed one) | ranked by T1 #4, T2 P6, T3 K4-c; **dropped by T3 by code reading**: vLLM ubatching needs DP>1, hybrid unsupported, HMAR off under ubatching | T3 review T3-1 | the vLLM feature is blocked, but a hand-written row split of GEMM+AR inside the model is not (see B) |

## B. What my sweep has that the others lack (or price differently)
1. **The GDN state-write mismatch quantified on the champion**: napkin 4.8 ms vs 3.69 ms measured per decode step (3-calibration B):
   the in-server kernel is faster than "all writes reach DRAM", consistent with T3's 70 us in-server vs 120 us cold. My worker is
   measuring DRAM read/write bytes with ncu (--cache-control none), the check T3 and T4 both named as the gating test.
2. **A DeltaLog layout that keeps the existing slot contract**: row 0 of every verify window is always accepted (it is the last
   sampled token), so S after row 0 is exact and final; write it to slot 0 and log only rows 1-2's inputs (~50 KB per request per
   layer) in slot 1. Readers then need one change: "read slot 0, replay (accepted-1) logged rows". T4 judged the contract too large;
   this layout reduces it to the readers of slots 1..2 (the fused CUDA decode kernel, the Triton spec path in mixed steps, and the
   mamba-align temporal copies), and it frees slot 2's capacity, the resource k=3 is blocked on (T4 update-1).
3. **Mixed-step AR/GEMM overlap without vLLM's ubatching** (new): split only the row-local "GEMM -> AR" pairs (out_proj/o_proj -> AR,
   down -> AR) of mixed steps into 2 row chunks on 2 streams: AR(chunk 1) overlaps GEMM(chunk 2). Mixed steps are compute-bound
   (each half > ridge), so the second weight read is hidden (06 §3 condition). Napkin: hides min(GEMM, AR) per site, at most
   ~13 ms of a ~120 ms mixed step -> <= +4.5%, realistic ~+2%. T3's "blocked" verdict applies to vLLM's DBO, not to this.
4. A host-CPU resource in the window cost (3-calibration C). Its use here was wrong (nsys-inflated), but the dimension is real
   (T2-001 and T3-007 were both host-side wins).

## C. What the other maps have that mine missed (findings about my use of the framework)
- **Calibrate mixed steps without nsys** (T4 update-1: vLLM's `--enable-logging-iteration-details` is a free per-step timer).
  I read PROFILING.md's warning, checked it with a Prometheus average (107 ms per mixed step real ~ nsys 113 ms), and concluded the
  ~110 ms floor was real. The average hid the error: real small steps are ~60 ms and large ones 130-200 ms; the average matched by
  coincidence of the size mix. Lesson: compare distributions, not means.
- The step-boundary CPU edges (scheduler -> input prep -> graph replay, drafter metadata rebuild) in T3's graph: T3-007's win came
  from exactly that node (FlashInfer's eager draft metadata), which my graph listed only as "[sched/prepare (CPU)]".
- T2's GEMM-efficiency analysis at mixed sizes (Triton reaches 210-230 TFLOP/s at M>=1024, above my 165 F constant): my
  F was too low for prefill; mixed steps are less compute-bound than my napkin said.
- T1's/T3's AR microbenchmarks: hmar ~41-47 us at decode sizes, NCCL ~205 us at 2.66 MB.

## D. Disagreements
- T4 update-1: "DeltaLog not attempted: the slot contract is too large". I disagree with the size: with the row-0 checkpoint layout
  (B.2) the decode-only path is one kernel change plus a log write; mixed steps can route spec rows to the same CUDA kernel (T4's own
  #1 suggestion) instead of adapting the Triton kernel. I agree the align-mode temporal copies are the risky reader.
- T3 K4-c "DBO dropped (blocked)": blocked only as the vLLM feature (B.3).
- T1/T2 DeltaLog at +1.6-2%: priced with all writes in DRAM and before the other decode costs fell; on the champion GDN is 3.69 of
  ~28-31 ms (12%), so the share is larger now; the ncu result decides.
