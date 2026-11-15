# T1 map update 3 (resync 2)

## Tested since update-2
| candidate | record / job | result | verdict |
|---|---|---|---|
| H-PREP: decode-step host stall (new dimension: Host, found with a graph-level trace) | **T1-008** (leader-1.w1), unofficial #1, prof t1-prof-t1008-graph | 758.8 / 762.4 vs T2-006 738.8 / 747.8 (+2.3%); decode idle 1.92 -> 0.61 ms/step, decode period 34.11 -> 32.78 ms | **win (candidate sent)**. Cause: the drafter's FlashInfer build did `seq_lens.cpu()` (pageable D2H) and blocked the host ~30 ms per step; planning with the CPU upper bound plus exact GPU-written indptr/indices removes it. Napkin "up to ~4%": measured -3.9% per decode step, +2.3% end to end |
| GDN state zeroing via torch.where (idea A) | none (worker, CPU-only) | the old line doesn't sync | no-op, dropped |
| W1b: graph-capture GDN prefill in mixed steps | not built | 400-600 lines estimate | deferred (see ranking) |

## What the framework got right / wrong
- Missing: the framework has no node for the **host** (scheduler, metadata building). Both decode-step
  wins in this lane after the comm work came from making a host->GPU dependency edge asynchronous
  (Rewrite of a sync edge into a bound + exact GPU fix-up). The window/Link model could not see it;
  a graph-level nsys trace (GPU idle in step window) could. Suggest: add Host as a resource with
  its own edges (D2H syncs) and price "GPU idle per step" from a trace before pricing kernels.
- The "upper bound + exact device fix-up" trick is a general pattern for speculative/async
  pipelines: plan with what the host knows, correct on the device.

## Current ranking (T1 lane, vs T1-008 ~760 unofficial)
1. Mixed-step GPU idle: 10-16 ms per ~120 ms mixed step (~30% of time) => <= ~3-4%. Find the causes with
   the same trace method (host stalls in mixed steps: other .cpu() syncs in the target build /
   GDN prefill metadata, eager breaks). Cheap first: CPU-side sampling of the worker thread
   during mixed steps; fix any sync with the same bound + fix-up pattern. M effort.
2. Remaining decode idle 0.61 ms (~1.8% of a decode step): small; check what is left (target build's
   own seq_lens.cpu() is exact-only by design).
3. Draft-side extra lm_head GEMM (1.58 ms "other GEMM" in decode is now visible outside graphs): check
   whether it can be captured / shrunk. Unknown, needs a look.
4. k=3 on the new stack: T2's lane (t2-k3-kv6).
