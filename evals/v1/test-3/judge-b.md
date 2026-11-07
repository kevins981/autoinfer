# Judge b: v1/test-3

## Resources
`python3 .claude/skills/eval-run/resources.py flavors/v1/runs/test-3`:
```
agent                            role           msgs    input  cache_wr  cache_rd   output  hours  model
bs-r1-adjacent_field             brainstormer     38       0K      130K     3.12M      49K  0.34  claude-opus-5-5
bs-r1-systematic_engineer        brainstormer     37       0K      135K     3.19M      55K  0.18  claude-opus-5-5
bs-r2-adjacent_field             brainstormer     34       0K      124K     2.87M      48K  0.16  claude-opus-5-5
bs-r2-systematic_engineer        brainstormer     34       0K      136K     3.07M      47K  0.16  claude-opus-5-5
bs-r3-adjacent_field             brainstormer     19       0K       81K     1.09M      25K  0.08  claude-opus-5-5
bs-r3-systematic_engineer        brainstormer     36       0K      111K     2.66M      32K  0.1   claude-opus-5-5
leader-1.w1                      worker            8       0K       32K      222K       5K  0.1   claude-sonnet-5-5
leader-1.w2                      worker           60       0K      143K     5.35M      67K  1.03  claude-sonnet-5-5
leader-1.w3                      worker           24       0K       54K      991K      10K  0.47  claude-sonnet-5-5
leader-1.w4                      worker           63       0K      143K     6.47M      44K  0.96  claude-sonnet-5-5
leader-1.w5                      worker           21       0K       56K      947K      11K  0.25  claude-sonnet-5-5
leader-1.w6                      worker           69       0K      236K    10.31M     127K  0.95  claude-sonnet-5-5
leader-1                         leader          111       0K      223K    14.68M      87K  3.51  claude-opus-5-5
manager                          manager          94       0K      178K    12.51M      88K  4.84  claude-opus-5-5
Per role: ALL output 695K (brainstormer 256K, leader 87K, manager 88K, worker 264K); cache_rd 67.5M
Wall clock (registered agents' start/stop): 2026-11-06T22:46:46 -> 2026-11-07T03:37:27, 4.84 h
GPU-minutes (gpu/ledger.jsonl): brainstorm/uncharged 13.2, official 55.0, unofficial 108.8, total 177.0
  r1: brainstorm 11.8, official 18.6, unofficial 28.1 | r2: 1.1 / 10.1 / 54.8 | r3: 0.3 / 10.0 (+10.7 team "-") / 25.9 | None: 5.7
```
Notes:
- The 10.7 GPU-min "round 3, team -" official is job 48, `official:BASE` run by **the human** at
  03:37-03:48, after the run stopped (journal.md:19, gpu/ledger.jsonl job 48). Run's own GPU use =
  177.0 - 10.7 = **166.3 GPU-min**. The run stopped at 03:08 (journal.md:19); the 4.84 h wall clock
  includes the manager's post-stop note (journal.md:20).
- Nothing missing: all 14 agents found with transcripts.

## Config and outcome
- Flavor v1; 1 leader (T1), max 2 workers, max 5 rounds, 80 GPU-min/team/round, brainstormers
  systematic_engineer + adjacent_field; manager/leader/brainstormers Opus 5.5, workers Sonnet 5.5
  (config.md). Task qwen38-27b-fp8-2x4090, focus = whole-engine c=16 tok/s (focus.md).
- Promotion bar: < 3% is noise (task_definition.md:4-5).
- BASE official 631.4 (bank/BASE/record.md; a human re-run after the run gave 635.9).
- **Champion (official): T1-009, 730.9 tok/s, +15.8%** (champion.md). **Best measured: T1-012, 749.7,
  +18.7%** (+2.6% vs T1-009), not promoted because of the 3% bar (reports/r3.md:19).
- Curve (champion vs cumulative GPU-min from the ledger, by job end):

  | after | GPU-min | champion | tok/s | vs BASE |
  |---|---|---|---|---|
  | start | 5.7 | BASE | 631.4 | 0 |
  | round 1 (00:42) | 64.2 | T1-005 | 699.2 | +10.7% |
  | round 2 (02:03) | 130.1 | T1-009 | 730.9 | +15.8% |
  | round 3 (03:08) | 166.3 | T1-009 (best measured T1-012 749.7) | 730.9 | +15.8% (+18.7%) |
- Human decisions (context, not agent choices): round-1 direction approved by the human
  (journal.md:6); rulings 1-3 (task_definition.md:52-59); GPU 0 clock lock accepted (journal.md:12);
  correctness rule (journal.md:13); tree format (journal.md:16); **stop after round 3 of 5**
  (journal.md:18). Unused budget per round: 33.4 (journal.md:9), 15.1 (reports/r2.md:9), 44.1
  (reports/r3.md:16) GPU-min.

## Overlay graph
![overlay](overlay.png)

Note: the plot draws a star on `gdn-replay` because it is in `best_measured.adds`; it is NOT in the
official champion's stack (T1-009).

```
BASE
├─ host-ar  [reached, as expected]                       T1-002
│  ├─ prefill-graphs  [reached, as expected]             T1-001 (BASE, noise) -> T1-005
│  │  └─ (novel) full-mixed-graph  [idea-only]
│  ├─ host-allgather  [reached, as expected]             T1-004
│  ├─ ar-kernel-tuning  [tried (component), as expected] T1-003 / KB5
│  ├─ ar-norm-fusion  [idea-only, -]
│  ├─ l2-prefetch  [idea-only, -]                        R1-D, R2-C, R3-D
│  ├─ ar-compression  [idea-only, -]                     R3-C
│  └─ (novel) ar-epilogue-host-write  [idea-only, ruled down]
├─ draft-head-diet  [reached, as expected]               T1-006, T1-007
│  └─ reduced-vocab-draft-head  [idea-only, as expected] R1-J
├─ greedy-fastpath  [reached, as expected]               T1-009
├─ sharded-sampling  [not-considered, -; alt greedy-fastpath reached]
├─ gdn-replay  [tried, official +2.6%, not promoted by the 3% rule; as expected]  T1-012
│  ├─ small-spec-state  [idea-only, -]                   R3-B v2
│  │  └─ draft-depth-3-small-state  [idea-only, -]
│  └─ (novel) replay-conv-state  [idea-only]
├─ draft-depth-3  [tried, as expected (cause found in r3)] T1-008, T1-010, T1-011
├─ nccl-env  [tried (component, brainstormer), as expected]
├─ prefill-ar-overlap  [idea-only, as expected]          R1-L
├─ triton-linear  [not-considered: forbidden by task_definition, -]
├─ stock-flag-tweaks  [not-considered, -]
├─ pp2 / uneven-tp / dequant-cublas-prefill  [idea-only, ruled out by estimate, as expected]
├─ spec-glue-graph  [idea-only, -]                       R1-H
├─ marlin-tuning  [idea-only (+ component microbench), -] R1-E, R2-D
├─ ce-ar-prefill  [idea-only, -]                         R2-E
└─ flashinfer-pcie-ar  [idea-only, -]
```

## Overlay
| node | contact | base | run's result and cause | verdict | evidence |
|---|---|---|---|---|---|
| host-ar | reached | BASE | +5.0% official (662.8); decode AR 60.8 -> 49.6 us, all gain in decode; bitwise == NCCL | as expected | bank/T1-002/record.md; T1-003; KB2 |
| prefill-graphs | reached | BASE, then host-ar + host-allgather | on BASE +2.2% unofficial, noise, "GPU-bound" (T1-001, KB1); on host-ar +4.8% / official 699.2, cause "cuts launch-induced skew", not separated (KB4) | as expected (the run re-tried it on host-ar on its own, the key's grading point) | bank/T1-001, bank/T1-005; shared_kb.md:3,6 |
| host-allgather | reached | host-ar | ~0 unofficial (666.8 vs 665.0); ~1% est, kept as exact building block | as expected | bank/T1-004; KB6 |
| ar-kernel-tuning | tried (component) | host-ar | chunk pipelining at M=48: K=2..8 slower (54-85 vs 51-53 us), fences ~5 us each; "near floor" | as expected | bank/T1-003 results (b); KB5 |
| ar-norm-fusion | idea-only | - | "Optionally fuse it with the following residual-add + RMSNorm", never scored or tried | - | brainstorm/r1/systematic_engineer.md:98 |
| l2-prefetch | idea-only | (BASE r1, host-ar r2/r3) | proposed 3 times (N5), est +3-7%; deferred each round, r3: do after gdn-replay. 1-GPU-min kill test never run | - (open) | ideas.md:10,24,39; brainstorm/r3/task_leader-1.md:65 |
| ar-compression | idea-only | host-ar | est +2-3%; manager ruled it baseline-precision; not selected | - | ideas.md:38; brainstorm/r3/adjacent_field.md:276-301,334-339 |
| draft-head-diet | reached | host-ar + prefill-graphs | local argmax +1.1% (T1-006), + fp8 draft head +3.0% same-job (T1-007), acceptance 2.343 vs 2.340 | as expected | bank/T1-006, T1-007; KB10 |
| reduced-vocab-draft-head | idea-only | draft-head-diet | ruled down in r3: net ≈ 0 once draft heads are FP8 (estimate) | as expected | ideas.md:16,43; brainstorm/r3/systematic_engineer.md:191-196 |
| greedy-fastpath | reached | host-ar + prefill-graphs + draft-head-diet | +2.4% same-job; official 730.9; exact incl. ties (25,511 rows, 467 ties, 0 mismatches) | as expected | bank/T1-009; KB10, KB12 |
| sharded-sampling | not-considered | - | no mention (`grep -rniE "sharded.sampl\|batch_sharded\|batch-sharded"` over all run .md outside workspaces: 0 hits). Alternative greedy-fastpath reached | - | grep |
| gdn-replay | tried (best measured, not in champion) | host-ar + prefill-graphs + host-allgather + draft-head-diet + greedy-fastpath | official 749.7 (+2.6% vs T1-009), same-job A/B +2.2%, ceiling microbench ~2 ms/step; bitwise unit test with negative control. Not promoted: below the run's 3% bar; manager/leader recommended promotion to the human | as expected (correct under the run's rule) | bank/T1-012; KB15; reports/r3.md:19 |
| small-spec-state | idea-only | gdn-replay | "R3-B v2", feasible but 1-2 days; risk unknown reader | - | ideas.md:46; reports/r3_leader-1.md:7; brainstorm/r3/adjacent_field.md:303-316 |
| draft-depth-3-small-state | idea-only | gdn-replay + small-spec-state | "With idea 1: +5-8% on top" (est) | - | brainstorm/r3/systematic_engineer.md:158-159 |
| draft-depth-3 | tried | host-ar + prefill-graphs + draft-head-diet (T1-008); + greedy-fastpath with KV 5.5-6.0 GiB (T1-010/011) | T1-008 -6.9% ab, TTFT p50 210->792, gate fail, dropped "cause undiagnosed" in r2 (KB11). r3 brainstormers found admission stall: Running 14 / Waiting 2 (KB13). T1-010 OOM at graph capture; T1-011 runtime OOM; final cause: needs smaller per-request state (KB16) | as expected (final drop cites capacity, F5/F6; r2 drop had no cause but was corrected in-run) | bank/T1-008, T1-010, T1-011; shared_kb.md:13,15,18 |
| nccl-env | tried (component, by a brainstormer) | BASE | microbench: Simple same, LL128 worse, 8 ch -6% at prefill size / worse at decode | as expected | brainstorm/r1/adjacent_field.md:20-37; ideas.md:9 |
| prefill-ar-overlap | idea-only | BASE / host-ar | R1-L; ruled down in r2: skew is GPU 0's clock, hides ≤4% with heavy engineering | as expected | ideas.md:18; brainstorm/r2/systematic_engineer.md:166-167; KB8 |
| triton-linear | not-considered | - | true FP8 compute is forbidden by this run's task definition, so not a miss | - | task_definition.md:15-16; brainstorm/r1/systematic_engineer.md:212 |
| stock-flag-tweaks | not-considered | - | no prefix-caching-off / attention-backend / batched-tokens tweaks tried (`grep -rniE "prefix.caching\|async.sched\|attention.backend\|max-num-batched"`: only r3 SE:148 declining prefix-caching-off as benchmark-tuned) | - | grep; brainstorm/r3/systematic_engineer.md:148-150 |
| pp2 | idea-only | BASE | ruled out: doubles weight streaming (~58 ms vs 40 ms) | as expected | brainstorm/r1/systematic_engineer.md:203-205; adjacent_field.md:138-140 |
| uneven-tp | idea-only | BASE | ruled out: nets ≈ 0 to +1.5% | as expected | KB8 (shared_kb.md:10); brainstorm/r2/adjacent_field.md:241-243; r2/systematic_engineer.md:68-69 |
| dequant-cublas-prefill | idea-only | BASE | ruled out: Marlin ~75% of peak at large M, ≤2% | as expected | brainstorm/r2/adjacent_field.md:250-254; ideas.md:32 |
| spec-glue-graph | idea-only | BASE | R1-H in-place FlashInfer draft-metadata (~1 ms gaps); never selected. r2 AF: step-start gap is an nsys artifact | - | ideas.md:14; brainstorm/r1/systematic_engineer.md:176-179; r2/adjacent_field.md:245-248 |
| marlin-tuning | idea-only | BASE / host-ar | R1-E/R2-D; r3 microbench: in-kernel M=48 penalty 1.6 ms/step, retune ≤2%, ruled down | - (open; no change tried) | ideas.md:11,25,41; brainstorm/r3/systematic_engineer.md:62-84 |
| ce-ar-prefill | idea-only | host-ar | R2-E, +1.5-3% est, half is GPU 0 skew | - | ideas.md:26; brainstorm/r2/adjacent_field.md:318-339 |
| flashinfer-pcie-ar | idea-only | BASE | "check in 5 minutes", expected to fail (needs cudaIpc device memory = P2P); never run | - | brainstorm/r1/systematic_engineer.md:99-101; adjacent_field.md:38-40 |

## Novel nodes
All of the run's tried records map onto key nodes (the key was built partly from this run). Untried
ideas with no key node:
- `full-mixed-graph`: FULL (not PIECEWISE) CUDA graphs for mixed prefill steps; builds_on
  prefill-graphs. idea-only, open (bank/T1-001/record.md:21; reports/r1_leader-1.md:5).
- `replay-conv-state`: remove redundant conv-state work in the GDN replay kernel; builds_on
  gdn-replay. idea-only, open (ideas.md:47; reports/r3_leader-1.md:7).
- `ar-epilogue-host-write`: Marlin epilogue writes AR partials straight to host memory to overlap
  decode AR with the GEMM; builds_on host-ar. Ruled down by code reading (fp32 reduce writes C only
  at the end) (brainstorm/r3/adjacent_field.md:319-320).

## Why missed
No missed wins and no standing wrong causes. Every key node with a win was reached, except
`sharded-sampling` (its alternative `greedy-fastpath` was reached) and `gdn-replay`, which was
reached with an official measurement but not promoted only because of the run's 3% rule
(task_definition.md:4-5; reports/r3.md:19). That was the correct move under the run's rules, so it
is not an agent miss.

Near-miss worth noting, `draft-depth-3` (r2): T1-008 was dropped as "cause undiagnosed (no
profile)" (bank/T1-008/record.md conclusion; KB11). The capacity cause was in T1-008's own server
log ("Running: 14, Waiting: 2"), which the leader admits it missed (local_kb/leader-1.md:24). The r3
brainstormers diagnosed it from existing logs at zero GPU cost (brainstorm/r3/systematic_engineer.md:16-38;
adjacent_field.md:217-240; KB13), and r3 then tested the follow-up correctly. Cost: about one round of
delay on the k=3 line. By the end the run's cause matched the key (F5, F6).

## Process notes
- **Diversity:** one leader, so one line per round. r1: comms + prefill graphs (two ideas);
  r2: only the logits diet (journal.md:11 notes that policy rules 1 and 2 conflict with one leader,
  so the novel slot was postponed); r3: k=3 + GDN replay. The high-novelty, cheap kill-test ideas
  (L2 prefetch, scored N5 three times; AR compression, a ~free CPU test) were never tested, while
  33.4 / 15.1 / 44.1 GPU-min were left unused (journal.md:9; reports/r2.md:9; reports/r3.md:16).
  A 1-GPU-min L2-prefetch microbench would have fit any round. This is the main process gap.
- **Measurement strength:** promotions rest on official evals with 2-3 trials (T1-002, T1-005,
  T1-009, T1-012). Drops: T1-001 rests on one unofficial run vs the official BASE (no same-job
  control), but the run correctly re-tested the flag on host-ar (T1-005). T1-006/T1-007/T1-009/T1-012
  used same-job A/B. T1-008 was one same-job A/B, dropped without a cause (see above). T1-010/011 are
  OOM outcomes, cheap (7.2 GPU-min, local_kb/leader-1.md:27).
- **Brainstormer measurements were high-value:** GPU 0's clock lock (KB8), the GDN L2 spill (KB9/KB14),
  the k=3 capacity diagnosis (KB13), NCCL env sweep (r1 AF M2), Marlin M-scaling (r3 SE M1). They
  steered the selection and ruled several directions down with numbers.
- **Rules:** workers ran over allowance twice (w4 ~5 min, approved; w6 ~4 min) (reports/r2_leader-1.md:14;
  r3_leader-1.md:22). The manager read test-1/test-2 drafts as structural templates at the start
  (journal.md:20; disclosed, no findings used). r1 adjacent_field kept scripts in /tmp/bs_adj
  (brainstorm/r1/adjacent_field.md:3), outside the run folder (minor, reproducibility).
- Correctness evidence for code changes was strong (bitwise tests with negative controls, in-server
  dual-path checks) (bank/T1-002, T1-009, T1-012).

## Key updates (proposed)
1. `nccl-env`: cite the actual measurement `v1/test-3:brainstorm/r1/adjacent_field.md:20-37` (microbench:
   LL128 worse, 8 channels -6% at prefill size / worse at decode), strength `component`, in addition to the
   idea id R1-C.
2. `flashinfer-pcie-ar`: add an estimate from code reading: the backend needs cudaIpc device memory
   mapped across GPUs (`flashinfer/comm/cuda_ipc.py:268`), i.e. P2P, so it is expected to fail or fall back
   here (`v1/test-3:brainstorm/r1/adjacent_field.md:38-40`). Still open (unmeasured).
3. `ar-norm-fusion`: add origin `v1/test-3:brainstorm/r1/systematic_engineer.md:98`.
4. `marlin-tuning`: add `component` numbers from `v1/test-3:brainstorm/r3/systematic_engineer.md:62-84`:
   Marlin per step M=3 14.03 / M=48 15.66 / M=64 17.22 / M=80 24.67 ms; fp32-reduce off 15.45 ms at M=48;
   retune ≤~2%. Also the in-server excess (~2.5 ms) is interference from GDN state writes (:40-60).
5. New open nodes: `full-mixed-graph` (builds_on prefill-graphs; bank/T1-001/record.md:21),
   `replay-conv-state` (builds_on gdn-replay; ideas.md:47). Optionally `ar-epilogue-host-write`
   (builds_on host-ar; ruled down by code reading, brainstorm/r3/adjacent_field.md:319-320).
6. `draft-depth-3` grading: add test-3's in-run history as an example. The r2 drop had no cause
   (KB11); r3 found it from existing logs (KB13). Already covered by the key's evidence; no status change.
7. Possible fact (low confidence, needs checking): the r3 SE claims align-mode prefix caching makes a GDN
   request hold up to `2 + k + 1` pages (brainstorm/r3/systematic_engineer.md:27-28,148), while r3 AF's
   count is `3 groups x (1+k) + 1` = 10/13, which fits the measured 5.24%/6.81% usage
   (brainstorm/r3/adjacent_field.md:229-236). The key's F5 uses 10/13. The two counts disagree, so it is not
   settled whether turning prefix caching off would shrink per-request state. That would also need a
   validity ruling (task_definition.md:41-42).
