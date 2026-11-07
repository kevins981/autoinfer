# Judge a: v1/test-4

All paths are relative to `flavors/v1/runs/test-4/` unless they start with `answer_keys/` or `evals/`.
Key = `answer_keys/qwen38-27b-fp8-2x4090/answer_key.md`. The key was partly built from this run, so many
matches are expected. I graded by what each change does.

## Resources
`python3 .claude/skills/eval-run/resources.py flavors/v1/runs/test-4`:
```
agent                       role          msgs  cache_wr cache_rd  output  hours model
bs-r1-adjacent_field        brainstormer    47     127K    4.34M     41K   0.47 claude-opus-5-5
bs-r1-systematic_engineer   brainstormer    59     118K    5.13M     42K   0.47 claude-opus-5-5
leader-1.w1                 worker          66     152K    6.02M     64K   1.47 claude-sonnet-5-5
leader-1.w2                 worker          54     113K    4.08M     37K   1.21 claude-sonnet-5-5
leader-1                    leader          66     150K    6.49M     56K   2.76 claude-opus-5-5
leader-2.w1                 worker          56     131K    5.28M     31K   0.37 claude-sonnet-5-5
leader-2.w2                 worker          54     120K    3.90M     45K   0.41 claude-sonnet-5-5
leader-2                    leader          85     175K    9.75M     51K   2.76 claude-opus-5-5
manager                     manager         52     102K    5.20M     43K   3.26 claude-opus-5-5
Per role: ALL msgs 539, cache_wr 1.19M, cache_rd 50.19M, output 411K
  brainstormer 106 msgs / 84K out; leader 151 / 107K; manager 52 / 43K; worker 230 / 177K
Wall clock: 2026-11-07T03:48:48Z -> 07:04:34Z, 3.26 h
GPU-minutes (gpu/ledger.jsonl): brainstorm/uncharged 19.8, official 41.5, unofficial 103.2, total 164.5
  r1: T1 official 20.9 + unofficial 29.9; T2 official 10.0 + unofficial 19.6; brainstorm 19.8
  r2: T1 unofficial 26.9; T2 official 10.6 + unofficial 26.8
```
Gaps: resources.py found transcripts for all 9 agents, so nothing was missing. Fresh input tokens show as ~0K because almost
everything is a cache read. 27 ledger entries (`gpu/ledger.jsonl`). Brainstormer job 2 (systematic_engineer, 8.0 GPU-min)
timed out with no output (`gpu/ledger.jsonl` line 2, `brainstorm/r1/systematic_engineer.md:24-25`).

## Config and outcome
- Flavor v1. Task qwen38-27b-fp8-2x4090. 2 leaders x 1 worker, 2 rounds, 55 GPU-min per team per round. Brainstormers:
  systematic_engineer and adjacent_field. Models: Opus 5.5 for manager, leaders and brainstormers; Sonnet 5.5 for workers
  (`config.md`). Focus: whole-engine speedup (`focus.md`).
- BASE: 635.9 tok/s official (`bank/BASE/record.md`).
- Champion: **T1-003, 691.3 tok/s official, +8.7%** (trials 696.1/686.4; all gates pass; `champion.md`, `bank/T1-003/record.md`).
  T1-003 = the K1 host-staged AR + the K3 drafter flags + `--enable-batch-sharded-sampling` (K4b).
- Curve, champion against cumulative GPU-min from the ledger:
  - start: 635.9 at 0 GPU-min;
  - end of r1: 691.3 at ~100 GPU-min (T1 50.8 + T2 29.6 + brainstorm 19.8);
  - end of r2: 691.3 (no change) at 164.5 GPU-min.
- Round 2 also produced T2-006 at 699.5 official, but an ABAB A/B measured +0.2% and it was not promoted (`journal.md:14`).
- The frontier in the key is 749.7. This run reached K1+K3+K4 but not K2 or K5.

## Overlay
| node | label | run evidence | note |
|---|---|---|---|
| K1 host-staged AR | **reached** | T1-001 (official 672.5, +5.7%); KB2, KB3 | Matches exactly: /dev/shm + cudaHostRegister Mapped, ≤2 MiB dispatch before PyNCCL, bitwise = NCCL (`bank/T1-001/record.md`). |
| K2 prefill CUDA graphs | **never-considered** | none | Searched records, KB, ideas, brainstorm/, journal, local_kb and reports for `prefill_cudagraph\|cudagraph_capture\|max_prefill\|cudagraph_mode\|prefill.*graph`: no hits. In agents/*/events.jsonl the string appears only in dumped server configs (`'max_prefill_cudagraph_capture_tokens': 96`). leader-1 noted "compilation mode NONE, graphs <=96 tokens; prefill steps eager" (`local_kb/leader-1.md:22`) but did not act on it. |
| K3 drafter head diet | **reached** | T2-001 (+2.4% ab, noted as within noise); T2-003 official; KB4 | Uses exactly the flags `use_local_argmax_reduction` and `draft_lm_head_quantization=fp8`. |
| K4 target-logits diet | **reached (both K4b and K4a)** | K4b: T2-003 (official 653.9 with K3), in the champion T1-003. K4a: T2-004 (exact, 0/9600 mismatches), T2-006 (official 699.5; ABAB +0.2% vs K4b); KB5 | Found that the two are equivalent, as the key says. |
| K5 GDN spec-state replay | **never-considered** | none | Searched records, KB, ideas, brainstorms, journal, local_kb and reports for `replay\|recurrent state\|ssm state\|spec.state\|state block\|mamba\|GDN`. GDN appears only as a profile line item (KB10; `brainstorm/r1/adjacent_field.md:10`; T2-005: "GDN 3.84->4.63"). No idea targets GDN state writes. |
| D1 3 draft tokens | **tried, dropped, wrong or no cause** | T2-002 (-3.9%, TTFT p99 2225); T2-005 (flat on the champion, TTFT p99 2185); KB6, KB13 | The drop was right, but the cause was given as step cost ("+21% step", later corrected to +12.6%) and prefill dilution. TTFT is "cause NOT established", with the hypothesis "requests queue behind 2048-token prefill steps" (`agents/leader-2.w2/REPORT_job1.md:30`; KB13). No mention of GDN spec-state blocks or KV capacity (F5). `grep -i "admission\|blocks per request\|kv pool"` over records/KB/local_kb: no hits. |
| D2 NCCL env knobs | **tried, dropped, right cause** | KB1; brainstorm/r1/adjacent_field.md:20-24 (decode sizes); T1-001 record (NCCL_PROTO=Simple at T=256-2048) | Measured at both decode and prefill sizes. |
| D3 AR kernel micro-tuning | **tried, dropped, right cause** | T1-002; KB3 | Cause given: ~10 us from the PCIe floor, link not full-duplex. Matches F3. |
| D4 prefill AR/compute overlap | **tried, dropped, right cause** | T1-004 (0.0% at bench 400; big prefill steps -10% under nsys); KB14, KB15 | Matches the key, including the finding that Marlin must reserve SMs (F9). |
| D5 host-staged logits all-gather | **not-tried** | Mentioned in passing ("same building block also serves the logits all-gather", `brainstorm/r1/adjacent_field.md:41`) | Superseded by K3/K4 in this run, same as in the key. |
| D6 `--linear-backend triton` | **not-tried** | none | `grep -i "linear.backend\|triton"` over records/KB/ideas/brainstorms: only T2-004's Triton greedy_verify kernel matched, which is unrelated. |
| D7 stock serve-flag tweaks | **not-tried** | none | `grep -i "prefix.caching\|max-num-batched\|attention.backend\|async.sched"`: only task_definition rules matched. max_num_batched_tokens was named as something to check (`local_kb/leader-2.md:18`) but never varied. |
| D8 PP=2 / uneven TP | **not-tried (ruled out by estimate, consistent with the key)** | KB10; `brainstorm/r1/systematic_engineer.md:28-30`; `brainstorm/r1/adjacent_field.md:99-101` | PP=2 by napkin math only. Uneven TP was not considered. |
| O1 one-graph spec-decode glue | **idea-only (component analysis)** | T2-007; KB12; `agents/leader-2.w2/REPORT_job1.md:32-33` (proposals A/B) | Analyzed (1.6 ms/step idle, 1.24 ms of it CPU Python) but not built. |
| O2 one GDN spec block/request | **not-tried (never surfaced)** | none | Same searches as K5 and D1. |
| O3 fuse residual+RMSNorm into AR | **idea-only** | `reports/r1_leader-1.md:10`; `reports/final.md:27` | Listed as left open. |
| O4 L2 prefetch during AR | **idea-only (with a component measurement)** | R1-D (`ideas.md:5`); `brainstorm/r1/adjacent_field.md:90-97` (job 3, `marlin_l2.py`) | L2-hot Marlin at M=48 measured 43.6/47.8/40.9 us vs ~53/56/50 in-server, so the ceiling is ~+3%. Not tried. |
| O5 lossless AR compression | **not-tried (never surfaced)** | none | `grep -i compress` over all run .md files: only the assumption ledger and task rules matched. |
| Marlin in-kernel (open) | **idea-only** | R1-C (`ideas.md:4`); `brainstorm/r1/systematic_engineer.md:96-109` | Dropped before testing, using the same L2-hot measurement. |

## Novel nodes
No run node gives a win outside the key. Smaller findings the key doesn't carry yet:
- **The batch-sharded sampling (K4b) path costs an NCCL SendRecv of 0.41 ms/step**
  (`agents/leader-2.w2/REPORT_job1.md:13,34`). This motivated T2-006.
- **Depth-3 step-cost correction.** The first estimate of +21% step cost (KB6, from throughput) was
  wrong; nsys measured +12.6% for +16.7% tokens/step (T2-005, KB13).
- **Node-level nsys artifact.** `--cuda-graph-trace=node` inflates cudaGraphLaunch to ~1.2 ms vs ~120 us
  at graph level (KB12). It is a measurement fact, close to F8.
- **NCCL graph-capture hang.** Brainstormer job 2 hung while capturing NCCL in a CUDA graph at prefill
  sizes (`brainstorm/r1/systematic_engineer.md:24-25`). It is a process fact, and 8 GPU-min were lost to it.

## Why missed
- **K2 (prefill CUDA graphs).**
  - Closest approach: leader-1 recorded that prefill steps run eager (compilation mode NONE, graphs only
    ≤96 tokens; `local_kb/leader-1.md:22`), and T1-004 targeted those same eager prefill steps.
    T1-004 tried to hide their ARs with chunked GEMMs instead of graphing them.
  - The profile said prefill-containing steps are 35-38% of wall (KB10, KB13). T2-005's report even named
    "cudagraph padding / capture sizes" as something to check (`local_kb/leader-2.md:17-18`).
  - What happened: neither brainstorm proposed raising the prefill capture size. In round 2 the brainstorm
    was skipped (`journal.md:12`) and both slots went to the leaders' own "next". So the fact was
    observed but never turned into a flag test.
  - The run did have K1 in hand, so the base was right. This is a miss in ideation, not in measurement.
- **K5 (GDN spec-state replay).**
  - Closest approach: GDN decode appears in every step anatomy (2.4-3.7 ms; KB10). T2-005 measured it
    growing 3.84 → 4.63 ms at depth 3.
  - What happened: no agent looked at what the GDN kernels write per draft position. The brainstorms
    ranked GDN below AR, lm_head and Marlin as a cost item. Never surfaced.
  - The two-round budget also limited this: round 2 had no brainstorm.
- **D1 cause** (not a K node, but a wrong-cause drop):
  - leader-2.w2 saw that TPOT fell 16% while total duration stayed the same: "the gain moved from decode
    to queueing" (`agents/leader-2.w2/REPORT_job1.md:30`). That is the admission-stall symptom.
  - But it was attributed to waves of prefill steps, not to the KV/GDN-state pool admitting only 14 of 16
    requests (F5). No one looked at num_waiting or KV block counts. The metrics snapshots
    (`agents/leader-2/abab/*/bench/metrics_*.txt`) existed for other jobs, but no depth-3 analysis read them.
  - As a result, O2 (the fix that would unblock depth 3) was never surfaced.

## Process notes
- **Diversity.**
  - Round 1: two directions (comms; drafter/logits head). Both brainstormers independently picked the same
    top two (`journal.md:6`), so they were not diverse.
  - Round 2: no brainstorm, so both leaders continued along their own lines (search policy rule 1; `journal.md:12`).
  - Never explored: compilation/graphs, GDN, scheduler/memory, and serve flags beyond spec-config.
- **Evidence behind decisions** (mostly strong):
  - Promotions were official evals: T1-001, T2-003, T1-003.
  - T2-006 was correctly held back after an order-controlled ABAB of fresh servers (`journal.md:14`, T2-006).
  - T1-004 was dropped after two pairs at 400 prompts plus an nsys explanation.
  - D2 and D3 drops rest on microbenchmarks (component level), which is appropriate.
  - Weak spot: T2-002 was dropped after one 150-prompt A/B. The direction was right, but the cause was not established.
  - T1-003's promotion rested on job 15, a single non-interleaved run, then official.
- **GPU time without information:**
  - Brainstormer job 2 hung (8.0 min).
  - T2-005 was a depth-3 profile on the champion. It closed the depth question but ran under nsys, which slowed the bench.
  - T1-004 cost ~27 GPU-min plus a C++ Marlin change for a result that was always going to be at most ~1-3% on prefill steps.
  - Budget headroom went unused: T1 29.2 min in r1 (`journal.md:11`), T1 28 and T2 ~18 in r2.
- **Rule issues:**
  - A worker edited workspace code during a GPU job, which contaminated arm C of job 1
    (`reports/r1_leader-2.md:10`, T2-001 caveat). It was re-measured cleanly.
  - T1-004 changed Marlin per-M fp ordering (≤1 ulp). The validity review discussed it and it was never
    taken to official. No other violations found.

## Key updates (proposed)
1. **D1 evidence note.** Add that test-4 dropped D1 with the wrong cause: step cost / prefill-wave
   queueing, "cause NOT established" (KB13; `agents/leader-2.w2/REPORT_job1.md:30`).
   - Add its observation as a supporting symptom of F5: TPOT -16% with unchanged total duration, and
     TTFT median 334 → 938 ms.
   - Today the key cites test-4 T2-005/KB13 for D1 without saying its cause differed.
2. **O4: add test-4's component measurement.** L2-hot Marlin at M=48 measured 43.6/47.8/40.9 us vs
   ~53/56/50 in-server, giving a ceiling of ~+3% (`brainstorm/r1/adjacent_field.md:90-97`, job 3).
   - That makes O4's +3-5% estimate the upper end. Strength: component.
3. **Marlin in-kernel (Open) / F7: add the same measurement.** L2-hot Marlin is near its own floor
   (`ideas.md:4`, R1-C).
4. **New fact.** Batch-sharded sampling (K4b) adds an NCCL SendRecv of ~0.41 ms per decode step
   (`agents/leader-2.w2/REPORT_job1.md:13`). It explains why K4a and K4b tie only roughly.
5. **K4: add test-4's official numbers on K1+K3.** K4a (T2-006) 699.5 vs K4b (T1-003) 691.3 official;
   order-controlled ABAB +0.2% (`journal.md:14`). Strength: abab, not significant.
6. **K2 recognition note.** On BASE the engine runs with compilation mode NONE and
   `max_prefill_cudagraph_capture_tokens: 96`, so prefill steps are eager (`local_kb/leader-1.md:22`;
   server config dump in `agents/leader-1/events.jsonl`).
   - This helps judges spot near-misses: a run that records "prefill steps eager" without testing the flag
     came close to K2.
7. **F8 addition.** Node-level nsys (`--cuda-graph-trace=node`) inflates cudaGraphLaunch to ~1.2 ms vs ~120 us
   at graph level (KB12). The key's F8 already cites KB11/KB9/KB15, but not this one.
8. **O1 detail.** Add the cheaper sub-option from `agents/leader-2.w2/REPORT_job1.md:32`: skip no-op eager
   launches (_zero_kv_blocks, _apply_write x5, mamba pre/post), est. 0.5-1.0 ms.
   - T2-007 later bounded the pure launch-fusion part at 0.1-0.2 ms. Keep both estimates, labelled `estimate`.
