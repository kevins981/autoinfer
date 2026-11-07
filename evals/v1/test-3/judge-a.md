# Judge a: v1/test-3

Run folder: `flavors/v1/runs/test-3/` (paths below are relative to it unless they start with `answer_keys/`).
Key: `answer_keys/qwen38-27b-fp8-2x4090/answer_key.md`. The key was built partly from this run, so many matches are expected. I graded each one against the run's own records anyway.

## Resources
`python3 .claude/skills/eval-run/resources.py flavors/v1/runs/test-3` (verbatim, abridged to the totals):
```
agent                       role          msgs  cache_wr  cache_rd  output  hours  model
bs-r1-adjacent_field        brainstormer    38    130K     3.12M     49K   0.34  opus-5-5
bs-r1-systematic_engineer   brainstormer    37    135K     3.19M     55K   0.18  opus-5-5
bs-r2-adjacent_field        brainstormer    34    124K     2.87M     48K   0.16  opus-5-5
bs-r2-systematic_engineer   brainstormer    34    136K     3.07M     47K   0.16  opus-5-5
bs-r3-adjacent_field        brainstormer    19     81K     1.09M     25K   0.08  opus-5-5
bs-r3-systematic_engineer   brainstormer    36    111K     2.66M     32K   0.1   opus-5-5
leader-1.w1..w6             worker     8/60/24/63/21/69  (663K wr, 24.30M rd, 264K out)  sonnet-5-5
leader-1                    leader         111    223K    14.68M     87K   3.51  opus-5-5
manager                     manager         94    178K    12.51M     88K   4.84  opus-5-5
Per role: ALL msgs 648, cache_wr 1.78M, cache_rd 67.48M, output 695K
  brainstormer 256K out; leader 87K; manager 88K; worker 264K
Wall clock: 2026-11-06T22:46:46Z -> 2026-11-07T03:37:27Z, 4.84 h
GPU-minutes: official 55.0, unofficial 108.8, brainstorm/uncharged 13.2, total 177.0
  r1: bs 11.8, T1 official 18.6, T1 unofficial 28.1
  r2: bs 1.1,  T1 official 10.1, T1 unofficial 54.8
  r3: bs 0.3,  official (manager) 10.7, T1 official 10.0, T1 unofficial 25.9
  no round (BASE official #1): 5.7
```
The script found all 14 agents' transcripts and the ledger. Nothing was missing. One ledger job was cancelled: job 2, bs-r1-adjacent_field `bs-adj-arbench`, 11.8 GPU-min, exit 130 (`gpu/ledger.jsonl`).

## Config and outcome
- Flavor v1. Task qwen38-27b-fp8-2x4090. Config (`config.md`): 1 leader, at most 2 workers, at most 5 rounds, 80 GPU-min per team per round, 2 brainstormers (systematic_engineer, adjacent_field). Opus 5.5 for manager, leader and brainstormers; Sonnet 5.5 for workers; all at medium effort.
- The run stopped after round 3 of 5, at the human checkpoint (`journal.md:19`, `reports/final.md:1`).
- **BASE** 631.4 tok/s official (`bank/BASE/record.md`, official #1). A re-measure at the end gave 635.9 (official #2).
- **Champion (official): T1-009, 730.9 tok/s, +15.8%** (`champion.md`).
- **Best measured: T1-012, 749.7, +18.7% vs BASE and +2.6% vs T1-009.** It passes all gates and the 128K test but was not promoted, because +2.6% is below the 3% bar. The run left the decision to the human (`reports/r3.md:4-5`, `tree.md:3`).
- Champion curve against cumulative GPU-minutes (from the ledger):

| point | champion (official) | cum. GPU-min (approx.) |
|---|---|---|
| BASE | 631.4 | 5.7 |
| end r1 | T1-005, 699.2 (+10.7%) | ~64 |
| end r2 | T1-009, 730.9 (+15.8%) | ~130 |
| end r3 | T1-009, 730.9 (best measured T1-012, 749.7, +18.7%) | 177 |

## Overlay
| node | label | run evidence | note |
|---|---|---|---|
| K1 host-staged AR | **reached** | T1-002 (official 662.8, +5.0%); KB2, KB5; T1-003 | Pinned /dev/shm segment, cudaHostRegister mapped, fp32 sum, NCCL fallback, bitwise equal to NCCL (`bank/T1-002/record.md`). |
| K2 prefill CUDA graphs | **reached** | T1-001 on BASE (+2.2%, noise; KB1), then T1-005 on K1 (official 699.2; KB4) | It was tried on BASE and found no gain there, then correctly retried on K1. The retry is the step the key's grading note asks for. |
| K3 drafter head diet | **reached** | T1-006 (local argmax, +1.1%), T1-007 (+FP8 draft head, +3.0% ab vs T1-005; KB10) | Both spec-config keys are present (`bank/T1-007/serve_args.txt`, `champion.md`). |
| K4 target-logits diet | **reached (K4a)** | T1-009 official 730.9; KB10, KB12 | Greedy local-argmax path with a cross-rank tie-break, checked on 25,511 real rows including 467 ties (`bank/T1-009/record.md`). K4b (`--enable-batch-sharded-sampling`) never appears: `grep -i "batch.sharded"` over tree, records, KB, ideas, brainstorms, journal and reports gives 0 hits. |
| K5 GDN spec-state replay | **reached** (measured, not promoted) | T1-012 official 749.7; KB15 | Replay plus flush kernels in csrc, bitwise unit test with a negative control (`bank/T1-012/record.md`). Not the official champion only because of the run's 3% bar. |
| D1 3 draft tokens | **tried, dropped, right cause** | T1-008 (-6.9%, TTFT gate fail); KB11 (cause "unknown"); KB13 (admission stall, 14 of 16 running, 10 vs 13 blocks per request); T1-010/T1-011 (more KV OOMs; KB16); k=4 Marlin cliff (KB14) | When T1-008 was dropped in r2, the cause was undiagnosed (KB11). The r3 brainstorm found the right cause (capacity, F5) from the T1-008 server log (KB13). The run then tested the remedy "more KV", which failed (KB16, F6). |
| D2 NCCL env knobs | **tried, dropped, right cause** | `brainstorm/r1/adjacent_field.md:24-36,141` microbench (Simple/LL128, channels: at most -6% at prefill size, worse at decode); ideas R1-C (`ideas.md:9`) | Measured by a brainstormer's microbench (component level), not in the server. |
| D3 AR micro-tuning | **tried, dropped, right cause** | T1-003 (b): chunk pipelining K=2..8 loses, about 5 us per fence; KB5 | The cause given matches F3 (near the PCIe floor). |
| D4 prefill AR/compute overlap | **not-tried** (idea, set aside by analysis) | idea R1-L (`ideas.md:18`); KB8; `brainstorm/r2/adjacent_field.md:62,163` | It was set aside because the skew is GPU 0's clock (F2), not by a measurement. The reason fits the key's account of this run. |
| D5 host-staged logits all-gather | **tried; kept as a neutral building block** | T1-004 (666.8 vs 665.0, about 0; KB6), carried into T1-005 and all later records | The cause given is right: the gathers are small (KB6). It stays in the champion stack but no gain is claimed for it, so this does not contradict the key. |
| D6 Triton linear backend | **not-tried** | `grep -iE "linear-backend\|linear_backend\|triton.*fp8 gemm"` over tree, records, KB, ideas, brainstorms, journal and reports: 0 hits | The nearest is "dequantize + cuBLAS for prefill GEMMs", ruled down by estimate (`brainstorm/r2/adjacent_field.md:75,165`, `ideas.md:32`). |
| D7 stock serve-flag tweaks | **not-tried** | `grep -iE "prefix.caching\|attention.backend\|max-num-batched\|async.sched"`: only context mentions (`brainstorm/r3/systematic_engineer.md:25,148`: disabling prefix caching considered for the GDN pages and not proposed) | |
| D8 PP=2 / uneven TP | **not-tried** (ruled out by estimate) | `brainstorm/r1/adjacent_field.md:138`, `brainstorm/r1/systematic_engineer.md:203` (PP=2); `brainstorm/r2/systematic_engineer.md:68`, `brainstorm/r2/adjacent_field.md:62` (uneven TP); `ideas.md:29-30,44`; KB8 | Napkin math only, as the key says. |
| O1 one-graph spec-decode glue | **idea-only (narrower version)** | R1-H, in-place FlashInfer draft-metadata update for about 1 ms of CPU gaps (`ideas.md:14`; `brainstorm/r1/systematic_engineer.md:176-180`) | Never scored high, never tried. In r2 the decode step-start gap was called an nsys artifact (`brainstorm/r2/adjacent_field.md:68,164`). |
| O2 one GDN spec block per request | **idea-only** | R3-B v2 (`ideas.md:46`; `bank/T1-012/record.md` conclusion; KB16, KB17) | The run is the source of this node. |
| O3 residual-add + RMSNorm in the AR | **idea-only (passing mention)** | `brainstorm/r1/systematic_engineer.md:98` ("Optionally fuse it with the following residual-add + RMSNorm") | Never scored or pursued. `grep -i rmsnorm` finds only this line and a kernel-sum row (`brainstorm/r2/systematic_engineer.md:55`). |
| O4 L2 prefetch during the AR | **idea-only** | R1-D, R2-C, R3-D (`ideas.md:10,25,39`) | Re-proposed in 3 rounds and never selected. |
| O5 lossless AR compression | **idea-only** | R3-C (`ideas.md:38`) | |
| O (Marlin in-kernel) | **idea-only, with a component measurement** | R1-E / R2-D (`ideas.md:11,26`); ruled down in r3 by microbench: the M=48 in-kernel penalty is only about 1.6 ms/step (KB14, `ideas.md:41`) | This is a measurement: see key updates. |

## Novel nodes (no matching key node)
- **Reduced-vocab draft lm_head (R1-J):** allowed by ruling 2. It was carried through r1-r2 as untried (`ideas.md:15`, `reports/r2_leader-1.md:6`). In r3 it was ruled down by estimate: "net ≈ 0 now that the drafts are FP8" (`ideas.md:43`). Never measured.
- **Copy-engine full-duplex AR for large M (R2-E):** idea only, estimated +1.5-3% (`ideas.md:27`; `brainstorm/r2/adjacent_field.md:139`). Copy-engine variants were also judged unlikely at M=48 and not built (`bank/T1-002/record.md` conclusion).
- **Dequantize + cuBLAS for prefill GEMMs:** ruled down by estimate at 2% or less, because Marlin already runs at about 75% of peak (`brainstorm/r2/adjacent_field.md:75,165`; `ideas.md:32`).
- **Uneven weight rebalancing for the GPU 0 skew:** at most 3% and costs memory, by estimate (`ideas.md:44`). This could merge into D8.
- **`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`:** proposed as a 5-minute check (`brainstorm/r1/systematic_engineer.md:99`). No record or worker note tests it: `grep -rl PCIE_IPC` hits only server-log env dumps and `envs.py`. Expected to fail without P2P. Never tested.
- **Prefill-graph cap 1024 to free memory (T1-011):** freed only 0.26 GiB. Already captured in F6 (KB16).

## Why missed
Every K node was reached, so no K node needs a why-missed entry. Two near misses are worth recording:
- **K5 not promoted.** It was reached and measured, but the run's own 3% promotion bar kept it out of the champion (`reports/r3.md:4`). The measured gain is consistent across three measurements:
  - same-job A/B: +2.2%;
  - microbench: about 2 ms/step;
  - 2 official trials each: 746.0/753.5 for T1-012 vs 729.7/732.1 for T1-009.

  For the overlay this is reached; the gap between the official champion and the best measured result is a flavor-policy effect.
- **K4b not found.** K4a (custom code) reached the same gain, so nothing was lost. The run wrote a code path while the flag existed, and K4a cost a worker about 1 h (leader-1.w4). The stock flag was never surfaced in any brainstorm (grep above).

Of the open nodes, O4 (L2 prefetch) was proposed in all three rounds and never selected. In r1 and r2 the higher-confidence lines won (`journal.md:11`). In r3 it was explicitly deferred until after R3-B (`ideas.md:39`). Then the run stopped after 3 of 5 rounds with budget left: r3 used 35.9/80 GPU-min (`reports/r3.md:2`).

## Process notes
- **Diversity.** One leader means one direction per round:
  - r1: comms, plus one flag test;
  - r2: logits/spec diet;
  - r3: spec capacity plus GDN state, two sub-directions.

  The directions differ round to round. Each continues the winning line (search policy rule 1, `journal.md:11,17`).
- **Evidence behind decisions.**
  - Promotions rest on official multi-trial evals (T1-002, T1-005, T1-009, T1-012).
  - Stacked steps used same-job A/Bs (T1-006, T1-007, T1-008, T1-012).
  - Drops rest on one same-job A/B (T1-008), a single run against an official number (T1-001, T1-004), or crashes (T1-010, T1-011).
  - Several rule-downs are estimates only: PP=2, uneven TP, reduced-vocab head, dequant+cuBLAS.
  - Good practice: the k=3 drop was re-diagnosed from logs in the next round (KB13) instead of being left as "slower".
- **GPU time without information.**
  - The cancelled bs-r1 job used 11.8 GPU-min (ledger job 2), although it may have produced the NCCL microbench numbers quoted in the brainstorm.
  - T1-010 used three OOM startups (5.9 GPU-min). That is cheap and informative (KB16).
  - Overall the GPU use was efficient: 177 GPU-min for +18.7% measured.
  - The run ended with 2 rounds of budget unused because of the human checkpoint stop.
- **Rule concerns.**
  - The manager read test-1/test-2's `task_definition.md`, `assumption_ledger.md` and `focus.md` at run start and used them as a structural template (`journal.md:20`, self-reported). This is an independence breach of structure only; no findings are mentioned.
  - The 6 distinct worker agents across 3 rounds don't by themselves violate "max 2 workers per leader". I did not check whether more than 2 ran at once.
  - The custom-logits-processor check `len(...)==3` in T1-009 is fragile (`bank/T1-009/record.md` validity). It is a code-quality note, not a violation.

## Key updates (proposed)
1. **K5:** add the A/B and microbench evidence: same-job A/B +2.2% `ab`, 712.4 vs 697.3 tok/s, TPOT p50 19.57 vs 20.41 ms (`v1/test-3:T1-012`). Note that in test-3 it was not promoted only because of the 3% bar. Keep the frontier as is (749.7 is official).
2. **D1:** add T1-010/T1-011 (`v1/test-3`) as evidence that the "more KV" remedy fails (it is already under F6/KB16). Mention them under D1's "Revisit when" so that "raise --kv-cache-memory" is not counted as a fresh idea.
3. **Marlin in-kernel (O bullet):** it now has a component measurement: the M=48 in-kernel penalty vs M=3 is about 1.6 ms/step, and the rest is GDN spill (`v1/test-3:KB14`; `ideas.md:41`). Suggest calling it "small upside by microbench" instead of "untested".
4. **New D (estimate only):** "dequantize + cuBLAS for prefill GEMMs" at 2% or less (`v1/test-3` brainstorm r2 adjacent_field:75,165). Also "reduced-vocab draft head", net about 0 once the draft head is FP8 (`v1/test-3` ideas R1-J, r3 rule-down). Both are weak evidence.
5. **New O (low):** copy-engine full-duplex AR for large M (prefill), est. +1.5-3% (`v1/test-3` idea R2-E). Also the untested `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` check (`brainstorm/r1/systematic_engineer.md:99`).
6. **O1:** note the narrower test-3 variant, R1-H (in-place FlashInfer draft-metadata update, about 1 ms of CPU gaps; `brainstorm/r1/systematic_engineer.md:176`). Also note the r2 counter-claim that the decode step-start gap is a `cudaGraphLaunch` nsys artifact (`brainstorm/r2/adjacent_field.md:68,164`), which is relevant to O1's 1.6 ms CPU-idle estimate.
7. **D5:** note that test-3 kept the all-gather as a neutral part of the champion stack (no gain, no harm, exact). This is not a contradiction.
