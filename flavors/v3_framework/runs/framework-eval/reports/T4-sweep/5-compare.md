# T4 sweep, step 5: my map vs the other teams' maps and the run's history

Read: reports/T{1,2,3}-writeup.md, T2-sweep/update-1..3, T3-sweep/3-calibration, 6-ranking, update-3, T2/T3 resyncs,
bank T2-001, T3-014 (and grep over all records). shared_kb.md is empty.

## 1. My candidates, marked
| My candidate (4-knobs) | Status | By whom, on what code | Comment |
|---|---|---|---|
| W: CUDA graphs past 512 tokens (76 prompts with P+45 > 512 run eager) | **tested in part, on old code**: T2-001 tried sizes up to 1024 with 83 sizes -> OOM during capture (T2 job 15, Marlin era); T2 update-1 then judged ">512-row steps already GPU-bound (231 ms at 763 rows vs 233 graphed estimate)" from a linear model 38 ms + 0.25 ms/row fitted on **Marlin** (BASE). T1-015 (row-split for eager >=512 steps) official -0.7% | My data on the champion: graphed 256-512-row steps take ~80-100 ms (~0.16 ms/row with W8A8), eager 700-900-row steps ~210 ms with 30-60 ms GPU idle (graph trace). The "already GPU-bound" verdict rests on the Marlin-era per-row slope; with W8A8 a 760-row step should be ~150 ms graphed. **Disagree; re-test with few sizes (8, not 83) on the champion: E1 running.** |
| W: small mixed steps CPU-bound in the eager GDN core | **disputed**: T2 update-3 "mixed-step idle: only ~2.5 ms of real gaps per mixed step (0.9% of wall); the ~9 ms step-window attribution is mostly async launch lag" (gaps.py on a T3-006 nsys). T2-001 (BASE) measured without nsys: graphed mixed step = 38 ms + 0.255 ms/row | My graph-trace profile of T3-009 shows ~22 ms GPU idle per small mixed step and ~1.4 ms wall vs ~0.2 ms GPU per eager GDN layer. nsys inflates eager launches (PROFILING.md), so neither of us has ground truth on current code. **Measuring with T2's non-nsys per-step logger (ws t4-steplog) inside E1.** If T2 is right, this candidate drops to <1%. |
| Rewrite: decode W8A8 GEMM efficiency / prefetch effectiveness | **tested**: T2-010 (decode-M tiles, split-K: "configs equal a plain streaming read, 850-890 GB/s achievable"), T3-013 (prefetch-aware configs: null), T3-012 (prefetch 56 MB: null). Current code | Agree: my 0.87 TB/s "inefficiency" was against a 1.0 TB/s spec; achievable is ~0.87 (T2). **Dropped.** |
| Speculate: 3 drafts | **tested 3x**: T1-003/T1-010 (-3.6%, -5.8%), T2-013 on current code (-11.7% screen, TTFT p99 gate broken). k=1 -9.2% (T1-016) | Agree: dead. My napkin (+3%) underpriced the verify rows: the ridge is ~80-165 rows, so 64 rows are not free, and DeltaLog caps at 2 logged rows. |
| Split: hostmem AR for the 1.4 MB row-split halves | **tested**: T3-001 hostmem for all sizes -6.6%; T3 job 74: hostmem 128 us vs NCCL 132 at 1.3 MB (equal), and hostmem spins SMs, which hurts overlap | Agree: dead. |
| Keep: ReplaySSM depth L | **tested as a ceiling**: T3-007 (DeltaLog L=2 ceiling ~0.33 ms/step) | Agree: <1%, not worth it. |
| Rewrite: lm_head / draft head BW | T2: target lm_head and decode GEMMs are at the achievable floor (~0.85 TB/s) | Agree: dropped. |
| Merge prefills (policy) | untried | Only matters if the small-mixed-step overhead is real. |
| PP / DP / disaggregation / RMSNorm distributivity / fp16 accumulation | dropped in every map (fp16 accumulation: T3 showed overflow, as I did) | Agree. |

## 2. What my sweep found that the other maps lack
1. **The prompt-length distribution of the actual benchmark as an input.** No other map priced mixed steps per prompt
   length bucket from the 400 real prompts (232 < 211 tokens, 92 mid, 76 >= 467). It is a workload dimension the
   framework's "snapshot" doesn't carry (a distribution over snapshots, not one snapshot), and it is what sizes the two
   mixed-step candidates.
2. **The eager >512 bucket on current code.** The other maps dropped it on a Marlin-era slope. W8A8 (T2-006) changed
   the per-row cost of mixed steps by ~1/3, which re-opens it. (A framework lesson the other teams also hit: verdicts are
   conditional on the code they were priced on.)
3. **The breakable-graph structure of a mixed step**: per GDN layer, one graph segment then an eager GDN core of ~30
   launches. Whether its CPU time is exposed depends on how much GPU work the step has (small P: exposed; P >= ~200:
   hidden). T2 measured gaps on aggregate; the dependence on P is what my calibration adds.

## 3. What they have that my sweep missed
1. **Achievable peaks**, measured: DRAM ~850-890 GB/s, SM-issued PCIe duplex ~31-34 GB/s total, FP8 block-scaled MMA
   ~275-285 TF/s realistic. I used spec 1.0 TB/s and saw a "15% GEMM inefficiency" that isn't there.
2. **L2 carry-over between nodes** (T2 DeltaLog, T3 prefetch): writes drain into the next kernel, and an idle-HBM
   window can pre-pay the next node. My per-node byte pricing would have mis-ranked DeltaLog.
3. **Verify slack is not free** at a ridge of 80-165 rows (k=3 dead). My napkin assumed 64 verify rows were ~free.
4. **Interactions through shared gates** (fused norm+quant silently disabled row-split). I will check feature log
   lines in every A/B server.

## 4. Where I disagree
- ">512-row steps already GPU-bound, graphing gains nothing" (T2 update-1, T3 5-compare agreed). It was computed with a
  Marlin per-row slope (0.25 ms/row) before W8A8 (0.16 ms/row in my data), and the only graph attempt used 83 sizes and
  OOMed. Testable cheaply (E1).
- "Mixed-step idle ~0.9% of wall" (T2 update-3): it may be right for the aggregate of mid and large steps, where the GPU
  hides the CPU. For the 232 small prompts my trace says otherwise. The non-nsys step log settles it.
