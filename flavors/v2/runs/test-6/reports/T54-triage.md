# T54 triage of research_handoff_1113.md (leader-54, 0 GPU-min)

Workload reminder (TASK.md): ShareGPT, ~260 prompt + ~228 output tokens, 16 concurrent, TP=2, k=3 MTP.
Champion T52-005 1212.4 tok/s. Median decode step (T51 nsys of T45-005, 64 rows; agents/leader-53/anat/, re-aggregated
by me with show.py): 25.1 ms = hostar 7.0, t31b GEMMs 6.8, bs_gemm 5.3, **GDN replay 1.85 (38.5 us/layer)**, target
lm_head 1.03, Marlin 0.54, draft head ~0.27, attention 0.31, conv 0.20, small ops rest.

## Fact check of the memo's inputs (config.json of Qwen/Qwen3.8-27B-FP8, text_config)
| Memo claim | Actual | Consequence |
|---|---|---|
| Full attn 40 Q / 8 KV heads | 24 Q / 4 KV heads (head_dim 256) | KV is 32 KB/token, not 64: KV offload premises halve again |
| GDN state 16-bit, 75.5 MB/req | `mamba_ssm_dtype: float32`: 151 MB/req (75.5 MB per GPU at TP=2) | bf16 state would be lossy vs the checkpoint's fp32: not allowed |
| Ridge ~330 tokens/weight load (660 TF FP8) | exact path is bf16 MMA or e4m3 with fp32 promotion, ~146-165 TF: ridge ~80 rows; measured: rows above 64 cost a step (KB118, KB120, T51) | every "verify M tokens for free" argument (Opt 2) is 4x too optimistic at c=16 |
| Weight read 13.5 ms/pass | ~12.2 GB/GPU, GEMMs ~12 ms + 7 ms AR | roughly right |
| Decode is B=1-4 latency regime | benchmark is c=16, 64 rows/step at k=3 | Opt 2's regime is not this task's |

## One row per idea
| # | Idea | Verdict | Evidence | What would have to be true to pay here |
|---|---|---|---|---|
| 1 | DeltaLog Keep (checkpoint + ring of k,v,α,β; replay; flush every W) | **Already tested in this run (it is the champion's GDN path)** | GDN replay = write one state per step (T1-012, KB15, +2.6%); lazy checkpoint every 8 accepted tokens (T1-035, KB48, +1.0%); two-slot rollback-free state (T37-002, KB111); RED8 kernel (KB51/54). Remaining: 38.5 us/layer vs ~25 us fp32 state-read floor (16 req x 24 heads x 64 KB = 25 MB/layer/GPU) -> ~0.65 ms/step headroom (~1.5% e2e). L2 state prefetch in the AR window costed <=1% (T12, T22: warm 23.4 vs 37.5 us not reproduced by prefetch). | The state read itself would have to shrink exactly (fp32 is fixed; lossless codec gave 0, T1-039) or GDN would have to be nearly free: even GDN = 0 is ~2.05/25 ms of decode = ~5.5% e2e ceiling, unreachable since the 1.2 ms read is a floor. |
| 2 | Dual-form tree speculation for GDN | **Wrong workload + already tested** | T38-002/003 exact offline tree acceptance (agents/leader-38.w1/tree/full_cpu_analysis.txt): at <=7 nodes the best static tree IS a chain; dynamic N=3 only +0.044 tokens over chain k=3. Trees pay only at N>=11 (4.21 tok/step) = 192 rows/step at c=16; rows above 64 cost ~0.14 ms each in TP (GEMM 0.07 + AR 0.07, T51) plus ~0.83 ms per extra draft level (KB111) -> step ~25 -> ~45 ms for +55% tokens: loss. PP+trees: T38 (decode 0.91-0.92x, needed 0.93), T49 sim. The ancestor-masked WY verification algebra was already noted by T38 (unbuilt). | Verify rows would need to be ~free up to ~200 rows/step (ridge 330 premise): false here; or c<=4 workload. |
| 3A | 3:1 strided PCIe KV prefetch (KV in DDR4) | **Wrong workload** | KV per request ~500 tokens x 32 KB = 16 MB; all 16 fit in the 5 GiB pool; 128K gate request = 4 GB, fits and passes. Nothing to offload. Host tier for freeing VRAM: T46 (~1% ceiling, KB125). | Memory would have to bind throughput (it did at k=3 pre-T37, fixed by two-slot state). |
| 3B | CPU-side full attention for micro-batch R2 | **Wrong workload; key assumption wrong** | Attention is 0.31 ms/step; microbatch split costs 1.31-1.47x (T53-007); PCIe host path already carries the ARs (KB104). | Attention would have to be a large share of the step. |
| 4 | Head-start Quest page prefetch for 64-256K | **Wrong workload, and lossy** | Context <= ~1K tokens in the benchmark; approximate top-K attention violates exactness (task_definition). | Long-context workload and an exact variant. |
| 5 | Shared-prefix lazy state materialization | **Wrong workload** | ShareGPT requests are independent; shared prefix is the ~44-token chat template; prefix-state caching already tried: +0.5% (KB86). No fan-out/beam. | Best-of-N / agent fan-out traffic. |
| 6.1 | Min prefill chunk >= 512 for GDN | **Wrong workload / already true** | Prompts ~260 tokens, budget 2048: prompts are not chunked; decode windows don't write S per step (lazy checkpoint). | Long prompts chunked into small pieces. |
| 6.2 | TP=2 -> asymmetric PP=2 at B>=16 | **Already tested in this run** | T38-004/005 (KB112): PP prefill 0.51x TP, decode 0.65x TP without trees, 0.91-0.92 with trees; T49 sim (KB123): one-microbatch PP doesn't beat TP; T38's "PP prefill + TP decode" hybrid ~0 on 300-token prompts. | A PP decode as good as TP's (both GPUs' bandwidth on one microbatch without an AR): not found. |
| 7A | FR-Spec top-16K drafter vocab | **Already tested; key assumption wrong** | 98K subset +1.0% (T1-013, KB21); 32K list -8.3% acceptance, -7.3% tok/s (KB71); superseded by the exact PCA-screened draft lm_head (T27-002, KB84): ~0.27 ms/step left for 3 draft steps. "Last-token-only prefill lm_head" is already vLLM's behaviour. | Draft head would need to be a large cost: it is ~1% of the step now. |
| 7B | Output-gate head skipping (||g||inf < eps) | **Not allowed (approximate)** | task_definition: no approximations/skipped work that changes outputs. GDN gate is swish(z) (zero only at z=0); attention gate sigmoid never 0 in practice. | An exact zero-gate rate > a few % (implausible). |

## Summary
No idea in the memo has a credible >=5% path on this task. The memo's two central quantitative premises are off for
this system: (a) the GDN state is fp32 (checkpoint config), so its per-step read is a hard 1.2 GB/GPU floor and the
write-avoidance the memo proposes is already in the champion; (b) the exact-arithmetic ridge on the 4090 is ~80 rows,
not 330, and the benchmark already runs 64 rows per decode step, so extra verify rows (trees) are not free. Ideas 3-6
target long-context, low-batch or fan-out regimes that this benchmark does not exercise.
Best remaining memo-derived item: GDN replay latency (38.5 vs ~25 us/layer), ceiling ~1.5% e2e, already costed by T12/T22.

## Addendum: idea 2 recomputed with measured costs (after sparring T54-1)
Step at 64 rows = 25.1 ms (T51 nsys). Measured costs above 64 rows:
- GEMM: all-bs chain +68 us/layer from M=64 to 128, with a +42 us step from 64 to 80 (KB120).
- AR: ~0.5-0.875 us per row per AR, x ~141 ARs (KB32, T51 job 434).
- Draft level: ~0.83 ms each (KB111).
Acceptance is the oracle (T38 full_cpu_analysis.txt); chain k=3 = 2.73.
| Candidate | Rows | Step estimate | Tokens/step | Net vs chain k=3 |
|---|---|---|---|---|
| dynamic N=3 | 64 | 25.1 (same depth) | 2.78 (+1.6%) | +1.6% decode at most (~+1% e2e); needs tree attention/GDN/conv kernels |
| dynamic N~5 (interpolated) | 96 | 25.1 + 2.7 GEMM + 2.3 AR + 1.7 (2 more levels) = ~31.8 (+27%) | ~3.25 (+19%) | ~-6% |
| dynamic N=7 | 128 | 25.1 + 4.4 + 4.5 + 3.3 = ~37.3 (+49%) | 3.73 (+37%) | ~-8% |
| chain k=4 | 80 | KB127: break-even at 0.115 ms/row | 3.01 | ~0 to +1% (not mine; KB127) |
Under measured costs no tree size is a +5% candidate at c=16 in TP. Branching only beats depth from N>=7 on, which is where rows stop being cheap.
Sparring T54-1 standing: AR row cost supported; "PP below gate" supported within the evaluated schedule (0.91-0.92 vs 0.93). End-to-end PP was never measured. The reopening condition is a GEMM that holds its floor at M=128-160.
