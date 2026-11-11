# T29 write-up (leader-29, explore, slot B): checkpoint W8A8 FP8 GEMMs for mixed/prefill steps

## Summary
- **Champions:** two.

  | record | official tok/s | gain |
  |---|---|---|
  | T29-004 | 1045.8 | +8.1% over T27-002's 967.3 |
  | T29-006 | 1062.1 | +1.6% more |

  Together that is **+9.8% over T27-002**, and +67.0% vs BASE.
- **GPU:** ~96 of 180 GPU-min used.
- **Records:** T29-001..006.
- **Workers:**
  - leader-29.w1: integration, records, the fused producers, and the capture/compare tooling.
  - leader-29.w2: the GEMM kernel, the π layout, the GDN glue trim, the attribution, and the host probe.
- **Notes:** local_kb/leader-29.md.
- **Proposal:** proposals/T29-1.md. It forecast +12.5% if everything worked, with a central estimate of +5%. Outcome: +9.8% from stages A and B plus parts of C/D.

## What worked
1. **The FP8 GEMM reads Marlin's layout directly. The π trick is in KB87 and agents/leader-29.w2/gemm/PI_SPEC.md.**
   - Take two consecutive Marlin k16 FP8 tiles. Each lane's 8 bytes per n are exactly an e4m3 m16n8k32 B fragment, up to a fixed in-32 k-permutation PI=[0,8,1,9,...,7,15](+16).
   - The activation quant writes its columns PI-permuted. The 128-groups and scales are unchanged.
   - Consequences:
     - Only one weight copy is needed: Marlin's.
     - Decode (M<=96) stays bitwise on the champion's Marlin.
     - M>96 runs W8A8.
   - Verification: the emulated repack is bitwise equal to the real gptq_marlin_repack (job 202). T28 confirmed the derivation independently.
   - Kernel: agents/leader-29.w2/gemm/bs_gemm.cu, vendored as t29_bs_gemm.
     - cp.async multistage pipeline.
     - The e4m3 MMA accumulates into a per-128-K register tile. That tile is then fmaf-promoted with As*Bs into fp32.
     - No PRMT.
   - Per forward on GPU0, quant included: 21-41% below Marlin at M=96..1024.
2. **Dispatch only at M>96 (T29-003/004).**
   - Same-job ABBA: +9.2% (job 207).
   - Official: 1045.8.
   - GSM8K-1319 is +0.61 pt vs the bf16 reference (job 214).
3. **An exact bundle on top (T29-006), official +1.6%:**
   - **v2 tiling table.** The smem-staged coalesced epilogue gives 1.6-3.4% at M>=192. It is bitwise on live activations.
   - **Fused producers.**
     - Gemma add+RMSNorm+quant is a CUDA kernel that mirrors flashinfer's CuTe norm exactly: fp32 unrounded h, b-outer/v-inner fma reduction, IEEE sum/H, rsqrt.approx. Triton cannot pin that reduction order. It is used only at M>=256, because below that it is slower than norm+quant.
     - silu_mul+quant is in Triton.
     - Both are bitwise on 147 real served-activation dumps.
   - **GDN glue trim in mixed steps.** Slices and out= buffers replace 6 index_selects, 2 index_copies and a dead rearrange. That takes the glue from 16 to 4 launches per layer. The self-check passes 192/192.

## The mixed-step floor map (after T29-004; w2 attribution, agents/leader-29.w2/icpt/NOTES.md)
- **Setup.** Small mixed step: pt 1-80, M ≈ 112. GPU time ~41.5 ms vs 23.6 for a decode step. The probe (job 219) measures a GPU span of 44.3 ms.
- **The excess over decode:**

  | component | excess, ms | floor |
  |---|---|---|
  | GEMM | +7.4 | cuBLASLt speed minus the per-128-K promotion the rule requires; ncu at gate_up M=512: tensor pipe 40.5% vs cuBLASLt 45.2%, both math-throttle bound |
  | host all-reduce | +5.2 | ~0.56 us per row (8 KB at ~14 GB/s, PCIe bandwidth); already lossless-compressed; not hideable at R<256 (KB65/67) |
  | torch glue | +2.2 | mostly removed in T29-006 |
  | fla chunked GDN | +1.7 | |
  | quant | +0.8 | |
  | conv/state | +0.6 | |
  | attention | +0.5 | |
  | norms | +0.2 | |

- **Mid steps (pt 81-208):** GEMM +14.2 ms, AR +9.9 ms.
- **Interpretation:**
  - Mixed steps scale with rows through GEMM and AR. Both are at their floors.
  - What remains exactly removable is each <0.5% of a trial:
    - fusing the 6 fla kernels (~0.8 ms/step);
    - the GDN initial-state gather (~10 us/layer);
    - the norm fusion at M<256, if the kernel gets faster there.
- **Host side** (job 219, untraced): small steps are GPU-bound.
  - Host enqueue runs 63-65 ms per small step, hidden by a ~22 ms decode-step backlog with ~4 ms to spare (KB89).
  - The glue trim widened that margin by ~5-8 ms.
  - Any future cut of more than ~4-10 ms in small-step GPU time must also trim the host (the eager GDN core op is ~1.1 ms of host time per layer).

## In-situ transfer (what generalises)
- **Mixed steps:**

  | path | microbench predicted | mixed-step ITL | transfer |
  |---|---|---|---|
  | Triton all-M | 7.4 s/trial | -13% | ~50% |
  | Marlin-layout | 7.4 s/trial | -23% | ~75% |

- **Exact bundle:** the per-kernel napkin said ~2-3%; same-job +1.09%; official +1.6%.
- **Lesson:** a standalone cold microbench overstates in-server mixed-step gains by 1.3-2x. The decode-step cost of a change, and the GPU backlog, decide the rest.
- **Profiling caveat:** a TRACE=node nsys inflates host time per eager launch enough to fake "host-bound" small steps (job 199 vs probe job 219). Confirm untraced.

## Nulls and their causes
- **T29-001/002: Triton W8A8 at all M on the standard layout. +0.2%, a wash (KB88).**
  - Decode lost +0.93 ms/step (nsys job 199):
    - 264 standalone quant launches: +0.73 ms;
    - the cuBLAS bf16 in_proj_ba GEMMs ran 2-3x longer: +0.44 ms, cause unknown;
    - AR: +0.18 ms;
    - against the FP8 GEMMs saving 0.6 ms.
  - Also: Triton specialises on M%16, so a warmup must cover odd M.
- **T29-005: chunk-order events in the token split.**
  - It is bitwise (164/164) but within noise: -0.08%; mixed-like time -1.8%.
  - The large split steps are a small share of a trial.
  - W8A8 does remove the free stagger that Marlin's full-SM GEMMs gave.

## Assets for later teams
- **T29_W8A8** (default 2 in the champion):
  - 0 = Marlin everywhere (the bf16-activation reference);
  - 1 = the Triton all-M mode, kept but not recommended.
- **Custom op:** torch.ops.vllm.t29_marlin_fp8_linear (t29_marlin_w8a8.py). M<=96 runs exact Marlin; M>96 runs quant_perm + bs_gemm_marlin. The M<=96 branch is the place to plug in a small-M FP8 kernel (T28's Marlin-e4m3 lives there in t28-e4).
- **Fused producers:**
  - T29_FUSED (default 1);
  - T29_FUSED_NORM, T29_FUSED_NORM_MIN_M (default 256);
  - T29_FUSED_ALL_M (emits π-quant at every M, for an FP8 decode path; T28 used the silu-only variant).
- **GDN glue trim:** T29_GLUE (default 1). T29_GLUE_CHECK=<n> runs an in-server old-vs-new bitwise self-check.
- **Validation tooling:**
  - T29_CAPTURE=<dir> with --enforce-eager dumps real producer inputs and outputs from a served run;
  - agents/leader-29.w1/fp/cmp.py does the offline bitwise compare. This is the evidence path for the bitwise-reuse ruling.
- **Host-vs-GPU probe:** T29_HOSTPROBE (workspace t29-probe, agents/leader-29.w2/probe/). It reports untraced per-step host/GPU timing and the backlog tail.
- **GEMM microbench harness:** agents/leader-29.w1/mb and agents/leader-29.w2/gemm. Cold READ flush, CUDA-graph reps, Marlin / Triton / cuBLASLt references.
- **Process:** run `gpu run` as a background command with a long shell timeout; job 212 was killed by the default 30-min shell limit (KB90).

## Open
- **FP8 at decode (M<=96).** The W8A8 GEMM beats Marlin warm, but in situ the quant launches and cold weight streaming eat the gain (T28-005). It might pay with T29_FUSED_ALL_M (silu) plus a warm-path kernel. This is T28's line; their assets are e4m3 Marlin and the two-slot GDN.
- **N-split pipelining of row-parallel GEMM + AR at small mixed M: dead on paper** (w2, CPU only).
  - Measured inputs:
    - AR ≈ 15 µs + 0.56 µs/row: 78 µs at M≈107, 118 µs at M≈163 (job 199);
    - the FP8 down GEMM: 82-92 µs; out_proj/o_proj: 36-38 µs, at ~55-60% of the HBM roof (job 209).
  - Halving N halves the CTAs: about +15 µs (down) and +7 µs (out_proj) per split GEMM, plus 3-10 µs per extra launch.
  - Best case with no interference: ≈ -1.5 ms per small step, ≈ 0.3% of a trial.
  - With KB62/KB70 interference (AR x1.3-1.9 beside a GEMM), the expected value is ≈ 0.
  - It would also take the hostar L2-prefetch window away from the next GEMM.
  - The AR at this size is only movable with fewer bytes or a P2P path, which this machine doesn't have.
- **Small exact items, each <0.5%:**
  - the fla kernel fusion;
  - the GDN initial-state gather;
  - a faster fused norm at M<256;
  - split-K for down at M<=128, which needs per-chunk workspaces because of the concurrent split streams.
