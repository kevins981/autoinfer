# T5 sweep, step 5: my map against the other maps and the history

Read after files 1-4: shared_kb.md, T1/T2/T3 write-ups, T3-sweep/update-3, T4-sweep (1-6), bank records
T1-007, T2-014, T3-001, T3-003 (and the write-ups' record tables).

## 1. Status of my candidates (4-knobs.md summary list)
| # | my candidate | status | evidence / code it was tested on |
|---|---|---|---|
| 1 | mixed-step CPU overhead (full graph / leaner eager) | **partly built, rest open** | T2-004 piecewise prefill graphs (+4.49%, in HEAD); T2-014 FLA fast-launch: CPU 0.30 -> 0.205 ms per GDN prefill layer, **null end-to-end** (on T2-012). Full graph of mixed steps untried (T3 open #2). See §3: my 70 ms/step estimate was wrong. |
| 2 | pipelined host all-reduce (write/read overlap) + extend to prefill sizes | **tested negative** | T1 jobs 40-46, T3-001 (jobs 159/161/164: ~10 us + 78 us/MB, flat in grid 4..32, pipelined variant no better), T3-006 copy engines no duplex. On this host, PCIe up + down share ~25.6 GB/s: my "Link up and Link down are separate resources" was wrong. |
| 3 | GEMM + all-reduce overlap (decode) | untested as a record; related negatives | T2-005..007 (L2 prefetch during host AR: negative, "link and HBM coupled"), T1-007/009 (AR+norm fusion null/neg). T4 lists it as fallback #5. |
| 4 | merge prefills (admission batching) | built, withdrawn | T3-002: +1.12% on ShareGPT but N picked on benchmark data; N=1 best on generic data. |
| 5 | no blocking sync in mixed steps | open, small | T1-010 did it for decode only. The measured post-sync bubble is ~1-2 ms per mixed step (job 201 graph trace: sync returns at 53.4 ms, first kernel 55.8 ms, under nsys): ≤ 0.5% now. Demoted. |
| 6 | decode W8A16 efficiency | tested to the limit of the kernel family | T1-020 (noise), T2-017, T2-013 (Marlin/Humming at the read ceiling, 0.77-0.90 TB/s); open only as a new kernel design (T4 fallback #3). |
| 7 | in_proj_ba kernel | **tested null** | T3-005: it runs on an aux stream under qkvz, off the critical path. My per-kernel sum double-counted it (my error: I priced an overlapped kernel). |
| 8 | 3 drafts | tested negative 4 times | T1-003/008/021, T2-008 (acceptance +15%, step +28%). |
| 9 | lazier GDN commit (K=3), lm_head kernel, small fusions | K=3 negative (T2-019, -0.59%); lm_head addmm negative (T1-014), Triton GEMV untried (T4 #4); fusions T1-007/009 null | |

## 2. Corrections to my files 1-4 (I don't edit them; corrections here)
- **Mixed-step cost (3-calibration §1, §3; 4-knobs §6).** I attributed ~70 ms extra per mixed step and 41% of wall to CPU. Two errors: (a) NVTX periods are start-to-start of `execute_model`; a mixed step's range contains a blocking wait for the *previous* decode step's GPU work (~20 ms), which then makes the next decode period short (job 201: 5 ms "next decode" periods after mixed steps). So the mixed-step extra is ~20 ms smaller than my binning. (b) The GPU is mostly busy in mixed steps (W8A8 ~22-31 ms + NCCL ~29 ms + a decode-shaped forward, T4 job 199 and my job 201 node trace: busy 92.8 of 97 ms in one window). T3's no-nsys client-timestamp measurement (job 198: +28 ms fixed + 0.17 ms/prompt token) is the better number. CPU-bound part: ~10 ms per mixed step (T2-014: 0.205 ms x 48 GDN layers), mostly overlapped.
- **Host all-reduce napkin (3-calibration §2).** The "pipelined" floor of 2.6 ms does not exist on this host: the serialized ~10 us + 78 us/MB (T3-001) is the floor, i.e. 135 x 48 us ≈ 5.7 ms measured = floor.
- **in_proj_ba** is overlapped (aux stream): 0 gain.
- FP8 fp32-acc peak: I wrote 330 TF "check"; T4 measured ≥ 182 TF achieved, consistent with 330.

## 3. What my sweep found that the other maps lack
1. **The link's bytes are a representation choice (lossless encoding of the all-reduce payload).** All maps treat AR bytes as fixed (M x 5120 x 2 B) once lossy compression is ruled out. But the payload is bf16 partial sums, whose sign/mantissa are incompressible while the **exponent byte is highly predictable** within a block of neighbouring channels (entropy ~2.5-3 bits of 8). A block code (per 32 values: max exponent; per value: sign, 4-bit exponent offset, 7-bit mantissa; raw fallback for a block with any value more than 14 binades below the block max, zeros/subnormals handled) is **bit-exact**: the peer reconstructs the identical bf16 tensor, the sum is computed exactly as now, outputs are bitwise those of HEAD. 16 -> ~12.3 bits per value = **~23% fewer link bytes**. Since the link time is ~10 us + 78 us/MB (bandwidth part ~80% of decode AR time), this is the only lever left on communication that doesn't need overlap (which failed 6 times). The framework's Rewrite knob covers representation changes only as lossy `approx` rows; a lossless re-encoding is a Rewrite of a data node's *format*, priced in Link_bytes, with p = 1 (no accuracy gate). Nobody listed it.
2. Per-kind step timing by NVTX-only nsys (job 202) as a cheap representative run (843.7 tok/s vs 863-874 unprofiled) — but see §2(a) for how not to bin it.

## 4. What they have that my sweep missed
- PCIe through the host is effectively **one** ~25.6 GB/s resource for up+down on this machine (T1/T3), and it is coupled with HBM traffic (T2-005..007). My graph split them and priced overlap that doesn't exist.
- Overlapped kernels must be excluded before pricing (T3-005) — I priced in_proj_ba from kernel sums.
- Pushing argmax before the collective (T3-004), single state write + lazy commit (T2-016/018), weight layout as a shared property (T1-017): all in HEAD; my 2-graph listed them as present, not as found.
- Layer-level two-micro-batch overlap in mixed steps (T4 P4): my 4-knobs pruned dual-batch overlap for decode correctly but treated prefill only as HEAD's GEMM-level chunking; T4's layer-level form is distinct and is T4's current work.

## 5. Where I disagree
- None of the earlier verdicts I checked look wrong on current code. One nuance: the AR "floor" (78 us/MB) is a floor on **bytes moved**, not on the tensor's information content; the compressed AR goes below it without overlap.
