# T3 map, update 2 (HEAD = T3-004)

## Results so far
| record | what | result |
|---|---|---|
| T3-004 | target local argmax for all-greedy batches (Rewrite: argmax commutes with the vocab split) | **PASS +2.11%, HEAD** |
| T3-002 | prefill admission batching (--prefill-schedule-interval) | withdrawn: N picked on benchmark data; on generic data N=1 best |
| T3-003 | chunked GEMM+AR inside captured mixed graphs | -0.72% |
| T3-001 | host AR grid / pipelining / mid sizes | null (serialized PCIe floor; NCCL ties at 256 rows) |

## What's left on my map (against the new code; worker profile job 188, 24 prompts, node trace: coarse)
| site | cost/step | why I don't pursue it |
|---|---|---|
| host AR (135/step) | ~5.7 ms | at the serialized PCIe floor (T3-001, T1 jobs 40-46); overlaps fail on this box (T2-005..007, T3-003) |
| W8A16/W8A8 GEMMs | ~15 ms | T1's kernels, actively tuned by T1 |
| GDN recurrence | 1.9 ms | at its byte floor for the keep set; K=3 measured negative |
| target lm_head | 1.58 vs 1.27 floor | T1-014 (addmm) lost end to end |
| GDN ba proj (bf16 GEMM + split-K reduce, 48/step) | 0.3-0.9 ms (profiles disagree) | a fused small GEMV might save ~0.15-0.4 ms (<1-1.5%); needs a clean profile first |
| SM2 decode GEMM || AR column split | napkin ~4% | every AR overlap has lost on this box: low confidence |

## Framework notes
- The win came from reading the decode timeline for transfers the graph doesn't need: a collective whose consumer only
  needs a reduction (argmax) can be replaced by reducing locally then combining the reduction's tag. In framework terms:
  Rewrite (associativity) of a reduction across a Split index — push the reduction before the communication.
  Both earlier maps listed it; neither built it.
- Policy-layer knobs (admission timing) are in the design space but the rules require choosing their parameters on
  non-benchmark data, where this one didn't pay.

## Addendum: clean HEAD profile (job ~193, agents/leader-3/prof_t3004_node) and the ba GEMM
GDN in_proj_ba looked like 0.88 ms/step but runs on an aux stream under in_proj_qkvz (T3-005, worker's overlap analysis):
off the critical path, cuBLAS already ~4 us isolated. Closed. Lesson: per-kernel time sums must exclude kernels overlapped
on other streams before they're priced as recoverable.
Remaining decode budget (GPU0, 27.7 ms period): W8A16/W8A8 GEMMs ~16.3 ms (T1), host AR 5.9 (floor), GDN 1.9 (floor),
lm_head 1.58 (T1-014 lost), drafter lm_head 0.63, small kernels ~1.5. Nothing on my map >= 1% that isn't T1's or at a floor.

## Addendum 2: sparring T3-1 and an unprofiled mixed-step measurement
Sparring (sparring/ T3-1) challenged "nothing >=1% left" and "mixed-step fixed cost can only fall via GEMMs".
- Matched NCCL ARs across GPUs (6,438 pairs): transfer ~266 us at ~270-350 rows = the serialized-PCIe rate; the apparent
  ~51 us start skew is not idle (the later rank was busy) and is partly a cross-GPU timestamp offset.
- Unprofiled step timing from client token timestamps (job 198, HEAD T3-004, 200 prompts, `--save-detailed`;
  agents/leader-3/detailed/): decode 28.2 ms; mixed steps +33 ms (<=100 prompt tokens), +49 (100-200), +73 (200-400),
  +122 (400-800), +175 (800+). Mixed steps = 14.3 s of 53.4 s. Model: **~28 ms fixed + ~0.17 ms/prompt token**
  (nsys had doubled the fixed part). Serialized AR alone ~0.10 ms/token -> communication is the biggest prefill term.
- Open item raised: full-duplex AR for prefill sizes via copy engines (napkin 2-4%). Go/no-go microbench in t3-w1 task 4.
