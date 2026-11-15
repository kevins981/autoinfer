# T2 map, update 4

HEAD = T2-016 (unofficial, +3.53% over T1-015). CONFIRMED = T1-012 (785.3), whose chain includes T2-011 and T2-012.

## Tested since update-3
| id | record | result | verdict |
|---|---|---|---|
| KR1 GDN single-write state | T2-015 | PASS +2.44% vs T1-012, then **reverted**: greedy output differed at the last token at max_model_len (c=1 OFF/ON/OFF, job 104) | design right, integration bug |
| KR1 fixed | **T2-016** | root cause (job 114): the runner sets num_draft_tokens_per_req=None when no row has drafts, which skipped the replay decision, so pending rows were dropped. Fixed; c=1 gate 16/16 bitwise (jobs 104+116); PASS +3.53% vs T1-015, GSM8K 93.2%, 128K pass | **won (HEAD)** |
| Re-profile of T1-012+KR1 | t2-w1 job 101 | decode step 35.8 -> 30.6 ms; GDN 3.68 -> 2.04 ms; found attention 16 -> 77 us/call after T1-010 (split-KV off) | handed to T1, who fixed it as T1-015 (+1.29%) |

## Verdicts changed / lessons
- KR1's napkin (~4%) held up: +2.44% then +3.53% on a faster base, since the GDN share of the step grew as other items shrank.
- Exactness testing (framework/test lessons):
  - a shadow check that feeds both kernels the same input state cannot see stale-state bugs;
  - compare logprobs, not just token ids;
  - unit tests that generate their own control flags cannot find bugs in how the server derives them;
  - run-to-run determinism needs c=1 (at c=16, timing-dependent batch shapes flip near-ties);
  - the decisive test was OFF/ON/OFF at c=1 against a length cap.
- A profile that compares the current HEAD against an old one finds regressions in other teams' changes (the T1-010 split-KV). That's worth doing after every few promotions.
- Drafter quantization: the MTP layer's GEMMs are already FP8 in the checkpoint (only norms/gates/fc are bf16), so it is dropped.

## Current ranking (napkin vs HEAD T2-016)
1. KE1b new decode FP8 W8A16 GEMM: Marlin is 57% of the decode step at ~0.80 of HBM peak; ~2 ms/step (~4%) at 0.92 TB/s. Prototype on gate_up with a go/no-go gate (task 15).
2. Items owned by T1 (all-reduce 5.7 ms/step; mixed ≤512 all-reduce +4.3 ms and small torch ops +1.7 ms since T1-005, per job 101): flag them, don't build.
3. GDN kernel 38 us/layer: near the read+write floor after KR1; no further exact saving seen.
