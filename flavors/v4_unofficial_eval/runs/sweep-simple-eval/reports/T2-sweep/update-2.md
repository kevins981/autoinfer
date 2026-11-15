# T2 map update 2 (resync 2)

## Tested since update-1
| id | what | result | evidence |
|---|---|---|---|
| R5 | reduced-vocab draft head, token ids [0,N) split across ranks + local draft argmax | N=98K with KV 4 GiB: 646.6 after 1 start (T2-003) — **cause: the 4 GiB KV pool (1,600-token blocks sized to the GDN state page) admits only 15 running requests**, not the head (decode step -1.4 ms at equal rows in profile). N=64K, KV 5 GiB, T2-001's 2048-token graphs: OOM during the benchmark. N=64K + max-num-batched-tokens 1024 + graphs to 1024: **660.0 / 664.2, promoted (T2-004)**, acceptance 2.351 -> 2.312, confirmed officially via T1-006 | T2-003, T2-004, profile agents/leader-2/prof_rv |
| R6 | W8A8 (checkpoint scheme) via vLLM's humming backend, all FP8 linears, one fp8 weight layout for every M | GEMM microbench (job 24): = Marlin at M=48, -33% at M=260, -42% at M=2048; server: **715.7 / 720.2 on T2-004 (T2-005, promoted)**, GSM8K 93.2%; on T1-006: **738.8 / 747.8 (T2-006, candidate)** | T2-005, T2-006, agents/leader-2.w1/humming_gemm.md |
| R4 | FP8 draft lm_head copy | 645.3 after 1 start with KV 4 GiB (same 15-slot cause, T2-002); superseded by R5 | T2-002 |
| W4 | NCCL_PROTO=Simple in the engine | no effect (T1 job 33; the default already matches) — standalone LL/Simple gap from my worker's table does not transfer | T1-002 (reverted) |
| W3/W-DBO | AR overlapped with a GEMM on a second stream | hides only ~50% (NCCL takes SMs), half GEMMs 25% less efficient: ~0 net | T1-004 |

## New dimension / framework notes
- **Capacity C is quantized by the hybrid page size**: with GDN state pages, KV blocks are 1,600 tokens and the number of
  concurrently admitted requests is a step function of the KV budget; a 20% KV cut lost 1 of 16 requests (-6%). The
  framework's capacity check `resident bytes <= C` needs the allocator's granularity to see this.
- **Memory headroom is a shared, scarce resource** in this config: any +0.3-0.6 GB (draft head copies, wider graphs)
  competes with the 2048-token activation peak. Trading max-num-batched-tokens 2048 -> 1024 bought ~0.3 GB.
- Prefill FP8 compute: settled at ~315 TFLOP/s (fp32 accumulate); the SOL file's 165 is too low.

## Current ranking (vs T2-006, ~743 tok/s; time split will have shifted toward decode: needs a fresh profile)
1. **S1 MTP k=3 again** on T2-006: draft steps now read 0.33 GB instead of 1.27 GB and have no logits gather; T1-001 (on BASE)
   had acceptance 2.70 (+15%) but step +22%; the third draft step now costs ~1 ms less. Cheapest test: serve flag
   (num_speculative_tokens 3) + 128K. Riskiest: GDN verify and AR growth at 64 rows, memory headroom.
2. **K1 GDN rollback-free state commit** (2-3%): kernel change; first ncu of gdn_decode_post_conv_mtp_kernel DRAM writes.
3. **humming tuning for sm89 prefill shapes** (no tuned configs; heuristic tiles): microbench a few tile configs at M=260-1024.
4. **W1b** graph-capture the eager GDN chunked-prefill core in mixed steps (host time); price with a TRACE=graph profile.
5. Draft vocab N: 64K vs 48K/80K trade (acceptance vs bytes), small.
