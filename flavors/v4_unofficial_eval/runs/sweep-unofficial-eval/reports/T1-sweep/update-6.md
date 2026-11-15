# T1 map update 6 (closing; HEAD = T3-004; CONFIRMED = T1-019 at 897.5 official)

## Tested since update-5
| candidate | record / job | result | verdict |
|---|---|---|---|
| Tuned W8A8 configs + W8A8 from M=80 | T1-018 | -6.10% (1 start), TTFT p99 2559: runtime Triton JIT (M%16 specialisation, unwarmed configs) | lesson: warm every config x specialisation through the real path |
| Same + qkvz/qkv on checkpoint layout + full warm-up | T1-019; FP8 A/B job 182 (93.2 -> 93.6); 0 runtime compiles (jobs 166-179) | PASS +2.12%; **confirmed 897.5 official** | **won** |
| W8A8 at decode M (16-64) | job 184 | gate_up 1.28x at M=48; others tie | closed: keep W8A8 from M=80 |
| W8A16 kernel bandwidth tuning (wave fit, split-K, BK=256, eviction) | jobs 187-192; T1-020 | 0.77-0.90 TB/s ceiling for this kernel family; -0.11 ms/step op-level; unofficial mixed -0.31% | parked (below eval resolution) |
| 3 MTP drafts on current code (sparring T1-3) | job 196 (M=64 tuning -0.23 ms only; M=64 is compute/dequant-bound); T1-021 | acceptance 2.673 (+15% as predicted) but -10.24%, TTFT p99 fails | closed (step cost +~28%, napkin said +18%) |

## Framework notes (final)
- Weight layout is a data-node property shared by every kernel that reads it. Pricing kernels one at a time hid the biggest
  late win (T1-017/T1-019: +5.2% unofficial combined).
- The host CPU belongs in the graph as a device with sync edges (T1-010).
- Every trade has a cost side; record it as a new candidate (T1-010 -> T1-015).
- Runtime JIT is a real cost node for Triton-heavy designs: "compile" must be a priced, warmed node, not an implementation detail.
- Per-step floors are not per-token floors: spec-decode depth changes M, and kernel cost vs M was not flat (compute/dequant
  bound from M~64 on this card). That ruled out k=3 here, with the evidence now on current code.

## Final ranking (for whoever continues)
1. A W8A16 decode GEMM above 0.9 TB/s needs a new kernel design (a persistent/warp-specialised pipeline; no TMA on sm_89).
   Prize ~1-1.5 ms/step (~3-4% of wall); high effort and risk.
2. lm_head bf16 GEMV toward 0.92 TB/s: ~0.3 ms/step; the addmm route lost in situ (T1-014), a Triton GEMV is untested in situ.
3. Drafter GEMMs: ~0.3 ms/step.
Everything else in my map is built, measured negative, or at a measured floor (host AR, NCCL tuning, chunking, k=3).
