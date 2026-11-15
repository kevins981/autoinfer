# T4 map, update 1 (closing; run wrapping up). HEAD = T3-004

## Tested since 6-ranking
| candidate | record / job | result | verdict |
|---|---|---|---|
| R4 fp16-accumulate W8A8 prefill | ruling | not allowed (human, final) | dropped |
| P3 duplex/pipelined host AR | (T1 jobs 40-46, T3-001, T3-006) | already negative; my worker stopped before any GPU spend | dropped |
| P4 two-micro-batch overlap of mixed-step ARs, NCCL per half | job 203 (microbench, 8 real-shape layers, graph) | 0 overlap: S ≈ GEMM-only + AR-only at 304/608/1024 rows; also explains T3-003's loss | NCCL path dead |
| P4, host-AR instance per half | job 204 (3 interleaved rounds, both ranks) | vs HEAD's serial NCCL per 8-layer stack: 304 rows 9.84->8.40 ms, 400 12.96->11.23, 608 17.28->14.36, 1024 27.24->24.74; bitwise equal | **mechanism confirmed in microbench: ~11-23 ms per mixed step, ~4% wall napkin** |
| P4 in the engine (T4-001) | jobs 205-212 | built; eager + graph capture work; 128K pass; c=1 greedy 5/8 prompts full agreement, 3 diverge late; hidden-state split-vs-plain: 0 at 2048 rows, max ~95-116 (mean 0.33) at 531/1159 rows, unexplained (not tile configs, job 210) | **open: correctness unresolved; no tok/s number** |

## Changed verdicts / new knowledge
- **The Window knob's "orthogonal resources" holds for the SM-driven host AR, not for NCCL** on this box: NCCL's
  SHM path overlaps nothing with W8A8 GEMMs, while the host AR overlaps ~35-55% of its time. Earlier "overlap
  doesn't work here" findings (T1-006/012, T3-003, T2's coupling) were measured with NCCL or within one GEMM.
- Splitting mixed-step GEMMs into halves is free at >= 152-row halves (job 203: 5.33 vs 5.53 ms).
- No mixer metadata split is needed: the row-independent chain between mixers carries both ARs.

## Ranking now (for whoever continues)
1. **Finish T4-001**: strip debug hooks (job 212's startup failure is a debug hook), then localize the 531/1159-row
   deviation. Compare per layer split vs plain. Suspects: an M-dependent bf16 cuBLAS kernel (in_proj_ba,
   M vs M/2), which is a legitimate reordering, or a fork/join race on a residual buffer. Then a same-job A/B and
   unofficial_eval --gsm8k --long. It stacks with T5's C1 (lossless AR encoding, same HostAllReduce contract).
2. Decode W8A16 kernel > 0.9 TB/s (new design), ~3-4%.
3. lm_head Triton GEMV in situ, ~0.8%.
