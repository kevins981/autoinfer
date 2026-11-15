team:         T3
builds_on:    T3-006
performance:  unofficial: +1.58% vs T3-006 (ABBA job 121, ABBAAB, all B > all A). Superseded by the port onto T2-011 (T3-009)
summary:
  hypothesis: T2-009's fused norm+quant made rowsplit_ok() False (the in_proj/gate_up/out_proj/down_proj input quant keys
              are now non-None), so mixed steps lost the row-split AR overlap: profile job 117 shows every mixed-step NCCL
              AR on the main stream, 27-39% of mixed-step time (~11% of the run). Restoring it should give ~3-5% napkin.
  change:     (worker leader-3.w1, ws t3-rs) model.py only: drop the four quant-key terms from rowsplit_ok(); in
              forward_split run the fused norm+quant per row half (forward_fused(x, r, in_proj / gate_up_proj,
              reduce=False)) and feed the pre-quantized activation to the linear; out_proj slices pre-quantized inputs
              per row via _row_slice (asserts a per-tensor scale; not exercised here: out/down inputs are bf16). A one-time
              info log of the rowsplit_ok conditions. Decode path unchanged.
  results:    Observed: "Row-split TP all-reduce pipelining active" in all B server logs, none in A. ABBA job 121 (200
              prompts, T3-006 args): B 882.0/895.9/891.1 vs A 877.7/873.5/876.1, +1.58%, all B > all A; TPOT p99 27.6-34.5
              (B) vs 35.1-40.3 ms (A); acceptance equal. GSM8K-40 on a B server 90.0%.
              Diagnosis: worker profile job 117 (T2-009 code), and leader-3's grep of server logs (row-split line in all
              T3-003-code servers, none in T2-009/T3-006 ones).
  conclusion: Interpretation: about half of the napkin. Row-split hides part of the exposed AR; the rest is AR on the
              critical path (first/last halves) and NCCL's slowdown under GEMM contention (cf. T3-002).
validity:     (leader-3) Same per-row computations as the decode path (fused norm+quant per row; quant groups along the
              hidden dim), so row-splitting is exact by construction; the AR of each half is the same 2-term bf16 sum.
              Numerics of mixed steps change from the T3-003-era unfused norm to T2-009's fused one = the champion's
              decode numerics (already accepted with T2-009). No memory change.
