team:         T45
builds_on:    T37-002 (kernels of T31-003/T37-002: t31 e4m3, Marlin fp8, T29 bs_gemm_marlin), harness T35-003 / T45-003
performance:  "microbench only (stage 1b of T45-1): e4m3-MMA gate_up holds ~120 us flat from M=80 to 128; at M=64
               (today's k=3 decode) bs_gemm gate_up is 82.6 vs Marlin 104.1 us/layer in chain (-21.5 us/layer)"
summary:
  hypothesis: (a) An e4m3-MMA gate_up keeps the M=96-128 band near its ~91 us weight-stream floor (opening left by
              T45-003). (b) If so, rows above 64 get cheap.
  change:     None to the server. GPU jobs 411 and 413 (GPU0, 1.7 GPU-min total). Weights are synthetic, timings cold
              with a read flush.
              - gu.py: gate_up (17408x5120 per rank) standalone at M = 48..128. Kernels: Marlin bf16-MMA, t31 e4m3
                (9 cfgs, M<=96), T29 bs_gemm_marlin e4m3 (43 cfgs).
              - shapes.py: best bs cfg for all four TP decode shapes at M = 64/96/128.
              - bench_m.py: the T45-003 64-layer chain with GU=t31 (gate_up on t31) or GU=bs (all four GEMMs on
                bs_gemm, best cfgs).
  results:    Observed.
    Standalone gate_up (us; floor 90.9 at 0.98 TB/s; e4m3 MMA roof at 291 TF / bf16 MMA roof at 146 TF):
      M     Marlin   t31 best  bs best (cfg)   e4m3 roof  bf16 roof
      48    106.5    107.4     <=101.4         29         59
      64    119.8    187.4      99.8 (127)     39         78
      80    136.2    193.5     118.8 (127)     49         98
      96    137.2    196.6     119.7 (127)     59        117
      112   165.3     --       120.8 (127)     69        137
      128   164.9     --       121.9 (127)     78        156
    bs, other shapes (us; floors 42.8 / 16.0 / 45.5):
      qkvz     M64 49.2, M96 51.2, M128 51.2
      gdn_out  M64 22.5, M96 23.6, M128 30.0
      down     M64 56.3, M96 58.4, M128 75.8
    Chain, us/layer (champion-like = t31 + Marlin gate_up, from T45-003):
      M      champion-like  GU=t31   all-bs   gate_up in all-bs
      48     356.8          362.3    -        -
      64     368.6          427.7    344.2    82.6 (Marlin 104.1)
      80     404.9          445.0    386.1    111.9
      96     411.6          448.5    389.8    113.5
      112    -              -        410.9    116.6
      128    -              -        412.2    118.2
  conclusion: Interpretation:
    - (a) Holds for gate_up: bs_gemm is flat at ~120 us (1.3x floor) from M=80 to 128, while Marlin climbs to 165.
      The band cost moves to the other three shapes (gdn_out/down jump at M>96 with cfg 103: +6/+14 us) and to the
      64->80 tile step (one more 64-row m-tile: +42 us/layer in all-bs).
    - (b) Rows stay costly. All-bs at M=128 vs M=64 is +68 us/layer = 68 us/row/pass of GEMM, plus ~70 us of AR =
      ~0.14 ms/row vs 0.166 ms in a prefill event. Piggyback saves ~15% of a prompt token's cost: <= ~+2% e2e. The
      T45-1 bet stays killed; the link-priced row rule (KB118) stands with a GEMM band of ~35-70 us/row above 64.
    - UNEXPECTED, exploit-sized: at M=64 (today's k=3 decode, 16 req x 4 rows) the champion's gate_up stays on bf16
      Marlin (T31_SKIP default 17408x5120, T29_W8A8_MAX_MARLIN_M=96) and sits on Marlin's M>48 step (KB94's tmb=4
      cliff): 104.1 us/layer in chain vs 82.6 for bs_gemm cfg 127 on the same quantized-activation W8A8 path the
      champion already uses above M=96. Estimate: 21.5 us x 64 layers = 1.38 ms/step, minus ~0.2 ms of extra
      per-128 activation-quant launches for gate_up inputs. Expected to reach the clock at 0.5-0.75x (KB93): ~0.6-0.9
      ms of ~26.2 ms, i.e. +2-3% decode, ~+1.5-2.5% e2e. Not measured in server.
    - Why it was missed: KB96/KB98/KB101 compared gate_up kernels at M=48 (k=2 era), where Marlin has no step. T37's
      move to k=3 put decode at M=64 without revisiting gate_up dispatch.
    - Caveats: synthetic weights; GDN-layer proxy for all 64 layers; AR as fixed spins; the bs path changes gate_up
      numerics from W8A16 to the checkpoint-scheme W8A8 (as the champion already does above M=96 and on five
      shapes), so it needs the GSM8K gate; the activation quant cost was not in the chain.
validity:     Microbenchmark only. A server change would route gate_up at 48 < M <= 96 through the existing
              w8a8_marlin path (exact up to reduction order w.r.t. that path; W8A8 per the checkpoint's block scheme).
