# T29 task 1: FP8 W8A8 (block-scaled) vs champion Marlin, per-rank linears at TP=2 (leader-29.w1)

Files: bench.py (job 190, 1.5 GPU-min), calib.py (job 191, 0.2 GPU-min), bench.json (raw), summary.md (all tables:
per-forward totals GPU0/GPU1, unsplit and with the champion's token split, per-shape us, best cfg, accuracy).
Workspace t29-fp8 (T27-002), unmodified; Marlin via apply_fp8_marlin_linear exactly as MarlinFP8ScaledMMLinearKernel
(process_fp8_weight_block_strategy + prepare_fp8_layer_for_marlin(size_k_first=False)), T6 policy default (AI_* unset).
Weights: REAL checkpoint, layer 0 (GDN + MLP) and layer 3 (attention), TP rank-0 slices.
Shapes (N x K) x count/forward: qkvz 8192x5120 x48, gdn_out 5120x3072 x48, attn_qkv 7168x5120 x16, attn_o 5120x3072 x16,
gate_up 17408x5120 x64, down 5120x8704 x64. in_proj_a/b are bf16 in the checkpoint (modules_to_not_convert): out of scope.
Timing: CUDA graph of 8 reps, CUDA events, median of 9; cold = 256 MB READ flush before each op, flush time subtracted.
W8A8 = vLLM per_token_group_quant_fp8 (CUDA op, group 128, row-major scales, no ue8m0) + _w8a8_triton_block_scaled_mm,
16 configs swept on GPU0 (BLOCK_K=128 always), GPU1 re-timed top-3 + default per (shape, M).

## gpu0 per-forward linear totals, ms per rank, cold (READ flush)
| M | marlin | w8a8_gemm | w8a8_e2e | triton_default | hybrid_e2e | scaled_mm_ref | saved e2e vs Marlin | saved hybrid |
|---|---|---|---|---|---|---|---|---|
| 48 | 16.35 | 15.61 | 16.53 | 24.49 | 15.95 | 15.49 | -0.18 (-1%) | +0.40 |
| 64 | 17.69 | 16.36 | 17.27 | 24.54 | 17.20 | 15.65 | +0.42 (+2%) | +0.49 |
| 96 | 24.08 | 20.40 | 21.34 | 28.21 | 21.34 | 17.72 | +2.75 (+11%) | +2.75 |
| 128 | 30.08 | 23.21 | 24.40 | 28.37 | 24.40 | 20.31 | +5.68 (+19%) | +5.68 |
| 192 | 41.54 | 29.39 | 30.92 | 34.36 | 30.92 | 23.79 | +10.62 (+26%) | +10.62 |
| 256 | 53.29 | 35.70 | 37.39 | 43.89 | 37.39 | 29.82 | +15.89 (+30%) | +15.89 |
| 320 | 66.51 | 45.11 | 47.29 | 51.20 | 47.29 | 37.36 | +19.22 (+29%) | +19.22 |
| 384 | 76.99 | 48.23 | 50.47 | 55.22 | 50.47 | 40.10 | +26.52 (+34%) | +26.52 |
| 512 | 98.00 | 61.58 | 64.24 | 73.90 | 64.24 | 52.49 | +33.77 (+34%) | +33.77 |
| 1024 | 190.85 | 115.92 | 121.06 | 130.55 | 121.06 | 94.59 | +69.80 (+37%) | +69.80 |

## gpu0 per-forward linear totals, ms per rank, cold (READ flush) , champion token split at M>=256
| M | marlin | w8a8_gemm | w8a8_e2e | triton_default | hybrid_e2e | scaled_mm_ref | saved e2e vs Marlin | saved hybrid |
|---|---|---|---|---|---|---|---|---|
| 48 | 16.35 | 15.61 | 16.53 | 24.49 | 15.95 | 15.49 | -0.18 (-1%) | +0.40 |
| 64 | 17.69 | 16.36 | 17.27 | 24.54 | 17.20 | 15.65 | +0.42 (+2%) | +0.49 |
| 96 | 24.08 | 20.40 | 21.34 | 28.21 | 21.34 | 17.72 | +2.75 (+11%) | +2.75 |
| 128 | 30.08 | 23.21 | 24.40 | 28.37 | 24.40 | 20.31 | +5.68 (+19%) | +5.68 |
| 192 | 41.54 | 29.39 | 30.92 | 34.36 | 30.92 | 23.79 | +10.62 (+26%) | +10.62 |
| 256 | 60.16 | 46.42 | 48.81 | 56.74 | 48.81 | 40.62 | +11.36 (+19%) | +11.36 |
| 320 | 71.62 | 52.60 | 55.32 | 62.73 | 55.32 | 44.10 | +16.30 (+23%) | +16.30 |
| 384 | 83.08 | 58.79 | 61.84 | 68.72 | 61.84 | 47.58 | +21.24 (+26%) | +21.24 |
| 512 | 106.58 | 71.39 | 74.79 | 87.78 | 74.79 | 59.64 | +31.79 (+30%) | +31.79 |
| 1024 | 196.01 | 123.17 | 128.48 | 147.80 | 128.48 | 104.98 | +67.53 (+34%) | +67.53 |

## gpu0 per shape cold us: Marlin / W8A8 e2e (quant+best GEMM) / best GEMM only [cfg] / quant warm / scaled_mm ref

qkvz:
  M=   48 marlin    54.1  e2e    56.9  gemm    54.0 (32, 64, 4, 4, 1)      quant   2.9  scaled_mm    54.5  ratio e2e/marlin 1.05
  M=   64 marlin    56.1  e2e    57.6  gemm    54.2 (32, 64, 4, 4, 1)      quant   2.9  scaled_mm    53.9  ratio e2e/marlin 1.03
  M=   96 marlin    76.3  e2e    69.1  gemm    65.4 (32, 128, 4, 3, 1)     quant   3.0  scaled_mm    57.1  ratio e2e/marlin 0.91
  M=  128 marlin   100.3  e2e    72.0  gemm    66.3 (128, 64, 4, 3, 1)     quant   3.3  scaled_mm    58.9  ratio e2e/marlin 0.72
  M=  192 marlin   154.4  e2e    95.8  gemm    90.3 (64, 64, 4, 3, 1)      quant   3.5  scaled_mm    90.3  ratio e2e/marlin 0.62
  M=  256 marlin   171.2  e2e   117.6  gemm   110.7 (64, 128, 4, 3, 1)     quant   4.2  scaled_mm    89.9  ratio e2e/marlin 0.69
  M=  320 marlin   236.6  e2e   147.9  gemm   141.1 (64, 64, 4, 3, 1)      quant   4.6  scaled_mm   130.9  ratio e2e/marlin 0.62
  M=  384 marlin   263.7  e2e   168.0  gemm   159.4 (64, 128, 4, 3, 1)     quant   4.9  scaled_mm   131.0  ratio e2e/marlin 0.64
  M=  512 marlin   335.2  e2e   216.9  gemm   206.3 (64, 128, 4, 3, 1)     quant   5.8  scaled_mm   171.8  ratio e2e/marlin 0.65
  M= 1024 marlin   661.2  e2e   415.4  gemm   393.5 (64, 128, 4, 3, 1)     quant   9.0  scaled_mm   333.8  ratio e2e/marlin 0.63

gdn_out:
  M=   48 marlin    28.9  e2e    25.0  gemm    22.6 (16, 128, 4, 4, 1)     quant   2.9  scaled_mm    22.7  ratio e2e/marlin 0.87
  M=   64 marlin    29.1  e2e    28.7  gemm    25.9 (64, 64, 4, 3, 1)      quant   2.9  scaled_mm    23.3  ratio e2e/marlin 0.99
  M=   96 marlin    38.8  e2e    30.3  gemm    27.1 (32, 128, 4, 3, 1)     quant   2.9  scaled_mm    25.2  ratio e2e/marlin 0.78
  M=  128 marlin    51.4  e2e    42.5  gemm    38.9 (64, 64, 4, 3, 1)      quant   3.0  scaled_mm    32.6  ratio e2e/marlin 0.83
  M=  192 marlin    67.0  e2e    45.8  gemm    41.3 (64, 64, 4, 3, 1)      quant   3.3  scaled_mm    32.1  ratio e2e/marlin 0.68
  M=  256 marlin    83.8  e2e    60.4  gemm    56.2 (64, 64, 4, 3, 1)      quant   3.5  scaled_mm    48.0  ratio e2e/marlin 0.72
  M=  320 marlin   107.1  e2e    74.0  gemm    68.9 (64, 128, 4, 3, 1)     quant   3.6  scaled_mm    57.6  ratio e2e/marlin 0.69
  M=  384 marlin   125.7  e2e    76.7  gemm    70.7 (64, 128, 4, 3, 1)     quant   4.1  scaled_mm    58.4  ratio e2e/marlin 0.61
  M=  512 marlin   146.0  e2e    96.9  gemm    89.5 (64, 64, 4, 3, 1)      quant   4.3  scaled_mm    76.9  ratio e2e/marlin 0.66
  M= 1024 marlin   271.7  e2e   174.7  gemm   162.2 (64, 128, 4, 3, 1)     quant   6.5  scaled_mm   136.9  ratio e2e/marlin 0.64

attn_qkv:
  M=   48 marlin    52.5  e2e    49.9  gemm    47.2 (32, 64, 4, 4, 1)      quant   2.9  scaled_mm    46.5  ratio e2e/marlin 0.95
  M=   64 marlin    54.6  e2e    50.2  gemm    46.9 (32, 64, 4, 4, 1)      quant   3.0  scaled_mm    47.9  ratio e2e/marlin 0.92
  M=   96 marlin    73.8  e2e    67.2  gemm    63.5 (64, 64, 4, 3, 1)      quant   3.1  scaled_mm    50.7  ratio e2e/marlin 0.91
  M=  128 marlin   100.3  e2e    68.6  gemm    64.0 (64, 64, 4, 3, 1)      quant   3.4  scaled_mm    51.5  ratio e2e/marlin 0.68
  M=  192 marlin   135.2  e2e    93.6  gemm    88.6 (64, 64, 4, 3, 1)      quant   3.6  scaled_mm    76.0  ratio e2e/marlin 0.69
  M=  256 marlin   165.3  e2e   115.6  gemm   109.1 (64, 128, 4, 3, 1)     quant   4.1  scaled_mm    88.5  ratio e2e/marlin 0.70
  M=  320 marlin   210.1  e2e   144.4  gemm   136.6 (64, 64, 4, 3, 1)      quant   4.6  scaled_mm   115.3  ratio e2e/marlin 0.69
  M=  384 marlin   253.4  e2e   165.9  gemm   157.3 (64, 128, 4, 3, 1)     quant   4.9  scaled_mm   128.9  ratio e2e/marlin 0.65
  M=  512 marlin   308.2  e2e   203.4  gemm   193.8 (64, 64, 4, 3, 1)      quant   5.8  scaled_mm   175.2  ratio e2e/marlin 0.66
  M= 1024 marlin   590.0  e2e   368.3  gemm   347.3 (64, 128, 4, 3, 1)     quant   8.9  scaled_mm   293.1  ratio e2e/marlin 0.62

attn_o:
  M=   48 marlin    28.9  e2e    24.9  gemm    22.1 (16, 128, 4, 4, 1)     quant   2.8  scaled_mm    22.6  ratio e2e/marlin 0.86
  M=   64 marlin    29.1  e2e    28.6  gemm    26.1 (64, 64, 4, 3, 1)      quant   2.9  scaled_mm    23.2  ratio e2e/marlin 0.99
  M=   96 marlin    38.7  e2e    30.2  gemm    26.9 (32, 128, 4, 3, 1)     quant   2.9  scaled_mm    25.5  ratio e2e/marlin 0.78
  M=  128 marlin    51.5  e2e    42.5  gemm    38.9 (64, 64, 4, 3, 1)      quant   3.1  scaled_mm    33.6  ratio e2e/marlin 0.83
  M=  192 marlin    67.0  e2e    45.4  gemm    41.2 (64, 64, 4, 3, 1)      quant   3.3  scaled_mm    32.6  ratio e2e/marlin 0.68
  M=  256 marlin    84.0  e2e    60.5  gemm    55.8 (64, 64, 4, 3, 1)      quant   3.4  scaled_mm    47.7  ratio e2e/marlin 0.72
  M=  320 marlin   106.6  e2e    74.2  gemm    68.9 (64, 128, 4, 3, 1)     quant   3.6  scaled_mm    63.9  ratio e2e/marlin 0.70
  M=  384 marlin   125.8  e2e    76.6  gemm    70.3 (64, 128, 4, 3, 1)     quant   4.1  scaled_mm    58.4  ratio e2e/marlin 0.61
  M=  512 marlin   146.0  e2e    96.5  gemm    89.4 (64, 64, 4, 3, 1)      quant   4.4  scaled_mm    76.7  ratio e2e/marlin 0.66
  M= 1024 marlin   271.6  e2e   174.7  gemm   162.0 (64, 128, 4, 3, 1)     quant   6.5  scaled_mm   137.1  ratio e2e/marlin 0.64

gate_up:
  M=   48 marlin   111.6  e2e   118.6  gemm   113.7 (64, 64, 4, 3, 1)      quant   2.9  scaled_mm   112.2  ratio e2e/marlin 1.06
  M=   64 marlin   123.3  e2e   119.5  gemm   115.6 (64, 64, 4, 3, 1)      quant   3.0  scaled_mm   114.0  ratio e2e/marlin 0.97
  M=   96 marlin   166.0  e2e   162.4  gemm   159.6 (128, 64, 4, 3, 1)     quant   3.0  scaled_mm   133.1  ratio e2e/marlin 0.98
  M=  128 marlin   202.2  e2e   164.5  gemm   161.2 (128, 64, 4, 3, 1)     quant   3.3  scaled_mm   153.9  ratio e2e/marlin 0.81
  M=  192 marlin   280.6  e2e   231.6  gemm   226.5 (64, 128, 4, 3, 1)     quant   3.6  scaled_mm   177.0  ratio e2e/marlin 0.83
  M=  256 marlin   375.7  e2e   256.8  gemm   251.9 (128, 64, 4, 3, 1)     quant   4.2  scaled_mm   210.9  ratio e2e/marlin 0.68
  M=  320 marlin   450.2  e2e   332.8  gemm   322.6 (64, 128, 4, 3, 1)     quant   4.6  scaled_mm   256.8  ratio e2e/marlin 0.74
  M=  384 marlin   531.7  e2e   354.2  gemm   347.3 (128, 64, 4, 3, 1)     quant   5.0  scaled_mm   295.6  ratio e2e/marlin 0.67
  M=  512 marlin   696.6  e2e   452.4  gemm   446.0 (128, 64, 4, 3, 1)     quant   5.7  scaled_mm   366.8  ratio e2e/marlin 0.65
  M= 1024 marlin  1370.2  e2e   866.9  gemm   853.0 (128, 64, 4, 3, 1)     quant   8.9  scaled_mm   673.6  ratio e2e/marlin 0.63

down:
  M=   48 marlin    61.2  e2e    59.4  gemm    55.4 (16, 64, 4, 5, 1)      quant   3.0  scaled_mm    54.6  ratio e2e/marlin 0.97
  M=   64 marlin    68.3  e2e    65.9  gemm    61.7 (32, 64, 4, 4, 1)      quant   3.2  scaled_mm    55.0  ratio e2e/marlin 0.97
  M=   96 marlin    95.8  e2e    72.1  gemm    67.2 (32, 128, 4, 3, 1)     quant   3.5  scaled_mm    63.0  ratio e2e/marlin 0.75
  M=  128 marlin   116.2  e2e   103.2  gemm    96.9 (64, 64, 4, 3, 1)      quant   4.1  scaled_mm    73.5  ratio e2e/marlin 0.89
  M=  192 marlin   151.8  e2e   110.5  gemm   101.7 (64, 64, 4, 3, 1)      quant   4.6  scaled_mm    75.8  ratio e2e/marlin 0.73
  M=  256 marlin   203.4  e2e   149.9  gemm   139.4 (64, 64, 4, 3, 1)      quant   5.2  scaled_mm   117.6  ratio e2e/marlin 0.74
  M=  320 marlin   252.1  e2e   185.1  gemm   173.5 (64, 128, 4, 3, 1)     quant   6.0  scaled_mm   140.8  ratio e2e/marlin 0.73
  M=  384 marlin   284.5  e2e   190.2  gemm   176.7 (64, 128, 4, 3, 1)     quant   6.6  scaled_mm   142.2  ratio e2e/marlin 0.67
  M=  512 marlin   360.2  e2e   241.0  gemm   223.6 (64, 64, 4, 3, 1)      quant   7.9  scaled_mm   203.8  ratio e2e/marlin 0.67
  M= 1024 marlin   696.8  e2e   446.3  gemm   414.2 (64, 128, 4, 3, 1)     quant  13.2  scaled_mm   343.7  ratio e2e/marlin 0.64

## gpu1 per-forward linear totals, ms per rank, cold (READ flush)
| M | marlin | w8a8_gemm | w8a8_e2e | triton_default | hybrid_e2e | scaled_mm_ref | saved e2e vs Marlin | saved hybrid |
|---|---|---|---|---|---|---|---|---|
| 48 | 16.07 | 14.93 | 15.74 | 21.40 | 15.56 | 14.96 | +0.34 (+2%) | +0.51 |
| 64 | 17.10 | 15.28 | 16.13 | 21.62 | 15.86 | 15.18 | +0.97 (+6%) | +1.24 |
| 96 | 20.92 | 18.48 | 19.56 | 24.85 | 19.13 | 16.36 | +1.35 (+6%) | +1.78 |
| 128 | 26.04 | 20.98 | 21.93 | 25.13 | 21.93 | 18.70 | +4.11 (+16%) | +4.11 |
| 192 | 36.16 | 26.24 | 27.32 | 29.62 | 27.32 | 21.86 | +8.84 (+24%) | +8.84 |
| 256 | 46.96 | 32.26 | 34.05 | 37.64 | 34.05 | 27.33 | +12.91 (+27%) | +12.91 |
| 320 | 57.74 | 40.28 | 42.22 | 43.78 | 42.22 | 34.40 | +15.52 (+27%) | +15.52 |
| 384 | 65.78 | 43.75 | 46.10 | 47.10 | 46.10 | 36.53 | +19.68 (+30%) | +19.68 |
| 512 | 83.89 | 56.89 | 59.93 | 62.71 | 59.93 | 48.03 | +23.96 (+29%) | +23.96 |
| 1024 | 162.62 | 104.31 | 109.25 | 111.06 | 109.25 | 84.35 | +53.36 (+33%) | +53.36 |

## gpu1 per-forward linear totals, ms per rank, cold (READ flush) , champion token split at M>=256
| M | marlin | w8a8_gemm | w8a8_e2e | triton_default | hybrid_e2e | scaled_mm_ref | saved e2e vs Marlin | saved hybrid |
|---|---|---|---|---|---|---|---|---|
| 48 | 16.07 | 14.93 | 15.74 | 21.40 | 15.56 | 14.96 | +0.34 (+2%) | +0.51 |
| 64 | 17.10 | 15.28 | 16.13 | 21.62 | 15.86 | 15.18 | +0.97 (+6%) | +1.24 |
| 96 | 20.92 | 18.48 | 19.56 | 24.85 | 19.13 | 16.36 | +1.35 (+6%) | +1.78 |
| 128 | 26.04 | 20.98 | 21.93 | 25.13 | 21.93 | 18.70 | +4.11 (+16%) | +4.11 |
| 192 | 36.16 | 26.24 | 27.32 | 29.62 | 27.32 | 21.86 | +8.84 (+24%) | +8.84 |
| 256 | 52.07 | 41.96 | 43.85 | 50.26 | 43.85 | 37.40 | +8.22 (+16%) | +8.22 |
| 320 | 62.20 | 47.22 | 49.24 | 54.75 | 49.24 | 40.56 | +12.95 (+21%) | +12.95 |
| 384 | 72.32 | 52.49 | 54.63 | 59.24 | 54.63 | 43.73 | +17.69 (+24%) | +17.69 |
| 512 | 93.91 | 64.51 | 68.09 | 75.28 | 68.09 | 54.66 | +25.82 (+27%) | +25.82 |
| 1024 | 167.78 | 113.78 | 119.87 | 125.42 | 119.87 | 96.05 | +47.91 (+29%) | +47.91 |

## gpu1 per shape cold us: Marlin / W8A8 e2e (quant+best GEMM) / best GEMM only [cfg] / quant warm / scaled_mm ref

qkvz:
  M=   48 marlin    52.9  e2e    56.5  gemm    53.7 (32, 64, 4, 4, 1)      quant   2.5  scaled_mm    53.4  ratio e2e/marlin 1.07
  M=   64 marlin    52.0  e2e    57.5  gemm    54.3 (32, 64, 4, 4, 1)      quant   2.6  scaled_mm    54.2  ratio e2e/marlin 1.11
  M=   96 marlin    65.3  e2e    64.3  gemm    60.6 (32, 128, 4, 3, 1)     quant   2.6  scaled_mm    57.3  ratio e2e/marlin 0.98
  M=  128 marlin    86.8  e2e    65.9  gemm    62.2 (64, 64, 4, 3, 1)      quant   2.9  scaled_mm    57.9  ratio e2e/marlin 0.76
  M=  192 marlin   139.6  e2e    86.6  gemm    82.9 (64, 64, 4, 3, 1)      quant   3.1  scaled_mm    81.3  ratio e2e/marlin 0.62
  M=  256 marlin   144.2  e2e   106.1  gemm    99.8 (128, 128, 8, 3, 8)    quant   3.6  scaled_mm    80.5  ratio e2e/marlin 0.74
  M=  320 marlin   208.5  e2e   135.9  gemm   130.2 (64, 64, 4, 3, 1)      quant   3.9  scaled_mm   122.7  ratio e2e/marlin 0.65
  M=  384 marlin   222.6  e2e   150.7  gemm   142.7 (64, 128, 4, 3, 1)     quant   4.2  scaled_mm   117.0  ratio e2e/marlin 0.68
  M=  512 marlin   282.0  e2e   192.2  gemm   182.5 (64, 128, 4, 3, 1)     quant   4.9  scaled_mm   155.4  ratio e2e/marlin 0.68
  M= 1024 marlin   556.3  e2e   364.0  gemm   345.5 (64, 128, 4, 3, 1)     quant   7.5  scaled_mm   298.1  ratio e2e/marlin 0.65

gdn_out:
  M=   48 marlin    28.2  e2e    23.5  gemm    21.1 (16, 64, 4, 5, 1)      quant   2.7  scaled_mm    21.7  ratio e2e/marlin 0.84
  M=   64 marlin    25.9  e2e    25.7  gemm    23.3 (32, 64, 4, 4, 1)      quant   2.5  scaled_mm    21.8  ratio e2e/marlin 0.99
  M=   96 marlin    33.4  e2e    27.3  gemm    24.3 (32, 128, 4, 3, 1)     quant   3.3  scaled_mm    22.4  ratio e2e/marlin 0.82
  M=  128 marlin    45.7  e2e    37.0  gemm    33.8 (64, 64, 4, 3, 1)      quant   3.8  scaled_mm    29.2  ratio e2e/marlin 0.81
  M=  192 marlin    58.5  e2e    39.5  gemm    35.7 (64, 64, 4, 3, 1)      quant   4.6  scaled_mm    28.2  ratio e2e/marlin 0.68
  M=  256 marlin    72.8  e2e    53.3  gemm    48.5 (64, 64, 4, 3, 1)      quant   4.6  scaled_mm    42.1  ratio e2e/marlin 0.73
  M=  320 marlin    94.3  e2e    64.6  gemm    59.3 (64, 128, 4, 3, 1)     quant   4.6  scaled_mm    54.0  ratio e2e/marlin 0.69
  M=  384 marlin   110.7  e2e    67.2  gemm    61.0 (64, 128, 4, 3, 1)     quant   4.6  scaled_mm    50.9  ratio e2e/marlin 0.61
  M=  512 marlin   125.1  e2e    85.2  gemm    78.9 (64, 64, 4, 3, 1)      quant   4.6  scaled_mm    75.2  ratio e2e/marlin 0.68
  M= 1024 marlin   230.1  e2e   153.0  gemm   140.6 (64, 128, 4, 3, 1)     quant   5.7  scaled_mm   123.0  ratio e2e/marlin 0.67

attn_qkv:
  M=   48 marlin    50.3  e2e    48.2  gemm    45.3 (32, 64, 4, 4, 1)      quant   2.5  scaled_mm    45.8  ratio e2e/marlin 0.96
  M=   64 marlin    51.6  e2e    48.9  gemm    46.2 (32, 64, 4, 4, 1)      quant   2.6  scaled_mm    46.7  ratio e2e/marlin 0.95
  M=   96 marlin    63.7  e2e    58.1  gemm    55.1 (32, 128, 4, 3, 1)     quant   2.6  scaled_mm    48.5  ratio e2e/marlin 0.91
  M=  128 marlin    87.9  e2e    59.1  gemm    54.9 (64, 64, 4, 3, 1)      quant   2.9  scaled_mm    50.0  ratio e2e/marlin 0.67
  M=  192 marlin   117.5  e2e    84.1  gemm    78.9 (64, 64, 4, 3, 1)      quant   3.1  scaled_mm    75.9  ratio e2e/marlin 0.72
  M=  256 marlin   141.0  e2e   100.2  gemm    93.9 (64, 128, 4, 3, 1)     quant   3.5  scaled_mm    76.2  ratio e2e/marlin 0.71
  M=  320 marlin   181.4  e2e   130.5  gemm   122.5 (64, 64, 4, 3, 1)      quant   4.0  scaled_mm   106.7  ratio e2e/marlin 0.72
  M=  384 marlin   221.2  e2e   145.6  gemm   136.8 (64, 128, 4, 3, 1)     quant   4.2  scaled_mm   114.8  ratio e2e/marlin 0.66
  M=  512 marlin   260.4  e2e   185.9  gemm   174.7 (64, 64, 4, 3, 1)      quant   4.9  scaled_mm   160.8  ratio e2e/marlin 0.71
  M= 1024 marlin   496.5  e2e   321.4  gemm   304.3 (64, 128, 4, 3, 1)     quant   7.5  scaled_mm   263.0  ratio e2e/marlin 0.65

attn_o:
  M=   48 marlin    28.6  e2e    23.4  gemm    21.3 (16, 64, 4, 5, 1)      quant   2.5  scaled_mm    21.5  ratio e2e/marlin 0.82
  M=   64 marlin    26.1  e2e    25.0  gemm    22.3 (32, 64, 4, 4, 1)      quant   2.5  scaled_mm    21.7  ratio e2e/marlin 0.96
  M=   96 marlin    33.6  e2e    27.0  gemm    24.1 (32, 128, 4, 3, 1)     quant   2.5  scaled_mm    22.4  ratio e2e/marlin 0.80
  M=  128 marlin    45.1  e2e    36.4  gemm    33.2 (64, 64, 4, 3, 1)      quant   2.6  scaled_mm    28.7  ratio e2e/marlin 0.81
  M=  192 marlin    57.5  e2e    38.5  gemm    35.0 (64, 64, 4, 3, 1)      quant   2.8  scaled_mm    27.6  ratio e2e/marlin 0.67
  M=  256 marlin    71.9  e2e    52.5  gemm    47.6 (64, 64, 4, 3, 1)      quant   3.0  scaled_mm    41.0  ratio e2e/marlin 0.73
  M=  320 marlin    93.3  e2e    63.9  gemm    58.5 (64, 128, 4, 3, 1)     quant   3.1  scaled_mm    53.9  ratio e2e/marlin 0.69
  M=  384 marlin   109.7  e2e    66.5  gemm    60.3 (64, 128, 4, 3, 1)     quant   3.5  scaled_mm    50.1  ratio e2e/marlin 0.61
  M=  512 marlin   124.3  e2e    84.9  gemm    78.1 (64, 64, 4, 3, 1)      quant   3.7  scaled_mm    75.4  ratio e2e/marlin 0.68
  M= 1024 marlin   229.7  e2e   153.1  gemm   140.8 (64, 128, 4, 3, 1)     quant   5.5  scaled_mm   122.5  ratio e2e/marlin 0.67

gate_up:
  M=   48 marlin   111.0  e2e   110.7  gemm   107.5 (64, 64, 4, 3, 1)      quant   2.6  scaled_mm   107.4  ratio e2e/marlin 1.00
  M=   64 marlin   124.7  e2e   112.3  gemm   108.1 (64, 64, 4, 3, 1)      quant   2.5  scaled_mm   109.1  ratio e2e/marlin 0.90
  M=   96 marlin   143.0  e2e   149.7  gemm   146.5 (128, 64, 4, 3, 1)     quant   2.6  scaled_mm   122.6  ratio e2e/marlin 1.05
  M=  128 marlin   173.5  e2e   151.8  gemm   149.5 (128, 64, 4, 3, 1)     quant   2.9  scaled_mm   142.6  ratio e2e/marlin 0.88
  M=  192 marlin   242.0  e2e   206.4  gemm   201.3 (64, 128, 4, 2, 32)    quant   3.1  scaled_mm   166.5  ratio e2e/marlin 0.85
  M=  256 marlin   339.7  e2e   240.5  gemm   233.6 (128, 64, 4, 3, 1)     quant   3.6  scaled_mm   199.5  ratio e2e/marlin 0.71
  M=  320 marlin   383.6  e2e   298.2  gemm   290.6 (64, 128, 4, 2, 32)    quant   3.9  scaled_mm   240.1  ratio e2e/marlin 0.78
  M=  384 marlin   451.4  e2e   335.2  gemm   327.6 (128, 64, 4, 3, 1)     quant   4.2  scaled_mm   276.9  ratio e2e/marlin 0.74
  M=  512 marlin   606.5  e2e   437.8  gemm   423.6 (64, 128, 4, 2, 32)    quant   5.1  scaled_mm   332.0  ratio e2e/marlin 0.72
  M= 1024 marlin  1182.6  e2e   809.2  gemm   790.3 (64, 128, 4, 2, 32)    quant   7.6  scaled_mm   597.8  ratio e2e/marlin 0.68

down:
  M=   48 marlin    59.6  e2e    57.3  gemm    52.9 (16, 64, 4, 5, 1)      quant   2.6  scaled_mm    53.2  ratio e2e/marlin 0.96
  M=   64 marlin    64.6  e2e    58.8  gemm    55.3 (32, 64, 4, 4, 1)      quant   2.8  scaled_mm    54.0  ratio e2e/marlin 0.91
  M=   96 marlin    85.4  e2e    66.0  gemm    58.7 (32, 64, 4, 4, 1)      quant   3.0  scaled_mm    55.5  ratio e2e/marlin 0.77
  M=  128 marlin   100.7  e2e    89.8  gemm    84.2 (64, 64, 4, 3, 1)      quant   3.5  scaled_mm    64.5  ratio e2e/marlin 0.89
  M=  192 marlin   130.6  e2e    95.2  gemm    91.3 (64, 128, 4, 3, 1)     quant   3.9  scaled_mm    67.1  ratio e2e/marlin 0.73
  M=  256 marlin   178.1  e2e   133.8  gemm   123.7 (64, 64, 4, 3, 1)      quant   4.5  scaled_mm   106.2  ratio e2e/marlin 0.75
  M=  320 marlin   222.8  e2e   162.5  gemm   151.3 (64, 128, 4, 3, 1)     quant   5.1  scaled_mm   124.7  ratio e2e/marlin 0.73
  M=  384 marlin   243.6  e2e   168.7  gemm   154.0 (128, 128, 8, 3, 8)    quant   5.5  scaled_mm   126.7  ratio e2e/marlin 0.69
  M=  512 marlin   302.7  e2e   222.9  gemm   206.1 (64, 64, 4, 3, 1)      quant   6.7  scaled_mm   186.4  ratio e2e/marlin 0.74
  M= 1024 marlin   587.0  e2e   391.5  gemm   363.6 (64, 128, 4, 3, 1)     quant  11.1  scaled_mm   307.9  ratio e2e/marlin 0.67

## accuracy GPU0 (vs fp32 ref of dequantized W, bf16 x with 1% outlier channels x20): relFrob marlin / w8a8 ; maxabs marlin / w8a8 ; ref max
  qkvz      M=   48 relF 2.28e-03 / 1.48e-02  maxabs 0.071 / 0.337  ref 27.8  default==best bitwise True
  qkvz      M=   64 relF 2.27e-03 / 1.49e-02  maxabs 0.0911 / 0.319  ref 32.4  default==best bitwise True
  qkvz      M=   96 relF 2.28e-03 / 1.55e-02  maxabs 0.0946 / 0.429  ref 29  default==best bitwise True
  qkvz      M=  128 relF 2.29e-03 / 1.64e-02  maxabs 0.0808 / 0.528  ref 30.7  default==best bitwise True
  qkvz      M=  192 relF 2.28e-03 / 1.60e-02  maxabs 0.0987 / 0.514  ref 31  default==best bitwise True
  qkvz      M=  256 relF 2.28e-03 / 1.61e-02  maxabs 0.0802 / 0.447  ref 35.4  default==best bitwise True
  qkvz      M=  320 relF 2.28e-03 / 1.47e-02  maxabs 0.0844 / 0.443  ref 27.7  default==best bitwise True
  qkvz      M=  384 relF 2.29e-03 / 1.63e-02  maxabs 0.102 / 0.509  ref 35.9  default==best bitwise True
  qkvz      M=  512 relF 2.28e-03 / 1.54e-02  maxabs 0.116 / 0.5  ref 37.6  default==best bitwise True
  qkvz      M= 1024 relF 2.28e-03 / 1.52e-02  maxabs 0.0874 / 0.421  ref 28.8  default==best bitwise True
  gdn_out   M=   48 relF 2.29e-03 / 1.58e-02  maxabs 0.17 / 1.47  ref 66.5  default==best bitwise True
  gdn_out   M=   64 relF 2.28e-03 / 1.55e-02  maxabs 0.267 / 1.6  ref 105  default==best bitwise True
  gdn_out   M=   96 relF 2.28e-03 / 1.62e-02  maxabs 0.27 / 2.27  ref 116  default==best bitwise True
  gdn_out   M=  128 relF 2.28e-03 / 1.51e-02  maxabs 0.291 / 1.42  ref 95.6  default==best bitwise True
  gdn_out   M=  192 relF 2.30e-03 / 1.46e-02  maxabs 0.244 / 1.48  ref 108  default==best bitwise True
  gdn_out   M=  256 relF 2.31e-03 / 1.60e-02  maxabs 0.509 / 1.97  ref 143  default==best bitwise True
  gdn_out   M=  320 relF 2.31e-03 / 1.59e-02  maxabs 0.389 / 2.23  ref 114  default==best bitwise True
  gdn_out   M=  384 relF 2.28e-03 / 1.54e-02  maxabs 0.234 / 1.72  ref 93  default==best bitwise True
  gdn_out   M=  512 relF 2.33e-03 / 1.47e-02  maxabs 0.456 / 2.62  ref 145  default==best bitwise True
  gdn_out   M= 1024 relF 2.29e-03 / 1.47e-02  maxabs 0.41 / 3.62  ref 168  default==best bitwise True
  attn_qkv  M=   48 relF 2.28e-03 / 1.53e-02  maxabs 0.0534 / 0.266  ref 16.5  default==best bitwise True
  attn_qkv  M=   64 relF 2.27e-03 / 1.54e-02  maxabs 0.0475 / 0.264  ref 15.8  default==best bitwise True
  attn_qkv  M=   96 relF 2.28e-03 / 1.62e-02  maxabs 0.0528 / 0.289  ref 17.5  default==best bitwise True
  attn_qkv  M=  128 relF 2.28e-03 / 1.55e-02  maxabs 0.0522 / 0.265  ref 19.2  default==best bitwise True
  attn_qkv  M=  192 relF 2.27e-03 / 1.60e-02  maxabs 0.0592 / 0.317  ref 19.1  default==best bitwise True
  attn_qkv  M=  256 relF 2.28e-03 / 1.59e-02  maxabs 0.0641 / 0.313  ref 19.5  default==best bitwise True
  attn_qkv  M=  320 relF 2.28e-03 / 1.54e-02  maxabs 0.0562 / 0.307  ref 18.6  default==best bitwise True
  attn_qkv  M=  384 relF 2.28e-03 / 1.50e-02  maxabs 0.0613 / 0.293  ref 17.3  default==best bitwise True
  attn_qkv  M=  512 relF 2.28e-03 / 1.57e-02  maxabs 0.0692 / 0.303  ref 18.4  default==best bitwise True
  attn_qkv  M= 1024 relF 2.28e-03 / 1.53e-02  maxabs 0.071 / 0.314  ref 19.2  default==best bitwise True
  attn_o    M=   48 relF 2.27e-03 / 1.45e-02  maxabs 0.0989 / 0.897  ref 37.7  default==best bitwise True
  attn_o    M=   64 relF 2.27e-03 / 1.61e-02  maxabs 0.12 / 1.03  ref 60.5  default==best bitwise True
  attn_o    M=   96 relF 2.27e-03 / 1.58e-02  maxabs 0.184 / 0.929  ref 71.9  default==best bitwise True
  attn_o    M=  128 relF 2.27e-03 / 1.50e-02  maxabs 0.187 / 1.23  ref 56.3  default==best bitwise True
  attn_o    M=  192 relF 2.27e-03 / 1.54e-02  maxabs 0.146 / 0.91  ref 58.1  default==best bitwise True
  attn_o    M=  256 relF 2.27e-03 / 1.57e-02  maxabs 0.182 / 1.72  ref 67.9  default==best bitwise True
  attn_o    M=  320 relF 2.26e-03 / 1.61e-02  maxabs 0.178 / 1.35  ref 73.3  default==best bitwise True
  attn_o    M=  384 relF 2.27e-03 / 1.52e-02  maxabs 0.153 / 0.963  ref 69.9  default==best bitwise True
  attn_o    M=  512 relF 2.31e-03 / 1.59e-02  maxabs 0.348 / 1.74  ref 99.4  default==best bitwise True
  attn_o    M= 1024 relF 2.27e-03 / 1.56e-02  maxabs 0.173 / 1.06  ref 55.7  default==best bitwise True
  gate_up   M=   48 relF 2.29e-03 / 1.50e-02  maxabs 0.0252 / 0.158  ref 9.62  default==best bitwise True
  gate_up   M=   64 relF 2.28e-03 / 1.55e-02  maxabs 0.0304 / 0.18  ref 9.91  default==best bitwise True
  gate_up   M=   96 relF 2.29e-03 / 1.52e-02  maxabs 0.0424 / 0.206  ref 15  default==best bitwise True
  gate_up   M=  128 relF 2.28e-03 / 1.52e-02  maxabs 0.0606 / 0.262  ref 11.7  default==best bitwise True
  gate_up   M=  192 relF 2.28e-03 / 1.58e-02  maxabs 0.0384 / 0.185  ref 11.6  default==best bitwise True
  gate_up   M=  256 relF 2.28e-03 / 1.55e-02  maxabs 0.0361 / 0.191  ref 12.9  default==best bitwise True
  gate_up   M=  320 relF 2.28e-03 / 1.61e-02  maxabs 0.045 / 0.236  ref 12.9  default==best bitwise True
  gate_up   M=  384 relF 2.29e-03 / 1.59e-02  maxabs 0.0424 / 0.203  ref 15.5  default==best bitwise True
  gate_up   M=  512 relF 2.28e-03 / 1.57e-02  maxabs 0.0383 / 0.212  ref 12.1  default==best bitwise True
  gate_up   M= 1024 relF 2.28e-03 / 1.58e-02  maxabs 0.0426 / 0.248  ref 15.7  default==best bitwise True
  down      M=   48 relF 2.28e-03 / 1.61e-02  maxabs 0.142 / 1.21  ref 74.9  default==best bitwise True
  down      M=   64 relF 2.29e-03 / 1.51e-02  maxabs 0.211 / 1.42  ref 60.8  default==best bitwise True
  down      M=   96 relF 2.28e-03 / 1.57e-02  maxabs 0.155 / 1.1  ref 67.4  default==best bitwise True
  down      M=  128 relF 2.29e-03 / 1.55e-02  maxabs 0.25 / 1.23  ref 87.1  default==best bitwise True
  down      M=  192 relF 2.29e-03 / 1.55e-02  maxabs 0.191 / 1.29  ref 75.4  default==best bitwise True
  down      M=  256 relF 2.28e-03 / 1.54e-02  maxabs 0.236 / 1.19  ref 76.2  default==best bitwise True
  down      M=  320 relF 2.29e-03 / 1.56e-02  maxabs 0.24 / 1.17  ref 81.1  default==best bitwise True
  down      M=  384 relF 2.28e-03 / 1.54e-02  maxabs 0.282 / 1.23  ref 96.8  default==best bitwise True
  down      M=  512 relF 2.28e-03 / 1.56e-02  maxabs 0.247 / 1.28  ref 87.2  default==best bitwise True
  down      M= 1024 relF 2.28e-03 / 1.57e-02  maxabs 0.212 / 1.21  ref 94.9  default==best bitwise True
