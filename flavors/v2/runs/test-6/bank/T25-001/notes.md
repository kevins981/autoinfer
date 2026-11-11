# G1: per-GPU Marlin N-subrange microbench (T25, worker leader-25.w1)
Script: mb.py, job.sh (this folder); raw: res_gpu0.json, res_gpu1.json, clocks.log, out*.log. Workspace t25-s1 (built, KB61 _C path printed in out*.log). One gpu job, both GPUs concurrently (one process each, CUDA_VISIBLE_DEVICES=0/1), ~0.4 GPU-min charged.
Method: random FP8 block-scaled weights (128x128 scales) repacked with prepare_fp8_layer_for_marlin, ops.marlin_gemm default AI_POLICY (no env set). Intermediate per rank 8704 -> nh=round(8704 f/128)*128 (f=.88->7680 ... 1.12->9728). gate_up N=2*nh,K=5120; down N=5120,K=nh. CUDA graph of 8 calls, median of 25 replays; cold = 256MB flush kernel before each call (flush time subtracted), warm = no flush.
Clocks (nvidia-smi every 2 s during job): GPU0 2235 MHz in all 12 samples; GPU1 2520-2685 (mostly 2670) while running (210 idle samples at start/end).

Observed (f=1, cold sum gate_up+down, us): GPU0/GPU1 = M48 168.9/162.6 (+3.9%), M128 254.8/217.3 (+17%), M256 520.3/467.8 (+11%; GPU1 f=1 point looks a high outlier), M384 743.3/637.2 (+17%), M640 1231/1034 (+19%).
Time vs f is NOT smooth (Marlin tile/thread-config heuristic switches with N; e.g. M=48 f=1.04 slower than f=1.08; M=256 f=1.0 > f=1.04 on GPU1). The balance fits below are least-squares lines through the 7 f points per GPU (extrapolated to f up to ~1.09 as needed), so they carry ~±3% noise-from-config-jumps; a real implementation should pick f values/N that land on fast configs (per-N tile choice) rather than trust the line.

Full tables and balance: tables.md (copied below).

## Summary balance (cold = main; warm in tables.md)
| M | f0 (GPU0) | f1 (GPU1) | MLP saving per layer vs f=1 slower GPU |
|---|---|---|---|
| 48 | 0.98 | 1.02 | 2 us (1.2%) |
| 128 | 0.96 | 1.04 | 10 us (4.1%) |
| 256 | 0.94 | 1.06 | 46 us (8.9%) |
| 384 | 0.94 | 1.06 | 38 us (5.2%) |
| 640 | 0.92 | 1.08 | 101 us (8.2%) |
Warm: f0 .97/.91/.91/.93/.92 for M=48/128/256/384/640, saving 3/25/57/53/97 us (2.1/8.9/10.5/6.8/7.6%).
Interpretation: in compute-bound mixed steps (M>=256) a ~6-8% column shift to GPU1 saves ~40-100 us per layer MLP cold (~6-9%). With 64 layers, M=640: ~6.5 ms per forward of MLP critical path (if comms/other parts unchanged and GPU1's extra columns are summed in the all-reduce; extra weight memory on GPU1 = f1-1 of MLP ~ +8% of 64*2*... ~ +0.3 GB, GPU0 would drop its share equivalently). Decode M=48: almost no gain (GPU gap only 4%, memory-bound; ~2 us/layer).
Caveat: the f-dependence is bumpy, so the savings are +-30%; and the two GPUs ran concurrently (power/thermal as in serving, but no NCCL).

## Subrange-of-bigger-buffer test (f=1 gate_up, M=48, K=5120,N=17408, repacked int32 weight [320,69632])
Weight copied into a view at +16384 B offset of a larger buffer (different data_ptr), same scales. Output bitwise equal. Times (us, 3 reps): GPU0 cold own 110.6/110.6/110.6 vs view 110.9/110.7/110.8; warm 107.9-108.0 vs 108.1-108.2. GPU1 cold own 107.8/107.6/107.6 vs view 107.6/107.7/107.7. Difference <=0.3 us (<0.3%): none. (Only tests offset view of the whole repacked tensor; a real column-slice of the repacked layout is not contiguous and would need to be repacked per-slice: not tested.)


### cold (us per call)

| M | shape | GPU | f=0.88 | f=0.92 | f=0.96 | f=1.0 | f=1.04 | f=1.08 | f=1.12 |
|---|---|---|---|---|---|---|---|---|---|
| 48 | gate_up | 0 | 99.0 | 97.7 | 106.6 | 110.7 | 125.3 | 119.7 | 123.5 |
| 48 | gate_up | 1 | 97.0 | 96.4 | 103.4 | 107.6 | 121.5 | 115.5 | 118.5 |
| 48 | down | 0 | 51.5 | 53.9 | 55.6 | 58.2 | 61.7 | 61.6 | 65.1 |
| 48 | down | 1 | 49.6 | 51.6 | 52.7 | 55.0 | 58.7 | 59.3 | 61.7 |
| 48 | **sum** | 0 | 150.5 | 151.6 | 162.2 | 168.9 | 187.0 | 181.3 | 188.6 |
| 48 | **sum** | 1 | 146.6 | 148.0 | 156.1 | 162.6 | 180.2 | 174.8 | 180.3 |
| 128 | gate_up | 0 | 141.5 | 133.6 | 153.6 | 164.4 | 179.8 | 187.3 | 197.3 |
| 128 | gate_up | 1 | 121.1 | 109.4 | 127.3 | 138.9 | 166.0 | 162.1 | 179.2 |
| 128 | down | 0 | 83.3 | 76.8 | 85.1 | 90.4 | 98.4 | 117.9 | 109.4 |
| 128 | down | 1 | 70.8 | 64.4 | 71.1 | 78.3 | 88.1 | 110.3 | 103.9 |
| 128 | **sum** | 0 | 224.8 | 210.3 | 238.7 | 254.8 | 278.2 | 305.2 | 306.8 |
| 128 | **sum** | 1 | 191.9 | 173.8 | 198.4 | 217.3 | 254.2 | 272.4 | 283.1 |
| 256 | gate_up | 0 | 281.0 | 297.2 | 309.0 | 337.1 | 339.3 | 349.7 | 365.3 |
| 256 | gate_up | 1 | 230.6 | 245.2 | 258.0 | 305.3 | 289.1 | 301.2 | 323.6 |
| 256 | down | 0 | 165.3 | 152.4 | 179.4 | 183.3 | 182.2 | 210.6 | 197.4 |
| 256 | down | 1 | 142.7 | 126.7 | 158.5 | 162.5 | 156.5 | 187.6 | 180.0 |
| 256 | **sum** | 0 | 446.3 | 449.6 | 488.5 | 520.3 | 521.6 | 560.3 | 562.7 |
| 256 | **sum** | 1 | 373.3 | 371.8 | 416.5 | 467.8 | 445.5 | 488.8 | 503.6 |
| 384 | gate_up | 0 | 432.0 | 455.8 | 469.1 | 490.6 | 519.2 | 530.3 | 560.5 |
| 384 | gate_up | 1 | 359.7 | 379.6 | 391.4 | 413.9 | 462.9 | 456.2 | 496.1 |
| 384 | down | 0 | 228.0 | 226.3 | 245.6 | 252.8 | 286.1 | 304.5 | 301.6 |
| 384 | down | 1 | 190.3 | 186.3 | 209.2 | 223.2 | 251.9 | 269.8 | 258.9 |
| 384 | **sum** | 0 | 660.0 | 682.2 | 714.7 | 743.3 | 805.3 | 834.8 | 862.2 |
| 384 | **sum** | 1 | 550.0 | 565.9 | 600.5 | 637.2 | 714.8 | 726.0 | 755.0 |
| 640 | gate_up | 0 | 728.6 | 765.7 | 793.1 | 829.8 | 872.3 | 893.9 | 930.4 |
| 640 | gate_up | 1 | 607.1 | 638.9 | 664.3 | 704.8 | 747.0 | 769.3 | 801.9 |
| 640 | down | 0 | 355.1 | 371.4 | 381.1 | 401.4 | 416.6 | 433.7 | 448.0 |
| 640 | down | 1 | 291.7 | 304.7 | 313.9 | 329.3 | 343.5 | 356.4 | 369.4 |
| 640 | **sum** | 0 | 1083.7 | 1137.1 | 1174.2 | 1231.2 | 1288.9 | 1327.5 | 1378.5 |
| 640 | **sum** | 1 | 898.8 | 943.5 | 978.1 | 1034.1 | 1090.4 | 1125.6 | 1171.2 |

### Balance (cold): f0+f1=2, equalize gate_up+down

| M | T0(f=1) us | T1(f=1) us | f0 | f1 | balanced T us | saving vs slower GPU us | % |
|---|---|---|---|---|---|---|---|
| 48 | 168.9 | 162.6 | 0.982 | 1.018 | 166.9 | 2.0 | 1.2% |
| 128 | 254.8 | 217.3 | 0.964 | 1.036 | 244.4 | 10.4 | 4.1% |
| 256 | 520.3 | 467.8 | 0.939 | 1.061 | 474.0 | 46.3 | 8.9% |
| 384 | 743.3 | 637.2 | 0.941 | 1.059 | 704.9 | 38.4 | 5.2% |
| 640 | 1231.2 | 1034.1 | 0.917 | 1.083 | 1129.9 | 101.3 | 8.2% |

### warm (us per call)

| M | shape | GPU | f=0.88 | f=0.92 | f=0.96 | f=1.0 | f=1.04 | f=1.08 | f=1.12 |
|---|---|---|---|---|---|---|---|---|---|
| 48 | gate_up | 0 | 94.4 | 98.3 | 103.7 | 107.9 | 122.9 | 116.9 | 120.4 |
| 48 | gate_up | 1 | 91.5 | 96.5 | 102.2 | 106.4 | 120.9 | 115.7 | 118.6 |
| 48 | down | 0 | 41.0 | 42.8 | 43.7 | 45.3 | 47.0 | 47.8 | 49.4 |
| 48 | down | 1 | 34.4 | 35.9 | 36.6 | 38.1 | 39.5 | 40.2 | 41.7 |
| 48 | **sum** | 0 | 135.4 | 141.1 | 147.5 | 153.2 | 169.9 | 164.6 | 169.8 |
| 48 | **sum** | 1 | 125.9 | 132.4 | 138.8 | 144.5 | 160.4 | 155.9 | 160.3 |
| 128 | gate_up | 0 | 160.1 | 162.7 | 171.9 | 181.2 | 188.9 | 193.3 | 200.1 |
| 128 | gate_up | 1 | 134.2 | 136.6 | 144.4 | 152.7 | 162.5 | 162.7 | 169.5 |
| 128 | down | 0 | 89.7 | 91.9 | 96.0 | 99.2 | 104.5 | 105.0 | 108.1 |
| 128 | down | 1 | 74.8 | 76.7 | 80.2 | 83.0 | 87.5 | 87.9 | 90.8 |
| 128 | **sum** | 0 | 249.8 | 254.6 | 268.0 | 280.3 | 293.4 | 298.3 | 308.2 |
| 128 | **sum** | 1 | 209.0 | 213.3 | 224.6 | 235.7 | 250.0 | 250.6 | 260.2 |
| 256 | gate_up | 0 | 307.2 | 323.0 | 333.9 | 365.4 | 363.7 | 375.2 | 391.9 |
| 256 | gate_up | 1 | 256.6 | 271.8 | 281.4 | 328.5 | 305.9 | 321.0 | 330.5 |
| 256 | down | 0 | 158.7 | 166.5 | 170.2 | 178.1 | 185.9 | 189.6 | 197.4 |
| 256 | down | 1 | 132.4 | 138.9 | 142.0 | 149.0 | 155.7 | 158.7 | 165.2 |
| 256 | **sum** | 0 | 466.0 | 489.5 | 504.2 | 543.5 | 549.6 | 564.9 | 589.3 |
| 256 | **sum** | 1 | 389.0 | 410.7 | 423.5 | 477.5 | 461.6 | 479.8 | 495.7 |
| 384 | gate_up | 0 | 458.3 | 482.1 | 493.4 | 518.1 | 546.0 | 557.6 | 587.8 |
| 384 | gate_up | 1 | 389.9 | 414.1 | 423.6 | 449.0 | 489.2 | 487.0 | 516.8 |
| 384 | down | 0 | 232.5 | 244.2 | 248.1 | 259.6 | 271.4 | 279.0 | 290.6 |
| 384 | down | 1 | 194.0 | 203.7 | 207.1 | 217.3 | 227.2 | 233.7 | 243.4 |
| 384 | **sum** | 0 | 690.8 | 726.2 | 741.5 | 777.7 | 817.3 | 836.6 | 878.4 |
| 384 | **sum** | 1 | 583.9 | 617.9 | 630.7 | 666.4 | 716.4 | 720.6 | 760.2 |
| 640 | gate_up | 0 | 755.4 | 790.8 | 818.2 | 857.7 | 899.6 | 919.5 | 954.7 |
| 640 | gate_up | 1 | 644.6 | 680.8 | 707.3 | 745.8 | 784.5 | 804.9 | 835.9 |
| 640 | down | 0 | 380.0 | 399.3 | 410.9 | 430.3 | 445.8 | 461.3 | 476.7 |
| 640 | down | 1 | 317.2 | 333.2 | 343.8 | 360.1 | 373.2 | 386.2 | 399.4 |
| 640 | **sum** | 0 | 1135.4 | 1190.1 | 1229.1 | 1287.9 | 1345.4 | 1380.8 | 1431.3 |
| 640 | **sum** | 1 | 961.8 | 1013.9 | 1051.1 | 1106.0 | 1157.6 | 1191.1 | 1235.3 |

### Balance (warm): f0+f1=2, equalize gate_up+down

| M | T0(f=1) us | T1(f=1) us | f0 | f1 | balanced T us | saving vs slower GPU us | % |
|---|---|---|---|---|---|---|---|
| 48 | 153.2 | 144.5 | 0.971 | 1.029 | 150.0 | 3.3 | 2.1% |
| 128 | 280.3 | 235.7 | 0.909 | 1.091 | 255.5 | 24.9 | 8.9% |
| 256 | 543.5 | 477.5 | 0.914 | 1.086 | 486.2 | 57.2 | 10.5% |
| 384 | 777.7 | 666.4 | 0.926 | 1.074 | 724.8 | 52.9 | 6.8% |
| 640 | 1287.9 | 1106.0 | 0.923 | 1.077 | 1190.5 | 97.4 | 7.6% |