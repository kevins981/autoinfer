# 4-CTA/SM variant in chain (job 438, 1.7 GPU-min; 7.5 left on the team)
New configs added to bs_gemm.cu (my workspace leader-51.pig only): base ids 32 = (2,2,32,32,2,64), 33 = (2,2,32,32,3,64), and epilogue versions 132, 133 (BMAR). Built outside the GPU job (82 s).
cuobjdump --dump-resource-usage: cfg132 126 regs, 16 KB smem/CTA(static field shows 128 B), STACK 0, LOCAL 0 (no spills); cfg133 124 regs, 24 KB smem, no spills. cfg127 (the table's gate_up pick) 136 regs, cfg103 240 regs.
Regs per CTA (128 thr): 132 -> 126 -> 16384 -> 4 CTAs/SM (512 slots); 127 -> 136 -> 17408 -> 3/SM (384, matches the assumption); 103 -> 240 -> 2/SM (256, matches).
Chain table: a = champion dispatch (M1 variant a); b132 = gate_up cfg 132 + down cfg 127 split-K 3; b133 = gate_up cfg 133 + down cfg 127 split-K 2. qkvz/gdn_out as in a. us/layer, GEMM profiler means:
var  M    layer  qkvz  gdn_out gate_up down | sum   dms/step vs a64  us/row
a    64   342.0  37.3  21.5    82.1   53.4 | 194.3  0     -
a    128  413.4  46.2  27.4   117.4   72.9 | 263.9  4.45  69.6
a    160  475.2  67.1  27.7   159.1   74.6 | 328.5  8.59  89.5
a    192  485.7  72.8  28.2   162.8   74.8 | 338.6  9.24  72.2
b132 64   344.2  37.4  21.5    84.0   53.6 | 196.5  0.14  -
b132 128  441.9  46.6  28.8   134.7   83.1 | 293.2  6.33  98.9
b132 160  487.6  67.0  30.1   158.8   83.9 | 339.8  9.31  97.0
b132 192  517.2  73.2  31.2   175.4   89.8 | 369.6  11.22 87.7
b133 64   344.2  37.2  21.5    83.8   53.5 | 196.0  0.11  -
b133 128  406.8  46.5  28.9   118.7   64.9 | 259.0  4.14  64.7
b133 160  483.4  67.2  28.7   157.8   82.7 | 336.4  9.09  94.7
b133 192  516.4  73.2  29.1   167.1   98.1 | 367.5  11.08 86.6
