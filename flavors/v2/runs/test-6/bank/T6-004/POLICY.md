# T6 Marlin policy (workspace t6-pol, marlin.cu t6_knobs; AI_POLICY unset = on, 0 = stock, manual = raw env knobs)
Shape class by weight size (N*K) / K: 0 gate_up-like (>=60M), 1 down-like (K>=8000), 2 qkvz/attn_qkv (>=30M), 3 o_proj/gdn_out (<30M).
M<=48: stock (bitwise identical).
49-64: classes 2,3: 2x32 parallel slabs in one launch; classes 0,1: stock.
65-128: all classes: parallel slabs in one launch (S=ceil(M/64) slabs of round16(ceil(M/S)) rows, tail rows masked in-kernel); class 0 additionally slab-INNER tile order.
129-150: class 0: slab + inner; others stock.
>150: classes 0,3: parallel slabs (AI_SLAB path); others stock.
Not used (measured invalid or no gain): stages=3 kernels with thread_k=64 (wrong results), bps=2, (128,128,256) 8-warp tiles.
