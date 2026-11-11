# T29 task 3 (leader-29.w1): nsys attribution, job 199 (5.2 GPU-min), B (T29_W8A8=1 + warmup fix) then A (gate off), 48 prompts each,
# TRACE=node, champion serve_args. Exports done outside the job. Analysis: attr.py -> attr_gpu0.txt, attr_gpu1.txt.
# Warmup fix: kernels/linear/scaled_mm/triton.py _t29_warmup (per (N,K): every table config at M=key and key+1, plus M=1,16,17).
# Result: no _w8a8 JIT during inference in B (--jit-monitor-verbose); only kernel_unified_attention JIT, as in A.
