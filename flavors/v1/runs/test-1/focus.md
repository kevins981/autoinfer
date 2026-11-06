# FOCUS
Speed up the FFN (MLP) layers of Qwen3.8-27B-FP8 on 2x4090: gate/up/down projections (hidden 5120, intermediate 17408, TP=2) and anything around them that makes them faster (Marlin FP8 GEMM kernels, fusion of gate/up + activation, weight layout, bytes read, the all-reduce after down_proj, overlap of comm with compute, CUDA graph/scheduling effects on FFN). Attention/GDN/lm_head work is out of scope unless it directly enables an FFN gain.

Clarification (human, round 1): "FFN only" is the GOAL (make the FFN layer faster), not a restriction on where changes may be made. Work touching other components (comm, scheduling, spec-decode, engine) is fine if it speeds up the FFN.
