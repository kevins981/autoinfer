# FOCUS (run framework-eval)
Whole-engine speedup of the official metric (output tok/s at concurrency 16) for Qwen3.8-27B-FP8
on 2x RTX 4090. Everything TASK.md lists as open is in scope (kernels, scheduler, memory,
communication, compilation, parallelism, serving flags, exact speculative decoding, max context /
KV budget), within the constraints of task_definition.md.
