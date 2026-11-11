# T29 task 2: T29_W8A8 gate in-server (leader-29.w1), workspace t29-fp8 (T27-002 + gate)
Change (Python + JSON only, no build):
- vllm/model_executor/kernels/linear/__init__.py: init_fp8_linear_kernel forces TritonFp8BlockScaledMMKernel for
  per-group(128) act x 128x128 block weights when T29_W8A8=1 (logs "T29_W8A8=1: ..."). Applies to every Fp8LinearMethod
  block linear = target + MTP-layer linears. Untouched: lm_head (unquantized), embeddings, bf16 in_proj_a/b,
  draft lm_head (Fp8PtpcOnlineLinearMethod -> Cutlass PTPC, still selected), T27 screen.
- configs/N=..,K=..,device_name=NVIDIA_GeForce_RTX_4090,dtype=fp8_w8a8,block_shape=[128,128].json (5 files: best GPU0
  cold config per M in {48..1024} from task-1 sweep; nearest-M lookup at call/capture time, no autotune).
- fp8_utils.w8a8_triton_block_scaled_mm: assert config BLOCK_SIZE_K == block_k (128).
- vllm/envs.py compile_factors: adds T29_W8A8 to the torch.compile cache key only when set (gate-off hash unchanged).
Gate off: no code path changes (Marlin selected as before) -> champion bitwise.
Jobs: 192 (order B,A + gsm8k50 on B), 195 (order A,B). Per-start table in report.
