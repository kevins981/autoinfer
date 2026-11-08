# Sourced by `ws build` (after the workspace's env.sh) for CMake builds of the engine; not by inference servers.
# vLLM's CMake configure git-clones ~8 external projects (2.9 GB, ~11 min) into every new build
# folder; none but CUTLASS and vllm-flash-attn even compile for sm_89. Point it at one shared,
# read-only copy of the exact pinned sources (copied from a configure of this vLLM commit).
# Source it yourself before running cmake by hand in a fresh build folder.
DEPS=$CACHE_ROOT/tasks/qwen38-27b-fp8-2x4090/deps
if [ -d "$DEPS" ]; then
  export VLLM_CUTLASS_SRC_DIR=$DEPS/cutlass-src VLLM_FLASH_ATTN_SRC_DIR=$DEPS/vllm-flash-attn-src
  export DEEPGEMM_SRC_DIR=$DEPS/deepgemm-src DEEPSELECT_SRC_DIR=$DEPS/deepselect-src
  export FLASH_MLA_SRC_DIR=$DEPS/flashmla-src FLASH_KDA_SRC_DIR=$DEPS/flashkda-src
  export FMHA_SM100_SRC_DIR=$DEPS/fmha_sm100-src TML_FA4_SRC_DIR=$DEPS/tml_fa4-src
  export TRITON_KERNELS_SRC_DIR=$DEPS/triton_kernels-src/python/triton_kernels/triton_kernels
fi
