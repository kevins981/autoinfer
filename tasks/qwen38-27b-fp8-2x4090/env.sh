# Environment for the qwen task: source this before building or running the engine.
# Reuses qwen38_opt's conda env and model weights READ-ONLY; every cache goes to autoinfer/.cache.
export AUTOINFER_ROOT=/scratch/kevinsong/learning/autoinfer
export QWEN38_OPT=/scratch/kevinsong/learning/qwen38_opt
export ENV_PREFIX=$QWEN38_OPT/env              # read-only: never install into it
export CACHE_ROOT=$AUTOINFER_ROOT/.cache

# Build caches
export CCACHE_DIR=$CACHE_ROOT/ccache
export CCACHE_BASEDIR=$AUTOINFER_ROOT          # hits across workspaces at different paths
export CCACHE_NOHASHDIR=true
export PIP_CACHE_DIR=$CACHE_ROOT/pip UV_CACHE_DIR=$CACHE_ROOT/uv CONDA_PKGS_DIRS=$CACHE_ROOT/conda-pkgs
export TMPDIR=$CACHE_ROOT/tmp

# Model weights and datasets: read from qwen38_opt's HF cache, offline (no downloads, no writes)
export HF_HOME=$CACHE_ROOT/huggingface
export HF_HUB_CACHE=$QWEN38_OPT/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Runtime caches
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/inductor
export VLLM_CACHE_ROOT=$CACHE_ROOT/vllm
export FLASHINFER_WORKSPACE_BASE=$CACHE_ROOT/flashinfer
export CUDA_CACHE_PATH=$CACHE_ROOT/nv-compute
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch_extensions   # torch cpp_extension load()/load_inline JIT builds
export PYTHONNOUSERSITE=1

# Build settings: RTX 4090 = sm_89; moderate parallelism (shared server)
export TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=16 NVCC_THREADS=2

mkdir -p $CCACHE_DIR $TMPDIR $HF_HOME $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $VLLM_CACHE_ROOT \
         $FLASHINFER_WORKSPACE_BASE $CUDA_CACHE_PATH $XDG_CACHE_HOME $TORCH_EXTENSIONS_DIR

# Activate the env + its CUDA toolkit (overrides /usr/local/cuda 12.6). See qwen38_opt/env.sh.
_autoinfer_flags=$-; set +u                   # conda's activate scripts break under set -u
source /home/kevinsong/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
[[ $_autoinfer_flags == *u* ]] && set -u; unset _autoinfer_flags
export CUDA_HOME=$ENV_PREFIX PATH=$ENV_PREFIX/bin:$PATH
export CC=$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export CUDA_TARGET_DIR=$ENV_PREFIX/targets/x86_64-linux CUDA_INC_PATH=$ENV_PREFIX/targets/x86_64-linux/include
export LIBRARY_PATH=$ENV_PREFIX/lib/stubs:$ENV_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
# Nsight Compute's Python API (`import ncu_report`) for reading .ncu-rep files
export NCU_PYTHON=$(ls -d "$ENV_PREFIX"/nsight-compute-*/extras/python 2>/dev/null | head -1)
