#!/bin/bash
R=$AUTOINFER_RUN; D=$R/agents/leader-47.w1/mb
source $R/workspaces/t47-waves2/env.sh || exit 1
export CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.9 OUT=$D/res_${PART:-12}.json
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem --format=csv
timeout 300 python $D/bench.py 2>&1 | grep -v W1111
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem --format=csv
