#!/bin/bash
cd /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-49/comp
source ../../../task/env.sh
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem --format=csv
CUDA_VISIBLE_DEVICES=0 ./cmp
