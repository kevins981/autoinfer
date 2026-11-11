#!/bin/bash
cd /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-45/rowband
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/t35-stream/env.sh
export CUDA_VISIBLE_DEVICES=0 STANDALONE=0 PROF=1 NL=64
for m in 48 64 80 96 112 128; do
  echo "=== M=$m"; M=$m OUT=res_M$m.json timeout 150 python bench_m.py 2>&1 | grep -v Warning | tail -6
done
