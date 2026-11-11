#!/bin/bash
cd /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-45/rowband
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/t35-stream/env.sh
export CUDA_VISIBLE_DEVICES=0
timeout 300 python gu.py 2>&1 | grep -v -i warn | tail -8
for m in 48 64 80 96; do
  echo "=== chain GU=t31 M=$m"; GU=t31 STANDALONE=0 PROF=1 NL=64 M=$m OUT=res_t31gu_M$m.json timeout 120 python bench_m.py 2>&1 | grep "us/layer\|prof\|Error" | head -4
done
