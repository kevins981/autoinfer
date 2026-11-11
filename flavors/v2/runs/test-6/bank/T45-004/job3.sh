#!/bin/bash
cd /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-45/rowband
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/t35-stream/env.sh
export CUDA_VISIBLE_DEVICES=0
timeout 300 python shapes.py 2>&1 | grep -v -i warn | tail -12
for m in 64 80 96 112 128; do
  echo "=== chain GU=bs M=$m"; GU=bs STANDALONE=0 PROF=1 NL=64 M=$m OUT=res_bs_M$m.json timeout 120 python bench_m.py 2>&1 | grep "bs cfgs\|us/layer\|prof\|Error" | head -5
done
