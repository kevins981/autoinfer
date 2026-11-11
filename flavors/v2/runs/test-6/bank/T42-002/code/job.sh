#!/bin/bash
# usage: job.sh <out> <nbatches> <timing>
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/task/env.sh
cd /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-42.w1/cs
export CUDA_VISIBLE_DEVICES=0
python cs_test.py "$@"
