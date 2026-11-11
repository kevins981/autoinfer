#!/bin/bash
source /scratch/kevinsong/learning/autoinfer/flavors/v1/runs/test-5/workspaces/eval-T1-020-1/env.sh
D=/scratch/kevinsong/learning/autoinfer/flavors/v1/runs/test-5/agents/bs-r7-adjacent_field/ceov
CUDA_VISIBLE_DEVICES=0 python $D/ceov.py $D/${1:-r1}
