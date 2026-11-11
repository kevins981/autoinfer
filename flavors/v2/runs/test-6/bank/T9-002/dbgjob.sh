#!/bin/bash
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/t9-tbo/env.sh >/dev/null 2>&1
python /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/bank/T9-002/dbg.py 2>&1 | grep -v "W1109\|FP8"
