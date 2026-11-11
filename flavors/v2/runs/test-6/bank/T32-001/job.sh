#!/bin/bash
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/task/env.sh >/dev/null 2>&1
cd /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-32/rsag
python -I rsag.py 128 256 512 734 1024
