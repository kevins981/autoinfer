#!/bin/bash
source /scratch/kevinsong/learning/autoinfer/flavors/v4_unofficial_eval/runs/review-eval/workspaces/t8-champ/env.sh
cd /scratch/kevinsong/learning/autoinfer/flavors/v4_unofficial_eval/runs/review-eval/agents/leader-8/mb
CUDA_VISIBLE_DEVICES=0 python f16acc.py
