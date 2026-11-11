#!/bin/bash
export AUTOINFER_RUN=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6
source $AUTOINFER_RUN/workspaces/t17-k/env.sh
cd $AUTOINFER_RUN/agents/leader-17.w1
VARS=${VARS:-0} python tm17.py
