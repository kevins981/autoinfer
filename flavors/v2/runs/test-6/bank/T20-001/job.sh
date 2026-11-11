#!/bin/bash
# usage: job.sh <tag> <smoke 0/1> [arms]   (env: STAGES, STRESS_ARS, STRESS_CFGS, BO)
R=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6
source $R/workspaces/t20/env.sh >/dev/null 2>&1
D=$R/agents/leader-20.w1/g1
python $D/bench.py 0 $2 $3 > $D/out_$1_r0.log 2>&1 &
P0=$!
sleep 3
python $D/bench.py 1 $2 $3 > $D/out_$1_r1.log 2>&1 &
P1=$!
wait $P0; E0=$?; wait $P1; E1=$?
echo exit $E0 $E1; grep -v "^\[rank\|Warning\|warn" $D/out_$1_r0.log | tail -50; grep -v "^\[rank" $D/out_$1_r1.log | tail -25
