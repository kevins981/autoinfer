#!/bin/bash
# usage: job.sh <tag> <smoke 0/1> [arms]
R=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6
source $R/workspaces/t16/env.sh >/dev/null 2>&1
D=$R/agents/leader-16.w1/h
python $D/bench.py 0 $2 $3 > $D/out_$1_r0.log 2>&1 &
P0=$!
sleep 3
python $D/bench.py 1 $2 $3 > $D/out_$1_r1.log 2>&1 &
P1=$!
wait $P0; E0=$?; wait $P1; E1=$?
echo exit $E0 $E1; tail -40 $D/out_$1_r0.log; tail -15 $D/out_$1_r1.log
