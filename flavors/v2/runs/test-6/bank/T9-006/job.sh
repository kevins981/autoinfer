#!/bin/bash
# usage: job.sh <tag> <smoke 0/1> [arms]
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/t9-tbo/env.sh >/dev/null 2>&1
D=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/bank/T9-006
export CE_BUILD=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-9.w1/ext6; mkdir -p $CE_BUILD
python $D/bench5.py 0 $2 $3 > $D/out_$1_r0.log 2>&1 &
P0=$!
sleep 3
python $D/bench5.py 1 $2 $3 > $D/out_$1_r1.log 2>&1 &
P1=$!
wait $P0; E0=$?; wait $P1; E1=$?
echo exit $E0 $E1; tail -30 $D/out_$1_r0.log; tail -12 $D/out_$1_r1.log
