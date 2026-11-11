#!/bin/bash
# usage: job.sh <tag> <smoke 0/1> <Rs> [arms]
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/t9-tbo/env.sh >/dev/null 2>&1
D=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/bank/T9-002
export CE_BUILD=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-9.w1/ext
SHM=/dev/shm/t9ce2_$$_$RANDOM
python $D/bench2.py 0 $SHM $2 $3 $4 $5 > $D/out_$1_r0.log 2>&1 &
P0=$!
sleep 3
python $D/bench2.py 1 $SHM $2 $3 $4 $5 > $D/out_$1_r1.log 2>&1 &
P1=$!
wait $P0; E0=$?; wait $P1; E1=$?
echo exit $E0 $E1; tail -30 $D/out_$1_r0.log; tail -12 $D/out_$1_r1.log
