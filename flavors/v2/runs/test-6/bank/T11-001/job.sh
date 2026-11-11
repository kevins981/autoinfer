#!/bin/bash
# usage: job.sh <tag> [smoke]
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/t11-lane/env.sh
D=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-11.w1
SHM=/dev/shm/lane_$$_$RANDOM
python $D/lane.py 0 $SHM $2 > $D/out_$1_r0.log 2>&1 &
P0=$!
sleep 3
python $D/lane.py 1 $SHM $2 > $D/out_$1_r1.log 2>&1 &
P1=$!
wait $P0; E0=$?; wait $P1; E1=$?
echo exit $E0 $E1; tail -40 $D/out_$1_r0.log; tail -25 $D/out_$1_r1.log
