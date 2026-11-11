#!/bin/bash
# usage: job.sh <tag> <smoke 0/1> <Rs>
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/t7-ovl/env.sh
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PWD}
D=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-7.w1
SHM=/dev/shm/t7ovl_$$_$RANDOM
python $D/bench.py 0 $SHM $2 $3 > $D/out_$1_r0.log 2>&1 &
P0=$!
sleep 3
python $D/bench.py 1 $SHM $2 $3 > $D/out_$1_r1.log 2>&1 &
P1=$!
wait $P0; E0=$?; wait $P1; E1=$?
echo exit $E0 $E1; tail -40 $D/out_$1_r0.log; tail -40 $D/out_$1_r1.log
