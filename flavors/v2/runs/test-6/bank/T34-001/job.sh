#!/bin/bash
R=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6
source $R/workspaces/t34-ar/env.sh >/dev/null 2>&1
D=$R/agents/leader-34.w1/h
python $D/bench2.py 0 > $D/out_$1_r0.log 2>&1 &
P0=$!
sleep 3
python $D/bench2.py 1 > $D/out_$1_r1.log 2>&1 &
P1=$!
wait -n $P0 $P1
sleep 6
kill -9 $P0 $P1 2>/dev/null
echo done; grep -E "timing|TIME|CHECK|TRACE|Error|=====" $D/out_$1_r0.log | tail -70; echo ---; grep -E "TIME|TRACE|Error" $D/out_$1_r1.log | tail -40
