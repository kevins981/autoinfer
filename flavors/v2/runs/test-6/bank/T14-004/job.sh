#!/bin/bash
export NP=300; D=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-14.w1/abba
T0=$SECONDS; i=0
for arm in A B B A; do
  i=$((i+1))
  echo "stray check (log only):"; pgrep -u $USER -fa "vllm (serve|bench)" | cut -c1-120 | head -3
  g=0
  bash $D/arm.sh $arm $D/w3_${i}$arm $g
  echo "=== start $i $arm done t=$((SECONDS-T0))"
done
