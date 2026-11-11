#!/bin/bash
# T42 stage 2b: nsys of A (T37-002) and B (t42-cs), 48 prompts each, same job.
R=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6
ARGS="$(cat $R/bank/T37-002/serve_args.txt)"
for arm in A B; do
  if [ $arm = A ]; then WS=$R/workspaces/t42-a; else WS=$R/workspaces/t42-cs; fi
  eval "EXTRA=( $ARGS )"
  ( bash $R/task/scripts/profile_nsys.sh $WS $R/agents/leader-42.w1/out/s2b/$arm 48 "${EXTRA[@]}" ) 2>&1 | grep -E "\[prof\]|died|error" 
done
