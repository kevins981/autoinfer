#!/bin/bash
# usage: arm.sh <A|B> <outdir> <gsm8k:0|1>
export AUTOINFER_RUN=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6
arm=$1; OUT=$2; G=$3; export OUT; mkdir -p $OUT
if [ $arm = A ]; then WS=$AUTOINFER_RUN/workspaces/leader-14.champ; else WS=$AUTOINFER_RUN/workspaces/leader-14.m1; fi
source $WS/env.sh || exit 1
source $AUTOINFER_RUN/task/scripts/lib.sh
unset AI_POLICY AI_LOG; unset VLLM_T9_PP_ROWS
python -c "
import vllm._C_stable_libtorch as c, vllm; print('ARM $arm _C =', c.__file__, 'vllm =', vllm.__file__)"
eval "EXTRA=( $(cat $AUTOINFER_RUN/bank/T9-005/serve_args.txt) )"
start_server_bench ${NP:-300} 16 $OUT "${EXTRA[@]}"
[ "$G" = 1 ] && gsm8k 50 $OUT/gsm8k50.json
stop_server
