#!/bin/bash
# T14-002: in-situ A/B (A: AI_POLICY=0 champion, B: policy on), nsys, 64 prompts c=16, order A B B A (fallback A B A)
export AUTOINFER_RUN=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6
WS=$AUTOINFER_RUN/workspaces/leader-14.m1
source $WS/env.sh || exit 1
OUTD=$AUTOINFER_RUN/agents/leader-14.w1/insitu
echo "KB61: $(ls -laL $WS/vllm/vllm/_C_stable_libtorch.abi3.so)"
python -c "
from vllm.model_executor.layers.mamba.gdn.replay import gdn_replay_op_available as f
import vllm._C_stable_libtorch as c; print('gdn_replay_op_available =', f(), c.__file__)"
eval "CH=($(cat $AUTOINFER_RUN/bank/T9-005/serve_args.txt))"
T0=$SECONDS
arm() { # name policyval
  if [ "$2" = 0 ]; then export AI_POLICY=0; else unset AI_POLICY; fi
  echo "=== arm $1 start t=$((SECONDS-T0))"
  TRACE=node bash $AUTOINFER_RUN/task/scripts/profile_nsys.sh $WS $OUTD/$1 64 "${CH[@]}" > $OUTD/$1.log 2>&1
  echo "=== arm $1 done t=$((SECONDS-T0))"
  rm -f $OUTD/$1/profile.nsys-rep
}
arm A1 0
arm B1 1
if [ $((SECONDS-T0)) -lt 330 ]; then arm B2 1; fi
arm A2 0
