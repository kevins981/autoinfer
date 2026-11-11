#!/bin/bash
R=$AUTOINFER_RUN; WS=$R/workspaces/t47-waves2; d=$R/agents/leader-47.w1/${RUNDIR:-s1}; mkdir -p $d; rm -f $d/*
export OUT=$d T47_WAVE_BENCH=1 T47_WAVE_REPS=${REPS:-12} T47_WAVE_TRACE=$d
source $WS/env.sh || exit 1
source $R/task/scripts/lib.sh
python -c "import torch, vllm._custom_ops; from vllm.model_executor.layers.mamba.gdn.replay import gdn_replay_op_available as f; print('gdn_replay_op_available', f())" 2>&1 | grep gdn_replay
eval "EXTRA=( $(cat $R/bank/T45-005/serve_args.txt) )"
start_server "${EXTRA[@]}" || exit 1
timeout ${CLIENT_T:-240} python $R/agents/leader-47.w1/drive.py $PORT $MODEL ${PROBES:-256,128,512,64} 2>&1 | tee $d/client.log
grep -h "T47PLAN\|T47BENCH\|T47TRACE\|T47SUM\|T47CHECK\|Traceback\|Error\|gdn_replay\|two.slot\|two_slot" $d/server.log | cut -c1-600 | head -120
stop_server
