#!/bin/bash
# usage: job.sh <sequence e.g. B or ABBA> <num_prompts> <outdir>
SEQ=$1; NP=$2; BASE=$3
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/leader-12.g1/env.sh
source $AUTOINFER_RUN/task/scripts/lib.sh
python -c "from vllm.model_executor.layers.mamba.gdn.replay import gdn_replay_op_available as f; print('gdn_replay_op_available:', f())"
SA=$(cat $AUTOINFER_RUN/bank/T1-042/serve_args.txt)
n=0
for L in $(echo $SEQ | fold -w1); do
  n=$((n+1)); OUT=$BASE/$n$L; mkdir -p $OUT
  export AUTOINFER_TIMERS=$OUT/timers
  if [ $L = A ]; then SPEC='{"method":"mtp","num_speculative_tokens":2,"use_local_argmax_reduction":true,"draft_lm_head_quantization":"fp8","draft_lm_head_vocab_size":98304}'
  else
    FCV=true; [ $L = C ] && FCV=false
    SPEC='{"method":"mtp","num_speculative_tokens":2,"use_local_argmax_reduction":true,"draft_lm_head_quantization":"fp8","draft_lm_head_vocab_size":32768,"draft_lm_head_token_file":"'$AUTOINFER_RUN'/bank/T12-002/draft_vocab_32768.npy","draft_fc_fp8":'$FCV'}'
  fi
  eval "ARGS=($SA)"
  # drop the champion's --speculative-config (last flag, replaced by ours)
  NEWARGS=(); skip=0
  for a in "${ARGS[@]}"; do if [ $skip = 1 ]; then skip=0; continue; fi; if [ "$a" = "--speculative-config" ]; then skip=1; continue; fi; NEWARGS+=("$a"); done
  echo "=== $n $L"
  start_server_bench $NP 16 $OUT/bench "${NEWARGS[@]}" --speculative-config "$SPEC"
  grep -E "Fp8|fc:|Drafter lm_head|Quantized a" $OUT/server.log | head -5
  stop_server
done
