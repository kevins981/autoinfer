#!/bin/bash
# T1-017 FP8-activation accuracy A/B: GSM8K-250 with bf16-activation reference (VLLM_W8A8_PREFILL=0) and with W8A8 (default), same code, same job.
WS=$AUTOINFER_RUN/workspaces/t1-w12
export OUT=$AUTOINFER_RUN/agents/leader-1/gsmab
source "$WS/env.sh"
source "$AUTOINFER_RUN/task/scripts/lib.sh"
mapfile -t ARGS < <(python -c 'import shlex,sys; print("\n".join(shlex.split(open(sys.argv[1]).read())))' "$AUTOINFER_RUN/bank/T1-017/serve_args.txt")
export VLLM_W8A8_PREFILL=0
export OUT=$AUTOINFER_RUN/agents/leader-1/gsmab/ref; mkdir -p $OUT; start_server "${ARGS[@]}" || exit 1
gsm8k 250 "$AUTOINFER_RUN/agents/leader-1/gsmab/ref_gsm8k.json"
stop_server
export VLLM_W8A8_PREFILL=1
export OUT=$AUTOINFER_RUN/agents/leader-1/gsmab/w8a8; mkdir -p $OUT; start_server "${ARGS[@]}" || exit 1
gsm8k 250 "$AUTOINFER_RUN/agents/leader-1/gsmab/w8a8_gsm8k.json"
stop_server
grep -h "w8a16\|W8A8" $AUTOINFER_RUN/agents/leader-1/gsmab/*/server.log | sort | uniq -c | head
