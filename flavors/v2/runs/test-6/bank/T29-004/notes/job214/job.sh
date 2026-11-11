#!/bin/bash
# T29 task 5 rerun (job 212 killed mid-A): GSM8K-1319 FP8-activation A/B for T29-004, no 128K.
# B = workspace default (T29_W8A8 unset -> mode 2): gsm8k 1319, long_request 128000 (+ health check);
# A = same code, T29_W8A8=0 (exact champion path = bf16-activation reference): gsm8k 1319.
# Client invocations as eval.sh: baseline tree on PYTHONPATH, harness scripts.
R=$AUTOINFER_RUN; WS=$R/workspaces/t29-fp8; D=$R/agents/leader-29.w1/t5r
source $WS/env.sh
TASKN=$(basename "$R/task"); BASELINE_TREE=$CACHE_ROOT/tasks/$(basename $(readlink -f $R/task 2>/dev/null || echo $R/task))/baseline/vllm
[ -d "$BASELINE_TREE" ] || BASELINE_TREE=$(ls -d $CACHE_ROOT/tasks/*/baseline/vllm | head -1)
echo "client tree: $BASELINE_TREE"
H=$R/task/harness
source $R/task/scripts/lib.sh
PORT=${PORT:-8000}
ARGS=$(cat $R/bank/T27-002/serve_args.txt)
eval "SARGS=($ARGS)"
run_side() {  # $1 = label, $2 = T29_W8A8 value or "default", $3 = do long request
  mkdir -p $D/$1; export OUT=$D/$1
  if [ "$2" = default ]; then unset T29_W8A8; else export T29_W8A8=$2; fi
  echo "== $1 (T29_W8A8=${T29_W8A8-unset}) $(date +%T)"
  start_server "${SARGS[@]}" || { echo "server failed"; return 1; }
  grep -hE "T29_W8A8|not eligible|KV cache size|Model loading took|JIT compilation" $D/$1/server.log | cut -c1-200 | head -6
  PYTHONPATH=$BASELINE_TREE python $H/gsm8k.py --n 1319 --url http://localhost:$PORT --out $D/$1/gsm8k.json > $D/$1/gsm8k.log 2>&1
  tail -2 $D/$1/gsm8k.log | head -1; date +%T
  if [ "$3" = 1 ]; then
    PYTHONPATH=$BASELINE_TREE python $H/long_request.py --tokens 128000 --url http://localhost:$PORT \
      --model "$MODEL" --out $D/$1/long_request.json > $D/$1/long_request.log 2>&1
    tail -1 $D/$1/long_request.log
  fi
  stop_server
}
run_side B default 0   # 128K skipped: passed in job 212 on identical code
run_side A 0 0
python - <<PY
import json
b=json.load(open("$D/B/gsm8k.json")); a=json.load(open("$D/A/gsm8k.json"))
d=100*(b["accuracy"]-a["accuracy"])
print(f"GSM8K-1319: B (W8A8 default) {100*b['accuracy']:.2f}%  A (T29_W8A8=0, bf16 act) {100*a['accuracy']:.2f}%  diff {d:+.2f} pts -> {'PASS' if abs(d)<=2.0 else 'FAIL'} (|diff|<=2.0)")
same=sum(x['out']==y['out'] for x,y in zip(a['items'],b['items'])); print(f"identical outputs {same}/{len(a['items'])}")
PY
