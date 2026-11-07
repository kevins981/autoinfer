#!/bin/bash
# Official eval of one workspace: TRIALS (default 3) trials, each on a freshly started server
# (start, bench c=16 ShareGPT 400 prompts, stop), so restart-to-restart drift averages out.
# GSM8K-250 runs once, on the last trial's server. Score = mean over trials (summarize.py).
# Run inside a GPU job (official_eval does this). Frozen: agents must not modify.
# usage: eval.sh <workspace dir> <output dir> [extra serve args file]
# output: <out>/trial<t>/{server.log,startup.txt,bench_c16.json,bench.log,metrics_*.txt,gpu_clocks.csv},
#         <out>/server_cmd.txt, <out>/gsm8k.{json,log}
WS=$1; OUT=$2; EXTRA=${3:-}
TRIALS=${TRIALS:-3}
HARNESS=$(cd "$(dirname "$0")" && pwd)
TASK=$(basename "$(dirname "$HARNESS")")
PORT=8100
mkdir -p "$OUT"
SRV=; CLK=
cleanup() { [ -n "$CLK" ] && kill $CLK 2>/dev/null; [ -n "$SRV" ] && { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }; }
trap cleanup EXIT
fail() { echo "$1" > "$OUT/failed_stage"; echo "FAILED at $1"; exit 1; }

source "$WS/env.sh" || fail env
BASELINE_TREE=$CACHE_ROOT/tasks/$TASK/baseline/vllm      # trusted client code (not the record's)
source "$HARNESS/serve_args.sh"
EXTRA_ARGS=()
if [ -n "$EXTRA" ] && [ -s "$EXTRA" ]; then
  mapfile -t EXTRA_ARGS < <(python -c 'import shlex,sys; print("\n".join(shlex.split(open(sys.argv[1]).read())))' "$EXTRA")
fi
SERVE=(vllm serve "$MODEL" --port $PORT "${SERVE_ARGS[@]}" "${EXTRA_ARGS[@]}")
{ printf '%q ' "${SERVE[@]}"; echo; python -c 'import vllm; print("server code:", vllm.__file__)'; } > "$OUT/server_cmd.txt"
DATA=$(ls "$HF_HUB_CACHE"/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/*/ShareGPT_V3_unfiltered_cleaned_split.json | head -1)

start_server() {   # $1 = trial dir; the server runs the workspace's code
  "${SERVE[@]}" > "$1/server.log" 2>&1 &
  SRV=$!
  local T0=$(date +%s)
  until curl -sf localhost:$PORT/health >/dev/null; do
    kill -0 $SRV 2>/dev/null || fail "server_start (trial $t)"
    [ $(( $(date +%s) - T0 )) -gt 900 ] && fail "server_start_timeout (trial $t)"
    sleep 2
  done
  echo "server ready after $(( $(date +%s) - T0 ))s" | tee "$1/startup.txt"
}
stop_server() {
  kill -INT $SRV 2>/dev/null
  for _ in $(seq 60); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
  kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=
  for _ in $(seq 30); do curl -sf localhost:$PORT/health >/dev/null || break; sleep 1; done
}

for t in $(seq 1 $TRIALS); do
  D=$OUT/trial$t; mkdir -p "$D"
  start_server "$D"
  # clients run the baseline code: a record's diff can't change how it is measured
  curl -s localhost:$PORT/metrics > "$D/metrics_before.txt"
  nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 30 > "$D/gpu_clocks.csv" &
  CLK=$!
  PYTHONPATH=$BASELINE_TREE vllm bench serve --port $PORT \
    --backend openai-chat --endpoint /v1/chat/completions --model "$MODEL" \
    --dataset-name sharegpt --dataset-path "$DATA" \
    --num-prompts 400 --max-concurrency 16 --request-rate inf \
    --seed 0 --temperature 0 --num-warmups 2 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$D" --result-filename bench_c16.json \
    > "$D/bench.log" 2>&1 || fail "bench (trial $t)"
  kill $CLK 2>/dev/null; CLK=
  curl -s localhost:$PORT/metrics > "$D/metrics_after.txt"
  grep "Output token throughput" "$D/bench.log"
  if [ "$t" -eq "$TRIALS" ]; then
    PYTHONPATH=$BASELINE_TREE python "$HARNESS/gsm8k.py" --n 250 --url http://localhost:$PORT --out "$OUT/gsm8k.json" \
      > "$OUT/gsm8k.log" 2>&1 || fail gsm8k
    tail -1 "$OUT/gsm8k.log"
  fi
  stop_server
done
echo "eval done"
