#!/bin/bash
# Official eval of one workspace: start the server from it, bench c=16 (ShareGPT, 400 prompts),
# GSM8K-250, stop. Run inside a GPU job (official_eval does this). Frozen: agents must not modify.
# usage: eval.sh <workspace dir> <output dir> [extra serve args file]
WS=$1; OUT=$2; EXTRA=${3:-}
HARNESS=$(cd "$(dirname "$0")" && pwd)
TASK=$(basename "$(dirname "$HARNESS")")
PORT=8100
mkdir -p "$OUT"
fail() { echo "$1" > "$OUT/failed_stage"; echo "FAILED at $1"; exit 1; }

source "$WS/env.sh" || fail env
BASELINE_TREE=$CACHE_ROOT/tasks/$TASK/baseline/vllm      # trusted client code (not the record's)
source "$HARNESS/serve_args.sh"
EXTRA_ARGS=()
if [ -n "$EXTRA" ] && [ -s "$EXTRA" ]; then
  mapfile -t EXTRA_ARGS < <(python -c 'import shlex,sys; print("\n".join(shlex.split(open(sys.argv[1]).read())))' "$EXTRA")
fi

# --- server (the workspace's code)
SERVE=(vllm serve "$MODEL" --port $PORT "${SERVE_ARGS[@]}" "${EXTRA_ARGS[@]}")
{ printf '%q ' "${SERVE[@]}"; echo; python -c 'import vllm; print("server code:", vllm.__file__)'; } > "$OUT/server_cmd.txt"
"${SERVE[@]}" > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
T0=$(date +%s)
until curl -sf localhost:$PORT/health >/dev/null; do
  kill -0 $SRV 2>/dev/null || fail server_start
  [ $(( $(date +%s) - T0 )) -gt 900 ] && fail server_start_timeout
  sleep 2
done
echo "server ready after $(( $(date +%s) - T0 ))s" | tee "$OUT/startup.txt"

# --- clients (baseline code: a record's diff can't change how it is measured)
export PYTHONPATH=$BASELINE_TREE
DATA=$(ls "$HF_HUB_CACHE"/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/*/ShareGPT_V3_unfiltered_cleaned_split.json | head -1)
curl -s localhost:$PORT/metrics > "$OUT/metrics_before.txt"
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 30 > "$OUT/gpu_clocks.csv" &
CLK=$!
vllm bench serve --port $PORT \
  --backend openai-chat --endpoint /v1/chat/completions --model "$MODEL" \
  --dataset-name sharegpt --dataset-path "$DATA" \
  --num-prompts 400 --max-concurrency 16 --request-rate inf \
  --seed 0 --temperature 0 --num-warmups 2 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir "$OUT" --result-filename bench_c16.json \
  > "$OUT/bench.log" 2>&1 || { kill $CLK; fail bench; }
kill $CLK 2>/dev/null
curl -s localhost:$PORT/metrics > "$OUT/metrics_after.txt"
python "$HARNESS/gsm8k.py" --n 250 --url http://localhost:$PORT --out "$OUT/gsm8k.json" \
  > "$OUT/gsm8k.log" 2>&1 || fail gsm8k
tail -1 "$OUT/gsm8k.log"
echo "eval done"
