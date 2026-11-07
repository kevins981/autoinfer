# Helpers for GPU jobs on the qwen task (read-only; source it inside a `gpu run` script).
#   source $AUTOINFER_RUN/task/scripts/lib.sh            # after `source <workspace>/env.sh`
#   start_server [extra vllm serve args...]   # task flags + yours; waits until ready (or fails).
#                                              Yours come last, so a flag you repeat REPLACES the
#                                              task's (e.g. your --speculative-config replaces the
#                                              whole JSON: include method and num_speculative_tokens)
#   bench <num_prompts> <concurrency> <out_dir>   # ShareGPT, T=0, seed 0, like the official bench
#   gsm8k <n> <out.json>                       # accuracy check (official gate: n=250); for changes
#                                              that may alter the math. Skip it for exact changes.
#                                              n=50 = the first 50 of the official 250 (same order):
#                                              BASE scores 88.0% (44/50) on it; SE ~+-4.6 points.
#   long_request <tokens> [out.json]           # one max-length request + a health check (official
#                                              gate: 128000). For changes that affect GPU memory.
#   stop_server                                # also happens automatically when the job ends
# Env: PORT (default 8000), OUT (default: current dir; server.log goes there).
# Unofficial numbers only: official ones come from `official_eval`. A short bench is NOT comparable
# with the official 400-prompt numbers (e.g. BASE: ~596 tok/s with 100 prompts vs 630.6 official):
# compare a change against BASE measured the same way, ideally in the same job.
TASK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$TASK_DIR/harness/serve_args.sh"           # MODEL, SERVE_ARGS
PORT=${PORT:-8000}
OUT=${OUT:-$PWD}
SHAREGPT=$(ls "$HF_HUB_CACHE"/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/*/ShareGPT_V3_unfiltered_cleaned_split.json | head -1)

start_server() {
  mkdir -p "$OUT"
  echo "[lib] vllm serve $MODEL --port $PORT ${SERVE_ARGS[*]} $* (code: $(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))'))"
  vllm serve "$MODEL" --port "$PORT" "${SERVE_ARGS[@]}" "$@" > "$OUT/server.log" 2>&1 &
  SERVER_PID=$!
  local t0=$SECONDS
  until curl -sf "localhost:$PORT/health" >/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "[lib] server died during startup; last lines of $OUT/server.log:"; tail -30 "$OUT/server.log"; return 1
    fi
    if (( SECONDS - t0 > 900 )); then echo "[lib] server not ready after 15 min"; return 1; fi
    sleep 2
  done
  echo "[lib] server ready after $((SECONDS - t0)) s (log: $OUT/server.log)"
}

stop_server() {
  [ -n "${SERVER_PID:-}" ] || return 0
  kill -INT $SERVER_PID 2>/dev/null
  for _ in $(seq 60); do kill -0 $SERVER_PID 2>/dev/null || break; sleep 1; done
  kill -KILL $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; SERVER_PID=
}

bench() {   # bench <num_prompts> <concurrency> <out_dir> [extra vllm bench serve args]
  local n=$1 c=$2 d=$3; shift 3; mkdir -p "$d"
  curl -s "localhost:$PORT/metrics" > "$d/metrics_before.txt"
  vllm bench serve --port "$PORT" --backend openai-chat --endpoint /v1/chat/completions \
    --model "$MODEL" --dataset-name sharegpt --dataset-path "$SHAREGPT" \
    --num-prompts "$n" --max-concurrency "$c" --request-rate inf --seed 0 --temperature 0 \
    --num-warmups 2 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$d" --result-filename bench.json "$@" > "$d/bench.log" 2>&1
  local rc=$?
  curl -s "localhost:$PORT/metrics" > "$d/metrics_after.txt"
  python "$TASK_DIR/harness/summarize.py" "$d" | grep -E '"(completed|output_throughput|median_ttft_ms|median_tpot_ms|mtp_acceptance_length)"'
  return $rc
}

gsm8k() {   # gsm8k <n> <out.json>
  python "$TASK_DIR/harness/gsm8k.py" --n "$1" --url "http://localhost:$PORT" --out "$2" | tail -2
}
long_request() {   # long_request <tokens> [out.json]   (official gate: 128000)
  python "$TASK_DIR/harness/long_request.py" --tokens "$1" --url "http://localhost:$PORT" \
    --out "${2:-$OUT/long_request.json}"
}
trap stop_server EXIT
