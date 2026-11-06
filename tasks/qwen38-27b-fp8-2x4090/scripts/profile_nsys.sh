#!/bin/bash
# nsys profile of a server under load (ShareGPT at c=16), then a kernel-time breakdown.
#   gpu run --timeout 20 --label prof -- bash profile_nsys.sh <workspace dir> <out dir> [num_prompts=64] [extra serve args]
# Output in <out dir>: profile.nsys-rep (open in Nsight Systems, or `nsys stats`), profile.sqlite
# (all the data; query it with Python, see the task's PROFILING.md), kern_sum.csv
# (per-kernel totals), breakdown.txt (time by category: GEMM, communication, attention, ...),
# server.log, bench.log. CUDA graphs are traced per kernel (--cuda-graph-trace=node).
# Profiling slows things down: use it for where-the-time-goes, not for throughput numbers.
WS=$1; export OUT=$2; N=${3:-64}; shift 3 2>/dev/null || shift $#
mkdir -p "$OUT"
source "$WS/env.sh"
source "$AUTOINFER_RUN/task/scripts/lib.sh"
NSYS=${NSYS:-$(command -v nsys || echo /usr/local/cuda-12.6/bin/nsys)}
# The server runs under nsys; capture starts/stops with the benchmark's --profile (cudaProfilerApi).
"$NSYS" profile -o "$OUT/profile" --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  vllm serve "$MODEL" --port "$PORT" "${SERVE_ARGS[@]}" --profiler-config '{"profiler":"cuda"}' "$@" \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
t0=$SECONDS
until curl -sf "localhost:$PORT/health" >/dev/null; do
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; tail -30 "$OUT/server.log"; exit 1; }
  (( SECONDS - t0 > 900 )) && { echo "server not ready after 15 min"; exit 1; }
  sleep 2
done
echo "[prof] server ready after $((SECONDS - t0)) s"
vllm bench serve --port "$PORT" --backend openai-chat --endpoint /v1/chat/completions --model "$MODEL" \
  --dataset-name sharegpt --dataset-path "$SHAREGPT" --num-prompts "$N" --max-concurrency 16 \
  --request-rate inf --seed 0 --temperature 0 --num-warmups 16 --profile > "$OUT/bench.log" 2>&1
echo "[prof] bench rc=$?"
stop_server
for _ in $(seq 120); do [ -f "$OUT/profile.nsys-rep" ] && break; sleep 2; done
"$NSYS" export --type sqlite --force-overwrite true --output "$OUT/profile.sqlite" "$OUT/profile.nsys-rep" >/dev/null 2>&1
"$NSYS" stats --report cuda_gpu_kern_sum --format csv --output "$OUT/kern" "$OUT/profile.sqlite" >/dev/null 2>&1
mv "$OUT/kern_cuda_gpu_kern_sum.csv" "$OUT/kern_sum.csv" 2>/dev/null
python "$TASK_DIR/scripts/kernel_breakdown.py" "$OUT/kern_sum.csv" | tee "$OUT/breakdown.txt"
