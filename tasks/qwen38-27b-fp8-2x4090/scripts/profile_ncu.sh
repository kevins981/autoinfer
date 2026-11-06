#!/bin/bash
# ncu (Nsight Compute) on selected kernels of a live server under steady c=16 decode load.
#   gpu run --timeout 30 --label ncu -- bash profile_ncu.sh <workspace dir> <out dir> <kernel regex> [skip=0] [count=12] [ncu args...]
#   e.g. ... profile_ncu.sh $WS $OUT Marlin 0 12          (3 layers x 4 FP8 GEMMs)
# CUDA graphs stay ON; graph nodes are profiled individually. Flow: server under ncu with
# --profile-from-start off -> 16 long decode requests -> POST /start_profile -> ncu captures the
# next <count> launches matching <kernel regex> (after <skip>) -> stop.
# Output: <out>/ncu.ncu-rep (full metric set; `ncu --import <rep> --page details`), ncu.log, server.log.
# Notes: GPU counters may only work on GPU 0 (add `--devices 0`); ncu serializes and replays
# kernels, so the server is very slow while profiling (startup counts too: give a long timeout).
# Source-level view (per-line stalls): build the workspace with -lineinfo and add
# `--import-source yes`, then `python ncu_hotlines.py <rep>`.
WS=$1; export OUT=$2; KRE=${3:?usage: profile_ncu.sh <ws> <out> <kernel regex> [skip] [count] [ncu args]}
SKIP=${4:-0}; CNT=${5:-12}; shift 5 2>/dev/null || shift $#
mkdir -p "$OUT"
source "$WS/env.sh"
source "$AUTOINFER_RUN/task/scripts/lib.sh"
NCU=${NCU:-$ENV_PREFIX/bin/ncu}
"$NCU" --target-processes all --profile-from-start off --kernel-name-base demangled -k "regex:$KRE" \
  --launch-skip "$SKIP" --launch-count "$CNT" --set full --clock-control none \
  -o "$OUT/ncu" --force-overwrite "$@" \
  vllm serve "$MODEL" --port "$PORT" "${SERVE_ARGS[@]}" --profiler-config '{"profiler":"cuda"}' \
  > "$OUT/ncu.log" 2>&1 &
SERVER_PID=$!
t0=$SECONDS
until curl -sf "localhost:$PORT/health" >/dev/null; do
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; tail -30 "$OUT/ncu.log"; exit 1; }
  (( SECONDS - t0 > 1500 )) && { echo "server not ready after 25 min"; exit 1; }
  sleep 5
done
echo "[ncu] server ready after $((SECONDS - t0)) s"
PORT=$PORT MODEL=$MODEL python - <<'PY' &
import json, os, threading, urllib.request
url = f"http://localhost:{os.environ['PORT']}/v1/chat/completions"
topics = ["the Roman Empire", "neural networks", "the water cycle", "Japan", "quantum computing",
          "the French Revolution", "vaccines", "urban planning", "Marie Curie", "supply and demand",
          "deep sea exploration", "the internet", "medieval castles", "airplanes", "renewable energy", "jazz"]
def go(t):
    body = {"model": os.environ["MODEL"], "temperature": 0, "max_tokens": 3000, "ignore_eos": True,
            "messages": [{"role": "user", "content": f"Write a very long, detailed essay about {t}."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(),
                               {"Content-Type": "application/json"}), timeout=3000).read()
    except Exception:
        pass
ts = [threading.Thread(target=go, args=(t,)) for t in topics]; [t.start() for t in ts]; [t.join() for t in ts]
PY
LOAD=$!
sleep 25
curl -s -X POST "localhost:$PORT/start_profile" >/dev/null; echo "[ncu] start_profile sent"
LAST=0; QUIET=0     # wait until ncu has been quiet for 90 s after the first profiled kernel
for _ in $(seq 300); do
  N=$(grep -c "Profiling " "$OUT/ncu.log")
  if [ "$N" -gt 0 ] && [ "$N" = "$LAST" ]; then QUIET=$((QUIET + 5)); else QUIET=0; fi
  LAST=$N; [ $QUIET -ge 90 ] && break; sleep 5
done
echo "[ncu] kernels profiled: $N"
curl -s -X POST "localhost:$PORT/stop_profile" >/dev/null
kill $LOAD 2>/dev/null
stop_server
ls -la "$OUT"
