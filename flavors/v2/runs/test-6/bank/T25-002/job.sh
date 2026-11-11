#!/bin/bash
export AUTOINFER_RUN=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6
source $AUTOINFER_RUN/workspaces/t25-s1/env.sh || exit 1
source $AUTOINFER_RUN/task/scripts/lib.sh
D=$AUTOINFER_RUN/agents/leader-25.w1/S1a; cd $D
echo "KB61: $(ls -laL $AUTOINFER_WS/vllm/vllm/_C_stable_libtorch.abi3.so)"
eval "EXTRA=( $(cat $AUTOINFER_RUN/bank/T14-004/serve_args.txt) )"
unset AI_POLICY AI_LOG
export AI_MLP_SPLIT=0.5367
mkdir -p $D/on; export OUT=$D/on
EXTRA+=(--kv-cache-memory 4616880128)
_launch_server "${EXTRA[@]}"
for i in $(seq 150); do grep -q "Model loading took" $D/on/server.log && break; kill -0 $SERVER_PID 2>/dev/null || break; sleep 3; done
sleep 3
if ! grep -q "AI_MLP_SPLIT: uneven" $D/on/server.log; then echo "FAIL FAST: split not active"; tail -5 $D/on/server.log; stop_server; exit 3; fi
grep -E "AI_MLP_SPLIT|Model loading took" $D/on/server.log
_wait_server || exit 1
bench 64 16 $D/on
if grep -q "Internal Server Error\|OutOfMemory" $D/on/server.log || ! kill -0 $SERVER_PID 2>/dev/null; then echo "FAIL FAST: bench errors/OOM"; grep -c "500 Internal" $D/on/server.log; stop_server; exit 4; fi
grep -o "Running: [0-9]* reqs, Waiting: [0-9]* reqs, GPU KV cache usage: [0-9.]*%" $D/on/server.log | sort | uniq -c | sort -rn | head -8
grep -o "usage: [0-9.]*%" $D/on/server.log | sort -t: -k2 -n | tail -1 | sed "s/^/PEAK KV /"
grep -E "Model loading took|Available KV|GPU KV cache size|Graph capturing|Maximum concurrency" $D/on/server.log
nvidia-smi --query-gpu=index,memory.used --format=csv
gsm8k 50 $D/on/gsm8k50.json
long_request 128000 $D/on/long.json
nvidia-smi --query-gpu=index,memory.used --format=csv
stop_server
