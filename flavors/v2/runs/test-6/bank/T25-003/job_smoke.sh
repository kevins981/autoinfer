#!/bin/bash
export AUTOINFER_RUN=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6
source $AUTOINFER_RUN/workspaces/t25-s1b/env.sh || exit 1
source $AUTOINFER_RUN/task/scripts/lib.sh
D=$AUTOINFER_RUN/agents/leader-25.w1/S1b; cd $D
echo "KB61: $(ls -laL $AUTOINFER_WS/vllm/vllm/_C_stable_libtorch.abi3.so)"
unset AI_POLICY AI_LOG AI_MLP_SPLIT AI_MLP_SPLIT_MODE VLLM_GDN_TWO_SLOT
eval "EXTRA=( $(cat $AUTOINFER_RUN/bank/T25-002/serve_args.txt) )"
mkdir -p $D/smoke; export OUT=$D/smoke
_launch_server "${EXTRA[@]}"
for i in $(seq 150); do grep -q "Model loading took" $OUT/server.log && break; kill -0 $SERVER_PID 2>/dev/null || break; sleep 3; done
sleep 3
grep -q "AI_MLP_SPLIT: uneven" $OUT/server.log || { echo "FAIL FAST: split not active"; tail -5 $OUT/server.log; stop_server; exit 3; }
grep -E "AI_MLP_SPLIT|Model loading took" $OUT/server.log | cut -c1-230
_wait_server || { tail -30 $OUT/server.log | cut -c1-250; exit 1; }
grep -E "mode b|Graph capturing|GPU KV cache size" $OUT/server.log | cut -c1-230
bench 64 16 $D/smoke
if grep -q "Internal Server Error\|OutOfMemory" $OUT/server.log || ! kill -0 $SERVER_PID 2>/dev/null; then echo "FAIL FAST: bench errors"; stop_server; exit 4; fi
grep -o "Running: [0-9]* reqs, Waiting: [0-9]* reqs, GPU KV cache usage: [0-9.]*%" $OUT/server.log | sort | uniq -c | sort -rn | head -5
grep -E "mode b" $OUT/server.log | cut -c1-230
gsm8k 50 $D/smoke/gsm8k50.json
nvidia-smi --query-gpu=index,memory.used --format=csv
stop_server
