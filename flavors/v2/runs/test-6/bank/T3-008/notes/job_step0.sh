#!/bin/bash
A=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-3.w1
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/t3-distill/env.sh
source $AUTOINFER_RUN/task/scripts/lib.sh
KV=${KV:-6174015488}   # 5.75 GiB
export OUT=$A/jobs/step0; mkdir -p $OUT
SPEC='{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true,"draft_lm_head_quantization":"fp8","draft_lm_head_vocab_size":98304}'
CCFG=$(sed -e "s/.*--compilation-config '\([^']*\)'.*/\1/" $AUTOINFER_RUN/bank/T1-042/serve_args.txt)
start_server_bench 100 16 $OUT/bench --attention-backend TRITON_ATTN --compilation-config "$CCFG" --kv-cache-memory $KV --speculative-config "$SPEC" || echo "bench failed"
echo "== running lines"; grep -o "Running: [0-9]* reqs, Waiting: [0-9]* reqs, GPU KV cache usage: [0-9.]*%" $OUT/server.log | sort | uniq -c
grep -n "GPU KV cache size\|Initial free\|Graph capturing\|OOM\|out of memory" $OUT/server.log | cut -c1-200
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
long_request 128000 $OUT/long_request.json; echo "long rc=$?"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
grep -i "error\|OOM\|out of memory" $OUT/server.log | head
curl -s localhost:$PORT/health -o /dev/null -w "health %{http_code}\n"
stop_server
