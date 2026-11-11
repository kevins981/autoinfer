#!/bin/bash
# B: junction prefix cache, unit 1. usage: B_only.sh <ws> <out>
WS=$1; export OUT=$2; mkdir -p "$OUT"
source "$WS/env.sh"; source "$AUTOINFER_RUN/task/scripts/lib.sh"
eval "CHAMP=($(cat $AUTOINFER_RUN/bank/T14-004/serve_args.txt))"
B_ARGS=(--prefix-match-unit 1 --enable-mamba-shared-prefix-checkpoint)
start_server_bench 400 16 "$OUT/bench" "${CHAMP[@]}" "${B_ARGS[@]}" || { tail -50 $OUT/server.log; exit 1; }
curl -s localhost:$PORT/metrics | grep -E "^vllm:prefix_cache_(hits|queries)_total|^vllm:spec_decode_num_accepted_tokens_total|^vllm:spec_decode_num_drafts_total"
python - <<P
import json,glob;f=glob.glob("$OUT/bench/*.json")[0];d=json.load(open(f));print("RESULT tok/s",d['output_throughput'],"ttft_mean",d['mean_ttft_ms'],"p99ttft",d['p99_ttft_ms'],"p99tpot",d['p99_tpot_ms'])
P
python $AUTOINFER_RUN/agents/leader-21.w1/jobs/exact.py http://localhost:$PORT "$MODEL" 2>&1 | tail -30
stop_server
