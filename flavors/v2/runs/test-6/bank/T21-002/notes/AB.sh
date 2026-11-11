#!/bin/bash
# usage: AB.sh <ws> <out> <order e.g. "A B"> <extra: yes|no for gsm8k+long on last B>
WS=$1; export OUT=$2; ORDER=$3; EXTRA=$4; mkdir -p "$OUT"
source "$WS/env.sh"; source "$AUTOINFER_RUN/task/scripts/lib.sh"
eval "CHAMP=($(cat $AUTOINFER_RUN/bank/T14-004/serve_args.txt))"
B_ARGS=(--prefix-match-unit 1 --enable-mamba-shared-prefix-checkpoint)
for X in $ORDER; do
  D=$OUT/$X; export OUT=$D; mkdir -p $D
  if [ $X = A ]; then ARGS=("${CHAMP[@]}"); else ARGS=("${CHAMP[@]}" "${B_ARGS[@]}"); fi
  start_server_bench 400 16 "$D/bench" "${ARGS[@]}" || { tail -50 $D/server.log; exit 1; }
  echo "== $X"
  curl -s localhost:$PORT/metrics | grep -E "^vllm:prefix_cache_(hits|queries)_total|^vllm:spec_decode_num_accepted_tokens_total|^vllm:spec_decode_num_drafts_total"
  python - <<P
import json,glob;f=glob.glob("$D/bench/*.json")[0];d=json.load(open(f));print("RESULT $X tok/s",d['output_throughput'],"ttft_mean",d['mean_ttft_ms'],"p99ttft",d['p99_ttft_ms'],"p99tpot",d['p99_tpot_ms'])
P
  if [ $X = B ] && [ "$EXTRA" = yes ]; then
    gsm8k 50 $D/gsm8k.json
    long_request 128000 $D/long.json
  fi
  stop_server
done
