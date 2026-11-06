#!/bin/bash
# Example GPU job (copy and adapt): start the server from a workspace, short bench, GSM8K, stop.
#   gpu run --timeout 15 --label smoke -- bash example_job.sh <workspace dir> <out dir> [extra serve args]
# A short bench (100 prompts) takes ~1 min; the official one is 400 prompts (~2.7 min).
WS=$1; export OUT=$2; shift 2
source "$WS/env.sh"
source "$AUTOINFER_RUN/task/scripts/lib.sh"
start_server "$@" || exit 1
bench 100 16 "$OUT/bench"
gsm8k 50 "$OUT/gsm8k.json"
