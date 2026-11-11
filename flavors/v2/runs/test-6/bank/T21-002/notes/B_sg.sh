#!/bin/bash
WS=$1; export OUT=$2; mkdir -p "$OUT"
source "$WS/env.sh"; source "$AUTOINFER_RUN/task/scripts/lib.sh"
eval "CHAMP=($(cat $AUTOINFER_RUN/bank/T14-004/serve_args.txt))"
start_server "${CHAMP[@]}" --prefix-match-unit 1 --enable-mamba-shared-prefix-checkpoint || exit 1
python $AUTOINFER_RUN/agents/leader-21.w1/jobs/sg.py http://localhost:$PORT "$MODEL" 2>&1 | tail -30
stop_server
