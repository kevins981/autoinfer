#!/bin/bash
A=/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-3.w1
source /scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/t3-distill.t/env.sh
source $AUTOINFER_RUN/task/scripts/lib.sh
D=$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))'); ls -laL $D/_C_stable_libtorch.abi3.so; ls -la $D/_C_stable_libtorch.abi3.so
CCFG=$(sed -e "s/.*--compilation-config '\([^']*\)'.*/\1/" $AUTOINFER_RUN/bank/T1-042/serve_args.txt)
run() { # k kv outdir
  local k=$1 kv=$2; export OUT=$3; mkdir -p $OUT; export AUTOINFER_TIMERS=$OUT/timers
  SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$k,\"use_local_argmax_reduction\":true,\"draft_lm_head_quantization\":\"fp8\",\"draft_lm_head_vocab_size\":98304}"
  start_server_bench 100 16 $OUT/bench --attention-backend TRITON_ATTN --compilation-config "$CCFG" --kv-cache-memory $kv --speculative-config "$SPEC"
  local rc=$?
  grep -o "Running: [0-9]* reqs, Waiting: [0-9]* reqs" $OUT/server.log | sort | uniq -c
  grep "GPU KV cache size" $OUT/server.log | head -1 | cut -c1-200
  if [ $rc = 0 ]; then
    grep -c "out of memory" $OUT/server.log
    if [ $k = 3 ]; then long_request 128000 $OUT/long_request.json; echo "long rc=$?"; curl -s localhost:$PORT/health -o /dev/null -w "health %{http_code}\n"; fi
  fi
  stop_server; return $rc
}
ok=0
for kv in 6066641305 6120328396; do
  if run 3 $kv $A/jobs/step0c/k3_$kv; then ok=1; break; fi
done
echo "k3 ok=$ok"
run 2 5368709120 $A/jobs/step0c/k2
