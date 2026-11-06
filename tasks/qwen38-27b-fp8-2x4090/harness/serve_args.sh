# The task's serve command (sourced by eval.sh). A record may add flags in its serve_args.txt;
# the leader's validity review checks them against the task definition.
MODEL=Qwen/Qwen3.8-27B-FP8
SERVE_ARGS=(
  --tensor-parallel-size 2
  --max-model-len 131072
  --max-num-seqs 16
  --gpu-memory-utilization 0.92
  --kv-cache-dtype fp8 --kv-cache-memory 5368709120
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
  --language-model-only
)
