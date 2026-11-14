## 2026-11-14T16:11:07+00:00 (manager)
team T1 (sweep) opened with 100 GPU-min (seed 20 + start 80)

## 2026-11-14T16:11:07+00:00 (manager)
team T2 (sweep) opened with 100 GPU-min (seed 20 + start 80)

## 2026-11-14T16:21:55+00:00 (manager)
map.md updated (proposals/T1-map-1.md)

## 2026-11-14T16:22:46+00:00 (manager)
[manager] Infra issue (found by T1, job 5): vLLM server starts fail in this run because TMPDIR + uuid socket path exceeds the 107-char unix-socket limit. Workaround for your own GPU jobs: export VLLM_RPC_BASE_PATH=$CACHE_ROOT/tmp before starting the server (changes only the socket directory; nothing in task/ may be changed). official_eval is likely affected too: message me before requesting one; I've raised it with the human.

## 2026-11-14T16:23:10+00:00 (manager)
map.md updated (proposals/T2-map-1.md)

## 2026-11-14T16:23:31+00:00 (manager)
[manager] Socket-path issue fixed by the human: task env.sh now exports VLLM_RPC_BASE_PATH=$CACHE_ROOT/tmp, so every server start (official evals included) gets a short socket path. The per-job workaround is no longer needed. Official evals may go ahead as usual.

