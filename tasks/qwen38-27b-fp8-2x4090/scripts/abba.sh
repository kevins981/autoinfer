#!/bin/bash
# abba.sh: same-job, interleaved, fresh-start A/B comparison of two workspaces, with early stopping.
# Run it INSIDE a `gpu run` job (it starts servers). Usage:
#   abba.sh <out_dir> --a <workspace dir> --b <workspace dir> [options]
# Options:
#   --prompts N        ShareGPT prompts per bench (default 300; same prompts for every start)
#   --order ABBA       start order (default ABBA; e.g. BAAB, ABBAAB). First two letters must be A,B
#                      in some order: the first pair decides early stopping.
#   --stop-loss PCT    after the first pair, stop if B trails A by more than PCT% (default 1.5).
#                      Only losses stop the run: a win still needs every planned start.
#   --args "..."       extra vllm serve args for both arms (e.g. "$(cat bank/<id>/serve_args.txt)")
#   --args-a / --args-b "..."   extra serve args for one arm only
#   --env-a / --env-b "K=V ..."  environment for one arm (e.g. "MY_FLAG=1")
#   --gsm8k N          after the last B start, run `gsm8k N` on that server (0 = skip, default 0)
# Each start is a fresh server (start_server_bench from lib.sh), so the comparison carries the
# task's fresh-start noise: per-start sigma ~0.3-0.5% at 300-400 prompts (TASK.md, Noise). The
# verdict lines at the end:
#   ABBA: SCREEN NEGATIVE after 1 pair: B -2.3% vs A  -> record it as a screen result, not a measured fact
#   ABBA: all B > all A: +0.8% (B 946.2/940.1 vs A 937.1/935.6)
#   ABBA: mixed: +0.3% (one B below one A)                   -> repeat, or not worth more GPU
# Output: <out_dir>/<k>_<A|B>/ (server.log, bench.json, results.json), <out_dir>/abba.json, abba.log.
set -u
OUTD=${1:?usage: abba.sh <out_dir> --a <ws> --b <ws> [options]}; shift
WSA= WSB= NP=300 ORDER=ABBA STOP=1.5 ARGS= ARGSA= ARGSB= ENVA= ENVB= GSM=0
while [ $# -gt 0 ]; do
  case "$1" in
    --a) WSA=$2; shift 2;; --b) WSB=$2; shift 2;; --prompts) NP=$2; shift 2;;
    --order) ORDER=$2; shift 2;; --stop-loss) STOP=$2; shift 2;; --args) ARGS=$2; shift 2;;
    --args-a) ARGSA=$2; shift 2;; --args-b) ARGSB=$2; shift 2;; --env-a) ENVA=$2; shift 2;;
    --env-b) ENVB=$2; shift 2;; --gsm8k) GSM=$2; shift 2;;
    *) echo "abba.sh: unknown option $1" >&2; exit 2;;
  esac
done
[ -n "$WSA" ] && [ -n "$WSB" ] || { echo "abba.sh: --a and --b are required" >&2; exit 2; }
[ -f "$WSA/env.sh" ] && [ -f "$WSB/env.sh" ] || { echo "abba.sh: --a/--b must be workspace dirs (with env.sh)" >&2; exit 2; }
case "$ORDER" in AB*|BA*) ;; *) echo "abba.sh: --order must start with A,B or B,A" >&2; exit 2;; esac
LIB=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh
mkdir -p "$OUTD"; LOG=$OUTD/abba.log
log() { echo "[abba $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log "A=$WSA  B=$WSB  prompts=$NP order=$ORDER stop-loss=$STOP%"

declare -a RES_A=() RES_B=()
run_start() {   # run_start <k> <A|B>
  local k=$1 arm=$2 ws extra env d="$OUTD/${1}_$2"
  if [ "$arm" = A ]; then ws=$WSA; extra="$ARGS $ARGSA"; env=$ENVA; else ws=$WSB; extra="$ARGS $ARGSB"; env=$ENVB; fi
  mkdir -p "$d"
  ( export OUT=$d
    [ -n "$env" ] && export $env
    source "$ws/env.sh" || exit 1
    source "$LIB"
    eval "EXTRA=( $extra )"
    start_server_bench "$NP" 16 "$d" "${EXTRA[@]}" || exit 1
    if [ "$arm" = B ] && [ "$GSM" -gt 0 ] && [ "$k" = "$LAST_B" ]; then gsm8k "$GSM" "$d/gsm8k.json"; fi
    stop_server ) >> "$d/start.log" 2>&1
  local rc=$?
  local tps acc
  if [ $rc -eq 0 ] && [ -f "$d/results.json" ]; then
    read -r tps acc < <(python -c '
import json,sys; r=json.load(open(sys.argv[1])); print(r.get("output_throughput",0), r.get("mtp_acceptance_length",0))' "$d/results.json")
  else
    log "start $k ($arm) FAILED (exit $rc); last lines of $d/start.log:"; tail -5 "$d/start.log" | tee -a "$LOG"
    echo "{\"order\": \"$ORDER\", \"verdict\": \"failed at start $k ($arm)\"}" > "$OUTD/abba.json"; exit 1
  fi
  log "start $k ($arm): $tps tok/s  acceptance $acc  (ttft/tpot p99: $(python -c '
import json,sys; r=json.load(open(sys.argv[1])); print(round(r.get("p99_ttft_ms",0)), "/", round(r.get("p99_tpot_ms",0),1), "ms")' "$d/results.json"))"
  if [ "$arm" = A ]; then RES_A+=("$tps"); else RES_B+=("$tps"); fi
}

n=${#ORDER}; LAST_B=0
for ((i=0; i<n; i++)); do [ "${ORDER:$i:1}" = B ] && LAST_B=$((i+1)); done
for ((i=0; i<n; i++)); do
  run_start $((i+1)) "${ORDER:$i:1}"
  if [ $i -eq 1 ]; then   # first pair done: a clear loss ends the job
    gain=$(python -c "print(round(100*(${RES_B[0]}/${RES_A[0]}-1),2))")
    if python -c "import sys; sys.exit(0 if $gain < -$STOP else 1)"; then
      log "SCREEN NEGATIVE after 1 pair: B ${gain}% vs A (B ${RES_B[0]} vs A ${RES_A[0]}; stop-loss $STOP%)."
      log "This is one fresh-start pair (sigma of the difference ~0.5-0.7%): a screen result, not a measured negative."
      echo "{\"order\": \"$ORDER\", \"starts_run\": 2, \"a\": [${RES_A[0]}], \"b\": [${RES_B[0]}], \"gain_pct\": $gain, \"verdict\": \"screen negative (1 pair)\"}" > "$OUTD/abba.json"
      exit 3
    fi
  fi
done
python - "$OUTD/abba.json" "$ORDER" "${RES_A[*]}" "${RES_B[*]}" <<'EOF' | tee -a "$LOG"
import json, sys, statistics as st
out, order, A, B = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split()], [float(x) for x in sys.argv[4].split()]
gain = 100 * (st.mean(B) / st.mean(A) - 1)
allb = min(B) > max(A); alla = max(B) < min(A)
verdict = ("all B > all A" if allb else "all B < all A" if alla else "mixed")
fa = "/".join(f"{x:.1f}" for x in A); fb = "/".join(f"{x:.1f}" for x in B)
print(f"ABBA: {verdict}: {gain:+.2f}% (B {fb} vs A {fa})" + ("" if allb or alla else "  -> repeat, or not worth more GPU"))
json.dump({"order": order, "starts_run": len(A) + len(B), "a": A, "b": B, "gain_pct": round(gain, 2),
           "verdict": verdict}, open(out, "w"), indent=1)
EOF
