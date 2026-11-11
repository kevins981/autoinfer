#!/bin/bash
# T29 fused producers: (0) PI-store probe; (1) eager capture server (unfused path, real activations dumped,
# v2-vs-T29-004 GEMM table compare) + offline bitwise compare; (2) T29_FUSED=1 server smoke (bench + gsm8k 50).
R=$AUTOINFER_RUN; WS=$R/workspaces/t29-fu; D=$R/agents/leader-29.w1/fp
source $WS/env.sh
echo "== probe2 $(date +%T)"; ( cd $D; timeout 150 python probe2.py 2>&1 | grep -v "^W1109\|deprecated" )
source $R/task/scripts/lib.sh
ARGS=$(cat $R/bank/T27-002/serve_args.txt); eval "SARGS=($ARGS)"
CAP=$D/cap; rm -rf $CAP; mkdir -p $CAP $D/srv_cap $D/srv_fused
echo "== capture server $(date +%T)"
export OUT=$D/srv_cap T29_CAPTURE=$CAP T29_CAPTURE_REF_TABLE=$D/ref_table_T29-004.json T29_CAPTURE_PER_KEY=3
start_server "${SARGS[@]}" --enforce-eager || { echo "capture server failed"; tail -30 $D/srv_cap/server.log; exit 1; }
bench 60 16 $D/srv_cap/bench
stop_server
unset T29_CAPTURE T29_CAPTURE_REF_TABLE
grep -hiE "error|Traceback" $D/srv_cap/server.log | head -5
echo "dumps: $(ls $CAP/cap_*.pt | wc -l), $(du -sh $CAP | cut -f1)"
echo "-- GEMM v2 vs T29-004 table (live activations, rank 0)"; python - <<PY
import collections, re
t = collections.defaultdict(lambda: [0, 0, 0.0, set()])
for l in open("$CAP/gemm_v2_vs_ref.txt"):
    m = re.match(r"(\S+) M=(\d+) cfg=(\d+) ref_cfg=(\d+) differ=(\d+)/(\d+) maxabs=(\S+)", l)
    k = m.group(1); v = t[k]; v[0] += int(m.group(5)); v[1] += int(m.group(6)); v[2] = max(v[2], float(m.group(7))); v[3].add(int(m.group(2)))
for k, v in sorted(t.items()):
    print(f"{k}: differing {v[0]}/{v[1]} ({100*v[0]/max(1,v[1]):.4f}%), max|diff| {v[2]:.4g}, calls over {len(v[3])} distinct M ({min(v[3])}..{max(v[3])})")
PY
echo "== offline compare $(date +%T)"; ( cd $D; T29_FUSED=1 timeout 300 python cmp.py $CAP 2>&1 | grep -v "^W1109\|deprecated" )
echo "== fused server smoke $(date +%T)"
export OUT=$D/srv_fused T29_FUSED=1
start_server "${SARGS[@]}" || { echo "fused server failed"; tail -30 $D/srv_fused/server.log; exit 1; }
grep -hE "T29_W8A8|JIT compilation during|KV cache size|Model loading took" $D/srv_fused/server.log | cut -c1-200 | head -8
bench 100 16 $D/srv_fused/bench
gsm8k 50 $D/srv_fused/gsm8k.json 2>&1 | tail -1
grep -hE "JIT compilation during|Traceback|Error" $D/srv_fused/server.log | cut -c1-200 | head -5
stop_server
