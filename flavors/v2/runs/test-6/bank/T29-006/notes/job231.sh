#!/bin/bash
# T29-006 bundle (t29-fu = T29-004 + v2 GEMM table + fused producers + w2's GDN glue trim, all default on):
# (1) eager capture server on the merged code (T29_CAPTURE dumps, v2 table compare, T29_GLUE_CHECK=1 self-check)
#     + offline bitwise compare; (2) abba.sh A = T29-004 (t29-fp8) vs B = t29-fu, ABBA, 300 prompts, gsm8k 50.
R=$AUTOINFER_RUN; A=$R/workspaces/t29-fp8; B=$R/workspaces/t29-fu; D=$R/agents/leader-29.w1/t6; FP=$R/agents/leader-29.w1/fp
source $B/env.sh; source $R/task/scripts/lib.sh
ARGS=$(cat $R/bank/T27-002/serve_args.txt); eval "SARGS=($ARGS)"
CAP=$D/cap; rm -rf $CAP; mkdir -p $CAP $D/srv_cap
echo "== capture server (merged code, eager) $(date +%T)"
export OUT=$D/srv_cap T29_CAPTURE=$CAP T29_CAPTURE_REF_TABLE=$FP/ref_table_T29-004.json T29_CAPTURE_PER_KEY=3 T29_GLUE_CHECK=96
start_server "${SARGS[@]}" --enforce-eager || { echo "capture server failed"; tail -30 $D/srv_cap/server.log; exit 1; }
bench 60 16 $D/srv_cap/bench
stop_server
unset T29_CAPTURE T29_CAPTURE_REF_TABLE T29_GLUE_CHECK
grep -hiE "Traceback|Error" $D/srv_cap/server.log | head -5
echo "-- glue self-check: lines $(grep -c T29GLUECHECK $D/srv_cap/server.log) (expect 192), ok=False $(grep T29GLUECHECK $D/srv_cap/server.log | grep -c ok=False) (must be 0)"; grep -h T29GLUECHECK $D/srv_cap/server.log | head -2 | cut -c1-240; echo "JIT during inference (capture server): $(grep -c "JIT compilation during inference" $D/srv_cap/server.log)"; grep -o "inference: [A-Za-z_0-9]*" $D/srv_cap/server.log | sort | uniq -c
echo "dumps: $(ls $CAP/cap_*.pt | wc -l), $(du -sh $CAP | cut -f1)"
echo "-- GEMM v2 vs T29-004 table (live activations)"; python - <<PY
import collections, re
t = collections.defaultdict(lambda: [0, 0, 0.0, set()])
for l in open("$CAP/gemm_v2_vs_ref.txt"):
    m = re.match(r"(\S+) M=(\d+) cfg=(\d+) ref_cfg=(\d+) differ=(\d+)/(\d+) maxabs=(\S+)", l)
    k = m.group(1); v = t[k]; v[0] += int(m.group(5)); v[1] += int(m.group(6)); v[2] = max(v[2], float(m.group(7))); v[3].add(int(m.group(2)))
for k, v in sorted(t.items()):
    print(f"{k}: differing {v[0]}/{v[1]}, max|diff| {v[2]:.4g}, {len(v[3])} distinct M ({min(v[3])}..{max(v[3])})")
PY
echo "== offline compare $(date +%T)"; ( cd $FP; timeout 300 python cmp.py $CAP 2>&1 | grep -v "^W1109\|deprecated" )
rm -rf $CAP/cap_*.pt   # 2 GB of dumps; the compare table above is the record
echo "== abba $(date +%T)"
$R/task/scripts/abba.sh $D/abba --a $A --b $B --order ABBA --prompts 300 --gsm8k 50 --args "$ARGS"
rc=$?
for d in $D/abba/*_*/; do echo "== $d"; grep -hE "T29_W8A8|not eligible|KV cache size|Model loading took|JIT compilation during" $d/server.log 2>/dev/null | cut -c1-160 | sort -u | head -8; done
exit $rc
