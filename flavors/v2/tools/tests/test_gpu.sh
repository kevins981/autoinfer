#!/bin/bash
# Tests for flavors/v2/tools/gpu against its spec. Fake jobs (sleep); no GPU use.
G=/scratch/kevinsong/learning/autoinfer/flavors/v2/tools/gpu
BASE=/scratch/kevinsong/learning/autoinfer/.cache/gpu2test
rm -rf $BASE; mkdir -p $BASE
export AUTOINFER_RUN=$BASE/run1
unset AUTOINFER_TEAM AUTOINFER_ROLE AUTOINFER_AGENT
pass=0; fail=0
check() { # desc expected actual
  if [ "$2" == "$3" ]; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1 (expected $2, got $3)"; fail=$((fail+1)); fi; }
H() { env -u AUTOINFER_TEAM -u AUTOINFER_ROLE -u AUTOINFER_AGENT "$@"; }
M() { AUTOINFER_ROLE=manager AUTOINFER_AGENT=manager "$@"; }
W() { local t=$1; shift; AUTOINFER_ROLE=worker AUTOINFER_TEAM=$t AUTOINFER_AGENT=w-$t "$@"; }

M $G team open T1 --type explore >/dev/null 2>&1; check "open before pool refused" 1 $?
M $G pool init 200 --seed 20 >/dev/null 2>&1; check "manager can't init pool" 1 $?
H $G pool init 200 --seed 20 >/dev/null; check "human pool init" 0 $?
H $G pool init 300 --seed 20 >/dev/null 2>&1; check "second init refused" 1 $?
W T1 $G team open T1 --type explore >/dev/null 2>&1; check "worker can't open team" 1 $?
M $G team open T1 --type explore >/dev/null; check "manager opens T1" 0 $?
M $G team open T1 --type exploit >/dev/null 2>&1; check "reused id refused" 1 $?
M $G team open T2 >/dev/null 2>&1; check "open without --type refused" 2 $?
free=$($G status --json | python3 -c "import json,sys;print(json.load(sys.stdin)['pool']['free'])"); check "pool free after seed" 180.0 $free
M $G grant T1 500 --kind start >/dev/null 2>&1; check "grant beyond pool refused" 1 $?
M $G grant T9 5 --kind start >/dev/null 2>&1; check "grant to unknown team refused" 1 $?
W T1 $G grant T1 5 --kind start >/dev/null 2>&1; check "worker can't grant" 1 $?
M $G grant T1 80 --kind start --note "proposal ok" >/dev/null; check "grant start 80" 0 $?

out=$(W T1 $G run --timeout 1 --label ok -- sleep 2); rc=$?
check "T1 job runs" 0 $rc
echo "$out" | grep -q "team T1 (explore): 100.0 GPU-min left of 100 granted"; check "balance printed at start" 0 $?
n=$(echo "$out" | grep -c "team T1 (explore): .* GPU-min left of 100 granted"); check "balance printed at start and end" 2 $n
W T1 $G run --timeout 500 -- true >/dev/null 2>&1; check "timeout over balance -> 76" 76 $?
W T1 $G run --timeout 0.05 -- sleep 30 >/dev/null 2>&1; check "timeout kill -> 124" 124 $?
W TX $G run --timeout 1 -- true >/dev/null 2>&1; check "unknown team -> 76" 76 $?
AUTOINFER_ROLE=worker AUTOINFER_AGENT=w $G run --timeout 1 -- true >/dev/null 2>&1; check "worker without team -> 76" 76 $?
M $G run --timeout 1 -- true >/dev/null 2>&1; check "manager with empty MGR -> 76" 76 $?
M $G grant MGR 10 --kind extension --note profiling >/dev/null; check "grant MGR" 0 $?
M $G run --timeout 1 --label mgrjob -- sleep 1 >/dev/null 2>&1; check "manager job charged to MGR" 0 $?
H $G run --timeout 1 --label human -- sleep 1 >/dev/null 2>&1; check "human job uncharged runs" 0 $?
$G status --json > $BASE/s1.json
python3 - $BASE/s1.json <<'P'
import json,sys
d=json.load(open(sys.argv[1])); a=d['accounts']
print("PASS MGR charged" if a['MGR']['used']>0 else "FAIL MGR not charged")
print("PASS human uncharged" if 'human' in d['uncharged'] and all('human' not in x['by_agent'] for x in a.values()) else "FAIL human charged")
print("PASS T1 timeout charged ~3s" if 0.04 < a['T1']['used'] < 0.2 else f"FAIL T1 used {a['T1']['used']}")
P

# close with a running and a waiting job
M $G team open T2 --type exploit >/dev/null; M $G grant T2 50 --kind start >/dev/null
W T2 $G run --timeout 5 --label long -- sleep 60 > $BASE/j_run.out 2>&1 & p1=$!
sleep 4
W T2 $G run --timeout 5 --label waiting -- sleep 1 > $BASE/j_wait.out 2>&1 & p2=$!
sleep 4
M $G team close T2 --note test >/dev/null; check "close T2" 0 $?
wait $p1; check "running job cancelled -> 130" 130 $?
wait $p2; check "waiting job cancelled -> 130" 130 $?
W T2 $G run --timeout 1 -- true >/dev/null 2>&1; check "closed team -> 76" 76 $?
M $G grant T2 5 --kind extension >/dev/null 2>&1; check "grant to closed team refused" 1 $?
M $G team close MGR >/dev/null 2>&1; check "MGR can't be closed" 1 $?
$G status --json > $BASE/s2.json
python3 - $BASE/s2.json <<'P'
import json,sys
d=json.load(open(sys.argv[1])); a=d['accounts']; p=d['pool']
used_closed=a['T2']['used']
exp=p['total']-sum(x['granted'] for x in a.values() if x['open'])-used_closed
print("PASS T2 running job charged" if 0.05 < used_closed < 0.3 else f"FAIL T2 used {used_closed}")
print("PASS pool free returns closed unspent" if abs(p['free']-exp)<1e-6 and p['free'] > 200-20-80-10-1 else f"FAIL pool {p}")
P
$G history T2 | grep -q "close by manager: test"; check "history shows close" 0 $?
$G history T1 | grep -q "grant start +80  by manager: proposal ok"; check "history shows grant note" 0 $?

# concurrent grants can't overdraw
free=$($G status --json | python3 -c "import json,sys;print(int(json.load(sys.stdin)['pool']['free']))")
half=$((free/2+1))
for i in 1 2 3; do M $G grant T1 $half --kind extension >/dev/null 2>&1 & done; wait
f2=$($G status --json | python3 -c "import json,sys;print(json.load(sys.stdin)['pool']['free']>=0)"); check "concurrent grants don't overdraw" True $f2
n=$(grep -c "\"minutes\": $half" $BASE/run1/gpu/accounts.jsonl); check "exactly one concurrent grant succeeded" 1 $n

# low-balance notice and admission re-check at the front of the queue (new pool, small seed)
export AUTOINFER_RUN=$BASE/run2
H $G pool init 100 --seed 0.3 >/dev/null; M $G team open T1 --type exploit >/dev/null
out=$(W T1 $G run --timeout 0.3 -- sleep 15 2>&1); echo "$out" | grep -q "low balance: consider requesting a resync"; check "low-balance notice" 0 $?
M $G team open T2 --type exploit >/dev/null
M $G team open T3 --type exploit >/dev/null
W T3 $G run --timeout 0.25 --label first -- sleep 12 > /dev/null 2>&1 & q1=$!
sleep 1
W T2 $G run --timeout 0.25 --label a -- sleep 12 > /dev/null 2>&1 & q2=$!
sleep 1
W T2 $G run --timeout 0.25 --label b -- sleep 1 > $BASE/q3.out 2>&1 & q3=$!
wait $q1; wait $q2; check "T2 first job ran" 0 $?
wait $q3; check "T2 second job refused at front (balance spent while waiting)" 76 $?
H $G status >/dev/null; check "status text" 0 $?
echo "== $pass passed, $fail failed (plus python checks above)"
