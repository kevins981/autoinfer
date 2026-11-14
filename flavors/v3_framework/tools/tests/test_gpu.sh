#!/bin/bash
# Tests for flavors/v3_framework/tools/gpu against its spec. Fake jobs (sleep); no GPU use.
G=/scratch/kevinsong/learning/autoinfer/flavors/v3_framework/tools/gpu
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

M $G team open T1 --type exploit-long >/dev/null 2>&1; check "open before pool refused" 1 $?
M $G pool init 200 --seed 20 >/dev/null 2>&1; check "manager can't init pool" 1 $?
H $G pool init 200 --seed 20 >/dev/null; check "human pool init" 0 $?
H $G pool init 300 --seed 20 >/dev/null 2>&1; check "second init refused" 1 $?
W T1 $G team open T1 --type exploit-long >/dev/null 2>&1; check "worker can't open team" 1 $?
M $G team open T1 --type exploit-long >/dev/null; check "manager opens T1" 0 $?
M $G team open T1 --type exploit >/dev/null 2>&1; check "reused id refused" 1 $?
M $G team open T2 >/dev/null 2>&1; check "open without --type refused" 2 $?
free=$($G status --json | python3 -c "import json,sys;print(json.load(sys.stdin)['pool']['free'])"); check "pool free after seed" 180.0 $free
M $G grant T1 500 --kind start --note t >/dev/null 2>&1; check "grant beyond pool refused" 1 $?
M $G grant T9 5 --kind start --note t >/dev/null 2>&1; check "grant to unknown team refused" 1 $?
W T1 $G grant T1 5 --kind start --note t >/dev/null 2>&1; check "worker can't grant" 1 $?
M $G grant T1 80 --kind start --note "proposal ok" >/dev/null; check "grant start 80" 0 $?

out=$(W T1 $G run --timeout 1 --label ok -- sleep 2); rc=$?
check "T1 job runs" 0 $rc
echo "$out" | grep -q "team T1 (exploit-long): 100.0 GPU-min left of 100 granted"; check "balance printed at start" 0 $?
n=$(echo "$out" | grep -c "team T1 (exploit-long): .* GPU-min left of 100 granted"); check "balance printed at start and end" 2 $n
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
M $G team open T2 --type exploit >/dev/null; M $G grant T2 50 --kind start --note t >/dev/null
W T2 $G run --timeout 5 --label long -- sleep 60 > $BASE/j_run.out 2>&1 & p1=$!
sleep 4
W T2 $G run --timeout 5 --label waiting -- sleep 1 > $BASE/j_wait.out 2>&1 & p2=$!
sleep 4
M $G team close T2 --note test >/dev/null; check "close T2" 0 $?
wait $p1; check "running job cancelled -> 130" 130 $?
wait $p2; check "waiting job cancelled -> 130" 130 $?
W T2 $G run --timeout 1 -- true >/dev/null 2>&1; check "closed team -> 76" 76 $?
M $G grant T2 5 --kind extension --note t >/dev/null 2>&1; check "grant to closed team refused" 1 $?
M $G team close MGR --note x >/dev/null 2>&1; check "MGR can't be closed" 1 $?
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
for i in 1 2 3; do M $G grant T1 $half --kind extension --note t >/dev/null 2>&1 & done; wait
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

# review fixes (2026-11-08)
export AUTOINFER_RUN=$BASE/run3
H $G pool init 100 --seed 10 >/dev/null; M $G team open T1 --type exploit >/dev/null
W T1 $G run --timeout 0.05 -- bash -c "trap '' TERM; sleep 60" >/dev/null 2>&1; check "TERM-ignoring job killed at timeout -> 124" 124 $?
c=$(python3 -c "import json;l=[json.loads(x) for x in open('$BASE/run3/gpu/ledger.jsonl')];print(l[-1]['gpu_minutes']<=0.05, l[-1]['wall_minutes']>0.05)"); check "charge capped at timeout, wall time kept" "True True" "$c"
W T1 $G run --timeout 2 --label dbl -- bash -c "trap '' TERM; sleep 61" > $BASE/dbl.out 2>&1 & d=$!
sleep 4; jid=$(ls $BASE/run3/gpu/jobs | sort -n | tail -1)
$G cancel $jid >/dev/null; sleep 1; $G cancel $jid >/dev/null 2>&1
wait $d; check "double cancel -> 130" 130 $?
grep -q Traceback $BASE/dbl.out; check "double cancel: no traceback" 1 $?
n=$(grep -c "\"job\": $jid," $BASE/run3/gpu/ledger.jsonl); check "double cancel: charged exactly once" 1 $n
st=$(python3 -c "import json;print(json.load(open('$BASE/run3/gpu/jobs/$jid/job.json'))['state'])"); check "double cancel: state cancelled" cancelled $st
pgrep -u $USER -f "^sleep 61$" >/dev/null; check "double cancel: job processes gone" 1 $?
M $G team open T2 --type exploit >/dev/null; M $G team open T3 --type exploit >/dev/null
W T3 $G run --timeout 0.25 -- sleep 12 >/dev/null 2>&1 & q1=$!; sleep 1
W T2 $G run --timeout 9 -- sleep 8 >/dev/null 2>&1 & q2=$!; sleep 1
W T2 $G run --timeout 9.9 -- sleep 1 >/dev/null 2>&1 & q3=$!
sleep 1; j3=$(ls $BASE/run3/gpu/jobs | sort -n | tail -1)
wait $q1; wait $q2; wait $q3; check "second T2 job refused at front" 76 $?
st=$(python3 -c "import json;print(json.load(open('$BASE/run3/gpu/jobs/$j3/job.json')).get('state'))"); check "refused-at-front job has state refused" refused $st
timeout 20 $G wait $j3 >/dev/null 2>&1; r=$?; [ $r -ne 124 ]; check "gpu wait returns for refused job" 0 $?
$G history MGR >/dev/null; check "history MGR" 0 $?
$G history T1 --json | python3 -c "import json,sys;d=json.load(sys.stdin);assert any(x.get('event')=='job' and 'gpu_minutes' in x for x in d)"; check "history --json structured" 0 $?
M $G team open "" --type exploit-long >/dev/null 2>&1; check "empty team id refused" 1 $?
M $G team open "a b" --type exploit-long >/dev/null 2>&1; check "bad team id refused" 1 $?
M $G team close T3 >/dev/null 2>&1; check "close without --note refused" 2 $?
M $G grant T1 1 --kind extension >/dev/null 2>&1; check "grant without --note refused" 2 $?
W T1 $G run --timeout 2 -- sleep 60 > $BASE/cl.out 2>&1 & c=$!; sleep 4
M $G team close T1 --note bye >/dev/null; wait $c
grep -q "cancelled: team T1 closed" $BASE/cl.out; check "close: job told why" 0 $?
grep -q "T1 (exploit) is closed" $BASE/cl.out; check "close: no stale balance shown" 0 $?
mv $BASE/run3/gpu/pool.json $BASE/run3/gpu/pool.json.bak
H $G pool init 100 --seed 10 >/dev/null 2>&1; check "re-init after pool.json lost refused" 1 $?
mv $BASE/run3/gpu/pool.json.bak $BASE/run3/gpu/pool.json
echo '{"foo": 1}' >> $BASE/run3/gpu/accounts.jsonl
$G status >/dev/null 2>&1; check "status survives odd accounts line" 0 $?
echo "== $pass passed, $fail failed (plus python checks above)"
