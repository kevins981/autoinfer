#!/bin/bash
# Tests for `inspire`. No agents, no GPU.
T=/scratch/kevinsong/learning/autoinfer/flavors/v2/tools
BASE=/scratch/kevinsong/learning/autoinfer/.cache/inspiretest
rm -rf $BASE; mkdir -p $BASE/run; cp $T/../config_default.md $BASE/run/config.md
export AUTOINFER_RUN=$BASE/run; unset AUTOINFER_TEAM AUTOINFER_AGENT
pass=0; fail=0
check() { if [ "$2" == "$3" ]; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1 (expected $2, got $3)"; fail=$((fail+1)); fi; }
I=$T/inspire
$I >/dev/null 2>&1; check "no team refused" 2 $?
out=$($I --team T1 2>/dev/null); check "manager draws for T1" 0 $?
check "9 cards" 9 $(echo "$out" | grep -c '^(card: ')
check "3 sections" 3 $(echo "$out" | grep -c '^## ')
echo "$out" | grep -vi "random number" | grep -qi "random"; check "never says random (framing)" 1 $?
echo "$out" | grep -q "verify any detail"; check "verify sentence" 0 $?
echo "$out" | grep -q "web research"; check "web research line" 0 $?
echo "$out" | grep -q "2-3 lines in your notes"; check "notes line" 0 $?
diff <(echo "$out") $BASE/run/inspiration/T1-1.md >/dev/null; check "saved file = output" 0 $?
AUTOINFER_TEAM=T2 $I --team T1 >/dev/null 2>&1; check "leader can't draw for another team" 2 $?
for i in 2 3 4 5 6 7; do AUTOINFER_TEAM=T1 AUTOINFER_AGENT=leader-1 $I >/dev/null 2>&1; done
check "7 draws logged" 7 $(wc -l < $BASE/run/inspiration/draws.jsonl)
python3 - <<P; check "no repeat before a full pass" 0 $?
import json
rs=[json.loads(l) for l in open("$BASE/run/inspiration/draws.jsonl")]
p=[c for r in rs for c in r["cards"]["principles"]]; h=[c for r in rs[:9] for c in r["cards"]["hardware"]]
s=[c for r in rs[:4] for c in r["cards"]["software"]]
assert len(set(p))==21 and len(set(h))==21 and len(set(s))==12, (len(set(p)),len(set(h)),len(set(s)))
assert rs[1]["by"]=="leader-1" and rs[0]["by"]=="human"
P
sed -i 's/principles 3, hardware 3, software 3/principles 2, hardware 0, software 1/' $BASE/run/config.md
out=$($I --team T9 2>/dev/null); check "config counts: 3 cards" 3 $(echo "$out" | grep -c '^(card: ')
check "hardware off: 2 sections" 2 $(echo "$out" | grep -c '^## ')
$I --list | grep -q "14-imbalance.md: Imbalance"; check "--list" 0 $?
$I --show 14-imbalance.md | head -1 | grep -qx "# Imbalance"; check "--show" 0 $?
$I --help | grep -qi "random"; check "help never says random" 1 $?
echo "passed $pass, failed $fail"; [ $fail == 0 ]
