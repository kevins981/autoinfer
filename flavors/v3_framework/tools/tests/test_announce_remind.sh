#!/bin/bash
# Tests for `agent announce`, `agent remind`, _remind_hook.py and the compact hook in new_run.
# Fake agents (meta.json + a live `sleep` as their process); no claude sessions, no GPU.
T=/scratch/kevinsong/learning/autoinfer/flavors/v3_framework/tools
A=$T/agent
BASE=/scratch/kevinsong/learning/autoinfer/.cache/announcetest
rm -rf $BASE; mkdir -p $BASE/run/agents
export AUTOINFER_RUN=$BASE/run
pass=0; fail=0
check() { if [ "$2" == "$3" ]; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1 (expected $2, got $3)"; fail=$((fail+1)); fi; }
sleep 600 & LIVE=$!
mk() { # name role registered stopped
  mkdir -p $BASE/run/agents/$1
  python3 -c "import json,sys;n,r,reg,st,pid=sys.argv[1:];m={'name':n,'role':r,'pid':int(pid)}
if reg=='1': m['registered']=True
if st=='1': m['stopped']='2026-11-08T00:00:00'; m['pid']=999999
json.dump(m,open('$BASE/run/agents/'+n+'/meta.json','w'))" $1 $2 $3 $4 $LIVE; }
mk manager manager 1 0
mk leader-1 leader_exploit_long 0 0
mk leader-2 leader_exploit 0 0
mk leader-1.w1 worker 0 0
mk leader-3 leader_exploit_long 0 1
mk odd brainstormer 0 0
M() { AUTOINFER_ROLE=manager AUTOINFER_AGENT=manager "$@"; }
inbox() { cat $BASE/run/agents/$1/inbox.jsonl 2>/dev/null | python3 -c "import json,sys;[print(json.loads(l)['message']['content']) for l in sys.stdin]"; }

AUTOINFER_ROLE=worker AUTOINFER_AGENT=leader-1.w1 $A announce hi >/dev/null 2>&1; check "worker can't announce" 1 $?
out=$(M $A announce "new champion T2-014 from team T2"); check "manager announces" 0 $?; out0="$out"
echo "$out" | grep -q "announced to 4 agent(s)"; check "delivered to the 4 live headless agents" 0 $?
inbox leader-1 | head -1 | grep -qx "\[announcement from manager; no reply needed\]"; check "announcement header" 0 $?
inbox leader-2 | grep -q "new champion T2-014"; check "announcement body" 0 $?
[ -f $BASE/run/agents/manager/inbox.jsonl ]; check "sender/interactive not written" 1 $?
[ -f $BASE/run/agents/leader-3/inbox.jsonl ]; check "stopped agent not written" 1 $?
grep -q "new champion T2-014" $BASE/run/announcements.md; check "logged in announcements.md" 0 $?
echo "team T3 (exploit-long) received 80 GPU-min" > $BASE/a.txt
M $A announce -f $BASE/a.txt >/dev/null; c=$(grep -c "^## " $BASE/run/announcements.md); check "announce -f, log has 2 entries" 2 $c
M $A announce "" >/dev/null 2>&1; check "empty announcement refused" 1 $?

out=$(M $A remind); check "remind all" 0 $?
echo "$out" | grep -q "reminded 1 agent(s)"; check "only the exploit-long leader reminded" 0 $?
echo "$out" | grep -q "skipped odd: no reminder file for role brainstormer"; check "role without file skipped" 0 $?
echo "$out" | grep -q "skipped leader-1.w1: no reminder file for role worker"; check "worker not reminded" 0 $?
inbox leader-1 | grep -q "^\[reminder; no reply needed\]"; check "reminder header" 0 $?
inbox leader-1 | grep -q "^<system message>: You lead an exploit-long team."; check "exploit-long reminder body" 0 $?
inbox leader-2 | grep -q "<system message>"; check "exploit leader not reminded" 1 $?
out=$(M $A remind --role leader_exploit_long); echo "$out" | grep -q "reminded 1 agent(s): leader-1"; check "remind --role" 0 $?
out=$(M $A remind leader-1 nobody); echo "$out" | grep -q "reminded 1 agent(s): leader-1"; check "remind by name" 0 $?
echo "$out" | grep -q "skipped nobody: no such agent"; check "unknown name reported" 0 $?
AUTOINFER_ROLE=leader_exploit_long AUTOINFER_AGENT=leader-1 $A remind >/dev/null 2>&1; check "leader can't remind" 1 $?

h=$(AUTOINFER_ROLE=leader_exploit_long python3 $T/_remind_hook.py | python3 -c "import json,sys;d=json.load(sys.stdin)['hookSpecificOutput'];print(d['hookEventName'], 'exploit-long team' in d['additionalContext'])")
check "compact hook: exploit-long reminder as additionalContext" "SessionStart True" "$h"
h=$(AUTOINFER_ROLE=manager python3 $T/_remind_hook.py | wc -c); check "compact hook silent for manager" 0 $h
h=$(AUTOINFER_ROLE=worker python3 $T/_remind_hook.py | wc -c); check "compact hook silent for worker" 0 $h
h=$(AUTOINFER_ROLE=brainstormer python3 $T/_remind_hook.py | wc -c); check "compact hook silent without file" 0 $h
h=$(env -u AUTOINFER_ROLE python3 $T/_remind_hook.py | wc -c); check "compact hook silent without role" 0 $h
s=$(cd $T && python3 -c "
import importlib.machinery,importlib.util,json,pathlib
l=importlib.machinery.SourceFileLoader('nr','new_run');m=importlib.util.module_from_spec(importlib.util.spec_from_loader('nr',l));l.exec_module(m)
s=m.confine(json.loads(pathlib.Path('agent_settings.json').read_text()),pathlib.Path('$BASE/run'))
h=s['hooks']['SessionStart'][0];print(h['matcher'],'_remind_hook.py' in h['hooks'][0]['command'], 'PreToolUse' in s['hooks'])")
check "new_run settings: compact hook + message guard" "compact True True" "$s"

# review fixes (2026-11-08)
echo "$out0" | grep -q "skipped leader-3: not running"; check "stopped agent reported" 0 $?
t=$(python3 -c "import json;print(json.load(open('$BASE/run/agents/leader-1/meta.json')).get('last_task'))"); check "announce/remind leave last_task alone" None "$t"
mkdir -p $BASE/run/agents/broken; echo '{"role": "worker"}' > $BASE/run/agents/broken/meta.json
mkdir -p $BASE/run/agents/broken2; echo '[1,2]' > $BASE/run/agents/broken2/meta.json
M $A announce "odd folders" >/dev/null 2>&1; check "odd meta.json doesn't crash announce" 0 $?
out=$(M $A remind manager); echo "$out" | grep -q "skipped manager: that is you"; check "remind yourself reported" 0 $?
M $A announce x -f $BASE/a.txt >/dev/null 2>&1; check "text and -f together refused" 1 $?
M $A announce -f $BASE/nope.txt 2>&1 | grep -q Traceback; check "missing -f file: no traceback" 1 $?
out=$(M $A remind --role nosuchrole); echo "$out" | grep -q "no running agent has role nosuchrole"; check "unknown role hint" 0 $?
pids=""; for i in 1 2 3 4 5 6; do M $A announce -f $BASE/a.txt >/dev/null 2>&1 & pids="$pids $!"; AUTOINFER_AGENT=human $A send leader-2 hello >/dev/null 2>&1 & pids="$pids $!"; done
wait $pids; n=$(grep -c "received 80" $BASE/run/announcements.md); check "concurrent announces all logged" 7 $n
n=$(inbox leader-1.w1 | grep -c "received 80"); check "concurrent announces all delivered" 7 $n
# moonshot teams get no announcements (history-free); their leader gets the moonshot reminder
mkdir -p $BASE/run/gpu $BASE/run/agents/leader-9
echo '{"event": "open", "team": "T9", "type": "moonshot", "by": "manager"}' > $BASE/run/gpu/accounts.jsonl
python3 -c "import json;json.dump({'name':'leader-9','role':'leader_moonshot','team':'T9','pid':$LIVE},open('$BASE/run/agents/leader-9/meta.json','w'))"
out=$(M $A announce "team T4 (exploit-long) closed"); echo "$out" | grep -q "skipped leader-9: moonshot team"; check "announce skips moonshot team" 0 $?
inbox leader-9 | grep -q "T4 (exploit-long) closed"; check "moonshot leader not written" 1 $?
grep -q "T4 (exploit-long) closed" $BASE/run/announcements.md; check "still logged in announcements.md" 0 $?
out=$(M $A remind leader-9); inbox leader-9 | grep -q "You lead a moonshot team."; check "moonshot reminder" 0 $?
kill $LIVE
echo "== $pass passed, $fail failed"
