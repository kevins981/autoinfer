#!/bin/bash
# Tests for per-team model slots (`agent start --slot`, config.md "slot" lines).
# A stub `claude` records its arguments; no real sessions, no GPU.
T=/scratch/kevinsong/learning/autoinfer/flavors/v2/tools
A=$T/agent
BASE=/scratch/kevinsong/learning/autoinfer/.cache/slottest
rm -rf $BASE; mkdir -p $BASE/run/agents $BASE/bin
export AUTOINFER_RUN=$BASE/run
cat > $BASE/bin/claude <<'S'
#!/bin/bash
echo "$@" > "$AUTOINFER_RUN/last_args"; sleep 30
S
chmod +x $BASE/bin/claude; export PATH=$BASE/bin:$PATH
echo '{}' > $BASE/run/agent_settings.json
cat > $BASE/run/config.md <<'C'
- models (role: model / effort):
  - manager: claude-opus-5-5 / medium
  - leader_exploit_long: claude-opus-5-5 / medium
  - worker: claude-sonnet-5-5 / medium
- model slots:
  - slot A: leader claude-fable-5-1 / medium, worker claude-sonnet-5-5 / medium
  - slot B: leader claude-opus-5-5 / high, worker claude-opus-5-5 / medium
C
pass=0; fail=0
check() { if [ "$2" == "$3" ]; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1 (expected $2, got $3)"; fail=$((fail+1)); fi; }
M() { AUTOINFER_ROLE=manager AUTOINFER_AGENT=manager "$@"; }
L() { AUTOINFER_ROLE=leader_exploit_long AUTOINFER_AGENT=$1 "${@:2}"; }
model() { python3 -c "import json;m=json.load(open('$BASE/run/agents/$1/meta.json'));print(m['model'],m['effort'])"; }
wait_args() { for i in $(seq 50); do [ -s $AUTOINFER_RUN/last_args ] && break; sleep 0.1; done; }

M $A start leader-1 --role leader_exploit_long --team T1 --task x >/dev/null 2>&1; check "leader without --slot refused" 1 $?
M $A start leader-1 --role leader_exploit_long --team T1 --slot C --task x >/dev/null 2>&1; check "unknown slot refused" 1 $?
rm -f $AUTOINFER_RUN/last_args
M $A start leader-1 --role leader_exploit_long --team T1 --slot A --task x >/dev/null; check "leader in slot A" 0 $?
check "slot A leader model" "claude-fable-5-1 medium" "$(model leader-1)"
wait_args
grep -q -- "--model claude-fable-5-1" $AUTOINFER_RUN/last_args; check "claude launched with fable" 0 $?
L leader-1 $A start leader-1.w1 --role worker --team T1 --task x >/dev/null; check "worker start" 0 $?
check "slot A worker model" "claude-sonnet-5-5 medium" "$(model leader-1.w1)"
L leader-1 $A start leader-1.w2 --role worker --team T1 --model claude-opus-5-5 --task x >/dev/null 2>&1; check "agent can't override slot model" 1 $?
M $A start leader-2 --role leader_exploit_long --team T2 --slot B --task x >/dev/null
check "slot B leader model" "claude-opus-5-5 high" "$(model leader-2)"
L leader-2 $A start leader-2.w1 --role worker --team T2 --task x >/dev/null
check "slot B worker model" "claude-opus-5-5 medium" "$(model leader-2.w1)"
M $A start leader-3 --role leader_exploit_long --team T2 --slot A --task x >/dev/null 2>&1; check "team can't change slot" 1 $?
M $A start leader-4 --role leader_exploit_long --team T3 --slot A --task x >/dev/null; check "replacement team in slot A" 0 $?
check "team_slots.json" '{"T1": "A", "T2": "B", "T3": "A"}' "$(python3 -c "import json;print(json.dumps(json.load(open('$BASE/run/team_slots.json'))))")"
M $A start leader-9 --role manager --task x >/dev/null 2>&1   # role without team: role line
check "no team: role line" "claude-opus-5-5 medium" "$(model leader-9)"
$A start leader-1.w3 --role worker --team T1 --model claude-opus-5-5 --task x >/dev/null; check "human may override" 0 $?
check "human override model" "claude-opus-5-5 medium" "$(model leader-1.w3)"

for n in $(ls $BASE/run/agents); do AUTOINFER_AGENT= $A stop $n >/dev/null 2>&1; done
echo "passed $pass, failed $fail"; [ $fail == 0 ]
