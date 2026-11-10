#!/bin/bash
# scout tool: starts a (stub) codex in the background, copies cards into the run, status, role check.
set -u
T=$(cd "$(dirname "$0")/../.." && pwd)
D=$(mktemp -d); trap 'rm -rf "$D"; rm -rf "$T/../../.cache/scout_work/run-TX-"*' EXIT
pass=0; fail=0
ok() { if eval "$2"; then pass=$((pass+1)); else fail=$((fail+1)); echo "FAIL $1"; fi; }
mkdir -p "$D/bin" "$D/run/agents"
cat > "$D/bin/codex" <<'S'
#!/bin/bash
cat > "$PWD/../stdin_seen.txt"
echo "# A" > 01-a.md; echo "# B" > 02-b.md; echo "# index" > index.md
echo "$CODEX_HOME" > "$PWD/../codex_home.txt"
S
chmod +x "$D/bin/codex"
printf -- "- scout: codex gpt-test / low, cards 2\n" > "$D/run/config.md"
export AUTOINFER_RUN="$D/run" PATH="$D/bin:$PATH"
out=$("$T/tools/scout" --team TX 2>&1)
ok "starts" '[[ "$out" == *"deck TX-1 (gpt-test / low, 2 cards"* ]]'
for i in $(seq 1 20); do grep -q '"done"' "$D/run/scout/TX-1.json" 2>/dev/null && break; sleep 0.5; done
ok "done" 'grep -q "\"state\": \"done\"" "$D/run/scout/TX-1.json"'
ok "cards copied" '[ -f "$D/run/scout/TX-1/01-a.md" ] && [ -f "$D/run/scout/TX-1/index.md" ]'
ok "card count excludes index" 'grep -q "\"cards\": 2" "$D/run/scout/TX-1.json"'
W=$(python3 -c "import json;print(json.load(open('$D/run/scout/TX-1.json'))['work'])")
ok "prompt filled" '! grep -q "{N_CARDS}\|{OUT_DIR}\|{WINDOW_START}" "$W/prompt.md" && grep -q "2 cards, then stop" "$W/prompt.md"'
ok "runs outside the run" '[[ "$W" != "$D/run"* ]]'
ok "private codex home" 'grep -q "/.cache/codex" "$W/codex_home.txt"'
ok "status" '"$T/tools/scout" --status | grep -q "TX-1 .*done"'
"$T/tools/scout" --team TX >/dev/null 2>&1
for i in $(seq 1 20); do [ -f "$D/run/scout/TX-2.json" ] && grep -q '"done"' "$D/run/scout/TX-2.json" && break; sleep 0.5; done
ok "second deck numbered" '[ -d "$D/run/scout/TX-2" ]'
ok "leader can't start" '! AUTOINFER_ROLE=leader_external "$T/tools/scout" --team TX 2>/dev/null'
echo "passed $pass, failed $fail"
