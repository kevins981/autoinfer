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

## 2026-11-14T17:15:20+00:00 (manager)
new champion T1-004 from team T1: 707.6 tok/s (+11.3% over BASE); how: bank/T1-004/

## 2026-11-14T17:57:54+00:00 (manager)
new champion T2-002 from teams T2 + T1 (combination of T1-005 and T2-001): 757.6 tok/s (+7.1% over T1-004, +19.1% over BASE); how: bank/T2-002/

## 2026-11-14T18:32:29+00:00 (manager)
new champion T1-009 from team T1: 786.4 tok/s (+3.8% over T2-002, +23.7% over BASE); how: bank/T1-009/

## 2026-11-14T18:32:29+00:00 (manager)
team T1 (sweep) received 80 GPU-min; resync report: reports/T1-1.md

## 2026-11-14T18:59:48+00:00 (manager)
new champion T2-006 from teams T2 + T1 (T2-004 + T1-009's TRITON_ATTN): 819.9 tok/s (+4.3% over T1-009, +28.9% over BASE); how: bank/T2-006/

## 2026-11-14T18:59:48+00:00 (manager)
team T2 (sweep) received 60 GPU-min; resync report: reports/T2-1.md

## 2026-11-14T20:29:32+00:00 (manager)
new champion T1-014 from team T1: 842.7 tok/s (+2.8% over T2-006, +32.5% over BASE); how: bank/T1-014/

## 2026-11-14T21:16:35+00:00 (manager)
team T1 (sweep) closed at its own recommendation after 3 official wins (T1-004, T1-009, T1-014); write-up: reports/T1-writeup.md

## 2026-11-14T21:16:35+00:00 (manager)
team T3 (sweep) opened with 100 GPU-min (seed 20 + start 80)

## 2026-11-14T21:23:56+00:00 (manager)
new champion T2-008 from team T2: 880.1 tok/s (+4.4% over T1-014, +38.4% over BASE); how: bank/T2-008/

## 2026-11-14T22:26:23+00:00 (manager)
new champion T3-003 from team T3: 897.4 tok/s (+1.97% over T2-008, +41.1% over BASE); how: bank/T3-003/

## 2026-11-14T23:18:21+00:00 (manager)
team T3 (sweep) received 60 GPU-min; resync report: reports/T3-1.md

## 2026-11-14T23:26:43+00:00 (manager)
new champion T2-009 from team T2: 906.9 tok/s (+1.06% over T3-003, +42.6% over BASE); how: bank/T2-009/

## 2026-11-14T23:26:43+00:00 (manager)
team T2 (sweep) received 60 GPU-min; resync report: reports/T2-2.md

## 2026-11-15T00:10:22+00:00 (manager)
new champion T3-006 from team T3: 919.0 tok/s (+1.33% over T2-009, +44.5% over BASE); how: bank/T3-006/

## 2026-11-15T00:35:09+00:00 (manager)
new champion T2-011 from team T2: 931.8 tok/s (+1.39% over T3-006, +46.5% over BASE); how: bank/T2-011/

## 2026-11-15T01:07:21+00:00 (manager)
new champion T3-009 from team T3: 951.5 tok/s (+2.11% over T2-011, +49.6% over BASE); how: bank/T3-009/

## 2026-11-15T01:07:21+00:00 (manager)
team T3 (sweep) received 50 GPU-min; resync report: reports/T3-2.md

## 2026-11-15T02:16:02+00:00 (manager)
team T2 (sweep) closed at its own recommendation after 5 official wins (T2-002, T2-006, T2-008, T2-009, T2-011); write-up: reports/T2-writeup.md

## 2026-11-15T02:16:02+00:00 (manager)
team T4 (sweep) opened with 100 GPU-min (seed 20 + start 80)

## 2026-11-15T02:30:19+00:00 (manager)
team T3 (sweep) received 35 GPU-min; resync report: reports/T3-3.md

## 2026-11-15T03:21:53+00:00 (manager)
run ended by the human; final report: reports/final.md

