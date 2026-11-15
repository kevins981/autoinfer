## 2026-11-15T15:19:45+00:00 (manager)
team T1 (sweep) opened with 30 GPU-min

## 2026-11-15T15:19:45+00:00 (manager)
team T2 (sweep) opened with 30 GPU-min

## 2026-11-15T15:27:51+00:00 (manager)
BASE unofficial reference: 634.5 / 643.4 tok/s (2 starts, 200 prompts), bank/BASE/unofficial/1/. This is the number to beat.

## 2026-11-15T15:36:27+00:00 (manager)
T2-001 (team T2) promoted to unofficial champion: --compilation-config max_prefill_cudagraph_capture_tokens=2048, unofficial 648.6 / 658.6 tok/s (+2.3% vs BASE 634.5 / 643.4), 128K pass. Number to beat: bank/T2-001/unofficial/1/. Official champion remains BASE 635.9.

## 2026-11-15T15:57:01+00:00 (manager)
T1-002 (team T1, builds on T2-001) promoted to unofficial champion: NCCL_PROTO=Simple default in pynccl.py; unofficial 662.3 / 655.4 tok/s (env variant 663.2 / 660.0), 4-start mean 660.2 vs T2-001 653.6 (+1.0%). Number to beat: bank/T1-002/unofficial/1/. Official champion remains BASE 635.9.

## 2026-11-15T15:58:09+00:00 (manager)
team T2 (sweep) received 30 GPU-min; resync report: reports/T2-1.md

## 2026-11-15T15:58:33+00:00 (manager)
team T1 (sweep) received 30 GPU-min; resync report: reports/T1-1.md

## 2026-11-15T16:14:22+00:00 (manager)
REVERT: T1-002 is no longer unofficial champion (its own leader found the NCCL_PROTO default in pynccl.py is set too late: the profile shows only LL kernels). Unofficial champion is T2-001 again: 648.6 / 658.6 tok/s, bank/T2-001/unofficial/1/. Official champion remains BASE 635.9.

## 2026-11-15T16:28:00+00:00 (manager)
T2-004 (team T2, builds on T2-001) promoted to unofficial champion: reduced-vocab MTP draft head (ids [0,65536)) + local draft argmax + max-num-batched-tokens 1024; unofficial 660.0 / 664.2 tok/s (+1.3% vs T2-001), 128K pass. Number to beat: bank/T2-004/unofficial/1/. Official champion remains BASE 635.9.

## 2026-11-15T16:41:50+00:00 (manager)
T1-006 (team T1, builds on T2-004) promoted to unofficial champion: target-side distributed argmax for greedy batches; unofficial 676.6 / 680.7 tok/s (+2.5% vs T2-004). Number to beat: bank/T1-006/unofficial/1/. Official confirmation eval of T1-006 is starting (MGR).

## 2026-11-15T16:48:56+00:00 (manager)
T2-005 (team T2, builds on T2-004) promoted to unofficial champion: checkpoint W8A8 via --quantization humming (+ 3 fixes); unofficial 715.7 / 720.2 tok/s (+5.8% vs T1-006), GSM8K 93.2%, 128K pass. Number to beat: bank/T2-005/unofficial/1/. T1-006's official eval is running.

## 2026-11-15T16:57:28+00:00 (manager)
CONFIRMED: T1-006 is the official champion: official 712.1 tok/s (+12.0% vs BASE 635.9), GSM8K 93.6%, all gates pass (bank/T1-006/official/1/). This confirms T2-001 and T2-004 as well. Unofficial champion remains T2-005.

## 2026-11-15T17:04:39+00:00 (manager)
T2-006 (team T2) promoted to unofficial champion: official champion T1-006 + T2-005's humming W8A8; unofficial 738.8 / 747.8 tok/s (+3.5% vs T2-005), GSM8K 93.2%. Number to beat: bank/T2-006/unofficial/1/. Official eval of T2-006 starting.

## 2026-11-15T17:05:02+00:00 (manager)
team T2 (sweep) received 60 GPU-min; resync report: reports/T2-2.md

## 2026-11-15T17:14:14+00:00 (manager)
CONFIRMED: T2-006 is the official champion: official 761.6 tok/s (+19.8% vs BASE 635.9), GSM8K 93.6%, all gates pass (bank/T2-006/official/1/). This confirms T2-005's W8A8. T2-006 is both unofficial and official champion.

## 2026-11-15T17:40:40+00:00 (manager)
T1-008 (team T1, builds on T2-006) promoted to unofficial champion: drafter attention metadata without the blocking seq_lens.cpu(); unofficial 758.8 / 762.4 tok/s (+2.3% vs T2-006). Number to beat: bank/T1-008/unofficial/1/. Official champion remains T2-006 761.6.

## 2026-11-15T17:41:11+00:00 (manager)
team T1 (sweep) received 30 GPU-min; resync report: reports/T1-2.md

## 2026-11-15T18:03:06+00:00 (manager)
T2-008 (team T2, builds on T1-008) promoted to unofficial champion: GDN replay commit in the MTP verify kernel; unofficial 792.8 / 789.1 tok/s (+4.0% vs T1-008), GSM8K 93.2%, 128K pass. Number to beat: bank/T2-008/unofficial/1/. Official eval of T2-008 starting.

## 2026-11-15T18:12:04+00:00 (manager)
FINAL: T2-008 confirmed official champion, 835.1 tok/s (+31.3% vs BASE 635.9), GSM8K 93.6%. Run stopped by the human; final report: reports/final.md

