# R10-A bitwise-lossless host-AR payload compression (leader-2.w18), ws leader-2.arcomp
Kernel: host_staged_ar.cu mode&16; py: VLLM_HOSTAR_COMP (default 1), VLLM_HOSTAR_COMP_MIN_BYTES (default 0).
Gate job 117 (v1 sparse-overflow layout, host_staged_ar.v1.cu): bitwise OK; M=108/150 only 0.90-0.95 (dependent PCIe read per raw block).
Gate job 120 (v2 group exception slots): bitwise 0 mismatches (300 real pairs + 30 adversarial); M>=108 0.76-0.78 of raw time;
  M=48 realistic pf 50.9->44.4 / 46.5->39.5 us; trace M=48 45.5->38.0 us.
ABBA job 121 (abba1/): tok/s A1 834.7, B1 848.7, B2 855.3, A2 835.5 -> pooled +2.0%; TPOT p50 A 16.84/16.68 B 16.39/16.43;
  ITL p50 A 27.38/27.37 B 27.02/26.97; TTFT p50 A 195.6/186.6 B 186.8/181.1; TTFT p99 A 1152/1311 B 1283/1279;
  accept A 2.34/2.341 B 2.329/2.346; gsm8k50 (B2) 90.0%.
