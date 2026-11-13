# LOCAL_KB leader-52 (T52, exploit-long, direction: megakernels)
## Cards (first set)
- Boundaries/Distance/Generality: per-kernel prologue+tail (~6.5 us/launch, KB98) and HBM idle between kernels are exactly the
  "crossing cost" a persistent kernel removes; weights of op N+1 don't depend on op N's activations -> can be in flight early.
- Fetching-without-waiting (cp.async ring) + no PDL/clusters/TMA on sm_89 -> a persistent kernel is the only way to overlap
  next-op weight fetch with the current op's tail on this chip.
## Starting facts (from KB)
- decode pass 24.9 ms @M=64 (k=3); nsys decode: bs_gemm 5.3 + t31b 6.7 GEMM, AR 7.0, busy 24.8 (T51-001 job 434)
- HBM bytes/step ~15.5 GB -> ~16.3 ms floor at 0.95 TB/s => HBM ~65% busy in decode. Prize ~ the other 35%.
- AR windows already prefetch ~32 MB each into L2 (KB97), so AR time isn't all HBM-idle.
- mixed steps ~30% of wall, ~51 ms floor, host gap 4-8 ms (KB131).
## Plan v0 (2026-11-11)
- Per-GDN-layer in situ (T51 nsys, node trace): qkvz 49 (floor 44), gdn_out 22, AR1 ~56, gate_up 82 (floor 94; ~9 MB from hint pf),
  down 53, AR2 ~46, GDN replay 41, small ~17 us. AR windows ~mostly HBM-idle (hint pf lands 11-30 MB; T31: real loads 32 MB cap
  is the AR-kernel-end coupling = boundary cost).
- Bet: fused [AR + add/norm + quant + consumer GEMM] persistent kernel; GEMM CTAs stream own slab during AR.
  Napkin: AR2+qkvz ~100 -> ~60 us, AR1+gate_up ~143 -> ~100 us => ~80 us/GDN layer ~ 4-5 ms/step (~17% decode, ~+10% e2e if decode ~65% wall).
- W1-1 to leader-52.w1 (task agents/leader-52/task_w1_1.md), workspace t52-w1 (=T50-002, built). Allowance 12 of 20 seed.
- Open Qs: decode share of wall (compute from T51 steplog); AR latency under concurrent HBM streaming; spar before proposal.
## Spar T52-1 (ended)
- Corrected napkin: warm qkvz ~26 us@M48 (T31-002) -> fused AR2+qkvz ~76 vs ~100; AR1+gate_up ~106 vs ~143 => ~60 us/GDN layer
  x48 = 2.9 ms + attn layers (16 x ~50?) ~0.8 ms => ~3.7 ms/step (assumed; count explicitly in proposal).
- Assumed/to test: AR latency with persistent GEMM grid resident + 40 MB stream (falsifier, W1-1a); warm qkvz M=64; 40 MB evict_last
  survival while gate_up streams the rest 49 MB.
- Decode share of wall 65% (T51 b2048 steplog: 29.4 s decode / 15.7 s mixed).
## Split with T53 (agreed): T52 = consumer side (AR read+norm+quant+next GEMM); T53 = producer side (chunked output posting + per-chunk flags; agents/leader-53/INTERFACE.md).
- 2026-11-11 manager ruling (via leader-53; see task_definition.md): deferring per-row 1/rms past the GEMM is ALLOWED, also on W8A8
  (group-128 dynamic scales on the un-normalized x*g, fp32 per-K-block promotion unchanged, r multiplies fp32 result; needs diffs +
  FP8 GSM8K A/B + 128K). => consumer GEMM may start on K-chunks as AR chunks arrive (cut-through), not only after the full row.
## W1-1 result (T52-001): NEGATIVE. fused whole-row AR2+qkvz 103.5 vs champion 88.5-90.4 us. Champion hint pf already brings qkvz to
  warm floor (34 vs 32.7). Full 42 MB stream in AR window stretches AR +10 us. My in-situ premise (qkvz 49) was nsys-node inflation?
  M=64 GEMMs near compute ridge: per layer 24.5 GFLOP ~ 80-120 us vs 201 us HBM; warm qkvz ~30 us ~ compute/latency-bound.
- Next: W1-2 K-chunk cut-through consumer (r-deferral). Est. saves ~read+add+norm+quant (~20-25 us) per AR -> ~3 ms/step if all 128.
- T53 INTERFACE.md v1: C=8 x 640 cols, each chunk hostar-0.78-codec encoded separately ([u32 nbytes][codec bytes]), u64 seq flag/chunk, slot seq&1, per-slot ack.
- T53 producer (T53-001..003): host flag polling by many CTAs = 3x slower GEMM (use 1 poller + device relay); host reads during GEMM +6-14 us; coupled down->AR->qkvz C2+codec saves 10-12 us/boundary; tail (last chunk post+read ~40 us) is the limiter.
## W1-2 (T52-002): PARITY. cut-through 89.0-91.3 vs 88.3-91.8. chunk usable ~20 us after publish; first publish 18-25 us -> GEMM starts ~45 us.
  => decode consumer fusion closed (with host-staged transport). Cause: chunk latency ~20-40 us ~ decode GEMM length.
## Pivot candidate: mixed/prefill steps. T38 champion node nsys (agents/leader-38/prof_champ_node, T37-era k=3): mixed step 319 prompt+60 dec
  85 ms (nsys): GEMM 53 ms (~175 TF), copies union 22 ms, AR kernels 9.6, GDN pre 5.4, quant 2.9; compute union 52, busy union 81
  => exposed comm ~29 ms. Mixed = 35% wall. Hiding 70% => ~+8% e2e. Producer N-chunk posting + consumer K-chunk cut-through, CE copies.
## Mixed-step layer timeline (mix2.py on T38 champ nsys, 319+60 rows): per layer ~1.35 ms; token-split halves' GEMMs run CONCURRENTLY,
  AR1 ~260 us + AR2 ~280 us of CE copies (8x480KB each dir, ~24 us/piece) with NO compute => ~550 us exposed (~40%).
  KB67's 86-96% hiding was microbench; in-situ not staggering (KB89).
- Proposal T52-1 submitted (+9.1% if works, central +3%, 30%; ask ~50 GPU-min). M1 = W1-3 gate 1 running.
- FUNDED: +50 GPU-min (proposal T52-1). Plan: M1 seed, M2 10, M3 25, M4 12, reserve 3.
- T53 CLOSED (reports/T53-writeup.md). For M2: CE D2H via wait-value on device flag lands 17-32 us after flag, GEMM ~0 cost;
  t31c chunk-ordered GEMM (agents/leader-53.w1/ext/t53c.cu) decode down C2 1.054x C4 1.081x; hazards: job 446 hung (wait-value order),
  450 lazy module loading (warm all kernels on both GPUs before first host-flag barrier). benchC.py coupled 2-GPU bench reusable.
## W1-3 gate 1 (jobs 460-465, 3.8 GPU-min; 3.3 lost to a spin bug): producer hooks +0.2..1.2% bitwise PASS; bs_gemm already n-major & K-ascending.
  Chunk release wave-quantized: first chunk at 60-75% of GEMM (out M384 31/50 us; down M384 80/130). Consumer hooks v4 +7-10% (suspect
  A-scales ld.cg vs __ldg) FAIL 3% -> ablate.
## Gate 2 timeline model (GPU-free, mine): per AR CE pieces 480KB ~24 us/dir, CE latency ~20 us.
  AR1: out rel 31 -> 1st chunk at peer ~110, last ~280; gate_up paced (22 us/chunk) ends ~300 vs today 50+260+10+180=500 => -200 us.
  AR2: down rel 80 -> last lands ~326; qkvz ends ~340 vs today 130+280+10+85=505 => -165 us. ~365 us/layer of ~1350 (27%), hidden ~66%.
  Consumer +8% costs ~21 us/layer. Gate 2 PASS on model (assumed CE rates in-situ; full-M GEMM replaces token split).
- W1-4(1) ablation (jobs 466-468): scales via cp.async stage cfg103 -0.7..-2.2% (faster than champ), cfg127 ld.cg +1-1.5%; r +0.3-2.8%;
  flag poll +7-8.7% unresolved (not latency; tid0 branch in 2-stage loop?). Best consumer combo +4-5% (out +6%). Fine for M2.
- CORRECTION to my gate-2 model (worker): consumer GEMM has ~3.2 waves; only wave 1 overlaps chunk arrival -> B (full-M, chunked CE, serial
  producer->transfer->consumer) est. 860-1030 us/layer vs A (token-split pingpong) 750-850 in chain. Transfer of 3.9 MB (~200 us/dir) > producer.
  Plan: (c) measure A in chain (natural vs server-like concurrent halves). If chain-A hides but server doesn't -> scheduling target + hybrid (b).
## M2(c) (jobs 469/470, 0.8 GPU-min): chain M=384 split 192/192, NL=4 tails. Layer/no-comm/exposed us: E 1198/740/458; G(concurrent) 1083/630/453;
  S(stagger) 1073/708/364; P(link priority chunk1 AR waits chunk0 AR) 1020/708/312; Q 1041/746/296. ONE D2H CE: lanes interleave -> both ARs
  finish together, no compute beside (= server window). Piece gap 7-8 us (~30 us/AR). Link floor ~375 us/layer vs compute 600-700 -> floor ~700+fill.
  Next: gap screen on P, then hybrid (b) on P'; single server ABBA of stack vs T50-002 (env flags P/P'/(b) separate).
- Gap screen (jobs 471-472): P 1013, P2 1020, P3 1010, Pe 1121: no gain, P'=P. Engine flag VLLM_T52_LINKPRI (default 0) in _qwen3_5_split_chain.
- (b) est 890-920 vs P 1013 (out/down single wave at M=192 -> producer release gains ~0). ON HOLD.
- Steplog T51 b2048 share of wall: decode 65.8%, mix<256 10.0% (49.7 ms), 256-383 2.5%, 384-639 8.4% (107 ms), 640+ 13.3% (174 ms).
  Split steps 24.2%. P alone ~1.5% e2e est. Cross-layer pipeline (remove per-layer join; = T30 waves + T32 captured core + P) up to ~5-6%.
  Reading feasibility (worker, GPU-free).
- Feasibility read (worker): T30 waves in workspaces/t30-wave (_t30_wave_layers, vllm/t30_wave.py, t30_wave_attn); state handoff via cache with
  truncated wave-0 request; T32 captured core (t32-cap, NOT in champion) needs per-wave meta copy + extra event; host est ~40-45 ms/step with
  captured cores (GPU span 100-170). Chain screen J/W/WP at M=384/736 NL=8 approved (<=0.7 GPU-min). Port to T50 if pass.
- Job 473 chain (NL=8): J/W/WP M384 1158/1219/984 (floors 833/765), M736 1894/2026/1664. WP -12..15% vs J. Gate missed (WP +220/+359 over floor;
  peer D2H lag ~100 us = rank skew). Est +3.5-4.5% e2e if host hidden. Filed T52-004. Port approved: steps 1-3 (8 GPU-min + 4 reserve).
- Step 2 (job 474, 2.4 GPU-min): wave port in t52-w1 (T52_WAVE, T52_WAVE_PRI, T52_WAVE_CTL). Host-bound: wave steps host 196 ms vs GPU span 178 (true GPU ~100-130). L0 stages match T30 (rel 1e-3..1e-2). Token streams inconclusive (on-on unstable at c=4). Step 3 = T32 cap port + c=1 sequential token test + mem check.
- Step 3 (job 475, 3.1 min): T32 cap ported (2160 graphs, free 1.00->0.45 GiB, bitwise vs eager); wave steps host 120 vs GPU span 131 ms; c=1 tokens on/on & off/off identical, on/off diverge (M-split rounding). ABBA abba1 launched (400 prompts, gsm8k 250 on last B, then 128K on B).
## ABBA job 477 (13.7 min): A = t52-w1 flags off (=T50-002), B = T52_WAVE=1 T30_WAVE_MIN=768 T32_GDN_CAP=1 NTMAX=16 LINKPRI=1:
  A 1182.6/1183.8, B 1213.0/1220.0, all B > all A +2.82%. GSM8K-250 (B) 93.6%. 128K pass (B). p99 TTFT/TPOT better.
  (Cancelled job 476 at 1.2 min to change MIN 384->768.) Balance 41.8. Next: defaults on, record T52-005, official.
- Record T52-005 filed (defaults on; validity reviewed). official_eval T52-005 running (~11 GPU-min).
- Next levers (worker est.): (b) host cost of eager wave pieces -> MIN 384: +0.8-1.2%; (a) rank skew +0.4-0.9%; codec on DMA +0.3-0.6%; 3 waves 0-0.6%; head/tail ~0. Approved (a) stamp test + (b) host trace, 4 GPU-min, in ws t52-w2.
- OFFICIAL T52-005 #1: 1212.4 (1210.1/1214.8) +2.6% vs T50-002; GSM8K 93.2; p99 TTFT 667 TPOT 26.6; 128K pass. Reported to manager as champion candidate. Balance 34.1. Job 479 (Q + b(i)) running.
- T52-005 PROMOTED champion (1212.4). New +5% bar = 1273.0.
- Q/WD negative (T52-006). b(i): chain eager issue 325-421 us/wave-layer vs server ~0.9-1.1 ms -> 0.5-0.7 ms vLLM overhead. b(ii) approved 2.5 min.
- b(ii) (job 480, T52-007): host 117 ms/wave step: GDN core glue 1/3, linear wrappers 1/3, AR issue 1/5. Fix to MIN 384 ~3-4 days for est ~+1%. Balance ~32.
- Resync: manager chose (i) host-cost follow-up, no new grant (31.9). Worker: route 1 a-d in t52-w2, <=5 GPU-min smoke; I run ABBA t52-w2 vs t52-w1 then official.
- Route 1 (job 481, T52-008): lean loop bitwise, host 117 -> 85-100 ms; MIN only ~650-700; est +0.2-0.5% -> stop (A). Closing team.
- 2026-11-12: write-up reports/T52-writeup.md; recommended close to manager; 28.5 GPU-min unspent.
- Spar T52-2: 'nothing left' ceiling withdrawn: WP residual is schedule-exposed (link ~375 us vs compute ~700/layer @384). Conditional mixed headroom ~6% e2e. Proposed to manager a <=0.5 GPU-min chain screen (3 waves; copy pieces at producer wave release) before closing.
- Screen job 482 (T52-009): W3P -9% @736, flat @384; residual vs Wn -42% (rule: halve) -> close. Writeup addendum done.
