# leader-21 (T21, explore) notes

## Direction: the new-request prefill path, attacked through *what* is prefilled
- The model's chat template (chat_template.jinja l.46-52) injects a default system prompt
  "Reasoning effort is set to xhigh. ..." whenever thinking is on (default). Excerpt: agents/leader-21/template_excerpt.txt.
  So every request has a 45-token shared prefix (17% of 104.8K prompt tokens/trial). Measured by tokenizing vLLM's ShareGPT sample (agents/leader-21/prompt_lens.txt).
- Prefix caching is on but hits 0% (T14-004 official metrics). The cause is hybrid attention block size 1600 + align-mode Mamba checkpoints.
- The vLLM commit has off-by-default `--prefix-match-unit N` and `--enable-mamba-shared-prefix-checkpoint`.
- Ruling: the manager sent it to the human (2026-11-09). Until it lands: measurement only, record validity "PENDING RULING", no official eval, no building on it.
- Step model (KB77, f(R)=38.5+0.229*max(R,49), decode 25.3) over the prompt distribution:
  - cache 45 tokens: -3.2 s/trial, +3.5%;
  - + feed residual prefills <=45 tokens as forced-accepted draft rows in the uniform decode graph (slot cost ceil(u/3)*25.3/16 ms vs mixed excess): +5.5% total.
  - Feed alone without the cache: 0 (every prompt is >=51 tokens).
- Item 2 (feed) needs no ruling; it must show exactness (bitwise/bounded vs the mixed-step prefill) + GSM8K.

## Jobs
- pc16: flags-only, unit 16, 400 prompts (agents/leader-21/out/pc16).
- job 141 (pc16): 917.3 tok/s, hits 144/104718 tokens (= ~9 hits x 16). The stock mechanism fails; T21-001 filed PENDING RULING.
- proposals/T21-1.md sent (conditional on the ruling): payoff +5.5%, central +2.5%, 25%. Holding GPU work (14.8 left).
- Ruling APPROVED (task_definition.md end). Conditions: nothing sized to the 45-token prompt; the record must show a different shared prefix, the origin of unit/positions, no warm-up. Grant 30 now + 50 on the M1 gate.
- Worker leader-21.w1 started on M1 (task: agents/leader-21/task_w1_m1.md), allowance 25 GPU-min, workspace t21-a.
- w1 M1 status: the junction index works. The sim gives 44/45 hits, 1.00 steps/req; 4 prefixes tested. But same-job job 176: B 851.7 vs A 907.6 (-6.2%), acceptance 2.208 vs 2.331, and hits 16.9% of prompt tokens. Exactness logprob diff 0.03-0.39.
  My top suspect: the shared partial 1600-token block is written by several requests (KV corruption). Asked for a code read + a c=1 hit vs no-hit token test. Granted +12 (allowance 37).
- The acceptance drop was a stale GDN replay flag on CoW destination blocks; fixed (acceptance equal). Partial hits are CoW, not shared.
- Same-job A/B: job 182 (A,B) -4.7%, job 184 (B,A) -4.2%, hits 16.9% of prompt tokens. Expected +3.5%: the ~19 ms/request loss is unexplained. Record bank/T21-002.
- Next: CPU sim timing of the scheduler (unit 1/16/off), code read of the copy path, then one traced A/B (T19 tracer). w1 allowance +8 (45 total).
- New champion T27-002 967.3 (a drafter lm_head change). Bar is now 1015.7. M1/M2 act on prefill, orthogonal, so they should stack. Rebase onto T27-002 before any official eval.
- CLOSED: M1 +0.5% (job 189) vs gate +2.5%. Write-up reports/T21-writeup.md. The run is pausing.
