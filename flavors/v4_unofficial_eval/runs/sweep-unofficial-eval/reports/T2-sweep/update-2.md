# T2 map, update 2

HEAD = T2-012 (unofficial; chain on CONFIRMED T1-005 760.8: T2-011 +0.52%, T2-012 +0.85%).

## Tested since update-1
| id | record | result | verdict change |
|---|---|---|---|
| SP1 3 drafts | T2-008 | negative (1 start) -7.34%, acceptance 2.724, TTFT p99 1930 ms gate fail | **dropped** (also T1-003; I duplicated T1's test: check the bank first) |
| KR1 GDN 1-write state | T2-009 (worker) | parked: state corruption in server (layers 17/40, one row) after ~13.8 GPU-min | parked; needs align-mode-aware slot bookkeeping, high integration risk |
| W4 drafter TRITON_ATTN | T2-010 | server start OOM | dropped |
| SM4 reduced-vocab draft argmax | **T2-011** | PASS +0.52% vs T1-005, 128K pass; acceptance 2.337 vs 2.354 | **won (promoted)**. Net = 1/4 of napkin: acceptance loss 0.7% (docs text predicted 0.4%) and the head saving below napkin |
| KE1a Marlin tile config at M 17-48 | **T2-012** (t2-w1) | PASS +0.85% vs T2-011; gate_up -5% at M=48 (job 56) | **won (promoted)**; matches napkin (~0.36 ms/step) |

## Observations (Observed -> Interpretation)
- Marlin config sweep (job 49): the 3 compiled variants at m_blocks=3 are within ~5%; decode GEMMs run at 0.74-0.84 of
  ~1 TB/s. The rest is structural (1 block/SM, 4 warps, 4-stage cp.async; ~101 KB smem): a new kernel with more bytes in
  flight could reach ~0.9 TB/s -> ~1.3 ms/step, ~3-4%.
- Tokenizer structure (CPU): ids < 98304 = base BPE vocab (99.6% of English docs, 99.95% of Python); 98K-151K CJK.
  The benchmark's slightly larger acceptance loss suggests some non-English text in it; a larger N only adds CJK rows.

## Framework notes
- The framework prices the draft head by bytes; with a reduced vocab it needs an acceptance term (bytes saved vs
  acceptance lost), i.e. Speculate and Rewrite interact. My napkin ignored the out-of-subset rate on the real workload.

## Current ranking (napkin vs HEAD T2-012)
1. KE1b new FP8 W8A16 decode GEMM (or Humming kernel if faster): ~3-4%; high effort. Probe running (worker Task 8).
2. Mixed steps <= 512 tokens: ~10 ms idle per 88 ms step (19.8% of time) -> up to ~2% if the idle is removable (CPU
   work between graph pieces; needs a profile of HEAD).
3. KR1: up to ~4%, parked (integration risk).
4. W1b: graph mixed steps > 512 tokens: small (those steps are GPU-bound).
