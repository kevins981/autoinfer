# T4 map, update 1 (closing; the run was ended by the human)

## Tested since 6-ranking.md
Nothing ran on the GPU after the two profiles of T3-009 (jobs 139, 140; ~6 GPU-min of 100).
E1 never ran: job 144 was cancelled to switch workspaces, and job 145 was cancelled while queued when the run was paused.
There are no A/B numbers.

CPU-only finding (worker, agents/leader-4/e1_graphcheck.md / e1.md, script graphcheck.py on the node trace of job 140).
Observed:
- mixed steps of <= 512 tokens run as piecewise graphs; every GDN kernel and the full-attention kernel are eager between
  the pieces;
- mixed steps of > 512 tokens are fully eager: 38 of 3,508 kernels in a graph, sample period 219 ms vs ~84 ms for a
  graphed 355-token step;
- the row-split path (>= 256 tokens) has the same piecewise structure.
This confirms the premise of candidate 1: the GEMMs of > 512-token steps are uncaptured.

## Verdicts
Unchanged from 6-ranking.md. Nothing new was measured, so nothing moves.

## Current ranking, handed to a later run (vs T3-009, 951.5 tok/s, 95.9 s)
| # | Candidate | Napkin | Status / cheapest test |
|---|---|---|---|
| 1 | CUDA-graph capture sizes 576..1024 (8 sizes) for the 76 benchmark prompts whose step exceeds 512 tokens | +5-6% | **untested on current code** (T2's 83-size attempt OOMed on Marlin-era code; their "GPU-bound anyway" verdict used a Marlin per-row slope). Run `agents/leader-4/e1_job.sh` (gpu run --timeout 30): long_request 128000 on B, then abba AB at 400 prompts with per-step logs. |
| 2 | Small mixed steps (232 prompts, P+45 < 256): eager GDN core is CPU-bound (graph trace ~1.4 ms CPU vs ~0.2 ms GPU per GDN layer) -> graph-capture the mixed GDN core, keyed on padded prompt bucket and number of spec decodes (padding rows with g=0, beta=0, k=v=0 leave the state exactly unchanged) | +9-11% if the nsys reading holds; <1% if T2's gap measurement (~2.5 ms/step) holds | **disputed**; the same E1 job settles it: `python agents/leader-4/buckets.py agents/leader-4/e1/abba` gives median non-nsys step time per bucket (decode / <256 / 256-512 / >512). Build only if <256-token mixed steps median >= ~60 ms vs ~26.5 ms decode. |
| 2b | Merge small prefills (scheduler admits pending prompts together) | up to half of #2 | fallback if #2 is real but too costly; watch TTFT p99 (gate 1609 ms) |
| 3 | Spec-decode rows of mixed steps on the CUDA MTP kernel instead of the Triton path + materialize | ~+0.8% | untried; moderate build |

E1 setup, ready (not run):
- **Workspace:** `workspaces/t4-steplog`, built: T3-009 plus T2's per-step GPU-event logger, measurement only.
- **Arm A:** T3-009's serve args.
- **Arm B:** the same args with capture sizes 576, 640, 704, 768, 832, 896, 960, 1024 added.
- **Script fix:** at closing I fixed the script's sed pattern (`448,512]`, no space); before that its guard would have
  aborted.
- **Memory:** graph memory is the risk, so the 128K request must pass on B, and the official eval reruns it.
