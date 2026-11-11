"""Frequency-ranked draft-vocab subset (ruling 2: non-benchmark sources only).
Ranking: (1) token counts in T3's TRAIN split of self-generated target outputs+prompts
(agents/leader-3.w1/data/seqs.jsonl, ids in split.json['train']; prompts are T3's self-generated
Magpie/self-instruct prompts, not ShareGPT); (2) token counts in vllm repo text (docs/**/*.md + vllm/**/*.py,
as T1-013 used); (3) fill by id order. Report held-out coverage on split.json['heldout'] (response ids).
usage: build_draft_vocab.py <N> <out.npy> [--repo <dir>]"""
import json, sys, glob, os, numpy as np
N=int(sys.argv[1]); out=sys.argv[2]
repo=sys.argv[sys.argv.index("--repo")+1]
D="/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-3.w1/data"
split=json.load(open(D+"/split.json")); train=set(split["train"]); held=set(split["heldout"])
V=248320
tc=np.zeros(V*2,np.int64); hc=np.zeros(V*2,np.int64)
for l in open(D+"/seqs.jsonl"):
    r=json.loads(l)
    if r["id"] in train:
        np.add.at(tc,np.array(r["prompt_ids"]+r["resp_ids"]),1)
    elif r["id"] in held:
        np.add.at(hc,np.array(r["resp_ids"]),1)
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B-FP8")
rc=np.zeros(V*2,np.int64); nt=0
files=glob.glob(repo+"/docs/**/*.md",recursive=True)+glob.glob(repo+"/vllm/**/*.py",recursive=True)
for f in files:
    try: ids=tok(open(f,errors="ignore").read(),add_special_tokens=False)["input_ids"]
    except Exception: continue
    np.add.at(rc,np.array(ids,dtype=np.int64),1); nt+=len(ids)
chosen=[]; seen=set()
def add(order):
    for t in order:
        if len(chosen)>=N: return
        if t not in seen: seen.add(t); chosen.append(int(t))
add([t for t in np.argsort(-tc,kind="stable") if tc[t]>0])
n_train=len(chosen)
add([t for t in np.argsort(-rc,kind="stable") if rc[t]>0])
n_repo=len(chosen)-n_train
add(range(V)); n_fill=len(chosen)-n_train-n_repo
chosen=np.array(sorted(chosen),dtype=np.int32); np.save(out,chosen)
cov=hc[chosen].sum()/hc.sum(); first=hc[:N].sum()/hc.sum()
print(f"N={N} files={len(files)} repo_tokens={nt} train_tokens={tc.sum()} heldout_tokens={hc.sum()}")
print(f"selected: from_train={n_train} from_repo={n_repo} from_id_fill={n_fill}")
print(f"held-out coverage: freq-ranked {cov:.4f} vs first-N-ids {first:.4f}")
