# Cross-request n-gram drafting sim on T3 self-generated data (CPU only).
import json, numpy as np
from collections import defaultdict
D='/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-3.w1/data/'
S=[json.loads(l) for l in open(D+'seqs.jsonl')]
z=np.load(D+'conf.npz'); sid,pos,match=z['sid'],z['pos'],z['match']
PL={d['id']:len(d['prompt_ids']) for d in S}
M=defaultdict(dict)
for i in range(len(sid)): M[int(sid[i])][int(pos[i])-PL[int(sid[i])]+1]=match[i]
NS=[3,4,6,8]
class Idx:
    def __init__(s): s.stream=[]; s.maps={n:{} for n in NS}
    def add(s,toks):
        for t in toks:
            s.stream.append(t); L=len(s.stream)
            for n in NS:
                if L>=n+1:
                    key=tuple(s.stream[L-n-1:L-1]); s.maps[n][key]=L-1  # position of next token
    def sep(s): s.stream.append(-1)
    def draft(s,ctx,K):
        for n in sorted(NS,reverse=True):
            if len(ctx)<n: continue
            p=s.maps[n].get(tuple(ctx[-n:]))
            if p is not None:
                out=[]
                for q in range(p,min(p+K,len(s.stream))):
                    if s.stream[q]<0: break
                    out.append(s.stream[q])
                if out: return out,n
        return [],0
def mtp_acc(sid_,p,k):
    m=M[sid_].get(p)
    if m is None: return None
    a=0
    for j in range(min(k,4)):
        if m[j]: a+=1
        else: break
    return a
glob=Idx()
res=defaultdict(lambda:[0,0,0]); hits=defaultdict(int)
for d in S:
    k_=d['id']
    if k_ not in M: continue
    prompt=d['prompt_ids']; resp=d['resp_ids']; T=len(resp)
    for pol in ['mtp2','oracle2_x','oracle6_x','x_long_or_mtp2']:
        p=0; tok=1; steps=0; rows=0
        loc=Idx(); loc.add(prompt)
        while p<T-1:
            a_m=mtp_acc(k_,p,2)
            if a_m is None: break
            ctx=prompt+resp[:p+1]
            dr,n=glob.draft(ctx,6)
            dl,nl=loc.draft(ctx,6)
            if nl>n: dr,n=dl,nl
            a_n=0
            for j,t in enumerate(dr):
                if p+1+j<T and resp[p+1+j]==t: a_n+=1
                else: break
            if pol=='mtp2': a=a_m; r=3
            elif pol=='oracle2_x': a=max(a_m,min(a_n,2)); r=3
            elif pol=='oracle6_x': a=max(a_m,a_n); r=3   # free-row upper bound
            else:
                if n>=6 and len(dr)>=3: a=a_n; r=1+len(dr); hits['long']+=1
                else: a=a_m; r=3
            a=max(0,min(a,T-2-p))
            p+=a+1; tok+=a+1; steps+=1; rows+=r
            loc.add([resp[p-a-1+i] for i in range(a+1)] if False else [])
        res[pol][0]+=tok; res[pol][1]+=steps; res[pol][2]+=rows
    glob.add(prompt); glob.sep(); glob.add(resp); glob.sep()
for pol,(t,s,r) in res.items():
    print(f'{pol:16s} E={t/s:.3f} rows/step/seq={r/s:.2f} tokens/row={t/r:.3f}')
print(dict(hits))
