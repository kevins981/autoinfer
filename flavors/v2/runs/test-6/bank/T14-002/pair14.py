# usage: pair14.py A1 B1 [B2 A2]  (dirs with steps.json). Pairs mixed steps by prefill tokens (pt) occurrence order, decode tokens as tiebreak.
import json, sys, random, statistics as st
from collections import defaultdict
SC=400/64
def load(n): return json.load(open(f'{n}/steps.json'))
def key_groups(steps):
    g=defaultdict(list)
    for o in steps:
        if o['p']: g[(o['pt'])].append(o)
    for k in g: g[k].sort(key=lambda o:(o['dt'],o['R']))
    return g
def pairs(a,b):
    ga,gb=key_groups(a),key_groups(b); out=[]
    for k in ga:
        if k in gb:
            n=min(len(ga[k]),len(gb[k]))
            out+= list(zip(ga[k][:n],gb[k][:n]))
    return out
def bucket(R): return 'R<256' if R<256 else ('256-600' if R<600 else '>=600')
def boot(vals,n=2000):
    random.seed(1); m=len(vals); r=[]
    for _ in range(n): r.append(sum(random.choice(vals) for _ in range(m)))
    r.sort(); return r[int(.025*n)],r[int(.975*n)]
def cmp(name,a,b):
    P=pairs(a,b)
    na=sum(1 for o in a if o['p']); nb=sum(1 for o in b if o['p'])
    print(f'\n### {name}: mixed steps {na} vs {nb}, paired {len(P)}')
    for metric in ('m_sum','m_un','m_noar','span','period','n','nbig'):
        print(f'-- {metric}')
        for bk in ('R<256','256-600','>=600','all'):
            sel=[(x,y) for x,y in P if bk=='all' or bucket(x['R'])==bk]
            if not sel: continue
            d=[x[metric]-y[metric] for x,y in sel]
            s=sum(d)
            lo,hi=boot(d)
            f=lambda v: f'{v:.1f}'
            print(f'  {bk:8s} n={len(sel):3d} A {sum(x[metric] for x,_ in sel)/len(sel):7.2f} B {sum(y[metric] for _,y in sel)/len(sel):7.2f} mean/step  saved/64p {s:7.1f} -> per 400p trial {s*SC/1000:6.2f} s  95%CI [{lo*SC/1000:.2f}, {hi*SC/1000:.2f}] s' if metric in('m_un','span','period','m_sum') else f'  {bk:8s} n={len(sel):3d} A {sum(x[metric] for x,_ in sel)/len(sel):7.2f} B {sum(y[metric] for _,y in sel)/len(sel):7.2f} mean/step')
def dec(name,s):
    d=[o for o in s if o['p']==0 and o['dt']>=40]
    print(name,'decode steps',len(d),'marlin sum median',round(st.median(o['m_sum'] for o in d),3) if d else None,'launches',st.median(o['n'] for o in d) if d else None,'span',round(st.median(o['span'] for o in d),2) if d else None)
names=sys.argv[1:]; S={n:load(n) for n in names}
for n in names: dec(n,S[n])
for a in names:
    for b in names:
        if a<b and a[0]==b[0]: cmp(f'{a} vs {b} (same arm, noise)',S[a],S[b])
As=[n for n in names if n[0]=='A']; Bs=[n for n in names if n[0]=='B']
for a in As:
    for b in Bs: cmp(f'{a} (champion) minus {b} (policy)',S[a],S[b])
