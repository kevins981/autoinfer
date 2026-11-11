import json, numpy as np
R=[json.load(open(f'res_gpu{g}.json'))['res'] for g in (0,1)]
FS=[0.88,0.92,0.96,1.0,1.04,1.08,1.12]; MS=[48,128,256,384,640]
out=[]
for mode in ('cold','warm'):
    out.append(f'\n### {mode} (us per call)\n\n| M | shape | GPU | '+' | '.join(f'f={f}' for f in FS)+' |\n|'+'---|'*(3+len(FS)))
    for M in MS:
        for sh in ('gate_up','down'):
            for g in (0,1):
                out.append(f'| {M} | {sh} | {g} | '+' | '.join(f"{R[g][f'{sh}|{M}|{f}'][mode]:.1f}" for f in FS)+' |')
        for g in (0,1):
            out.append(f'| {M} | **sum** | {g} | '+' | '.join(f"{sum(R[g][f'{sh}|{M}|{f}'][mode] for sh in ('gate_up','down')):.1f}" for f in FS)+' |')
    out.append(f'\n### Balance ({mode}): f0+f1=2, equalize gate_up+down\n\n| M | T0(f=1) us | T1(f=1) us | f0 | f1 | balanced T us | saving vs slower GPU us | % |\n|---|---|---|---|---|---|---|---|')
    for M in MS:
        T=[np.array([sum(R[g][f'{sh}|{M}|{f}'][mode] for sh in ('gate_up','down')) for f in FS]) for g in (0,1)]
        # linear fit (extrapolate) 
        p=[np.polyfit(FS,T[g],1) for g in (0,1)]
        # p0(f0)=p1(2-f0)
        f0=(np.polyval(p[1],2)-p[0][1])/(p[0][0]+p[1][0]); f1=2-f0
        tb=np.polyval(p[0],f0); t0,t1=T[0][3],T[1][3]; slow=max(t0,t1)
        out.append(f'| {M} | {t0:.1f} | {t1:.1f} | {f0:.3f} | {f1:.3f} | {tb:.1f} | {slow-tb:.1f} | {100*(slow-tb)/slow:.1f}% |')
open('tables.md','w').write('\n'.join(out)); print('\n'.join(out))
