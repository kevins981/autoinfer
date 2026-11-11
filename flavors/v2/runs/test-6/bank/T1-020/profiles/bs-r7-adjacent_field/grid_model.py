# Cost model: Marlin (microbench mpad/r1) + host AR (0.112 ms/row: 0.875 us/token/AR x 128, bs-r7-systematic_engineer) per mixed step,
# for prefill CUDA-graph token grids. Sample of mixed-step token counts: T1-016 graph trace (49 steps, M = prefill + 3*decode reqs).
import json, math, csv
res = {int(k): v for k, v in json.load(open('mpad/r1/res.json')).items()}
T64 = {m: res[m]['stock_ms'] for m in res if m % 64 == 0}
def t64(m):
    if m in T64: return T64[m]
    top = max(T64); return T64[top] + (m - top) * 0.18
R = {16: res[80]['stock_ms'] - T64[64], 32: res[96]['stock_ms'] - T64[64]}
R[48] = R[32] + (R[32] - R[16])  # ~extrapolated (M=105 measured 29.7 ->48-row rem ~ 12.6 at 41 rows)
def marlin(m):
    r = m % 64
    if r == 0: return t64(m)
    base = t64(m - r) if m > r else 0
    rr = min(R, key=lambda x: abs(x - r)); return base + (R[rr] if m > 64 else t64(64) * 0)  # m<=64 single launch
def marlin_small(m):  # m<=64: single launch, roughly linear 48->64: 15.3->17.1
    return 15.3 + (m - 48) * (17.1 - 15.3) / 16 if m >= 48 else 15.3
def cost(m):
    mar = marlin_small(m) if m <= 64 else marlin(m)
    return mar + 0.112 * m
rows = list(csv.DictReader(open('../leader-2.w7/prof_stack/graph/steps.csv')))
Ms = [int(r['prefill_tokens']) + 3 * int(r['decode_reqs']) for r in rows if r['kind'] == 'prefill']
Ms = [m for m in Ms if m < 1500]
def walk_grid(mx):
    g = [1, 2, 4] + list(range(8, 65, 8)); low = 64
    while low < 1024:
        g += list(range(low + low // 4, 2 * low + 1, low // 4)); low *= 2
    g += list(range(1280, mx + 1, 256)); return sorted(set(x for x in g if x <= mx))
G0 = walk_grid(2048)
G64 = sorted(set([1, 2, 4] + list(range(8, 65, 8)) + list(range(128, 2049, 64))))
G32 = sorted(set([1, 2, 4] + list(range(8, 65, 8)) + list(range(96, 2049, 32))))
def pad(m, G): return min(g for g in G if g >= m)
print('G0', G0)
for name, G in [('G0 current', G0), ('G64', G64), ('G32', G32), ('exact (no pad)', None)]:
    tot = 0; tm = 0; ta = 0
    for m in Ms:
        mp = m if G is None else pad(m, G)
        tot += cost(mp)
    print(f'{name:16s} mean Marlin+AR per mixed step {tot/len(Ms):6.1f} ms')
for m in Ms[:0]: pass
print('padding waste current grid, mean rows', sum(pad(m, G0) - m for m in Ms) / len(Ms), ' G64', sum(pad(m, G64) - m for m in Ms) / len(Ms))
def ideal_marlin(m, gran=16):
    m = math.ceil(m / gran) * gran
    if m <= 64: return marlin_small(m)
    r = m % 64; return t64(m - r) + 0.18 * r
for name, G, gran in [('G0 + masked-kernel(16)', G0, 16), ('G16 + masked-kernel(16)', sorted(set(list(range(8, 2049, 16)))), 16), ('exact + masked(16)', None, 16)]:
    tot = 0
    for m in Ms:
        mp = m if G is None else pad(m, G)
        tot += ideal_marlin(mp, gran) + 0.112 * mp
    print(f'{name:26s} {tot/len(Ms):6.1f} ms')
import statistics
print('n', len(Ms), 'mean M', statistics.mean(Ms), 'median', statistics.median(Ms))
R[8] = R[16]  # assume
def stock_any(m):
    if m <= 64: return marlin_small(m)
    r = m % 64
    if r == 0: return t64(m)
    rr = min([16, 32, 48], key=lambda x: abs(x - r)); return t64(m - r) + R[rr]
for name, G in [('G0 stock', G0), ('G64 stock', G64), ('G32 stock', G32), ('G16 stock', sorted(set([1,2,4]+list(range(8, 2049, 16))))), ('G8 stock', list(range(8, 2049, 8)))]:
    tot = sum(stock_any(pad(m, G)) + 0.112 * pad(m, G) for m in Ms)
    print(f'{name:12s} {tot/len(Ms):6.1f} ms/step  ngraphs {len(G)}')
# benchmark-mean-like distribution check: scale sample to mean M ~306 by dropping largest? just report median-region
print('--- AR sends only real rows (device-side count), Marlin at padded size')
for name, G in [('G0', G0), ('G64', G64), ('G32', G32)]:
    tot = sum(stock_any(pad(m, G)) + 0.112 * m for m in Ms)
    print(f'{name:6s} {tot/len(Ms):6.1f} ms/step')
print('--- + masked Marlin (16-row granularity)')
for name, G in [('G0', G0), ('G32', G32)]:
    tot = sum(ideal_marlin(pad(m, G)) + 0.112 * m for m in Ms)
    print(f'{name:6s} {tot/len(Ms):6.1f} ms/step')
# also: restrict sample to the 'benchmark-mean' by reweighting? report savings in the M<=400 subset
sub = [m for m in Ms if m <= 450]
for name, f in [('G0 stock', lambda m: stock_any(pad(m, G0)) + 0.112 * pad(m, G0)), ('G32 stock', lambda m: stock_any(pad(m, G32)) + 0.112 * pad(m, G32)),
                ('G32+ARreal', lambda m: stock_any(pad(m, G32)) + 0.112 * m)]:
    print('subset M<=450 n', len(sub), name, round(sum(map(f, sub)) / len(sub), 1))
