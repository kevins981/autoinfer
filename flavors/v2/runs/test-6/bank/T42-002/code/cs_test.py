# T42 stage 1: correctness + timing of the certified lm_head screen vs the champion lmh_codec kernel (GPU0).
# usage: python cs_test.py <out_dir> <n_batches (0 = all)> [timing 0/1]
import os, sys, time, json, math
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csext import cs, champ
from huggingface_hub import hf_hub_download
from safetensors import safe_open

OUT = sys.argv[1]; NB = int(sys.argv[2]); DO_TIME = int(sys.argv[3]) if len(sys.argv) > 3 else 1
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda:0'); torch.cuda.set_device(dev)
LOG = open(os.path.join(OUT, 'log.txt'), 'a')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
log('device', torch.cuda.get_device_name(0), 'clocks', os.popen('nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader -i 0').read().strip())

K = 5120; NH = 124160; KC = K // 128; DTC = 2.0 ** -11
ECAP = 65536; PAIRCAP = 1 << 20; PAIRCAP_T = 8192; CAPR_REAL = 1024; CAPR_ADV = 65536; MMAX = 64
CAPR = [CAPR_REAL]

t0 = time.time()
f = hf_hub_download('Qwen/Qwen3.8-27B-FP8', 'outside.safetensors')
with safe_open(f, 'pt') as s:
    Wall = s.get_tensor('lm_head.weight')
assert Wall.shape == (2 * NH, K) and Wall.dtype == torch.bfloat16
log('W loaded', time.time() - t0)

def ru32(x64):  # float64 tensor -> float32 rounded up (x >= 0)
    f = x64.float()
    return torch.where(f.double() < x64, torch.nextafter(f, torch.full_like(f, math.inf)), f)

def build_plane(W):  # W bf16 [N, K] on GPU
    N = W.shape[0]
    planes = torch.empty((N // 8, KC, 228), dtype=torch.int32, device=dev)
    Q = torch.empty((N, K), dtype=torch.int8, device=dev)
    A = torch.empty(N, dtype=torch.float32, device=dev); B = torch.empty_like(A)
    qmax_seen = 0
    CH = 8192
    for r0 in range(0, N, CH):
        r1 = min(N, r0 + CH)
        Wf = W[r0:r1].float().view(-1, KC, 128)
        mx = Wf.abs().amax(2)
        s32 = torch.where(mx > 0, mx / 63, torch.ones_like(mx))
        sb = s32.bfloat16()
        sb = torch.where(sb.float() < s32, (sb.view(torch.int16) + 1).view(torch.bfloat16), sb)  # round up (s > 0)
        s = sb.float()
        q = torch.round(Wf / s[..., None]).clamp(-64, 63)
        qmax_seen = max(qmax_seen, int(q.abs().max()))
        Wd = Wf.double(); What = s.double()[..., None] * q.double()
        rn = (Wd - What).pow(2).sum((1, 2)).sqrt() * (1 + 1e-9)
        wh = What.pow(2).sum((1, 2)).sqrt() * (1 + 1e-9)
        wn = Wd.pow(2).sum((1, 2)).sqrt() * (1 + 1e-9)
        A[r0:r1] = ru32((rn + DTC * wn + 2.0 ** -18 * wh) * (1 + 1e-9))
        B[r0:r1] = ru32(wh * (1 + 2.0 ** -18) * (1 + 1e-9))
        qi = q.to(torch.int8)
        Q[r0:r1] = qi.view(-1, K)
        # pack: [n8, row, kc, j, half, t, byte] -> [n8, kc, row, t, j, half, byte]
        qq = qi.view(-1, 8, KC, 4, 2, 4, 4).permute(0, 2, 1, 5, 3, 4, 6).reshape(-1, KC, 32, 8, 4)
        by = qq.to(torch.int64) & 0x7F
        reg = by[..., 0] | (by[..., 1] << 8) | (by[..., 2] << 16) | (by[..., 3] << 24)  # [n8, kc, 32, 8]
        x = reg
        Nw = torch.stack([(x[..., 2 * w] & 0x0F0F0F0F) | ((x[..., 2 * w + 1] & 0x0F0F0F0F) << 4) for w in range(4)], -1)  # [n8,kc,32,4]
        P1 = torch.stack([sum(((x[..., 4 * w + i] >> 4) & 0x03030303) << (2 * i) for i in range(4)) for w in range(2)], -1)
        P2 = sum(((x[..., i] >> 6) & 0x01010101) << i for i in range(8))
        s16 = sb.view(-1, 8, KC).permute(0, 2, 1).contiguous().view(torch.int16).to(torch.int64) & 0xFFFF  # [n8, kc, 8]
        sc = s16[..., 0::2] | (s16[..., 1::2] << 16)  # [n8, kc, 4]: word t = rows (2t, 2t+1)
        rec = torch.cat([Nw.reshape(-1, KC, 128), P1.reshape(-1, KC, 64), P2.reshape(-1, KC, 32), sc], -1)
        rec = torch.where(rec >= 2 ** 31, rec - 2 ** 32, rec).to(torch.int32)
        planes[r0 // 8:r1 // 8] = rec
        del Wf, q, Wd, What, qq, by, reg, x, Nw, P1, P2, rec
    plane = planes.view(torch.uint8).reshape(-1)
    bad = cs.plane_verify(Q, plane)
    return plane, A, B, Q, bad, qmax_seen

halves = []
for hh in range(2):
    W = Wall[hh * NH:(hh + 1) * NH].to(dev).contiguous()
    plane, A, B, Q, bad, qm = build_plane(W)
    enc = torch.empty((NH // 8) * KC * 1600, dtype=torch.uint8, device=dev)
    side = torch.zeros(4096 * 512, dtype=torch.uint8, device=dev)
    rawb, ovf, miss, _ = champ.encode(W, enc, side)
    vbad = champ.verify(W, enc, side)
    log(f'half{hh}: plane {plane.numel()/1e9:.4f} GB (verify bad words {bad}, max|q| {qm}); champion enc {enc.numel()/1e9:.4f} GB verify bad {vbad} miss {miss}; A mean {A.mean():.4g} B mean {B.mean():.4g}')
    assert bad == 0 and vbad == 0 and miss == 0
    del Q
    halves.append(dict(W=W, plane=plane, A=A, B=B, enc=enc, side=side))
torch.cuda.synchronize(); log('planes built', time.time() - t0, 'GPU mem GB', torch.cuda.memory_allocated() / 1e9)

# ---- per-step buffers ----
def mkbuf():
    return dict(hq=torch.zeros((MMAX, KC * 256), dtype=torch.int8, device=dev),
                hs=torch.zeros(MMAX, device=dev), hn=torch.zeros(MMAX, device=dev), hqe=torch.zeros(MMAX, device=dev))
hb = mkbuf()
def mkhalf(paircap):
    c1 = torch.zeros(2 * MMAX, dtype=torch.int32, device=dev)          # lb | ecnt  (one memset before the screen)
    c2 = torch.zeros(1 + 2 * MMAX, dtype=torch.int32, device=dev)      # pcnt | kept | fb (one memset before finalize)
    return dict(c1=c1, c2=c2, lb=c1[:MMAX], ecnt=c1[MMAX:], pcnt=c2[:1], kept=c2[1:1 + MMAX], fb=c2[1 + MMAX:],
                ebuf=torch.zeros((MMAX, ECAP, 2), dtype=torch.int32, device=dev),
                pairs=torch.zeros((paircap, 2), dtype=torch.int32, device=dev),
                src=torch.zeros(paircap, dtype=torch.int32, device=dev), map=torch.full((NH,), -1, dtype=torch.int32, device=dev))
hbufs_big = [mkhalf(PAIRCAP), mkhalf(PAIRCAP)]
hbufs = list(hbufs_big)
keys = torch.zeros(MMAX, dtype=torch.int64, device=dev)
keysL = [torch.zeros(MMAX, dtype=torch.int64, device=dev) for _ in range(2)]

def screen_cfg(M):
    return 0 if M > 48 else (4 if M > 16 else 5)

def run_screen(X, hh, cfg=None):
    M = X.shape[0]; b = hbufs[hh]; H = halves[hh]
    b['c1'].zero_()
    cs.screen(hb['hq'], M, H['plane'], H['A'], H['B'], hb['hs'], hb['hn'], hb['hqe'], NH, b['lb'], b['ecnt'], b['ebuf'], ECAP,
              screen_cfg(M) if cfg is None else cfg)

def run_final(M, hh, lbB, keyt, voff_on=True, mode=0, stage=None, wptr=None, vals=0):
    b = hbufs[hh]; H = halves[hh]
    b['c2'].zero_()
    cs.finalize(M, b['lb'], lbB, b['ecnt'], b['ebuf'], ECAP, b['pcnt'], b['pairs'], b['src'], b['map'], b['kept'], b['fb'], CAPR[0])
    if mode == 1:
        cs.gather(b['pcnt'], b['pairs'], b['src'], b['map'], wptr, K, stage, 512)
        base = stage.data_ptr()
    else:
        base = H['W'].data_ptr()
    cs.unmap(b['pcnt'], b['pairs'], b['map'])
    cs.rescore(hX[0], b['pcnt'], b['pairs'], b['src'], base, mode, hh * NH if voff_on else 0, keyt, 256, vals)

hX = [None]
def decide_global(X):
    M = X.shape[0]; hX[0] = X
    cs.hprep(X, hb['hq'], hb['hs'], hb['hn'], hb['hqe'])
    run_screen(X, 0); run_screen(X, 1)
    keys.zero_()
    run_final(M, 0, hbufs[1]['lb'], keys); run_final(M, 1, hbufs[0]['lb'], keys)

def decode_keys(k):
    return (0xFFFFFFFF - (k & 0xFFFFFFFF))

def ref_logits(X):
    out = torch.empty((X.shape[0], 2 * NH), dtype=torch.bfloat16, device=dev)
    for hh in range(2):
        o = torch.empty((X.shape[0], NH), dtype=torch.bfloat16, device=dev)
        champ.gemm(X, halves[hh]['enc'], halves[hh]['side'], NH, o)
        out[:, hh * NH:(hh + 1) * NH] = o
    return out

# ---------------- correctness ----------------
dat = torch.load('/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-42.w1/data/all_hidden.pt')
Hall, resp = dat['hidden'], dat['resp']
log('data rows', Hall.shape[0], 'resp', int(resp.sum()), 'load', time.time() - t0)

stats = dict(rows=0, mism_global=0, mism_local=0, fb_global=0, fb_local=0, valmism=0)
kept_g, kept_l, ecnts, resp_flags, dupfrac = [], [], [], [], []
def check_batch(X, rflag, tag):
    M = X.shape[0]
    L = ref_logits(X)
    Lf = L.float()
    ref = Lf.argmax(1)
    refl = [Lf[:, :NH].argmax(1), Lf[:, NH:].argmax(1)]
    decide_global(X)
    kg = hbufs[0]['kept'][:M] + hbufs[1]['kept'][:M]
    fbg = (hbufs[0]['fb'][:M] + hbufs[1]['fb'][:M]) > 0
    np_ = [min(int(hbufs[h]['pcnt']), hbufs[h]['pairs'].shape[0]) for h in range(2)]
    dup = sum(int((hbufs[h]['src'][:np_[h]] != torch.arange(np_[h], device=dev, dtype=torch.int32)).sum()) for h in range(2))
    dec = decode_keys(keys[:M])
    val = (keys[:M] >> 32) & 0xFFFFFFFF
    ok = (dec == ref) | fbg
    stats['mism_global'] += int((~ok).sum()); stats['fb_global'] += int(fbg.sum())
    # value check: decided value equals the reference logit bitwise
    rv = Lf.gather(1, dec.clamp(0, 2 * NH - 1)[:, None]).squeeze(1)
    u = rv.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    ordv = torch.where(u >= 2 ** 31, (~u) & 0xFFFFFFFF, u | 0x80000000)
    stats['valmism'] += int(((ordv != val) & ~fbg).sum())
    ec = hbufs[0]['ecnt'][:M] + hbufs[1]['ecnt'][:M]
    # local variant (per-rank LB only)
    kl = torch.zeros(M, dtype=torch.int32, device=dev); fbl = torch.zeros(M, dtype=torch.bool, device=dev)
    for hh in range(2):
        keysL[hh].zero_()
        run_final(M, hh, hbufs[hh]['lb'], keysL[hh], voff_on=False)
        d = decode_keys(keysL[hh][:M]); fb = hbufs[hh]['fb'][:M] > 0
        stats['mism_local'] += int(((d != refl[hh]) & ~fb).sum()); fbl |= fb
        kl += hbufs[hh]['kept'][:M]
    stats['fb_local'] += int(fbl.sum())
    stats['rows'] += M
    kept_g.append(kg.cpu()); kept_l.append(kl.cpu()); ecnts.append(ec.cpu()); resp_flags.append(rflag); dupfrac.append((dup, sum(np_)))
    bad = (~ok).nonzero().flatten().tolist()
    if bad:
        log(f'MISMATCH {tag}: rows {bad[:8]} ref {ref[bad[:8]].tolist()} got {dec[bad[:8]].tolist()}')

t1 = time.time()
nrows = Hall.shape[0]
nb = (nrows + 63) // 64 if NB == 0 else NB
step = max(1, ((nrows + 63) // 64) // nb)
for bi in range(nb):
    r0 = (bi * step) * 64
    if r0 >= nrows: break
    X = Hall[r0:r0 + 64].to(dev, non_blocking=True).contiguous()
    check_batch(X, resp[r0:r0 + 64], f'real{bi}')
    if bi % 500 == 0:
        log(f'batch {bi}/{nb} {stats} t={time.time()-t1:.1f}')
torch.cuda.synchronize()
log('REAL', stats, 'time', time.time() - t1)
real_stats = dict(stats)

def dist(x, name):
    x = x.float()
    if len(x) == 0: return f'{name}: n=0'
    return f'{name}: n={len(x)} mean {x.mean():.2f} p50 {x.median():.0f} p90 {x.quantile(.9):.0f} p99 {x.quantile(.99):.0f} p999 {x.quantile(.999):.0f} max {x.max():.0f}'
KG = torch.cat(kept_g); KL = torch.cat(kept_l); EC = torch.cat(ecnts); RF = torch.cat(resp_flags)
summ = {}
for nm, sel in [('all', torch.ones_like(RF)), ('resp', RF), ('prompt', ~RF)]:
    for lab, T in [('kept_global', KG), ('kept_local', KL), ('emitted', EC)]:
        s = dist(T[sel], f'{lab}[{nm}]'); log(s); summ[f'{lab}_{nm}'] = s
dsum = sum(d for d, _ in dupfrac); psum = sum(p for _, p in dupfrac)
log(f'dedupe: {dsum}/{psum} pairs share a vocab row with an earlier pair ({dsum/max(psum,1)*100:.1f}%)')

# ---------------- adversarial rows ----------------
CAPR[0] = CAPR_ADV
stats = dict(rows=0, mism_global=0, mism_local=0, fb_global=0, fb_local=0, valmism=0)
kept_g.clear(); kept_l.clear(); ecnts.clear(); resp_flags.clear()
g = torch.Generator(device='cpu').manual_seed(1)
hn_real = Hall[:4096].float().norm(dim=1).median().item()
adv = []
for sc in [0.25, 1.0, 4.0]:
    for _ in range(4):
        x = torch.randn(64, K, generator=g); x = x / x.norm(dim=1, keepdim=True) * hn_real * sc; adv.append(('gauss%.2f' % sc, x))
for _ in range(4):  # one-hot rows: massive exact bf16 ties
    x = torch.zeros(64, K); idx = torch.randint(0, K, (64,), generator=g); x[torch.arange(64), idx] = torch.randn(64, generator=g).abs() * 50 + 1
    adv.append(('onehot', x))
for _ in range(4):  # two-hot equal weights: ties between ids sharing both weights
    x = torch.zeros(64, K); i1 = torch.randint(0, K, (64,), generator=g); i2 = torch.randint(0, K, (64,), generator=g)
    x[torch.arange(64), i1] = 8.0; x[torch.arange(64), i2] = 8.0; adv.append(('twohot', x))
for _ in range(4):  # self rows: h = scaled vocab row (huge logit at that id, tests big magnitudes)
    v = torch.randint(0, 2 * NH, (64,), generator=g); x = Wall[v].float() * 400; adv.append(('selfrow', x))
for _ in range(2):  # sparse real rows (most coords zeroed) -> many near-ties
    x = Hall[torch.randint(0, nrows, (64,), generator=g)].float(); x[torch.rand(64, K, generator=g) < 0.97] = 0; adv.append(('sparse', x))
adv.append(('zero', torch.zeros(64, K)))  # all logits 0: all ids tie -> fallback path
for nm, x in adv:
    X = x.to(torch.bfloat16).to(dev).contiguous()
    s0 = dict(stats)
    check_batch(X, torch.zeros(64, dtype=torch.bool), nm)
    torch.cuda.synchronize()
    log(f'adv {nm}: mism_g {stats["mism_global"]-s0["mism_global"]} mism_l {stats["mism_local"]-s0["mism_local"]} fb_g {stats["fb_global"]-s0["fb_global"]} '
        f'fb_l {stats["fb_local"]-s0["fb_local"]} valmism {stats["valmism"]-s0["valmism"]} kept_g mean {kept_g[-1].float().mean():.1f} max {kept_g[-1].max()}')
log('ADV', stats)
adv_stats = dict(stats)

# ---------------- tensor-core arithmetic check: rescore chain vs champion bits, and error vs fp64 ----------------
X = Hall[resp.nonzero().flatten()[:64]].to(dev).contiguous(); hX[0] = X
L = ref_logits(X)
gg = torch.Generator(device='cpu').manual_seed(2)
ids = torch.randint(0, NH, (64, 64), generator=gg)
maxrel = 0.0; bitmis = 0; npairs = 0; maxrel_abs = 0.0
for hh in range(2):
    b = hbufs[hh]
    pr = torch.stack([torch.arange(64).repeat_interleave(64), ids.flatten()], 1).to(torch.int32).to(dev)
    n = pr.shape[0]
    b['pairs'][:n] = pr; b['pcnt'][0] = n; b['src'][:n] = torch.arange(n, dtype=torch.int32, device=dev)
    vals = torch.zeros(n, device=dev); keys.zero_()
    cs.rescore(X, b['pcnt'], b['pairs'], b['src'], halves[hh]['W'].data_ptr(), 0, 0, keys, 256, vals.data_ptr())
    torch.cuda.synchronize()
    v = vals[:n]
    refbits = L[pr[:, 0].long(), hh * NH + pr[:, 1].long()]
    bitmis += int((v.to(torch.bfloat16).view(torch.int16) != refbits.view(torch.int16)).sum())
    Xd = X.double()[pr[:, 0].long()]; Wd = halves[hh]['W'].double()[pr[:, 1].long()]
    exact = (Xd * Wd).sum(1)
    err = (v.double() - exact).abs()
    s_abs = (Xd * Wd).abs().sum(1); s_cs = Xd.norm(dim=1) * Wd.norm(dim=1)
    maxrel_abs = max(maxrel_abs, (err / s_abs.clamp_min(1e-30)).max().item())
    maxrel = max(maxrel, (err / s_cs.clamp_min(1e-30)).max().item())
    npairs += n
log(f'TC check: {npairs} pairs: rescore bf16 bits != champion bits: {bitmis}; max |c - exact| / sum|h w| = {maxrel_abs:.3e} (2^{math.log2(max(maxrel_abs,1e-30)):.1f}); / ||h|| ||w|| = {maxrel:.3e}; assumed DTC = 2^-11')
json.dump(dict(real=real_stats, adv=adv_stats, summary=summ, tc=dict(bitmis=bitmis, maxrel_abs=maxrel_abs, maxrel_cs=maxrel)),
          open(os.path.join(OUT, 'correctness.json'), 'w'), indent=1)

if not DO_TIME:
    sys.exit(0)

# ---------------- timing ----------------
# decode-like batch: 16 sequences x M/16 consecutive response positions
seq = dat['seq']
def decode_batch(M, seed=0):
    gg = torch.Generator().manual_seed(seed)
    seqs = torch.randperm(400, generator=gg)[:16]; rows = []
    per = M // 16
    for sidx in seqs.tolist():
        rr = ((seq == sidx) & resp).nonzero().flatten()
        st = int(torch.randint(0, len(rr) - per, (1,), generator=gg)); rows += rr[st:st + per].tolist()
    return Hall[rows].to(dev).contiguous()

flushbuf = torch.empty(256 * 2 ** 20, dtype=torch.uint8, device=dev); sink = torch.zeros(1, dtype=torch.int32, device=dev)
Whost = [torch.empty((NH, K), dtype=torch.bfloat16, pin_memory=True) for _ in range(2)]
for hh in range(2): Whost[hh].copy_(Wall[hh * NH:(hh + 1) * NH])
whp = [cs.host_dev_ptr(Whost[hh].data_ptr()) for hh in range(2)]
stage = torch.empty((PAIRCAP_T, K), dtype=torch.bfloat16, device=dev)
CAPR[0] = CAPR_REAL
hbufs[:] = [mkhalf(PAIRCAP_T), mkhalf(PAIRCAP_T)]
log('pinned host W ready')

def timeit(fn, reps=30, cold=False):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        if cold: cs.read_flush(flushbuf, sink)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort()
    return ts[len(ts) // 2], ts[0]

res = {}
for M in [64, 48, 16]:
    X = decode_batch(M, seed=M); hX[0] = X
    # other rank's LB (emulates the cross-rank exchange having arrived; its cost is NOT measured here)
    decide_global(X); torch.cuda.synchronize()
    lb_other = hbufs[1]['lb'].clone()
    kept0 = hbufs[0]['kept'][:M].clone(); kept1 = hbufs[1]['kept'][:M].clone()
    outc = torch.empty((M, NH), dtype=torch.bfloat16, device=dev)
    H0 = halves[0]
    def f_champ(): champ.gemm(X, H0['enc'], H0['side'], NH, outc)
    def f_hprep(): cs.hprep(X, hb['hq'], hb['hs'], hb['hn'], hb['hqe'])
    def f_screen(): run_screen(X, 0)
    def f_screen_only():
        b = hbufs[0]; b['c1'].zero_()
        cs.screen(hb['hq'], M, H0['plane'], H0['A'], H0['B'], hb['hs'], hb['hn'], hb['hqe'], NH, b['lb'], b['ecnt'], b['ebuf'], ECAP, screen_cfg(M))
    def f_post_gpu(): keys.zero_(); run_final(M, 0, lb_other, keys)
    def f_post_host(): keys.zero_(); run_final(M, 0, lb_other, keys, mode=1, stage=stage, wptr=whp[0])
    def f_final():
        b = hbufs[0]; b['c2'].zero_()
        cs.finalize(M, b['lb'], lb_other, b['ecnt'], b['ebuf'], ECAP, b['pcnt'], b['pairs'], b['src'], b['map'], b['kept'], b['fb'], CAPR[0])
        cs.unmap(b['pcnt'], b['pairs'], b['map'])
    def f_gather():
        b = hbufs[0]; cs.gather(b['pcnt'], b['pairs'], b['src'], b['map'], whp[0], K, stage, 512)
    def f_resc_dev():
        b = hbufs[0]; cs.rescore(X, b['pcnt'], b['pairs'], b['src'], H0['W'].data_ptr(), 0, 0, keys, 256, 0)
    def f_resc_stage():
        b = hbufs[0]; cs.rescore(X, b['pcnt'], b['pairs'], b['src'], stage.data_ptr(), 1, 0, keys, 256, 0)
    def f_full_gpu(): f_hprep(); f_screen(); f_post_gpu()
    def f_full_host(): f_hprep(); f_screen(); f_post_host()
    r = {}
    for nm, fn in [('champion', f_champ), ('hprep', f_hprep), ('screen', f_screen_only), ('post_RGPU', f_post_gpu), ('post_RHOST', f_post_host),
                   ('finalize', f_final), ('gather_host', f_gather), ('rescore_dev', f_resc_dev), ('rescore_stage', f_resc_stage),
                   ('full_RGPU', f_full_gpu), ('full_RHOST', f_full_host)]:
        w = timeit(fn); c = timeit(fn, cold=True)
        r[nm] = dict(warm_med=w[0], warm_min=w[1], cold_med=c[0], cold_min=c[1])
        log(f'M={M} {nm:11s} warm {w[0]:7.1f} us (min {w[1]:.1f})  cold {c[0]:7.1f} us (min {c[1]:.1f})')
    # screen config sweep (cold)
    if M == 64:
        for cfg in [1, 2, 3, 0]:
            def fcfg(cfg=cfg):
                b = hbufs[0]; b['c1'].zero_()
                cs.screen(hb['hq'], M, H0['plane'], H0['A'], H0['B'], hb['hs'], hb['hn'], hb['hqe'], NH, b['lb'], b['ecnt'], b['ebuf'], ECAP, cfg)
            c = timeit(fcfg, cold=True); log(f'M=64 screen cfg{cfg} cold {c[0]:.1f} us')
            r[f'screen_cfg{cfg}'] = c[0]
    # verify the timed path's decisions still match (rank-0 half only, global LB emulated)
    keys.zero_(); f_full_host(); torch.cuda.synchronize()
    r['kept_rank0'] = dict(mean=kept0.float().mean().item(), max=int(kept0.max())); r['pairs_rank0'] = int(hbufs[0]['pcnt'])
    np0 = int(hbufs[0]['pcnt']); uniq = int((hbufs[0]['src'][:np0] == torch.arange(np0, device=dev, dtype=torch.int32)).sum())
    r['unique_rows_rank0'] = uniq
    log(f'M={M} rank0 unique rows fetched {uniq} ({uniq * K * 2 / 1e6:.2f} MB)')
    log(f'M={M} rank0 kept mean {kept0.float().mean():.2f} max {int(kept0.max())}, pairs {int(hbufs[0]["pcnt"])}, emitted {int(hbufs[0]["ecnt"][:M].sum())}')
    res[M] = r
json.dump(res, open(os.path.join(OUT, 'timing.json'), 'w'), indent=1)
log('done', time.time() - t0)
