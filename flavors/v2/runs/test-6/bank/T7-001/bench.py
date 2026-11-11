"""T7-001: split-half RS/AG pipeline on copy engines vs stock AR. usage: bench.py <rank> <shm> <smoke 0/1> <Rs comma> """
import os, sys, statistics as st, json
import numpy as np
import torch, torch.distributed as dist
rank = int(sys.argv[1]); SHM = sys.argv[2]; SMOKE = sys.argv[3] == '1'; RS = [int(x) for x in sys.argv[4].split(',')]
NL = 3 if SMOKE else 8
REPS = 2 if SMOKE else 5
NREP = 8 if SMOKE else 25
HERE = os.path.dirname(os.path.abspath(__file__))
torch.cuda.set_device(rank); dev = torch.device('cuda', rank); bf = torch.bfloat16
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', os.environ.get('OVL_PORT', '29577'))
dist.init_process_group('gloo', rank=rank, world_size=2)
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
import vllm.distributed.device_communicators.host_staged_ar as Hm
import vllm.distributed.device_communicators.dma_staged_ar as Dm
from torch.utils.cpp_extension import load
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
hmod = Hm._load_ext(); dmod = Dm._load_ext()
X = load('half_ar', [HERE + '/half_ar.cu'], build_directory=HERE + '/ext/half_ar', extra_ldflags=['-lcuda'], verbose=False)
SLOT = 24 << 20
def init2(m, path, slot):
    if rank == 0: m.init(path, slot, 0, True) if m is not X else m.init(path, slot, 0, True)
    dist.barrier()
    if rank == 1: m.init(path, slot, 1, False)
    dist.barrier()
    if rank == 0: os.unlink(path)
init2(hmod, SHM, SLOT)
# host mod init signature has extra bool arg for create (rank0 True)
init2(dmod, SHM + '_d', SLOT)
init2(X, SHM + '_x', 2 << 20)
def log(*a): print(f'[r{rank}]', *a, flush=True)

def mk(N, K):
    m = torch.nn.Module()
    m.output_size_per_partition = N; m.input_size_per_partition = K; m.weight_block_size = [128, 128]; m.orig_dtype = bf
    m.weight = torch.nn.Parameter(torch.randint(0, 255, (N, K), device=dev, dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn), requires_grad=False)
    m.weight_scale_inv = torch.nn.Parameter(torch.rand(N // 128, K // 128, device=dev) * 0.02 + 0.005, requires_grad=False)
    prepare_fp8_layer_for_marlin(m, size_k_first=False)
    m.N = N; m.K = K
    return m
wsp = marlin_make_workspace_new(dev)
def lin(x, m):
    return apply_fp8_marlin_linear(x, m.weight, m.weight_scale_inv, wsp, m.N, m.K, None)
SH = dict(qkvz=(8192, 5120), gu=(17408, 5120), down=(5120, 8704), out=(5120, 3072))
S_L = [{k: mk(*v) for k, v in SH.items()} for _ in range(NL)]
HS = dict(op=(2560, 3072), oo=(2560, 3072), gA=(17408, 2560), gB=(17408, 2560), dp=(2560, 8704), do=(2560, 8704), qA=(8192, 2560), qB=(8192, 2560))
H_L = [{k: mk(*v) for k, v in HS.items()} for _ in range(NL)]
main = torch.cuda.Stream(); cs = torch.cuda.Stream(); aux = torch.cuda.Stream()
nw = torch.randn(5120, device=dev, dtype=bf) * 0.1 + 1
gh = torch.randn(2560, device=dev, dtype=bf) * 0.1 + 1
def cfg(nbytes):
    return (8, 1) if nbytes <= 600 * 1024 else ((16, 4) if nbytes <= 8 << 20 else (32, 8))

class Ev:
    def __init__(s): s.l = []; s.i = 0
    def get(s):
        if s.i == len(s.l): s.l.append(torch.cuda.Event())
        e = s.l[s.i]; s.i += 1; return e
    def reset(s): s.i = 0
EV = Ev()
def fork(src, dst):
    e = EV.get(); e.record(src); dst.wait_event(e)

def build(arm, R, stampbuf=None):
    M = R
    xa = torch.randn(M, 3072, device=dev, dtype=bf)
    resid = torch.randn(M, 5120, device=dev, dtype=bf)
    r1 = [torch.empty(M, 5120, device=dev, dtype=bf) for _ in range(NL)]
    r2 = [torch.empty(M, 5120, device=dev, dtype=bf) for _ in range(NL)]
    ridx = [torch.randn(M, 2560, device=dev, dtype=bf) for _ in range(2)]
    bufs = {}
    def B(name, *shape, dt=bf):
        if name not in bufs: bufs[name] = torch.empty(*shape, device=dev, dtype=dt)
        return bufs[name]
    nb = M * 2560 * 2
    def stamp(i):
        if stampbuf is not None: X.stamp(stampbuf, i)
    def addnorm(x, rs): ops.fused_add_rms_norm(x, rs, nw, 1e-6)
    def hostar(x, out):
        g, k = cfg(x.numel() * 2); hmod.allreduce(x, out, g, k, 512, 20)
    c0 = 64 * -(-M // 128); c1 = M - c0
    def run():
        EV.reset()
        t = 0
        for i in range(NL):
            L = S_L[i]; nxt = S_L[(i + 1) % NL]; Hh = H_L[i]; Hn = H_L[(i + 1) % NL]
            if arm in ('S', 'N'):
                h1 = lin(xa, L['out'])
                if arm == 'S': hostar(h1, r1[i])
                else: r1[i].copy_(h1)
                addnorm(r1[i], resid)
                gu = lin(r1[i], L['gu'])
                act = B('act', M, 8704); torch.ops._C.silu_and_mul(act, gu)
                h2 = lin(act, L['down'])
                if arm == 'S': hostar(h2, r2[i])
                else: r2[i].copy_(h2)
                addnorm(r2[i], resid)
                q = lin(r2[i], nxt['qkvz'])
                xa.copy_(q[:, :3072])
            elif arm == 'S2':   # champion token-split path: chunk0 DMA AR on main, chunk1 host AR on aux
                fork(main, aux)
                def chunk(a, b, dma):
                    p = lin(xa[a:b], L['out'])
                    if dma: pr = torch.empty_like(p); dmod.allreduce(p, pr, 0, 4, 0)
                    else: pr = torch.empty_like(p); hostar(p, pr)
                    addnorm(pr, resid[a:b])
                    gu = lin(pr, L['gu']); act = torch.empty(b - a, 8704, device=dev, dtype=bf); torch.ops._C.silu_and_mul(act, gu)
                    d = lin(act, L['down'])
                    if dma: dmod.allreduce(d, r2[i][a:b], 0, 4, 0)
                    else: hostar(d, r2[i][a:b])
                chunk(0, c0, True)
                with torch.cuda.stream(aux): chunk(c0, M, False)
                fork(aux, main)
                addnorm(r2[i], resid)
                q = lin(r2[i], nxt['qkvz']); xa.copy_(q[:, :3072])
            else:  # H, HG (no comm), H32 (fp32 carry upper bound), HT same as H with stamps
                comm = arm != 'HG'; f32 = arm == 'H32'
                def boundary(inp, wp, wo, ridx_, cA, cB, tt, sb):
                    # producer: peer half first
                    stamp(sb + 0)
                    pp = lin(inp, wp)
                    stamp(sb + 1)
                    if comm:
                        fork(main, cs)
                        with torch.cuda.stream(cs):
                            X.send(pp, 4 * i + tt * 2, nb); stamp(sb + 2)
                    po = lin(inp, wo)
                    stamp(sb + 3)
                    tmp = B(f'tmp{tt}', M, 2560)
                    if comm:
                        with torch.cuda.stream(cs):
                            X.recv(tmp, 4 * i + tt * 2, nb); stamp(sb + 4)
                            e = EV.get(); e.record(cs)
                        main.wait_event(e)
                    y = B(f'y{tt}', M, 2560)
                    X.rsn(po, tmp if comm else pp, ridx_, gh, y)
                    stamp(sb + 5)
                    yp = B(f'yp{tt}', M, 2560)
                    if comm:
                        fork(main, cs)
                        with torch.cuda.stream(cs):
                            X.send(y, 4 * i + tt * 2 + 1, nb); stamp(sb + 6)
                            X.recv(yp, 4 * i + tt * 2 + 1, nb); stamp(sb + 7)
                            e = EV.get(); e.record(cs)
                    oA = lin(y, cA)
                    stamp(sb + 8)
                    if f32:
                        a32 = B(f'a32{tt}', M, oA.shape[1], dt=torch.float32); X.tof32(oA, a32); oA = a32
                    if comm: main.wait_event(e)
                    stamp(sb + 9)
                    oB = lin(yp if comm else y, cB)
                    stamp(sb + 10)
                    return oA, oB
                oA, oB = boundary(xa, Hh['op'], Hh['oo'], ridx[0], Hh['gA'], Hh['gB'], 0, i * 32)
                act = B('act', M, 8704); X.add_silu_mul(oA, oB, act)
                oA, oB = boundary(act, Hh['dp'], Hh['do'], ridx[1], Hn['qA'], Hn['qB'], 1, i * 32 + 16)
                q = B('q', M, 8192); X.add2(oA, oB, q)
                xa.copy_(q[:, :3072])
    return run, xa

def capture(run):
    with torch.cuda.stream(main):
        for _ in range(2): run()
        main.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=main): run()
    main.synchronize(); return g
def tgraph(g, n):
    ts = []
    with torch.cuda.stream(main):
        for _ in range(3): g.replay()
        main.synchronize()
        for _ in range(n):
            dist.barrier()
            a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(main); g.replay(); b.record(main); b.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts)

# round-trip sanity of send/recv
t0 = torch.full((1024,), float(rank + 1), device=dev, dtype=bf); t1 = torch.zeros(1024, device=dev, dtype=bf)
with torch.cuda.stream(cs):
    X.send(t0, 63, 2048); X.recv(t1, 63, 2048)
torch.cuda.synchronize(); assert (t1 == float(2 - rank)).all(), t1[:4]; log('sendrecv ok')

OUT = {}
for R in RS:
    arms = ['S', 'N', 'H', 'HG', 'H32'] + (['S2'] if (R >= 256 and os.environ.get('WITH_S2')) else [])
    stampbuf = torch.zeros(NL * 32, dtype=torch.int64, device=dev)
    gr = {}; keep = []
    for a in arms:
        run, xa = build(a, R); keep.append((run, xa)); gr[a] = capture(run)
    # stamped H for timeline
    runT, xaT = build('H', R, stampbuf); keep.append((runT, xaT)); gT = capture(runT)
    res = {a: [] for a in arms}; res['HT'] = []
    for rep in range(REPS):
        for a in arms:
            dist.barrier(); res[a].append(tgraph(gr[a], NREP))
        dist.barrier(); res['HT'].append(tgraph(gT, NREP))
    log(f'R={R} us/layer (median over {REPS} reps of {NREP}-replay medians):', {a: round(st.median(v) * 1000 / NL, 1) for a, v in res.items()}, 'raw', {a: [round(x * 1000 / NL, 1) for x in v] for a, v in res.items()})
    # timeline: layer 1 (steady), stamps relative to its first stamp
    torch.cuda.synchronize(); sb = stampbuf.cpu().numpy()
    base = sb[32 * 1]
    names = ['pp_done(L1 end)', 'send1 D2H done', 'po_done', 'recv1 H2D done', 'rsn done', 'send2 D2H done', 'recv2 H2D done', 'oA done', 'oB done']
    tl = {}
    for l in (1, 2):
        o = l * 32
        tl[l] = {k: (int(sb[o + k] - base) / 1000.0 if sb[o + k] else None) for k in range(0, 11)}
    log(f'R={R} TIMELINE us rel. to layer1 start (stamp idx: 0 start,1 prodA done,2 send1 done,3 prodB done,4 recv1 done,5 rsn done,6 send2 done,7 recv2 done,8 consA done,9 pre-consB,10 consB done; +16 for the 2nd boundary)')
    for k in sorted(tl[1]): log(f'   boundary1 stamp{k}: {tl[1][k]}', f'| boundary2 stamp{k}: {tl[1][16+k] if (16+k) in tl[1] else None}' if True else '')
    del gr, gT, keep
    torch.cuda.empty_cache()
