"""T9-001: CE two-chunk ping-pong microbench. usage: bench.py <rank> <shm> <smoke 0/1> <Rs> <arms comma>"""
import os, sys, statistics as st, re
rank = int(sys.argv[1]); SHM = sys.argv[2]; SMOKE = sys.argv[3] == '1'; RS = [int(x) for x in sys.argv[4].split(',')]
ARMS = sys.argv[5].split(',') if len(sys.argv) > 5 else ['N', 'S', 'S2', 'J', 'P', 'NP']
# ---- NUMA: before any CUDA / pinned allocation
def cpulist(s):
    r = []
    for p in s.strip().split(','):
        a, _, b = p.partition('-'); r += list(range(int(a), int(b or a) + 1))
    return r
os.sched_setaffinity(0, cpulist(open('/sys/devices/system/node/node0/cpulist').read()))
import ctypes
try:
    ln = ctypes.CDLL('libnuma.so.1'); NUMA = ln.numa_available() >= 0
    if NUMA: ln.numa_set_preferred(0)
except OSError:
    NUMA = False
import torch, torch.distributed as dist
NL = 3 if SMOKE else 8
REPS = 2 if SMOKE else 5
NREP = 6 if SMOKE else 20
HERE = os.path.dirname(os.path.abspath(__file__))
PC = int(os.environ.get('PC', '2'))   # pieces for J/P CE ARs
torch.cuda.set_device(rank); dev = torch.device('cuda', rank); bf = torch.bfloat16
def log(*a): print(f'[r{rank}]', *a, flush=True)
log('affinity', sorted(os.sched_getaffinity(0))[:4], '... n=', len(os.sched_getaffinity(0)), 'libnuma preferred(0):', NUMA)
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', os.environ.get('OVL_PORT', '29581'))
dist.init_process_group('gloo', rank=rank, world_size=2)
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.distributed.device_communicators.host_staged_ar import HostStagedAllReduce
from vllm.distributed.device_communicators.dma_staged_ar import DmaStagedAllReduce
from torch.utils.cpp_extension import load
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
X = load('t9ce', [HERE + '/ce.cu'], build_directory=os.environ.get('CE_BUILD', HERE + '/ext'), extra_ldflags=['-lcuda'], verbose=False)
# the champion's own communicator classes (workspace code), 4 DMA pieces (VLLM_DMA_PIPE default 4)
host = HostStagedAllReduce(dist.group.WORLD, dev)
dma = DmaStagedAllReduce(dist.group.WORLD, dev)
log('host.disabled', host.disabled, 'dma.disabled', dma.disabled, 'dma pieces', dma.pieces)
SLOT = 2 << 20
if rank == 0: X.init(SHM, SLOT, 0, True)
dist.barrier()
if rank == 1: X.init(SHM, SLOT, 1, False)
dist.barrier()
if rank == 0: os.unlink(SHM)
# NUMA proof: pages of every shm mapping by node (from /proc/self/numa_maps)
def numa_proof():
    out = {}
    for ln_ in open('/proc/self/numa_maps'):
        if '/dev/shm' in ln_:
            f = ln_.split(); nm = [x for x in f if re.match(r'N\d+=', x)]
            out[f[0] + ' ' + f[-1] if 'N' not in f[-1] else f[0]] = ' '.join(nm)
    return out
for k, v in numa_proof().items(): log('numa_maps shm', k, v, '(pages per node, 4KB)')
cs_ = [l for l in open('/proc/self/numa_maps') if '/dev/shm' in l]
tot = {}
for l in cs_:
    for m in re.finditer(r'N(\d+)=(\d+)', l): tot[m.group(1)] = tot.get(m.group(1), 0) + int(m.group(2))
log('NUMA PROOF: /dev/shm mapped pages total by node:', tot, f'= {{node: MB}}', {k: round(v * 4 / 1024) for k, v in tot.items()})

def mk(N, K):
    m = torch.nn.Module()
    m.output_size_per_partition = N; m.input_size_per_partition = K; m.weight_block_size = [128, 128]; m.orig_dtype = bf
    m.weight = torch.nn.Parameter(torch.randint(0, 255, (N, K), device=dev, dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn), requires_grad=False)
    m.weight_scale_inv = torch.nn.Parameter(torch.rand(N // 128, K // 128, device=dev) * 0.02 + 0.005, requires_grad=False)
    prepare_fp8_layer_for_marlin(m, size_k_first=False)
    m.N = N; m.K = K
    return m
SH = dict(qkvz=(8192, 5120), gu=(17408, 5120), down=(5120, 8704), out=(5120, 3072))
S_L = [{k: mk(*v) for k, v in SH.items()} for _ in range(NL)]
main = torch.cuda.Stream(); sA = torch.cuda.Stream(); sB = torch.cuda.Stream()
dA = torch.cuda.Stream(); hA = torch.cuda.Stream(); dB = torch.cuda.Stream(); hB = torch.cuda.Stream()
WS = {s: marlin_make_workspace_new(dev) for s in ('main', 'A', 'B')}   # separate Marlin lock workspaces per stream
nw = torch.randn(5120, device=dev, dtype=bf) * 0.1 + 1

# ---- lone copy bandwidth (pinned shm slot, node-0 pages)
def bw():
    for nbytes in (655360, 4 << 20):
        if nbytes > SLOT: x = torch.empty(nbytes // 2, device=dev, dtype=bf)
        else: x = torch.empty(nbytes // 2, device=dev, dtype=bf)
        n = min(nbytes, SLOT)
        for tag, fn in (('D2H', X.d2h_raw), ('H2D', X.h2d_raw)):
            for mode in ('alone(r0 only)', 'both ranks'):
                dist.barrier()
                active = (mode == 'both ranks') or rank == 0
                ts = []
                for _ in range(30):
                    dist.barrier()
                    if active:
                        a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); fn(x, 62, n); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
                if active: log(f'lone {tag} {n} B {mode}: median {st.median(ts)*1000:.1f} us = {n/st.median(ts)/1e6:.1f} GB/s')
# 4MB test must fit slot: SLOT is 2MB -> test 655KB and 2MB
def bw2():
    for n in (655360, 2 << 20):
        x = torch.empty(n // 2, device=dev, dtype=bf)
        for tag, fn in (('D2H', X.d2h_raw), ('H2D', X.h2d_raw)):
            for mode in ('rank0 only', 'both ranks'):
                active = (mode == 'both ranks') or rank == 0
                ts = []
                for _ in range(30):
                    dist.barrier()
                    if active:
                        a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); fn(x, 62, n); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
                if active: log(f'lone {tag} {n} B {mode}: median {st.median(ts)*1000:.1f} us = {n/st.median(ts)/1e6:.1f} GB/s')
bw2()

# ---- numeric check: CE send/recv/add2 AR (as used in J/P, 2 pieces) == the champion's host AR (same fp32 add -> bf16)
def ce_check():
    torch.manual_seed(100 + rank)
    x = torch.randn(192, 5120, device=dev, dtype=bf); tmp_ = torch.empty_like(x); o = torch.empty_like(x)
    nb = x.numel() * 2; half = nb // 2
    with torch.cuda.stream(dA):
        for k in range(2): X.send(x, 60 + k, k * half, half)
    with torch.cuda.stream(hA):
        for k in range(2): X.recvwait(60 + k); X.recvcopy(tmp_, 60 + k, k * half, half)
    torch.cuda.synchronize(); X.add2(x, tmp_, o); ref = host.all_reduce(x.clone()); torch.cuda.synchronize()
    log('CE AR vs host AR bitwise equal:', bool((o.view(torch.int16) == ref.view(torch.int16)).all()), 'dma.all_reduce equal:', bool((dma.all_reduce(x).view(torch.int16) == ref.view(torch.int16)).all()))
ce_check()

class Ev:
    def __init__(s): s.l = []; s.i = 0
    def get(s):
        if s.i == len(s.l): s.l.append(torch.cuda.Event())
        e = s.l[s.i]; s.i += 1; return e
    def reset(s): s.i = 0
EV = Ev()
def rec(s):
    e = EV.get(); e.record(s); return e
def fork(src, *dsts):
    e = rec(src)
    for d in dsts: d.wait_event(e)

def build(arm, R, core_us, stampbuf=None):
    M = R; keep = []
    def K(t): keep.append(t); return t
    c0 = 64 * -(-M // 128); c1 = M - c0
    chunks = [(0, c0), (c0, M)]
    xa = torch.randn(M, 3072, device=dev, dtype=bf)
    resid = torch.randn(M, 5120, device=dev, dtype=bf)
    rq = torch.empty(M, 5120, device=dev, dtype=bf)
    tmp = {c: torch.empty(chunks[c][1] - chunks[c][0], 5120, device=dev, dtype=bf) for c in (0, 1)}
    r1 = torch.empty(M, 5120, device=dev, dtype=bf); r2 = torch.empty(M, 5120, device=dev, dtype=bf)
    def lin(x, m, w): return K(apply_fp8_marlin_linear(x, m.weight, m.weight_scale_inv, WS[w], m.N, m.K, None))
    def addnorm(x, rs): ops.fused_add_rms_norm(x, rs, nw, 1e-6)
    def silu(gu):
        act = K(torch.empty(gu.shape[0], 8704, device=dev, dtype=bf)); torch.ops._C.silu_and_mul(act, gu); return act
    def core(rows):
        if core_us > 0: X.core(int(core_us * 1000 * rows / 128))
    sx = (main, sA, sB)
    # ---- CE AR of chunk c on compute stream s, copies on (d, h); writes out
    def ce_ar(p, out, s, d, h, c, lid, st_idx=None):
        nb = p.numel() * 2
        psz = ((nb + PC * 16 - 1) // (PC * 16)) * 16
        fork(s, d, h)
        for k in range(PC):
            o = k * psz
            if o >= nb: break
            n = min(psz, nb - o); tid = lid * PC + k
            with torch.cuda.stream(d):
                if stampbuf is not None and st_idx is not None and k == 0: X.stamp(stampbuf, st_idx + 0)
                X.send(p, tid, o, n)
                if stampbuf is not None and st_idx is not None and k == PC - 1: X.stamp(stampbuf, st_idx + 1)
            with torch.cuda.stream(h):
                X.recvwait(tid)
                if stampbuf is not None and st_idx is not None and k == 0: X.stamp(stampbuf, st_idx + 2)
                X.recvcopy(tmp[c], tid, o, n)
                if stampbuf is not None and st_idx is not None and k == PC - 1: X.stamp(stampbuf, st_idx + 3)
        s.wait_event(rec(h)); s.wait_event(rec(d))
        X.add2(p, tmp[c], out)
    def lid(i, b, c): return (i * 2 + b) * 2 + c
    def run():
        EV.reset(); keep.clear()
        if arm in ('N', 'S'):
            for i in range(NL):
                L = S_L[i]; nxt = S_L[(i + 1) % NL]
                core(M)
                h1 = lin(xa, L['out'], 'main')
                if arm == 'S': r1.copy_(host.all_reduce(h1))
                else: r1.copy_(h1)
                addnorm(r1, resid)
                act = silu(lin(r1, L['gu'], 'main'))
                h2 = lin(act, L['down'], 'main')
                if arm == 'S': r2.copy_(host.all_reduce(h2))
                else: r2.copy_(h2)
                addnorm(r2, resid)
                q = lin(r2, nxt['qkvz'], 'main'); xa.copy_(q[:, :3072])
        elif arm in ('S2', 'J'):
            for i in range(NL):
                L = S_L[i]; nxt = S_L[(i + 1) % NL]
                core(M)
                fork(main, sA, sB)
                def chunk(c, s, w, kind):
                    a, b = chunks[c]
                    with torch.cuda.stream(s):
                        p = lin(xa[a:b], L['out'], w)
                        if kind == 'dma': p = K(dma.all_reduce(p))      # champion: DMA lane, 4 pieces
                        elif kind == 'host': p = K(host.all_reduce(p))
                        else:
                            o = K(torch.empty_like(p)); ce_ar(p, o, s, dA if c == 0 else dB, hA if c == 0 else hB, c, lid(i, 0, c)); p = o
                        addnorm(p, resid[a:b])
                        act = silu(lin(p, L['gu'], w))
                        d = lin(act, L['down'], w)
                        if kind == 'dma': dma.all_reduce(d, r2[a:b])
                        elif kind == 'host': r2[a:b].copy_(host.all_reduce(d))
                        else:
                            o = K(torch.empty_like(d)); ce_ar(d, o, s, dA if c == 0 else dB, hA if c == 0 else hB, c, lid(i, 1, c)); r2[a:b].copy_(o)
                if arm == 'S2':
                    # champion _qwen3_5_split_chain: chunk0 on the current stream with DMA AR, chunk1 on aux with host AR
                    chunk(0, main if False else sA, 'A', 'dma'); chunk(1, sB, 'B', 'host')
                else:
                    chunk(0, sA, 'A', 'ce'); chunk(1, sB, 'B', 'ce')
                fork(sA, main); fork(sB, main)
                addnorm(r2, resid)
                q = lin(r2, nxt['qkvz'], 'main'); xa.copy_(q[:, :3072])
        else:  # P / NP / PT (stamped P)
            comm = arm != 'NP'
            xs = [xa[a:b] for a, b in chunks]; rs = [resid[a:b] for a, b in chunks]
            fork(main, sA, sB)
            S = [sA, sB]; W = ['A', 'B']; D = [dA, dB]; H = [hA, hB]
            evcore = [None] * NL
            for i in range(NL):
                L = S_L[i]; nxt = S_L[(i + 1) % NL]
                # issue order: A's whole layer then B's: B interleaves on its own stream
                for c in (0, 1):
                    s = S[c]; w = W[c]; rows = chunks[c][1] - chunks[c][0]
                    with torch.cuda.stream(s):
                        if c == 1: s.wait_event(evcore[i])
                        core(rows)
                        if c == 0: evcore[i] = rec(s)
                        p = lin(xs[c], L['out'], w)
                        if comm:
                            o = K(torch.empty_like(p)); ce_ar(p, o, s, D[c], H[c], c, lid(i, 0, c), (i * 4 + 0 * 2 + c) * 4 if (arm == 'PT') else None); p = o
                        addnorm(p, rs[c])
                        act = silu(lin(p, L['gu'], w))
                        d = lin(act, L['down'], w)
                        if comm:
                            o = K(torch.empty_like(d)); ce_ar(d, o, s, D[c], H[c], c, lid(i, 1, c), (i * 4 + 1 * 2 + c) * 4 if (arm == 'PT') else None); d = o
                        addnorm(d, rs[c])
                        q = lin(d, nxt['qkvz'], w); xs[c].copy_(q[:, :3072])
            fork(sA, main); fork(sB, main)
    return run, (xa, resid, rq, tmp, r1, r2, keep)

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
        for _ in range(2): dist.barrier(); g.replay()
        main.synchronize()
        for _ in range(n):
            dist.barrier()
            a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(main); g.replay(); b.record(main); b.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts)

for R in RS:
    for core_total in (0, 90):
        core_us = core_total   # per 128 rows; core(rows) scales by rows/128
        arms = [a for a in ARMS if not (a == 'S2' and R < 256)]
        gr = {}; keep = []
        for a in arms:
            if a == 'P' and False: pass
            run, kp = build(a, R, core_us); keep.append((run, kp)); gr[a] = capture(run)
        stampbuf = torch.zeros(NL * 16, dtype=torch.int64, device=dev)
        if 'P' in arms and core_total == 90 and not SMOKE:
            runT, kpT = build('PT', R, core_us, stampbuf); keep.append((runT, kpT)); gT = capture(runT)
        res = {a: [] for a in arms}
        for rep in range(REPS):
            for a in arms:
                dist.barrier(); res[a].append(tgraph(gr[a], NREP))
        log(f'RESULT R={R} core={core_total}us/128rows us/layer (median of {REPS} reps x {NREP} replays):', {a: round(st.median(v) * 1000 / NL, 1) for a, v in res.items()}, 'raw', {a: [round(x * 1000 / NL, 1) for x in v] for a, v in res.items()})
        if 'gT' in dir() and core_total == 90 and not SMOKE and 'P' in arms:
            for _ in range(3): dist.barrier(); gT.replay()
            torch.cuda.synchronize(); dist.barrier(); gT.replay(); torch.cuda.synchronize()
            sb = stampbuf.cpu().numpy()
            # layer 1 stamps; per (b,c): idx = (1*4+b*2+c)*4 + [d2h_first_start, d2h_last_end, h2d_first_start(after wait), h2d_last_end]
            base = None
            for b in (0, 1):
                for c in (0, 1):
                    o = (1 * 4 + b * 2 + c) * 4
                    rows = (64 * -(-R // 128)) if c == 0 else R - 64 * -(-R // 128)
                    nbytes = rows * 5120 * 2
                    t = [int(x) for x in sb[o:o + 4]]
                    if base is None: base = t[0]
                    d2h = (t[1] - t[0]) / 1000; h2d = (t[3] - t[2]) / 1000
                    log(f'COPYRATE R={R} layer1 boundary{b} chunk{"AB"[c]} {nbytes} B: D2H span {d2h:.1f} us = {nbytes/d2h/1e3:.1f} GB/s | H2D (after peer doorbell) span {h2d:.1f} us = {nbytes/h2d/1e3:.1f} GB/s | t0 {(t[0]-base)/1000:.1f} d2hEnd {(t[1]-base)/1000:.1f} h2dStart {(t[2]-base)/1000:.1f} h2dEnd {(t[3]-base)/1000:.1f}')
        del gr, keep
        if 'gT' in dir(): del gT
        torch.cuda.empty_cache()
log('done')
