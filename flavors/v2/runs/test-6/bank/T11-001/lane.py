"""T11-001 lane microbench (based on leader-4.w1 ovl.py / T4-001). usage: lane.py <rank> <shm> [smoke]"""
import os, sys, re, json
def _cpulist(s):
    r = []
    for p in s.strip().split(','):
        a, _, b = p.partition('-'); r += list(range(int(a), int(b or a) + 1))
    return r
os.sched_setaffinity(0, _cpulist(open('/sys/devices/system/node/node0/cpulist').read()))
import ctypes
try:
    _ln = ctypes.CDLL('libnuma.so.1'); NUMA = _ln.numa_available() >= 0
    if NUMA: _ln.numa_set_preferred(0)
except OSError:
    NUMA = False
import os, sys, time, types, statistics as st
import numpy as np
import torch, torch.distributed as dist
rank = int(sys.argv[1]); SHM = sys.argv[2]; SMOKE = len(sys.argv) > 3
NL = 4 if SMOKE else 16
REPS = 2 if SMOKE else 6
NREP = 10 if SMOKE else 30
M = 48
torch.cuda.set_device(rank)
dev = torch.device('cuda', rank)
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', os.environ.get('OVL_PORT', '29533'))
dist.init_process_group('gloo', rank=rank, world_size=2)
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
import vllm.distributed.device_communicators.host_staged_ar as H
mod = H._load_ext()
SLOT = 24 << 20
if rank == 0: mod.init(SHM, SLOT, 0, True)
dist.barrier()
if rank == 1: mod.init(SHM, SLOT, 1, False)
dist.barrier()
if rank == 0: os.unlink(SHM)
def numa_proof():
    tot = {}
    for l in open('/proc/self/numa_maps'):
        if '/dev/shm' in l:
            for m in re.finditer(r'N(\d+)=(\d+)', l): tot[m.group(1)] = tot.get(m.group(1), 0) + int(m.group(2))
    return {k: f'{round(v*4/1024)}MB' for k, v in tot.items()}
print(f'[r{rank}] affinity n={len(os.sched_getaffinity(0))} libnuma={NUMA} NUMA PROOF /dev/shm pages by node:', numa_proof(), flush=True)
TRCAP = 1 << 18
mod.trace_enable(TRCAP)
def log(*a):
    print(f'[r{rank}]', *a, flush=True)

def mk(N, K):
    m = torch.nn.Module()
    m.output_size_per_partition = N; m.input_size_per_partition = K; m.weight_block_size = [128, 128]; m.orig_dtype = torch.bfloat16
    m.weight = torch.nn.Parameter(torch.randint(0, 255, (N, K), device=dev, dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn), requires_grad=False)
    m.weight_scale_inv = torch.nn.Parameter(torch.rand(N // 128, K // 128, device=dev) * 0.02 + 0.005, requires_grad=False)
    prepare_fp8_layer_for_marlin(m, size_k_first=False)
    return m
SH = dict(qkvz=(8192, 5120), gu=(17408, 5120), down=(5120, 8704), out=(5120, 3072))
layers = [{k: mk(*v) for k, v in SH.items()} for _ in range(NL)]
wsp = marlin_make_workspace_new(dev)
def lin(x, name, L, sms=None):
    N, K = SH[name]
    if sms is None: os.environ.pop('MARLIN_SMS', None)
    else: os.environ['MARLIN_SMS'] = str(sms)
    r = apply_fp8_marlin_linear(x, L[name].weight, L[name].weight_scale_inv, wsp, N, K, None)
    os.environ.pop('MARLIN_SMS', None)
    return r
bf = torch.bfloat16
xa = torch.randn(M, 3072, device=dev, dtype=bf)
resid = torch.randn(M, 5120, device=dev, dtype=bf)
nw = torch.randn(5120, device=dev, dtype=bf) * 0.1 + 1
r1 = [torch.randn(M, 5120, device=dev, dtype=bf) for _ in range(NL)]
r2 = [torch.randn(M, 5120, device=dev, dtype=bf) for _ in range(NL)]
r2[-1] = r2[-1]
main = torch.cuda.Stream(priority=-1)
lane_s = torch.cuda.Stream()
evs = [torch.cuda.Event() for _ in range(2 * NL * 2)]
class Cnt: n = 0   # eager + replayed AR launches (epoch = 1 + n)
STEP = 2; PFMAX = 88 << 20
def ar_pf(x, out, pf, pf2=None, grid=8, K=1, mode=20, pfb=8):
    if pf is None: mod.allreduce(x, out, grid, K, 512, mode)
    else: mod.allreduce_pf(x, out, grid, K, 512, mode, pf, PFMAX, pfb, STEP, pf2)
def addnorm(x): ops.fused_add_rms_norm(x, resid, nw, 1e-6)

NEV = 8 * NL + 8
evs = [torch.cuda.Event() for _ in range(NEV)]
wsp2 = marlin_make_workspace_new(dev)
xl = torch.randn(256, 5120, device=dev, dtype=bf)
def mk_pool(N, nbytes=400 << 20):
    P = max(32, -(-nbytes // (5120 * N)))
    return [mk(N, 5120) for _ in range(P)]
def lane_lin(pool, w, S):
    m = pool[w % len(pool)]
    if S is None: os.environ.pop('MARLIN_SMS', None)
    else: os.environ['MARLIN_SMS'] = str(S)
    r = apply_fp8_marlin_linear(xl, m.weight, m.weight_scale_inv, wsp2, m.output_size_per_partition, 5120, None)
    os.environ.pop('MARLIN_SMS', None)
    return r

def build(cfg):
    """cfg: ar='pf'|'plain'|'none'; lane=None|(pool,S,joined)"""
    ar = cfg['ar']; lane = cfg.get('lane')
    mode = 20
    def run():
        e = [0]; w = [0]; last = [None]
        def pre():      # fork lane at the AR producer
            if lane is None: return None
            pool, S, joined = lane
            ev = evs[e[0]]; e[0] += 1; ev.record(main); lane_s.wait_event(ev)
            with torch.cuda.stream(lane_s): lane_lin(pool, w[0], S)
            w[0] += 1
            ev2 = evs[e[0]]; e[0] += 1; ev2.record(lane_s); last[0] = ev2
            return ev2
        def post(ev2):
            if lane is not None and lane[2]: main.wait_event(ev2)
        for i, L in enumerate(layers):
            nxt = layers[(i + 1) % NL]
            h1 = lin(xa, 'out', L)
            ev2 = pre()
            if ar == 'pf': ar_pf(h1, r1[i], L['gu'].weight, mode=mode)
            elif ar == 'plain': ar_pf(h1, r1[i], None, mode=mode)
            else: r1[i].copy_(h1)
            post(ev2)
            addnorm(r1[i])
            gu = lin(r1[i], 'gu', L)
            act = torch.empty(M, 8704, device=dev, dtype=bf)
            torch.ops._C.silu_and_mul(act, gu)
            h2 = lin(act, 'down', L)
            ev2 = pre()
            if ar == 'pf': ar_pf(h2, r2[i], nxt['qkvz'].weight, nxt['out'].weight, mode=mode)
            elif ar == 'plain': ar_pf(h2, r2[i], None, mode=mode)
            else: r2[i].copy_(h2)
            post(ev2)
            addnorm(r2[i])
            q = lin(r2[i], 'qkvz', nxt)
            xa.copy_(q[:, :3072])
        if last[0] is not None: main.wait_event(last[0])   # join forks (unjoined arm: chain tail)
    run.nar = 0 if ar == 'none' else 2 * NL
    return run

def capture(run):
    with torch.cuda.stream(main):
        for _ in range(2): run()
        main.synchronize()
        Cnt.n += 2 * run.nar
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=main): run()
    main.synchronize(); g.nar = run.nar; return g

def time_graph(g, n=NREP):
    ts = []
    with torch.cuda.stream(main):
        for _ in range(3): g.replay()
        main.synchronize(); Cnt.n += 3 * g.nar
        for _ in range(n):
            dist.barrier()
            a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(main); g.replay(); b.record(main); b.synchronize(); ts.append(a.elapsed_time(b)); Cnt.n += g.nar
    return st.median(ts), ts

# ---- lane GEMM calibration: standalone, cold (pool cycles >=400MB), graph of P back-to-back GEMMs
NCAND = [256, 768, 1536] if SMOKE else [256, 512, 768, 1024, 1536]
SS = [112] if SMOKE else [112, 96]
pools = {}
cal = {}
for N in NCAND:
    pools[N] = mk_pool(N)
    for S in [None] + SS:
        P = len(pools[N])
        def run(N=N, S=S, P=P):
            for j in range(P): lane_lin(pools[N], j, S)
        g = capture_c = None
        with torch.cuda.stream(main):
            run(); main.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=main): run()
        main.synchronize()
        ts = []
        for _ in range(3): g.replay()
        torch.cuda.synchronize()
        for _ in range(10):
            a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
        cal[(N, S)] = st.median(ts) / P * 1000
        del g
    log(f'calib N={N} pool={len(pools[N])}: full-grid {cal[(N,None)]:.1f} us ' + ' '.join(f'S={S}: {cal[(N,S)]:.1f}us' for S in SS) + f'  TFLOP/s(full)={2*256*5120*N/cal[(N,None)]/1e6:.1f}')
TT = [25] if SMOKE else [15, 25, 35]
pickN = {T: min(NCAND, key=lambda N: abs(cal[(N, None)] - T)) for T in TT}
log('picked N for T:', pickN, {T: round(cal[(pickN[T], None)], 1) for T in TT})
for N in NCAND:
    if N not in pickN.values(): pools[N] = None
torch.cuda.empty_cache()

CFGS = {'A_pf': dict(ar='pf'), 'A_nopf': dict(ar='plain'), 'N_full': dict(ar='none')}
for T, N in pickN.items():
    for S in SS:
        CFGS[f'L_T{T}_S{S}'] = dict(ar='pf', lane=(pools[N], S, True), T=T, N=N, S=S)
    CFGS[f'U_T{T}_S{SS[0]}'] = dict(ar='pf', lane=(pools[N], SS[0], False), T=T, N=N, S=SS[0])
    CFGS[f'Lnp_T{T}_S{SS[0]}'] = dict(ar='plain', lane=(pools[N], SS[0], True), T=T, N=N, S=SS[0])
graphs = {k: capture(build(v)) for k, v in CFGS.items()}
log('captured', list(graphs))
results = {}; tr_rows = {}
for rep in range(REPS):
    for k, g in graphs.items():
        dist.barrier(); e0 = 1 + Cnt.n
        t, ts = time_graph(g); e1 = 1 + Cnt.n
        results.setdefault(k, []).append(t)
        tr_rows.setdefault(k, []).append((e0 + 3 * g.nar, e1))
    log('rep', rep, {k: round(v[-1], 3) for k, v in results.items()})
torch.cuda.synchronize()
TR = mod.trace_dump().numpy()
def trstats(rows):
    tot, wr = [], []
    for (a, b) in rows:
        T_ = TR[a:b]; T_ = T_[(T_[:, 1] > 0) & (T_[:, 6] >= T_[:, 1])]
        tot += list((T_[:, 6] - T_[:, 1]) / 1000); wr += list((T_[:, 2] - T_[:, 1]) / 1000)
    return dict(total_us=round(float(np.median(tot)), 1) if tot else None, n=len(tot))
log('===== RESULTS ms/%d-layer chain (median of %d replays; %d reps)' % (NL, NREP, REPS))
Am = st.median(results['A_pf'])
for k, v in results.items():
    m = st.median(v); c = CFGS[k]
    extra = ''
    if 'lane' in c:
        wk = 2 * NL * cal[(c['N'], None)] / 1000
        extra = f" lane_work(standalone fullgrid)={wk:.3f}ms  dChain={m-Am:+.3f}ms ({(m-Am)/Am*100:+.1f}%)  net={wk-(m-Am):+.3f}ms"
    log(f'{k:16s} med {m:.3f} min {min(v):.3f} max {max(v):.3f} reps {[round(x,3) for x in v]} trace {trstats(tr_rows[k]) if c["ar"]!="none" else ""}{extra}')

# ---- profiler pass: per-kernel durations & gaps
from torch.profiler import profile, ProfilerActivity
def classify(n):
    l = n.lower()
    if 'marlin' in l: return 'marlin'
    if 'norm' in l: return 'norm'
    if 'silu' in l: return 'silu'
    if 'memcpy' in l or 'copy' in l or 'elementwise' in l: return 'copy'
    return 'AR:' + n[:40]
prof_out = {}
HERE_ = os.path.dirname(os.path.abspath(__file__))
for k, g in graphs.items():
    if CFGS[k]['ar'] == 'none': continue
    dist.barrier()
    with torch.cuda.stream(main):
        for _ in range(2): g.replay(); Cnt.n += g.nar
        main.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            g.replay(); Cnt.n += g.nar; main.synchronize()
    fn = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'prof_r{rank}_{k}.json')
    pr.export_chrome_trace(fn)
    ev = [x for x in json.load(open(fn))['traceEvents'] if x.get('cat') == 'kernel']
    os.unlink(fn)
    kern = [(x['ts'], x['dur'], str(x['args'].get('grid')), x['name'], x['args'].get('stream')) for x in ev]
    kern.sort()
    json.dump(kern, open(os.path.join(HERE_, f'rawprof_r{rank}_{k}.json'), 'w'))
    try:
        S_ = CFGS[k].get('S')
        isl = lambda x: ('marlin' in x[3].lower()) and S_ is not None and x[2].startswith('[%d,' % S_)
        lane_k = [x for x in kern if isl(x)]
        mar = [x for x in kern if 'marlin' in x[3].lower() and not isl(x)]
        ars = [x for x in kern if classify(x[3]).startswith('AR')]
        roles = {}
        for j, x in enumerate(mar): roles.setdefault(['out', 'gu', 'down', 'qkvz'][j % 4], []).append(x[1])
        gaps = []
        for a_ in ars:
            end = a_[0] + a_[1]
            nxt_m = [x for x in mar if x[0] >= end - 1]
            if nxt_m: gaps.append(nxt_m[0][0] - end)
        md = lambda v: round(float(np.median(v)), 1) if len(v) else None
        prof_out[k] = dict(n_ar=len(ars), AR_us=md([a_[1] for a_ in ars]), out=md(roles.get('out', [])), gu=md(roles.get('gu', [])),
            down=md(roles.get('down', [])), qkvz=md(roles.get('qkvz', [])), gap_AR_end_to_consumer_start_us=md(gaps),
            gap_p90=round(float(np.percentile(gaps, 90)), 1) if gaps else None,
            lane_n=len(lane_k), lane_us=md([x[1] for x in lane_k]), lane_sum_ms=round(sum(x[1] for x in lane_k) / 1000, 3),
            main_marlin_grids=sorted(set(x[2] for x in mar))[:3])
        log('PROF', k, prof_out[k])
    except Exception as ex:
        log('PROF parse failed', k, repr(ex))
torch.cuda.synchronize()
if rank == 0 or True: log('done')
