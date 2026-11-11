"""T20-001 tagged fence-free AR microbench (adapted from T16-001 bench). usage: bench.py <rank> <smoke> [arms]
stages: check (bitwise vs hostar), standalone latency (graph of 64 ARs), chain (16 layers, 32 ARs), timeline."""
import os, sys, statistics as st
rank = int(sys.argv[1]); SMOKE = sys.argv[2] == '1'
DEF = ['N','A_pf','T_15','Tpf_15','Tpf_8','Tpf_4']
ARMS = sys.argv[3].split(',') if len(sys.argv) > 3 else DEF
BO = int(os.environ.get('BO', '0'))
def cpulist(s):
    r = []
    for p in s.strip().split(','):
        a, _, b = p.partition('-'); r += list(range(int(a), int(b or a) + 1))
    return r
os.sched_setaffinity(0, cpulist(open('/sys/devices/system/node/node0/cpulist').read()))
import ctypes
ln = ctypes.CDLL('libnuma.so.1'); ln.numa_set_preferred(0)
import re, torch, torch.distributed as dist
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
NL = 4 if SMOKE else 16; REPS = 2 if SMOKE else 6; NREP = 8 if SMOKE else 30; M = 48
torch.cuda.set_device(rank); dev = torch.device('cuda', rank); bf = torch.bfloat16
def log(*a): print(f'[r{rank}]', *a, flush=True)
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29717')
dist.init_process_group('gloo', rank=rank, world_size=2)
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.distributed.device_communicators.host_staged_ar import HostStagedAllReduce
import build
HY = build.get()
host = HostStagedAllReduce(dist.group.WORLD, dev)
path = ['/dev/shm/t20tag_%s' % os.environ['MASTER_PORT']]
if rank == 0: HY.init(path[0], 2 << 20, 0, True)
dist.barrier()
if rank == 1: HY.init(path[0], 2 << 20, 1, False)
dist.barrier()
if rank == 0: os.unlink(path[0])
def numa():
    tot = {}
    for l in open('/proc/self/numa_maps'):
        if '/dev/shm' in l:
            for m in re.finditer(r'N(\d+)=(\d+)', l): tot[m.group(1)] = tot.get(m.group(1), 0) + int(m.group(2))
    return tot
log('affinity n=', len(os.sched_getaffinity(0)), 'host.disabled', host.disabled)
log('NUMA PROOF: /dev/shm pages by node (4KB pages):', numa())
state = {'idx': 0}
PFMAX = 88 << 20
main = torch.cuda.Stream(); side = torch.cuda.Stream()
evs = [torch.cuda.Event() for _ in range(4 * NL + 4)]
def parse(arm):
    if arm in ('N', 'A_pf', 'A_nopf'): return dict(kind=arm)
    m = re.match(r'T(pf)?_(\d+)$', arm); assert m, arm
    return dict(kind='T', pf=bool(m.group(1)), G=int(m.group(2)))
def ar_fn(arm, mode=0, delay_us=0, delay_rank=0):
    c = parse(arm)
    def ar(h, pfw, pfw2, e):
        if c['kind'] == 'N': return h.clone()
        if c['kind'] == 'A_pf': return host.all_reduce(h, pfw, pfw2)
        if c['kind'] == 'A_nopf': return host.all_reduce(h)
        out = torch.empty_like(h)
        usepf = c['pf'] and pfw is not None
        HY.allreduce(h, out, c['G'], mode, delay_us, delay_rank, BO, pfw if usepf else None, PFMAX, 8, 2, pfw2 if usepf else None)
        return out
    return ar
def gen(it, n=M):
    g = torch.Generator(device=dev); g.manual_seed(1000 * rank + it)
    scale = [1.0, 0.01, 30.0][it % 3]
    x = (torch.randn(n, 5120, device=dev, dtype=bf, generator=g) * scale)
    if it % 7 == 0: x[:, ::5] = 0
    if it % 11 == 0: x[0, 0] = float('inf') if rank == 0 else 1.0
    if it % 13 == 0: x = torch.randint(-32768, 32767, (n, 5120), device=dev, dtype=torch.int16, generator=g).view(bf)  # overflow-heavy / NaN / Inf patterns
    return x
# ---------------- stage 1: bitwise check vs hostar (eager, fresh data each iter)
def check():
    for arm in [a for a in ARMS if a.startswith('T')]:
        ar = ar_fn(arm); bad = 0; n = 40 if SMOKE else 300
        for it in range(n):
            x = gen(it)
            ref = host.all_reduce(x.clone())
            out = ar(x, None, None, 0)
            torch.cuda.synchronize()
            bad += (ref.view(torch.int16) != out.view(torch.int16)).sum().item() > 0
        log(f'CHECK {arm}: {bad}/{n} iterations differ bitwise vs hostar'); 
def neg():
    """negative control: delay producer BEFORE its writes (rank 1 then rank 0), tag check on vs forced off."""
    n = 20 if SMOKE else 60
    for dr in (1, 0):
        for mode, name in ((0, 'tag ON '), (1, 'tag OFF')):
            ar = ar_fn('Tpf_15', mode, 150, dr); bad = 0
            for it in range(n):
                x = gen(500 + it)
                ref = host.all_reduce(x.clone())
                out = ar(x, None, None, 0)
                torch.cuda.synchronize()
                bad += (ref.view(torch.int16) != out.view(torch.int16)).sum().item() > 0
                dist.barrier()
            log(f'NEG delay producer rank {dr} by 150us, {name}: {bad}/{n} iterations differ vs hostar')
# ---------------- stage 3 model
def mk(N, K):
    m = torch.nn.Module()
    m.output_size_per_partition = N; m.input_size_per_partition = K; m.weight_block_size = [128, 128]; m.orig_dtype = bf
    m.weight = torch.nn.Parameter(torch.randint(0, 255, (N, K), device=dev, dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn), requires_grad=False)
    m.weight_scale_inv = torch.nn.Parameter(torch.rand(N // 128, K // 128, device=dev) * 0.02 + 0.005, requires_grad=False)
    prepare_fp8_layer_for_marlin(m, size_k_first=False)
    return m
SH = dict(qkvz=(8192, 5120), gu=(17408, 5120), down=(5120, 8704), out=(5120, 3072))
layers = [{k: mk(*v) for k, v in SH.items()} for _ in range(NL)]
wsp = marlin_make_workspace_new(dev, 4)
def lin(x, name, L):
    N, K = SH[name]; return apply_fp8_marlin_linear(x, L[name].weight, L[name].weight_scale_inv, wsp, N, K, None)
xa = torch.randn(M, 3072, device=dev, dtype=bf); resid = torch.randn(M, 5120, device=dev, dtype=bf)
nw = torch.randn(5120, device=dev, dtype=bf) * 0.1 + 1
def addnorm(x): ops.fused_add_rms_norm(x, resid, nw, 1e-6)
def build_chain(arm):
    ar = ar_fn(arm)
    def run():
        state['idx'] = 0; e = 0
        for i, L in enumerate(layers):
            nxt = layers[(i + 1) % NL]
            h1 = lin(xa, 'out', L); r1 = ar(h1, L['gu'].weight, None, e); e += 2
            addnorm(r1)
            gu = lin(r1, 'gu', L)
            act = torch.empty(M, 8704, device=dev, dtype=bf); torch.ops._C.silu_and_mul(act, gu)
            h2 = lin(act, 'down', L); r2 = ar(h2, nxt['qkvz'].weight, nxt['out'].weight, e); e += 2
            addnorm(r2)
            q = lin(r2, 'qkvz', nxt); xa.copy_(q[:, :3072])
    return run
def build_alone(arm, n=64):
    ar = ar_fn(arm); x = torch.randn(M, 5120, device=dev, dtype=bf)
    def run():
        state['idx'] = 0
        for i in range(n): ar(x, None, None, 0)
    return run
def capture(run):
    with torch.cuda.stream(main):
        for _ in range(2): run()
        main.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=main): run()
    main.synchronize(); return g
def time_graph(g):
    ts = []
    with torch.cuda.stream(main):
        for _ in range(3): g.replay()
        main.synchronize()
        for _ in range(NREP):
            dist.barrier()
            a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(main); g.replay(); b.record(main); b.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts)
def stage(name, builder, nar):
    graphs = {}
    for a in ARMS:
        if name == 'alone' and a == 'A_pf': pass
        graphs[a] = capture(builder(a))
    log('captured', name, list(graphs)); res = {}
    for rep in range(REPS):
        for k, g in graphs.items():
            dist.barrier(); res.setdefault(k, []).append(time_graph(g))
    log(f'===== {name}: ms per graph (median of {REPS} reps), {nar} ARs')
    for k, v in res.items():
        log(f'{k:9s} med {st.median(v):.3f} min {min(v):.3f} max {max(v):.3f}  per-AR(us) {st.median(v)/nar*1000:.1f}')
    if 'A_pf' in res:
        for k in res: log(f'{name} per-AR vs A_pf: A_pf - {k} = {(st.median(res["A_pf"]) - st.median(res[k])) / nar * 1000:+.1f} us')
    return graphs
def stress():
    NARS = int(os.environ.get('STRESS_ARS', '100000')); B = 32; G = int(os.environ.get('STRESS_G', '15'))
    cfgs = os.environ.get('STRESS_CFGS', 'r0late_hbm,r1late_hbm,jitter').split(',')
    HY.stat_enable()
    pfw = torch.empty(40 << 20, device=dev, dtype=torch.uint8)
    hb_a = torch.randn(64 << 20, device=dev, dtype=torch.float32); hb_b = torch.empty_like(hb_a)   # 256 MB each
    bank = torch.empty(B, M, 5120, device=dev, dtype=bf)
    d1 = torch.zeros(B, device=dev, dtype=torch.int32); d2 = torch.zeros(B, device=dev, dtype=torch.int32)
    acc = torch.zeros(1, device=dev, dtype=torch.int64)
    side = torch.cuda.Stream(); ev = torch.cuda.Event()
    ar = ar_fn(f'Tpf_{G}')
    def body():
        for i in range(B):
            HY.spin(d1, i); o_t = ar(bank[i], pfw, None, 0)
            HY.spin(d2, i); o_r = host.all_reduce(bank[i], pfw)
            HY.cmpacc(o_t, o_r, acc)
    g = capture(body)
    def refill(it, late, jitter):
        gg = torch.Generator(device=dev); gg.manual_seed(7919 * it + rank)
        bank.normal_(generator=gg)
        sc = torch.tensor([[1.0, 0.01, 30.0][(it + i) % 3] for i in range(B)], device=dev, dtype=bf).view(B, 1, 1)
        bank.mul_(sc)
        bank[::7, :, ::5] = 0
        bank[::11, 0, 0] = float('inf') if rank == 0 else 1.0
        bank[3::5] = torch.randint(-32768, 32767, (B // 5, M, 5120), device=dev, dtype=torch.int16, generator=gg).view(bf) if B > 5 else bank[3::5]
        ds = torch.randint(0, 80, (2, B), device=dev, generator=gg, dtype=torch.int32)
        for k, d in enumerate((d1, d2)):
            if jitter: d.copy_(ds[k] // 2 if rank == (it & 1) else ds[k] // 2)   # both ranks random, different draws via rank seed
            elif rank == late: d.copy_(ds[k])
            else: d.zero_()
    for cfg in cfgs:
        hbm = cfg.endswith('_hbm'); late = 0 if cfg.startswith('r0') else 1; jit = cfg.startswith('jitter')
        acc.zero_(); HY.stat_get(True); nrep = max(1, NARS // B); t0 = time.time()
        with torch.cuda.stream(main):
            for it in range(nrep):
                refill(it, late, jit)
                if hbm:
                    ev.synchronize()
                    with torch.cuda.stream(side): HY.hbm(hb_a, hb_b, 128, 2); ev.record(side)
                g.replay()
                if it % 20 == 19: main.synchronize()
            main.synchronize()
        side.synchronize(); torch.cuda.synchronize()
        st_ = HY.stat_get(True).tolist(); mism = acc.item(); nar = nrep * B
        log(f'STRESS {cfg}: ARs {nar} (T and hostar-ref each), output u32 words compared bitwise {nar * M * 5120 // 2}, 8B wire words accepted {st_[1]} (+O words {st_[2]}), poll retries (stale/partial reads seen) {st_[0]}, MISMATCHED u32 words {mism}, secs {time.time() - t0:.1f}')
        dist.barrier()
import time
only = os.environ.get('STAGES', 'check,neg,alone,chain').split(',')
if 'check' in only: check()
if 'neg' in only: neg()
if 'alone' in only: stage('alone', lambda a: build_alone(a), 64)
if 'chain' in only:
    gs = stage('chain', build_chain, 2 * NL)
    log('NUMA after:', numa())
if 'stress' in only: stress()
if 'trace' in only:
    import numpy as np
    for arm in [a for a in ARMS if a.startswith('T')][:2]:
        HY.trace_enable(1 << 17); dist.barrier()
        g = capture(build_chain(arm)); dist.barrier()
        with torch.cuda.stream(main):
            for _ in range(10): dist.barrier(); g.replay(); main.synchronize()
        t = HY.trace_dump().numpy(); t = t[t[:, 4] > 0]
        r0 = -t[:, 1]   # reader start (inverted min)
        f = lambda c, inv=False: ((-t[:, c] if inv else t[:, c]) - r0) / 1000.0
        log(f'TRACE {arm} n={len(t)} (us from earliest reader-block start, rank-local globaltimer; med/p10/p90): writer first store {np.median(f(7,True)):.1f} | first word observed {np.median(f(2,True)):.1f} p10 {np.percentile(f(2,True),10):.1f} p90 {np.percentile(f(2,True),90):.1f} | last word observed {np.median(f(3)):.1f} p90 {np.percentile(f(3),90):.1f} | reader end {np.median(f(4)):.1f} | writer last store {np.median(f(6)):.1f} | kernel end {np.median(f(5)):.1f}')
