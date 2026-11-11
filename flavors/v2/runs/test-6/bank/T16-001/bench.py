"""T16-001 hybrid AR (copy-engine write + SM read) microbench. usage: bench.py <rank> <smoke> [arms]
stages: check (bitwise vs hostar), standalone latency (graph of 64 ARs), chain (16 layers, 32 ARs), timeline."""
import os, sys, statistics as st
rank = int(sys.argv[1]); SMOKE = sys.argv[2] == '1'
DEF = ['N','A_pf','H1_1','H1_2','H1_4','H1pf_1','H1pf_2','H1pf_4','H1kpf_2','H2pf_1','H2pf_2','H2_1']
ARMS = sys.argv[3].split(',') if len(sys.argv) > 3 else DEF
FLAGMODE = int(os.environ.get('FLAGMODE', '0'))
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
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29617')
dist.init_process_group('gloo', rank=rank, world_size=2)
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.distributed.device_communicators.host_staged_ar import HostStagedAllReduce
import build
HY = build.get(); PFK = build.getpf()
host = HostStagedAllReduce(dist.group.WORLD, dev)
path = ['/dev/shm/t16hyb_%s' % os.environ['MASTER_PORT']]
if rank == 0: HY.init(path[0], 1 << 20, 0, True)
dist.barrier()
if rank == 1: HY.init(path[0], 1 << 20, 1, False)
dist.barrier()
if rank == 0: os.unlink(path[0])
def numa():
    tot = {}
    for l in open('/proc/self/numa_maps'):
        if '/dev/shm' in l:
            for m in re.finditer(r'N(\d+)=(\d+)', l): tot[m.group(1)] = tot.get(m.group(1), 0) + int(m.group(2))
    return tot
log('affinity n=', len(os.sched_getaffinity(0)), 'host.disabled', host.disabled, 'flagmode', FLAGMODE)
log('NUMA PROOF: /dev/shm pages by node (4KB pages):', numa())
state = {'idx': 0}
PFMAX = 88 << 20
main = torch.cuda.Stream(); side = torch.cuda.Stream()
evs = [torch.cuda.Event() for _ in range(4 * NL + 4)]
def parse(arm):
    # N | A_pf | H{1,2}[k][pf]_{P}
    if arm in ('N', 'A_pf', 'A_nopf'): return dict(kind=arm)
    m = re.match(r'H([12])(k?)(pf)?_?(\d)$', arm); assert m, arm
    return dict(kind='H', comp=int(m.group(1)) == 2, pfk=bool(m.group(2)), pf=bool(m.group(3)), P=int(m.group(4)))
def ar_fn(arm):
    c = parse(arm)
    def ar(h, pfw, pfw2, e):
        if c['kind'] == 'N': return h.clone()
        if c['kind'] == 'A_pf': return host.all_reduce(h, pfw, pfw2)
        if c['kind'] == 'A_nopf': return host.all_reduce(h)
        out = torch.empty_like(h); par = state['idx'] & 1; state['idx'] += 1
        if c['pfk'] and pfw is not None:
            evs[e].record(main); side.wait_event(evs[e])
            with torch.cuda.stream(side): PFK.l2pf(pfw, PFMAX, 8, 2, pfw2)
            evs[e + 1].record(side)
        usepf = c['pf'] and pfw is not None
        HY.allreduce(h, out, c['P'], 8, 512, par, int(c['comp']), FLAGMODE, 8,
                     pfw if usepf else None, PFMAX, 8, 2, pfw2 if usepf else None)
        if c['pfk'] and pfw is not None: main.wait_event(evs[e + 1])
        return out
    return ar
# ---------------- stage 1: bitwise check vs hostar (eager, fresh data each iter)
def check():
    res = {}
    x0 = torch.randn(M, 5120, device=dev, dtype=bf)
    for arm in [a for a in ARMS if a.startswith('H')]:
        ar = ar_fn(arm); bad = 0; mx = 0.0; n = 40 if SMOKE else 300
        for it in range(n):
            g = torch.Generator(device=dev); g.manual_seed(1000 * rank + it)
            scale = [1.0, 0.01, 30.0][it % 3]
            x = (torch.randn(M, 5120, device=dev, dtype=bf, generator=g) * scale)
            if it % 7 == 0: x[:, ::5] = 0
            if it % 11 == 0: x[0, 0] = float('inf') if rank == 0 else 1.0
            ref = host.all_reduce(x.clone())
            out = ar(x, None, None, 0)
            torch.cuda.synchronize()
            ne = (ref.view(torch.int16) != out.view(torch.int16)).sum().item()
            bad += ne > 0
        res[arm] = bad; log('timeouts', HY.timeouts()); log(f'CHECK {arm}: {bad}/{n} iterations differ bitwise vs hostar')
    return res
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
wsp = marlin_make_workspace_new(dev)
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
only = os.environ.get('STAGES', 'check,alone,chain').split(',')
if 'check' in only: check()
if 'alone' in only: stage('alone', lambda a: build_alone(a), 64)
if 'chain' in only:
    gs = stage('chain', build_chain, 2 * NL)
    log('NUMA after:', numa())
if 'trace' in only:
    import numpy as np
    HY.trace_enable(1 << 14); dist.barrier()
    for arm in [a for a in ARMS if a.startswith('H')][:3]:
        g = capture(build_chain(arm)); dist.barrier()
        HY.trace_enable(1 << 14); dist.barrier()
        with torch.cuda.stream(main):
            for _ in range(10): dist.barrier(); g.replay(); main.synchronize()
        t = HY.trace_dump().numpy(); t = t[t[:, 2] > 0]
        d1 = (t[:, 1] - t[:, 0]) / 1000; d3 = (t[:, 3] - t[:, 0]) / 1000; d2 = (t[:, 2] - t[:, 0]) / 1000
        log(f'TRACE {arm} n={len(t)} (us from reader start; globaltimer): first flag seen med {np.median(d1):.1f} p10 {np.percentile(d1,10):.1f} p90 {np.percentile(d1,90):.1f} | last flag seen med {np.median(d3):.1f} | reader end med {np.median(d2):.1f} p90 {np.percentile(d2,90):.1f}')
