"""T9-006 decode copy-engine AR chain microbench (M=48, 16 layers, CUDA graphs, real FP8 Marlin, NUMA-bound).
Arms (per 16-layer chain; 32 ARs): A_pf = champion hostar + compression + L2 prefetch (HostStagedAllReduce.all_reduce(prefetch=next GEMM weights));
D_pf{2,4} = DMA AR (pieces 2/4) + separate evict_last prefetch kernel (8 blocks, same 88 MB / step 2) on a side stream during the copies;
D{2,4} = DMA, no prefetch; N = no AR (copy only). Gate: D_pf - A_pf <= -8 us per AR.
usage: bench5.py <rank> <smoke 0/1> [arms comma]   (shm for the dma ext is handled by DmaStagedAllReduce itself)"""
import os, sys, statistics as st
rank = int(sys.argv[1]); SMOKE = sys.argv[2] == '1'
ARMS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['A_pf', 'A_nopf', 'D_pf2', 'D_pf4', 'D2', 'D4', 'N']
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
except OSError: NUMA = False
import re, torch, torch.distributed as dist
NL = 4 if SMOKE else 16; REPS = 2 if SMOKE else 6; NREP = 8 if SMOKE else 30; M = 48
HERE = os.path.dirname(os.path.abspath(__file__))
torch.cuda.set_device(rank); dev = torch.device('cuda', rank); bf = torch.bfloat16
def log(*a): print(f'[r{rank}]', *a, flush=True)
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', os.environ.get('OVL_PORT', '29591'))
dist.init_process_group('gloo', rank=rank, world_size=2)
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.distributed.device_communicators.host_staged_ar import HostStagedAllReduce
from vllm.distributed.device_communicators.dma_staged_ar import DmaStagedAllReduce
from torch.utils.cpp_extension import load
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
P = load('t9pfk', [HERE + '/pfk.cu'], build_directory=os.environ.get('CE_BUILD', HERE + '/ext'), verbose=False)
host = HostStagedAllReduce(dist.group.WORLD, dev)
dma = DmaStagedAllReduce(dist.group.WORLD, dev)
log('affinity n=', len(os.sched_getaffinity(0)), 'libnuma preferred(0):', NUMA, 'host.disabled', host.disabled, 'dma.disabled', dma.disabled, 'dma pieces', dma.pieces)
tot = {}
for l in open('/proc/self/numa_maps'):
    if '/dev/shm' in l:
        for m in re.finditer(r'N(\d+)=(\d+)', l): tot[m.group(1)] = tot.get(m.group(1), 0) + int(m.group(2))
log('NUMA PROOF: /dev/shm pages by node (4KB pages):', tot)
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
    N, K = SH[name]
    return apply_fp8_marlin_linear(x, L[name].weight, L[name].weight_scale_inv, wsp, N, K, None)
xa = torch.randn(M, 3072, device=dev, dtype=bf); resid = torch.randn(M, 5120, device=dev, dtype=bf)
nw = torch.randn(5120, device=dev, dtype=bf) * 0.1 + 1
main = torch.cuda.Stream(); side = torch.cuda.Stream()
PFMAX = 88 << 20
evs = [torch.cuda.Event() for _ in range(4 * NL + 4)]
def addnorm(x): ops.fused_add_rms_norm(x, resid, nw, 1e-6)

def build(arm):
    kind = arm.split('_')[0].rstrip('0123456789'); pf = '_pf' in arm
    pieces = int(arm[-1]) if arm[-1] in '24' else None
    def ar(h, pfw, pfw2, e):
        """returns reduced tensor; e = event slot base"""
        if kind == 'A':
            return host.all_reduce(h, pfw if arm == 'A_pf' else None, pfw2 if arm == 'A_pf' else None)
        if kind == 'N':
            return h.clone()
        dma.pieces = pieces
        if pf:
            evs[e].record(main); side.wait_event(evs[e])
            with torch.cuda.stream(side): P.l2pf(pfw, PFMAX, 8, 2, pfw2)
            evs[e + 1].record(side)
        o = dma.all_reduce(h)
        if pf: main.wait_event(evs[e + 1])
        return o
    def run():
        e = 0
        for i, L in enumerate(layers):
            nxt = layers[(i + 1) % NL]
            h1 = lin(xa, 'out', L)
            r1 = ar(h1, L['gu'].weight, None, e); e += 2
            addnorm(r1)
            gu = lin(r1, 'gu', L)
            act = torch.empty(M, 8704, device=dev, dtype=bf); torch.ops._C.silu_and_mul(act, gu)
            h2 = lin(act, 'down', L)
            r2 = ar(h2, nxt['qkvz'].weight, nxt['out'].weight, e); e += 2
            addnorm(r2)
            q = lin(r2, 'qkvz', nxt)
            xa.copy_(q[:, :3072])
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
graphs = {a: capture(build(a)) for a in ARMS}
log('captured', list(graphs))
res = {}
for rep in range(REPS):
    for k, g in graphs.items():
        dist.barrier(); res.setdefault(k, []).append(time_graph(g))
    log('rep', rep, {k: round(v[-1], 3) for k, v in res.items()})
nar = 2 * NL
log('===== ms per %d-layer chain (median of %d reps), %d ARs' % (NL, REPS, nar))
for k, v in res.items(): log(f'{k:7s} med {st.median(v):.3f} min {min(v):.3f} max {max(v):.3f}')
if 'A_pf' in res:
    for k in res:
        log(f'per-AR vs A_pf: A_pf - {k} = {(st.median(res["A_pf"]) - st.median(res[k])) / nar * 1000:+.1f} us  (gate: D_pf >= +8)')
