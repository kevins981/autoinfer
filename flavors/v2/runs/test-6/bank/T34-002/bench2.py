"""T34-001: mixed-step AR transport microbench. usage: bench2.py <rank>   env: RS=1100,512,.. ARMS=.. STAGES=check,time,trace
Arms: N (producer only) | A_host (hostar, compressed) | A_dma1 (copy-engine, pieces=4, one lane) | A_split (champion mixed-step path:
2 chunks, lane0 on main + lane1 on aux concurrently, c0=64*ceil(R/128)) | H_<P>_<C> (CE D2H P pieces + flags, SM reader C CTAs polls)
| F_ce (CE D2H only) | F_sm (SM read only) | F_both (CE + SM read, no waiting)  [F_* use P=4, C=32]"""
import os, sys, statistics as st
rank = int(sys.argv[1])
RS = [int(v) for v in os.environ.get('RS', '1100,512,128,256,768').split(',')]
NCH = int(os.environ.get('NCH', '16')); REPS = int(os.environ.get('REPS', '5')); NREP = int(os.environ.get('NREP', '8'))
HC = os.environ.get('HARMS', 'H_2_32,H_4_32,H_8_32,H_4_16,H_4_64').split(',')
ARMS = os.environ.get('ARMS', 'N,' + ','.join(HC) + ',F_ce,F_sm,F_both,A_dma1,A_split,A_host').split(',')
NSL = int(os.environ.get('NSL', '300')); FR = [int(v) for v in os.environ.get('FLOOR_RS', '').split(',') if v]
STAGES = os.environ.get('STAGES', 'check,time,trace').split(',')
def cpulist(s):
    r = []
    for p in s.strip().split(','):
        a, _, b = p.partition('-'); r += list(range(int(a), int(b or a) + 1))
    return r
os.sched_setaffinity(0, cpulist(open('/sys/devices/system/node/node0/cpulist').read()))
import ctypes
ctypes.CDLL('libnuma.so.1').numa_set_preferred(0)
import re, torch, torch.distributed as dist
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
torch.cuda.set_device(rank); dev = torch.device('cuda', rank); bf = torch.bfloat16
def log(*a): print(f'[r{rank}]', *a, flush=True)
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29634')
dist.init_process_group('gloo', rank=rank, world_size=2)
WITH_A = os.environ.get('WITH_A', '0') == '1'
if WITH_A:
    from vllm.distributed.device_communicators.host_staged_ar import HostStagedAllReduce
    from vllm.distributed.device_communicators.dma_staged_ar import DmaStagedAllReduce
import build
HY = build.get()
host = HostStagedAllReduce(dist.group.WORLD, dev) if WITH_A else None
dma = DmaStagedAllReduce(dist.group.WORLD, dev) if WITH_A else None
KEEP = []; PTRS = {}   # every tensor a graph touches stays alive until os._exit
def die(code, msg):
    print(f'[r{rank}] FATAL: {msg}', flush=True); os._exit(code)
path = '/dev/shm/t34hyb_%s' % os.environ['MASTER_PORT']
if rank == 0: HY.init(path, 12 << 20, 0, True)
dist.barrier()
if rank == 1: HY.init(path, 12 << 20, 1, False)
dist.barrier()
if rank == 0: os.unlink(path)
def numa():
    tot = {}
    for l in open('/proc/self/numa_maps'):
        if '/dev/shm' in l:
            for m in re.finditer(r'N(\d+)=(\d+)', l): tot[m.group(1)] = tot.get(m.group(1), 0) + int(m.group(2))
    return tot
log('affinity n=', len(os.sched_getaffinity(0)), 'WITH_A', WITH_A)
log('NUMA PROOF: /dev/shm pages by node (4KB pages):', numa())
main = torch.cuda.Stream(); aux = torch.cuda.Stream()
state = {'idx': 0}
def parse(a):
    if a.startswith('H_'): _, P, C = a.split('_'); return ('H', int(P), int(C), 0)
    if a.startswith('H1_'): _, P, C = a.split('_'); return ('H1', int(P), int(C), 0)
    return {'F_ce': ('H', 4, 32, 1), 'F_sm': ('H', 4, 32, 2), 'F_both': ('H', 4, 32, 3)}.get(a, (a, 0, 0, 0))
def make_ar(arm, R):
    k, P, C, mode = parse(arm); W = 5120
    c0 = 64 * -(-R // 128)
    def ar(x, out):
        if k == 'N': return
        if k == 'A_host': return host.all_reduce(x)   # hostar allocates its own out
        if k == 'A_dma1': dma.all_reduce(x, out, lane=0); return
        if k == 'A_split':
            cur = torch.cuda.current_stream(); aux.wait_stream(cur)
            dma.all_reduce(x[:c0], out[:c0], lane=0)
            with torch.cuda.stream(aux): dma.all_reduce(x[c0:], out[c0:], lane=1)
            cur.wait_stream(aux); return
        par = state['idx'] & 1; state['idx'] += 1
        HY.allreduce(x, out, P, C, 512, par, mode, int(k == 'H1'), NSL)
    return ar
def check():
    for R in [r for r in RS if r not in FR]:
        for arm in [a for a in ARMS if a.startswith(('H_', 'H1_'))]:
            ar = make_ar(arm, R); bad = 0; n = 24
            for it in range(n):
                g = torch.Generator(device=dev); g.manual_seed(1000 * rank + it)
                x = torch.randn(R, 5120, device=dev, dtype=bf, generator=g) * [1.0, 0.01, 30.0][it % 3]
                if it % 7 == 0: x[:, ::5] = 0
                if it % 11 == 0: x[0, 0] = float('inf') if rank == 0 else 1.0
                if it % 5 == 1: x[R // 2] = 1e30 * (1 if rank == 0 else -1)
                xc = x.cpu(); both = [torch.empty_like(xc), torch.empty_like(xc)]
                dist.all_gather(both, xc)   # gloo, no GPU waits
                pr = both[1 - rank].to(dev)
                ref = (x.float() + pr.float()).to(bf)   # same math: bf16(float(x)+float(peer))
                out = torch.empty_like(x); ar(x, out); torch.cuda.synchronize()
                if HY.timeouts() > 0: die(2, f'flag timeout in check R={R} {arm}')
                a = (ref.view(torch.int16) != out.view(torch.int16)).sum().item()
                bad += (a > 0)
                if a: log(f'  mismatch R={R} {arm} it={it}: {a} elements')
            log(f'CHECK R={R} {arm}: {bad}/{n} iterations differ bitwise vs fp32-add reference; timeouts {HY.timeouts()}')
def build_chain(arm, R, n=None):
    n = n or NCH
    ar = make_ar(arm, R)
    xs = [torch.randn(R, 5120, device=dev, dtype=bf), torch.randn(R, 5120, device=dev, dtype=bf)]
    outs = [torch.zeros(R, 5120, device=dev, dtype=bf), torch.zeros(R, 5120, device=dev, dtype=bf)]
    KEEP.append((xs, outs)); PTRS[id(xs)] = [t.data_ptr() for t in xs + outs]
    def run():
        state['idx'] = 0
        for i in range(n):
            x, o = xs[i & 1], outs[i & 1]
            HY.produce(outs[(i - 1) & 1], x, 1024)
            ar(x, o)   # A_host's own output is not consumed: persistent buffers only (no graph reads a freed pointer)
    return run
def capture(run):
    with torch.cuda.stream(main):
        for _ in range(2): run()
        main.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=main): run()
    main.synchronize(); KEEP.append(g); return g
def time_graph(g):
    ts = []
    with torch.cuda.stream(main):
        for _ in range(2): g.replay()
        main.synchronize()
        for _ in range(NREP):
            dist.barrier()
            a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(main); g.replay(); b.record(main); b.synchronize(); ts.append(a.elapsed_time(b))
    return st.median(ts)
def smoke():
    # eager (sync after every call) then 1 graph capture + ONE replay per arm (short chain), error check
    SA = os.environ.get('SMARMS', ','.join(ARMS)).split(',')
    for R in [int(v) for v in os.environ.get('SMRS', ','.join(map(str, RS))).split(',')]:
        for arm in [a for a in SA if a.startswith(('H_', 'H1_', 'F_', 'A_'))]:
            if arm == 'A_split' and R < 256: continue
            log('smoke start', R, arm)
            run = build_chain(arm, R, 2)
            xs_, outs_ = KEEP[-1]
            for _ in range(3): run(); torch.cuda.synchronize()
            if HY.timeouts() > 0: die(2, f'timeouts after eager {R} {arm}')
            g = capture(run)
            if PTRS[id(xs_)] != [t.data_ptr() for t in xs_ + outs_]: die(3, 'buffer pointer changed after capture')
            dist.barrier(); g.replay(); torch.cuda.synchronize()
            if HY.timeouts() > 0: die(2, f'timeouts after graph replay {R} {arm}')
            log(f'SMOKE ok R={R} {arm} (eager x3 + graph x1)')
def timing():
    for R in RS:
        graphs = {}
        for a in ([x for x in ARMS if x in ('N', 'F_ce', 'F_sm', 'F_both')] if R in FR else ARMS):
            if a == 'A_split' and R < 256: continue
            graphs[a] = capture(build_chain(a, R))
        res = {}
        for rep in range(REPS):
            for k, g in graphs.items():
                if HY.timeouts() > 0: die(2, f'timeouts before {k}')
                if rep == 0: log('timing', R, k)
                dist.barrier(); res.setdefault(k, []).append(time_graph(g))
        n0 = st.median(res['N'])
        log(f'===== R={R} ({R*10240/1e6:.2f} MB/rank) chain of {NCH} ARs, ms med of {REPS}x{NREP}; per-AR us = (chain - N)/{NCH}; timeouts {HY.timeouts()}')
        for k, v in res.items():
            m = st.median(v); log(f'TIME R={R} {k:9s} chain {m:.3f} ms  per-AR {(m-n0)/NCH*1000:8.1f} us   (min {(min(v)-n0)/NCH*1000:.1f} max {(max(v)-n0)/NCH*1000:.1f})')
def trace():
    import numpy as np
    for R in [r for r in RS if r not in FR]:
        HY.trace_enable(1 << 12)
        for arm in [a for a in ARMS if a in ('H1_4_32', 'H1_8_32', 'H_4_32', 'F_both')][:int(os.environ.get('NTR', '2'))]:
            g = capture(build_chain(arm, R)); dist.barrier()
            HY.trace_enable(1 << 12); dist.barrier()
            with torch.cuda.stream(main):
                for _ in range(10): dist.barrier(); g.replay(); main.synchronize()
            t = HY.trace_dump().numpy(); t = t[t[:, 2] > 0]
            s = t[:, 0]; f = lambda c: (t[:, c] - s) / 1000
            pe = (s - t[:, 4]) / 1000
            log(f'TRACE R={R} {arm} n={len(t)} us rel. reader start: producer-end->reader-start med {np.median(pe):.1f} | first flag seen med {np.median(f(1)):.1f} p10 {np.percentile(f(1),10):.1f} | last flag seen med {np.median(f(3)):.1f} | reader end med {np.median(f(2)):.1f} p90 {np.percentile(f(2),90):.1f}')
import traceback
try:
    if 'check' in STAGES: check()
    if 'smoke' in STAGES: smoke()
    if 'time' in STAGES: timing()
    if 'trace' in STAGES: trace()
except BaseException:
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
log('NUMA after:', numa()); sys.stdout.flush(); os._exit(0)
