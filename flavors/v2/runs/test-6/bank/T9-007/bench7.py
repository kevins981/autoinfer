"""T9-007 (from T9-004 bench4): S3c vs S3cc (codec on the CE payload). T9-002: ping-pong microbench with the REAL core (conv1d + post-conv prep + FLA chunk_gated_delta_rule + gated norm; FA2 attention every 4th layer).
usage: bench2.py <rank> <shm> <smoke 0/1> <Rs> <arms comma> [check]"""
import os, sys, statistics as st, re
rank = int(sys.argv[1]); SHM = sys.argv[2]; SMOKE = sys.argv[3] == '1'; RS = [int(x) for x in sys.argv[4].split(',')]
ARMS = sys.argv[5].split(',') if len(sys.argv) > 5 else ['N', 'S2', 'S3c', 'S3cc']
CHECK = len(sys.argv) > 6 and sys.argv[6] == 'check'
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
NL = 4 if SMOKE else 8
REPS = 2 if SMOKE else 5
NREP = 6 if SMOKE else 20
HERE = os.path.dirname(os.path.abspath(__file__))
PC = int(os.environ.get('PC', '2'))
torch.cuda.set_device(rank); dev = torch.device('cuda', rank); bf = torch.bfloat16
def log(*a): print(f'[r{rank}]', *a, flush=True)
log('affinity n=', len(os.sched_getaffinity(0)), 'libnuma preferred(0):', NUMA)
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', os.environ.get('OVL_PORT', '29587'))
dist.init_process_group('gloo', rank=rank, world_size=2)
import numpy as np
import vllm._custom_ops as ops
from types import SimpleNamespace
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.distributed.device_communicators.host_staged_ar import HostStagedAllReduce
from vllm.distributed.device_communicators.dma_staged_ar import DmaStagedAllReduce
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule as fla_gdr, fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
from vllm.models.qwen3_5.nvidia.ops.fused_gdn_gated_norm import fused_gdn_gated_rms_norm
from vllm.vllm_flash_attn import flash_attn_varlen_func
from torch.utils.cpp_extension import load
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
X = load('t9ce7', [HERE + '/ce.cu'], build_directory=os.environ.get('CE_BUILD', HERE + '/ext'), extra_ldflags=['-lcuda'], verbose=False)
host = HostStagedAllReduce(dist.group.WORLD, dev)
dma = DmaStagedAllReduce(dist.group.WORLD, dev)
log('host.disabled', host.disabled, 'dma.disabled', dma.disabled, 'dma pieces', dma.pieces)
SLOT = 6 << 20
if rank == 0: X.init(SHM, SLOT, 0, True)
dist.barrier()
if rank == 1: X.init(SHM, SLOT, 1, False)
dist.barrier()
if rank == 0: os.unlink(SHM)
tot = {}
for l in open('/proc/self/numa_maps'):
    if '/dev/shm' in l:
        for m in re.finditer(r'N(\d+)=(\d+)', l): tot[m.group(1)] = tot.get(m.group(1), 0) + int(m.group(2))
log('NUMA PROOF: /dev/shm pages by node:', tot, 'MB:', {k: round(v * 4 / 1024) for k, v in tot.items()})

def mk(N, K):
    m = torch.nn.Module()
    m.output_size_per_partition = N; m.input_size_per_partition = K; m.weight_block_size = [128, 128]; m.orig_dtype = bf
    m.weight = torch.nn.Parameter((torch.randint(0, 0x78, (N, K), device=dev, dtype=torch.int32) | (torch.randint(0, 2, (N, K), device=dev, dtype=torch.int32) << 7)).to(torch.uint8).view(torch.float8_e4m3fn), requires_grad=False)
    m.weight_scale_inv = torch.nn.Parameter(torch.rand(N // 128, K // 128, device=dev) * 8e-5 + 2e-5, requires_grad=False)
    prepare_fp8_layer_for_marlin(m, size_k_first=False)
    m.N = N; m.K = K
    return m
def is_attn(i): return i % 4 == 3
S_L = [dict(qkvz=mk(7168 if is_attn(i) else 8192, 5120), gu=mk(17408, 5120), down=mk(5120, 8704), out=mk(5120, 3072)) for i in range(NL)]
OVF = torch.zeros(2, dtype=torch.int32, device=torch.device('cuda', rank))
main = torch.cuda.Stream(); sA = torch.cuda.Stream(); sB = torch.cuda.Stream()
sAh = torch.cuda.Stream(priority=-1); dA = torch.cuda.Stream(); hA = torch.cuda.Stream(); dB = torch.cuda.Stream(); hB = torch.cuda.Stream()
WS = {s: marlin_make_workspace_new(dev) for s in ('main', 'A', 'B')}
nw = torch.randn(5120, device=dev, dtype=bf) * 0.1 + 1
# GDN per-layer params (random): conv weight [5120,4], A_log [24] f32, dt_bias [24], gated-norm weight [128]
torch.manual_seed(1234)
GP = [dict(cw=(torch.randn(5120, 4, device=dev, dtype=bf) * 0.3).contiguous(), A_log=torch.randn(24, device=dev, dtype=torch.float32) * 0.3,
           dt_bias=torch.randn(24, device=dev, dtype=bf) * 0.1, nw=torch.ones(128, device=dev, dtype=bf)) for _ in range(NL)]

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

class Seg:
    """metadata for one core call over `rows` rows of a single sequence"""
    def __init__(self, rows, has_init):
        self.rows = rows
        cu_cpu = torch.tensor([0, rows], dtype=torch.int32)
        self.cu = cu_cpu.to(dev)
        nd, bp, tp = compute_causal_conv1d_metadata(cu_cpu, device=dev)
        self.meta = SimpleNamespace(nums_dict=nd, batch_ptr=bp, token_chunk_offset_ptr=tp)
        self.ci = prepare_chunk_indices(cu_cpu, 64).to(dev); self.co = prepare_chunk_offsets(cu_cpu, 64).to(dev)
        self.has_init = torch.tensor([has_init], dtype=torch.bool, device=dev)
        self.cidx = torch.ones(1, dtype=torch.int32, device=dev)  # index 0 is NULL_BLOCK_ID (skipped)
        self.cuk = None
torch.cuda.synchronize()

def gdn_core(qout, z, a, b, gp, seg, conv_state, h0):
    """real GDN prefill core on rows of qout[:, :5120] (+ z) -> ([rows,3072], final_state)"""
    mixed = qout[:, :5120].contiguous()
    x = causal_conv1d_fn(mixed.transpose(0, 1), gp['cw'], None, activation='silu', conv_states=conv_state, has_initial_state=seg.has_init,
                         cache_indices=seg.cidx, query_start_loc=seg.cu, metadata=seg.meta).transpose(0, 1)
    q, k, v, g, beta = fused_post_conv_prep(conv_output=x, a=a, b=b, A_log=gp['A_log'], dt_bias=gp['dt_bias'], num_k_heads=8, head_k_dim=128, head_v_dim=128,
                                            apply_l2norm=True, output_g_exp=False)
    o, fs = fla_gdr(q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0), g=g.unsqueeze(0), beta=beta.unsqueeze(0), initial_state=h0, output_final_state=True,
                    cu_seqlens=seg.cu, chunk_indices=seg.ci, chunk_offsets=seg.co, use_qk_l2norm_in_kernel=False)
    out = torch.empty(seg.rows, 3072, device=dev, dtype=bf)
    fused_gdn_gated_rms_norm(x=o.view(seg.rows, 3072), z=qout[:, 5120:8192], weight=gp['nw'], out=out, eps=1e-6, num_v_heads=24, head_v_dim=128,
                             norm_before_gate=True, activation='silu')
    return out, fs

def attn_core(qout, Kb, Vb, a0, rows, cuq, cuk):
    """FA2 prefill over rows [a0, a0+rows) attending to KV rows [0, a0+rows) (chunk B sees chunk A's KV)"""
    b = a0 + rows
    Kb[a0:b].copy_(qout[:, 6144:6656].view(rows, 2, 256)); Vb[a0:b].copy_(qout[:, 6656:7168].view(rows, 2, 256))
    return qout[:, :3072].view(rows, 12, 256), b

def attn_call(q, Kb, Vb, b, rows, cuq, cuk):
    o = flash_attn_varlen_func(q, Kb[:b], Vb[:b], max_seqlen_q=rows, cu_seqlens_q=cuq, max_seqlen_k=b, cu_seqlens_k=cuk, causal=True, fa_version=2)
    return o.reshape(rows, 3072)

def build(arm, R, split=None, nl=None, stampbuf=None):
    nl = NL if nl is None else nl
    M = R; keep = []
    torch.manual_seed(99 + rank)
    def K(t): keep.append(t); return t
    c0 = 64 * -(-M // 128) if split is None else split
    chunks = [(0, c0), (c0, M)]
    resid = torch.randn(M, 5120, device=dev, dtype=bf)
    r2 = torch.randn(M, 5120, device=dev, dtype=bf)
    r2init = r2.clone(); residinit = resid.clone()
    tmp = {c: torch.empty(chunks[c][1] - chunks[c][0], 5120, device=dev, dtype=bf) for c in (0, 1)}
    # per-layer persistent inputs / state
    ab = [(torch.randn(M, 24, device=dev, dtype=bf), torch.randn(M, 24, device=dev, dtype=bf)) for _ in range(NL)]
    segs = {('f',): Seg(M, False), (0,): Seg(c0, False), (1,): Seg(M - c0, True)}
    cst = {(i, c): torch.zeros(2, 5120, 3, device=dev, dtype=bf) for i in range(NL) for c in ('f', 0, 1)}
    h00 = torch.zeros(1, 24, 128, 128, device=dev, dtype=torch.float32)
    Kb = [torch.empty(M, 2, 256, device=dev, dtype=bf) for _ in range(NL)]; Vb = [torch.empty(M, 2, 256, device=dev, dtype=bf) for _ in range(NL)]
    cuq = {c: torch.tensor([0, n], dtype=torch.int32, device=dev) for c, n in (('f', M), (0, c0), (1, M - c0))}
    cukf = {c: torch.tensor([0, n], dtype=torch.int32, device=dev) for c, n in (('f', M), (0, c0), (1, M))}
    fs_store = {}
    outs = {}
    def lin(x, m, w): return K(apply_fp8_marlin_linear(x, m.weight, m.weight_scale_inv, WS[w], m.N, m.K, None))
    def addnorm(x, rs): ops.fused_add_rms_norm(x, rs, nw, 1e-6)
    def silu(gu):
        act = K(torch.empty(gu.shape[0], 8704, device=dev, dtype=bf)); torch.ops._C.silu_and_mul(act, gu); return act
    def core(i, qout, c):
        """c in 'f' (full rows), 0 (chunk A), 1 (chunk B); qout = the rows of this call's qkvz output"""
        a0, b0 = (0, M) if c == 'f' else chunks[c]
        rows = b0 - a0
        if is_attn(i):
            q, b = attn_core(qout, Kb[i], Vb[i], a0, rows, None, None)
            if c == 1: pass
            return K(attn_call(q, Kb[i], Vb[i], b, rows, cuq[c], cukf[c]))
        if c == 1: h0 = fs_store[(i, 0)]
        else: h0 = h00
        o, fs = gdn_core(qout, qout[:, 5120:8192], ab[i][0][a0:b0], ab[i][1][a0:b0], GP[i], segs[(c,) if c != 'f' else ('f',)], cst[(i, c)], h0)
        fs_store[(i, c)] = fs; outs[(i, c)] = (o, fs)
        return K(o)
    def lid(i, b, c): return (i * 2 + b) * 2 + c
    def stp(s, idx):
        if stampbuf is not None:
            with torch.cuda.stream(s): X.stamp(stampbuf, idx)
    def ce_ar(p, out, s, d, h, c, tid0, sb=None):
        nb = p.numel() * 2
        psz = ((nb + PC * 16 - 1) // (PC * 16)) * 16
        fork(s, d, h)
        for k in range(PC):
            o = k * psz
            if o >= nb: break
            n = min(psz, nb - o); tid = tid0 * PC + k
            if sb is not None and k == 0: stp(d, sb + 0)
            with torch.cuda.stream(d): X.send(p, tid, o, n)
            if sb is not None and k == PC - 1: stp(d, sb + 1)
            with torch.cuda.stream(h): X.recvwait(tid)
            if sb is not None and k == 0: stp(h, sb + 2)
            with torch.cuda.stream(h): X.recvcopy(tmp[c], tid, o, n)
            if sb is not None and k == PC - 1: stp(h, sb + 3)
        s.wait_event(rec(h)); s.wait_event(rec(d))
        X.add2(p, tmp[c], out)
    encb = {c: torch.empty(X.enc_bytes((chunks[c][1] - chunks[c][0]) * 5120) // 2, device=dev, dtype=bf) for c in (0, 1)}
    rxb = {c: torch.empty_like(encb[c]) for c in (0, 1)}
    keep.extend(list(encb.values()) + list(rxb.values()))
    def cc_ar(p, out, s, d, h, c, tid0):
        # encode (main chain stream s) -> CE moves 0.781x bytes -> fused decode+add
        e = encb[c]; cb = X.enc_bytes(p.numel())
        X.enc(p, e, OVF)
        psz = ((cb + PC * 16 - 1) // (PC * 16)) * 16
        fork(s, d, h)
        for k in range(PC):
            o = k * psz
            if o >= cb: break
            n = min(psz, cb - o); tid = tid0 * PC + k
            with torch.cuda.stream(d): X.send(e, tid, o, n)
            with torch.cuda.stream(h): X.recvwait(tid)
            with torch.cuda.stream(h): X.recvcopy(rxb[c], tid, o, n)
        s.wait_event(rec(h)); s.wait_event(rec(d))
        X.dec_add(p, rxb[c], out, OVF)
    def run():
        EV.reset(); keep.clear(); fs_store.clear()
        if arm in ('N', 'S', 'S2'):
            for i in range(nl):
                L = S_L[i]
                q = lin(r2, L['qkvz'], 'main')
                x = core(i, q, 'f')
                if arm == 'S2':
                    fork(main, sA, sB)
                    def chunk(c, s, w, kind):
                        a, b = chunks[c]
                        with torch.cuda.stream(s):
                            p = lin(x[a:b], L['out'], w)
                            p = K(dma.all_reduce(p)) if kind == 'dma' else K(host.all_reduce(p))
                            addnorm(p, resid[a:b])
                            act = silu(lin(p, L['gu'], w)); d = lin(act, L['down'], w)
                            if kind == 'dma': dma.all_reduce(d, r2[a:b])
                            else: r2[a:b].copy_(host.all_reduce(d))
                    chunk(0, sA, 'A', 'dma'); chunk(1, sB, 'B', 'host')
                    fork(sA, main); fork(sB, main)
                    addnorm(r2, resid)
                else:
                    h1 = lin(x, L['out'], 'main')
                    if arm == 'S': h1 = K(host.all_reduce(h1))
                    addnorm(h1, resid)
                    act = silu(lin(h1, L['gu'], 'main')); h2 = lin(act, L['down'], 'main')
                    if arm == 'S': h2 = K(host.all_reduce(h2))
                    r2.copy_(h2); addnorm(r2, resid)
        elif arm in ('S3', 'S3c', 'S3cc'):
            # champion-style chain per chunk (core on full rows, join per layer) EXTENDED with the next layer's qkvz GEMM per chunk inside the chain
            qbuf = [torch.empty(M, 7168 if is_attn(j) else 8192, device=dev, dtype=bf) for j in range(NL)]
            keep.extend(qbuf)
            q = lin(r2, S_L[0]['qkvz'], 'main')
            for i in range(nl):
                L = S_L[i]; nj = (i + 1) % NL
                x = core(i, q, 'f')
                fork(main, sA, sB)
                def chunk(c, s, w, kind):
                    a, b = chunks[c]
                    with torch.cuda.stream(s):
                        p = lin(x[a:b], L['out'], w)
                        if kind == 'dma': p = K(dma.all_reduce(p))
                        elif kind == 'host': p = K(host.all_reduce(p))
                        elif kind == 'cc':
                            o = K(torch.empty_like(p)); cc_ar(p, o, s, D_[c], H_[c], c, lid(i, 0, c)); p = o
                        else:
                            o = K(torch.empty_like(p)); ce_ar(p, o, s, D_[c], H_[c], c, lid(i, 0, c)); p = o
                        addnorm(p, resid[a:b])
                        act = silu(lin(p, L['gu'], w)); d = lin(act, L['down'], w)
                        if kind == 'dma': dma.all_reduce(d, r2[a:b])
                        elif kind == 'host': r2[a:b].copy_(host.all_reduce(d))
                        elif kind == 'cc':
                            o = K(torch.empty_like(d)); cc_ar(d, o, s, D_[c], H_[c], c, lid(i, 1, c)); r2[a:b].copy_(o)
                        else:
                            o = K(torch.empty_like(d)); ce_ar(d, o, s, D_[c], H_[c], c, lid(i, 1, c)); r2[a:b].copy_(o)
                        addnorm(r2[a:b], resid[a:b])
                        qbuf[nj][a:b].copy_(lin(r2[a:b], S_L[nj]['qkvz'], w))
                D_ = [dA, dB]; H_ = [hA, hB]
                if arm == 'S3': chunk(0, sA, 'A', 'dma'); chunk(1, sB, 'B', 'host')
                else:
                    kd = 'cc' if arm == 'S3cc' else 'ce'
                    chunk(0, sA, 'A', kd); chunk(1, sB, 'B', kd)
                fork(sA, main); fork(sB, main)
                q = qbuf[nj]
        else:  # P / NP / PD (no core dependency) / PT (stamped)
            comm = arm != 'NP'
            nodep = arm == 'PD'
            S = [sA, sB]; W = ['A', 'B']; D = [dA, dB]; H = [hA, hB]
            fork(main, S[0], S[1])
            evc = [None] * NL
            for i in range(nl):
                L = S_L[i]
                for c in (0, 1):
                    s = S[c]; w = W[c]; a, b = chunks[c]
                    sbase = (((i - 1) * 2 + c) * 16) if (stampbuf is not None and i in (1, 2)) else None
                    def T(k):
                        if sbase is not None: stp(s, sbase + k)
                    with torch.cuda.stream(s):
                        T(0)
                        q = lin(r2[a:b], L['qkvz'], w)
                        if c == 1 and not nodep: s.wait_event(evc[i])
                        T(1)
                        x = core(i, q, c)
                        if c == 0: evc[i] = rec(s)
                        T(2)
                        p = lin(x, L['out'], w)
                        T(3)
                        if comm:
                            o = K(torch.empty_like(p)); ce_ar(p, o, s, D[c], H[c], c, lid(i, 0, c), None if sbase is None else sbase + 4); p = o
                        addnorm(p, resid[a:b])
                        act = silu(lin(p, L['gu'], w))
                        T(8)
                        d = lin(act, L['down'], w)
                        T(9)
                        if comm:
                            o = K(torch.empty_like(d)); ce_ar(d, o, s, D[c], H[c], c, lid(i, 1, c), None if sbase is None else sbase + 10); d = o
                        r2[a:b].copy_(d); addnorm(r2[a:b], resid[a:b])
                        T(14)
            fork(S[0], main); fork(S[1], main)
    def reset():
        r2.copy_(r2init); resid.copy_(residinit)
        for v in cst.values(): v.zero_()
    return run, reset, (r2, resid, outs, keep)

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

def mdiff(a, b):
    a = a.float(); b = b.float(); d = (a - b).abs(); ok = ~(torch.isnan(a) | torch.isnan(b)); same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    return dict(bitwise=bool(same.all()), maxabs=float(d[ok].max()) if ok.any() else None, max_ref=float(b[ok].abs().max()) if ok.any() else None, neq=int((~same).sum()), nan_a=int(torch.isnan(a).sum()), nan_b=int(torch.isnan(b).sum()), n=a.numel())

def gdn_handoff_check(R=384):
    """single-call GDN core vs two-call (state handoff) at a 64-aligned and a non-64-aligned split (eager)."""
    torch.manual_seed(7)
    qout = (torch.randn(R, 8192, device=dev, dtype=bf)); a = torch.randn(R, 24, device=dev, dtype=bf); b = torch.randn(R, 24, device=dev, dtype=bf)
    gp = GP[0]
    def one():
        cst = torch.zeros(2, 5120, 3, device=dev, dtype=bf)
        return gdn_core(qout, qout[:, 5120:8192], a, b, gp, Seg(R, False), cst, torch.zeros(1, 24, 128, 128, device=dev, dtype=torch.float32))
    o1, f1 = one()
    res = {}
    for split in (192, 200, 128, 256, 64):
        cst = torch.zeros(2, 5120, 3, device=dev, dtype=bf); h0 = torch.zeros(1, 24, 128, 128, device=dev, dtype=torch.float32)
        oA, fA = gdn_core(qout[:split], qout[:split, 5120:8192], a[:split], b[:split], gp, Seg(split, False), cst, h0)
        oB, fB = gdn_core(qout[split:], qout[split:, 5120:8192], a[split:], b[split:], gp, Seg(R - split, True), cst, fA)
        o2 = torch.cat([oA, oB])
        res[split] = dict(out=mdiff(o2, o1), final_state=mdiff(fB, f1))
    torch.cuda.synchronize()
    return res

if False:
    r = gdn_handoff_check(384)
    for sp, d in r.items(): log(f'GDN HANDOFF CHECK R=384 split={sp} ({"64-aligned" if sp % 64 == 0 else "NOT 64-aligned"}): out {d["out"]} | final_state {d["final_state"]}')

def chain_check(R=384, nl=1):
    out = {}
    for arm in ('S2', 'P', 'S2b'):
        run, reset, (r2, resid, outs, keep) = build(arm.rstrip('b'), R, nl=nl)
        reset(); torch.cuda.synchronize(); dist.barrier()
        with torch.cuda.stream(main): run()
        torch.cuda.synchronize(); out[arm] = (r2.clone(), resid.clone())
        dist.barrier()
    log(f'CHAIN CHECK R={R} layers={nl}: P vs S2 final r2: {mdiff(out["P"][0], out["S2"][0])} | resid: {mdiff(out["P"][1], out["S2"][1])} || determinism S2 vs S2 rerun: r2 {mdiff(out["S2b"][0], out["S2"][0])}')


# core time alone (per layer), real GDN / attention
def core_time(R, kind):
    run_, reset_, _ = None, None, None
    M = R
    qout = torch.randn(M, 8192, device=dev, dtype=bf); a = torch.randn(M, 24, device=dev, dtype=bf); b = torch.randn(M, 24, device=dev, dtype=bf)
    seg = Seg(M, False); cst = torch.zeros(2, 5120, 3, device=dev, dtype=bf); h0 = torch.zeros(1, 24, 128, 128, device=dev, dtype=torch.float32)
    Kb = torch.empty(M, 2, 256, device=dev, dtype=bf); Vb = torch.empty(M, 2, 256, device=dev, dtype=bf); cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
    def f():
        for _ in range(8):
            if kind == 'gdn': gdn_core(qout, qout[:, 5120:8192], a, b, GP[0], seg, cst, h0)
            else:
                q, bb = attn_core(qout, Kb, Vb, 0, M, None, None); attn_call(q, Kb, Vb, bb, M, cu, cu)
    g = capture(f)
    ts = [tgraph(g, 10) for _ in range(3)]
    return st.median(ts) * 1000 / 8


def codec_check(rows):
    """eager: codec AR vs engine DMA AR (the champion's bitwise-equal path) on a Marlin-like payload and on randn"""
    out = {}
    for name, gen in (('randn', lambda: torch.randn(rows, 5120, device=dev, dtype=bf)),
                      ('randn_x1e3_tail', lambda: (torch.randn(rows, 5120, device=dev, dtype=bf) * torch.exp(torch.randn(rows, 5120, device=dev) * 2).to(bf)))):
        torch.manual_seed(5 + rank); p = gen()
        ref = dma.all_reduce(p)
        torch.cuda.synchronize(); dist.barrier()
        OVF.zero_(); e = torch.empty(X.enc_bytes(p.numel()) // 2, device=dev, dtype=bf); rx = torch.empty_like(e); o = torch.empty_like(p)
        cb = X.enc_bytes(p.numel()); X.enc(p, e, OVF)
        with torch.cuda.stream(main):
            main.wait_stream(torch.cuda.current_stream())
            fork(main, dA, hA)
            with torch.cuda.stream(dA): X.send(e, 60, 0, cb)
            with torch.cuda.stream(hA): X.recvwait(60); X.recvcopy(rx, 60, 0, cb)
            main.wait_event(rec(hA)); main.wait_event(rec(dA))
            X.dec_add(p, rx, o, OVF)
        torch.cuda.synchronize(); dist.barrier()
        out[name] = dict(mdiff(o, ref), ovf_enc=int(OVF[0]), ovf_dec=int(OVF[1]), ratio=round(cb / (p.numel() * 2), 4))
    return out
for rows in (384, 1024):
    for k, v in codec_check(rows).items(): log(f'CODEC CHECK rows={rows} {k}: {v}')
for R in RS:
    arms = [x for x in ARMS if not (x == 'S2' and R < 256) and not (x == 'S' and R > 192)]
    gr = {}; keep = []
    for x in arms:
        run, reset, kp = build(x, R); keep.append((run, kp)); gr[x] = capture(run)
    if 'S3c' in arms and 'S3cc' in arms and CHECK:
        o = {}
        for x in ('S3c', 'S3cc'):
            run, reset, (r2, resid, outs, kp) = build(x, R, nl=1)
            reset(); torch.cuda.synchronize(); dist.barrier(); OVF.zero_()
            with torch.cuda.stream(main): run()
            torch.cuda.synchronize(); o[x] = (r2.clone(), resid.clone()); dist.barrier()
        log(f'CHAIN CHECK R={R} 1 layer: S3cc vs S3c r2 {mdiff(o["S3cc"][0], o["S3c"][0])} resid {mdiff(o["S3cc"][1], o["S3c"][1])} ovf(enc,dec)={OVF.tolist()}')
    OVF.zero_()
    res = {x: [] for x in arms}
    for rep in range(REPS):
        for x in arms:
            dist.barrier(); res[x].append(tgraph(gr[x], NREP))
    log(f'RESULT R={R} us/layer (median of {REPS} reps x {NREP} replays):', {x: round(st.median(v) * 1000 / NL, 1) for x, v in res.items()}, 'raw', {x: [round(t * 1000 / NL, 1) for t in v] for x, v in res.items()}, 'ovf(enc,dec) in timed runs', OVF.tolist())
    del gr, keep
    torch.cuda.empty_cache()
log('done')
