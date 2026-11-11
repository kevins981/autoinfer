# Stage 0 microbench: capture GDN chunked-prefill core per (N, NTpad) bucket. One layer, TP=2 rank shapes.
import os, sys, time, json, numpy as np, torch
import torch.nn.functional as F
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule as fla_cgdr
Hk, Hv, D = 8, 24, 128
dev = "cuda"
SD = torch.bfloat16 if os.environ.get("SSM_DT","bf16")=="bf16" else torch.float32
SLOTS = 16; SCRATCH = SLOTS-1
torch.manual_seed(0)

def mk_inputs(T):
    q = F.normalize(torch.randn(1,T,Hk,D,device=dev,dtype=torch.float32),dim=-1).bfloat16()
    k = F.normalize(torch.randn(1,T,Hk,D,device=dev,dtype=torch.float32),dim=-1).bfloat16()
    v = (torch.randn(1,T,Hv,D,device=dev)*0.5).bfloat16()
    g = -(torch.rand(1,T,Hv,device=dev)*0.3+0.01)       # fp32 log-decay
    beta = torch.sigmoid(torch.randn(1,T,Hv,device=dev)).bfloat16()
    return q,k,v,g,beta

def meta_np(lens, NTpad, nreal_slots, has_init):
    """cu_seqlens(N+2), chunk_indices, chunk_offsets, state idx, has_init: all built on host with numpy."""
    N = len(lens); Treal = sum(lens)
    NTreal = sum(-(-l//64) for l in lens); Dn = NTpad-NTreal; assert Dn>=1
    L = list(lens)+[Dn*64]
    cu = np.zeros(len(L)+1, np.int32); cu[1:] = np.cumsum(L)
    cnt = [-(-l//64) for l in L]
    ci = np.concatenate([np.stack([np.full(c,i), np.arange(c)],1) for i,c in enumerate(cnt)]).astype(np.int32)
    co = np.zeros(len(L)+1, np.int32); co[1:] = np.cumsum(cnt)
    idx = np.array(nreal_slots+[SCRATCH], np.int32)
    hi = np.array(list(has_init)+[False], np.int32)
    return cu, ci, co, idx, hi

def eager_core(q,k,v,g,beta, cu,ci,co, idx, hi, ssm, o_out=None):
    """The champion's eager block 2.3 (stock: boolean-mask zeroing; precomputed chunk metadata)."""
    init = ssm[idx]
    init[~hi.bool(), ...] = 0
    o, fs = fla_cgdr(q=q,k=k,v=v,g=g,beta=beta,initial_state=init,output_final_state=True,
        cu_seqlens=cu,chunk_indices=ci,chunk_offsets=co,use_qk_l2norm_in_kernel=False)
    ssm[idx] = fs.to(ssm.dtype)
    return o, fs

def graph_core(q,k,v,g,beta, cu,ci,co, idx, hi, ssm):
    """Capture-safe variant: no boolean-mask indexing (sync); where() instead (bitwise same)."""
    init = torch.where(hi.bool()[:,None,None,None], ssm[idx], torch.zeros((),dtype=ssm.dtype,device=ssm.device))
    o, fs = fla_cgdr(q=q,k=k,v=v,g=g,beta=beta,initial_state=init,output_final_state=True,
        cu_seqlens=cu,chunk_indices=ci,chunk_offsets=co,use_qk_l2norm_in_kernel=False)
    ssm[idx] = fs.to(ssm.dtype)
    return o, fs

class Bucket:
    def __init__(s, N, NTpad, ssm):
        s.N, s.NTpad = N, NTpad
        Tcap = NTpad*64
        s.q,s.k,s.v,s.g,s.beta = mk_inputs(Tcap)   # static input buffers (finite filler for the dummy)
        s.ssm = ssm
        # one static int32 device buffer for all metadata, one pinned host staging buffer, ONE h2d per step
        Nt = N+1
        s.sizes = [Nt+1, NTpad*2, Nt+1, Nt, Nt]
        tot = sum(s.sizes)
        s.dbuf = torch.zeros(tot, dtype=torch.int32, device=dev)
        s.hbuf = torch.zeros(tot, dtype=torch.int32).pin_memory()
        o = 0; views=[]
        for sz in s.sizes: views.append(s.dbuf[o:o+sz]); o+=sz
        s.cu, ci, s.co, s.idx, s.hi = views
        s.ci = ci.view(NTpad,2)
        s.set_meta_np = None
        # warm (compile) then capture
        s.fill([min(64*(NTpad-1), 64)]*1 + [64]*(N-1), [0]*N, list(range(N)))
        for _ in range(2): graph_core(s.q,s.k,s.v,s.g,s.beta,s.cu,s.ci,s.co,s.idx,s.hi,s.ssm)
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        s.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(s.graph):
            s.o, s.fs = graph_core(s.q,s.k,s.v,s.g,s.beta,s.cu,s.ci,s.co,s.idx,s.hi,s.ssm)
        torch.cuda.synchronize()
        s.mem = torch.cuda.max_memory_allocated()-m0
    def fill(s, lens, has_init, slots):
        cu, ci, co, idx, hi = meta_np(lens, s.NTpad, list(slots), has_init)
        flat = np.concatenate([cu, ci.reshape(-1), co, idx, hi])
        s.hbuf.numpy()[:] = flat
        s.dbuf.copy_(s.hbuf, non_blocking=True)

def run_case(buckets, ssm0, lens, NTpad, log):
    N = len(lens); T = sum(lens)
    NTreal = sum(-(-l//64) for l in lens)
    key=(N,NTpad)
    if key not in buckets: buckets[key] = Bucket(N,NTpad,ssm0.clone())
    B = buckets[key]
    slots = [2*i+1 for i in range(N)]; hi = [i%2==0 for i in range(N)]  # mix of with/without initial state
    q,k,v,g,beta = mk_inputs(T)
    # real tokens into static buffers
    for dst,src in ((B.q,q),(B.k,k),(B.v,v),(B.g,g),(B.beta,beta)): dst[:,:T].copy_(src)
    # initial states: random "history" in used slots
    base = (torch.randn_like(ssm0)*0.1).to(ssm0.dtype)
    # --- eager unpadded (stock), with its own copy of the pool
    ssm_e = base.clone()
    cu,ci,co,idx,hi_ = meta_np(lens, NTreal+0 if False else NTreal+1, slots, hi)  # build unpadded:
    # unpadded = real seqs only
    cu_u = torch.tensor(cu[:N+1],device=dev); 
    cnt=[-(-l//64) for l in lens]
    ci_u = torch.tensor(np.concatenate([np.stack([np.full(c,i),np.arange(c)],1) for i,c in enumerate(cnt)]).astype(np.int32),device=dev)
    co_u = torch.tensor(np.concatenate([[0],np.cumsum(cnt)]).astype(np.int32),device=dev)
    idx_u = torch.tensor(slots,dtype=torch.int32,device=dev); hi_u=torch.tensor(hi,dtype=torch.int32,device=dev)
    oe, fse = eager_core(q,k,v,g,beta,cu_u,ci_u,co_u,idx_u.long(),hi_u,ssm_e)
    # --- eager on padded layout (same metadata as graph, no graph)
    B.fill(lens, hi, slots); torch.cuda.synchronize()
    ssm_p = base.clone()
    op, fsp = graph_core(B.q,B.k,B.v,B.g,B.beta,B.cu,B.ci,B.co,B.idx.long(),B.hi,ssm_p)
    # --- captured replay
    B.ssm.copy_(base)
    B.graph.replay(); torch.cuda.synchronize()
    oc, fsc = B.o, B.fs
    def cmp(a,b):
        return bool(torch.equal(a,b)), float((a.float()-b.float()).abs().max())
    r = dict(lens=lens, N=N, NTreal=NTreal, NTpad=NTpad, T=T)
    r["G1_out_graph_vs_eager"] = cmp(oc[:,:T], oe)
    r["G1_out_padeager_vs_eager"] = cmp(op[:,:T], oe)
    r["G1_out_graph_vs_padeager"] = cmp(oc[:,:T], op[:,:T])
    r["G1_fs_graph_vs_eager"] = cmp(fsc[:N], fse)
    r["G1_fs_padeager_vs_eager"] = cmp(fsp[:N], fse)
    real = torch.tensor(slots,device=dev)
    r["G1_ssmslots_graph_vs_eager"] = cmp(B.ssm[real], ssm_e[real])
    other = torch.ones(SLOTS,dtype=torch.bool,device=dev); other[SCRATCH]=False
    r["G1_pool_noscratch_graph_vs_eager"] = cmp(B.ssm[other], ssm_e[other])
    # --- timing
    # host metadata build + h2d
    def mt():
        B.fill(lens, hi, slots)
    n=200
    t0=time.perf_counter()
    for _ in range(n): mt()
    r["host_meta_build_ms"] = (time.perf_counter()-t0)/n*1e3
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(n): B.graph.replay()
    r["host_replay_ms"] = (time.perf_counter()-t0)/n*1e3
    torch.cuda.synchronize()
    # eager host time: launch cost of the stock call (precomputed metadata), queued asynchronously
    ssm_e2 = base.clone(); idxl = idx_u.long()
    for _ in range(3): eager_core(q,k,v,g,beta,cu_u,ci_u,co_u,idxl,hi_u,ssm_e2)
    torch.cuda.synchronize()
    ths=[]
    for _ in range(20):
        torch.cuda.synchronize()
        t0=time.perf_counter(); eager_core(q,k,v,g,beta,cu_u,ci_u,co_u,idxl,hi_u,ssm_e2); ths.append(time.perf_counter()-t0)
    r["host_eager_ms"] = float(np.median(ths))*1e3
    # eager call w/o the boolean-mask (sync) -> same-as-graph body
    ths=[]
    for _ in range(20):
        torch.cuda.synchronize()
        t0=time.perf_counter(); graph_core(q,k,v,g,beta,cu_u,ci_u,co_u,idxl,hi_u,ssm_e2); ths.append(time.perf_counter()-t0)
    r["host_eager_nosync_ms"] = float(np.median(ths))*1e3
    # GPU time via events, back-to-back x20
    def gpu(fn, reps=20):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps): fn()
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1)/reps
    r["gpu_eager_ms"] = gpu(lambda: graph_core(q,k,v,g,beta,cu_u,ci_u,co_u,idxl,hi_u,ssm_e2))
    r["gpu_graph_ms"] = gpu(lambda: B.graph.replay())
    r["gpu_padeager_ms"] = gpu(lambda: graph_core(B.q,B.k,B.v,B.g,B.beta,B.cu,B.ci,B.co,B.idx.long(),B.hi,ssm_p))
    # true GPU time of the unpadded eager call: capture it in a throwaway graph (removes host-bound launch gaps)
    ge = torch.cuda.CUDAGraph()
    ssm_g = base.clone()
    for _ in range(2): graph_core(q,k,v,g,beta,cu_u,ci_u,co_u,idxl,hi_u,ssm_g)
    torch.cuda.synchronize()
    with torch.cuda.graph(ge): graph_core(q,k,v,g,beta,cu_u,ci_u,co_u,idxl,hi_u,ssm_g)
    r["gpu_eager_graphed_ms"] = gpu(lambda: ge.replay(), 50)
    r["gpu_graph_ms"] = gpu(lambda: B.graph.replay(), 50)
    del ge
    r["graph_mem_MB"] = B.mem/2**20
    log(r)

def bucket_for(NTreal, grid):
    for g_ in grid:
        if g_>NTreal: return g_
    return None

if __name__=="__main__":
    out = open(sys.argv[1],"w")
    def log(r):
        print(json.dumps(r)); out.write(json.dumps(r)+"\n"); out.flush()
    ssm0 = torch.zeros(SLOTS,Hv,D,D,dtype=SD,device=dev)
    grid = [4,6,8,12,16,24]
    # (lens) cases: totals 260,300,520,734,1000 for N=1,2,3
    def split(T,N):
        base=[T//N]*N; base[-1]+=T-sum(base); return base
    cases=[]
    for T in (260,300,520,734,1000):
        for N in (1,2,3):
            cases.append(split(T,N))
    buckets={}
    for lens in cases:
        NTreal=sum(-(-l//64) for l in lens)
        NTp = bucket_for(NTreal,[4,6,8,12,16,24,32])
        try: run_case(buckets, ssm0, lens, NTp, log)
        except Exception as e:
            import traceback; traceback.print_exc(); log(dict(lens=lens,error=repr(e)))
