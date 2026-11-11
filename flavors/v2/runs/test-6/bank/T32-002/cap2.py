# Follow-up: exact (N,NTreal) keyed graphs, no dummy; (A) core only, (B) whole prefill block
# (flush + conv + fused_post_conv_prep + core + state writes). Shared graph pool.
import os, sys, time, json, numpy as np, torch
import torch.nn.functional as F
from types import SimpleNamespace
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule as fla_cgdr
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.model_executor.layers.mamba.gdn.replay import gdn_red8_flush_bit, get_stats_tensor
Hk, Hv, D = 8, 24, 128
QKV = 2*Hk*D + Hv*D   # 5120
QKVZ = QKV + Hv*D     # 8192 (row stride of the split view)
dev = "cuda"; SLOTS = 16; PAD=-1
torch.manual_seed(0)
POOL = torch.cuda.graph_pool_handle()

class Layer:   # weights / states of one layer
    def __init__(s):
        s.cw = (torch.randn(QKV,4,device=dev)*0.4).bfloat16()
        s.A_log = torch.randn(Hv,device=dev)*0.5; s.dt_bias = torch.randn(Hv,device=dev)*0.5
        s.ssm = torch.zeros(SLOTS,Hv,D,D,device=dev,dtype=torch.bfloat16)
        s.conv = torch.zeros(SLOTS,QKV,3,device=dev,dtype=torch.bfloat16)
        s.flags = torch.zeros(SLOTS,Hv,dtype=torch.uint8,device=dev)
        s.sp = torch.tensor([s.ssm.data_ptr()],dtype=torch.int64,device=dev)
        s.fp = torch.tensor([s.flags.data_ptr()],dtype=torch.int64,device=dev)
    def reset(s, ssm0, conv0):
        s.ssm.copy_(ssm0); s.conv.copy_(conv0); s.flags.zero_()

def flush(L, keys):
    torch.ops._C.gdn_replay_flush(L.sp, L.fp, L.ssm.stride(0), Hv, keys, keys, keys, keys, keys,
        keys.numel(), 3 | gdn_red8_flush_bit(), 0, 0, get_stats_tensor(L.ssm.device))

def core(L, q,k,v,g,beta, cu,ci,co, idx, hi, out, bool_mask):
    init = L.ssm[idx]
    if bool_mask: init[~hi.bool(), ...] = 0          # stock (host-sync) form
    else: init = torch.where(hi.bool()[:,None,None,None], init, torch.zeros((),dtype=init.dtype,device=dev))
    o, fs = fla_cgdr(q=q,k=k,v=v,g=g,beta=beta,initial_state=init,output_final_state=True,
        cu_seqlens=cu,chunk_indices=ci,chunk_offsets=co,use_qk_l2norm_in_kernel=False)
    L.ssm[idx] = fs.to(L.ssm.dtype)
    return o, fs

def block(L, qkvz, ba, cu, ci, co, idx, hi, md, out, bool_mask, flush_keys):
    """whole prefill block as the champion runs it (glue path, prefill-only part)"""
    if flush_keys is not None: flush(L, flush_keys)
    mixed = qkvz[:, :QKV]
    xT = mixed.transpose(0,1)
    conv_out = causal_conv1d_fn(xT, L.cw, None, activation="silu", conv_states=L.conv,
        has_initial_state=hi.bool(), cache_indices=idx, query_start_loc=cu, metadata=md,
        out=torch.empty(mixed.shape,dtype=mixed.dtype,device=dev).transpose(0,1)).transpose(0,1)
    b = ba[:, :Hv]; a = ba[:, Hv:]
    q,k,v,g,beta = fused_post_conv_prep(conv_output=conv_out,a=a,b=b,A_log=L.A_log,dt_bias=L.dt_bias,
        num_k_heads=Hk,head_k_dim=D,head_v_dim=D,apply_l2norm=True,output_g_exp=False)
    q,k,v,g,beta = q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),g.unsqueeze(0),beta.unsqueeze(0)
    o, fs = core(L,q,k,v,g,beta,cu,ci,co,idx.long() if bool_mask else idx,hi,out,bool_mask)
    out.copy_(o.squeeze(0))   # stands in for core_attn_out in-place write
    return o

def meta_np(lens, NT):
    N=len(lens)
    cu=np.zeros(N+1,np.int32); cu[1:]=np.cumsum(lens)
    cnt=[-(-l//64) for l in lens]
    ci=np.concatenate([np.stack([np.full(c,i),np.arange(c)],1) for i,c in enumerate(cnt)]).astype(np.int32)
    co=np.zeros(N+1,np.int32); co[1:]=np.cumsum(cnt)
    nums=[-(-l//8) for l in lens]
    tot=sum(nums); totpad=8*NT      # tot <= 8*NT
    mlist=np.full(totpad,PAD,np.int32); mlist[:tot]=np.repeat(np.arange(N),nums)
    off=np.full(totpad,PAD,np.int32); off[:tot]=np.concatenate([np.arange(n) for n in nums])
    return cu,ci,co,mlist,off,tot,totpad

class Bucket:
    def __init__(s, kind, L, N, NT, lens0):
        s.kind,s.L,s.N,s.NT=kind,L,N,NT; Tc=NT*64; s.Tc=Tc
        # static input buffers (producer pieces would write here)
        s.qkvz=(torch.randn(Tc,QKVZ,device=dev)*0.5).bfloat16(); s.ba=torch.randn(Tc,2*Hv,device=dev).bfloat16()
        s.q,s.k,s.v,s.g,s.beta=mk_core_inputs(Tc)
        s.out=torch.zeros(Tc,Hv,D,device=dev,dtype=torch.bfloat16)
        totpad=8*NT
        s.sizes=[N+1,NT*2,N+1,N,N,totpad,totpad]
        s.dbuf=torch.zeros(sum(s.sizes),dtype=torch.int32,device=dev)
        s.hbuf=torch.zeros(sum(s.sizes),dtype=torch.int32).pin_memory()
        o=0; vs=[]
        for sz in s.sizes: vs.append(s.dbuf[o:o+sz]); o+=sz
        s.cu,ci,s.co,s.idx,s.hi,s.mlist,s.off=vs; s.ci=ci.view(NT,2)
        s.md=SimpleNamespace(nums_dict={8:dict(tot=totpad,mlist=None,mlist_len=totpad,offsetlist=None,
              batch_ptr=s.mlist,token_chunk_offset_ptr=s.off)},batch_ptr=s.mlist,token_chunk_offset_ptr=s.off)
        s.fill(lens0,list(range(1,N+1)),[False]*N)
        fk = s.idx
        def body(): 
            if kind=="core": return core(L,s.q,s.k,s.v,s.g,s.beta,s.cu,s.ci,s.co,s.idx,s.hi,s.out,False)
            return block(L,s.qkvz,s.ba,s.cu,s.ci,s.co,s.idx,s.hi,s.md,s.out,False,fk)
        s.body=body
        for _ in range(2): body()
        torch.cuda.synchronize()
        s.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(s.graph,pool=POOL): s.res=body()
        torch.cuda.synchronize()
    def fill(s,lens,slots,hi):
        cu,ci,co,ml,off,tot,totpad=meta_np(lens,s.NT)
        s.hbuf.numpy()[:]=np.concatenate([cu,ci.reshape(-1),co,np.array(slots,np.int32),np.array(hi,np.int32),ml,off])
        s.dbuf.copy_(s.hbuf,non_blocking=True)

def mk_core_inputs(T):
    q=F.normalize(torch.randn(1,T,Hk,D,device=dev),dim=-1).bfloat16()
    k=F.normalize(torch.randn(1,T,Hk,D,device=dev),dim=-1).bfloat16()
    v=(torch.randn(1,T,Hv,D,device=dev)*0.5).bfloat16()
    g=-(torch.rand(1,T,Hv,device=dev)*0.3+0.01)
    beta=torch.sigmoid(torch.randn(1,T,Hv,device=dev)).bfloat16()
    return q,k,v,g,beta

def tm_host(fn,n=20):
    ts=[]
    for _ in range(n):
        torch.cuda.synchronize(); t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return float(np.median(ts))*1e3
def tm_gpu(fn,reps=50):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1)/reps
def cmp(a,b): return [bool(torch.equal(a,b)), float((a.float()-b.float()).abs().max())]

def run_case(kind, lens, Le, Lg, buckets, ssm0, conv0, log):
    N=len(lens); T=sum(lens); NT=sum(-(-l//64) for l in lens)
    key=(kind,N,NT)
    if key not in buckets: buckets[key]=Bucket(kind,Lg,N,NT,lens)
    B=buckets[key]
    slots=[2*i+1 for i in range(N)]; hi=[i%2==0 for i in range(N)]
    # inputs: write the real rows into the static buffers (rows >= T keep filler)
    if kind=="core":
        q,k,v,g,beta=mk_core_inputs(T)
        for d,s_ in ((B.q,q),(B.k,k),(B.v,v),(B.g,g),(B.beta,beta)): d[:,:T].copy_(s_)
    else:
        qkvz=(torch.randn(T,QKVZ,device=dev)*0.5).bfloat16(); ba=torch.randn(T,2*Hv,device=dev).bfloat16()
        B.qkvz[:T].copy_(qkvz); B.ba[:T].copy_(ba)
    # eager, unpadded, stock (bool mask; metadata as the builder gives it)
    cu,ci,co,ml,off,tot,totpad=meta_np(lens,NT)
    t=lambda x:torch.tensor(x,device=dev)
    cu_u,ci_u,co_u=t(cu),t(ci),t(co); idx_u=t(np.array(slots,np.int32)); hi_u=t(np.array(hi,np.int32))
    keys=idx_u
    nums=[-(-l//8) for l in lens]
    md_u=SimpleNamespace(nums_dict={8:dict(tot=tot,mlist=None,mlist_len=tot,offsetlist=None,
         batch_ptr=t(np.concatenate([ml[:tot],np.full(1024,PAD,np.int32)])),
         token_chunk_offset_ptr=t(np.concatenate([off[:tot],np.full(1024,PAD,np.int32)])))})
    md_u.batch_ptr=md_u.nums_dict[8]["batch_ptr"]; md_u.token_chunk_offset_ptr=md_u.nums_dict[8]["token_chunk_offset_ptr"]
    out_e=torch.zeros(T,Hv,D,device=dev,dtype=torch.bfloat16)
    def eager(Lx):
        if kind=="core": return core(Lx,q,k,v,g,beta,cu_u,ci_u,co_u,idx_u.long(),hi_u,out_e,True)
        return block(Lx,qkvz,ba,cu_u,ci_u,co_u,idx_u,hi_u,md_u,out_e,True,keys)
    # hmm: for the block, hi must be bool for conv; block() does hi.bool()
    base_s=(torch.randn_like(ssm0)*0.1).to(ssm0.dtype); base_c=(torch.randn_like(conv0)*0.5).to(conv0.dtype)
    Le.reset(base_s,base_c); eager(Le); torch.cuda.synchronize()
    Lg.reset(base_s,base_c); B.fill(lens,slots,hi); torch.cuda.synchronize()
    B.graph.replay(); torch.cuda.synchronize()
    r=dict(kind=kind,lens=lens,N=N,NT=NT,T=T)
    # outputs
    # recompute eager outputs again for compare (state already consumed): rerun on fresh state
    Le.reset(base_s,base_c); oe=eager(Le); torch.cuda.synchronize()
    Lg.reset(base_s,base_c); B.graph.replay(); torch.cuda.synchronize()
    og=B.res[0] if kind=="core" else B.res
    if isinstance(oe,tuple): oe=oe[0]
    og=og if og.dim()==3 else og
    oe_=oe[0] if oe.dim()==4 else oe; og_=og[0] if og.dim()==4 else og
    r["G1_out"]=cmp(og_[:T],oe_[:T])
    rs=torch.tensor(slots,device=dev)
    r["G1_ssm_realslots"]=cmp(Lg.ssm[rs],Le.ssm[rs]); r["G1_ssm_pool"]=cmp(Lg.ssm,Le.ssm)
    r["G1_conv_realslots"]=cmp(Lg.conv[rs],Le.conv[rs]); r["G1_conv_pool"]=cmp(Lg.conv,Le.conv)
    r["G1_flags"]=cmp(Lg.flags,Le.flags)
    # host
    n=100
    t0=time.perf_counter()
    for _ in range(n): B.fill(lens,slots,hi)
    r["host_meta_ms"]=(time.perf_counter()-t0)/n*1e3
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): B.graph.replay()
    r["host_replay_ms"]=(time.perf_counter()-t0)/n*1e3; torch.cuda.synchronize()
    # eager host: stock (bool mask => includes a sync inside) and no-sync variant launch cost
    r["host_eager_ms"]=tm_host(lambda: eager(Le))
    # GPU time: eager captured in a throwaway graph (stock bool-mask form is not capturable -> where() form)
    ge=torch.cuda.CUDAGraph(); Lx=Le
    def eager_ns():
        if kind=="core": return core(Lx,q,k,v,g,beta,cu_u,ci_u,co_u,idx_u,hi_u,out_e,False)
        return block(Lx,qkvz,ba,cu_u,ci_u,co_u,idx_u,hi_u,md_u,out_e,False,keys)
    for _ in range(2): eager_ns()
    torch.cuda.synchronize()
    with torch.cuda.graph(ge): eager_ns()
    r["gpu_eager_ms"]=tm_gpu(ge.replay); r["gpu_graph_ms"]=tm_gpu(B.graph.replay)
    r["gpu_pct"]=(r["gpu_graph_ms"]/r["gpu_eager_ms"]-1)*100
    r["host_eager_nosync_ms"]=tm_host(eager_ns)
    del ge
    log(r)

if __name__=="__main__":
    out=open(sys.argv[1],"w")
    def log(r): print(json.dumps(r),flush=True); out.write(json.dumps(r)+"\n"); out.flush()
    Le,Lg=Layer(),Layer(); Lg.cw=Le.cw; Lg.A_log=Le.A_log; Lg.dt_bias=Le.dt_bias
    ssm0=torch.zeros_like(Le.ssm); conv0=torch.zeros_like(Le.conv)
    def split(T,N):
        b=[T//N]*N; b[-1]+=T-sum(b); return b
    cases=[split(T,N) for T in (260,300,520,734,1000) for N in (1,2,3)]
    buckets={}; m0=torch.cuda.memory_reserved()
    for kind in sys.argv[2].split(","):
        for lens in cases:
            try: run_case(kind,lens,Le,Lg,buckets,ssm0,conv0,log)
            except Exception as e:
                import traceback; traceback.print_exc(); log(dict(kind=kind,lens=lens,error=repr(e)))
        log(dict(kind=kind,graphs=len([k for k in buckets if k[0]==kind]),reserved_MB_after=(torch.cuda.memory_reserved()-m0)/2**20))
