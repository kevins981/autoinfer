import os, sys, json, statistics, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build
ext = build.get()
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
torch.manual_seed(0)

def make_layer(N, K, wstd=0.04):
    L = torch.nn.Module()
    w = (torch.randn(N, K, device=dev) * wstd).to(torch.float8_e4m3fn)
    L.weight = torch.nn.Parameter(w, requires_grad=False)
    L.weight_scale_inv = torch.nn.Parameter((torch.rand(N // 128, K // 128, device=dev) * 1.5 + 0.5) * 0.01 , requires_grad=False)
    L.output_size_per_partition = N; L.input_size_per_partition = K
    L.weight_block_size = [128, 128]; L.orig_dtype = torch.bfloat16
    prepare_fp8_layer_for_marlin(L, size_k_first=False)
    return L

NG, KG = 17408, 5120      # gate_up per rank
ND, KD = 5120, 8704       # down per rank
Lg = make_layer(NG, KG); Ld = make_layer(ND, KD)
wsp = marlin_make_workspace_new(dev)
print("weight", Lg.weight.shape, Lg.weight.dtype, "scale", Lg.weight_scale_inv.shape, Lg.weight_scale_inv.dtype, flush=True)
Bg = Lg.weight.data; Sg = Lg.weight_scale_inv.data
ws = torch.zeros(8 * (NG // 64) * 3072, dtype=torch.float32, device=dev)
cnt = torch.zeros(NG // 64, dtype=torch.int32, device=dev)

def marlin_g(x): return apply_fp8_marlin_linear(x, Lg.weight, Lg.weight_scale_inv, wsp, NG, KG, None)
def marlin_d(x): return apply_fp8_marlin_linear(x, Ld.weight, Ld.weight_scale_inv, wsp, ND, KD, None)

def ulps(a, b):
    def key(t):
        i = t.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -32768 - i, i)
    return (key(a) - key(b)).abs()

res = {}
DBG=int(os.environ.get('DBG','0'))
TM=torch.zeros(1024*4,dtype=torch.int64,device=dev)
for M in (48, 32, 16):
    x = torch.randn(M, KG, device=dev, dtype=torch.bfloat16) * 1.0
    ref = marlin_g(x)
    y = torch.zeros(M, NG, device=dev, dtype=torch.bfloat16)
    ext.gemm(x, Bg, Sg, y, ws, cnt, DBG, TM); torch.cuda.synchronize()
    y1 = y.clone(); ext.gemm(x, Bg, Sg, y, ws, cnt, DBG, TM); torch.cuda.synchronize()
    u = ulps(y, ref)
    print(f"M={M} exact: max|d|={ (y.float()-ref.float()).abs().max().item():.4g} ref absmax={ref.float().abs().max().item():.3g} "
          f"differ={ (u>0).float().mean().item()*100:.2f}% max_ulp={u.max().item()} >1ulp={(u>1).float().mean().item()*100:.4f}% determ={torch.equal(y,y1)} cnt_ok={int(cnt.abs().sum())==0}", flush=True)

# ---- timing in CUDA graphs with chain: (flush | pf) -> gemm -> down marlin
flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
def spin_us(): pass
def build_graph(M, mode, impl):
    x = torch.randn(M, KG, device=dev, dtype=torch.bfloat16); xd = torch.randn(M, KD, device=dev, dtype=torch.bfloat16)
    y = torch.zeros(M, NG, device=dev, dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    def body():
        if mode == "cold": flush_buf.view(torch.int32).max()
        else: ext.l2pf(Bg, 88 << 20, 8)
        if impl == "mine": ext.gemm(x, Bg, Sg, y, ws, cnt, DBG, TM)
        else: marlin_g(x)
        marlin_d(xd)
    with torch.cuda.stream(s):
        for _ in range(3): body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): body()
    return g

from torch.profiler import profile, ProfilerActivity
def run(graphs, reps):
    out = {k: [] for k in graphs}
    keys = list(graphs)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for r in range(reps):
            for k in keys: graphs[k].replay(); torch.cuda.synchronize()
            prof.step() if False else None
    return prof
sel = [int(a) for a in sys.argv[1:]] or [48, 32, 16]
for M in sel:
    graphs = {}
    for mode in ("cold", "warm"):
        for impl in ("marlin", "mine"):
            graphs[(mode, impl)] = build_graph(M, mode, impl)
    reps = 15
    # profile each graph separately for clean attribution, but interleave replays across reps
    times = {k: [] for k in graphs}
    for r in range(reps):
        for k, g in graphs.items():
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                g.replay(); torch.cuda.synchronize()
            ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            ev.sort(key=lambda e: e.time_range.start)
            gem = [e for e in ev if ("arlin" in e.name or "g2w" in e.name)]
            times[k].append((gem[0].time_range.elapsed_us(), gem[1].time_range.elapsed_us(), [e.name[:30] for e in gem]))
    for k in graphs:
        g1 = sorted(t[0] for t in times[k]); g2 = sorted(t[1] for t in times[k])
        print(f"M={M} {k[0]:5s} {k[1]:6s}: gate_up med {statistics.median(g1):6.1f} us (min {g1[0]:.1f} max {g1[-1]:.1f}) | down med {statistics.median(g2):6.1f} | {times[k][0][2]}", flush=True)
