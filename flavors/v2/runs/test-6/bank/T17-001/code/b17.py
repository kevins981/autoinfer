import os, sys, statistics, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build17
ext = build17.get()
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
import vllm; print("vllm:", vllm.__file__, flush=True)
dev = torch.device("cuda:0"); torch.cuda.set_device(dev); torch.manual_seed(0)
def make_layer(N, K):
    L = torch.nn.Module()
    L.weight = torch.nn.Parameter((torch.randn(N, K, device=dev) * 0.04).to(torch.float8_e4m3fn), requires_grad=False)
    L.weight_scale_inv = torch.nn.Parameter((torch.rand(N // 128, K // 128, device=dev) * 1.5 + 0.5) * 0.01, requires_grad=False)
    L.output_size_per_partition = N; L.input_size_per_partition = K
    L.weight_block_size = [128, 128]; L.orig_dtype = torch.bfloat16
    prepare_fp8_layer_for_marlin(L, size_k_first=False); return L
SHAPES = {"gate_up": (17408, 5120), "qkvz": (8192, 5120)}
sel = os.environ.get("SHAPES", "gate_up").split(",")
GRID = int(os.environ.get("GRID", "128"))
wsp = marlin_make_workspace_new(dev, max_blocks_per_sm=4)
ws = torch.zeros(256 * 32 * 512, dtype=torch.float32, device=dev); flags = torch.zeros(256, dtype=torch.int32, device=dev)
def ulps(a, b):
    def key(t):
        i = t.view(torch.int16).to(torch.int32); return torch.where(i < 0, -32768 - i, i)
    return (key(a) - key(b)).abs()
flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
from torch.profiler import profile, ProfilerActivity
for name in sel:
    N, K = SHAPES[name]; L = make_layer(N, K); Bw = L.weight.data; Sw = L.weight_scale_inv.data
    wbytes = Bw.numel() * Bw.element_size(); pf = min(40 << 20, wbytes // 2)
    print(name, "weight", tuple(Bw.shape), Bw.dtype, "scale", tuple(Sw.shape), Sw.dtype, "bytes", wbytes, flush=True)
    for M in (1, 16, 32, 48, 64):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ref = apply_fp8_marlin_linear(x, L.weight, L.weight_scale_inv, wsp, N, K, None)
        y = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
        ext.gemm(x, Bw, Sw, y, ws, flags, GRID); torch.cuda.synchronize()
        y1 = y.clone(); ext.gemm(x, Bw, Sw, y, ws, flags, GRID); torch.cuda.synchronize()
        u = ulps(y, ref)
        print(f"EXACT {name} M={M}: max|d|={(y.float()-ref.float()).abs().max().item():.3g} absmax={ref.float().abs().max().item():.3g} differ={(u>0).float().mean().item()*100:.3f}% max_ulp={u.max().item()} >1ulp={(u>1).float().mean().item()*100:.4f}% determ={torch.equal(y,y1)} flags_clear={int(flags.abs().sum())==0}", flush=True)
    def build_graph(M, mode, impl):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16); y = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
        s = torch.cuda.Stream(); s2 = torch.cuda.Stream()
        def body():
            if mode == "cold": flush_buf.view(torch.int32).max()
            elif mode == "warm": ext.l2pf(Bw, pf, 8)
            elif mode == "chain":   # flush, then AR-like 45us kernel with concurrent 40MB evict_last prefetch
                flush_buf.view(torch.int32).max()
                ev0 = torch.cuda.Event(); ev0.record(torch.cuda.current_stream())
                s2.wait_event(ev0)
                with torch.cuda.stream(s2): ext.l2pf(Bw, pf, 8)
                ext.spin(45000)
                torch.cuda.current_stream().wait_stream(s2)
            if impl == "mine": ext.gemm(x, Bw, Sw, y, ws, flags, GRID)
            else: apply_fp8_marlin_linear(x, L.weight, L.weight_scale_inv, wsp, N, K, None)
        with torch.cuda.stream(s):
            for _ in range(3): body()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s): body()
        return g
    for M in (48, 64):
        graphs = {(m, i): build_graph(M, m, i) for m in ("cold", "warm", "chain") for i in ("marlin", "mine")}
        times = {k: [] for k in graphs}
        for r in range(17):
            for k, g in graphs.items():
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    g.replay(); torch.cuda.synchronize()
                ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and ("arlin" in e.name or "g17" in e.name)]
                times[k].append(ev[-1].time_range.elapsed_us())
        for k in graphs:
            t = sorted(times[k]); print(f"TIME {name} M={M} {k[0]:5s} {k[1]:6s}: med {statistics.median(t):6.1f} us (min {t[0]:.1f} max {t[-1]:.1f})", flush=True)
