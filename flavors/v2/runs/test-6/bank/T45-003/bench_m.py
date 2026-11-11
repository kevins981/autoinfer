import os, sys, json, time, statistics, importlib.util
import torch
sys.path.insert(0, "/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-35.w1/ms1/build")
import ms1ext as X
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import t31_cfg
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.scalar_type import scalar_types

dev = torch.device("cuda:0")
M = int(os.environ.get("M", "48"))
MB = 1 << 20
NL = int(os.environ.get("NL", "64"))           # distinct layers of weights
GL = NL // 2                                    # layers per graph
OUT = os.environ.get("OUT", "res.json")
SHAPES = {"qkvz": (8192, 5120), "gdn_out": (5120, 3072), "gate_up": (17408, 5120), "down": (5120, 8704)}
W = {k: [] for k in SHAPES}
WS = {k: [] for k in SHAPES}
for k, (N, K) in SHAPES.items():
    for l in range(NL):
        W[k].append(torch.empty(N * K // 4, dtype=torch.int32, device=dev).random_())
        if k == "gate_up":
            W[k][-1] = W[k][-1].view(K // 16, N * 4)
            WS[k].append((torch.rand(K // 128, N, device=dev) * 0.02 + 0.001).to(torch.bfloat16))
        else:
            WS[k].append(torch.full((N // 128, K // 128), 0.01, device=dev, dtype=torch.float32))
nbytes = {k: W[k][0].numel() * 4 for k in W}
GDN_BYTES = 12_500_000 // 16 * 16
GS = [(torch.empty(GDN_BYTES // 4, dtype=torch.int32, device=dev).random_(), torch.empty(GDN_BYTES // 4, dtype=torch.int32, device=dev)) for _ in range(NL)]
act = {}
for k in ("qkvz", "gdn_out", "down"):
    N, K = SHAPES[k]
    xq = torch.randn(M, K, device=dev).clamp(-4, 4).to(torch.float8_e4m3fn)
    xs = torch.full((M, K // 128), 0.05, device=dev, dtype=torch.float32)
    allc = {t31_cfg.pick(N, K, M, r) for r in ("cold", "mid", "r48", "warm")}
    ws = torch.zeros(max(max(t31_cfg.ws_numel(N, K, M, c) for c in allc), 1), device=dev, dtype=torch.float32)
    cnt = torch.zeros(4096, device=dev, dtype=torch.int32)
    act[k] = (xq, xs, ws, cnt)
xg = torch.randn(M, 5120, device=dev, dtype=torch.bfloat16)
marlin_ws = marlin_make_workspace_new(dev, 4)

def t31(k, l, regime):
    N, K = SHAPES[k]
    xq, xs, ws, cnt = act[k]
    return ops.t31_gemm_e4m3(xq, xs, W[k][l].view(torch.uint8) if False else W[k][l], WS[k][l], ws, cnt, M, N, K, t31_cfg.pick(N, K, M, regime))

def marlin(l):
    N, K = SHAPES["gate_up"]
    return torch.ops._C.marlin_gemm(xg, None, W["gate_up"][l], None, WS["gate_up"][l], None, None, marlin_ws,
                                    scalar_types.float8_e4m3fn.id, M, N, K, False, True, False)

def gemm(k, l, regime):
    return marlin(l) if k == "gate_up" else t31(k, l, regime)

def gp(k, l):
    return W[k][l].data_ptr()

# ---------------- standalone cold/warm per GEMM ----------------
flush = torch.empty(512 * MB // 4, dtype=torch.int32, device=dev)
def timeit(fn, pre=None, reps=15):
    ts = []
    for _ in range(reps):
        if pre: pre()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts), min(ts)
def fl(): flush.add_(1)
res = {"standalone": {}}
if os.environ.get("STANDALONE", "1") == "1":
    for k in SHAPES:
        regs = ["cold"] if k == "gate_up" else ["cold", "mid", "r48", "warm"]
        for r in regs:
            l = [0]
            def run(): gemm(k, l[0] % NL, r)
            def pre_cold():
                l[0] += 1; fl()
            # cold: flush then run (distinct layer each time)
            for _ in range(3): pre_cold(); run()
            torch.cuda.synchronize()
            cold = timeit(run, pre_cold)
            def run2():
                run(); 
            # warm: run once (untimed) then timed
            ts = []
            for _ in range(15):
                l[0] += 1; fl(); run()
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); run(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
            # warm via prefetch stream kernel (ld mode, whole weight, up to 64MB) -> "L2-resident" proxy
            res["standalone"][f"{k}/{r}"] = {"cold_med": cold[0], "cold_min": cold[1], "warm_med": statistics.median(ts), "warm_min": min(ts)}
            print(f"standalone {k:8s} cfg-regime={r:5s} cold {cold[0]:7.1f} (min {cold[1]:7.1f})  warm {statistics.median(ts):7.1f} (min {min(ts):7.1f}) us", flush=True)
    # "fully prefetched by ld kernel then GEMM" (what the streamer can achieve at best), per GEMM
    for k in SHAPES:
        regs = ["cold"] if k == "gate_up" else ["cold", "warm"]
        for r in regs:
            for cap_mb in (32, 48, 9999):
                cap = min(nbytes[k], cap_mb * MB)
                ts = []
                for i in range(12):
                    l = i + 1; fl()
                    X.stream_pf(gp(k, l), cap, 64, 1, 8, 32)   # ld mode, 64 blocks
                    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record(); gemm(k, l, r); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
                res["standalone"][f"{k}/{r}/preld{cap_mb}"] = statistics.median(ts)
                print(f"standalone {k:8s} cfg={r:5s} after ld-prefetch {min(cap_mb, nbytes[k]//MB):4d}MB: {statistics.median(ts):7.1f} us", flush=True)

# ---------------- layer chain in CUDA graphs ----------------
OPS = ["ar1", "norm1", "small", "qkvz", "conv", "gdn", "quant", "gdn_out", "ar2", "norm2", "gate_up", "silu", "down"]
GEMMS = ["qkvz", "gdn_out", "gate_up", "down"]
DEF_START = {"qkvz": "down", "gdn_out": "qkvz", "gate_up": "gdn_out", "down": "gate_up"}
side = torch.cuda.Stream()
def regime_for(k, capb, auto):
    if auto != "auto": return auto
    c = min(capb, nbytes[k])
    if c >= nbytes[k] - 1: return "warm"
    if c >= 44 * MB: return "r48"
    if c >= 16 * MB: return "mid"
    return "cold"

def build_graph(v, base):
    """v: dict(ar=(t1,t2), mode='champ'|'none'|'stream', ...). layers base..base+GL-1."""
    g = torch.cuda.CUDAGraph()
    ar1, ar2 = v["ar"]
    mode = v["mode"]
    sm = v.get("small", {"norm1": 3, "small": 4, "conv": 4, "quant": 1, "norm2": 3.5, "silu": 2.5})
    seq = [(o, base + i) for i in range(GL) for o in OPS]
    forks = {}
    if mode == "stream":
        st = dict(DEF_START); st.update(v.get("start", {}))
        for t, (o, l) in enumerate(seq):
            if o in GEMMS:
                j = t - 1
                while j >= 0 and seq[j][0] != st[o]: j -= 1
                forks.setdefault(max(j, 0) if j >= 0 else 0, []).append(t)
    cap = int(v.get("cap_mb", 48) * MB)
    reg = v.get("reg", "auto")
    with torch.cuda.graph(g):
        cur = torch.cuda.current_stream()
        for t, (o, l) in enumerate(seq):
            if t in forks:
                ev = torch.cuda.Event(); ev.record(cur); side.wait_event(ev)
                with torch.cuda.stream(side):
                    for tt in forks[t]:
                        k, ll = seq[tt]
                        n = min(nbytes[k], cap)
                        X.stream_pf(gp(k, ll), n, v["blocks"], 0 if v["smode"] == "hint" else 1, 8, v.get("step", 32))
            if o == "ar1":
                if mode == "champ":
                    X.ar_pf(ar1, gp("qkvz", l), min(nbytes["qkvz"], 88 * MB), 0, 0, 8, 0, 0, 32)
                else:
                    X.spin(ar1)
            elif o == "ar2":
                if mode == "champ":
                    X.ar_pf(ar2, gp("gate_up", l), min(nbytes["gate_up"], 88 * MB), 0, 0, 8, 0, 0, 32)
                else:
                    X.spin(ar2)
            elif o == "gdn":
                X.gdn(GS[l][0].data_ptr(), GS[l][1].data_ptr(), GDN_BYTES, v.get("gdn_us", 37))
            elif o in GEMMS:
                if mode == "champ":
                    r = "warm" if o == "qkvz" else "cold"
                elif mode == "none":
                    r = "cold"
                else:
                    r = regime_for(o, cap, reg)
                gemm(o, l, r)
            else:
                X.spin(sm[o])
        if mode == "stream":
            ev = torch.cuda.Event(); ev.record(side); cur.wait_event(ev)
    return g

def run_variant(v, reps=5, inner=3):
    ga, gb = build_graph(v, 0), build_graph(v, GL)
    for _ in range(2): ga.replay(); gb.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(inner): ga.replay(); gb.replay()
        b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000 / (inner * 2 * GL))
    return statistics.median(out), min(out), max(out), (ga, gb)

def profile_variant(v, graphs):
    from torch.profiler import profile, ProfilerActivity
    ga, gb = graphs
    gb.replay(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        gb.replay(); torch.cuda.synchronize()
    ev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return ev

def summarize_prof(ev, v):
    # main-chain kernels = not stream_kernel; per-layer index mod 13
    rows = sorted([(e.time_range.start, e.time_range.end, e.name) for e in ev])
    main = [r for r in rows if "stream_kernel" not in r[2]]
    n = len(main)
    if n < 13 * GL: return {"err": f"main kernels {n} < {13*GL}"}
    per = {o: [] for o in OPS}
    for i, r in enumerate(main[:13 * GL]):
        if i // 13 in (0, GL - 1): continue
        per[OPS[i % 13]].append(r[1] - r[0])
    stm = [r[1] - r[0] for r in rows if "stream_kernel" in r[2]]
    d = {o: round(statistics.mean(x), 1) for o, x in per.items()}
    d["stream_kernel_mean_dur"] = round(statistics.mean(stm), 1) if stm else None
    d["span_us"] = round((main[-1][1] - main[0][0]) / GL, 1)
    return d

variants = []
def add(name, **kw): variants.append((name, kw))
AR = (12, 12)
add("V0b champ AR39/50 + in-AR pf", mode="champ", ar=(39, 50))
add("V1b serial no-pf AR39/50 cold", mode="none", ar=(39, 50))
res["variants"] = {}
for name, kw in variants:
    t0 = time.time()
    med, mn, mx, graphs = run_variant(kw)
    r = {"med": med, "min": mn, "max": mx}
    if os.environ.get("PROF", "1") == "1" and (name.startswith(("V0", "V1")) or med < 470):
        try:
            r["prof"] = summarize_prof(profile_variant(kw, graphs), kw)
        except Exception as e:
            r["prof"] = {"err": repr(e)}
    res["variants"][name] = r
    print(f"{name:75s} {med:7.1f} us/layer (min {mn:7.1f} max {mx:7.1f}) [{time.time()-t0:.1f}s]", flush=True)
    if "prof" in r: print("     prof:", json.dumps(r["prof"]), flush=True)
    del graphs
    json.dump(res, open(OUT, "w"), indent=1)
