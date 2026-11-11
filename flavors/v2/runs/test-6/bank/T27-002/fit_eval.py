# Fit PCA screen on TRAIN self-generated seqs; evaluate recall + draft acceptance on HELD-OUT seqs (CPU).
import sys, json, torch, time
sys.path.insert(0, "/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-3.w1")
import mtp_model as mm
torch.set_num_threads(24)
D3 = "/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-3.w1/data"
OUT = "/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-27/lr"
split = json.load(open(f"{D3}/split.json"))
m = mm.MTP(mm.load_weights(torch.float32)); W = m.lm_head.float()
def run(sids):
    R = []
    with torch.no_grad():
        for sid in sids:
            d = torch.load(f"{D3}/seq/{sid}.pt"); H = d["hidden"].float(); ids = d["ids"]; P = int(d["P"])
            o1, K1, V1 = m.step1(H, ids); d1 = (o1 @ W.T).argmax(-1)
            o2 = m.step2(o1[:-1], d1[:-1], K1[:-1], V1[:-1])
            R.append(dict(o1=o1, o2=o2, ids=ids, P=P))
    return R
t0 = time.time()
ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 40
TR = run(split["train"][:ntr]); print("train done", time.time() - t0, flush=True)
TE = run(split["heldout"]); print("heldout done", time.time() - t0, flush=True)
X = torch.cat([torch.cat([r["o1"], r["o2"]]) for r in TR])
C = X.T @ X / X.shape[0]; ev, Vh = torch.linalg.eigh(C.double()); Vh = Vh.flip(1).float()
torch.save({"Vh": Vh[:, :1024].bfloat16(), "n_train_pos": X.shape[0], "train_ids": split["train"][:ntr]}, f"{OUT}/pca_train_n{ntr}.pt")
def drafts(o, Vr=None, U=None, k=64):
    ex = o @ W.T
    if Vr is None: return ex.argmax(-1)
    ap = (o @ Vr) @ U.T; idx = ap.topk(k, -1).indices
    return idx.gather(1, ex.gather(1, idx).argmax(-1, keepdim=True)).squeeze(1)
def accept(cfg):
    a1 = a2 = n = 0
    for r in TE:
        ids, P = r["ids"], r["P"]; L = ids.shape[0]
        d1 = drafts(r["o1"], *cfg)          # pos i -> predicts ids[i+2]
        d2 = drafts(r["o2"], *cfg)          # pos i (uses exact d1 recursion) -> predicts ids[i+3]
        for i in range(max(P - 1, 0), L - 3):
            n += 1; ok1 = int(d1[i] == ids[i + 2]); a1 += ok1; a2 += ok1 * int(d2[i] == ids[i + 3])
    return a1 / n, a2 / n, n
print("exact      a1 %.4f a2(cum) %.4f n=%d" % accept((None, None)), flush=True)
for r in (768, 1024):
    Vr = Vh[:, :r]; U = W @ Vr
    for k in (64, 128):
        Ote = torch.cat([torch.cat([t["o1"], t["o2"]]) for t in TE])
        top = (Ote @ W.T).argmax(-1); ap = (Ote @ Vr) @ U.T
        rec = (ap.topk(k, -1).indices == top[:, None]).any(-1).float().mean().item()
        print(f"r={r} k={k} recall {rec:.4f}  a1 %.4f a2 %.4f n=%d" % accept((Vr, U, k)), flush=True)
print("done", time.time() - t0)
