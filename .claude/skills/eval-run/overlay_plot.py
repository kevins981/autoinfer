"""Draw a run's overlay on the task's answer key (PNG).

usage:
  python overlay_plot.py map    <answer_key.md> <overlay.json> <out.png>
  python overlay_plot.py matrix <answer_key.md> <out.png> <overlay.json> [<overlay.json> ...]
  python overlay_plot.py run    <answer_key.md> <overlay.json> <out.png>
Needs matplotlib (autoinfer/.cache/venv; MPLCONFIGDIR under .cache, never $HOME).
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keygraph import parse

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#d8d7d2"
SURFACE = "#fcfcfb"
# fill by how far the run got (categorical slots 1, 2, 4, 7 of the reference palette)
CONTACT = {
    "reached":        ("#2a78d6", "white", "reached"),
    "tried":          ("#eb6834", "white", "tried"),
    "idea-only":      ("#fbe7b8", INK, "idea only"),
    "not-considered": ("#efeeea", MUTED, "not considered"),
    "novel":          ("#6250d6", "white", "novel (not in key)"),
}
CRIT = "#c62828"
VERDICT = {"missed win": ("✗ missed win", CRIT), "wrong cause": ("! wrong cause", CRIT),
           "new info": ("+ new info", "#6250d6")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def tree_rows(nodes, novel=()):
    """Outline order: (id, depth, first parent). Novel nodes hang under their first parent."""
    kids = {}
    for n, d in nodes.items():
        kids.setdefault(d["builds_on"][0] if d["builds_on"] else "BASE", []).append(n)
    for v in novel:
        kids.setdefault(v["builds_on"][0] if v["builds_on"] else "BASE", []).append(v["id"])
    rows = []
    def walk(n, depth):
        for c in kids.get(n, []):
            rows.append((c, depth, n)); walk(c, depth + 1)
    walk("BASE", 0)
    return rows


def box(ax, x, y, w, h, fill, edge, lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fill, ec=edge, lw=lw, ls=ls, zorder=3))


def header(fig, ov, extra=""):
    c = ov["champion"]
    t = f"{ov['run']}: champion {c['record']} {c['tok_s']} tok/s (+{c['pct']}% vs BASE {ov['base']})"
    r = ov.get("resources", {})
    sub = ""
    if ov.get("best_measured"):
        b = ov["best_measured"]
        sub = f"best measured {b['record']} {b['tok_s']} (+{b['pct']}%, not promoted: ☆);  "
    sub += f"resources: {r.get('wall_h')} h wall, {r.get('gpu_min')} GPU-min, {r.get('output_tokens')} output tokens" + extra
    fig.text(0.02, 0.995, t, ha="left", va="top", fontsize=11, weight="bold", color=INK)
    fig.text(0.02, 0.995 - 0.3 / fig.get_figheight(), sub, ha="left", va="top", fontsize=9, color=INK2)


def legend(fig, y, items):
    handles = [Patch(fc=CONTACT[k][0], ec=GRID, label=CONTACT[k][2]) for k in items]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, y), ncol=5, frameon=False,
               fontsize=8.5, handlelength=1.4, columnspacing=1.2)
    fig.text(0.02, y - 0.3 / fig.get_figheight(), "★ in the champion's stack     ✗ missed win     ! wrong cause"
             "     dashed border = known dead end", fontsize=8.5, color=INK2, va="top")


def plot_map(key_path, ov_path, out):
    nodes = parse(open(key_path).read())
    ov = json.load(open(ov_path))
    novel = ov.get("novel", [])
    rows = tree_rows(nodes, novel)
    champ = set(ov["champion"]["stack"])
    best_adds = set((ov.get("best_measured") or {}).get("adds", [])) - champ
    RH, BW, BH, DX, TOP = 0.42, 2.9, 0.32, 0.32, 1.05
    W = 0.3 + 4 * DX + BW + 3.6
    H = TOP + RH * (len(rows) + 1) + 0.2
    fig = plt.figure(figsize=(W, H), dpi=150); fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    header(fig, ov)
    legend(fig, 1 - 0.55 / H, ["reached", "tried", "idea-only", "not-considered", "novel"])
    y0 = TOP + RH * 0.5
    pos = {"BASE": (0.3, y0)}
    ax.text(0.3, y0, "BASE", va="center", ha="left", weight="bold", color=INK)
    for i, (n, depth, parent) in enumerate(rows):
        x, y = 0.3 + DX * (depth + 1), y0 + RH * (i + 1)
        pos[n] = (x, y)
    xa = 0.3 + 4 * DX + BW + 0.2      # annotation column
    for i, (n, depth, parent) in enumerate(rows):
        x, y = pos[n]
        px, py = pos[parent]
        on_path = n in champ and (parent == "BASE" or parent in champ)
        ax.plot([px + 0.08, px + 0.08, x], [py + BH / 2 if parent != "BASE" else py + 0.1, y, y],
                color=INK if on_path else GRID, lw=2.2 if on_path else 1.0, zorder=1, solid_capstyle="butt")
        if n in nodes:
            d, o = nodes[n], ov["nodes"].get(n, {"contact": "not-considered", "verdict": "-", "records": ""})
            contact = o["contact"]; status = d["status"]; recs = o.get("records", "")
        else:
            v = next(v for v in novel if v["id"] == n)
            contact, status, recs, o = "novel", v.get("title", ""), v.get("result", ""), {"verdict": "new info"}
        fill, txt, _ = CONTACT[contact]
        dead = any(s in status for s in ("no gain", "harmful", "ruled out"))
        box(ax, x, y, BW, BH, fill, INK2 if contact != "not-considered" else GRID, lw=0.8,
            ls="--" if dead and contact in ("not-considered", "idea-only") else "-")
        star = "★ " if n in champ else "☆ " if n in best_adds else ""
        ax.text(x + 0.08, y - 0.045, f"{star}{n}", va="center", ha="left", color=txt, weight="bold", fontsize=8.6, zorder=4)
        extra = ""
        if n in nodes:
            extra = "".join(f"  + {p}" for p in nodes[n]["builds_on"][1:]) + "".join(f"  alt: {a}" for a in nodes[n]["alt"])
        ax.text(x + 0.08, y + 0.095, (status + extra)[:60], va="center", ha="left", color=txt, fontsize=6.8, zorder=4, alpha=0.95)
        vt = VERDICT.get(o.get("verdict"))
        xx = xa
        if vt:
            ax.text(xx, y, vt[0], va="center", ha="left", color=vt[1], weight="bold", fontsize=8.6)
            xx += 1.15
        if recs:
            ax.text(xx, y, recs, va="center", ha="left", color=INK2, fontsize=8)
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)


def plot_matrix(key_path, out, ov_paths):
    nodes = parse(open(key_path).read())
    ovs = [json.load(open(p)) for p in ov_paths]
    rows = tree_rows(nodes)
    RH, NAMEW, COLW, TOP = 0.3, 3.3, 1.9, 1.25
    W = NAMEW + COLW * len(ovs) + 0.3
    H = TOP + RH * len(rows) + 0.3
    fig = plt.figure(figsize=(W, H), dpi=150); fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.text(0.02, 0.995, "Answer key coverage by run", ha="left", va="top", fontsize=11, weight="bold", color=INK)
    legend(fig, 1 - 0.3 / H, ["reached", "tried", "idea-only", "not-considered"])
    for j, ov in enumerate(ovs):
        c = ov["champion"]; cx = NAMEW + COLW * j + 0.1
        ax.text(cx, TOP - 0.28, ov["run"], weight="bold", color=INK, va="center")
        ax.text(cx, TOP - 0.1, f"+{c['pct']}%  ({ov['resources']['gpu_min']} GPU-min)", color=INK2, fontsize=8, va="center")
    for i, (n, depth, parent) in enumerate(rows):
        y = TOP + RH * (i + 0.5)
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.1, y - RH / 2), W - 0.2, RH, fc="#f3f2ee", ec="none", zorder=0))
        st = nodes[n]["status"]
        ax.text(0.2 + 0.22 * depth, y, ("└ " if depth else "") + n, va="center", color=INK, fontsize=8.6,
                weight="bold" if st.startswith("win") else "normal")
        ax.text(NAMEW - 0.1, y, st[:22], va="center", ha="right", color=MUTED, fontsize=7)
        for j, ov in enumerate(ovs):
            o = ov["nodes"].get(n, {"contact": "not-considered", "verdict": "-"})
            fill, txt, label = CONTACT[o["contact"]]
            cx = NAMEW + COLW * j + 0.1
            box(ax, cx, y, 0.95, RH * 0.72, fill, GRID, lw=0.6)
            star = "★" if n in ov["champion"]["stack"] else ""
            ax.text(cx + 0.06, y, f"{star}{label}", va="center", color=txt, fontsize=7, zorder=4)
            vt = VERDICT.get(o.get("verdict"))
            if vt:
                ax.text(cx + 1.0, y, vt[0], va="center", color=vt[1], fontsize=7.2, weight="bold")
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)


def plot_run(key_path, ov_path, out):
    nodes = parse(open(key_path).read())
    ov = json.load(open(ov_path))
    rg = {r["id"]: r for r in ov["run_graph"]}
    kids = {}
    for r in ov["run_graph"]:
        kids.setdefault(r["builds_on"][0], []).append(r["id"])
    rows = []
    def walk(n, d):
        for c in kids.get(n, []):
            rows.append((c, d, n)); walk(c, d + 1)
    walk("BASE", 0)
    missed = [n for n, o in ov["nodes"].items() if o.get("verdict") in ("missed win", "wrong cause")]
    RH, BW, BH, DX, TOP = 0.5, 3.1, 0.4, 0.35, 1.05
    W = 0.3 + DX * 4 + BW + 0.6 + 3.8
    H = TOP + RH * (max(len(rows), len(missed) * 2) + 1) + 0.3
    fig = plt.figure(figsize=(W, H), dpi=150); fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    header(fig, ov, "   |   boxes = the run's records, labeled with the key nodes they match")
    y0 = TOP + RH * 0.5
    pos = {"BASE": (0.3, y0)}
    ax.text(0.3, y0, "BASE", va="center", weight="bold")
    for i, (n, d, p) in enumerate(rows):
        pos[n] = (0.3 + DX * (d + 1), y0 + RH * (i + 1))
    champ = ov["champion"]["record"]
    for n, d, p in rows:
        x, y = pos[n]; px, py = pos[p]
        ax.plot([px + 0.08, px + 0.08, x], [py + (BH / 2 if p != "BASE" else 0.1), y, y], color=GRID, lw=1, zorder=1)
        r = rg[n]
        key_st = [nodes[m]["status"] for m in r["matches"] if m in nodes]
        win = any(s.startswith("win") for s in key_st) and "gate" not in r["result"] and "null" not in r["result"]
        fill, txt = (CONTACT["reached"][0], "white") if n == champ else (("#d6e6f8", INK) if win else ("#efeeea", INK))
        box(ax, x, y, BW, BH, fill, INK2, lw=1.6 if n == champ else 0.7)
        plus = "".join(f" + {p}" for p in r["builds_on"][1:])
        ax.text(x + 0.08, y - 0.07, f"{n}{plus}  {r['result']}", va="center", color=txt, fontsize=8, weight="bold", zorder=4)
        ax.text(x + 0.08, y + 0.1, " + ".join(r["matches"]), va="center", color=txt, fontsize=7.2, zorder=4)
    mx = 0.3 + DX * 4 + BW + 0.6
    ax.text(mx, y0, "Known wins missed / wrong causes", weight="bold", color=CRIT, va="center")
    for i, n in enumerate(missed):
        o = ov["nodes"][n]; y = y0 + RH * (1 + 2 * i)
        box(ax, mx, y, 3.6, BH, "white", CRIT, lw=1.2, ls="--")
        ax.text(mx + 0.08, y - 0.07, n, va="center", color=INK, weight="bold", fontsize=8.4, zorder=4)
        ax.text(mx + 0.08, y + 0.1, f"{VERDICT[o['verdict']][0]}: key says {nodes[n]['status'][:40]}", va="center",
                color=CRIT, fontsize=7.2, zorder=4)
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "map":
        plot_map(*sys.argv[2:5])
    elif mode == "matrix":
        plot_matrix(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif mode == "run":
        plot_run(*sys.argv[2:5])
