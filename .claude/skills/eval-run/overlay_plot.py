"""Draw a run's overlay on the task's answer key (PNG): the key as a tree, one row per node.

usage: python overlay_plot.py <answer_key.md> <overlay.json> <out.png>
Needs matplotlib (autoinfer/.cache/venv; MPLCONFIGDIR under .cache, never $HOME).

Win nodes are colored by the last stage the run passed (EVAL.md, "Grading a run against the key"):
none / idea / implementation / result. A win node that didn't pass `result` (and whose alternative
didn't either) is a missed win (red ✗). Dead-end nodes have a dashed border and are graded by the
run's reason: right / wrong / not considered. Novel nodes (measured changes not in the key) are
purple, under their parent. ★ = in the champion's stack; ☆ = in the best measured stack only.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keygraph import parse

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#d8d7d2", "#fcfcfb"
CRIT, GOOD = "#c62828", "#2e7d32"
# win nodes: fill by the last stage passed (categorical slots 1, 2, 4 of the reference palette + gray)
STAGE = {
    "result":         ("#2a78d6", "white", "known result reached"),
    "implementation": ("#eb6834", "white", "implemented, known result not reached"),
    "idea":           ("#fbe7b8", INK, "idea only"),
    "none":           ("#efeeea", MUTED, "not proposed"),
}
DEAD = {"not-considered": ("#efeeea", MUTED), "considered": ("#e2e0d9", INK)}
NOVEL = ("#6250d6", "white")
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


def stage_of(o):
    return ("result" if o.get("result") else "implementation" if o.get("implementation") else
            "idea" if o.get("idea") else "none")


def plot(key_path, ov_path, out):
    nodes = parse(open(key_path).read())
    ov = json.load(open(ov_path))
    novel = ov.get("novel", [])
    rows = tree_rows(nodes, novel)
    champ = set(ov["champion"]["stack"])
    best_adds = set((ov.get("best_measured") or {}).get("adds", [])) - champ
    RH, BW, BH, DX, TOP = 0.42, 2.9, 0.32, 0.32, 1.3
    W = 0.3 + 4 * DX + BW + 4.6
    H = TOP + RH * (len(rows) + 1) + 0.2
    fig = plt.figure(figsize=(W, H), dpi=150); fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

    c = ov["champion"]
    fig.text(0.02, 1 - 0.08 / H, f"{ov['run']}: champion {c['record']} {c['tok_s']} tok/s "
             f"(+{c['pct']}% vs BASE {ov['base']})", ha="left", va="top", fontsize=11, weight="bold", color=INK)
    r = ov.get("resources", {})
    sub = ""
    if ov.get("best_measured"):
        b = ov["best_measured"]
        sub = f"best measured {b['record']} {b['tok_s']} (+{b['pct']}%, not promoted: ☆);  "
    sub += f"resources: {r.get('wall_h')} h wall, {r.get('gpu_min')} GPU-min, {r.get('output_tokens')} output tokens"
    fig.text(0.02, 1 - 0.36 / H, sub, ha="left", va="top", fontsize=9, color=INK2)
    hs = [Patch(fc=STAGE[k][0], ec=GRID, label=STAGE[k][2]) for k in STAGE]
    hs += [Patch(fc=DEAD["considered"][0], ec=INK2, ls="--", label="dead end (considered)"),
           Patch(fc=NOVEL[0], ec=GRID, label="novel (measured, not in key)")]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.02, 1 - 0.58 / H), ncol=3, frameon=False,
               fontsize=8.2, handlelength=1.4, columnspacing=1.2)
    fig.text(0.02, 1 - 1.05 / H, "win nodes: idea → impl → result;  ★ champion's stack   ☆ best measured only   "
             "✗ missed win;  dead ends: ✓ right reason / ! wrong reason", fontsize=8.2, color=INK2, va="top")

    y0 = TOP + RH * 0.5
    pos = {"BASE": (0.3, y0)}
    ax.text(0.3, y0, "BASE", va="center", ha="left", weight="bold", color=INK)
    for i, (n, depth, parent) in enumerate(rows):
        pos[n] = (0.3 + DX * (depth + 1), y0 + RH * (i + 1))
    xa = 0.3 + 4 * DX + BW + 0.2      # annotation column
    for n, depth, parent in rows:
        x, y = pos[n]
        px, py = pos[parent]
        on_path = n in champ and (parent == "BASE" or parent in champ)
        ax.plot([px + 0.08, px + 0.08, x], [py + BH / 2 if parent != "BASE" else py + 0.1, y, y],
                color=INK if on_path else GRID, lw=2.2 if on_path else 1.0, zorder=1, solid_capstyle="butt")
        mark, mark_c, ls = "", INK2, "-"
        if n not in nodes:                                   # novel
            v = next(v for v in novel if v["id"] == n)
            fill, txt = NOVEL
            line2, ev = v.get("title", ""), v.get("result", "")
            mark, mark_c = "+ novel", NOVEL[0]
        else:
            d = nodes[n]
            o = ov["nodes"].get(n, {})
            extra = "".join(f"  + {p}" for p in d["builds_on"][1:]) + "".join(f"  alt: {a}" for a in d["alt"])
            ev = o.get("evidence", "")
            if o.get("type") == "dead-end":
                reason = o.get("reason", "not-considered")
                fill, txt = DEAD["not-considered" if reason == "not-considered" else "considered"]
                ls = "--"
                line2 = d["status"] + extra
                if reason == "right":
                    mark, mark_c = f"✓ right reason ({o.get('how', '?')})", GOOD
                elif reason == "wrong":
                    mark, mark_c = f"! wrong reason ({o.get('how', '?')})", CRIT
            else:
                fill, txt, _ = STAGE[stage_of(o)]
                if o.get("result") and o.get("result_at_gpu_min") is not None:
                    ev = f"@{o['result_at_gpu_min']:g} GPU-min  " + ev
                ok = lambda k: "✓" if o.get(k) else "✗"
                line2 = f"idea {ok('idea')}  impl {ok('implementation')}  result {ok('result')}   {d['status']}{extra}"
                if not o.get("result"):
                    if o.get("covered_by_alt"):
                        mark, mark_c = f"= via alt {o['covered_by_alt']}", MUTED
                    else:
                        mark, mark_c = "✗ missed win", CRIT
                if o.get("wrong_cause"):
                    mark = (mark + "  " if mark else "") + "! wrong cause"; mark_c = CRIT
        crit = mark_c == CRIT
        edge = CRIT if crit else (GRID if fill == DEAD["not-considered"][0] else INK2)
        box(ax, x, y, BW, BH, fill, edge, lw=1.8 if crit else 0.8, ls=ls)
        star = "★ " if n in champ else "☆ " if n in best_adds else ""
        ax.text(x + 0.08, y - 0.045, f"{star}{n}", va="center", ha="left", color=txt, weight="bold",
                fontsize=8.6, zorder=4)
        ax.text(x + 0.08, y + 0.095, line2[:68], va="center", ha="left", color=txt, fontsize=6.5, zorder=4)
        xx = xa
        if mark:
            t = ax.text(xx, y, mark, va="center", ha="left", color=mark_c, weight="bold", fontsize=8.4)
            fig.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transData.inverted())
            xx = bb.x1 + 0.2
        if ev:
            ax.text(xx, y, ev[:80], va="center", ha="left", color=INK2, fontsize=7.8)
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)
    print(out)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    plot(*sys.argv[1:])
