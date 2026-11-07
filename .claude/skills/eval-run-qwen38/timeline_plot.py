"""Timeline view of an AutoInfer run against wall clock.
usage: python timeline_plot.py <timeline.json> <ledger.jsonl> <out.png> [activity.json]
  timeline.json: written by the judge (schema in SKILL.md); activity.json: from activity.py [activity.json]
Panels (shared x = wall-clock minutes since the run started):
  1. speedup vs BASE: champion step line + every official eval
  2. cumulative GPU-minutes (slope 1 = GPU always busy)
  3. swimlanes: each GPU job as a bar in its agent group's lane, colored by what it served
  4. key nodes: when each was proposed / first run / reached result (win), or settled (dead end)
"""
import json, sys
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e4de"
WIN, DEAD, STACK, PROF, OTHER, BASEC = "#2a78d6", "#eb6834", "#1b4f8f", "#b9b6ad", "#8f83d9", "#4a4a4a"

spec = json.load(open(sys.argv[1]))
ledger = [json.loads(l) for l in open(sys.argv[2])]
t0 = datetime.fromisoformat(spec["start"])
mins = lambda iso: (datetime.fromisoformat(iso) - t0).total_seconds() / 60
end = spec["end_min"]
nodes = spec["nodes"]


def color_of(tag):
    if tag in nodes:
        return WIN if nodes[tag]["type"] == "win" else DEAD
    return {"stack": STACK, "profile": PROF, "base": BASEC}.get(tag, OTHER)


act = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
nl = len(act["agents"]) if act else len(spec["lanes"])
fig, axes = plt.subplots(4, 1, figsize=(13, 11 + (0.32 * nl if act else 0)), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1.0, (0.42 * nl) if act else 1.6, 3.4], "hspace": 0.12})
ax1, ax2, ax3, ax4 = axes
for ax in axes:
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for a, b, label in spec["phases"]:
        ax.axvspan(a, b, color="#f6e6e3" if "wait" in label else "#f3f1ec", zorder=0)
    for x, label in spec["rounds"]:
        ax.axvline(x, color=INK, lw=1.0, ls=(0, (3, 2)), zorder=1)

# phase / round labels on top panel
for k, (a, b, label) in enumerate(spec["phases"]):
    ax1.text((a + b) / 2, 0.97 - 0.07 * (k % 2), label, transform=ax1.get_xaxis_transform(), ha="center", va="top",
             fontsize=8, color=MUTED)
for x, label in spec["rounds"]:
    ax1.text(x + 1, 1.07, label, transform=ax1.get_xaxis_transform(), fontsize=8.5, color=INK, va="bottom")

# 1. speedup
base = spec["base"]
pct = lambda v: (v / base - 1) * 100
xs, ys, cur = [0], [0.0], 0.0
for t, v, rec, status in spec["official"]:
    if status == "promoted":
        xs += [t, t]; ys += [cur, pct(v)]; cur = pct(v)
xs.append(end); ys.append(cur)
ax1.plot(xs, ys, color=INK, lw=2, drawstyle="default", zorder=3, label="champion (official, promoted)")
for t, v, rec, status in spec["official"]:
    filled = status == "promoted"
    ax1.plot(t, pct(v), "o", ms=8, mfc=INK if filled else "white", mec=INK, mew=1.6, zorder=4)
    note = f"{rec} {pct(v):+.1f}%" + ("" if filled else f"\n({status})")
    # below-right of the point: the step line runs right at the point's height and rises later,
    # so this corner stays free even when results come close together
    ax1.annotate(note, (t, pct(v)), xytext=((6 if filled else -6), -8), textcoords="offset points",
                 fontsize=8.5, color=INK, ha="left" if filled else "right", va="top")
ax1.set_ylabel("speedup vs BASE (%)", fontsize=9.5, color=INK)
ax1.set_ylim(-1, max(ys + [pct(o[1]) for o in spec["official"]]) + 4)
ax1.grid(axis="y", color=GRID, lw=0.8)
ax1.set_title(f"{spec['run']}: wall clock   (BASE {base} tok/s; run ended at {end/60:.1f} h)",
              loc="left", fontsize=12, color=INK, pad=24)

# 2. cumulative GPU-min
ev = sorted(((mins(e["start"]), mins(e["end"])) for e in ledger if mins(e["start"]) <= end + 1))
grid = [i * 0.5 for i in range(int(end * 2) + 2)]
cum = [sum(max(0, min(t, b) - a) for a, b in ev) for t in grid]
ax2.plot(grid, cum, color=INK, lw=1.6)
ax2.plot([0, end], [0, end], color=MUTED, lw=0.8, ls=(0, (4, 3)))
ax2.text(end * 0.97, end * 0.97, "GPU always busy", fontsize=8, color=MUTED, ha="right", va="bottom", rotation=0)
ax2.text(grid[-1], cum[-1], f" {cum[-1]:.0f} GPU-min\n ({cum[-1]/end*100:.0f}% busy)", fontsize=8.5, color=INK, va="center")
ax2.set_ylabel("GPU-min used", fontsize=9.5, color=INK)
ax2.grid(axis="y", color=GRID, lw=0.8)

# 3. swimlanes
if act:
    STATE = [("llm", "#2f5fb3", "model generating"), ("tools", "#9cc0ee", "running tools (read / edit / shell / message)"),
             ("build", "#7a5cc7", "building"), ("gpu_q", "#d33f3f", "waiting for the GPU (queued)"),
             ("gpu_run", "#f0a54a", "waiting on its own GPU job"), ("idle", "#e9e6df", "idle: waiting for messages / others")]
    SC = {k: c for k, c, _ in STATE}
    def order_key(n):
        if n == "manager": return (0, n)
        if n.startswith("bs-"): return (1, n)
        return (2, n.split(".")[0], n)
    names = sorted(act["agents"], key=order_key)
    endt = t0.timestamp() + end * 60
    for i, n in enumerate(names):
        y = len(names) - 1 - i
        for a, b, st in act["agents"][n]["intervals"]:
            a, b = min(a, endt), min(b, endt)
            if b <= a: continue
            ax3.barh(y + 0.08, (b - a) / 60, left=(a - t0.timestamp()) / 60, height=0.58, color=SC[st], lw=0, zorder=3)
        for j in act["jobs"]:
            if j["agent"] == n:
                ax3.barh(y - 0.33, (j["ended"] - j["started"]) / 60, left=(j["started"] - t0.timestamp()) / 60,
                         height=0.14, color=INK, lw=0, zorder=3)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels([("    " if "." in n else "") + n for n in reversed(names)], fontsize=8.5, color=INK)
    ax3.set_ylim(-0.6, len(names) - 0.4)
    ax3.legend(handles=[Patch(color=c, label=l) for _, c, l in STATE] + [Patch(color=INK, label="its GPU job running (thin bar)")],
               loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8, frameon=False)
    ax3.text(-0.005, 1.005, "what each agent is doing", transform=ax3.transAxes, fontsize=8.5, color=MUTED)
lanes = [] if act else list(spec["lanes"])
for e in ([] if act else ledger):
    j = str(e["job"])
    if j not in spec["jobs"]:
        continue
    lane, tag = spec["jobs"][j]
    y = len(lanes) - 1 - lanes.index(lane)
    a, b = mins(e["start"]), mins(e["end"])
    official = (e.get("label") or "").startswith("official")
    ax3.barh(y, max(b - a, 0.6), left=a, height=0.62, color=color_of(tag), edgecolor=INK if official else "white",
             lw=1.4 if official else 0.6, zorder=3)
if not act:
  ax3.set_yticks(range(len(lanes)))
  ax3.set_yticklabels([spec["lanes"][l] for l in reversed(lanes)], fontsize=9, color=INK)
  ax3.set_ylim(-0.6, len(lanes) - 0.4)
  ax3.legend(handles=[Patch(color=WIN, label="win-node work"), Patch(color=DEAD, label="dead-end work"),
                    Patch(color=STACK, label="stacking / combining"), Patch(color=PROF, label="profiling"),
                    Patch(color=OTHER, label="other (unmeasured idea)"),
                    Patch(facecolor="white", edgecolor=INK, lw=1.4, label="official eval (outlined)")],
           loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8, frameon=False)
  ax3.text(-0.005, 1.02, "GPU jobs (gaps = GPU idle)", transform=ax3.transAxes, fontsize=8.5, color=MUTED)

# 4. key nodes
order = [n for n in nodes if nodes[n]["type"] == "win"] + [n for n in nodes if nodes[n]["type"] != "win"]
for i, n in enumerate(order):
    o = nodes[n]
    y = len(order) - 1 - i
    if o["type"] == "win":
        ax4.text(-0.01, y, n, transform=ax4.get_yaxis_transform(), ha="right", va="center", fontsize=9, color=INK)
        if "covered" in o:
            ax4.text(4, y, f"covered by {o['covered']}", va="center", fontsize=8.5, color=MUTED, style="italic")
            continue
        if "idea" not in o:
            ax4.text(4, y, "✗ missed: never proposed", va="center", fontsize=8.5, color="#c62828", fontweight="bold")
            continue
        ax4.plot([o["idea"], o["result"]], [y, y], color=WIN, lw=2.2, alpha=0.35, zorder=2)
        ax4.plot([o["impl"], o["result"]], [y, y], color=WIN, lw=2.2, zorder=3)
        ax4.plot(o["idea"], y, "D", ms=7, mfc="white", mec=WIN, mew=1.6, zorder=4)
        ax4.plot(o["impl"], y, "o", ms=7, color=WIN, zorder=4)
        ax4.plot(o["result"], y, "*", ms=14, color=WIN, mec="white", mew=0.6, zorder=5)
        ax4.text(o["result"] + 3, y, f"{o['result'] - o['idea']:.0f} min idea→result", va="center", fontsize=8, color=MUTED)
    else:
        ax4.text(-0.01, y, n, transform=ax4.get_yaxis_transform(), ha="right", va="center", fontsize=9, color=MUTED)
        if "settled" not in o:
            ax4.text(4, y, "not considered", va="center", fontsize=8, color=MUTED, style="italic")
            continue
        if "first" in o:
            ax4.plot([o["first"], o["settled"]], [y, y], color=DEAD, lw=2, alpha=0.6, zorder=2)
        mark, col = ("✓", "#2e7d32") if o["reason"] == "right" else ("!", "#c62828")
        ax4.plot(o["settled"], y, "s", ms=7, mfc="white", mec=DEAD, mew=1.6, zorder=4)
        ax4.text(o["settled"] + 3, y, f"{mark} {o['reason']} reason", va="center", fontsize=8, color=col, fontweight="bold")
nwin = sum(1 for n in nodes if nodes[n]["type"] == "win")
ax4.axhline(len(order) - nwin - 0.5, color=GRID, lw=1)
ax4.set_yticks([])
ax4.set_ylim(-0.7, len(order) - 0.3)
ax4.spines["left"].set_visible(False)
ax4.legend(handles=[Line2D([], [], marker="D", ls="", mfc="white", mec=WIN, mew=1.6, label="idea proposed"),
                    Line2D([], [], marker="o", ls="", color=WIN, label="first run (implementation)"),
                    Line2D([], [], marker="*", ls="", color=WIN, ms=12, label="result reached"),
                    Line2D([], [], color=DEAD, lw=2, alpha=0.6, label="dead end: tested"),
                    Line2D([], [], marker="s", ls="", mfc="white", mec=DEAD, mew=1.6, label="dead end: settled")],
           loc="lower left", bbox_to_anchor=(1.005, 0.0), fontsize=8, frameon=False)
ax4.text(-0.005, 1.01, "answer-key nodes (win nodes above the line, dead ends below)", transform=ax4.transAxes,
         fontsize=8.5, color=MUTED)
ax4.set_xlabel("wall clock since run start (minutes)", fontsize=9.5, color=INK)
ax4.set_xlim(-2, end + 8)
ax4.set_xticks(range(0, int(end) + 1, 30))

fig.savefig(sys.argv[3], dpi=140, bbox_inches="tight", facecolor="white")
print("wrote", sys.argv[3])
