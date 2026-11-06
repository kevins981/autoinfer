"""Diagrams for flavors/v1/design/example_round.md.

Regenerate (matplotlib in a venv under autoinfer/.cache, never in $HOME):
    MPLCONFIGDIR=../../../../.cache/mpl ../../../../.cache/venv/bin/python make_diagrams.py
All numbers are illustrative (a made-up round 1 on the qwen task with focus "reduce FFN latency").
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# Colors by who writes the thing.
HUMAN = ("#eeeeee", "#777777")
MANAGER = ("#dbe9f6", "#2f6ea5")
LEADER = ("#e2f0d9", "#4b8a2e")
EVAL = ("#fde9d9", "#c0601a")
BRAIN = ("#ece2f4", "#7a4fa0")
WORKER = ("#fff6cc", "#a68a00")
OFFICIAL = "#b34700"


def canvas(w, h, title, subtitle=None):
    fig = plt.figure(figsize=(w / 10, h / 10), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(2, h - 2.5, title, fontsize=17, weight="bold", va="top")
    if subtitle:
        ax.text(2, h - 7, subtitle, fontsize=10, va="top", color="#444444", style="italic")
    return fig, ax


def box(ax, x, y, w, h, title, body="", color=MANAGER, tsize=11, bsize=8.6, mono=False, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=color[0], ec=color[1], lw=lw))
    if title:
        ax.text(x + 1, y + h - 0.9, title, fontsize=tsize, weight="bold", va="top", color="#111111")
    if body:
        off = 3.6 if title else 1.0
        ax.text(x + 1, y + h - off, body, fontsize=bsize, va="top", linespacing=1.35,
                family="DejaVu Sans Mono" if mono else "DejaVu Sans")


def arrow(ax, p, q, color="#333333", rad=0.0, ls="-", lw=1.3, label=None, lsize=8, both=False):
    a = FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=11,
                        color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(mx, my + 0.6, label, fontsize=lsize, ha="center", va="bottom", color=color,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))


def legend(ax, x, y, items):
    for i, (name, c) in enumerate(items):
        xx = x + i * 30
        ax.add_patch(FancyBboxPatch((xx, y), 3, 2, boxstyle="round,pad=0.1", fc=c[0], ec=c[1]))
        ax.text(xx + 4, y + 1, name, fontsize=9, va="center")


# ---------------------------------------------------------------------------------------------
# 1. Global state and how it links together (after round 1)
# ---------------------------------------------------------------------------------------------
def data_structures():
    W, H = 200, 128
    fig, ax = canvas(W, H, "v1 global state, and how it links together (after round 1 of the example)",
                     "Focus: \"reduce FFN latency\". Box color = who writes it. Numbers are illustrative.")
    legend(ax, 2, H - 12, [("set by the human", HUMAN), ("written by the manager", MANAGER),
                           ("written by leaders", LEADER), ("evaluator / GPU queue (append-only)", EVAL)])

    # Left column: inputs
    box(ax, 2, 92, 40, 19, "TASK  (tasks/<task>/)",
        "TASK.md: objective = output tok/s,\n  ShareGPT, 400 prompts, c=16\nconstraints: FP8 weights + FP8 KV,\n  same math (fp reorder OK)\nharness (frozen): bench, GSM8K-250",
        HUMAN)
    box(ax, 2, 82, 40, 7.5, "FOCUS", "\"reduce FFN latency\"", HUMAN)
    box(ax, 2, 64, 40, 15.5, "CONFIG",
        "3 teams (leaders), <=3 workers/leader\nmax 5 rounds, k = 3 directions/round\n40 GPU-min per team per round\npersonas: first principles, physicist,\n  systems, cross-domain, lit scout, combiner",
        HUMAN)
    box(ax, 2, 46, 40, 15.5, "SEARCH_POLICY  (search_policy.md)",
        "1. a leader improved the champion ->\n   continue its \"what I'd do next\"\n2. rest by upside; >=1 slot for a\n   high-novelty idea\n3. nothing improved -> upside + novelty",
        HUMAN)
    box(ax, 2, 27, 40, 16.5, "LEDGER  (assumptions)",
        "frozen: 2x RTX 4090, FP8 W + FP8 KV,\n  model math, ShareGPT workload\nopen: how the model is split across\n  the 2 GPUs (baseline: TP=2), GEMM\n  kernels, fusion, weight layout,\n  tokens per pass, comm pattern",
        HUMAN)
    ax.text(2, 24, "Human writes these at run start (manager may\npropose ledger changes; the human approves).",
            fontsize=8, va="top", color="#555555", style="italic")

    # Tree
    tx, ty, tw, th = 46, 46, 58, 65
    box(ax, tx, ty, tw, th, "TREE  (search_tree.md, the index)", "", MANAGER)
    nodes = [
        ("FOCUS: reduce FFN latency", None, "#111111"),
        ("+- N1 Mixed GEMM kernel by tokens/pass", "[win]", "#2e7d32"),
        ("|    nov 2 . up 4 . by: lit scout + Team 1", None, "#555555"),
        ("|    experiments: T1-001, T1-002  (T1-002 = champion)", None, "#555555"),
        ("+- N2 Fuse SiLU*mul into gate/up GEMM", "[no gain]", "#9e6a00"),
        ("|    nov 2 . up 3 . by: cross-domain", None, "#555555"),
        ("|    experiments: T2-001", None, "#555555"),
        ("+- N3 Sequence-parallel FFN (RS + AG)", "[slower]", "#b71c1c"),
        ("|    nov 5 . up 2 . by: systems (novelty slot)", None, "#555555"),
        ("|    experiments: T3-001", None, "#555555"),
        ("+- N4 More tokens per weight read", "[proposed]", "#666666"),
        ("|    nov 3 . up 2 . by: first principles", None, "#555555"),
        ("+- N5 Split-K for the small down GEMM", "[proposed]", "#666666"),
        ("     nov 1 . up 2 . by: physicist", None, "#555555"),
    ]
    node_y = {}
    y = ty + th - 5
    for text, status, col in nodes:
        ax.text(tx + 1.5, y, text, fontsize=8.6, family="DejaVu Sans Mono", va="center", color="#111111")
        if status:
            ax.text(tx + tw - 1.5, y, status, fontsize=8.6, weight="bold", va="center", ha="right", color=col)
        if text.startswith("+- N"):
            node_y[text[3:5]] = y
        y -= 4.1
    ax.text(tx + 1.5, ty + 1.5, "A node is an idea; experiments hang off it.\nBottleneck = attribute, not structure.",
            fontsize=7.8, color="#555555", style="italic")

    # Journal
    box(ax, 46, 22, 58, 20, "JOURNAL  (decisions, in order, and why)",
        "R1 start: picked N1, N2 (upside), N3 (novelty slot,\n   rule 3: nothing improved yet); N4, N5 -> backlog\nR1 end: T1-002 -> champion (+5%). N2 closed (gain < noise),\n   N3 closed (extra NCCL calls cost more on PCIe)\nR2 start: rule 1 -> Team 1 continues (tune threshold\n   per GEMM shape); ...",
        MANAGER, bsize=8.2)
    arrow(ax, (75, 42.3), (75, 46), color=MANAGER[1], ls="--", label="refers to nodes")

    # Bank
    bx, by, bw, bh = 108, 22, 48, 89
    box(ax, bx, by, bw, bh, "BANK  (experiment bank)", "", LEADER)
    ax.text(bx + 1, by + bh - 4.2, "one folder per experiment: record.md + change.diff", fontsize=8, color="#333333")
    recs = [
        ("BASE  baseline (task)", "vllm@4527515, empty diff", "official: 656 tok/s, GSM8K 93.2%", HUMAN),
        ("T1-001  Team 1 / W1   start: BASE", "microbench Marlin vs FP8 tensor-core GEMM,\nreal FFN shapes, M = 1..2048\nObserved: FP8-TC faster above ~64 tok/pass", None, LEADER),
        ("T1-002  Team 1 / W2   start: BASE", "dispatch: FP8-TC if tokens/pass >= 64\nreview: valid (leader)", "official: 689 tok/s (+5%), GSM8K 93.6%", LEADER),
        ("T2-001  Team 2 / W1   start: BASE", "SiLU*mul fused into gate/up epilogue\nunofficial bench: +1% (within noise)\nnot submitted for official eval", None, LEADER),
        ("T3-001  Team 3 / W1   start: BASE", "reduce-scatter + all-gather around FFN\nunofficial bench: -6%\nObserved: 2 NCCL calls ~ 2x latency of 1", None, LEADER),
    ]
    rec_y = {}
    ry = by + bh - 6.5
    for title, body, off, col in recs:
        h = 13.2
        box(ax, bx + 1.5, ry - h, bw - 3, h, title, body, col, tsize=9, bsize=8)
        if off:
            ax.text(bx + 2.5, ry - h + 1.2, off, fontsize=8, weight="bold", color=OFFICIAL)
        rec_y[title.split()[0]] = ry - h / 2
        ry -= h + 1.8
    ax.text(bx + 1, by + 1.2,
            "record = id, team, performance, short summary\n(results = Observed, conclusion = Interpretation),\nvalidity; change.diff is vs the task baseline",
            fontsize=7.8, color="#555555", style="italic")

    # Tree -> bank links
    for n, es in [("N1", ["T1-001", "T1-002"]), ("N2", ["T2-001"]), ("N3", ["T3-001"])]:
        for e in es:
            arrow(ax, (tx + tw + 0.3, node_y[n]), (bx + 1.4, rec_y[e]), color=MANAGER[1], lw=1.0, rad=-0.05)

    # Right column
    rx, rw = 160, 38
    box(ax, rx, 93, rw, 18, "CHAMPION  (a bank record id)",
        "-> T1-002: 689 tok/s, GSM8K 93.6%\nprofile: GEMMs 44%, NCCL 30%, ...\nA performance reference, not a\nrequired starting point. Set between\nrounds; part of every leader's task.",
        MANAGER, bsize=8.2)
    arrow(ax, (rx - 0.3, 100), (bx + bw - 1.3, rec_y["T1-002"] + 2), color=MANAGER[1], lw=1.8, rad=0.25)
    box(ax, rx, 62, rw, 28, "SHARED_KB  (manager only)",
        "K1 FFN GEMMs are DRAM-bound (~80%\n   of peak): little headroom in the\n   kernel itself. [BASE profile]\nK2 FP8-TC beats Marlin above ~64\n   tokens/pass on sm_89.\n   [T1-001, T1-002]\nK3 Extra NCCL calls cost more than\n   they save over PCIe. [T3-001]",
        MANAGER, bsize=8.2)
    arrow(ax, (rx - 0.3, 76), (bx + bw - 1.3, rec_y["T1-001"]), color=MANAGER[1], lw=1.0, ls="--", rad=0.1)
    arrow(ax, (rx - 0.3, 66), (bx + bw - 1.3, rec_y["T3-001"]), color=MANAGER[1], lw=1.0, ls="--", rad=0.1)
    ax.text(rx + 1, 60, "entries cite the records behind them", fontsize=7.8, color="#555555", style="italic")
    box(ax, rx, 37, rw, 20, "Local KBs  (one per leader)",
        "Team 1: crossover table; next: tune\n   threshold per GEMM shape\nTeam 2: epilogue fusion saves ~6 us/\n   layer, below bench noise\nTeam 3: NCCL latency numbers",
        LEADER, bsize=8.2)
    box(ax, rx, 15, rw, 19, "GPU queue usage  (round 1)",
        "Team 1   32 / 40 GPU-min\nTeam 2   22 / 40\nTeam 3   40 / 40  (exhausted)\nmanager, brainstormers:\n         unbudgeted, still queued\none model-loading job at a time",
        EVAL, bsize=8.4, mono=True)

    # Who reads what
    box(ax, 2, 2, 154, 16, "Who reads what", "", ("#ffffff", "#999999"))
    ax.text(4, 12.5,
            "Manager: everything; the only writer of TREE, SHARED_KB, JOURNAL, CHAMPION.\n"
            "Leaders: all global state (read), their own local KB (write); they file BANK records.\n"
            "Workers: only their workspace (the baseline + the starting record's diff, plus brief files the leader copied in).\n"
            "Brainstormers: the manager's briefing + a random sample of BANK records (Observed sections only); may query BANK, chat with leaders, and profile any record (GPU queue).",
            fontsize=8.6, va="top", linespacing=1.45)

    fig.savefig(os.path.join(HERE, "1_global_state.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 2. One round, phase by phase, with the example
# ---------------------------------------------------------------------------------------------
def round_overview():
    W, H = 200, 160
    fig, ax = canvas(W, H, "v1: one round, phase by phase (example: round 1, focus \"reduce FFN latency\")",
                     "Left: the algorithm (DESIGN.md). Right: what it could look like. Numbers are illustrative.")
    legend(ax, 2, H - 12, [("manager", MANAGER), ("leaders", LEADER), ("brainstormers", BRAIN),
                           ("workers", WORKER), ("evaluator / GPU queue", EVAL)])
    L, LW, R = 7, 36, 46  # left column x, width; right column x
    RW = W - R - 2

    def phase(y, h, title, body, color=MANAGER):
        box(ax, L, y, LW, h, title, body, color, tsize=11, bsize=8.4)

    # 0. initialize
    phase(129, 14, "initialize()  (once)", "official eval of the baseline,\nprofile, bound analysis, ledger")
    box(ax, R, 129, RW, 14, "",
        "Baseline (BASE): 656 tok/s, GSM8K 93.2%.  Profile: Marlin FP8 GEMMs 49% of kernel time, NCCL all-reduce 27%, GDN 6%, attention 5%.\n"
        "Gate/up GEMM 117 us/layer, DRAM-bound at ~80% of peak bandwidth; tensor pipe ~25% busy.\n"
        "Bound: reading the gate/up weights at peak bandwidth takes ~90 us -> the kernel itself has ~25% headroom; the rest must come from\n"
        "reading fewer bytes per token (more tokens per weight read) or from the other 50% of time.   Ledger reviewed by the human.",
        MANAGER, bsize=8.6)
    arrow(ax, (L + LW / 2, 129), (L + LW / 2, 125.5))

    # round start
    phase(119, 6, "round r: CHAMPION -> tasks", "")
    box(ax, R, 119, RW, 6, "", "CHAMPION = BASE (656 tok/s). Set between rounds and included in every leader's task; agents may still start from any BANK record.", MANAGER, bsize=8.6)
    arrow(ax, (L + LW / 2, 119), (L + LW / 2, 116.5))

    # brainstorm
    phase(66, 50, "brainstorm()",
              "1. leaders' \"what I'd do next\"\n   (round 1: asked directly)\n\n2. manager writes the briefing\n   (facts + what was tried and\n   observed; also a status report)\n\n3. brainstormers, one per persona,\n   fresh sessions, in parallel, each\n   with a random BANK sample\n   (Observed only); may query BANK,\n   chat with any leader, and profile\n   any record (GPU queue, unbudgeted)\n\n4. manager dedupes and scores:\n   novelty 1-5, upside 1-5\n\n5. manager selects k directions\n   with SEARCH_POLICY; the rest\n   -> TREE as `proposed` (backlog)")
    # right side of brainstorm
    box(ax, R, 105, 62, 11, "1. Leaders", "Team 1: \"Triton FP8 cut TTFT 14% but raised ITL 14%:\nuse each kernel where it wins.\"  Teams 2, 3: ...", LEADER, tsize=9.5, bsize=8.2)
    box(ax, R + 64, 105, RW - 64, 11, "2. Briefing (to brainstormers)",
        "\"FFN GEMMs are half the time and already near DRAM peak. NCCL over PCIe is\n27%. Tried: spec tokens 2 (+12%), Triton FP8 backend (net -4%) ...\"",
        MANAGER, tsize=9.5, bsize=8.2)
    rows = [
        ("persona / source", "idea", "nov", "up", "decision"),
        ("lit scout + Team 1", "a  Mixed GEMM kernel: FP8 tensor-core GEMM for big passes, Marlin for small", "2", "4", "selected -> N1"),
        ("cross-domain", "b  Fuse SiLU*mul (+ FP8 act quant) into the gate/up GEMM epilogue", "2", "3", "selected -> N2"),
        ("systems", "c  Sequence-parallel FFN: reduce-scatter + all-gather instead of all-reduce", "5", "2", "novelty slot -> N3"),
        ("first principles", "d  \"One weight read per verify pass\" is an assumption: more tokens per read", "3", "2", "backlog N4"),
        ("physicist", "e  Split-K for the small down GEMM (only ~62% of DRAM peak)", "1", "2", "backlog N5"),
        ("combiner", "f  spec tokens 2 + Triton FP8 backend", "1", "1", "dropped (both tried)"),
    ]
    box(ax, R, 66, RW, 37, "3-5. Brainstormers propose, manager scores and selects (k = 3)", "", BRAIN, tsize=9.5)
    cols = [R + 1.5, R + 25, R + 120, R + 126.5, R + 133]
    yy = 97.5
    for i, row in enumerate(rows):
        for c, txt in zip(cols, row):
            ax.text(c, yy, txt, fontsize=8.3, va="center", weight="bold" if i == 0 else "normal",
                    color="#2e7d32" if (c == cols[4] and "selected" in txt) else
                          ("#7a4fa0" if c == cols[4] and "novelty" in txt else "#111111"))
        yy -= 3.9
    ax.text(R + 1.5, 70, "Selection (search_policy.md rule 3, nothing has improved yet): 2 slots by upside (a, b) + 1 protected "
            "high-novelty slot (c),\neven though napkin math is pessimistic about c. d, e stay visible in the TREE as backlog. "
            "JOURNAL records why.", fontsize=8.3, va="center", color="#333333")
    arrow(ax, (L + LW / 2, 66), (L + LW / 2, 62.5))

    # assign
    phase(52, 10.5, "manager.assign()", "one task string per leader:\ndirection + rationale + GPU budget")
    box(ax, R, 52, RW, 10.5, "",
        "Team 1 <- N1: \"Use the FP8 tensor-core GEMM when a forward pass has many tokens and Marlin otherwise. Why: Triton FP8 cut TTFT 14%\n"
        "   but raised decode ITL 14%. Budget: 40 GPU-min. Related: BASE profile, the Triton backend sweep.\"\n"
        "Team 2 <- N2 (fused epilogue), Team 3 <- N3 (RS + AG). Not in the task (global): constraints, metric, CHAMPION, BANK, SHARED_KB.",
        MANAGER, bsize=8.4)
    arrow(ax, (L + LW / 2, 52), (L + LW / 2, 48.5))

    # execute
    phase(18, 30, "execute()",
          "leaders in parallel; barrier\n\neach leader: decompose ->\nworkspaces -> workers ->\nfile records -> review diff ->\n(maybe) official eval ->\nreport\n\nall GPU jobs go through the\nqueue, charged to the team",
          LEADER)
    lw3 = (RW - 4) / 3
    lanes = [
        ("Team 1  (N1)", "W1: microbench both kernels on the\n   FFN shapes -> crossover ~64 tok/pass\n   (T1-001, 3 GPU-min)\nW2: dispatch by tokens/pass, 2 iterations\n   of smoke + bench + nsys (22 GPU-min)\nleader: files T1-001, T1-002; reviews T1-002's diff:\n   valid -> official eval (7 GPU-min)\n   => 689 tok/s (+5%), GSM8K 93.6%\nused 32 / 40.  next: tune threshold\n   per GEMM shape"),
        ("Team 2  (N2)", "W1: fused SiLU*mul epilogue;\n   microbench: -6 us/layer\n   unofficial bench: +1% (noise)  (T2-001)\nleader: not worth an official eval;\n   stops at a checkpoint: \"gain below\n   noise; revisit if FFN share grows\"\nused 22 / 40.  next: fuse the FP8\n   activation quant too"),
        ("Team 3  (N3)", "W1: RS + AG around the FFN;\n   unofficial bench: -6%  (T3-001)\nW2: nsys: 2 NCCL calls ~ 2x the\n   latency of 1 all-reduce on PCIe\nbudget exhausted (40 / 40): no more\n   GPU jobs, but still analyzes the\n   profile and writes the report\nnext: none, close the idea"),
    ]
    for i, (t, b) in enumerate(lanes):
        box(ax, R + i * (lw3 + 2), 18, lw3, 30, t, b, LEADER, tsize=9.5, bsize=8.1)
    arrow(ax, (L + LW / 2, 18), (L + LW / 2, 15.5))

    # collect
    phase(2, 13.5, "collect_and_update()", "TREE, promote CHAMPION,\nSHARED_KB, JOURNAL,\nreport to the human")
    box(ax, R, 2, RW, 13.5, "",
        "TREE: N1 win, N2 no gain, N3 slower (each with its experiments).   Manager promotes T1-002 (689 tok/s, +5% > noise) to CHAMPION.\n"
        "SHARED_KB += K2 (\"FP8-TC beats Marlin above ~64 tok/pass\" [T1-001, T1-002]), K3 (\"extra NCCL calls cost more over PCIe\" [T3-001]).\n"
        "JOURNAL: round summary.   Report to the human: what was tried, what was learned, the new champion, the backlog.\n"
        "Next round: rule 1 (Team 1 improved the champion) -> Team 1's \"tune threshold per GEMM shape\" gets a slot. Combining wins = a brainstorm idea.",
        MANAGER, bsize=8.4)
    # loop arrow
    ax.plot([L, 3.5, 3.5], [8, 8, 122], color="#555555", lw=1.5)
    arrow(ax, (3.5, 122), (L - 0.3, 122), color="#555555", lw=1.5)
    ax.text(2.2, 66, "next round", rotation=90, fontsize=8.5, color="#555555", va="center")

    fig.savefig(os.path.join(HERE, "2_round_overview.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 3. Inside one team (Team 1), as a sequence diagram
# ---------------------------------------------------------------------------------------------
def inside_team():
    W, H = 200, 140
    fig, ax = canvas(W, H, "v1: inside one team during execute() (example: Team 1, direction N1)",
                     "Time flows down. Budget: 40 GPU-min for the round. Numbers are illustrative.")
    lanes = [("Manager", "Manager", MANAGER, 14), ("Leader", "Leader (Team 1)", LEADER, 48),
             ("W1", "Worker W1", WORKER, 84), ("W2", "Worker W2", WORKER, 116),
             ("GPU", "GPU queue", EVAL, 150), ("BANK", "BANK", LEADER, 184)]
    X = {}
    top, bottom = 122, 10
    for key, name, c, x in lanes:
        box(ax, x - 12, top, 24, 6, "", "", c)
        ax.text(x, top + 3, name, fontsize=10.5, weight="bold", ha="center", va="center")
        ax.plot([x, x], [bottom, top], color="#999999", lw=1, ls=":")
        X[key] = x

    def msg(y, a, b, text, color="#333333", ls="-"):
        arrow(ax, (X[a], y), (X[b], y), color=color, ls=ls, label=text, lsize=8.2)

    def note(y, who, text, color=LEADER, w=40, h=6):
        box(ax, X[who] - w / 2, y - h / 2, w, h, "", "", color)
        ax.text(X[who], y, text, fontsize=8, ha="center", va="center", linespacing=1.35)

    y = 116
    msg(y, "Manager", "Leader", "task: N1 + rationale + 40 GPU-min")
    y -= 9
    note(y, "Leader", "reads BANK (BASE, Triton sweep), SHARED_KB\nK1, its local KB; plans 2 experiments", w=46, h=7)
    y -= 9
    note(y, "Leader", "sets up 2 workspaces from CHAMPION (= BASE)\n+ BRIEF.md (metric, scripts, excerpts)", w=46, h=7)
    y -= 8
    msg(y, "Leader", "W1", "\"Microbench Marlin vs FP8-TC, 4 FFN shapes,\nM=1..2048. Read BRIEF.md first. 5 GPU-min\"")
    y -= 9
    msg(y, "Leader", "W2", "\"Add a dispatch by tokens/pass, threshold\nconfigurable. Read BRIEF.md first. 30 GPU-min\"")
    y -= 7
    msg(y, "W1", "GPU", "6 microbench jobs (0.5 each)", color=WORKER[1])
    y -= 6
    msg(y, "W1", "Leader", "report: crossover ~64 tok/pass + table", color=WORKER[1])
    y -= 6
    msg(y, "Leader", "W2", "follow-up: \"use threshold 64 (from W1)\"")
    y -= 7
    msg(y, "W2", "GPU", "smoke 2 + bench 4.5 + nsys 4.5, twice = 22", color=WORKER[1])
    y -= 6
    msg(y, "W2", "Leader", "report: commit, 690 tok/s unofficial, profile", color=WORKER[1])
    y -= 7
    msg(y, "Leader", "BANK", "files T1-001 (microbench) and T1-002 (dispatch): record.md + change.diff vs baseline")
    y -= 8
    note(y, "Leader", "reviews T1-002's raw diff vs TASK constraints:\nsame math, FP8 W + KV kept -> valid", w=46, h=7)
    y -= 8
    msg(y, "Leader", "GPU", "official eval of T1-002 (clean build, 7 GPU-min)", color=OFFICIAL)
    y -= 6
    msg(y, "GPU", "BANK", "append: 689 tok/s, GSM8K 93.6%", color=OFFICIAL)
    y -= 7
    msg(y, "Leader", "Manager", "report: summary, T1-001/T1-002 links, proposed insight K2, next: tune threshold per shape", color=LEADER[1])
    y -= 6
    note(y, "GPU", "Team 1 used 3 + 22 + 7 = 32 / 40", color=EVAL, w=34, h=4.5)
    ax.text(2, 1.5, "Workers only see their worktree. Workers may run any unofficial GPU job within their share; only the leader "
            "requests official evals, and only after its own validity review.", fontsize=8.4, color="#444444", style="italic")

    fig.savefig(os.path.join(HERE, "3_inside_a_team.png"))
    plt.close(fig)


if __name__ == "__main__":
    data_structures()
    round_overview()
    inside_team()
