"""Diagrams for flavors/v1/design/example_round.md.

Regenerate (matplotlib in a venv under autoinfer/.cache, never in $HOME):
    MPLCONFIGDIR=../../../../.cache/mpl ../../../../.cache/venv/bin/python make_diagrams.py
All numbers are illustrative (a made-up round 1 on the qwen task with focus "reduce FFN latency").

Layout: 1 data unit = 0.1 inch on both axes. y grows downward from 0 (negative values); each
figure is sized to its content at the end. Cards size themselves to their wrapped text.
"""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 120

# Colors by who writes the thing: (fill, edge)
HUMAN = ("#f0f0f0", "#8a8a8a")
MANAGER = ("#e3eef9", "#3a72a8")
LEADER = ("#e6f2df", "#4f8a33")
EVAL = ("#fcebdf", "#c0601a")
BRAIN = ("#efe6f6", "#7a4fa0")
WORKER = ("#fff7d1", "#a68a00")
PLAIN = ("#ffffff", "#9a9a9a")
OFFICIAL = "#b34700"
GREEN = "#2e7d32"
GRAY = "#555555"

FS = 11.5       # body text
TFS = 13        # card titles
PADX, PADY = 1.4, 1.2


def cw(fs, mono=False):
    """Approximate character width in data units."""
    return fs * (0.0845 if mono else 0.076)


def lh(fs):
    """Line height in data units."""
    return fs * 0.139 * 1.32


def wrap(text, width, fs, mono=False, indent=3):
    n = max(10, int((width - 2 * PADX) / cw(fs, mono)))
    out = []
    for line in text.split("\n"):
        if not line.strip():
            out.append("")
            continue
        ind = len(line) - len(line.lstrip())
        out += textwrap.wrap(line, n, subsequent_indent=" " * (ind + indent)) or [""]
    return out


def measure(w, title, body, fs=FS, tfs=TFS, mono=False):
    lines = wrap(body, w, fs, mono) if body else []
    h = 2 * PADY + (tfs * 0.139 * 1.45 if title else 0) + len(lines) * lh(fs)
    if title and lines:
        h += 0.5
    return h


def card(ax, x, ytop, w, title, body="", color=MANAGER, fs=FS, tfs=TFS, mono=False, h=None,
         lw=1.3, title_color="#111111"):
    """Draw a card with its top-left at (x, ytop). Lines starting with '@o ' are drawn in the
    'official' color, '@g ' green, '@b ' bold, '@i ' gray italic. Returns the height."""
    lines = wrap(body, w, fs, mono) if body else []
    if h is None:
        h = measure(w, title, body, fs, tfs, mono)
    ax.add_patch(FancyBboxPatch((x, ytop - h), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=color[0], ec=color[1], lw=lw))
    y = ytop - PADY
    if title:
        ax.text(x + PADX, y, title, fontsize=tfs, weight="bold", va="top", color=title_color)
        y -= tfs * 0.139 * 1.45 + 0.5
    for line in lines:
        kw = dict(fontsize=fs, va="top", family="DejaVu Sans Mono" if mono else "DejaVu Sans")
        for tag, extra in (("@o ", dict(color=OFFICIAL, weight="bold")), ("@g ", dict(color=GREEN, weight="bold")),
                           ("@b ", dict(weight="bold")), ("@i ", dict(color=GRAY, style="italic"))):
            if line.lstrip().startswith(tag):
                line = line.replace(tag, "", 1)
                kw.update(extra)
        if "::" in line:
            label, rest = line.split("::", 1)
            t = ax.text(x + PADX, y, label + ":", **dict(kw, weight="bold"))
            wpx = t.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width
            ax.text(x + PADX + wpx / DPI * 10, y, rest, **kw)
        else:
            ax.text(x + PADX, y, line, **kw)
        y -= lh(fs)
    return h


def arrow(ax, p, q, color="#333333", rad=0.0, ls="-", lw=1.4, ms=13):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def new_fig(W, title, subtitle):
    fig = plt.figure(dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(2, -2, title, fontsize=19, weight="bold", va="top")
    ax.text(2, -6.6, subtitle, fontsize=12, va="top", color=GRAY, style="italic")
    return fig, ax


def legend(ax, y, items, x=2, step=34):
    for i, (name, c) in enumerate(items):
        xx = x + i * step
        ax.add_patch(FancyBboxPatch((xx, y - 2.2), 3.2, 2.2, boxstyle="round,pad=0,rounding_size=0.4",
                                    fc=c[0], ec=c[1], lw=1.2))
        ax.text(xx + 4.4, y - 1.1, name, fontsize=12, va="center")


def section(ax, x, y, text):
    ax.text(x, y, text, fontsize=12.5, weight="bold", color="#444444", va="top")


def finish(fig, ax, W, ybottom, name):
    H = -ybottom + 2
    ax.set_xlim(0, W)
    ax.set_ylim(-H, 0)
    fig.set_size_inches(W / 10, H / 10)
    fig.savefig(os.path.join(HERE, name), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 1. Global state and how it links together (after round 1)
# ---------------------------------------------------------------------------------------------
def data_structures():
    W = 172
    fig, ax = new_fig(W, "v1 global state, and how it links together",
                      "After round 1 of the example (focus: \"reduce FFN latency\"). Box color = who writes it. "
                      "Numbers are illustrative.")
    legend(ax, -11, [("set by the human", HUMAN), ("written by the manager", MANAGER),
                     ("written by leaders", LEADER), ("evaluator / GPU queue (append-only)", EVAL)], step=40)

    # Row A: inputs
    y = -17
    section(ax, 2, y, "INPUTS  -  set by the human at run start; read-only to agents")
    y -= 3.6
    inputs = [
        ("TASK  (tasks/<task>/)",
         "Objective: output tok/s on ShareGPT, 400 prompts at concurrency 16\n"
         "Constraints: FP8 weights + FP8 KV; same math (fp reordering OK)\nHarness (frozen): bench, GSM8K-250\n"
         "Baseline profile (provided): where the time goes"),
        ("FOCUS", "\"Reduce FFN latency\""),
        ("CONFIG", "3 teams, <= 3 workers each\nMax 5 rounds, k = 3 directions\n40 GPU-min per team per round\n"
                   "Brainstormers (persona prompts): systematic engineer, adjacent-field colleague"),
        ("SEARCH_POLICY", "1. A leader improved the champion -> continue that line\n"
                          "2. Rest by upside; >= 1 slot for a high-novelty idea\n3. Nothing improved -> upside + novelty"),
        ("LEDGER  (assumptions)", "Frozen:: 2x RTX 4090, FP8 W + FP8 KV, model math, ShareGPT workload\n"
                                  "Open:: how the model is split across the 2 GPUs (baseline: TP=2), GEMM kernels, "
                                  "fusion, weight layout, tokens per pass, comm pattern"),
    ]
    widths = [36, 20, 33, 37, 38]
    gap = (W - 4 - sum(widths)) / 4
    hA = max(measure(w, t, b) for (t, b), w in zip(inputs, widths))
    x = 2
    for (t, b), w in zip(inputs, widths):
        card(ax, x, y, w, t, b, HUMAN, h=hA)
        x += w + gap
    y -= hA + 5

    # Row B: tree | bank | champion + shared KB
    section(ax, 2, y, "RUN STATE")
    y -= 3.6
    TX, TW = 2, 56
    BX, BW = 70, 52
    RX, RW = 128, 42
    topB = y

    # Bank (measure first; it sets the row height)
    recs = [
        ("BASE  -  baseline (task)", "vllm@4527515, empty diff\n@o Official: 656 tok/s, GSM8K 93.2%", HUMAN),
        ("T1-001  -  Team 1 / W1  -  from BASE", "Microbench Marlin vs FP8 tensor-core GEMM on the real FFN shapes\n"
                                                 "Observed: FP8-TC faster above ~64 tokens/pass", LEADER),
        ("T1-002  -  Team 1 / W2  -  from BASE", "Dispatch: FP8-TC if tokens/pass >= 64\nValidity: reviewed by the leader, OK\n"
                                                 "@o Official: 689 tok/s (+5%), GSM8K 93.6%", LEADER),
        ("T2-001  -  Team 2 / W1  -  from BASE", "SiLU*mul fused into the gate/up GEMM epilogue\n"
                                                 "Unofficial: +1% (within noise); not submitted", LEADER),
        ("T3-001  -  Team 3 / W1  -  from BASE", "Reduce-scatter + all-gather around the FFN\nUnofficial: -6%\n"
                                                 "Observed: 2 NCCL calls ~ 2x the latency of 1", LEADER),
    ]
    rfs, rtfs = 10.8, 11.5
    inner = BW - 4
    rec_h = [measure(inner, t, b, rfs, rtfs) for t, b, _ in recs]
    bank_title_h = 7.5
    foot = "@i Record = id, team, performance, short summary (results = Observed, conclusion = Interpretation), validity. " \
           "change.diff is against the task baseline."
    foot_lines = wrap(foot, BW, 10.5)
    bank_h = bank_title_h + sum(rec_h) + 1.6 * len(recs) + len(foot_lines) * lh(10.5) + 2.5
    card(ax, BX, topB, BW, "BANK  (experiment bank)", "", LEADER, h=bank_h)
    ax.text(BX + PADX, topB - 4.3, "one folder per experiment: record.md + change.diff", fontsize=10.8, va="top", color=GRAY)
    ry = topB - bank_title_h
    rec_mid = {}
    for (t, b, c), h in zip(recs, rec_h):
        card(ax, BX + 2, ry, inner, t, b, c, fs=rfs, tfs=rtfs)
        rec_mid[t.split()[0]] = (ry - h / 2, ry)
        ry -= h + 1.6
    ax.text(BX + PADX, ry - 0.3, "\n".join(l.replace("@i ", "") for l in foot_lines), fontsize=10.5, va="top",
            color=GRAY, style="italic", linespacing=1.3)

    # Tree (same height as the bank)
    card(ax, TX, topB, TW, "TREE  (search_tree.md, the index)", "", MANAGER, h=bank_h)
    nodes = [
        ("N1  Mixed GEMM kernel by tokens/pass", "win", GREEN, "nov 2 . up 4 . by systematic engineer", ["T1-001", "T1-002"]),
        ("N2  Fuse SiLU*mul into gate/up GEMM", "no gain", "#9e6a00", "nov 2 . up 3 . by systematic engineer", ["T2-001"]),
        ("N3  Sequence-parallel FFN (RS + AG)", "slower", "#b71c1c", "nov 5 . up 2 . by adjacent field (novelty slot)", ["T3-001"]),
        ("N4  More tokens per weight read", "proposed", "#777777", "nov 3 . up 2 . by adjacent field", []),
        ("N5  Split-K for the small down GEMM", "proposed", "#777777", "nov 1 . up 2 . by systematic engineer", []),
    ]
    ny = topB - 6.5
    root_h = card(ax, TX + 2, ny, TW - 4, "", "@b FOCUS: reduce FFN latency", PLAIN, fs=11)
    spine_x = TX + 5
    spine_top = ny - root_h
    ny -= root_h + 2.5
    node_mid = {}
    for title, status, col, detail, exps in nodes:
        body = detail + ("\nexperiments: " + ", ".join(exps) if exps else "\nno experiments yet (backlog)")
        nh = card(ax, TX + 8, ny, TW - 10, title, body, ("#ffffff", MANAGER[1]), fs=10.5, tfs=11.3, lw=1.0)
        ax.text(TX + TW - 3.5, ny - PADY, status, fontsize=11, weight="bold", color=col, va="top", ha="right")
        mid = ny - nh / 2
        ax.plot([spine_x, TX + 8], [mid, mid], color=MANAGER[1], lw=1.0)
        node_mid[title[:2]] = (mid, exps)
        last_mid = mid
        ny -= nh + 2.2
    ax.plot([spine_x, spine_x], [spine_top, last_mid], color=MANAGER[1], lw=1.0)
    ax.text(TX + PADX, topB - bank_h + 1.2, "A node is an idea; experiments hang off it.\nBottleneck = attribute, not structure.",
            fontsize=10.5, va="bottom", color=GRAY, style="italic", linespacing=1.3)

    # tree -> bank links
    for n, (mid, exps) in node_mid.items():
        for e in exps:
            arrow(ax, (TX + TW - 0.2, mid), (BX + 1.8, rec_mid[e][0]), color=MANAGER[1], lw=1.1, ms=11)

    # Right column: champion + shared KB (stretched to the row height)
    ch = card(ax, RX, topB, RW, "CHAMPION  (a bank record id)",
              "@b -> T1-002: 689 tok/s, GSM8K 93.6%\nA performance reference, not a required starting point. "
              "Set between rounds; included in every leader's task.", MANAGER)
    arrow(ax, (RX - 0.2, topB - ch / 2), (BX + BW - 1.8, rec_mid["T1-002"][0] + 1.5), color=MANAGER[1], lw=2.0, rad=0.2)
    ky = topB - ch - 3
    kb_body = ("K1  FFN GEMMs are DRAM-bound (~80% of peak); little headroom in the kernel itself.  [BASE]\n"
               "K2  FP8-TC beats Marlin above ~64 tokens/pass on sm_89.  [T1-001, T1-002]\n"
               "K3  Extra NCCL calls cost more than they save over PCIe.  [T3-001]\n\n"
               "@i Entries cite the records behind them.")
    kh = card(ax, RX, ky, RW, "SHARED_KB  (manager only)", kb_body, MANAGER)
    card(ax, RX, ky - kh - 3, RW, "LOCAL KBs  (one per leader)",
         "Team 1:: crossover table; open: one threshold for all GEMM shapes\n"
         "Team 2:: epilogue fusion saves ~6 us/layer, below bench noise\nTeam 3:: NCCL latency numbers",
         LEADER, h=bank_h - ch - kh - 6)
    for e, dy, rad in (("T1-001", 0, 0.15), ("T3-001", 0, -0.15)):
        arrow(ax, (RX - 0.2, ky - 14 if e == "T1-001" else ky - 22), (BX + BW - 1.8, rec_mid[e][0] + dy),
              color=MANAGER[1], lw=1.1, ls="--", rad=rad, ms=11)
    y = topB - bank_h - 6

    # Row C: journal | GPU queue
    rowC = [
        (TX, BX + BW - TX, "JOURNAL  (decisions, in order, and why)",
         "R1 start:: picked N1, N2 (upside) and N3 (novelty slot; rule 3, nothing improved yet). N4, N5 -> backlog.\n"
         "R1 end:: T1-002 -> champion (+5%). N2 closed (gain < noise). N3 closed (extra NCCL calls cost more on PCIe).\n"
         "R2 start:: rule 1 -> continue Team 1's line: a brainstormer proposed a threshold per GEMM shape.", MANAGER, False),
        (RX, RW, "GPU QUEUE  (usage, round 1)",
         "Team 1    32 / 40 GPU-min\nTeam 2    22 / 40\nTeam 3    40 / 40  exhausted\n"
         "Manager and brainstormers:\n          unbudgeted, still queued\n\nOne model-loading job at a time.", EVAL, True),
    ]
    hC = max(measure(w, t, b, fs=10.8 if m else FS, mono=m) for _, w, t, b, _, m in rowC)
    for x, w, t, b, c, m in rowC:
        card(ax, x, y, w, t, b, c, h=hC, fs=10.8 if m else FS, mono=m)
    arrow(ax, (TX + TW / 2, y + 0.2), (TX + TW / 2, topB - bank_h - 0.2), color=MANAGER[1], ls="--", lw=1.2)
    ax.text(TX + TW / 2 + 1, y + 3, "refers to tree nodes", fontsize=10.5, color=MANAGER[1], va="center")
    y -= hC + 5

    # Row D: who reads what
    who = ("Manager::  everything; the only writer of TREE, SHARED_KB, JOURNAL and CHAMPION.\n"
           "Leaders::  all global state; write their own local KB; file BANK records.\n"
           "Workers::  only their workspace (the baseline + the starting record's diff + brief files the leader copied in).\n"
           "Brainstormers::  everything (TASK incl. the baseline profile, BANK, KBs, TREE, JOURNAL), plus the manager's "
           "briefing; chat with leaders; profile any record through the GPU queue. Variety comes from the personas.")
    h = card(ax, 2, y, W - 4, "WHO READS WHAT", who, PLAIN)
    finish(fig, ax, W, y - h, "1_global_state.png")


# ---------------------------------------------------------------------------------------------
# 2. One round, phase by phase, with the example
# ---------------------------------------------------------------------------------------------
def round_overview():
    W = 182
    fig, ax = new_fig(W, "v1: one round, phase by phase",
                      "Example: round 1, focus \"reduce FFN latency\". Left: the algorithm (DESIGN.md). "
                      "Right: what it could look like. Numbers are illustrative.")
    legend(ax, -11, [("manager", MANAGER), ("leaders", LEADER), ("brainstormers", BRAIN),
                     ("workers", WORKER), ("evaluator / GPU queue", EVAL)], step=30)
    L, LW = 8, 46
    R = L + LW + 4
    RW = W - R - 2
    y = -17
    phase_tops = {}

    def row(key, left_title, left_body, right_fn, color=MANAGER):
        nonlocal y
        phase_tops[key] = y
        rh = right_fn(y, measure_only=True)
        lhh = measure(LW, left_title, left_body)
        h = max(rh, lhh)
        card(ax, L, y, LW, left_title, left_body, color, h=h)
        right_fn(y, h=h)
        top = y
        y -= h + 6
        return top, h

    def simple(body, color=MANAGER):
        def f(top, measure_only=False, h=None):
            if measure_only:
                return measure(RW, "", body)
            card(ax, R, top, RW, "", body, color, h=h)
        return f

    def down_arrow(top, h):
        arrow(ax, (L + LW / 2, top - h - 0.2), (L + LW / 2, top - h - 5.8), color="#444444", lw=1.6)

    t, h = row("init", "initialize()  (once)", "Official eval of the baseline, bound analysis, ledger "
               "(the human reviews it). The baseline profile comes with the task.",
               simple("@b Baseline (BASE): 656 tok/s, GSM8K 93.2%\n"
                      "Profile: Marlin FP8 GEMMs 49% of kernel time, NCCL all-reduce 27%, GDN 6%, attention 5%.\n"
                      "Gate/up GEMM: 117 us/layer, DRAM-bound at ~80% of peak; tensor pipe ~25% busy.\n"
                      "Bound: reading the gate/up weights at peak bandwidth takes ~90 us, so the kernel itself has "
                      "~25% headroom. The rest must come from fewer bytes per token, or from the other half of the time."))
    down_arrow(t, h)
    t, h = row("start", "Round r begins", "CHAMPION goes into every leader's task.",
               simple("CHAMPION = BASE (656 tok/s). Set between rounds. Agents may still start from any BANK record."))
    down_arrow(t, h)

    # brainstorm
    ideas = [
        ("systematic eng.", "Mixed GEMM kernel: FP8 tensor-core for big passes, Marlin for small", "2", "4", "selected -> N1", "sel"),
        ("systematic eng.", "Fuse SiLU*mul (+ FP8 act quant) into the gate/up GEMM epilogue", "2", "3", "selected -> N2", "sel"),
        ("adjacent field", "Sequence-parallel FFN: reduce-scatter + all-gather", "5", "2", "novelty slot -> N3", "nov"),
        ("adjacent field", "Challenge \"one weight read per pass\": more tokens per read", "3", "2", "backlog (N4)", ""),
        ("systematic eng.", "Split-K for the small down GEMM (~62% of DRAM peak)", "1", "2", "backlog (N5)", ""),
        ("adjacent field", "Spec tokens 2 + Triton FP8 backend", "1", "1", "dropped: both tried", ""),
    ]
    tfs_ = 11
    trow = 3.3
    sel_note = ("Selection (search_policy.md rule 3: nothing has improved yet): 2 slots by upside (a, b) + 1 protected "
                "high-novelty slot (c), even though napkin math is pessimistic about c. d and e stay in the TREE as "
                "backlog; the JOURNAL records why.")
    brief_body = ("\"FFN GEMMs are half the time and already near DRAM peak. NCCL over PCIe is 27%. Tried so far: "
                  "spec tokens 2 (+12%), Triton FP8 backend (net -4%) ...\"")

    def brainstorm_right(top, measure_only=False, h=None):
        h1 = measure(RW, "1. Briefing (round 1: task, focus, baseline profile)", brief_body)
        note_lines = wrap(sel_note, RW, 11)
        h2 = 6 + (len(ideas) + 1) * trow + 2 + len(note_lines) * lh(11) + 2
        total = h1 + 3 + h2
        if measure_only:
            return total
        h2 = (h or total) - h1 - 3
        card(ax, R, top, RW, "1. Briefing (round 1: task, focus, baseline profile)", brief_body, MANAGER, h=h1)
        ty = top - h1 - 3
        card(ax, R, ty, RW, "2-4. Two brainstormers propose; the manager scores and selects (k = 3)", "", BRAIN, h=h2)
        cols = [R + 2, R + 28, R + RW - 33, R + RW - 27.5, R + RW - 22]
        yy = ty - 6
        for c, txt in zip(cols, ("persona", "idea", "nov", "up", "decision")):
            ax.text(c, yy, txt, fontsize=tfs_, weight="bold", va="center")
        ax.plot([R + 1.5, R + RW - 1.5], [yy - trow / 2, yy - trow / 2], color=BRAIN[1], lw=0.8)
        for i, (src, idea, nov, up, dec, kind) in enumerate(ideas):
            yy -= trow
            if kind:
                ax.add_patch(FancyBboxPatch((R + 1.2, yy - trow / 2 + 0.15), RW - 2.4, trow - 0.3,
                                            boxstyle="round,pad=0,rounding_size=0.4",
                                            fc="#ffffff", ec="none", alpha=0.85))
            col = GREEN if kind == "sel" else (BRAIN[1] if kind == "nov" else "#333333")
            for c, txt in zip(cols, (src, f"{'abcdef'[i]}.  {idea}", nov, up, dec)):
                ax.text(c, yy, txt, fontsize=tfs_, va="center", color=col if c == cols[4] else "#111111",
                        weight="bold" if (c == cols[4] and kind) else "normal")
        yy -= trow / 2 + 1.5
        for line in wrap(sel_note, RW, 11):
            ax.text(R + 2, yy, line, fontsize=11, va="top", color="#333333")
            yy -= lh(11)

    bs_left = ("1. The manager writes the briefing: facts + what was tried and observed (also a status report).\n\n"
               "2. Brainstormers, one per persona in CONFIG (the only idea source): fresh sessions in parallel. "
               "All can access everything, chat with any leader, and profile any record (GPU queue, unbudgeted). "
               "Round 1: same inputs for all (task, focus, baseline profile); only the persona differs.\n\n"
               "3. The manager dedupes and scores: novelty 1-5, upside 1-5.\n\n"
               "4. The manager selects k directions with SEARCH_POLICY; the rest go to the TREE as `proposed` (backlog).")
    t, h = row("bs", "brainstorm()", bs_left, brainstorm_right)
    down_arrow(t, h)

    t, h = row("assign", "manager.assign()", "One task string per leader: direction + rationale + GPU budget + the "
               "current CHAMPION.",
               simple("@b Team 1 <- N1:\n"
                      "\"Use the FP8 tensor-core GEMM when a pass has many tokens, and Marlin otherwise. Why: Triton FP8 "
                      "cut TTFT 14% but raised decode ITL 14%. Budget: 40 GPU-min. Champion: BASE (656 tok/s). "
                      "Related: the BASE profile, the Triton backend sweep.\"\n"
                      "@b Team 2 <- N2 (fused epilogue).    Team 3 <- N3 (RS + AG).\n"
                      "@i Not in the task (global, read on demand): constraints, metric, BANK, SHARED_KB."))
    down_arrow(t, h)

    lanes = [
        ("Team 1  (N1)", "W1:: microbench both kernels on the FFN shapes -> crossover at ~64 tokens/pass "
                         "(T1-001, 3 GPU-min).\n"
                         "W2:: dispatch by tokens/pass; 2 iterations of smoke + bench + nsys (22 GPU-min).\n"
                         "Leader:: files T1-001, T1-002; reviews T1-002's diff: valid -> official eval (7 GPU-min).\n"
                         "@o => 689 tok/s (+5%), GSM8K 93.6%\n"
                         "Used 32 / 40. Left open: one threshold for all GEMM shapes."),
        ("Team 2  (N2)", "W1:: fused SiLU*mul epilogue. Microbench: -6 us/layer. Unofficial bench: +1%, within "
                         "noise (T2-001).\n"
                         "Leader:: not worth an official eval. Stops at a checkpoint: \"gain below noise; revisit if "
                         "the FFN share grows.\"\n"
                         "Used 22 / 40. Left open: the FP8 activation quant is not fused yet."),
        ("Team 3  (N3)", "W1:: RS + AG around the FFN. Unofficial bench: -6% (T3-001).\n"
                         "W2:: nsys: 2 NCCL calls cost ~2x one all-reduce on PCIe.\n"
                         "Budget exhausted (40 / 40): no more GPU jobs, but it still analyzes the profile and writes "
                         "the report.\nLeft open: nothing; the idea looks closed."),
    ]
    lw3 = (RW - 6) / 3

    def exec_right(top, measure_only=False, h=None):
        hh = max(measure(lw3, t_, b_) for t_, b_ in lanes)
        if measure_only:
            return hh
        for i, (t_, b_) in enumerate(lanes):
            card(ax, R + i * (lw3 + 3), top, lw3, t_, b_, LEADER, h=h)

    t, h = row("exec", "execute()", "Leaders work in parallel; barrier at the end.\n\n"
               "Each leader: decompose -> set up workspaces -> workers -> file records -> review diffs -> "
               "(maybe) official eval -> report.\n\n"
               "All GPU jobs go through the queue, charged to the team.", exec_right, color=LEADER)
    down_arrow(t, h)

    t, h = row("collect", "collect_and_update()", "Update the TREE, promote the CHAMPION, update SHARED_KB and "
               "the JOURNAL, report to the human.",
               simple("TREE: N1 win, N2 no gain, N3 slower, each with its experiments.\n"
                      "@b CHAMPION <- T1-002 (689 tok/s: +5% beats the ~3% noise, and GSM8K holds).\n"
                      "SHARED_KB += K2 [T1-001, T1-002], K3 [T3-001].   JOURNAL: round summary.\n"
                      "Report to the human: what was tried, what was learned, the new champion, the backlog.\n"
                      "Next round: rule 1 -> a slot continues Team 1's line (e.g. a threshold per GEMM shape, if a brainstormer "
                      "proposes it). Brainstormers profile the new champion. Combining wins is just another brainstorm idea."))

    # loop arrow back to "Round r begins"
    ys = t - h / 2
    ye = phase_tops["start"] - 3
    ax.plot([L, 3.5, 3.5], [ys, ys, ye], color="#444444", lw=1.6)
    arrow(ax, (3.5, ye), (L - 0.2, ye), color="#444444", lw=1.6)
    ax.text(2.3, (ys + ye) / 2, "next round", rotation=90, fontsize=12, color="#444444", va="center", ha="center",
            bbox=dict(fc="white", ec="none", pad=1))
    finish(fig, ax, W, t - h, "2_round_overview.png")


# ---------------------------------------------------------------------------------------------
# 3. Inside one team (Team 1), as a sequence diagram
# ---------------------------------------------------------------------------------------------
def inside_team():
    W = 172
    fig, ax = new_fig(W, "v1: inside one team during execute()",
                      "Example: Team 1, direction N1, budget 40 GPU-min. Time flows down. Numbers are illustrative.")
    lanes = [("Manager", "Manager", MANAGER, 13), ("Leader", "Leader (Team 1)", LEADER, 45),
             ("W1", "Worker W1", WORKER, 76), ("W2", "Worker W2", WORKER, 104),
             ("GPU", "GPU queue", EVAL, 133), ("BANK", "BANK", LEADER, 160)]
    X = {k: x for k, _, _, x in lanes}
    top = -12
    for key, name, c, x in lanes:
        ax.add_patch(FancyBboxPatch((x - 11, top - 5), 22, 5, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=c[0], ec=c[1], lw=1.3))
        ax.text(x, top - 2.5, name, fontsize=12.5, weight="bold", ha="center", va="center")
    y = top - 10
    LFS = 11.3

    def msg(a, b, text, color="#333333"):
        nonlocal y
        span = abs(X[b] - X[a])
        lw_ = max(span - 2, 60)
        lines = wrap(text, lw_, LFS, indent=0)
        y -= len(lines) * lh(LFS) + 0.6
        mx = min(max((X[a] + X[b]) / 2, 2 + lw_ / 2), W - 2 - lw_ / 2)
        ax.text(mx, y + 0.9, "\n".join(lines), fontsize=LFS, ha="center", va="bottom", color=color,
                linespacing=1.3, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9), zorder=3)
        arrow(ax, (X[a], y), (X[b], y), color=color, lw=1.6)
        y -= 4.2

    def note(who, text, color=LEADER, w=52):
        nonlocal y
        y -= 0.5
        h = measure(w, "", text, fs=LFS)
        card(ax, X[who] - w / 2, y, w, "", text, color, fs=LFS)
        y -= h + 3.5

    msg("Manager", "Leader", "task: N1 + rationale + champion + 40 GPU-min")
    note("Leader", "Reads BANK (BASE, the Triton sweep), SHARED_KB K1 and its local KB; plans 2 experiments.")
    note("Leader", "Sets up 2 workspaces from the CHAMPION (= BASE) + BRIEF.md (metric, scripts, excerpts).")
    msg("Leader", "W1", "\"Microbench Marlin vs FP8-TC on the 4 FFN shapes, M = 1..2048. Read BRIEF.md first. "
                        "Budget: 5 GPU-min.\"")
    msg("Leader", "W2", "\"Add a dispatch by tokens/pass with a configurable threshold. Read BRIEF.md first. "
                        "Budget: 30 GPU-min.\"")
    msg("W1", "GPU", "6 microbench jobs, 0.5 GPU-min each", color=WORKER[1])
    msg("W1", "Leader", "report: crossover at ~64 tokens/pass + table", color=WORKER[1])
    msg("Leader", "W2", "follow-up: \"use threshold 64 (from W1)\"")
    msg("W2", "GPU", "smoke 2 + bench 4.5 + nsys 4.5, twice = 22 GPU-min", color=WORKER[1])
    msg("W2", "Leader", "report: 690 tok/s unofficial, profile; the change is in its workspace", color=WORKER[1])
    msg("Leader", "BANK", "files T1-001 (microbench) and T1-002 (dispatch): record.md + change.diff vs the baseline")
    note("Leader", "Reviews T1-002's raw diff against the TASK constraints: same math, FP8 W + KV kept -> valid.")
    msg("Leader", "GPU", "official eval of T1-002: clean build of baseline + diff, 7 GPU-min", color=OFFICIAL)
    msg("GPU", "BANK", "append: 689 tok/s, GSM8K 93.6%", color=OFFICIAL)
    msg("Leader", "Manager", "report: summary, links to T1-001 and T1-002, proposed insight K2, "
                             "left open: one threshold for all shapes", color=LEADER[1])
    note("GPU", "@b Team 1 used 3 + 22 + 7 = 32 / 40 GPU-min", color=EVAL, w=46)
    for key, name, c, x in lanes:
        ax.plot([x, x], [top - 5, y + 1], color="#a0a0a0", lw=1.1, ls=(0, (2, 3)), zorder=0)
    foot = ("Workers only see their workspace and may run any unofficial GPU job within their share. Only the leader "
            "requests official evals, and only after its own validity review.")
    lines = wrap(foot, W - 4, 11.5)
    ax.text(2, y, "\n".join(lines), fontsize=11.5, va="top", color=GRAY, style="italic", linespacing=1.3)
    finish(fig, ax, W, y - len(lines) * lh(11.5), "3_inside_a_team.png")


if __name__ == "__main__":
    data_structures()
    round_overview()
    inside_team()
