#!/usr/bin/env python3
"""Build the human handoff packet (reports/packet/) for an AutoInfer run.

Reads the run's records (bank/, tree spec, KB, ideas, rulings, reports) and writes a self-contained
folder: index.html (inline CSS/JS, no network) + diffs/<id>.diff + data/*.md copies.
Usage (from the run folder): python3 reports/packet_src/build_packet.py
"""
import html, json, os, re, shutil

RUN = os.environ.get("AUTOINFER_RUN") or os.getcwd()
SRC = os.path.join(RUN, "reports", "packet_src")
OUT = os.path.join(RUN, "reports", "packet")


def rd(p):
    p = os.path.join(RUN, p)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


KEYS = ["team", "performance", "summary", "hypothesis", "change", "results", "conclusion", "validity"]


def parse_record(text):
    out, cur, official = {}, None, []
    for line in text.splitlines():
        if line.startswith("official #"):
            official.append(line.strip()); cur = None; continue
        m = re.match(r"^\s{0,4}(%s):\s?(.*)$" % "|".join(KEYS), line)
        if m:
            cur = m.group(1); out[cur] = m.group(2).strip(); continue
        if cur and line.strip():
            out[cur] += "\n" + line.strip()
    for k in list(out):
        out[k] = out[k].strip().strip('"')
    out["official_lines"] = official
    return out


def official_results(rid):
    d = os.path.join(RUN, "bank", rid, "official")
    if not os.path.isdir(d):
        return []
    res = []
    for n in sorted(os.listdir(d), key=lambda s: int(s) if s.isdigit() else 0):
        p = os.path.join(d, n, "results.json")
        if not os.path.exists(p):
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        keep = {k: j.get(k) for k in [
            "ok", "output_throughput", "output_throughput_trials", "output_throughput_cv_pct",
            "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms",
            "mtp_acceptance_length", "gsm8k_accuracy", "gate_gsm8k", "gate_latency",
            "gate_long_request", "vs_baseline_pct"]}
        keep["n"] = n
        res.append(keep)
    return res


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s


def md(text):
    lines = text.splitlines()
    out, i, stack = [], 0, []

    def close_lists(to=-1):
        while stack and stack[-1] > to:
            stack.pop(); out.append("</li></ul>")

    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1; continue
        if ln.startswith("```"):
            close_lists(); j = i + 1; buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j]); j += 1
            out.append("<pre>" + html.escape("\n".join(buf)) + "</pre>"); i = j + 1; continue
        m = re.match(r"^(#{1,6})\s+(.*)", ln)
        if m:
            close_lists(); lvl = min(len(m.group(1)) + 1, 6)
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>"); i += 1; continue
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            close_lists()
            hdr = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")]); j += 1
            t = "<div class='tw'><table><thead><tr>" + "".join(f"<th>{inline(h)}</th>" for h in hdr) + "</tr></thead><tbody>"
            for r in rows:
                t += "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
            out.append(t + "</tbody></table></div>"); i = j; continue
        m = re.match(r"^(\s*)(?:[-*]|\d+\.)\s+(.*)", ln)
        if m:
            ind = len(m.group(1))
            if not stack or ind > stack[-1]:
                stack.append(ind); out.append("<ul><li>")
            else:
                close_lists(ind)
                if stack and stack[-1] == ind:
                    out.append("</li><li>")
                else:
                    stack.append(ind); out.append("<ul><li>")
            out.append(inline(m.group(2))); i += 1; continue
        if stack and ln.startswith(" "):
            out.append(" " + inline(ln.strip())); i += 1; continue
        close_lists()
        buf, j = [ln.strip()], i + 1
        while j < len(lines) and lines[j].strip() and not re.match(r"^(\s*([-*]|\d+\.)\s|#|\||```)", lines[j]):
            buf.append(lines[j].strip()); j += 1
        out.append("<p>" + inline(" ".join(buf)) + "</p>"); i = j
    close_lists()
    return "\n".join(out)


def sections(diff):
    out, cur = {}, None
    for line in diff.splitlines(keepends=True):
        m = re.match(r"^diff --git a/(\S+) b/", line)
        if m:
            cur = m.group(1); out[cur] = ""
        if cur:
            out[cur] += line
    return out


def main():
    spec = json.load(open(os.path.join(SRC, "tree_spec.json")))
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "diffs")); os.makedirs(os.path.join(OUT, "data"))
    records = {}
    for d in spec["directions"]:
        for n in d["nodes"]:
            rid, parent = n["id"], n.get("parent")
            diff = rd(f"bank/{rid}/change.diff")
            pdiff = rd(f"bank/{parent}/change.diff") if parent else ""
            if diff:
                open(os.path.join(OUT, "diffs", f"{rid}.diff"), "w").write(diff)
            records[rid] = dict(
                rec=parse_record(rd(f"bank/{rid}/record.md")), official=official_results(rid),
                serve_args=rd(f"bank/{rid}/serve_args.txt").strip(), diff=diff,
                diff_files=re.findall(r"^\+\+\+ b/(.*)$", diff, re.M),
                same_code_as_parent=bool(parent) and diff == pdiff,
                delta_vs_parent="".join(v for f, v in sections(diff).items() if sections(pdiff).get(f) != v),
                files_changed_vs_parent=[f for f, v in sections(diff).items() if sections(pdiff).get(f) != v])
    files = ["task_definition.md", "assumption_ledger.md", "focus.md", "shared_kb.md", "ideas.md",
             "tree.md", "champion.md", "journal.md"]
    files += [f"reports/{x}" for x in sorted(os.listdir(os.path.join(RUN, "reports"))) if x.endswith(".md")]
    for f in files:
        t = rd(f)
        if t:
            open(os.path.join(OUT, "data", os.path.basename(f)), "w").write(t)
    pages = {
        "overview": md(open(os.path.join(SRC, "overview.md")).read()),
        "kb": md(re.sub(r"^(KB\d+)\.\s*", r"- **\1.** ", rd("shared_kb.md"), flags=re.M)),
        "ideas": md(rd("ideas.md")),
        "rules": md(rd("task_definition.md")) + "<hr>" + md(rd("assumption_ledger.md")),
    }
    data = dict(spec=spec, records=records, base=official_results("BASE"), pages=pages)
    tpl = open(os.path.join(SRC, "template.html")).read()
    blob = json.dumps(data).replace("</", "<\\/")
    open(os.path.join(OUT, "index.html"), "w").write(tpl.replace("/*__DATA__*/null", blob))
    open(os.path.join(OUT, "README.txt"), "w").write(
        "AutoInfer run test-3: handoff packet. Open index.html in a browser (works offline, no server).\n"
        "diffs/<record>.diff: each record's change.diff against the task's baseline vLLM tree.\n"
        "data/: copies of the run's key markdown files (task definition, ledger, KB, ideas, tree, reports, journal).\n")
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT) for f in fs)
    print("wrote", OUT, size, "bytes")


if __name__ == "__main__":
    main()
