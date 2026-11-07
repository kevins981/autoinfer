r"""Regenerate an answer key's graph outline from its nodes' builds_on / alternative_to edges.

usage: python3 keygraph.py <answer_key.md> [--check]
Nodes are `### \`id\`: title` sections with `- builds_on: BASE | id [+ id]`, optional
`- alternative_to: id`, and `- status: ...`. The outline (between <!-- graph:start --> and
<!-- graph:end -->) puts each node under its first parent, names other parents "+ X", and is rewritten
in place. Fails on unknown ids and cycles. --check: only report whether the outline is current.
"""
import re, sys


def parse(text):
    nodes, cur, fence = {}, None, False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        m = re.match(r"### `([^`]+)`", line)
        if m:
            cur = m.group(1); nodes[cur] = {"builds_on": [], "alt": [], "status": "?"}
            continue
        if line.startswith("## "):
            cur = None
        if not cur:
            continue
        m = re.match(r"- (builds_on|alternative_to|status):\s*(.*)", line)
        if m:
            k, v = m.groups()
            if k == "status":
                nodes[cur]["status"] = v.strip()
            else:
                ids = [x.strip() for x in v.split("+") if x.strip()]
                nodes[cur]["builds_on" if k == "builds_on" else "alt"] = ids
    return nodes


def outline(nodes):
    for n, d in nodes.items():
        for p in d["builds_on"] + d["alt"]:
            if p != "BASE" and p not in nodes:
                sys.exit(f"{n}: unknown id {p!r}")
    seen, stack = set(), set()
    def visit(n):                                   # cycle check over builds_on
        if n in stack:
            sys.exit(f"cycle through {n}")
        if n in seen or n == "BASE":
            return
        stack.add(n)
        for p in nodes[n]["builds_on"]:
            visit(p)
        stack.discard(n); seen.add(n)
    for n in nodes:
        visit(n)
    kids = {"BASE": []}
    for n, d in nodes.items():                      # file order = sibling order
        kids.setdefault(d["builds_on"][0] if d["builds_on"] else "BASE", []).append(n)
    out = ["BASE"]
    def walk(n, pre):
        ch = kids.get(n, [])
        for i, c in enumerate(ch):
            last = i == len(ch) - 1
            d = nodes[c]
            extra = "".join(f"  + {p}" for p in d["builds_on"][1:])
            extra += "".join(f"  alt: {a}" for a in d["alt"])
            out.append(f"{pre}{'└─ ' if last else '├─ '}{c}  [{d['status']}]{extra}")
            walk(c, pre + ("   " if last else "│  "))
    walk("BASE", "")
    return "\n".join(out)


def main():
    path = sys.argv[1]
    text = open(path).read()
    new = "<!-- graph:start -->\n```\n" + outline(parse(text)) + "\n```\n<!-- graph:end -->"
    upd = re.sub(r"<!-- graph:start -->.*?<!-- graph:end -->", lambda _: new, text, flags=re.S)
    if "--check" in sys.argv:
        print("current" if upd == text else "STALE: rerun without --check"); sys.exit(upd != text)
    open(path, "w").write(upd)
    print(new)


if __name__ == "__main__":
    main()
