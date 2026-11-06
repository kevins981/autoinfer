"""Turn an eval output folder into results.json (and print it).
usage: python summarize.py <output dir> [baseline results.json]"""
import json, re, sys
from pathlib import Path

SPEC = "vllm:spec_decode_num_"
GSM8K_MIN = 0.896          # TASK.md accuracy gate
LATENCY_MAX_WORSE = 0.20   # TASK.md latency guard (p99 TTFT / TPOT vs BASE)


def read_metrics(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.startswith(SPEC) or "_created" in line:
            continue
        m = re.match(r"(vllm:spec_decode_num_[a-z_]+?)(?:_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        pos = re.search(r'position="(\d+)"', labels)
        key = f"{name}[{pos.group(1)}]" if pos else name
        out[key] = out.get(key, 0.0) + val
    return out


def main():
    d = Path(sys.argv[1])
    res = {"ok": not (d / "failed_stage").exists()}
    if not res["ok"]:
        res["failed_stage"] = (d / "failed_stage").read_text().strip()
    b = d / "bench_c16.json"
    if not b.exists():                              # unofficial benches (scripts/lib.sh): bench.json
        b = d / "bench.json"
    if b.exists():
        r = json.loads(b.read_text())
        res.update({k: r.get(k) for k in (
            "completed", "output_throughput", "total_token_throughput", "median_ttft_ms", "p99_ttft_ms",
            "median_tpot_ms", "p99_tpot_ms", "median_itl_ms", "median_e2el_ms")})
        before, after = read_metrics(d / "metrics_before.txt"), read_metrics(d / "metrics_after.txt")
        delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
        drafts = delta.get(SPEC + "drafts", 0)
        if drafts:
            res["mtp_acceptance_length"] = round(1 + delta.get(SPEC + "accepted_tokens", 0) / drafts, 3)
            res["mtp_acceptance_per_position"] = [
                round(delta[k] / drafts, 3) for k in sorted(k for k in delta if "accepted_tokens_per_pos[" in k)]
    g = d / "gsm8k.json"
    if g.exists():
        gj = json.loads(g.read_text())
        res["gsm8k_accuracy"] = round(gj["accuracy"], 4)
        res["gsm8k_no_answer"] = gj["no_answer"]
    # Gates (TASK.md): GSM8K >= 89.6%; TTFT p99 and TPOT p99 at most 20% worse than BASE.
    if "gsm8k_accuracy" in res:
        res["gate_gsm8k"] = "pass" if res["gsm8k_accuracy"] >= GSM8K_MIN else "FAIL"
    if len(sys.argv) > 2 and Path(sys.argv[2]).exists() and res.get("output_throughput"):
        base = json.loads(Path(sys.argv[2]).read_text())
        if base.get("output_throughput"):
            res["vs_baseline_pct"] = round(100 * (res["output_throughput"] / base["output_throughput"] - 1), 2)
        worse = {k: res[k] / base[k] for k in ("p99_ttft_ms", "p99_tpot_ms") if base.get(k) and res.get(k)}
        res["p99_vs_baseline"] = {k: round(v, 3) for k, v in worse.items()}
        res["gate_latency"] = "pass" if all(v <= 1 + LATENCY_MAX_WORSE for v in worse.values()) else "FAIL"
    (d / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
