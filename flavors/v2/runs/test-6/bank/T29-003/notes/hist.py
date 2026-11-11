# Per start of an abba dir: tok/s, latencies, acceptance, ITL histogram split (Prometheus deltas), memory lines.
import json, re, sys, glob, os
def parse(f):
    d = {}
    for l in open(f):
        if l.startswith('vllm:inter_token_latency_seconds') or l.startswith('vllm:iteration_tokens_total'):
            m = re.match(r'(\S+?)(\{.*\})? (\S+)$', l.strip())
            if not m: continue
            name, lab, v = m.groups(); le = re.search(r'le="([^"]+)"', lab or '')
            d[(name, le.group(1) if le else '')] = float(v)
    return d
for r in sorted(glob.glob(os.path.join(sys.argv[1], '*_*/'))):
    try:
        a = parse(r + 'metrics_before.txt'); b = parse(r + 'metrics_after.txt'); bj = json.load(open(r + 'bench.json'))
    except Exception as e:
        print(r, 'missing', e); continue
    dd = {k: b[k] - a.get(k, 0) for k in b}
    n = dd[('vllm:inter_token_latency_seconds_count', '')]; tot = dd[('vllm:inter_token_latency_seconds_sum', '')]
    le50 = dd[('vllm:inter_token_latency_seconds_bucket', '0.05')]; le25 = dd[('vllm:inter_token_latency_seconds_bucket', '0.025')]
    med = bj['median_itl_ms'] / 1000
    dec = le50 * med; mix = tot - dec
    srv = open(r + 'server.log').read()
    kv = re.search(r'GPU KV cache size: ([\d,]+) tokens', srv); ml = re.findall(r'Model loading took ([\d.]+) GiB', srv)
    print(f"{os.path.basename(r.rstrip('/'))}: {bj['output_throughput']:.1f} tok/s | median ITL {bj['median_itl_ms']:.2f} mean {bj['mean_itl_ms']:.2f} p99 {bj['p99_itl_ms']:.0f} | "
          f"TTFT p50/p99 {bj['median_ttft_ms']:.0f}/{bj['p99_ttft_ms']:.0f} TPOT p99 {bj['p99_tpot_ms']:.1f} | acc {bj['spec_decode_acceptance_length']:.3f} | "
          f"ITL n {n:.0f} sum {tot:.1f}s, <=25ms {le25:.0f}, <=50ms {le50:.0f}, >50ms {n - le50:.0f}; mixed-like sum {mix:.1f}s (mean {1000 * mix / max(1, n - le50):.1f} ms), "
          f"decode-like est {dec:.1f}s | KV {kv.group(1) if kv else '?'} | model load {ml} GiB")
