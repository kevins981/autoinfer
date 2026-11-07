"""Long-request check: one max-length request must still work (the memory picture can't shrink it).

Usage: python long_request.py [--tokens 128000] [--max-tokens 64] --out FILE.json [--url URL]
Builds a prompt of exactly --tokens token ids from ShareGPT text (deterministic), sends it to
/v1/completions (temperature 0), then checks the server still answers a short request.
Pass = both requests return HTTP 200 with at least one generated token. Exit 0 on pass, 1 on fail.
"""
import argparse, glob, json, os, time, urllib.error, urllib.request


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:2000]}
    except Exception as e:                                   # connection reset = server died
        return None, {"error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    data_path = glob.glob(os.path.join(os.environ["HF_HUB_CACHE"], "datasets--anon8231489123--ShareGPT_Vicuna_unfiltered",
                                       "snapshots", "*", "ShareGPT_V3_unfiltered_cleaned_split.json"))[0]
    ids = []
    for d in json.load(open(data_path)):
        for c in d.get("conversations", []):
            ids += tok.encode(c["value"], add_special_tokens=False)
        if len(ids) >= a.tokens:
            break
    ids = ids[:a.tokens]
    t0 = time.time()
    st, r = post(a.url, {"model": a.model, "prompt": ids, "max_tokens": a.max_tokens, "temperature": 0}, 1800)
    secs = round(time.time() - t0, 1)
    st2, r2 = post(a.url, {"model": a.model, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}, 120)
    gen = (r.get("usage") or {}).get("completion_tokens", 0)
    ok = st == 200 and gen > 0 and st2 == 200
    res = {"pass": ok, "prompt_tokens": len(ids), "status": st, "completion_tokens": gen, "seconds": secs,
           "text": (r.get("choices") or [{}])[0].get("text"), "finish_reason": (r.get("choices") or [{}])[0].get("finish_reason"),
           "error": r.get("error"), "after_status": st2, "after_error": r2.get("error")}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"long request {len(ids)} tokens: {'pass' if ok else 'FAIL'} (status {st}, {gen} tokens, {secs}s; after: {st2})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
