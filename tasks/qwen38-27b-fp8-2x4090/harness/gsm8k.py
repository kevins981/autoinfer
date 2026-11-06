"""Quick accuracy gate: GSM8K subset against the running server (temperature 0, thinking off).

Usage: python gsm8k.py [--n 250] [--conc 16] --out FILE.json [--url URL]
Writes per-question outputs + accuracy to --out and prints accuracy. Fixed question subset (seed 0), so scores are
comparable across configs. With n=250 the standard error is ~2.5 points: treat drops under ~4 points as noise.
Stdlib only. Data: data/gsm8k_test.jsonl (openai/grade-school-math test split).
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import time
import urllib.request

PROMPT = ("Solve the following math problem step by step. Finish with a final line of the form "
          "'Answer: <number>' containing only the final number.\n\n{q}")


def ask(q: str, model: str, url: str) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
            "temperature": 0, "max_tokens": 1500,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def num(s: str):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    rows = [json.loads(l) for l in open(f"{root}/data/gsm8k_test.jsonl")]
    random.Random(0).shuffle(rows)
    rows = rows[:a.n]
    gold = [num(r["answer"].split("####")[-1]) for r in rows]

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as ex:
        outs = list(ex.map(lambda r: ask(r["question"], a.model, a.url), rows))
    dt = time.time() - t0

    preds = []
    for o in outs:
        m = re.search(r"Answer:\s*(.+)", o)
        preds.append(num(m.group(1)) if m else num(o))
    ok = [p is not None and g is not None and abs(p - g) < 1e-4 for p, g in zip(preds, gold)]
    acc = sum(ok) / len(ok)
    no_answer = sum(1 for o in outs if not re.search(r"Answer:", o))
    print(f"GSM8K n={len(rows)} accuracy {100*acc:.1f}% (+-{100*(acc*(1-acc)/len(ok))**.5:.1f})  "
          f"no 'Answer:' line: {no_answer}  time {dt:.0f}s")
    path = a.out
    json.dump({"n": len(rows), "accuracy": acc, "no_answer": no_answer, "seconds": dt,
               "items": [{"q": r["question"], "gold": g, "pred": p, "out": o} for r, g, p, o in zip(rows, gold, preds, outs)]},
              open(path, "w"))
    print("saved", path)


if __name__ == "__main__":
    main()
