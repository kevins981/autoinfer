"""T47 stage 1 client: 15 long decoders, then single prompts of controlled token length."""
import json, random, sys, threading, time, urllib.request

PORT, MODEL = sys.argv[1], sys.argv[2]
PROBES = [int(x) for x in sys.argv[3].split(",")]
URL = f"http://localhost:{PORT}/v1/completions"
rng = random.Random(0)


def post(ids, max_tokens, timeout=600):
    body = json.dumps({"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": 0,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return time.time() - t0, out["usage"]


def toks(n):
    return [rng.randrange(1000, 100000) for _ in range(n)]


bg = []
for i in range(15):
    ids = toks(16)
    t = threading.Thread(target=lambda ids=ids: post(ids, 6000), daemon=True)
    t.start()
    bg.append(t)
    time.sleep(0.15)  # stagger so the decoders' own prefills are small steps
time.sleep(8)
for p in PROBES:
    dt, u = post(toks(p), 1)
    print(f"probe P={p}: {dt:.2f} s usage {u}", flush=True)
    time.sleep(1.5)
print("done", flush=True)
