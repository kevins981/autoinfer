import json, sys, urllib.request, uuid, random, glob, os, time
from concurrent.futures import ThreadPoolExecutor
url, model = sys.argv[1], sys.argv[2]
sg = glob.glob(os.environ["HF_HUB_CACHE"]+"/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/*/ShareGPT_V3_unfiltered_cleaned_split.json")[0]
data = json.load(open(sg)); rnd = random.Random(1); rnd.shuffle(data)
prompts = []
for c in data:
    cv = c.get("conversations", [])
    if len(cv) >= 2 and cv[0]["from"] == "human" and 20 < len(cv[0]["value"]) < 3000: prompts.append(cv[0]["value"])
    if len(prompts) == 96: break
def post(body):
    r = urllib.request.Request(url+"/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
def metric(name):
    t = urllib.request.urlopen(url+"/metrics").read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(name) and not l.startswith("#"))
def run(salted, label):
    d0, a0 = metric("vllm:spec_decode_num_drafts_total"), metric("vllm:spec_decode_num_accepted_tokens_total")
    h0 = metric("vllm:prefix_cache_hits_total"); t0 = time.time()
    def one(p):
        b = dict(model=model, messages=[{"role":"user","content":p}], max_tokens=200, temperature=0)
        if salted: b["cache_salt"] = uuid.uuid4().hex
        return post(b)["usage"]["completion_tokens"]
    with ThreadPoolExecutor(16) as ex: toks = sum(ex.map(one, prompts))
    d1, a1 = metric("vllm:spec_decode_num_drafts_total"), metric("vllm:spec_decode_num_accepted_tokens_total")
    print(f"{label}: tokens={toks} drafts={int(d1-d0)} acc_len={1+(a1-a0)/(d1-d0):.3f} hit_tokens={int(metric('vllm:prefix_cache_hits_total')-h0)} time={time.time()-t0:.1f}s tok/s={toks/(time.time()-t0):.0f}")
run(True, "salted(no hits) ")
run(False, "unsalted(hits)  ")
run(True, "salted(no hits) ")
run(False, "unsalted(hits)  ")
