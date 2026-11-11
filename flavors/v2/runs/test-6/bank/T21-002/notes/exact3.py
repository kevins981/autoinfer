import json, sys, urllib.request, uuid
url, model = sys.argv[1], sys.argv[2]
def post(body):
    r = urllib.request.Request(url+"/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))
def metric(name):
    t = urllib.request.urlopen(url+"/metrics").read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(name) and not l.startswith("#"))
def req(system, user, salt=None, mt=160):
    msgs = ([{"role":"system","content":system}] if system else []) + [{"role":"user","content":user}]
    b = dict(model=model, messages=msgs, max_tokens=mt, temperature=0, logprobs=True, top_logprobs=1)
    if salt: b["cache_salt"] = salt
    r = post(b)
    c = r["choices"][0]["logprobs"]["content"]
    return [x["token"] for x in c], [x["logprob"] for x in c], r["usage"]["prompt_tokens"]
def cmp(a, b):
    n = min(len(a[0]), len(b[0])); k = 0
    while k < n and a[0][k] == b[0][k]: k += 1
    return k, n, max([abs(a[1][i]-b[1][i]) for i in range(k)] or [0.0])
def acc(fn):
    d0, a0 = metric("vllm:spec_decode_num_drafts_total"), metric("vllm:spec_decode_num_accepted_tokens_total")
    out = fn()
    d1, a1 = metric("vllm:spec_decode_num_drafts_total"), metric("vllm:spec_decode_num_accepted_tokens_total")
    return out, 1 + (a1-a0)/max(d1-d0, 1), int(d1-d0)

from concurrent.futures import ThreadPoolExecutor
tails = ["Explain how a hash map handles collisions in detail.", "Write a short story about a lighthouse keeper and a storm.",
 "Describe how photosynthesis works step by step.", "List and explain five common sorting algorithms.",
 "What happened during the French Revolution? Give a summary.", "Explain the difference between TCP and UDP with examples.",
 "How do vaccines train the immune system?", "Explain what a mutex is and when deadlocks occur.",
 "Summarize the plot of Romeo and Juliet.", "Why do we have leap years? Explain the math.",
 "Describe how a transformer neural network works.", "Give a recipe for a simple tomato soup with steps."]
def par(fn, items):
    with ThreadPoolExecutor(16) as ex: return list(ex.map(fn, items))
for rnd in range(2):
    sysm = "You are a patient tutor named Sage who explains ideas clearly, uses simple words, and gives concrete examples. Unique-%s" % uuid.uuid4().hex[:6]
    refs, a_ref, _ = acc(lambda: par(lambda t: req(sysm, t, salt=uuid.uuid4().hex, mt=200), tails))
    req(sysm, "Say hello.")
    req(sysm, "Say goodbye.")   # creator
    h0 = metric("vllm:prefix_cache_hits_total")
    hits, a_hit, nd = acc(lambda: par(lambda t: req(sysm, t, mt=200), tails))
    print(f"round {rnd}: concurrent 12 hit reqs, hit_tokens total {int(metric('vllm:prefix_cache_hits_total')-h0)}, acc(hit)={a_hit:.3f} drafts={nd}; acc(ref full prefill)={a_ref:.3f}")
    print("   vs ref (same_prefix, n, max dlogprob):", [cmp(h, r)[:2]+(round(cmp(h,r)[2],3),) for h, r in zip(hits, refs)])
