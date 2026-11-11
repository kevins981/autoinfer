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
tails = ["Explain how a hash map handles collisions in detail.", "Write a short story about a lighthouse keeper and a storm.",
         "Describe how photosynthesis works step by step.", "List and explain five common sorting algorithms.",
         "What happened during the French Revolution? Give a summary.", "Explain the difference between TCP and UDP with examples."]
sysm = "You are a patient tutor named Sage who explains ideas clearly, uses simple words, and gives concrete examples whenever possible. Unique-%s" % uuid.uuid4().hex[:6]
ref, accr = [], []
# 1. references: salted full prefill, twice (determinism)
refs, a_ref, _ = acc(lambda: [req(sysm, t, salt=uuid.uuid4().hex) for t in tails])
refs2 = [req(sysm, t, salt=uuid.uuid4().hex) for t in tails]
print("determinism ref vs ref2:", [cmp(x, y) for x, y in zip(refs, refs2)][:3], "acc(ref)=%.3f" % a_ref)
# 2. sibling 1 (no junction), creator (chunk split), then hits
req(sysm, "Say hello.")
cre, a_cre, _ = acc(lambda: req(sysm, tails[0]))
print("creator vs ref[0]:", cmp(cre, refs[0]), "prompt_tokens", cre[2], "acc=%.3f" % a_cre)
h0 = metric("vllm:prefix_cache_hits_total")
hits, a_hit, nd = acc(lambda: [req(sysm, t) for t in tails[1:]])
print("hit_tokens total", int(metric("vllm:prefix_cache_hits_total")-h0), "for", len(tails)-1, "requests")
for i, h in enumerate(hits, 1): print(f" hit req{i} vs ref: (same_prefix_toks, n, max|dlogprob| over common) =", cmp(h, refs[i]))
print("acc(hit)=%.3f (drafts %d)  acc(ref, full prefill)=%.3f" % (a_hit, nd, a_ref))
