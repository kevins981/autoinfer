import json, sys, urllib.request, re, uuid
url, model = sys.argv[1], sys.argv[2]
def post(body):
    r = urllib.request.Request(url+"/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))
def metric(name):
    t = urllib.request.urlopen(url+"/metrics").read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(name) and not l.startswith("#"))
def req(system, user, salt=None, kw=None):
    msgs = ([{"role":"system","content":system}] if system else []) + [{"role":"user","content":user}]
    b = dict(model=model, messages=msgs, max_tokens=48, temperature=0, logprobs=True, top_logprobs=1)
    if salt: b["cache_salt"] = salt
    if kw: b["chat_template_kwargs"] = kw
    return post(b)
prefixes = {
 "default_template(45tok)": (None, None),
 "custom_sys_short": ("You are a terse assistant. Answer in one sentence.", None),
 "custom_sys_long": ("You are a meticulous senior engineer. Always reason step by step, cite assumptions, keep answers short, avoid speculation, and never invent facts. If a question is ambiguous, state the most likely interpretation first, then answer it directly and concisely.", None),
 "reasoning_effort_low": (None, {"reasoning_effort":"low"}),
}
tails = ["What is the capital of France and why is it famous?", "Explain how a hash map handles collisions.", "Write a haiku about autumn rain.",
         "How many legs do three spiders have in total?", "Give two tips to sleep better.", "Why is the sky blue?"]
allok = True
for name,(sysm,kw) in prefixes.items():
    # two siblings create the junction, then compare hits vs salted full prefill
    req(sysm, tails[0], kw=kw); req(sysm, tails[1], kw=kw)
    for t in tails[2:]:
        h0, q0 = metric("vllm:prefix_cache_hits_total"), metric("vllm:prefix_cache_queries_total")
        a = req(sysm, t, kw=kw)
        h1, q1 = metric("vllm:prefix_cache_hits_total"), metric("vllm:prefix_cache_queries_total")
        b = req(sysm, t, salt=uuid.uuid4().hex, kw=kw)
        ta = [x["token"] for x in a["choices"][0]["logprobs"]["content"]]
        tb = [x["token"] for x in b["choices"][0]["logprobs"]["content"]]
        la = [x["logprob"] for x in a["choices"][0]["logprobs"]["content"]]
        lb = [x["logprob"] for x in b["choices"][0]["logprobs"]["content"]]
        n = min(len(la), len(lb)); k = 0
        while k < n and ta[k] == tb[k]: k += 1
        d0 = abs(la[0]-lb[0]); dk = max([abs(la[i]-lb[i]) for i in range(k)] or [0])
        print(f"{name:26s} prompt_tokens={a['usage']['prompt_tokens']:4d} hit_tokens={int(h1-h0):4d}/{int(q1-q0)} first_tok_same={ta[0]==tb[0]} same_prefix_tokens={k}/{n} |dlogprob| first={d0:.4f} max_over_common={dk:.4f}")
