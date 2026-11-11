import sys, random, time
sys.path.insert(0, "/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-21.w1")
import sim
from sim import *
lens = [int(x) for x in open("/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/agents/leader-21/prompt_lens.txt").read().split() if x.strip().isdigit()]
STEPS=[]; HITS=[]
def trace(unit, shared, nreq=400, conc=16, nblocks=200000, la=1):
    init_none_hash(sha256)
    cfg = KVCacheConfig(num_blocks=nblocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["full"], FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32)),
        KVCacheGroupSpec(["mamba"], MambaSpec(block_size=BS, shapes=((1,1),), dtypes=(torch.float32,), mamba_cache_mode="align", num_speculative_blocks=2)),
    ])
    m = KVCacheManager(cfg, max_model_len=262144, scheduler_block_size=BS, hash_block_size=unit, enable_caching=True,
                       use_eagle=False, num_prefill_lookahead=la, num_hash_lookahead_tokens=la, enable_mamba_shared_prefix_checkpoint=shared)
    stub = SimpleNamespace(block_size=BS, cache_config=SimpleNamespace(block_size=BS), use_eagle_block_drop=False,
        num_hash_lookahead_tokens=la, max_num_scheduled_tokens=2048, scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=unit<BS, mamba_shared_prefix_checkpoint=shared, hash_block_size=unit,
        mamba_has_prefill_checkpoint_blocks=False, mamba_prefill_checkpoint_alignment=None)
    rnd = random.Random(0); sysp=[1000+i for i in range(45)]
    run=[]; t_admit=0; t_step=0; nsteps=0; nadm=0; maxstep=0
    pending = list(range(nreq)); done=0
    while done < nreq:
        t0=time.perf_counter()
        # admit
        while len(run) < conc and pending:
            i = pending.pop(0)
            L = max(10, lens[i % len(lens)]-45)
            r = make_request(str(i), sysp+[rnd.randrange(3000,3004), rnd.randrange(3010,3014), rnd.randrange(3020,3023)]+[rnd.randrange(2000,50000) for _ in range(L)], unit, sha256, la)
            blocks, nc, jb = m.get_computed_blocks(r); r.shared_prefix_boundary = jb
            r.num_computed_tokens = nc
            first=True; nst=0
            while r.num_computed_tokens < r.num_tokens:
                n = Scheduler._mamba_block_aligned_split(stub, r, r.num_tokens-r.num_computed_tokens)
                if n==0: break
                a = m.allocate_slots(r, n, nc, blocks, num_lookahead_tokens=2) if first else m.allocate_slots(r, n, num_lookahead_tokens=2)
                first=False
                assert a is not None, (i, L, nc, jb, r.num_computed_tokens, n)
                r.num_computed_tokens += n; m.new_step_starts(); nst+=1
            STEPS.append(nst); HITS.append(nc); r._left = 228; run.append(r); nadm+=1
        t1=time.perf_counter(); t_admit+=t1-t0
        # decode step: ~2.33 tokens per request
        t0=time.perf_counter()
        for r in list(run):
            k = 2 if rnd.random()<0.67 else 3
            m.allocate_slots(r, 1+2, num_lookahead_tokens=2)  # scheduled tokens incl drafts
            r.append_output_token_ids([rnd.randrange(2000,50000) for _ in range(k)])
            r.num_computed_tokens += k
            r._left -= k
            if r._left <= 0: run.remove(r); m.free(r); done+=1
        m.new_step_starts()
        dt=time.perf_counter()-t0; t_step+=dt; nsteps+=1; maxstep=max(maxstep,dt)
    return t_admit/nadm*1e3, t_step/nsteps*1e3, maxstep*1e3
for name,(u,sh) in {"A(unit1600,off)":(1600,False),"B unit1":(1,True),"B unit16":(16,True)}.items():
    a,s,mx = trace(u,sh)
    print(f'  steps/request {sum(STEPS)/len(STEPS):.2f}, hit>=40: {sum(h>=40 for h in HITS)}/{len(HITS)}'); STEPS.clear(); HITS.clear()
    print(f"{name:18s} admit {a:.2f} ms/request   decode-step CPU {s:.3f} ms/step (max {mx:.1f})")
