import sys, random
from types import SimpleNamespace
import torch
sys.path.insert(0, ".")
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
def make_request(rid, toks, bs, hf, la=0):
    sp = SamplingParams(max_tokens=17); sp.update_from_generation_config({}, eos_token_id=100)
    return Request(request_id=rid, prompt_token_ids=toks, mm_features=None, sampling_params=sp, pooling_params=None,
        lora_request=None, cache_salt=None, block_hasher=get_request_block_hasher(bs, hf, la), session_id=None)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import *

BS = 1600; HB = 16
def run(lookahead, drop, shared, nblocks, conc=16, nreq=60, plen_sys=45, seed=0, unit=HB, junction_on=True):
    init_none_hash(sha256)
    cfg = KVCacheConfig(num_blocks=nblocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["full"], FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32)),
        KVCacheGroupSpec(["mamba"], MambaSpec(block_size=BS, shapes=((1,1),), dtypes=(torch.float32,), mamba_cache_mode="align", num_speculative_blocks=2)),
    ])
    m = KVCacheManager(cfg, max_model_len=262144, scheduler_block_size=BS, hash_block_size=unit, enable_caching=True,
                       use_eagle=drop, num_prefill_lookahead=lookahead, num_hash_lookahead_tokens=lookahead,
                       enable_mamba_shared_prefix_checkpoint=shared)
    stub = SimpleNamespace(block_size=BS, cache_config=SimpleNamespace(block_size=BS), use_eagle_block_drop=drop,
        num_hash_lookahead_tokens=lookahead, max_num_scheduled_tokens=2048, scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=True, mamba_shared_prefix_checkpoint=shared and junction_on, hash_block_size=unit,
        mamba_has_prefill_checkpoint_blocks=False, mamba_prefill_checkpoint_alignment=None)
    rnd = random.Random(seed)
    sysp = [1000+i for i in range(plen_sys)]
    jbs=[]; hits=[]; steps=[]; running=[]
    for i in range(nreq):
        tail = [rnd.randrange(2000, 50000) for _ in range(rnd.randrange(30, 400))]
        r = make_request(str(i), sysp+tail, unit, sha256, lookahead)
        # admit; retire oldest if full
        if len(running) >= conc:
            old = running.pop(0); m.free(old)
        blocks, ncomp, jb = m.get_computed_blocks(r)
        r.shared_prefix_boundary = jb
        res = m.allocate_slots(r, 0, ncomp, blocks) if False else None
        hits.append(ncomp); jbs.append(jb)
        r.num_computed_tokens = ncomp
        nsteps=0; first=True
        while r.num_computed_tokens < r.num_tokens:
            n = Scheduler._mamba_block_aligned_split(stub, r, r.num_tokens - r.num_computed_tokens)
            if n == 0: break
            if first:
                a = m.allocate_slots(r, n, ncomp, blocks, num_lookahead_tokens=2); first=False
            else:
                a = m.allocate_slots(r, n, num_lookahead_tokens=2)
            assert a is not None, "alloc fail"
            r.num_computed_tokens += n; nsteps+=1
            m.new_step_starts()
        steps.append(nsteps); running.append(r)
        if False:
            hb=r.block_hashes
            print('req',i,'jb',jb,'steps',nsteps, 'cached@43/44/45/32/31:', [(k,[bool(m.block_pool.get_cached_block(hb[k],[g])) for g in (0,1)]) for k in (31,32,42,43,44) if k < len(hb)] if unit==1 else '')
    print('jb',jbs[:8]); return hits, steps
if __name__=="__main__":
    for nb in (140, 120, 100, 90):
      for ps in (45,):
        try:
            h,s_=run(1,False,True,nb,unit=1,nreq=200,plen_sys=ps)
            print(f"nblocks={nb} sys={ps}: n>=ps-2:{sum(x>=ps-2 for x in h)}/{len(h)} mean_hit={sum(h)/len(h):.1f} mean_steps={sum(s_)/len(s_):.2f}")
        except AssertionError as e: print(nb, ps, "alloc fail")
