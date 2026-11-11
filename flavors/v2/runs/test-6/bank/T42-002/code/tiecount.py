# CPU: count exact top-1 ties in the adversarial one-hot / two-hot rows (same generator sequence as cs_test.py)
import torch
torch.set_num_threads(32)
from huggingface_hub import hf_hub_download
from safetensors import safe_open
f = hf_hub_download('Qwen/Qwen3.8-27B-FP8', 'outside.safetensors')
with safe_open(f, 'pt') as s: W = s.get_tensor('lm_head.weight').float()
K = 5120
g = torch.Generator(device='cpu').manual_seed(1)
for sc in [0.25, 1.0, 4.0]:
    for _ in range(4): torch.randn(64, K, generator=g)
tot = {}
for nm in ['onehot'] * 4 + ['twohot'] * 4:
    x = torch.zeros(64, K)
    if nm == 'onehot':
        idx = torch.randint(0, K, (64,), generator=g); x[torch.arange(64), idx] = torch.randn(64, generator=g).abs() * 50 + 1
    else:
        i1 = torch.randint(0, K, (64,), generator=g); i2 = torch.randint(0, K, (64,), generator=g); x[torch.arange(64), i1] = 8.0; x[torch.arange(64), i2] = 8.0
    L = (x.bfloat16().float() @ W.T).bfloat16().float()
    mx = L.max(1, keepdim=True).values
    nt = (L == mx).sum(1)
    t = tot.setdefault(nm, [0, 0, 0]); t[0] += 64; t[1] += int((nt > 1).sum()); t[2] = max(t[2], int(nt.max()))
print({k: f'rows {v[0]}, rows with a top-1 tie {v[1]}, max tied ids {v[2]}' for k, v in tot.items()})
