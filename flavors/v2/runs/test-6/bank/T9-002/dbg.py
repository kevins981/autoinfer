import os, torch
from types import SimpleNamespace
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule as fla_gdr, fused_post_conv_prep
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices, prepare_chunk_offsets
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
from vllm.models.qwen3_5.nvidia.ops.fused_gdn_gated_norm import fused_gdn_gated_rms_norm
dev = torch.device('cuda'); bf = torch.bfloat16; R = 384
torch.manual_seed(0)
qout = torch.randn(R, 8192, device=dev, dtype=bf); a = torch.randn(R, 24, device=dev, dtype=bf); b = torch.randn(R, 24, device=dev, dtype=bf)
cw = (torch.randn(5120, 4, device=dev, dtype=bf) * 0.3).contiguous(); A_log = torch.randn(24, device=dev) * 0.3; dtb = torch.randn(24, device=dev, dtype=bf) * 0.1
cu_cpu = torch.tensor([0, R], dtype=torch.int32); cu = cu_cpu.to(dev)
nd, bp, tp = compute_causal_conv1d_metadata(cu_cpu, device=dev); torch.cuda.synchronize()
meta = SimpleNamespace(nums_dict=nd, batch_ptr=bp, token_chunk_offset_ptr=tp)
ci = prepare_chunk_indices(cu_cpu, 64).to(dev); co = prepare_chunk_offsets(cu_cpu, 64).to(dev)
print('ci', ci.shape, ci.dtype, ci[:3].tolist(), 'co', co.tolist(), 'nd tot', nd[8]['tot'], 'bp[:5]', bp[:5].tolist())
def S(n, t): t = t.float(); print(n, tuple(t.shape), 'absmax', t.abs().max().item(), 'nan', torch.isnan(t).any().item(), 'nzero', (t == 0).float().mean().item())
mixed = qout[:, :5120].contiguous(); S('mixed', mixed)
cst = torch.zeros(1, 5120, 3, device=dev, dtype=bf)
x = causal_conv1d_fn(mixed.transpose(0, 1), cw, None, activation='silu', conv_states=cst, has_initial_state=torch.tensor([False], device=dev), cache_indices=torch.zeros(1, dtype=torch.int32, device=dev), query_start_loc=cu, metadata=meta).transpose(0, 1)
S('conv', x); S('convstate', cst)
q, k, v, g, beta = fused_post_conv_prep(conv_output=x, a=a, b=b, A_log=A_log, dt_bias=dtb, num_k_heads=8, head_k_dim=128, head_v_dim=128, apply_l2norm=True, output_g_exp=False)
for n, t in zip('qkvg', (q, k, v, g)): S(n, t)
S('beta', beta)
o, fs = fla_gdr(q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0), g=g.unsqueeze(0), beta=beta.unsqueeze(0), initial_state=torch.zeros(1, 24, 128, 128, device=dev, dtype=torch.float32), output_final_state=True, cu_seqlens=cu, chunk_indices=ci, chunk_offsets=co, use_qk_l2norm_in_kernel=False)
S('o', o); S('fs', fs)
out = torch.empty(R, 3072, device=dev, dtype=bf)
fused_gdn_gated_rms_norm(x=o.view(R, 3072), z=qout[:, 5120:8192], weight=torch.ones(128, device=dev, dtype=bf), out=out, eps=1e-6, num_v_heads=24, head_v_dim=128, norm_before_gate=True, activation='silu')
S('out', out)
