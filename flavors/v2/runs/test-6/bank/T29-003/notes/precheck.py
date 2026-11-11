# T29_W8A8=2 pre-check (GPU, ~20 s): the layer path at M<=96 is bitwise the champion's Marlin; at M>96
# (incl. odd M) W8A8 matches an fp32 reference of (group-quantized x) @ (dequantized W).
import os, torch
assert os.environ.get('T29_W8A8') == '2'
from vllm.model_executor.layers.quantization.utils import t29_marlin_w8a8 as T
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
dev = torch.device('cuda:0'); torch.cuda.set_device(dev); torch.manual_seed(0)
bad = 0
for N, K in ((17408, 5120), (5120, 8704), (5120, 3072)):
    L = torch.nn.Module(); L.output_size_per_partition = N; L.input_size_per_partition = K
    L.weight_block_size = [128, 128]; L.orig_dtype = torch.bfloat16
    w = torch.randint(0, 255, (N, K), device=dev, dtype=torch.uint8).clamp(max=0x77).view(torch.float8_e4m3fn)
    sc = torch.rand(N // 128, K // 128, device=dev) * 0.02 + 0.005
    Wdq = (w.float() * sc.repeat_interleave(128, 0).repeat_interleave(128, 1)).t().contiguous()
    L.weight = torch.nn.Parameter(w.clone(), requires_grad=False); L.weight_scale_inv = torch.nn.Parameter(sc.clone(), requires_grad=False)
    T.prepare_layer(L, L.weight_scale_inv)
    prepare_fp8_layer_for_marlin(L, size_k_first=False)
    T.warmup_layer(L)
    for M in (48, 96, 97, 131, 200, 1537):
        x = torch.randn(M, K, device=dev); x[:, torch.randperm(K, device=dev)[:K // 100]] *= 20; x = x.bfloat16()
        champ = apply_fp8_marlin_linear(x, L.weight, L.weight_scale_inv, None, N, K, None)
        gate = apply_fp8_marlin_linear(x, L.weight, L.weight_scale_inv, None, N, K, None, t29_ws_f32=L.t29_ws_f32)
        q, s = per_token_group_quant_fp8(x, 128, use_ue8m0=False)
        ref = (q.float() * s.repeat_interleave(128, 1)) @ Wdq
        relF = ((gate.float() - ref).norm() / ref.norm()).item()
        if M <= 96:
            ok = torch.equal(champ, gate); bad += not ok
            print(f'{N}x{K} M={M}: gate path == champion Marlin bitwise: {ok}', flush=True)
        else:
            ok = relF < 2e-3; bad += not ok
            print(f'{N}x{K} M={M}: W8A8 relF vs fp32 ref of quantized x {relF:.2e} (bf16 out ~1e-3 expected) '
                  f'maxabs {(gate.float() - ref).abs().max().item():.3g}  [Marlin vs same ref relF {((champ.float()-ref).norm()/ref.norm()).item():.2e}]', flush=True)
print('PRECHECK', 'PASS' if bad == 0 else f'FAIL ({bad})', flush=True)
