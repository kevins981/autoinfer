import torch
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.models.qwen3_5.nvidia.ops.fused_gdn_gated_norm import fused_gdn_gated_rms_norm
from vllm.config import VllmConfig, set_current_vllm_config
dev="cuda"; torch.manual_seed(0); H=24; D=128
with set_current_vllm_config(VllmConfig()):
    qf=QuantFP8(static=False,group_shape=GroupShape(1,128),num_token_padding=None,use_ue8m0=False)
    for scale_amp in [1.0, 30.0, 1e-3]:
      for M in [0,1,2,16,17,48,320,1024,2048]:
        x=(torch.randn(M,H*D,device=dev)*scale_amp).bfloat16()
        packed=(torch.randn(M,8192+H*D,device=dev)*2).bfloat16(); z=packed[:,8192:]   # strided view like the model
        w=(torch.rand(D,device=dev)+0.5).float()
        kw=dict(z=z,weight=w,eps=1e-6,num_v_heads=H,head_v_dim=D,norm_before_gate=True,activation="silu")
        ref=torch.empty(M,H*D,device=dev,dtype=torch.bfloat16)
        fused_gdn_gated_rms_norm(x=x,out=ref,**kw)
        if M==0:
            q=torch.empty(0,H*D,device=dev,dtype=torch.float8_e4m3fn); s=torch.empty(0,H,device=dev)
            fused_gdn_gated_rms_norm(x=x,out=q,dyn_scales=s,**kw); continue
        q1,s1=qf(ref)
        q=torch.empty(M,H*D,device=dev,dtype=torch.float8_e4m3fn); s=torch.empty(M,H,device=dev)
        fused_gdn_gated_rms_norm(x=x,out=q,dyn_scales=s,**kw)
        cd=(q1.view(torch.uint8)!=q.view(torch.uint8)).sum().item(); sd=(s1.view(M,H)!=s).sum().item()
        print(f"amp {scale_amp} M {M}: code mismatches {cd} scale mismatches {sd}",flush=True)
        if cd:
            idx=(q1.view(torch.uint8)!=q.view(torch.uint8)).nonzero()
            for r,c in idx[:4].tolist():
                g=c//128; yv=ref[r,g*128:(g+1)*128].float(); sc=s1.view(M,H)[r,g]
                print('  mismatch',r,c,'y',ref[r,c].item(),'scale',sc.item(),'ref code',q1[r,c].float().item(),'fused',q[r,c].float().item(),'y/scale',(ref[r,c].float()/sc).item(),'y*(1/sc)',(ref[r,c].float()*(1.0/sc)).item(),'amax',yv.abs().max().item())
print("OK")
