import os, torch
from torch.utils.cpp_extension import load
D = os.path.dirname(os.path.abspath(__file__))
WS = '/scratch/kevinsong/learning/autoinfer/flavors/v2/runs/test-6/workspaces/t42-cs/vllm/vllm/model_executor/layers/lmh_codec.cu'
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
def _load(name, src):
    bd = os.path.join(D, 'build', name); os.makedirs(bd, exist_ok=True)
    return load(name, [src], build_directory=bd, verbose=False, extra_cuda_cflags=['-O3', '-lineinfo'])
cs = _load('t42_cs', os.path.join(D, 'cs.cu'))
champ = _load('t42_champ_lmh', WS)
