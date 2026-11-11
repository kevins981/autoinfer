import os, sys, hashlib
os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
from torch.utils.cpp_extension import load
H=os.path.dirname(os.path.abspath(__file__))
def get():
    t=hashlib.sha1(open(H+'/hyb2.cu','rb').read()).hexdigest()[:10]
    d=f'{H}/ext_hyb_{t}'; os.makedirs(d,exist_ok=True)
    return load('hyb2_'+t,[H+'/hyb2.cu'],build_directory=d,verbose=False,extra_ldflags=['-lcuda'])
if __name__=='__main__':
    get(); print('ok')
