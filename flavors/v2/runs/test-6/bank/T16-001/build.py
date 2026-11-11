import os, sys, hashlib
os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
from torch.utils.cpp_extension import load
H=os.path.dirname(os.path.abspath(__file__))
def get():
    t=hashlib.sha1(open(H+'/hyb.cu','rb').read()).hexdigest()[:10]
    d=f'{H}/ext_hyb_{t}'; os.makedirs(d,exist_ok=True)
    return load('hyb_'+t,[H+'/hyb.cu'],build_directory=d,verbose=False)
def getpf():
    t=hashlib.sha1(open(H+'/pfk.cu','rb').read()).hexdigest()[:10]
    d=f'{H}/ext_pfk_{t}'; os.makedirs(d,exist_ok=True)
    return load('pfk_'+t,[H+'/pfk.cu'],build_directory=d,verbose=False)
if __name__=='__main__':
    get(); getpf(); print('ok')
