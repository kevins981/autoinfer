import os, hashlib
os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
from torch.utils.cpp_extension import load
H=os.path.dirname(os.path.abspath(__file__))
def get():
    t=hashlib.sha1(open(H+'/tag.cu','rb').read()+open(H+'/codec.cuh','rb').read()).hexdigest()[:10]
    d=f'{H}/ext_tag_{t}'; os.makedirs(d,exist_ok=True)
    return load('tag_'+t,[H+'/tag.cu'],build_directory=d,verbose=False,extra_include_paths=[H])
if __name__=='__main__':
    get(); print('ok')
