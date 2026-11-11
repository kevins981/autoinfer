import os, sys, hashlib
os.environ["TORCH_CUDA_ARCH_LIST"]="8.9"
from torch.utils.cpp_extension import load
def get(src="g17.cu", abl=0, nst=4):
    p=os.path.join(os.path.dirname(os.path.abspath(__file__)), src)
    h=hashlib.md5(open(p,'rb').read()).hexdigest()[:8]
    return load(name="g17_"+h+"_a%d_n%d"%(abl,nst), sources=[p], extra_cuda_cflags=["-O3","-lineinfo","--resource-usage","-DABL=%d"%abl,"-DNST=%d"%nst], verbose=True)
if __name__=="__main__": get()
