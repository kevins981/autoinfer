// T7-001 microbench ext: copy-engine send/recv with stream-memop doorbells + small fused kernels.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#define FS 16
#define NT 64   // transfers
struct Ctx { char* hbase; char* dbase; size_t slot; int rank; };
static Ctx C;
static size_t tot(size_t slot){ return 2*(size_t)NT*slot + 4*NT*FS*4; }
static char* slotp(int r,int t){ return C.hbase + ((size_t)(r*NT+t))*C.slot; }
static size_t offF(int r,int t){ return 2*NT*C.slot + ((size_t)(r*NT+t))*FS*4; }
static size_t offA(int r,int t){ return 2*NT*C.slot + ((size_t)(2*NT + r*NT+t))*FS*4; }
static CUdeviceptr fF(int r,int t){ return (CUdeviceptr)(C.dbase+offF(r,t)); }
static CUdeviceptr fA(int r,int t){ return (CUdeviceptr)(C.dbase+offA(r,t)); }
#define CK(x) TORCH_CHECK((x)==CUDA_SUCCESS, #x)
#define CKR(x) TORCH_CHECK((x)==cudaSuccess, #x)
void init(std::string path, int64_t slot, int64_t rank, bool create){
  size_t t=tot(slot);
  int fd=open(path.c_str(), O_RDWR|(create?O_CREAT:0),0600); TORCH_CHECK(fd>=0,"open");
  if(create) TORCH_CHECK(ftruncate(fd,t)==0,"ftruncate");
  void* m=mmap(nullptr,t,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); TORCH_CHECK(m!=MAP_FAILED,"mmap"); close(fd);
  CKR(cudaHostRegister(m,t,cudaHostRegisterMapped|cudaHostRegisterPortable));
  void* d; CKR(cudaHostGetDevicePointer(&d,m,0));
  C.hbase=(char*)m; C.dbase=(char*)d; C.slot=slot; C.rank=rank;
  if(create) for(int r=0;r<2;r++) for(int k=0;k<NT;k++){ *(volatile int*)(C.hbase+offA(r,k))=1; *(volatile int*)(C.hbase+offF(r,k))=0; }
}
// A[r][t]: set to 1 by reader (r's peer) after consuming r's slot; F[r][t]: set by r when data is in slot.
void send(torch::Tensor x, int64_t t, int64_t nbytes){
  cudaStream_t s=at::cuda::getCurrentCUDAStream(); CUstream cs=(CUstream)s; int me=C.rank;
  CK(cuStreamWaitValue32(cs,fA(me,t),1,CU_STREAM_WAIT_VALUE_EQ));
  CK(cuStreamWriteValue32(cs,fA(me,t),0,CU_STREAM_WRITE_VALUE_DEFAULT));
  CKR(cudaMemcpyAsync(slotp(me,t),x.data_ptr(),nbytes,cudaMemcpyDeviceToHost,s));
  CK(cuStreamWriteValue32(cs,fF(me,t),1,CU_STREAM_WRITE_VALUE_DEFAULT));
}
void recv(torch::Tensor dst, int64_t t, int64_t nbytes){
  cudaStream_t s=at::cuda::getCurrentCUDAStream(); CUstream cs=(CUstream)s; int pr=1-C.rank;
  CK(cuStreamWaitValue32(cs,fF(pr,t),1,CU_STREAM_WAIT_VALUE_EQ));
  CK(cuStreamWriteValue32(cs,fF(pr,t),0,CU_STREAM_WRITE_VALUE_DEFAULT));
  CKR(cudaMemcpyAsync(dst.data_ptr(),slotp(pr,t),nbytes,cudaMemcpyHostToDevice,s));
  CK(cuStreamWriteValue32(cs,fA(pr,t),1,CU_STREAM_WRITE_VALUE_DEFAULT));
}
__device__ __forceinline__ float bf(uint32_t w,int hi){ return __uint_as_float(hi? (w&0xffff0000u):(w<<16)); }
__device__ __forceinline__ uint32_t pk(float a,float b){ __nv_bfloat162 s=__floats2bfloat162_rn(a,b); return *reinterpret_cast<uint32_t*>(&s); }
// s=bf16(po+tmp); r=bf16(resid+s) (written back); y=bf16(r*g)
__global__ void rsn_k(const uint4* po,const uint4* tmp,uint4* resid,const uint4* g,uint4* y,long n,int vpr){
  for(long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x){
    uint4 a=po[i],b=tmp[i],r=resid[i],gg=g[i%vpr],o;
    uint32_t*pa=(uint32_t*)&a,*pb=(uint32_t*)&b,*pr=(uint32_t*)&r,*pg=(uint32_t*)&gg,*po_=(uint32_t*)&o;
    #pragma unroll
    for(int k=0;k<4;k++){
      float s0=__bfloat162float(__float2bfloat16(bf(pa[k],0)+bf(pb[k],0))), s1=__bfloat162float(__float2bfloat16(bf(pa[k],1)+bf(pb[k],1)));
      float r0=__bfloat162float(__float2bfloat16(bf(pr[k],0)+s0)), r1=__bfloat162float(__float2bfloat16(bf(pr[k],1)+s1));
      pr[k]=pk(r0,r1); po_[k]=pk(r0*bf(pg[k],0), r1*bf(pg[k],1));
    }
    resid[i]=r; y[i]=o;
  }
}
void rsn(torch::Tensor po,torch::Tensor tmp,torch::Tensor resid,torch::Tensor g,torch::Tensor y){
  long n=po.numel()/8; int vpr=po.size(1)/8; cudaStream_t s=at::cuda::getCurrentCUDAStream();
  int blocks=(int)std::min<long>((n+255)/256,512);
  rsn_k<<<blocks,256,0,s>>>((const uint4*)po.data_ptr(),(const uint4*)tmp.data_ptr(),(uint4*)resid.data_ptr(),(const uint4*)g.data_ptr(),(uint4*)y.data_ptr(),n,vpr);
}
__global__ void add2_k(const uint4* a,const uint4* b,uint4* o,long n){
  for(long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x){
    uint4 x=a[i],y=b[i],r; uint32_t*px=(uint32_t*)&x,*py=(uint32_t*)&y,*pr=(uint32_t*)&r;
    #pragma unroll
    for(int k=0;k<4;k++) pr[k]=pk(bf(px[k],0)+bf(py[k],0),bf(px[k],1)+bf(py[k],1));
    o[i]=r;
  }
}
// a is fp32 (carry): 8 floats per vec of 8 outputs
__global__ void add2f_k(const float4* a,const uint4* b,uint4* o,long n){
  for(long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x){
    float4 x0=a[2*i],x1=a[2*i+1]; uint4 y=b[i],r; uint32_t*py=(uint32_t*)&y,*pr=(uint32_t*)&r;
    pr[0]=pk(x0.x+bf(py[0],0),x0.y+bf(py[0],1)); pr[1]=pk(x0.z+bf(py[1],0),x0.w+bf(py[1],1));
    pr[2]=pk(x1.x+bf(py[2],0),x1.y+bf(py[2],1)); pr[3]=pk(x1.z+bf(py[3],0),x1.w+bf(py[3],1));
    o[i]=r;
  }
}
__global__ void tof32_k(const uint4* a,float4* o,long n){
  for(long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x){
    uint4 x=a[i]; uint32_t*p=(uint32_t*)&x;
    o[2*i]=make_float4(bf(p[0],0),bf(p[0],1),bf(p[1],0),bf(p[1],1)); o[2*i+1]=make_float4(bf(p[2],0),bf(p[2],1),bf(p[3],0),bf(p[3],1));
  }
}
void add2(torch::Tensor a,torch::Tensor b,torch::Tensor o){
  long n=o.numel()/8; cudaStream_t s=at::cuda::getCurrentCUDAStream(); int blocks=(int)std::min<long>((n+255)/256,512);
  if(a.scalar_type()==torch::kFloat) add2f_k<<<blocks,256,0,s>>>((const float4*)a.data_ptr(),(const uint4*)b.data_ptr(),(uint4*)o.data_ptr(),n);
  else add2_k<<<blocks,256,0,s>>>((const uint4*)a.data_ptr(),(const uint4*)b.data_ptr(),(uint4*)o.data_ptr(),n);
}
void tof32(torch::Tensor a,torch::Tensor o){
  long n=a.numel()/8; cudaStream_t s=at::cuda::getCurrentCUDAStream(); int blocks=(int)std::min<long>((n+255)/256,512);
  tof32_k<<<blocks,256,0,s>>>((const uint4*)a.data_ptr(),(float4*)o.data_ptr(),n);
}
// act[r][j]=bf16(silu(a0+b0)*(a1+b1)), a,b [R,2D] ; D=act.size(1)
__global__ void asm_k(const uint4* a,const uint4* b,uint4* o,int R,int vd){ // vd = D/8
  long n=(long)R*vd;
  for(long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x){
    long r=i/vd,c=i%vd; long b0=r*2*vd+c,b1=b0+vd;
    uint4 g0=a[b0],g1=a[b1],h0=b[b0],h1=b[b1],res; uint32_t*pg=(uint32_t*)&g0,*pu=(uint32_t*)&g1,*qg=(uint32_t*)&h0,*qu=(uint32_t*)&h1,*pr=(uint32_t*)&res;
    #pragma unroll
    for(int k=0;k<4;k++){
      float o2[2];
      #pragma unroll
      for(int h=0;h<2;h++){ float g=bf(pg[k],h)+bf(qg[k],h), u=bf(pu[k],h)+bf(qu[k],h); o2[h]=g/(1.f+__expf(-g))*u; }
      pr[k]=pk(o2[0],o2[1]);
    }
    o[i]=res;
  }
}
void add_silu_mul(torch::Tensor a,torch::Tensor b,torch::Tensor act){
  int R=act.size(0),vd=act.size(1)/8; long n=(long)R*vd; cudaStream_t s=at::cuda::getCurrentCUDAStream(); int blocks=(int)std::min<long>((n+255)/256,512);
  asm_k<<<blocks,256,0,s>>>((const uint4*)a.data_ptr(),(const uint4*)b.data_ptr(),(uint4*)act.data_ptr(),R,vd);
}
__global__ void stamp_k(long long* buf,int idx){ long long t; asm volatile("mov.u64 %0, %globaltimer;":"=l"(t)); buf[idx]=t; }
void stamp(torch::Tensor buf,int64_t idx){ stamp_k<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>((long long*)buf.data_ptr(),(int)idx); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("init",&init);m.def("send",&send);m.def("recv",&recv);m.def("rsn",&rsn);m.def("add2",&add2);m.def("tof32",&tof32);m.def("add_silu_mul",&add_silu_mul);m.def("stamp",&stamp);
}
