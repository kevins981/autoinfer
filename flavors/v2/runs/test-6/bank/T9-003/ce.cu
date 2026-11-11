// T9-001 ext: multi-transfer copy-engine send/recv (doorbells via stream mem-ops), raw copies, small kernels, core spin.
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
#define NT 64
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
// send piece: wait slot free, D2H x[off:off+n] -> my slot t, doorbell
void send(torch::Tensor x, int64_t t, int64_t off, int64_t n){
  cudaStream_t s=at::cuda::getCurrentCUDAStream(); CUstream cs=(CUstream)s; int me=C.rank;
  CK(cuStreamWaitValue32(cs,fA(me,t),1,CU_STREAM_WAIT_VALUE_EQ));
  CK(cuStreamWriteValue32(cs,fA(me,t),0,CU_STREAM_WRITE_VALUE_DEFAULT));
  CKR(cudaMemcpyAsync(slotp(me,t),(char*)x.data_ptr()+off,n,cudaMemcpyDeviceToHost,s));
  CK(cuStreamWriteValue32(cs,fF(me,t),1,CU_STREAM_WRITE_VALUE_DEFAULT));
}
void recvwait(int64_t t){
  cudaStream_t s=at::cuda::getCurrentCUDAStream(); CUstream cs=(CUstream)s; int pr=1-C.rank;
  CK(cuStreamWaitValue32(cs,fF(pr,t),1,CU_STREAM_WAIT_VALUE_EQ));
  CK(cuStreamWriteValue32(cs,fF(pr,t),0,CU_STREAM_WRITE_VALUE_DEFAULT));
}
void recvcopy(torch::Tensor dst, int64_t t, int64_t off, int64_t n){
  cudaStream_t s=at::cuda::getCurrentCUDAStream(); CUstream cs=(CUstream)s; int pr=1-C.rank;
  CKR(cudaMemcpyAsync((char*)dst.data_ptr()+off,slotp(pr,t),n,cudaMemcpyHostToDevice,s));
  CK(cuStreamWriteValue32(cs,fA(pr,t),1,CU_STREAM_WRITE_VALUE_DEFAULT));
}
// raw copies (bandwidth test, no flags)
void d2h_raw(torch::Tensor x,int64_t t,int64_t n){ CKR(cudaMemcpyAsync(slotp(C.rank,t),x.data_ptr(),n,cudaMemcpyDeviceToHost,at::cuda::getCurrentCUDAStream())); }
void h2d_raw(torch::Tensor x,int64_t t,int64_t n){ CKR(cudaMemcpyAsync(x.data_ptr(),slotp(C.rank,t),n,cudaMemcpyHostToDevice,at::cuda::getCurrentCUDAStream())); }
std::string shm_range(){ return std::to_string((unsigned long)C.hbase)+" "+std::to_string(tot(C.slot)); }
__device__ __forceinline__ float bf(uint32_t w,int hi){ return __uint_as_float(hi? (w&0xffff0000u):(w<<16)); }
__device__ __forceinline__ uint32_t pk(float a,float b){ __nv_bfloat162 s=__floats2bfloat162_rn(a,b); return *reinterpret_cast<uint32_t*>(&s); }
__global__ void add2_k(const uint4* a,const uint4* b,uint4* o,long n){
  for(long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x){
    uint4 x=a[i],y=b[i],r; uint32_t*px=(uint32_t*)&x,*py=(uint32_t*)&y,*pr=(uint32_t*)&r;
    #pragma unroll
    for(int k=0;k<4;k++) pr[k]=pk(bf(px[k],0)+bf(py[k],0),bf(px[k],1)+bf(py[k],1));
    o[i]=r;
  }
}
void add2(torch::Tensor a,torch::Tensor b,torch::Tensor o){
  long n=o.numel()/8; cudaStream_t s=at::cuda::getCurrentCUDAStream(); int blocks=(int)std::min<long>((n+255)/256,256);
  add2_k<<<blocks,256,0,s>>>((const uint4*)a.data_ptr(),(const uint4*)b.data_ptr(),(uint4*)o.data_ptr(),n);
}
__global__ void stamp_k(long long* buf,int idx){ long long t; asm volatile("mov.u64 %0, %globaltimer;":"=l"(t)); buf[idx]=t; }
void stamp(torch::Tensor buf,int64_t idx){ stamp_k<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>((long long*)buf.data_ptr(),(int)idx); }
// "core": occupies every SM (1 block/SM via 64KB smem, 1024 threads) for ns of wall time
__global__ void core_k(long long ns){
  extern __shared__ char sm[]; if(threadIdx.x==0) sm[0]=1;
  long long t0; asm volatile("mov.u64 %0, %globaltimer;":"=l"(t0));
  while(true){ long long t; asm volatile("mov.u64 %0, %globaltimer;":"=l"(t)); if(t-t0>=ns) break; __nanosleep(200); }
}
void core(int64_t ns){
  static bool set=false; if(!set){ cudaFuncSetAttribute(core_k,cudaFuncAttributeMaxDynamicSharedMemorySize,64*1024); set=true; }
  int nsm; cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0);
  core_k<<<nsm,1024,64*1024,at::cuda::getCurrentCUDAStream()>>>(ns);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("init",&init);m.def("send",&send);m.def("recvwait",&recvwait);m.def("recvcopy",&recvcopy);m.def("d2h_raw",&d2h_raw);m.def("h2d_raw",&h2d_raw);
  m.def("shm_range",&shm_range);m.def("add2",&add2);m.def("stamp",&stamp);m.def("core",&core);
}
