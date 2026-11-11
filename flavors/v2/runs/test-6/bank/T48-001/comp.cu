// Does the 4090 offer hardware generic (lossless, transparent) memory compression, and does it raise
// effective DRAM read bandwidth on real LLM data? Plain vs COMP_GENERIC allocation, same bytes.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#define CK(x) do{CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s: %s\n",#x,s);exit(1);}}while(0)

__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* out){
  unsigned acc=0;
  for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){
    uint4 v; asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p+i));
    acc^=v.x^v.y^v.z^v.w; }
  if(acc==0x12345678) out[0]=acc;
}
__global__ void flush(uint4* p,size_t n){ for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){ uint4 v=p[i]; if(v.x==0xdeadbeef) p[i].y=1; } }

static CUdeviceptr alloc(size_t sz,bool comp,bool* granted){
  CUmemAllocationProp pr={}; pr.type=CU_MEM_ALLOCATION_TYPE_PINNED; pr.location.type=CU_MEM_LOCATION_TYPE_DEVICE; pr.location.id=0;
  if(comp) pr.allocFlags.compressionType=CU_MEM_ALLOCATION_COMP_GENERIC;
  size_t g; CK(cuMemGetAllocationGranularity(&g,&pr,CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); sz=(sz+g-1)/g*g;
  CUmemGenericAllocationHandle h; CK(cuMemCreate(&h,sz,&pr,0));
  CUmemAllocationProp got={}; CK(cuMemGetAllocationPropertiesFromHandle(&got,h)); *granted=got.allocFlags.compressionType==CU_MEM_ALLOCATION_COMP_GENERIC;
  CUdeviceptr d; CK(cuMemAddressReserve(&d,sz,0,0,0)); CK(cuMemMap(d,sz,0,h,0));
  CUmemAccessDesc a={}; a.location=pr.location; a.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE; CK(cuMemSetAccess(d,sz,&a,1));
  return d;
}
int main(int argc,char**argv){
  CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev,0)); CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx,dev)); CK(cuCtxSetCurrent(ctx));
  int sup=0; CK(cuDeviceGetAttribute(&sup,CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,dev));
  printf("GENERIC_COMPRESSION_SUPPORTED=%d\n",sup);
  const size_t SZ=512ull<<20; size_t n=SZ/16;
  std::vector<unsigned char> h(SZ);
  std::vector<std::string> kinds={"zeros","random","fp8w","bf16lm","fp8w_absmax_rows"};
  unsigned* out; cudaMalloc(&out,4); uint4* fl; cudaMalloc(&fl,256<<20);
  for(int comp=0;comp<=1;comp++){
    if(comp && !sup){printf("skip comp\n");break;}
    bool g; CUdeviceptr d=alloc(SZ,comp,&g); printf("alloc comp=%d granted=%d\n",comp,(int)g);
    for(auto&k:kinds){
      if(k=="zeros") memset(h.data(),0,SZ);
      else if(k=="random"){ srand(1); for(size_t i=0;i<SZ;i++) h[i]=rand()&255; }
      else { std::string f=std::string(argv[1])+"/"+(k=="bf16lm"?"bf16lm.bin":"fp8w.bin"); FILE*fp=fopen(f.c_str(),"rb"); size_t r=fread(h.data(),1,SZ,fp); fclose(fp); for(size_t i=r;i<SZ;i++) h[i]=h[i-r];
        if(k=="fp8w_absmax_rows"){ /* sign bit cleared: a slightly more compressible variant */ for(size_t i=0;i<SZ;i++) h[i]&=0x7f; } }
      cudaMemcpy((void*)d,h.data(),SZ,cudaMemcpyHostToDevice); cudaDeviceSynchronize();
      cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float best=1e9;
      for(int it=0;it<8;it++){ flush<<<512,256>>>(fl,(256<<20)/16); cudaEventRecord(a); rd<<<128*8,256>>>((uint4*)d,n,out); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); if(it>0&&ms<best)best=ms; }
      printf("comp=%d %-18s %.3f ms  %.1f GB/s\n",comp,k.c_str(),best,SZ/best/1e6);
    }
  }
  printf("err=%s\n",cudaGetErrorString(cudaGetLastError()));
}
