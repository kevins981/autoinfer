// Probe: does Ada (sm_89) generic compute-data compression reduce DRAM read time for our data?
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cstring>
#define CK(x) do{CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s line %d: %s\n",#x,__LINE__,s);exit(1);}}while(0)
__global__ void rd(const uint4* __restrict__ p, size_t n, unsigned* out){
  unsigned acc=0; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){uint4 v=__ldcs(p+i); acc^=v.x^v.y^v.z^v.w;}
  if(acc==0x12345678) out[0]=acc;
}
CUdeviceptr alloc(size_t sz,bool comp,size_t*gran_out,CUmemGenericAllocationHandle*h){
  CUmemAllocationProp pr={}; pr.type=CU_MEM_ALLOCATION_TYPE_PINNED; pr.location.type=CU_MEM_LOCATION_TYPE_DEVICE; pr.location.id=0;
  if(comp) pr.allocFlags.compressionType=CU_MEM_ALLOCATION_COMP_GENERIC;
  size_t g; CK(cuMemGetAllocationGranularity(&g,&pr,CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); sz=(sz+g-1)/g*g;
  CK(cuMemCreate(h,sz,&pr,0));
  CUmemAllocationProp got={}; CK(cuMemGetAllocationPropertiesFromHandle(&got,*h));
  if(comp) printf("  compressible requested -> granted compressionType=%d\n",(int)got.allocFlags.compressionType);
  CUdeviceptr d; CK(cuMemAddressReserve(&d,sz,0,0,0)); CK(cuMemMap(d,sz,0,*h,0));
  CUmemAccessDesc a={}; a.location=pr.location; a.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE; CK(cuMemSetAccess(d,sz,&a,1));
  *gran_out=sz; return d;
}
int main(int argc,char**argv){
  CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev,0)); CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx,dev)); CK(cuCtxSetCurrent(ctx));
  int sup=0; CK(cuDeviceGetAttribute(&sup,CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,dev));
  int vmm=0; CK(cuDeviceGetAttribute(&vmm,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev));
  printf("GENERIC_COMPRESSION_SUPPORTED=%d VMM=%d\n",sup,vmm);
  const size_t S=1ull<<30; std::vector<unsigned char> host(S);
  unsigned* out; cudaMalloc(&out,4);
  const char* names[]={"zeros","const_0x3c","fp8_weights","fp8_hi_nibble_plane","fp8_lo_nibble_plane","random_bytes","fp32_normal","fp32_normal_smooth","bf16_normal"};
  std::mt19937_64 rng(1); std::normal_distribution<float> nd(0,1);
  for(int t=0;t<9;t++){
    if(t==0) memset(host.data(),0,S); else if(t==1) memset(host.data(),0x3c,S);
    else if(t>=2&&t<=4){const char*fn[]={"w_fp8.bin","w_hi.bin","w_lo.bin"}; FILE*f=fopen(fn[t-2],"rb"); fseek(f,0,SEEK_END); size_t n=ftell(f); fseek(f,0,SEEK_SET); std::vector<unsigned char> b(n); fread(b.data(),1,n,f); fclose(f); for(size_t i=0;i<S;i+=n) memcpy(host.data()+i,b.data(),std::min(n,S-i));}
    else if(t==5){uint64_t*q=(uint64_t*)host.data(); for(size_t i=0;i<S/8;i++) q[i]=rng();}
    else if(t==6){float*q=(float*)host.data(); for(size_t i=0;i<S/4;i++) q[i]=nd(rng)*0.01f;}
    else if(t==7){float*q=(float*)host.data(); float v=0; for(size_t i=0;i<S/4;i++){v=0.999f*v+0.01f*nd(rng); q[i]=v;}}
    else {uint16_t*q=(uint16_t*)host.data(); for(size_t i=0;i<S/2;i++){float x=nd(rng); uint32_t u; memcpy(&u,&x,4); q[i]=u>>16;}}
    for(int comp=0;comp<2;comp++){
      size_t sz; CUmemGenericAllocationHandle h; CUdeviceptr d=alloc(S,comp,&sz,&h);
      CK(cuMemcpyHtoD(d,host.data(),S)); cudaDeviceSynchronize();
      // flush L2 by reading another buffer
      cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
      float best=1e9; for(int r=0;r<8;r++){ cudaEventRecord(a); rd<<<128*8,512>>>((const uint4*)d,S/16,out); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); if(r>0&&ms<best)best=ms;}
      cudaError_t e=cudaGetLastError();
      printf("%-22s comp=%d  read 1 GiB best %.3f ms = %.1f GB/s %s\n",names[t],comp,best,S/best/1e6,e?cudaGetErrorString(e):"");
      CK(cuMemUnmap(d,sz)); CK(cuMemAddressFree(d,sz)); CK(cuMemRelease(h));
    }
  }
  return 0;
}
