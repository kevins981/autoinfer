// W1-1: swap-AB register-direct-weights microbench. K0 (no-loads MMA/dequant loop) and K1 (weight stream).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdint.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA err %s line %d\n",cudaGetErrorString(e),__LINE__); exit(1);}}while(0)
__device__ __forceinline__ void mma(float* c, uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3, uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t& r0,uint32_t& r1,uint32_t& r2,uint32_t& r3, uint32_t addr){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(addr));
}
// Marlin/g2w style dequant of 4 fp8 -> 2 x bf16x2 (exponent bias folded into scale)
__device__ __forceinline__ void deq(uint32_t q, uint32_t s, uint32_t& r0, uint32_t& r1){
  uint32_t o1=(q&0x80008000u)|((q&0x7F007F00u)>>4);
  q<<=8;
  uint32_t o2=(q&0x80008000u)|((q&0x7F007F00u)>>4);
  __nv_bfloat162 s2=*(__nv_bfloat162*)&s;
  __nv_bfloat162 b1=__hmul2(*(__nv_bfloat162*)&o1,s2), b0=__hmul2(*(__nv_bfloat162*)&o2,s2);
  r0=*(uint32_t*)&b0; r1=*(uint32_t*)&b1;
}
constexpr int SROW=520; // bf16 elems per token row in smem chunk (512 + pad)
template<int W,int T,int DEQ>
__global__ void __launch_bounds__(W*32,1) k0(int KS, float* out, uint32_t seed){
  extern __shared__ __align__(16) unsigned char sm[];
  __nv_bfloat16* tok=(__nv_bfloat16*)sm;
  int tid=threadIdx.x, l=tid&31;
  for(int i=tid;i<48*SROW;i+=W*32) tok[i]=__float2bfloat16(((i*2654435761u)>>20)*1e-3f-2.f);
  float acc[T][6][4];
  #pragma unroll
  for(int t=0;t<T;t++)
    #pragma unroll
    for(int n=0;n<6;n++)
      #pragma unroll
      for(int j=0;j<4;j++) acc[t][n][j]=0.f;
  uint4 w[T];
  #pragma unroll
  for(int t=0;t<T;t++) w[t]=make_uint4(seed*(t+1)^tid, seed*3+tid*7+t, seed+tid*11, seed^(tid*13+t));
  uint32_t sc[T];
  #pragma unroll
  for(int t=0;t<T;t++) sc[t]=0x7B807B80u; // 2^120 bf16x2
  uint32_t base=(uint32_t)__cvta_generic_to_shared(tok)+(((l&7)+((l>>4)<<3))*SROW+((l>>3)&1)*8)*2;
  __syncthreads();
  for(int it=0; it<KS; it+=2){
    uint32_t mix=it*0x01010101u+seed;
    if((it&15)==0){ __syncthreads(); // one barrier per 256-K chunk (16 ksteps)
      #pragma unroll
      for(int t=0;t<T;t++) sc[t]=0x7B807B80u+((it>>4)&1)*0u + (((const uint32_t*)sm)[(it+t)&63]&0x00010001u)*0; }
    #pragma unroll
    for(int sub=0;sub<2;sub++){
      uint32_t b[6][2];
      uint32_t ad=base+(((it&15)+sub)*16)*2;
      #pragma unroll
      for(int p=0;p<3;p++) ldsm4(b[2*p][0],b[2*p][1],b[2*p+1][0],b[2*p+1][1], ad+p*16*SROW*2);
      #pragma unroll
      for(int t=0;t<T;t++){
        uint32_t a0,a1,a2,a3;
        uint32_t x=sub?w[t].z:w[t].x, y=sub?w[t].w:w[t].y;
        if(DEQ){ deq(x+mix,sc[t],a0,a1); deq(y+mix,sc[t],a2,a3); }
        else { a0=x^mix; a1=y^mix; a2=x+mix; a3=y+mix; }
        #pragma unroll
        for(int n=0;n<6;n++) mma(acc[t][n],a0,a1,a2,a3,b[n][0],b[n][1]);
      }
    }
  }
  float s=0;
  #pragma unroll
  for(int t=0;t<T;t++)
    #pragma unroll
    for(int n=0;n<6;n++)
      #pragma unroll
      for(int j=0;j<4;j++) s+=acc[t][n][j];
  out[blockIdx.x*W*32+tid]=s;
}
template<int W,int T,int DEQ> double run_k0(int dev_sms, float* dout, const char* nm){
  const double UNITS=348160.0; // (m16 tile, k16) fragment units for gate_up
  int KS=(int)(UNITS/(128.0*W*T)+0.5); KS=(KS+15)/16*16; // multiple of 16 (chunk)
  double norm=UNITS/(128.0*W*T*KS);
  size_t smem=48*SROW*2;
  auto f=k0<W,T,DEQ>;
  CK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
  cudaFuncAttributes at; cudaFuncGetAttributes(&at,f);
  for(int i=0;i<5;i++) f<<<128,W*32,smem>>>(KS,dout,1);
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  std::vector<float> ts;
  for(int r=0;r<15;r++){
    cudaEventRecord(e0); for(int i=0;i<10;i++) f<<<128,W*32,smem>>>(KS,dout,r+1); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms,e0,e1); ts.push_back(ms*100.f);
  }
  std::sort(ts.begin(),ts.end()); double med=ts[7];
  printf("K0 W=%2d T=%d deq=%d regs=%3d spill=%zu KS=%d raw=%.1f us normalized(gate_up work)=%.1f us  =%.1f%% of 58.7\n",W,T,DEQ,at.numRegs,at.localSizeBytes,KS,med,med*norm,58.7/(med*norm)*100);
  return med*norm;
}
// ---------------- K1 weight stream
__device__ __forceinline__ uint64_t gt(){ uint64_t t; asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t)); return t; }
template<int L>
__global__ void k1(const uint4* __restrict__ wp, size_t nchunks_per_iter_total, int iters, uint32_t* sink, uint64_t* tm){
  int gw=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5), nw=gridDim.x*(blockDim.x>>5), l=threadIdx.x&31;
  if(threadIdx.x==0) tm[blockIdx.x*2]=gt();
  uint32_t x=0;
  for(int i=0;i<iters;i++){
    uint4 v[L];
    #pragma unroll
    for(int j=0;j<L;j++){
      const uint4* p=wp+(((size_t)(i*nw+gw)*L+j)*32+l);
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v[j].x),"=r"(v[j].y),"=r"(v[j].z),"=r"(v[j].w):"l"(p));
    }
    #pragma unroll
    for(int j=0;j<L;j++) x^=v[j].x^v[j].y^v[j].z^v[j].w;
  }
  sink[blockIdx.x*blockDim.x+threadIdx.x]=x;
  __syncthreads();
  if(threadIdx.x==0) tm[blockIdx.x*2+1]=gt();
}
__global__ void flushk(const uint4* p, size_t n, uint32_t* sink){
  uint32_t x=0; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){uint4 v=p[i]; x^=v.x^v.y^v.z^v.w;}
  if(x==0x12345678u) sink[0]=x;
}
__global__ void pfk(const unsigned char* p, size_t nbytes){
  for(size_t i=(blockIdx.x*(size_t)blockDim.x+threadIdx.x)*128;i<nbytes;i+=(size_t)gridDim.x*blockDim.x*128)
    asm volatile("prefetch.global.L2::evict_last [%0];"::"l"(p+i));
}
template<int L> void run_k1(const unsigned char* buf, size_t bytes, int W, bool warm, uint8_t* flush, size_t fbytes, uint32_t* sink, uint64_t* tm, const char* nm){
  int nw=128*W; size_t per=(size_t)L*512; // bytes per warp-iteration
  int iters=(int)(bytes/(per*nw)); if(iters<1) iters=1;
  size_t used=(size_t)iters*per*nw;
  std::vector<double> ts;
  for(int r=0;r<19;r++){
    flushk<<<1024,256>>>((const uint4*)flush,fbytes/16,sink);
    if(warm){ pfk<<<512,256>>>(buf,std::min(used,(size_t)48<<20)); }
    k1<L><<<128,W*32>>>((const uint4*)buf,0,iters,sink,tm);
    CK(cudaDeviceSynchronize());
    uint64_t h[256]; cudaMemcpy(h,tm,sizeof(h),cudaMemcpyDeviceToHost);
    uint64_t s=~0ull,e=0; for(int b=0;b<128;b++){s=std::min(s,h[2*b]);e=std::max(e,h[2*b+1]);}
    if(r>=4) ts.push_back((e-s)/1000.0);
  }
  std::sort(ts.begin(),ts.end()); double med=ts[ts.size()/2];
  printf("K1 %-8s %s W=%2d L=%d bytes=%.1fMB (used %.1f) %.1f us  %.2f TB/s\n",nm,warm?"warm":"cold",W,L,bytes/1e6,used/1e6,med,used/med/1e6);
}
int main(){
  CK(cudaSetDevice(0)); cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  int clk; cudaDeviceGetAttribute(&clk,cudaDevAttrClockRate,0);
  printf("device %s SMs=%d\n",p.name,p.multiProcessorCount);
  float* dout; CK(cudaMalloc(&dout,128*512*4));
  printf("--- K0\n");
  #define R(W,T) run_k0<W,T,1>(128,dout,""); run_k0<W,T,0>(128,dout,"");
  R(8,1) R(8,2) R(8,4) R(12,1) R(12,2) R(12,4) R(16,1) R(16,2) R(16,4)
  printf("--- K1\n");
  size_t fb=256<<20; uint8_t* flush; CK(cudaMalloc(&flush,fb)); CK(cudaMemset(flush,1,fb));
  unsigned char* buf; size_t mx=89ull<<20; mx=(size_t)89.0e6; CK(cudaMalloc(&buf,mx+(1<<20))); CK(cudaMemset(buf,3,mx));
  uint32_t* sink; CK(cudaMalloc(&sink,128*1024*4)); uint64_t* tm; CK(cudaMalloc(&tm,4096));
  struct S{const char*n; size_t b;} sz[]={{"gate_up",89128960},{"qkvz",41943040},{"out_proj",15728640}};
  for(auto&s:sz) for(int warm=0;warm<2;warm++) for(int W:{8,16}){
    run_k1<2>(buf,s.b,W,warm,flush,fb,sink,tm,s.n);
    run_k1<4>(buf,s.b,W,warm,flush,fb,sink,tm,s.n);
    run_k1<8>(buf,s.b,W,warm,flush,fb,sink,tm,s.n);
  }
  return 0;
}
