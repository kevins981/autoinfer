// leader-10: tensor-pipe rate + accumulation precision, bf16 m16n8k16 vs e4m3 m16n8k32 (f32 acc) vs e4m3 f16 acc.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("err %s line %d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));}
__device__ __forceinline__ void mma_e4m3(float* c, const uint32_t* a, const uint32_t* b){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));}
__device__ __forceinline__ void mma_e4m3_h(uint32_t* c, const uint32_t* a, const uint32_t* b){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
    :"+r"(c[0]),"+r"(c[1]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));}
// rate kernels: NM m-tiles x NN n-tiles of independent accumulators
template<int KIND,int NM,int NN>
__global__ void rate(int iters, float* out){
  uint32_t a[NM][4], b[NN][2];
  for(int m=0;m<NM;m++) for(int i=0;i<4;i++) a[m][i]=0x3c003c00u^(threadIdx.x*(m+1)*(i+3));
  for(int n=0;n<NN;n++) for(int i=0;i<2;i++) b[n][i]=0x38383838u^(threadIdx.x*(n+7)*(i+1));
  float c[NM][NN][4]={}; uint32_t h[NM][NN][2]={};
  for(int it=0;it<iters;it++){
#pragma unroll
    for(int m=0;m<NM;m++)
#pragma unroll
      for(int n=0;n<NN;n++){
        if(KIND==0) mma_bf16(c[m][n],a[m],b[n]);
        else if(KIND==1) mma_e4m3(c[m][n],a[m],b[n]);
        else mma_e4m3_h(h[m][n],a[m],b[n]);
      }
  }
  float s=0; for(int m=0;m<NM;m++) for(int n=0;n<NN;n++){ for(int i=0;i<4;i++) s+=c[m][n][i]; s+=h[m][n][0]+h[m][n][1]; }
  if(s==123.456f) out[0]=s;
}
// precision: one warp, D = A(16x32) * B(32x8) chained over KSTEPS, e4m3 vs bf16 on identical values
__global__ void prec(const uint8_t* A8, const uint8_t* B8, const __nv_bfloat16* Abf, const __nv_bfloat16* Bbf, int ksteps, float* D8, float* Dbf){
  int lane=threadIdx.x; int g=lane>>2, t=lane&3;
  float c8[4]={0,0,0,0}, cb[4]={0,0,0,0};
  int K=32*ksteps;
  for(int ks=0;ks<ksteps;ks++){
    int k0=ks*32;
    uint32_t a[4], b[2];
    // A fragment e4m3 m16n8k32: a0: row g, k t*4..+3 ; a1: row g+8 ; a2: row g, k 16+t*4 ; a3: row g+8, k+16
    auto pa=[&](int r,int k){ uint32_t v=0; for(int j=0;j<4;j++) v|=(uint32_t)A8[r*K+k0+k+j]<<(8*j); return v; };
    a[0]=pa(g,t*4); a[1]=pa(g+8,t*4); a[2]=pa(g,16+t*4); a[3]=pa(g+8,16+t*4);
    auto pb=[&](int k,int n){ uint32_t v=0; for(int j=0;j<4;j++) v|=(uint32_t)B8[n*K+k0+k+j]<<(8*j); return v; };
    b[0]=pb(t*4,g); b[1]=pb(16+t*4,g);
    mma_e4m3(c8,a,b);
    // bf16: two k16 halves
    for(int hh=0;hh<2;hh++){
      int kb=k0+hh*16; uint32_t ab[4], bb[2];
      auto qa=[&](int r,int k){ __nv_bfloat16 x0=Abf[r*K+kb+k], x1=Abf[r*K+kb+k+1]; return (uint32_t)__bfloat16_as_ushort(x0)|((uint32_t)__bfloat16_as_ushort(x1)<<16); };
      ab[0]=qa(g,t*2); ab[1]=qa(g+8,t*2); ab[2]=qa(g,8+t*2); ab[3]=qa(g+8,8+t*2);
      auto qb=[&](int k,int n){ __nv_bfloat16 x0=Bbf[n*K+kb+k], x1=Bbf[n*K+kb+k+1]; return (uint32_t)__bfloat16_as_ushort(x0)|((uint32_t)__bfloat16_as_ushort(x1)<<16); };
      bb[0]=qb(t*2,g); bb[1]=qb(8+t*2,g);
      mma_bf16(cb,ab,bb);
    }
  }
  // C layout: c0,c1 row g cols t*2,t*2+1 ; c2,c3 row g+8
  D8[g*8+t*2]=c8[0]; D8[g*8+t*2+1]=c8[1]; D8[(g+8)*8+t*2]=c8[2]; D8[(g+8)*8+t*2+1]=c8[3];
  Dbf[g*8+t*2]=cb[0]; Dbf[g*8+t*2+1]=cb[1]; Dbf[(g+8)*8+t*2]=cb[2]; Dbf[(g+8)*8+t*2+1]=cb[3];
}
template<int KIND,int NM,int NN> void runrate(const char* name,int warps_per_sm){
  int sms; CK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));
  float* o; CK(cudaMalloc(&o,4)); int iters=20000;
  rate<KIND,NM,NN><<<sms,32*warps_per_sm>>>(100,o); CK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0); rate<KIND,NM,NN><<<sms,32*warps_per_sm>>>(iters,o); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
  float ms; cudaEventElapsedTime(&ms,e0,e1);
  double kk = KIND==0?16:32; double flop=2.0*16*8*kk*NM*NN*(double)iters*warps_per_sm*sms;
  printf("%-10s NM=%d NN=%d warps/SM=%2d : %7.1f TFLOP/s (%.3f ms)\n",name,NM,NN,warps_per_sm,flop/ms/1e9,ms);
  cudaFree(o);
}
int main(int argc,char**argv){
  int dev=argc>1?atoi(argv[1]):0; CK(cudaSetDevice(dev));
  for(int w: {4,8,16}){
    runrate<0,3,4>("bf16",w); runrate<1,3,4>("e4m3/f32",w); runrate<2,3,4>("e4m3/f16",w);
  }
  runrate<0,2,4>("bf16",8); runrate<1,2,4>("e4m3/f32",8); runrate<1,6,4>("e4m3/f32",4); runrate<1,6,4>("e4m3/f32",8);
  // precision
  for(int ksteps: {4,40,160}){
    int K=32*ksteps; std::vector<uint8_t> A8(16*K),B8(8*K); std::vector<__nv_bfloat16> Ab(16*K),Bb(8*K);
    srand(1+ksteps);
    for(int i=0;i<16*K;i++){ uint8_t v=rand()&0xff; if((v&0x7f)==0x7f) v^=1; A8[i]=v; }
    for(int i=0;i<8*K;i++){ uint8_t v=rand()&0xff; if((v&0x7f)==0x7f) v^=1; B8[i]=v; }
    auto f8=[](uint8_t v){ __nv_fp8_e4m3 x; x.__x=v; return (float)x; };
    for(int i=0;i<16*K;i++) Ab[i]=__float2bfloat16(f8(A8[i]));
    for(int i=0;i<8*K;i++) Bb[i]=__float2bfloat16(f8(B8[i]));
    uint8_t *dA8,*dB8; __nv_bfloat16 *dAb,*dBb; float *dD8,*dDb;
    CK(cudaMalloc(&dA8,16*K)); CK(cudaMalloc(&dB8,8*K)); CK(cudaMalloc(&dAb,32*K)); CK(cudaMalloc(&dBb,16*K)); CK(cudaMalloc(&dD8,512)); CK(cudaMalloc(&dDb,512));
    cudaMemcpy(dA8,A8.data(),16*K,cudaMemcpyHostToDevice); cudaMemcpy(dB8,B8.data(),8*K,cudaMemcpyHostToDevice);
    cudaMemcpy(dAb,Ab.data(),32*K,cudaMemcpyHostToDevice); cudaMemcpy(dBb,Bb.data(),16*K,cudaMemcpyHostToDevice);
    prec<<<1,32>>>(dA8,dB8,dAb,dBb,ksteps,dD8,dDb); CK(cudaDeviceSynchronize());
    float D8[128],Db[128]; cudaMemcpy(D8,dD8,512,cudaMemcpyDeviceToHost); cudaMemcpy(Db,dDb,512,cudaMemcpyDeviceToHost);
    double e8=0,eb=0,d8b=0; int same=0;
    for(int r=0;r<16;r++) for(int n=0;n<8;n++){
      double ref=0, mag=0; for(int k=0;k<K;k++){ double p=(double)f8(A8[r*K+k])*f8(B8[n*K+k]); ref+=p; mag+=fabs(p);} 
      e8=fmax(e8,fabs(D8[r*8+n]-ref)/mag); eb=fmax(eb,fabs(Db[r*8+n]-ref)/mag); d8b=fmax(d8b,fabs(D8[r*8+n]-Db[r*8+n])/mag); same+=(D8[r*8+n]==Db[r*8+n]);
    }
    printf("prec K=%5d: max |err|/sum|p|: e4m3 %.3g  bf16 %.3g  e4m3-vs-bf16 %.3g  bitwise-equal %d/128  (2^-24=%.3g)\n",K,e8,eb,d8b,same,ldexp(1.0,-24));
  }
  return 0;
}
