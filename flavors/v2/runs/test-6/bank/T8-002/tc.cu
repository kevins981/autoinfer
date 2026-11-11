// Premise microbench for the M=48 decode GEMM warm path on sm_89.
// P1: raw HMMA bf16->fp32 rate (operands in registers), warps/scheduler 1,2,4.
// P2: Marlin-like inner loop from SMEM: ldmatrix A (3 m16 tiles), FP8 B word from smem -> bf16 (shift/mask) * scale
//     (HMUL2.BF16), 3 mma per n8 tile; warps/scheduler 1,2.
// P3: cp.async.cg L2->SMEM streaming bandwidth from an L2-resident buffer vs an HBM-cold buffer.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
// P1
template<int NT>
__global__ void p1(int iters, float* out){
  uint32_t a[3][4], b[NT][2]; float c[3][NT][4];
  for(int m=0;m<3;m++) for(int j=0;j<4;j++) a[m][j]=0x3f803f80u ^ (threadIdx.x*(m+1)+j);
  for(int n=0;n<NT;n++){ b[n][0]=0x3f803f80u+n; b[n][1]=0x3f80bf80u+n; }
  for(int m=0;m<3;m++) for(int n=0;n<NT;n++) for(int j=0;j<4;j++) c[m][n][j]=0.f;
  for(int i=0;i<iters;i++)
    #pragma unroll
    for(int m=0;m<3;m++)
      #pragma unroll
      for(int n=0;n<NT;n++) mma(c[m][n],a[m],b[n]);
  float s=0; for(int m=0;m<3;m++) for(int n=0;n<NT;n++) for(int j=0;j<4;j++) s+=c[m][n][j];
  if(s==1.2345f) out[0]=s;
}
// P2: smem holds A (48 x 64 bf16 = 6 KB, ldmatrix layout irrelevant for timing) and B (fp8, 64 k x 8*NT n per warp).
template<int NT>
__global__ void p2(int iters, float* out){
  extern __shared__ __align__(16) unsigned char sm[];
  __nv_bfloat16* As = (__nv_bfloat16*)sm;                     // 48*64*2 = 6144 B
  uint32_t* Bs = (uint32_t*)(sm + 6144);                     // per warp: 4 k16 steps * NT * 32 lanes words
  int w = threadIdx.x/32, l = threadIdx.x%32, nw = blockDim.x/32;
  for(int i=threadIdx.x;i<48*64;i+=blockDim.x) As[i]=__float2bfloat16(0.01f*(i%7));
  for(int i=threadIdx.x;i<nw*4*NT*32;i+=blockDim.x) Bs[i]=0x38403c44u ^ i;
  __syncthreads();
  float c[3][NT][4]; for(int m=0;m<3;m++) for(int n=0;n<NT;n++) for(int j=0;j<4;j++) c[m][n][j]=0.f;
  __nv_bfloat162 sc = __floats2bfloat162_rn(1.5f, 1.5f);
  uint32_t* Bw = Bs + w*4*NT*32;
  for(int it=0; it<iters; it++){
    #pragma unroll
    for(int ks=0; ks<4; ks++){
      uint32_t a[3][4];
      #pragma unroll
      for(int m=0;m<3;m++){
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(As + (m*16 + (l%16))*64 + ks*16 + (l/16)*8);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(a[m][0]),"=r"(a[m][1]),"=r"(a[m][2]),"=r"(a[m][3]) : "r"(addr));
      }
      #pragma unroll
      for(int n=0;n<NT;n++){
        uint32_t q = Bw[(ks*NT+n)*32 + l];
        // e4m3 -> bf16 (exact, unscaled by 2^120 here; the real kernel folds that into the scale): byte -> (s<<8)|(em<<4)
        uint32_t lo = __byte_perm(q, 0, 0x5150);   // bytes 0,1 into high bytes of each half: 0xB1 00 B0 00 pattern
        uint32_t hi = __byte_perm(q, 0, 0x7372);
        uint32_t b0 = (lo & 0x80008000u) | ((lo & 0x7f007f00u) >> 4);
        uint32_t b1 = (hi & 0x80008000u) | ((hi & 0x7f007f00u) >> 4);
        __nv_bfloat162 x0 = __hmul2(*(__nv_bfloat162*)&b0, sc), x1 = __hmul2(*(__nv_bfloat162*)&b1, sc);
        uint32_t b[2] = {*(uint32_t*)&x0, *(uint32_t*)&x1};
        #pragma unroll
        for(int m=0;m<3;m++) mma(c[m][n], a[m], b);
      }
    }
  }
  float s=0; for(int m=0;m<3;m++) for(int n=0;n<NT;n++) for(int j=0;j<4;j++) s+=c[m][n][j];
  if(s==1.2345f) out[0]=s;
}
// P3: each CTA streams its slice of buf through an 8-stage smem ring with cp.async.cg 16 B (7 groups in flight).
__global__ void p3(const int4* __restrict__ buf, size_t n16, int reps, float* out){
  extern __shared__ __align__(16) int4 ring[]; // ST * blockDim.x
  const int ST=8; size_t per = n16/gridDim.x; size_t base = blockIdx.x*per; size_t nch = per/blockDim.x;
  int acc=0;
  for(int r=0;r<reps;r++){
    for(size_t c=0;c<nch+ST-1;c++){
      if (c<nch){
        uint32_t d=(uint32_t)__cvta_generic_to_shared(&ring[(c%ST)*blockDim.x+threadIdx.x]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(buf+base+c*blockDim.x+threadIdx.x));
      }
      asm volatile("cp.async.commit_group;\ncp.async.wait_group 7;\n" ::);
      if (c>=ST-1) acc ^= ring[((c-ST+1)%ST)*blockDim.x+threadIdx.x].x;
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
  }
  if(acc==0x7777777) out[0]=acc;
}

int main(int argc,char**argv){
  int dev=argc>1?atoi(argv[1]):0; CK(cudaSetDevice(dev));
  int sms, clk; cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev); cudaDeviceGetAttribute(&clk,cudaDevAttrClockRate,dev);
  float* out; CK(cudaMalloc(&out,64));
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  printf("SMs %d, attr clock %d MHz. Spec bf16xfp32 dense 165.2 TF @2520 MHz -> per-SM-clk 512 flop\n", sms, clk/1000);
  // P1
  for(int wps : {1,2,4}){
    int threads = 128*wps, iters=20000; const int NT=4;
    p1<NT><<<sms,threads>>>(10,out); CK(cudaDeviceSynchronize());
    cudaEventRecord(a); p1<NT><<<sms,threads>>>(iters,out); cudaEventRecord(b); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms,a,b);
    double fl = 4096.0*3*NT*iters*(threads/32)*(double)sms;
    printf("P1 regs-only HMMA  warps/sched %d: %.1f TFLOP/s\n", wps, fl/ms/1e9);
  }
  // P2
  for(int wps : {1,2}){
    const int NT=4; int threads=128*wps, iters=4000; int nw=threads/32;
    size_t smem = 6144 + (size_t)nw*4*NT*32*4;
    CK(cudaFuncSetAttribute(p2<NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    p2<NT><<<sms,threads,smem>>>(10,out); CK(cudaDeviceSynchronize());
    cudaEventRecord(a); p2<NT><<<sms,threads,smem>>>(iters,out); cudaEventRecord(b); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms,a,b);
    double fl = 4096.0*3*NT*4*iters*nw*(double)sms;
    printf("P2 smem ldmatrix+fp8 dequant+HMMA (M=48) warps/sched %d: %.1f TFLOP/s\n", wps, fl/ms/1e9);
  }
  // P3
  size_t big = 1ull<<30, l2 = 32ull<<20;
  int4* buf; CK(cudaMalloc(&buf,big)); CK(cudaMemset(buf,1,big));
  for(int ctas_per_sm : {1,2}){
    int threads=256; size_t smem = 8*threads*16;
    for(int mode=0; mode<2; mode++){
      size_t sz = mode? big : l2; int reps = mode? 1 : 32;
      p3<<<sms*ctas_per_sm,threads,smem>>>(buf, sz/16, 1, out); CK(cudaDeviceSynchronize());
      cudaEventRecord(a); p3<<<sms*ctas_per_sm,threads,smem>>>(buf, sz/16, reps, out); cudaEventRecord(b); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms,a,b);
      printf("P3 cp.async ring %s %3zu MB x%d, %d CTA/SM: %.0f GB/s\n", mode?"HBM ":"L2  ", sz>>20, reps, ctas_per_sm, (double)sz*reps/ms/1e6);
    }
  }
  return 0;
}
