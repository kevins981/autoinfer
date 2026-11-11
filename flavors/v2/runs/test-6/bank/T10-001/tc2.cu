// leader-10: FP8 (e4m3) mma.sync f32-accumulate precision: alignment window inside one MMA, and C handling.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
__device__ __forceinline__ void mma_e4m3(float* c, const uint32_t* a, const uint32_t* b){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));}
// A: 16x32 bytes row-major, B: 8x32 bytes (n-major), C in: 16x8 floats; out D
__global__ void one(const uint8_t* A,const uint8_t* B,const float* C,float* D){
  int lane=threadIdx.x,g=lane>>2,t=lane&3;
  auto pa=[&](int r,int k){uint32_t v=0;for(int j=0;j<4;j++)v|=(uint32_t)A[r*32+k+j]<<(8*j);return v;};
  auto pb=[&](int k,int n){uint32_t v=0;for(int j=0;j<4;j++)v|=(uint32_t)B[n*32+k+j]<<(8*j);return v;};
  uint32_t a[4]={pa(g,t*4),pa(g+8,t*4),pa(g,16+t*4),pa(g+8,16+t*4)}, b[2]={pb(t*4,g),pb(16+t*4,g)};
  float c[4]={C[g*8+t*2],C[g*8+t*2+1],C[(g+8)*8+t*2],C[(g+8)*8+t*2+1]};
  mma_e4m3(c,a,b);
  D[g*8+t*2]=c[0];D[g*8+t*2+1]=c[1];D[(g+8)*8+t*2]=c[2];D[(g+8)*8+t*2+1]=c[3];
}
static float f8(uint8_t v){__nv_fp8_e4m3 x;x.__x=v;return (float)x;}
static uint8_t tof8(float f){__nv_fp8_e4m3 x(f);return x.__x;}
uint8_t *dA,*dB;float *dC,*dD;
void run(uint8_t*A,uint8_t*B,float*C,float*D){cudaMemcpy(dA,A,512,cudaMemcpyHostToDevice);cudaMemcpy(dB,B,256,cudaMemcpyHostToDevice);cudaMemcpy(dC,C,512,cudaMemcpyHostToDevice);one<<<1,32>>>(dA,dB,dC,dD);cudaMemcpy(D,dD,512,cudaMemcpyDeviceToHost);}
int main(){
  cudaMalloc(&dA,512);cudaMalloc(&dB,256);cudaMalloc(&dC,512);cudaMalloc(&dD,512);
  uint8_t A[512],B[256];float C[128],D[128];
  // Test 1: two products: big (2^8) and small (2^-s); C=0. find s where small is lost.
  printf("T1 single MMA, C=0: product 2^8 + product 2^-s (exact sum representable in fp32 for s<=15)\n");
  for(int s=0;s<=20;s+=1){
    for(int i=0;i<512;i++)A[i]=0; for(int i=0;i<256;i++)B[i]=0; for(int i=0;i<128;i++)C[i]=0;
    A[0]=tof8(16.f);B[0]=tof8(16.f); // 256
    float sm=ldexpf(1.f,-s); float a2=fminf(sm,1.f); float b2=sm/a2; // split small into representable factors
    if(b2<ldexpf(1,-9)){a2=ldexpf(1,-(s/2)); b2=ldexpf(1,-(s-s/2));}
    A[1]=tof8(a2);B[1]=tof8(b2);
    run(A,B,C,D); double ref=(double)f8(A[0])*f8(B[0])+(double)f8(A[1])*f8(B[1]);
    printf("  s=%2d  got %.10g  ref %.10g  %s\n",s,D[0],ref,D[0]==(float)ref?"exact":"LOST");
  }
  // Test 2: C=2^20, single product 2^0 ... C handling
  printf("T2 C=2^p, one product 1.0: D==C+1 ?\n");
  for(int p=0;p<=26;p+=2){
    for(int i=0;i<512;i++)A[i]=0; for(int i=0;i<256;i++)B[i]=0; for(int i=0;i<128;i++)C[i]=0;
    A[0]=tof8(1.f);B[0]=tof8(1.f);C[0]=ldexpf(1.f,p);
    run(A,B,C,D); double ref=ldexp(1.0,p)+1.0; printf("  p=%2d got %.10g ref %.10g %s\n",p,D[0],ref,D[0]==(float)ref?"exact":"LOST");
  }
  // Test 3: C small, product big: does C survive? C=1.0, product 2^p
  printf("T3 C=1, product 2^p: D==2^p+1 ?\n");
  for(int p=0;p<=16;p+=2){
    for(int i=0;i<512;i++)A[i]=0; for(int i=0;i<256;i++)B[i]=0; for(int i=0;i<128;i++)C[i]=0;
    A[0]=tof8(ldexpf(1.f,p/2));B[0]=tof8(ldexpf(1.f,p-p/2));C[0]=1.f;
    run(A,B,C,D); double ref=ldexp(1.0,p)+1.0; printf("  p=%2d got %.10g ref %.10g %s\n",p,D[0],ref,D[0]==(float)ref?"exact":"LOST");
  }
  return 0;
}
