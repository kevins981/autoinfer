// Toy: can bf16 mma.sync (f32 acc) and FFMA pipes run concurrently on sm_89?
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdint>
__global__ void k(int nt, int ITm, int ITf, float* out, long long* clk) {
  int w = threadIdx.x / 32;
  long long t0 = clock64();
  float res = 0.f;
  if (w < nt) {
    uint32_t a[4] = {0x3f803f80u ^ threadIdx.x, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u};
    uint32_t b[2] = {0x3f803f80u, 0x3f803f80u ^ threadIdx.x};
    float c[8][4];
    #pragma unroll
    for (int j = 0; j < 8; j++) for (int q = 0; q < 4; q++) c[j][q] = 0.f;
    for (int i = 0; i < ITm; i++) {
      #pragma unroll
      for (int j = 0; j < 8; j++)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
          : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) res += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  } else {
    float x[16]; float m = 1.0001f + threadIdx.x * 1e-7f, ad = 1e-7f;
    #pragma unroll
    for (int j = 0; j < 16; j++) x[j] = j;
    for (int i = 0; i < ITf; i++) {
      #pragma unroll
      for (int j = 0; j < 16; j++) x[j] = fmaf(x[j], m, ad);
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) res += x[j];
  }
  long long t1 = clock64();
  if (res == 12345.f) out[0] = res;
  if (threadIdx.x % 32 == 0) clk[blockIdx.x * 64 + w] = t1 - t0;
}
int main(int argc, char** argv) {
  int dev = 0; cudaGetDevice(&dev);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
  int sms = p.multiProcessorCount;
  float* out; long long* clk; cudaMalloc(&out, 4); cudaMalloc(&clk, sms * 64 * 8 * 4);
  // configs: (nt tensor warps, nf ffma warps) per block, 1 block/SM (grid = sms * bps)
  int cfg[][2] = {{4,0},{8,0},{0,4},{0,8},{0,16},{4,4},{4,8},{8,8},{4,12},{8,16}};
  int ITm = 4000, ITf = 16000;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int rep = 0; rep < 2; rep++)
  for (auto& c : cfg) {
    int nt = c[0], nf = c[1];
    int threads = 32 * (nt + nf);
    k<<<sms, threads>>>(nt, ITm, ITf, out, clk);  // warm
    cudaEventRecord(e0);
    int R = 20;
    for (int r = 0; r < R; r++) k<<<sms, threads>>>(nt, ITm, ITf, out, clk);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= R;
    double tflop_t = 2.0 * nt * 32 * 8 * (double)ITm * 2048.0 / 32 /*per-thread share*/ * sms;  // warp mma = 16*8*16 FMA = 2048 FMA
    // per warp per iter: 8 mma * 2048 FMA ; per block nt warps
    tflop_t = 2.0 * nt * 8.0 * ITm * 2048.0 * sms;
    double tflop_f = 2.0 * nf * 32.0 * 16.0 * ITf * sms;
    printf("dev%d nt=%d nf=%d  %.3f ms  tensor %.1f TF  ffma %.1f TF  total %.1f TF\n", dev, nt, nf, ms,
      tflop_t / ms / 1e9, tflop_f / ms / 1e9, (tflop_t + tflop_f) / ms / 1e9);
  }
  cudaError_t err = cudaGetLastError(); printf("err %s\n", cudaGetErrorString(err));
}
