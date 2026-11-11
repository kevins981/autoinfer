// 2-warps/scheduler FP8 x bf16 decode GEMM (M<=48) reading Marlin's repacked FP8 weights + group-128 scales.
// y[M,N] = x[M,K] * dequant(W). CTA = 8 warps (2 n32 halves x 4 k-warps), A fragments of the CTA's K slice held in registers.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

#define NST 5
#define STAGE_B 8192
#define STAGE_BYTES (8192 + 128)
#define RED_BYTES (8 * 48 * 32 * 4)

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void arrive_release(int* p) { asm volatile("atom.add.release.gpu.global.u32 _, [%0], 1;\n" :: "l"(p) : "memory"); }
__device__ __forceinline__ int ld_acquire(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }

// GS = groups(128 K) per K slice, NM = m16 tiles, CPC = 64-col chunks per CTA, NS = K slices (=8)
template <int GS, int NM, int CPC, int NS>
__global__ void __launch_bounds__(256, 1)
g2w_kernel(const __nv_bfloat16* __restrict__ x, const int4* __restrict__ B, const __nv_bfloat16* __restrict__ S,
           __nv_bfloat16* __restrict__ y, float* __restrict__ ws, int* __restrict__ cnt,
           int M, int N, int K, int NCH, int dbg, unsigned long long* tm) {
  unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  extern __shared__ __align__(16) unsigned char smem[];
  float* red = (float*)(smem + NST * STAGE_BYTES);
  const int tid = threadIdx.x, warp = tid >> 5, l = tid & 31;
  const int h = warp & 1, kw = warp >> 1;
  const int per = NCH / CPC;                    // CTAs per K slice
  const int ks = blockIdx.x / per, chunk0 = (blockIdx.x % per) * CPC;
  const uint32_t smem_base = (uint32_t)__cvta_generic_to_shared(smem);
  float* wsk = ws + (size_t)ks * NCH * 3072;

  const int T = CPC * GS;
  auto issue = [&](int t) {
    if (t < T) {
      int chunk = chunk0 + t / GS, g = ks * GS + t % GS;
      uint32_t sb = smem_base + (t % NST) * STAGE_BYTES;
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        int p = tid + j * 256, sub = p >> 6, q = p & 63;
        const char* src = (const char*)B + ((size_t)((g * 8 + sub) * NCH + chunk)) * 1024 + q * 16;
        cp16(sb + sub * 1024 + ((q & 1) * 32 + (q >> 1)) * 16, src);
      }
      if (tid < 8) cp16(sb + STAGE_B + tid * 16, (const char*)S + ((size_t)g * N + (size_t)chunk * 64) * 2 + tid * 16);
    }
    cp_commit();
  };
  #pragma unroll
  for (int t = 0; t < NST - 1; t++) issue(t);

  uint32_t a[GS * 2][NM][4];
  auto loadA = [&](int gg) {
    #pragma unroll
    for (int ii = 0; ii < 2; ii++) {
      int s = gg * 2 + ii;
      int kt = (ks * GS + gg) * 8 + 2 * kw + ii;
      int k0 = kt * 16 + (l & 3) * 2;
      #pragma unroll
      for (int m = 0; m < NM; m++) {
        int r0 = m * 16 + (l >> 2), r1 = r0 + 8;
        a[s][m][0] = r0 < M ? __ldg((const uint32_t*)(x + (size_t)r0 * K + k0)) : 0u;
        a[s][m][1] = r1 < M ? __ldg((const uint32_t*)(x + (size_t)r1 * K + k0)) : 0u;
        a[s][m][2] = r0 < M ? __ldg((const uint32_t*)(x + (size_t)r0 * K + k0 + 8)) : 0u;
        a[s][m][3] = r1 < M ? __ldg((const uint32_t*)(x + (size_t)r1 * K + k0 + 8)) : 0u;
      }
    }
  };
  loadA(0);
  if (tid == 0) tm[blockIdx.x * 4 + 0] = t0;

  // epilogue pieces of chunk `cc` (red buffer holds the 4 k-warp partials of this CTA)
  auto e1 = [&](int cc) {   // sum 4 k-warps in fixed order, store the partial for this K slice
    #pragma unroll
    for (int i = 0; i < NM; i++) {
      int f = tid + 256 * i, row = f >> 4, cg = (f & 15) * 4, hh = cg >> 5, cc4 = (cg & 31) >> 2;
      if (row < M) {
        const float* p0 = red + (hh * 48 + row) * 32 + ((cc4 ^ (row & 7)) << 2);
        float4 v = *(const float4*)p0;
        #pragma unroll
        for (int w = 1; w < 4; w++) {
          float4 u = *(const float4*)(p0 + w * 2 * 48 * 32);
          v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w;
        }
        __stcg((float4*)(wsk + (size_t)(chunk0 + cc) * 3072 + row * 64 + cg), v);
      }
    }
  };
  auto e2 = [&](int cc) { if (tid == 0) arrive_release(&cnt[chunk0 + cc]); };

  int t = 0;
  for (int ci = 0; ci < CPC; ci++) {
    float c[NM][4][4];
    #pragma unroll
    for (int m = 0; m < NM; m++)
      #pragma unroll
      for (int n = 0; n < 4; n++)
        #pragma unroll
        for (int j = 0; j < 4; j++) c[m][n][j] = 0.f;

    #pragma unroll
    for (int g = 0; g < GS; g++, t++) {
      if (ci == 0 && g + 1 < GS) loadA(g + 1);
      cp_wait<NST - 2>();
      __syncthreads();
      issue(t + NST - 1);
      if (ci > 0 && g == 0) e1(ci - 1);
      if (ci > 0 && g == 1) e2(ci - 1);
      const unsigned char* sb = smem + (t % NST) * STAGE_BYTES;
      uint2 scw = *(const uint2*)(sb + STAGE_B + ((l >> 2) * 8 + h * 4) * 2);
      uint32_t sc[4] = {(scw.x & 0xffffu) * 0x10001u, (scw.x >> 16) * 0x10001u, (scw.y & 0xffffu) * 0x10001u, (scw.y >> 16) * 0x10001u};
      #pragma unroll
      for (int i = 0; i < 2; i++) {
        uint4 q4 = *(const uint4*)(sb + (2 * kw + i) * 1024 + (h * 32 + l) * 16);
        uint32_t qq[4] = {q4.x, q4.y, q4.z, q4.w};
        #pragma unroll
        for (int n = 0; n < 4; n++) {
          uint32_t q = qq[n];
          uint32_t o1 = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 4);
          q <<= 8;
          uint32_t o2 = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 4);
          __nv_bfloat162 s2 = *(__nv_bfloat162*)&sc[n];
          __nv_bfloat162 b1 = __hmul2(*(__nv_bfloat162*)&o1, s2), b0 = __hmul2(*(__nv_bfloat162*)&o2, s2);
          uint32_t ub0 = *(uint32_t*)&b0, ub1 = *(uint32_t*)&b1;
          #pragma unroll
          for (int m = 0; m < NM; m++) mma16816(c[m][n], a[g * 2 + i][m], ub0, ub1);
        }
      }
    }
    if (ci == 0 && tid == 0) { unsigned long long t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); tm[blockIdx.x * 4 + 1] = t1; }
    // acc -> red (swizzled), consumed by e1 after the next barrier
    #pragma unroll
    for (int m = 0; m < NM; m++)
      #pragma unroll
      for (int n = 0; n < 4; n++) {
        int r = m * 16 + (l >> 2), gr = (n * 2 + ((l & 3) >> 1));
        float* b0p = red + ((kw * 2 + h) * 48 + r) * 32;
        *(float2*)(b0p + ((gr ^ (r & 7)) << 2) + (l & 1) * 2) = make_float2(c[m][n][0], c[m][n][1]);
        *(float2*)(b0p + 8 * 32 + ((gr ^ ((r + 8) & 7)) << 2) + (l & 1) * 2) = make_float2(c[m][n][2], c[m][n][3]);
      }
  }
  __syncthreads();
  e1(CPC - 1);
  __syncthreads();
  e2(CPC - 1);
  if (tid == 0) { unsigned long long t3; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t3)); tm[blockIdx.x * 4 + 3] = t3; }
  // ---- owner phase: this CTA reduces chunks ci with ci % NS == ks over the NS K slices (fixed order)
  for (int ci = ks; ci < CPC; ci += NS) {
    int chunk = chunk0 + ci;
    if (tid == 0) { while (ld_acquire(&cnt[chunk]) < NS) { } }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < NM; i++) {
      int f = tid + 256 * i, row = f >> 4, cg = (f & 15) * 4;
      if (row < M) {
        float4 v[NS];
        #pragma unroll
        for (int s = 0; s < NS; s++) v[s] = __ldcg((const float4*)(ws + ((size_t)s * NCH + chunk) * 3072 + row * 64 + cg));
        float4 r = v[0];
        #pragma unroll
        for (int s = 1; s < NS; s++) { r.x += v[s].x; r.y += v[s].y; r.z += v[s].z; r.w += v[s].w; }
        __nv_bfloat162 o0 = __floats2bfloat162_rn(r.x, r.y), o1 = __floats2bfloat162_rn(r.z, r.w);
        *(uint2*)(y + (size_t)row * N + (size_t)chunk * 64 + cg) = make_uint2(*(uint32_t*)&o0, *(uint32_t*)&o1);
      }
    }
    __syncthreads();
    if (tid == 0) cnt[chunk] = 0;
  }
  if (tid == 0) { unsigned long long t2; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t2)); tm[blockIdx.x * 4 + 2] = t2; }
}

template <int GS, int NM, int CPC, int NS>
static void launch(const torch::Tensor& x, const torch::Tensor& B, const torch::Tensor& S, torch::Tensor& y,
                   torch::Tensor& ws, torch::Tensor& cnt, int M, int N, int K, int dbg, torch::Tensor& tm) {
  int NCH = N / 64;
  int grid = (NCH / CPC) * NS;
  size_t smem = NST * STAGE_BYTES + RED_BYTES;
  auto kfn = g2w_kernel<GS, NM, CPC, NS>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kfn<<<grid, 256, smem, c10::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), (const int4*)B.data_ptr(), (const __nv_bfloat16*)S.data_ptr(),
      (__nv_bfloat16*)y.data_ptr(), (float*)ws.data_ptr(), (int*)cnt.data_ptr(), M, N, K, NCH, dbg, (unsigned long long*)tm.data_ptr());
}

// gate_up: K=5120 (40 groups), NS=8 -> GS=5; N=17408 -> 272 chunks, CPC=17 -> grid 128
void g2w_gemm(torch::Tensor x, torch::Tensor B, torch::Tensor S, torch::Tensor y, torch::Tensor ws, torch::Tensor cnt, int64_t dbg, torch::Tensor tm) {
  int M = x.size(0), K = x.size(1), N = y.size(1);
  TORCH_CHECK(M <= 48 && K % 128 == 0 && N % 64 == 0);
  int G = K / 128;
  int nm = (M + 15) / 16;
  if (G == 40 && N == 17408) {
    if (nm == 1) launch<5, 1, 17, 8>(x, B, S, y, ws, cnt, M, N, K, dbg, tm);
    else if (nm == 2) launch<5, 2, 17, 8>(x, B, S, y, ws, cnt, M, N, K, dbg, tm);
    else launch<5, 3, 17, 8>(x, B, S, y, ws, cnt, M, N, K, dbg, tm);
  } else TORCH_CHECK(false, "unsupported shape");
}

__global__ void spin_kernel(long ns){ unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t0)); unsigned long long t; do { __nanosleep(200); asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)); } while ((long)(t-t0) < ns); }
void spin(int64_t ns){ spin_kernel<<<1,32,0,c10::cuda::getCurrentCUDAStream()>>>(ns); }
__global__ void l2pf_kernel(const uint4* p, long vec, long step) {
  long nt = (long)gridDim.x * blockDim.x;
  for (long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * step; i < vec; i += nt * step)
    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + i));
}
void l2pf(torch::Tensor w, int64_t bytes, int64_t blocks) {
  bytes = std::min<int64_t>(bytes, w.numel() * w.element_size());
  l2pf_kernel<<<blocks, 512, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)w.data_ptr(), bytes / 16, 2);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &g2w_gemm); m.def("l2pf", &l2pf); m.def("spin", &spin); }
