// T17 M0: stream-K FP8(Marlin layout) x bf16 decode GEMM. CTA = 16 warps, tile = 256 cols x M rows (NM m16 tiles),
// warp (c,h,nn) owns 16 cols (2 n8 tiles) x all NM m-tiles; 4-stage cp.async ring (64-k stages: 16KB B + 8KB A + 0.5KB scales).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#ifndef NST
#define NST 4
#endif
#ifndef ABL
#define ABL 0
#endif
#define B_BYTES 16384
#define A_BYTES 8192
#define S_BYTES 512
#define ST_BYTES (B_BYTES + A_BYTES + S_BYTES)

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t* a, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}
__device__ __forceinline__ void arrive_release(int* p) { asm volatile("atom.add.release.gpu.global.u32 _, [%0], 1;\n" :: "l"(p) : "memory"); }
__device__ __forceinline__ int ld_acquire(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }

__device__ unsigned long long g_tm[256*4];
__device__ __forceinline__ unsigned long long gt(){unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)); return t;}
template <int NM>
__global__ void __launch_bounds__(512, 1)
g17_kernel(const __nv_bfloat16* __restrict__ x, const char* __restrict__ B, const char* __restrict__ S,
           __nv_bfloat16* __restrict__ y, float* __restrict__ ws, int* __restrict__ flags, int M, int N, int K) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, w = tid >> 5, l = tid & 31;
  const int c = w >> 2, h = (w >> 1) & 1, nn = w & 1;
  const int G = K >> 7, NT = N >> 8, NCH = N >> 6;
  const int nb = gridDim.x, b = blockIdx.x;
  const long U = (long)NT * G;
  const int u0 = (int)(U * b / nb), u1 = (int)(U * (b + 1) / nb);
  const int S_total = 2 * (u1 - u0);
  const uint32_t smem_base = (uint32_t)__cvta_generic_to_shared(smem);
  if (tid==0) g_tm[b*4+0]=gt();

  auto issue = [&](int s) {
    if (s < S_total && !(ABL & 2)) {
      int u = u0 + (s >> 1), hs = s & 1;
      int tile = u / G, g = u - tile * G;
      uint32_t sb = smem_base + (s % NST) * ST_BYTES;
      if (!(ABL & 32))
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        int p = tid + j * 512, sub = p >> 8, q256 = p & 255, cc = q256 >> 6, q = q256 & 63;
        const char* src = B + ((size_t)((g * 8 + hs * 4 + sub) * NCH + tile * 4 + cc)) * 1024 + q * 16;
        cp16(sb + sub * 4096 + cc * 1024 + ((q & 1) * 32 + (q >> 1)) * 16, src);
      }
      if (tid < NM * 128 && !(ABL & 16)) {
        int row = tid >> 3, ch = tid & 7, srow = row < M ? row : M - 1;
        const char* src = (const char*)(x + (size_t)srow * K + g * 128 + hs * 64 + ch * 8);
        cp16(sb + B_BYTES + row * 128 + ((ch ^ (row & 7)) << 4), src);
      }
      if (tid < 32) cp16(sb + B_BYTES + A_BYTES + tid * 16, S + ((size_t)g * N + (size_t)tile * 256) * 2 + tid * 16);
    }
    cp_commit();
  };
  #pragma unroll
  for (int t = 0; t < NST - 1; t++) issue(t);

  float acc[NM * 8];
  int s = 0;
  int seg_u = u0;
  int fin_tile[2] = {-1, -1};
  for (int sg = 0; sg < 2 && seg_u < u1; sg++) {
    const int tile = seg_u / G, ka = seg_u - tile * G;
    const int seg_end = min(u1, (tile + 1) * G), kb = seg_end - tile * G;
    (void)ka;
    #pragma unroll
    for (int i = 0; i < NM * 8; i++) acc[i] = 0.f;
    for (int st = 0; st < 2 * (seg_end - seg_u); st++, s++) {
      cp_wait<NST - 2>();
      __syncthreads();
      issue(s + NST - 1);
      const unsigned char* sb = smem + (s % NST) * ST_BYTES;
      const uint32_t sbs = smem_base + (s % NST) * ST_BYTES;
      uint32_t scw = *(const uint32_t*)(sb + B_BYTES + A_BYTES + c * 128 + ((l >> 2) * 8 + h * 4) * 2 + nn * 4);
      uint32_t sc[2] = {(scw & 0xffffu) * 0x10001u, (scw >> 16) * 0x10001u};
      if (!(ABL & 8))
      #pragma unroll
      for (int step = 0; step < 4; step++) {
        uint2 q2 = *(const uint2*)(sb + step * 4096 + c * 1024 + (h * 32 + l) * 16 + nn * 8);
        uint32_t qq[2] = {q2.x, q2.y};
        uint32_t ub0[2], ub1[2];
        #pragma unroll
        for (int n = 0; n < 2; n++) {
          uint32_t q = qq[n];
          uint32_t o1 = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 4);
          q <<= 8;
          uint32_t o2 = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 4);
          __nv_bfloat162 s2 = *(__nv_bfloat162*)&sc[n];
          __nv_bfloat162 bb1 = __hmul2(*(__nv_bfloat162*)&o1, s2), bb0 = __hmul2(*(__nv_bfloat162*)&o2, s2);
          ub0[n] = *(uint32_t*)&bb0; ub1[n] = *(uint32_t*)&bb1;
        }
        #pragma unroll
        for (int m = 0; m < NM; m++) {
          uint32_t a[4];
          int row = m * 16 + (l & 7) + ((l >> 3) & 1) * 8, kc = step * 2 + (l >> 4);
          ldsm4(a, sbs + B_BYTES + row * 128 + ((kc ^ (row & 7)) << 4));
          #pragma unroll
          for (int n = 0; n < 2; n++) { if (ABL & 1) { acc[(m*2+n)*4] += __uint_as_float(a[0]^ub0[n]^ub1[n]^a[3]); } else mma16816(acc + (m * 2 + n) * 4, a, ub0[n], ub1[n]); }
        }
      }
    }
    if (ABL & 4) { seg_u = seg_end; if (tid==0 && acc[0]==12345.f) flags[0]=1; continue; }
    const bool last = (seg_end == u1);
    const bool fin = (kb == G);
    if (fin && last) { fin_tile[sg] = tile; /* keep acc in regs; finish after loop */ }
    else {
      float* slot = ws + (size_t)(b * 2 + sg) * (32 * 512);
      #pragma unroll
      for (int j = 0; j < NM * 8; j++) __stcg(slot + j * 512 + tid, acc[j]);
      if (!fin) {
        __threadfence();
        __syncthreads();
        if (tid == 0) arrive_release(&flags[b * 2 + sg]);
      } else fin_tile[sg] = tile;
    }
    seg_u = seg_end;
  }
  __syncthreads(); if (tid==0) g_tm[b*4+1]=gt();
  // finisher duties: segments (in order) that complete a tile
  for (int sg = 0; sg < 2; sg++) {
    if (ABL & 4) break;
    const int tile = fin_tile[sg];
    if (tile < 0) continue;
    int cb[8], cs[8], nc = 0;
    for (int bb = b - 1; bb >= 0 && nc < 8; bb--) {
      long ub1 = U * (bb + 1) / nb;
      if (ub1 <= (long)tile * G) break;
      long ub0 = U * bb / nb;
      cb[nc] = bb; cs[nc] = ((int)(ub0 / G) == tile) ? 0 : 1; nc++;
    }
    if (tid == 0) { for (int i = 0; i < nc; i++) { while (ld_acquire(&flags[cb[i] * 2 + cs[i]]) < 1) { } } }
    __syncthreads();
    float tot[NM * 8];
    if (nc == 0) {
      #pragma unroll
      for (int j = 0; j < NM * 8; j++) tot[j] = 0.f;
    } else {
      const float* p = ws + (size_t)(cb[nc - 1] * 2 + cs[nc - 1]) * (32 * 512);
      #pragma unroll
      for (int j = 0; j < NM * 8; j++) tot[j] = __ldcg(p + j * 512 + tid);
      for (int i = nc - 2; i >= 0; i--) {
        const float* pp = ws + (size_t)(cb[i] * 2 + cs[i]) * (32 * 512);
        #pragma unroll
        for (int j = 0; j < NM * 8; j++) tot[j] += __ldcg(pp + j * 512 + tid);
      }
    }
    {
      const float* pm = ws + (size_t)(b * 2 + sg) * (32 * 512);
      bool inreg = (sg == 1) || (fin_tile[1] < 0 && sg == 0 && (u1 == (tile + 1) * G));
      #pragma unroll
      for (int j = 0; j < NM * 8; j++) tot[j] += inreg ? acc[j] : __ldcg(pm + j * 512 + tid);
    }
    __syncthreads();
    if (tid == 0) { for (int i = 0; i < nc; i++) flags[cb[i] * 2 + cs[i]] = 0; }
    #pragma unroll
    for (int m = 0; m < NM; m++)
      #pragma unroll
      for (int n = 0; n < 2; n++) {
        int r0 = m * 16 + (l >> 2), r1 = r0 + 8;
        int col = (tile * 4 + c) * 64 + h * 32 + (nn * 2 + n) * 8 + (l & 3) * 2;
        __nv_bfloat162 o0 = __floats2bfloat162_rn(tot[(m * 2 + n) * 4 + 0], tot[(m * 2 + n) * 4 + 1]);
        __nv_bfloat162 o1 = __floats2bfloat162_rn(tot[(m * 2 + n) * 4 + 2], tot[(m * 2 + n) * 4 + 3]);
        if (r0 < M) *(uint32_t*)(y + (size_t)r0 * N + col) = *(uint32_t*)&o0;
        if (r1 < M) *(uint32_t*)(y + (size_t)r1 * N + col) = *(uint32_t*)&o1;
      }
  }
  __syncthreads(); if (tid==0) g_tm[b*4+2]=gt();
}

template <int NM>
static void launch(const torch::Tensor& x, const torch::Tensor& B, const torch::Tensor& S, torch::Tensor& y,
                   torch::Tensor& ws, torch::Tensor& flags, int grid) {
  int M = x.size(0), K = x.size(1), N = y.size(1);
  size_t smem = (size_t)NST * ST_BYTES;
  auto kfn = g17_kernel<NM>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  kfn<<<grid, 512, smem, c10::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)x.data_ptr(), (const char*)B.data_ptr(), (const char*)S.data_ptr(),
      (__nv_bfloat16*)y.data_ptr(), (float*)ws.data_ptr(), (int*)flags.data_ptr(), M, N, K);
}
void g17_gemm(torch::Tensor x, torch::Tensor B, torch::Tensor S, torch::Tensor y, torch::Tensor ws, torch::Tensor flags, int64_t grid) {
  int M = x.size(0), K = x.size(1), N = y.size(1);
  TORCH_CHECK(M <= 64 && K % 128 == 0 && N % 256 == 0);
  int nm = (M + 15) / 16;
  if (nm <= 3) launch<3>(x, B, S, y, ws, flags, grid); else launch<4>(x, B, S, y, ws, flags, grid);
}
__global__ void l2pf_kernel(const uint4* p, long vec, long step) {
  long nt = (long)gridDim.x * blockDim.x;
  for (long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * step; i < vec; i += nt * step)
    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + i));
}
torch::Tensor get_tm(){ torch::Tensor t=torch::empty({256*4},torch::dtype(torch::kInt64).device(torch::kCUDA)); cudaMemcpyFromSymbol(t.data_ptr(),g_tm,256*4*8); return t; }
void l2pf(torch::Tensor w, int64_t bytes, int64_t blocks) {
  bytes = std::min<int64_t>(bytes, w.numel() * w.element_size());
  l2pf_kernel<<<blocks, 512, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)w.data_ptr(), bytes / 16, 2);
}
__global__ void spin_kernel(long ns){ unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t0)); unsigned long long t; do { __nanosleep(200); asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)); } while ((long)(t-t0) < ns); }
void spin(int64_t ns){ spin_kernel<<<1,32,0,c10::cuda::getCurrentCUDAStream()>>>(ns); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &g17_gemm); m.def("l2pf", &l2pf); m.def("get_tm", &get_tm); m.def("spin", &spin); }
