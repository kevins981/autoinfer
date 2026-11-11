
// W1-3: swap-AB register-direct FP8 decode GEMM, 8 warps/SM, LDG.128 weight ring (L=8 pairs in flight), even work split.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#define SROW 264
#define TOKB (48 * SROW * 2)
#define RING 8
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ uint4 ldw(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ void deq(uint32_t q, uint32_t s2u, uint32_t& r0, uint32_t& r1) {
  uint32_t o1 = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 4);
  q <<= 8;
  uint32_t o2 = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 4);
  __nv_bfloat162 s2 = *(__nv_bfloat162*)&s2u;
  __nv_bfloat162 b1 = __hmul2(*(__nv_bfloat162*)&o1, s2), b0 = __hmul2(*(__nv_bfloat162*)&o2, s2);
  r0 = *(uint32_t*)&b0; r1 = *(uint32_t*)&b1;
}
struct Args {
  const __nv_bfloat16* x; const unsigned char* W; const uint32_t* SC; __nv_bfloat16* y; float* part; int* flags;
  int N, K, tiles, TG, KSP, cps, KP, G;
};
// S streams (tiles) per warp, NP pairs per stream per chunk; HASE: extra tile E (gate_up mode)
template <int S, int NP, int HASE, int ABL>
__global__ void __launch_bounds__(256, 1) swab3(Args a) {
  constexpr int PPC = S * NP;           // ring pairs per chunk
  constexpr int U = (PPC >= RING) ? 1 : RING / PPC;
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, l = tid & 31, w = tid >> 5, bid = blockIdx.x;
  const int ks = a.KSP - 1 - bid / a.TG, gid = bid % a.TG;
  const int cps = a.cps, chunk0 = ks * cps;
  const int t0 = gid * (HASE ? 17 : 4);
  int tl[S], qb;
  if (HASE) { tl[0] = t0 + w; if (S > 1) tl[S - 1] = t0 + w + 8; qb = 0; }
  else { tl[0] = t0 + (w & 3); qb = 4 * (w >> 2); }
  const unsigned char* wp[S];
  const uint32_t* sp[S];
  #pragma unroll
  for (int s = 0; s < S; s++) { wp[s] = a.W + (size_t)tl[s] * a.KP * 512 + l * 16; sp[s] = a.SC + (size_t)tl[s] * a.G * 8 + (l >> 2); }
  const unsigned char* wpE = a.W + (size_t)(t0 + 16) * a.KP * 512 + l * 16;
  const uint32_t* spE = a.SC + (size_t)(t0 + 16) * a.G * 8 + (l >> 2);
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  auto issue = [&](int c) {
    uint32_t dst = sbase + (c & 1) * TOKB;
    for (int i = tid; i < 48 * 32; i += 256) {
      int row = i >> 5, pc = i & 31;
      cp16(dst + (row * SROW + pc * 8) * 2, a.x + (size_t)row * a.K + (size_t)(chunk0 + c) * 256 + pc * 8);
    }
    cp_commit();
  };
  float acc[S][6][4], accE[6][4];
  #pragma unroll
  for (int s = 0; s < S; s++)
    #pragma unroll
    for (int n = 0; n < 6; n++)
      #pragma unroll
      for (int j = 0; j < 4; j++) acc[s][n][j] = 0.f;
  #pragma unroll
  for (int n = 0; n < 6; n++)
    #pragma unroll
    for (int j = 0; j < 4; j++) accE[n][j] = 0.f;
  // pair p-major ordering: pp = p*S + s
  uint4 ring[RING];
  auto qof = [&](int p) { return qb + p; };
  auto load_pair = [&](int pp, int cabs) -> uint4 {
    int p = pp / S, s = pp % S;
    return ldw(wp[s] + (size_t)(cabs * 8 + qof(p)) * 512);
  };
  constexpr int NG = (S == 2 && HASE) ? 2 : 1;   // scale groups per stream per chunk
  uint32_t scn[S][NG], scc[S][NG], scEn = 0, scEc = 0;
  uint4 wEn = make_uint4(0, 0, 0, 0), wEc;
  auto load_scales = [&](int cabs) {
    #pragma unroll
    for (int s = 0; s < S; s++)
      #pragma unroll
      for (int g = 0; g < NG; g++) scn[s][g] = __ldg(sp[s] + (size_t)(cabs * 2 + (HASE ? g : (w >> 2))) * 8);
    if (HASE) { scEn = __ldg(spE + (size_t)(cabs * 2 + (w >> 2)) * 8); wEn = ldw(wpE + (size_t)(cabs * 8 + w) * 512); }
  };
  if (ABL == 1) {
    #pragma unroll
    for (int r = 0; r < RING; r++) ring[r] = make_uint4(tid * 7 + r, tid * 11, tid * 13 + 5, tid * 17 + r);
    #pragma unroll
    for (int s = 0; s < S; s++)
      #pragma unroll
      for (int g = 0; g < NG; g++) scn[s][g] = 0x7B807B80u;
    scEn = 0x7B807B80u; wEn = make_uint4(tid, tid * 3, tid * 5, tid * 7);
  } else {
    #pragma unroll
    for (int r = 0; r < RING; r++) ring[r] = load_pair(r % PPC, chunk0 + r / PPC);
    load_scales(chunk0);
  }
  const uint32_t lbase = sbase + (((l & 7) + ((l >> 4) << 3)) * SROW + ((l >> 3) & 1) * 8) * 2;
  uint32_t sink = 0;
  issue(0);
  for (int c0 = 0; c0 < cps; c0 += U) {
    #pragma unroll
    for (int u = 0; u < U; u++) {
      const int c = c0 + u;
      cp_wait0();
      __syncthreads();
      if (c + 1 < cps) issue(c + 1); else cp_commit();
      #pragma unroll
      for (int s = 0; s < S; s++)
        #pragma unroll
        for (int g = 0; g < NG; g++) scc[s][g] = scn[s][g];
      scEc = scEn; wEc = wEn;
      if (ABL != 1 && c + 1 < cps) load_scales(chunk0 + c + 1);
      const uint32_t tb = lbase + (c & 1) * TOKB;
      #pragma unroll
      for (int p = 0; p < NP; p++) {
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
          uint32_t b[6][2];
          if (ABL != 2) {
            const int ksl = qof(p) * 2 + sub;
            #pragma unroll
            for (int pi = 0; pi < 3; pi++) ldsm4(b[2 * pi][0], b[2 * pi][1], b[2 * pi + 1][0], b[2 * pi + 1][1], tb + ksl * 32 + pi * 16 * SROW * 2);
          }
          #pragma unroll
          for (int s = 0; s < S; s++) {
            const int pp = p * S + s, slot = (u * PPC + pp) % RING;
            uint4 wv = ring[slot];
            if (sub == 1) {
              // last use of this pair: refill the slot RING pairs ahead
              const int rel = pp + RING, dc = rel / PPC, pp2 = rel % PPC;
              if (ABL == 1) { ring[slot].x += 0x01010101u; ring[slot].y ^= c; ring[slot].w ^= ring[slot].x; }
              else if (c + dc < cps) ring[slot] = load_pair(pp2, chunk0 + c + dc);
            }
            const uint32_t x = sub ? wv.z : wv.x, y = sub ? wv.w : wv.y;
            const uint32_t sc = scc[s][HASE ? (p >> 2) : 0];
            if (ABL == 2) { sink ^= x ^ y; continue; }
            uint32_t a0, a1, a2, a3;
            deq(x, (sc & 0xffffu) * 0x10001u, a0, a2);
            deq(y, (sc >> 16) * 0x10001u, a1, a3);
            #pragma unroll
            for (int n = 0; n < 6; n++) mma16816(acc[s][n], a0, a1, a2, a3, b[n][0], b[n][1]);
          }
        }
      }
      if (HASE) {
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
          if (ABL == 2) { sink ^= wEc.x ^ wEc.y ^ wEc.z ^ wEc.w; break; }
          uint32_t b[6][2];
          const int ksl = w * 2 + sub;
          #pragma unroll
          for (int pi = 0; pi < 3; pi++) ldsm4(b[2 * pi][0], b[2 * pi][1], b[2 * pi + 1][0], b[2 * pi + 1][1], tb + ksl * 32 + pi * 16 * SROW * 2);
          const uint32_t x = sub ? wEc.z : wEc.x, y = sub ? wEc.w : wEc.y;
          uint32_t a0, a1, a2, a3;
          deq(x, (scEc & 0xffffu) * 0x10001u, a0, a2);
          deq(y, (scEc >> 16) * 0x10001u, a1, a3);
          #pragma unroll
          for (int n = 0; n < 6; n++) mma16816(accE[n], a0, a1, a2, a3, b[n][0], b[n][1]);
        }
        if (ABL == 1) { wEn.x += 0x01010101u; wEn.y ^= c; }
      }
    }
  }
  if (ABL == 2) acc[0][0][0] += (float)(sink & 0xff);
  // ---------------- epilogue
  float* red = (float*)smem;
  __syncthreads();
  auto colof = [&](int tile, int g) { return (tile >> 2) * 64 + (tile & 3) * 16 + g; };
  if (!HASE) {
    if (w >= 4) {
      #pragma unroll
      for (int n = 0; n < 6; n++)
        #pragma unroll
        for (int j = 0; j < 4; j++) red[(((w - 4) * 24 + n * 4 + j) << 5) + l] = acc[0][n][j];
    }
    __syncthreads();
    if (w < 4) {
      const int col0 = colof(tl[0], l >> 2);
      #pragma unroll
      for (int n = 0; n < 6; n++) {
        const int tok = n * 8 + 2 * (l & 3);
        float v0 = acc[0][n][0] + red[((w * 24 + n * 4 + 0) << 5) + l], v1 = acc[0][n][1] + red[((w * 24 + n * 4 + 1) << 5) + l];
        float v2 = acc[0][n][2] + red[((w * 24 + n * 4 + 2) << 5) + l], v3 = acc[0][n][3] + red[((w * 24 + n * 4 + 3) << 5) + l];
        a.y[(size_t)tok * a.N + col0] = __float2bfloat16_rn(v0);
        a.y[(size_t)(tok + 1) * a.N + col0] = __float2bfloat16_rn(v1);
        a.y[(size_t)tok * a.N + col0 + 8] = __float2bfloat16_rn(v2);
        a.y[(size_t)(tok + 1) * a.N + col0 + 8] = __float2bfloat16_rn(v3);
      }
    }
    return;
  }
  // gate_up mode: E tile reduced over the 8 warps (fixed order) then split-K exchange with the partner CTA
  #pragma unroll
  for (int n = 0; n < 6; n++)
    #pragma unroll
    for (int j = 0; j < 4; j++) red[((w * 24 + n * 4 + j) << 5) + l] = accE[n][j];
  __syncthreads();
  float sumE[3];
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    int e = tid + 256 * i; float v = red[e];
    #pragma unroll
    for (int ww = 1; ww < 8; ww++) v += red[ww * 768 + e];
    sumE[i] = v;
  }
  const int kh = ks;
  float* mypart = a.part + ((size_t)gid * 2 + kh) * 13056;
  const float* opart = a.part + ((size_t)gid * 2 + (1 - kh)) * 13056;
  const bool ownAB = ((w & 1) == kh);     // tiles w, w+8 owned by CTA with ks == w&1
  if (!ownAB) {
    #pragma unroll
    for (int s = 0; s < 2; s++)
      #pragma unroll
      for (int n = 0; n < 6; n++)
        #pragma unroll
        for (int j = 0; j < 4; j++) __stcg(&mypart[((w * 2 + s) * 24 + n * 4 + j) * 32 + l], acc[s][n][j]);
  }
  if (kh == 1) {
    #pragma unroll
    for (int i = 0; i < 3; i++) __stcg(&mypart[12288 + tid + 256 * i], sumE[i]);
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;\n" :: "l"(a.flags + gid * 2 + kh), "r"(1) : "memory");
    int v = 0, it = 0;
    do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(a.flags + gid * 2 + (1 - kh)) : "memory"); } while (v == 0 && ++it < (1 << 22));
  }
  __syncthreads();
  if (ownAB) {
    #pragma unroll
    for (int s = 0; s < 2; s++) {
      const int col0 = colof(tl[s], l >> 2);
      #pragma unroll
      for (int n = 0; n < 6; n++) {
        const int tok = n * 8 + 2 * (l & 3);
        float v[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) v[j] = acc[s][n][j] + __ldcg(&opart[((w * 2 + s) * 24 + n * 4 + j) * 32 + l]);
        a.y[(size_t)tok * a.N + col0] = __float2bfloat16_rn(v[0]);
        a.y[(size_t)(tok + 1) * a.N + col0] = __float2bfloat16_rn(v[1]);
        a.y[(size_t)tok * a.N + col0 + 8] = __float2bfloat16_rn(v[2]);
        a.y[(size_t)(tok + 1) * a.N + col0 + 8] = __float2bfloat16_rn(v[3]);
      }
    }
  }
  if (kh == 0) {
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      int e = tid + 256 * i; float v = sumE[i] + __ldcg(&opart[12288 + e]);
      int n = e >> 7, j = (e >> 5) & 3, ln = e & 31;
      int tok = n * 8 + 2 * (ln & 3) + (j & 1), col = colof(t0 + 16, ln >> 2) + 8 * (j >> 1);
      a.y[(size_t)tok * a.N + col] = __float2bfloat16_rn(v);
    }
  }
  __syncthreads();
  if (tid == 0) a.flags[gid * 2 + (1 - kh)] = 0;
}
template <int S, int NP, int HASE, int ABL>
static void launch(torch::Tensor x, torch::Tensor W, torch::Tensor SC, torch::Tensor y, torch::Tensor part, torch::Tensor flags, int tiles, int TG, int KSP) {
  int N = y.size(1), K = x.size(1);
  Args a{(const __nv_bfloat16*)x.data_ptr(), (const unsigned char*)W.data_ptr(), (const uint32_t*)SC.data_ptr(), (__nv_bfloat16*)y.data_ptr(),
         (float*)part.data_ptr(), (int*)flags.data_ptr(), N, K, tiles, TG, KSP, K / 256 / KSP, K / 32, K / 128};
  size_t smem = 2 * TOKB;
  auto f = swab3<S, NP, HASE, ABL>;
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  f<<<TG * KSP, 256, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
}
void swab_gemm(torch::Tensor x, torch::Tensor W, torch::Tensor SC, torch::Tensor y, torch::Tensor part, torch::Tensor flags, int64_t tiles, int64_t TG, int64_t KSP, int64_t abl) {
  if (KSP == 2) {
    TORCH_CHECK(tiles == TG * 17);
    if (abl == 0) launch<2, 8, 1, 0>(x, W, SC, y, part, flags, tiles, TG, KSP);
    else if (abl == 1) launch<2, 8, 1, 1>(x, W, SC, y, part, flags, tiles, TG, KSP);
    else launch<2, 8, 1, 2>(x, W, SC, y, part, flags, tiles, TG, KSP);
  } else {
    TORCH_CHECK(tiles == TG * 4);
    if (abl == 0) launch<1, 4, 0, 0>(x, W, SC, y, part, flags, tiles, TG, KSP);
    else if (abl == 1) launch<1, 4, 0, 1>(x, W, SC, y, part, flags, tiles, TG, KSP);
    else launch<1, 4, 0, 2>(x, W, SC, y, part, flags, tiles, TG, KSP);
  }
}
__global__ void spin_kernel(long ns) { unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0)); unsigned long long t; do { __nanosleep(200); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while ((long)(t - t0) < ns); }
void spin(int64_t ns) { spin_kernel<<<1, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(ns); }
__global__ void pf_kernel(const unsigned char* p, long tbytes, long total_lines, long lines_per_tile) {
  long nt = (long)gridDim.x * blockDim.x;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total_lines; i += nt) {
    long tile = i / lines_per_tile, off = (i % lines_per_tile) * 128;
    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + tile * tbytes + off));
  }
}
void l2pf_tiles(torch::Tensor W, int64_t ntiles, int64_t P) {
  long tbytes = W.numel() / ntiles, per = std::min<long>(tbytes, (P / ntiles) / 128 * 128);
  if (per <= 0) return;
  long lpt = per / 128;
  pf_kernel<<<512, 512, 0, c10::cuda::getCurrentCUDAStream()>>>((const unsigned char*)W.data_ptr(), tbytes, lpt * ntiles, lpt);
}
__global__ void l2pf_kernel(const uint4* p, long vec, long step) {
  long nt = (long)gridDim.x * blockDim.x;
  for (long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * step; i < vec; i += nt * step)
    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p + i));
}
void l2pf(torch::Tensor w, int64_t bytes) {
  bytes = std::min<int64_t>(bytes, w.numel() * w.element_size());
  l2pf_kernel<<<512, 512, 0, c10::cuda::getCurrentCUDAStream()>>>((const uint4*)w.data_ptr(), bytes / 16, 2);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &swab_gemm); m.def("spin", &spin); m.def("l2pf_tiles", &l2pf_tiles); m.def("l2pf", &l2pf); }
