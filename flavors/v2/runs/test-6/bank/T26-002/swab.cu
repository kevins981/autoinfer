// W1-2: swap-AB register-direct-weights FP8 decode GEMM, M=48 tokens. A=weights (m16 = 16 features x k16) from registers (LDG.128),
// B=tokens via smem+ldmatrix (cp.async double buffer, 1 barrier / 256-K chunk). Reads Marlin FP8 repacked weights (re-permuted offline) + Marlin scales.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#define SROW 264                 // bf16 per token row in smem chunk (256 + 8 pad)
#define TOKB (48 * SROW * 2)     // bytes per token chunk buffer
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
// identical to Marlin / g2w.cu dequant of 4 fp8 -> (b0,b1) bf16x2 each, scale multiply included
__device__ __forceinline__ void deq(uint32_t q, uint32_t s2u, uint32_t& r0, uint32_t& r1) {
  uint32_t o1 = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 4);
  q <<= 8;
  uint32_t o2 = (q & 0x80008000u) | ((q & 0x7F007F00u) >> 4);
  __nv_bfloat162 s2 = *(__nv_bfloat162*)&s2u;
  __nv_bfloat162 b1 = __hmul2(*(__nv_bfloat162*)&o1, s2), b0 = __hmul2(*(__nv_bfloat162*)&o2, s2);
  r0 = *(uint32_t*)&b0; r1 = *(uint32_t*)&b1;
}
struct Args {
  const __nv_bfloat16* x; const unsigned char* W; const uint32_t* SC; __nv_bfloat16* y; float* wsy; int* flags;
  int N, K, tiles, TG, KSP, cps, KP, G;
};
// warp roles. T = tiles per warp (2 pair-slot, 1 single, 0 idle)
template <int T, int ABL>
__device__ __forceinline__ void run(const Args& a, unsigned char* smem, int tid, int l, int kq, int tl0, int ks, int gid, int tpos_pairbase,
                                    int redoff, int nthreads) {
  const int cps = a.cps, chunk0 = ks * a.cps;
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  auto issue = [&](int c) {
    uint32_t dst = sbase + (c & 1) * TOKB;
    for (int i = tid; i < 48 * 32; i += nthreads) {
      int row = i >> 5, pc = i & 31;
      cp16(dst + (row * SROW + pc * 8) * 2, a.x + (size_t)row * a.K + (size_t)(chunk0 + c) * 256 + pc * 8);
    }
    cp_commit();
  };
  float acc[T > 0 ? T : 1][6][4];
  #pragma unroll
  for (int t = 0; t < (T > 0 ? T : 1); t++)
    #pragma unroll
    for (int n = 0; n < 6; n++)
      #pragma unroll
      for (int j = 0; j < 4; j++) acc[t][n][j] = 0.f;
  const unsigned char* wp[T > 0 ? T : 1];
  const uint32_t* sp[T > 0 ? T : 1];
  uint4 wn[T > 0 ? T : 1], wc[T > 0 ? T : 1];
  uint32_t scn[T > 0 ? T : 1], scc0[T > 0 ? T : 1], scc1[T > 0 ? T : 1];
  uint32_t sink = 0;
  #pragma unroll
  for (int t = 0; t < T; t++) {
    wp[t] = a.W + (size_t)(tl0 + t) * a.KP * 512 + l * 16;
    sp[t] = a.SC + (size_t)(tl0 + t) * a.G * 8 + (l >> 2);
    if (ABL == 1) { wn[t] = make_uint4(tid * 7 + t, tid * 11, tid * 13 + 5, tid * 17 + t); scn[t] = 0x7B807B80u; }
    else {
      wn[t] = ldw(wp[t] + (size_t)((chunk0) * 8 + kq * 2) * 512);
      scn[t] = __ldg(sp[t] + (size_t)((chunk0) * 2 + (kq >> 1)) * 8);
    }
  }
  const uint32_t lbase = sbase + (((l & 7) + ((l >> 4) << 3)) * SROW + ((l >> 3) & 1) * 8) * 2;
  issue(0);
  for (int c = 0; c < cps; c++) {
    cp_wait0();
    __syncthreads();
    if (c + 1 < cps) issue(c + 1); else cp_commit();
    if (T > 0) {
      const uint32_t tb = lbase + (c & 1) * TOKB;
      #pragma unroll
      for (int j = 0; j < 2; j++) {
        #pragma unroll
        for (int t = 0; t < T; t++) {
          wc[t] = wn[t];
          if (j == 0) {
            uint32_t s = scn[t];
            scc0[t] = (s & 0xffffu) * 0x10001u; scc1[t] = (s >> 16) * 0x10001u;
          }
          if (ABL == 1) { wn[t].x += 0x01010101u; wn[t].y ^= c; wn[t].z += j; wn[t].w ^= wn[t].x; }
          else {
            if (j == 0) {
              wn[t] = ldw(wp[t] + (size_t)((chunk0 + c) * 8 + kq * 2 + 1) * 512);
              if (c + 1 < cps) scn[t] = __ldg(sp[t] + (size_t)((chunk0 + c + 1) * 2 + (kq >> 1)) * 8);
            } else if (c + 1 < cps) wn[t] = ldw(wp[t] + (size_t)((chunk0 + c + 1) * 8 + kq * 2) * 512);
          }
        }
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
          if (ABL == 2) {
            #pragma unroll
            for (int t = 0; t < T; t++) sink ^= wc[t].x ^ wc[t].y ^ wc[t].z ^ wc[t].w;
            break;
          }
          const int ksl = kq * 4 + j * 2 + sub;
          uint32_t b[6][2];
          #pragma unroll
          for (int p = 0; p < 3; p++) ldsm4(b[2 * p][0], b[2 * p][1], b[2 * p + 1][0], b[2 * p + 1][1], tb + ksl * 32 + p * 16 * SROW * 2);
          #pragma unroll
          for (int t = 0; t < T; t++) {
            uint32_t x = sub ? wc[t].z : wc[t].x, y = sub ? wc[t].w : wc[t].y;
            uint32_t a0, a1, a2, a3;
            deq(x, scc0[t], a0, a2);
            deq(y, scc1[t], a1, a3);
            #pragma unroll
            for (int n = 0; n < 6; n++) mma16816(acc[t][n], a0, a1, a2, a3, b[n][0], b[n][1]);
          }
        }
      }
    }
  }
  if (ABL == 2 && T > 0) acc[0][0][0] += (float)(sink & 0xff);
  // ---- reduce the 4 kq warps of each tile slot through smem (fixed order), then split-K partner
  float* red = (float*)smem;
  __syncthreads();   // token buffers no longer needed
  if (T > 0 && kq > 0) {
    #pragma unroll
    for (int t = 0; t < T; t++)
      #pragma unroll
      for (int n = 0; n < 6; n++)
        #pragma unroll
        for (int j = 0; j < 4; j++) red[redoff + (((t * 6 + n) * 4 + j) << 5) + l] = acc[t][n][j];
  }
  __syncthreads();
  if (T > 0 && kq == 0) {
    const int pairbase = tpos_pairbase; // float offset of the red region of this slot (kq=1 first)
    #pragma unroll
    for (int kk = 0; kk < 3; kk++)
      #pragma unroll
      for (int t = 0; t < T; t++)
        #pragma unroll
        for (int n = 0; n < 6; n++)
          #pragma unroll
          for (int j = 0; j < 4; j++) acc[t][n][j] += red[pairbase + kk * (T * 768) + (((t * 6 + n) * 4 + j) << 5) + l];
  }
  // split-K
  if (a.KSP == 2) {
    if (ks == 1) {
      if (T > 0 && kq == 0) {
        #pragma unroll
        for (int t = 0; t < T; t++) {
          int tl = tl0 + t, col0 = (tl >> 2) * 64 + (tl & 3) * 16 + (l >> 2);
          #pragma unroll
          for (int n = 0; n < 6; n++) {
            int tok = n * 8 + 2 * (l & 3);
            __stcg(&a.wsy[(size_t)tok * a.N + col0], acc[t][n][0]);
            __stcg(&a.wsy[(size_t)(tok + 1) * a.N + col0], acc[t][n][1]);
            __stcg(&a.wsy[(size_t)tok * a.N + col0 + 8], acc[t][n][2]);
            __stcg(&a.wsy[(size_t)(tok + 1) * a.N + col0 + 8], acc[t][n][3]);
          }
        }
      }
      __threadfence();
      __syncthreads();
      if (tid == 0) asm volatile("st.release.gpu.global.u32 [%0], %1;\n" :: "l"(a.flags + gid), "r"(1) : "memory");
      return;
    } else {
      if (tid == 0) { int v; do { asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(a.flags + gid) : "memory"); } while (v == 0); }
      __syncthreads();
      if (T > 0 && kq == 0) {
        #pragma unroll
        for (int t = 0; t < T; t++) {
          int tl = tl0 + t, col0 = (tl >> 2) * 64 + (tl & 3) * 16 + (l >> 2);
          #pragma unroll
          for (int n = 0; n < 6; n++) {
            int tok = n * 8 + 2 * (l & 3);
            acc[t][n][0] += __ldcg(&a.wsy[(size_t)tok * a.N + col0]);
            acc[t][n][1] += __ldcg(&a.wsy[(size_t)(tok + 1) * a.N + col0]);
            acc[t][n][2] += __ldcg(&a.wsy[(size_t)tok * a.N + col0 + 8]);
            acc[t][n][3] += __ldcg(&a.wsy[(size_t)(tok + 1) * a.N + col0 + 8]);
          }
        }
      }
      __syncthreads();
      if (tid == 0) a.flags[gid] = 0;
    }
  }
  if (T > 0 && kq == 0) {
    #pragma unroll
    for (int t = 0; t < T; t++) {
      int tl = tl0 + t, col0 = (tl >> 2) * 64 + (tl & 3) * 16 + (l >> 2);
      #pragma unroll
      for (int n = 0; n < 6; n++) {
        int tok = n * 8 + 2 * (l & 3);
        a.y[(size_t)tok * a.N + col0] = __float2bfloat16_rn(acc[t][n][0]);
        a.y[(size_t)(tok + 1) * a.N + col0] = __float2bfloat16_rn(acc[t][n][1]);
        a.y[(size_t)tok * a.N + col0 + 8] = __float2bfloat16_rn(acc[t][n][2]);
        a.y[(size_t)(tok + 1) * a.N + col0 + 8] = __float2bfloat16_rn(acc[t][n][3]);
      }
    }
  }
}
template <int NW, int ABL>
__global__ void __launch_bounds__(NW * 32, 1) swab_kernel(Args a) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, l = tid & 31, warp = tid >> 5, bid = blockIdx.x;
  const int ks = a.KSP - 1 - bid / a.TG, gid = bid % a.TG;
  const int base = a.tiles / a.TG, rem = a.tiles % a.TG;
  const int nt = base + (gid < rem ? 1 : 0), t0 = gid * base + (gid < rem ? gid : rem);
  const int npairs = nt >> 1, single = nt & 1, nslots = npairs + single;
  const int slot = warp >> 2, kq = warp & 3;
  const int nth = NW * 32;
  if (slot < npairs) {
    int redoff = (slot * 3 + kq - 1) * 1536;       // valid for kq>0
    run<2, ABL>(a, smem, tid, l, kq, t0 + 2 * slot, ks, gid, slot * 3 * 1536, redoff, nth);
  } else if (slot < nslots) {
    int redoff = npairs * 3 * 1536 + (kq - 1) * 768;
    run<1, ABL>(a, smem, tid, l, kq, t0 + 2 * npairs, ks, gid, npairs * 3 * 1536, redoff, nth);
  } else {
    run<0, ABL>(a, smem, tid, l, kq, 0, ks, gid, 0, 0, nth);
  }
}
template <int NW, int ABL>
static void launch(torch::Tensor x, torch::Tensor W, torch::Tensor SC, torch::Tensor y, torch::Tensor wsy, torch::Tensor flags, int tiles, int TG, int KSP) {
  int N = y.size(1), K = x.size(1);
  Args a{(const __nv_bfloat16*)x.data_ptr(), (const unsigned char*)W.data_ptr(), (const uint32_t*)SC.data_ptr(), (__nv_bfloat16*)y.data_ptr(),
         (float*)wsy.data_ptr(), (int*)flags.data_ptr(), N, K, tiles, TG, KSP, K / 256 / KSP, K / 32, K / 128};
  int nt = (tiles + TG - 1) / TG, np = nt >> 1, sg = nt & 1;
  size_t red = 4 * ((size_t)np * 3 * 1536 + (size_t)sg * 3 * 768);
  size_t smem = std::max<size_t>(2 * TOKB, red);
  TORCH_CHECK(NW == 4 * (np + sg) && smem <= 101376, "bad cfg");
  auto f = swab_kernel<NW, ABL>;
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  f<<<TG * KSP, NW * 32, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
}
void swab_gemm(torch::Tensor x, torch::Tensor W, torch::Tensor SC, torch::Tensor y, torch::Tensor wsy, torch::Tensor flags, int64_t tiles, int64_t TG, int64_t KSP, int64_t abl) {
  int nt = (tiles + TG - 1) / TG, nw = 4 * ((nt >> 1) + (nt & 1));
#define L(NW) if (nw == NW) { if (abl == 0) launch<NW, 0>(x, W, SC, y, wsy, flags, tiles, TG, KSP); else if (abl == 1) launch<NW, 1>(x, W, SC, y, wsy, flags, tiles, TG, KSP); else launch<NW, 2>(x, W, SC, y, wsy, flags, tiles, TG, KSP); return; }
  L(8) L(12) L(16) L(20)
  TORCH_CHECK(false, "unsupported warps");
}
__global__ void spin_kernel(long ns) { unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0)); unsigned long long t; do { __nanosleep(200); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while ((long)(t - t0) < ns); }
void spin(int64_t ns) { spin_kernel<<<1, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(ns); }
// prefetch: `ntiles` regions of `tbytes`, first `per` bytes of each region (early-K of every tile; == Marlin's first-P-MB for its k-major layout)
__global__ void pf_kernel(const unsigned char* p, long tbytes, long per, long total_lines, long lines_per_tile) {
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
  pf_kernel<<<512, 512, 0, c10::cuda::getCurrentCUDAStream()>>>((const unsigned char*)W.data_ptr(), tbytes, per, lpt * ntiles, lpt);
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
