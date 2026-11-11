// W1-2: real producer/consumer handoff. 128 CTAs, CTA c owns cols [136c,136c+136), full K (40 K128-blocks).
// producers (NSW warps): cp.async (B stage 17408 B + A stage 6144 B) -> smem ring of S stages; consumers (NW-NSW warps) = 3 mt x G nt-groups.
// sync: mbarrier (MB=1) or named barriers (MB=0). Packed weight layout: [cta][kb][136 rows x 128 B swizzled], A: [kb][48 rows x 128 B swizzled].
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

constexpr int M = 48, N = 17408, K = 5120, BL = 40, CN = 136, NTT = 17, NCTA = 128;
constexpr int BSTAGE = CN * 128, ASTAGE = M * 128, STAGE = BSTAGE + ASTAGE, NCH = STAGE / 16;  // 17408, 6144, 23552, 1472
constexpr int BCH = BSTAGE / 16;  // 1088

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int Nn> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(Nn)); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ uint32_t lds32(uint32_t addr) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(addr)); return v; }
__device__ __forceinline__ void mma_fp8(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mbar_init(uint32_t a, int c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(c) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t par) {
  uint32_t r;
  do {
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }\n" : "=r"(r) : "r"(a), "r"(par) : "memory");
  } while (!r);
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arr(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }

enum { F_NOMMA = 1, F_LOCAL = 2, F_NOWAIT = 4 };

template <int NW, int NSW, int S, bool MB>
__global__ void __launch_bounds__(NW * 32, 1)
k2(const uint8_t* __restrict__ Wp, const uint8_t* __restrict__ Ap, const float* __restrict__ Xs, const float* __restrict__ Ws,
   __nv_bfloat16* __restrict__ C, int nkb, int flags) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ __align__(8) uint64_t mb_full[S], mb_empty[S];
  constexpr int NM = NW - NSW, G = NM / 3, NTMAX = (NTT + G - 1) / G, NB = NW * 32;
  static_assert(NM % 3 == 0, "mma warps must be 3*G");
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, cta = blockIdx.x;
  if constexpr (MB) {
    if (tid == 0) {
      for (int s = 0; s < S; ++s) { mbar_init(smem_u32(&mb_full[s]), NSW * 32); mbar_init(smem_u32(&mb_empty[s]), NM * 32); }
    }
  }
  __syncthreads();
  const uint32_t sbase = smem_u32(smem);
  if (wid < NSW) {
    if (flags & F_NOWAIT) return;
    // ---------------- producers ----------------
    const int pt = tid;
    constexpr int NPT = NSW * 32;
    auto issue = [&](int kb) {
      const int s = kb % S;
      const int kbs = (flags & F_LOCAL) ? (kb & 3) : kb;
      const uint8_t* wsrc = Wp + ((size_t)cta * BL + kbs) * BSTAGE;
      const uint8_t* asrc = Ap + (size_t)kbs * ASTAGE;
      const uint32_t dst = sbase + s * STAGE;
#pragma unroll 4
      for (int i = pt; i < NCH; i += NPT) {
        const uint8_t* src = (i < BCH) ? (wsrc + i * 16) : (asrc + (i - BCH) * 16);
        cp_async16(dst + i * 16, src);
      }
    };
#pragma unroll
    for (int s = 0; s < S - 1; ++s) { if (s < nkb) issue(s); cp_commit(); }
    for (int kb = 0; kb < nkb; ++kb) {
      cp_wait<S - 2>();
      const int s = kb % S;
      if constexpr (MB) mbar_arrive(smem_u32(&mb_full[s])); else bar_arr(1 + s, NB);
      const int nx = kb + S - 1;
      if (nx < nkb) {
        if (kb >= 1) {
          const int ps = (kb - 1) % S, u = (kb - 1) / S;
          if constexpr (MB) mbar_wait(smem_u32(&mb_empty[ps]), u & 1); else bar_sync(1 + S + ps, NB);
        }
        issue(nx);
      }
      cp_commit();
    }
    cp_wait<0>();
    return;
  }
  // ---------------- consumers ----------------
  const int mw = wid - NSW, mt = mw / G, g = mw % G;
  const int nt0 = (g * NTT) / G, nt1 = ((g + 1) * NTT) / G, ntw = nt1 - nt0;
  float acc[NTMAX][4], tmp[NTMAX][4];
#pragma unroll
  for (int a = 0; a < NTMAX; ++a)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[a][e] = tmp[a][e] = 0.f;
  const int r0 = mt * 16 + (lane >> 2), r1 = r0 + 8;
  const int a_r = lane & 15, a_c = lane >> 4;
  const float* xs0 = Xs + (size_t)r0 * BL;
  const float* xs1 = Xs + (size_t)r1 * BL;
  const float* wsp[NTMAX];
#pragma unroll
  for (int a = 0; a < NTMAX; ++a) wsp[a] = Ws + (size_t)(((cta * CN + (nt0 + a) * 8) >> 7)) * BL;
  float sa0n = xs0[0], sa1n = xs1[0], sbn[NTMAX];
#pragma unroll
  for (int a = 0; a < NTMAX; ++a) sbn[a] = (a < ntw) ? wsp[a][0] : 0.f;
  for (int kb = 0; kb < nkb; ++kb) {
    const float sa0 = sa0n, sa1 = sa1n;
    float sb[NTMAX];
#pragma unroll
    for (int a = 0; a < NTMAX; ++a) sb[a] = sbn[a];
    if (kb + 1 < nkb) {
      sa0n = xs0[kb + 1]; sa1n = xs1[kb + 1];
#pragma unroll
      for (int a = 0; a < NTMAX; ++a) sbn[a] = (a < ntw) ? wsp[a][kb + 1] : 0.f;
    }
    const int s = kb % S, u = kb / S;
    if (!(flags & F_NOWAIT)) { if constexpr (MB) mbar_wait(smem_u32(&mb_full[s]), u & 1); else bar_sync(1 + s, NB); }
    if (!(flags & F_NOMMA)) {
      const uint32_t st = sbase + s * STAGE, ab = st + BSTAGE;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t af[4];
        {
          const int r = mt * 16 + a_r, c = j * 2 + a_c;
          ldsm_x4(ab + r * 128 + ((c ^ (r & 7)) << 4), af[0], af[1], af[2], af[3]);
        }
#ifdef BLDM
#pragma unroll
        for (int a = 0; a < NTMAX; a += 2) {
          if (a < ntw) {
            const int i4 = lane >> 3;
            int tl = a + (i4 >> 1); if (tl >= ntw) tl = a;
            const int row = (nt0 + tl) * 8 + (lane & 7);
            uint32_t q0, q1, q2, q3;
            ldsm_x4(st + row * 128 + ((((2 * j) + (i4 & 1)) ^ (lane & 7)) << 4), q0, q1, q2, q3);
            mma_fp8(tmp[a], af, q0, q1);
            if (a + 1 < NTMAX) { if (a + 1 < ntw) mma_fp8(tmp[a + 1], af, q2, q3); }
          }
        }
#else
#pragma unroll
        for (int a = 0; a < NTMAX; ++a) {
          if (a < ntw) {
            const int row = (nt0 + a) * 8 + (lane >> 2), sw = lane >> 2;
            const uint32_t rb = st + row * 128 + (lane & 3) * 4;
            const uint32_t b0 = lds32(rb + (((2 * j) ^ sw) << 4));
            const uint32_t b1 = lds32(rb + (((2 * j + 1) ^ sw) << 4));
            mma_fp8(tmp[a], af, b0, b1);
          }
        }
#endif
      }
    }
    if (!(flags & F_NOWAIT) && kb + S < nkb) { if constexpr (MB) mbar_arrive(smem_u32(&mb_empty[s])); else bar_arr(1 + S + s, NB); }
    if (!(flags & F_NOMMA)) {
#pragma unroll
      for (int a = 0; a < NTMAX; ++a) {
        const float s0 = sa0 * sb[a], s1 = sa1 * sb[a];
        acc[a][0] = fmaf(tmp[a][0], s0, acc[a][0]);
        acc[a][1] = fmaf(tmp[a][1], s0, acc[a][1]);
        acc[a][2] = fmaf(tmp[a][2], s1, acc[a][2]);
        acc[a][3] = fmaf(tmp[a][3], s1, acc[a][3]);
        tmp[a][0] = tmp[a][1] = tmp[a][2] = tmp[a][3] = 0.f;
      }
    }
  }
  // epilogue: bf16 C[48][N]
#pragma unroll
  for (int a = 0; a < NTMAX; ++a) {
    if (a < ntw) {
      const int col = cta * CN + (nt0 + a) * 8 + (lane & 3) * 2;
      *reinterpret_cast<__nv_bfloat162*>(C + (size_t)r0 * N + col) = __floats2bfloat162_rn(acc[a][0], acc[a][1]);
      *reinterpret_cast<__nv_bfloat162*>(C + (size_t)r1 * N + col) = __floats2bfloat162_rn(acc[a][2], acc[a][3]);
    }
  }
}

__global__ void read_kernel(const uint4* __restrict__ p, size_t n16, unsigned* out, int evict_last) {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
  unsigned x = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v;
    if (evict_last) asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i), "l"(pol));
    else asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (x == 0x12345678u) out[0] = x;
}
// per-CTA prefetch of the first `bytes` of every CTA's contiguous packed range (stream order), evict_last
__global__ void pf_kernel(const uint4* __restrict__ p, size_t cta_bytes, size_t bytes, unsigned* out) {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
  const int c = blockIdx.x >> 2, part = blockIdx.x & 3;
  const uint4* base = p + (size_t)c * (cta_bytes / 16);
  const size_t n16 = bytes / 16;
  unsigned x = 0;
  for (size_t i = (size_t)part * 512 + threadIdx.x; i < n16; i += 4 * 512) {
    uint4 v;
    asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(base + i), "l"(pol));
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (x == 0x12345678u) out[0] = x;
}
__global__ void empty_kernel() {}

static uint8_t *d_Wp, *d_Ap; static float *d_Xs, *d_Ws; static __nv_bfloat16* d_C; static uint4* d_flush; static unsigned* d_sink;
static const size_t FLUSH = 384ull << 20;
static const size_t CTAB = (size_t)BL * BSTAGE;       // 696,320 B per CTA
static const size_t PFB = (48ull << 20) / NCTA;       // 393,216 B per CTA (= first 56.5% of each CTA's stream; 48 MiB total)

template <int NW, int NSW, int S, bool MB>
struct R2 {
  static size_t smem() { return (size_t)S * STAGE; }
  static void init() { CK(cudaFuncSetAttribute(k2<NW, NSW, S, MB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem())); }
  static void go(int nkb, int flags, cudaStream_t st) { k2<NW, NSW, S, MB><<<NCTA, NW * 32, smem(), st>>>(d_Wp, d_Ap, d_Xs, d_Ws, d_C, nkb, flags); }
};

template <class F>
static float med_time(F&& f, int pre, int reps) {
  cudaStream_t st = 0;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> t;
  for (int r = 0; r < reps + 2; ++r) {
    CK(cudaCtxResetPersistingL2Cache());
    read_kernel<<<NCTA * 8, 512, 0, st>>>(d_flush, FLUSH / 16, d_sink, 0);
    if (pre) pf_kernel<<<NCTA * 4, 512, 0, st>>>((const uint4*)d_Wp, CTAB, PFB, d_sink);
    CK(cudaEventRecord(e0, st));
    f(st);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    if (r >= 2) t.push_back(ms * 1000.f);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

static int host_main_check = 0;
template <int NW, int NSW, int S, bool MB>
static void run_cfg(const char* name, int reps, bool check, const char* dir) {
  using R = R2<NW, NSW, S, MB>;
  R::init();
  int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k2<NW, NSW, S, MB>, NW * 32, R::smem()));
  cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k2<NW, NSW, S, MB>));
  if (check) {
    CK(cudaMemset(d_C, 0, (size_t)M * N * 2));
    R::go(BL, 0, 0); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::vector<uint16_t> h((size_t)M * N); CK(cudaMemcpy(h.data(), d_C, h.size() * 2, cudaMemcpyDeviceToHost));
    char fn[512]; snprintf(fn, sizeof fn, "%s/out_%s.bin", dir, name);
    FILE* f = fopen(fn, "wb"); fwrite(h.data(), 2, h.size(), f); fclose(f);
  }
  const float tcold = med_time([&](cudaStream_t s) { R::go(BL, 0, s); }, 0, reps);
  const float t883 = med_time([&](cudaStream_t s) { R::go(35, 0, s); }, 0, reps);
  const float tpf = med_time([&](cudaStream_t s) { R::go(BL, 0, s); }, 1, reps);
  const float tnm = med_time([&](cudaStream_t s) { R::go(BL, F_NOMMA, s); }, 0, reps);       // handshake only, HBM stream
  const float tnm883 = med_time([&](cudaStream_t s) { R::go(35, F_NOMMA, s); }, 0, reps);
  const float tnmpf = med_time([&](cudaStream_t s) { R::go(BL, F_NOMMA, s); }, 1, reps);
  const float tloc = med_time([&](cudaStream_t s) { R::go(BL, F_LOCAL, s); }, 0, reps);      // weights from a 4-block L2-hot window: compute+handoff only
  const float tlocnm = med_time([&](cudaStream_t s) { R::go(BL, F_LOCAL | F_NOMMA, s); }, 0, reps);  // copy+handoff only, L2-hot
  const float tnw = med_time([&](cudaStream_t s) { R::go(BL, F_NOWAIT, s); }, 0, reps);           // consumers only, no producers/waits (lds from stale smem)
  printf("%-14s regs=%3d occ=%d smem=%zu | REAL cold %6.1f  0.883x %6.1f  pf48 %6.1f | stream-only(no MMA) cold %6.1f 0.883x %6.1f pf48 %6.1f | L2-local weights (MMA+handoff floor) %6.1f | L2-local copy-only %6.1f | consumers-only(no wait) %6.1f\n",
         name, fa.numRegs, occ, R::smem(), tcold, t883, tpf, tnm, tnm883, tnmpf, tloc, tlocnm, tnw);
  fflush(stdout);
}

static uint8_t rnd_fp8(uint32_t& st) {
  st = st * 1664525u + 1013904223u; uint8_t b = (st >> 16) & 0xff; if ((b & 0x7f) == 0x7f) b &= 0xbf; return b;
}

int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 21;
  int mode = argc > 2 ? atoi(argv[2]) : 0;   // 0 timing, 1 timing + write check files
  const char* dir = argc > 3 ? argv[3] : ".";
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device %s SMs %d\n", p.name, p.multiProcessorCount);
  // ---- host data
  std::vector<uint8_t> W((size_t)N * K), A((size_t)M * K);
  uint32_t st = 12345;
  for (auto& b : W) b = rnd_fp8(st);
  for (auto& b : A) b = rnd_fp8(st);
  std::vector<float> xs((size_t)M * BL), ws((size_t)(N / 128) * BL);
  for (auto& v : xs) { st = st * 1664525u + 1013904223u; v = 0.005f + (st >> 8) * (0.01f / 16777216.f); }
  for (auto& v : ws) { st = st * 1664525u + 1013904223u; v = 0.005f + (st >> 8) * (0.01f / 16777216.f); }
  std::vector<uint8_t> Wp((size_t)NCTA * CTAB), Ap((size_t)BL * ASTAGE);
  for (int c = 0; c < NCTA; ++c)
    for (int kb = 0; kb < BL; ++kb)
      for (int n = 0; n < CN; ++n)
        for (int ch = 0; ch < 8; ++ch)
          memcpy(&Wp[((size_t)c * BL + kb) * BSTAGE + n * 128 + ((ch ^ (n & 7)) << 4)], &W[(size_t)(c * CN + n) * K + kb * 128 + ch * 16], 16);
  for (int kb = 0; kb < BL; ++kb)
    for (int r = 0; r < M; ++r)
      for (int ch = 0; ch < 8; ++ch) memcpy(&Ap[(size_t)kb * ASTAGE + r * 128 + ((ch ^ (r & 7)) << 4)], &A[(size_t)r * K + kb * 128 + ch * 16], 16);
  if (mode == 1) {
    char fn[512];
    snprintf(fn, sizeof fn, "%s/W.bin", dir); FILE* f = fopen(fn, "wb"); fwrite(W.data(), 1, W.size(), f); fclose(f);
    snprintf(fn, sizeof fn, "%s/A.bin", dir); f = fopen(fn, "wb"); fwrite(A.data(), 1, A.size(), f); fclose(f);
    snprintf(fn, sizeof fn, "%s/xs.bin", dir); f = fopen(fn, "wb"); fwrite(xs.data(), 4, xs.size(), f); fclose(f);
    snprintf(fn, sizeof fn, "%s/ws.bin", dir); f = fopen(fn, "wb"); fwrite(ws.data(), 4, ws.size(), f); fclose(f);
  }
  CK(cudaMalloc(&d_Wp, Wp.size())); CK(cudaMalloc(&d_Ap, Ap.size())); CK(cudaMalloc(&d_Xs, xs.size() * 4)); CK(cudaMalloc(&d_Ws, ws.size() * 4));
  CK(cudaMalloc(&d_C, (size_t)M * N * 2)); CK(cudaMalloc(&d_flush, FLUSH)); CK(cudaMalloc(&d_sink, 64));
  CK(cudaMemcpy(d_Wp, Wp.data(), Wp.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_Ap, Ap.data(), Ap.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_Xs, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_Ws, ws.data(), ws.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(d_flush, 0x22, FLUSH));
  { // empty bracket
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); std::vector<float> t;
    for (int r = 0; r < 25; ++r) { read_kernel<<<NCTA * 8, 512>>>(d_flush, FLUSH / 16, d_sink, 0); cudaEventRecord(e0); empty_kernel<<<NCTA, 256>>>(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); if (r > 2) t.push_back(ms * 1000); }
    std::sort(t.begin(), t.end()); printf("empty-kernel event bracket: median %.1f us\n", t[t.size() / 2]);
  }
  if (mode == 2) {  // drift test: alternate REAL / stream-only(nomma) / L2-local, print every rep (B16 S3 nb, cold)
    using R = R2<16, 4, 3, false>; R::init();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < reps; ++r) {
      float t[3];
      for (int v = 0; v < 3; ++v) {
        read_kernel<<<NCTA * 8, 512>>>(d_flush, FLUSH / 16, d_sink, 0);
        cudaEventRecord(e0); R::go(BL, v == 0 ? 0 : (v == 1 ? F_NOMMA : F_LOCAL), 0); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); t[v] = ms * 1000;
      }
      printf("rep %3d real %6.1f nomma %6.1f local %6.1f\n", r, t[0], t[1], t[2]);
    }
    return 0;
  }
  bool chk = mode == 1;
  int only = argc > 4 ? atoi(argv[4]) : -1;
#define RUN(NW, NSW, S, MB, NAME) if (only < 0 || only == __COUNTER__) run_cfg<NW, NSW, S, MB>(NAME, reps, chk, dir);
  RUN(16, 4, 2, true, "B16-4+12 S2 mb")  RUN(16, 4, 3, true, "B16-4+12 S3 mb")  RUN(16, 4, 4, true, "B16-4+12 S4 mb")
  RUN(16, 4, 2, false, "B16-4+12 S2 nb") RUN(16, 4, 3, false, "B16-4+12 S3 nb") RUN(16, 4, 4, false, "B16-4+12 S4 nb")
  RUN(8, 2, 2, true, "A8-2+6 S2 mb")     RUN(8, 2, 3, true, "A8-2+6 S3 mb")     RUN(8, 2, 4, true, "A8-2+6 S4 mb")
  RUN(8, 2, 2, false, "A8-2+6 S2 nb")    RUN(8, 2, 3, false, "A8-2+6 S3 nb")    RUN(8, 2, 4, false, "A8-2+6 S4 nb")
  return 0;
}
