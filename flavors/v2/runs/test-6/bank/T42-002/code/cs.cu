// T42 stage 1: certified lm_head argmax screen (standalone extension, leader-42.w1).
// Plane: q = round(w / s), s = max|w_g| / 63 (fp32) per (vocab row, group of 128 along K), q in [-64, 63] (7 bits),
// packed per (n8 tile, kc) group record of 928 B laid out as int8 m16n8k32 B fragments:
//   [0,512) nibble plane (uint4/lane) [512,768) 2-bit plane (uint2/lane) [768,896) 1-bit plane (u32/lane)
//   [896,912) 8 bf16 scales (rows n8*8 + 0..7), rounded up
// Lane l: B row n8*8 + (l>>2); reg i = 2j + half holds q[row][kc*128 + 32j + 16half + 4(l&3) + 0..3].
// Screen: h quantized per row h ~ s_h (128 a + b), a, b int8; int8 MMA (exact int32) per group, fp32 combine.
// Bound (Cauchy-Schwarz, per-row scalars): |h.w - approx| <= hn*A_v + hqe*B_v  (see record for proof).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define REC 912
#define KC_MAX 64

static __device__ __forceinline__ uint4 ldg_nc(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
static __device__ __forceinline__ uint2 ldg_nc2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
static __device__ __forceinline__ uint32_t ldg_nc1(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
static __device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(s), "l"(gmem), "r"(sz));
}
static __device__ __forceinline__ void ldsm_x4(const void* smem, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}
static __device__ __forceinline__ void mma_s8(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// ordered encoding of floats (monotone as unsigned); 0 = "below everything"
static __device__ __forceinline__ unsigned f2ord(float f) {
  unsigned u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static __device__ __forceinline__ float ord2f(unsigned k) {
  if (k == 0) return -INFINITY;
  unsigned u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return __uint_as_float(u);
}
static __device__ __forceinline__ float bf_ru(float x) { return __bfloat162float(__float2bfloat16_ru(x)); }
static __device__ __forceinline__ float bf_rd(float x) { return __bfloat162float(__float2bfloat16_rd(x)); }

// ---------------- plane decode ----------------
struct PRec { uint4 n; uint2 p1; uint32_t p2; float2 sc; };
static __device__ __forceinline__ void load_rec(const uint8_t* G, int lane, PRec& r) {
  r.n = ldg_nc(G + 16 * lane);
  r.p1 = ldg_nc2(G + 512 + 8 * lane);
  r.p2 = ldg_nc1(G + 768 + 4 * lane);
  const uint32_t s = ldg_nc1(G + 896 + 4 * (lane & 3));
  r.sc = make_float2(__uint_as_float(s << 16), __uint_as_float(s & 0xFFFF0000u));
}
static __device__ __forceinline__ void decode_rec(const PRec& r, uint32_t q[8]) {
  const uint32_t nw[4] = {r.n.x, r.n.y, r.n.z, r.n.w};
  const uint32_t pw[2] = {r.p1.x, r.p1.y};
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint32_t lo = (nw[i >> 1] >> (4 * (i & 1))) & 0x0F0F0F0Fu;
    uint32_t mid = ((pw[i >> 2] >> (2 * (i & 3))) & 0x03030303u) << 4;
    uint32_t hi = (r.p2 >> i) & 0x01010101u;
    q[i] = lo | mid | (hi << 6) | (hi << 7);
  }
}
// verification of the packing: decode every group and compare with the int8 q [N][K]
__global__ void plane_verify_kernel(const int8_t* __restrict__ Q, long N, long K, const uint8_t* __restrict__ plane,
                                    unsigned long long* bad) {
  const int lane = threadIdx.x & 31;
  const long KC = K / 128, ng = (N / 8) * KC;
  const long g = (long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (g >= ng) return;
  const long n8 = g / KC, kc = g % KC;
  PRec r; load_rec(plane + g * REC, lane, r);
  uint32_t q[8]; decode_rec(r, q);
  const long row = n8 * 8 + (lane >> 2); const int t = lane & 3;
  int nb = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int j = i >> 1, half = i & 1;
    const uint32_t ref = *reinterpret_cast<const uint32_t*>(Q + row * K + kc * 128 + 32 * j + 16 * half + 4 * t);
    nb += (ref != q[i]);
  }
  if (nb) atomicAdd(bad, (unsigned long long)nb);
}

// ---------------- h prep: one block per row ----------------
// hq [M][KC][2][128] int8 (a then b), hs = s_h, hn = ||h||_2 (ru), hqe = ||h - s_h qh||_2 (ru)
__global__ void hprep_kernel(const __nv_bfloat16* __restrict__ X, int K, int8_t* __restrict__ hq, float* hs, float* hn, float* hqe) {
  const int m = blockIdx.x, tid = threadIdx.x;
  const __nv_bfloat16* x = X + (long)m * K;
  __shared__ float smax[32];
  __shared__ double s2[32], e2[32];
  float mx = 0.f;
  for (int i = tid; i < K; i += blockDim.x) mx = fmaxf(mx, fabsf(__bfloat162float(x[i])));
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((tid & 31) == 0) smax[tid >> 5] = mx;
  __syncthreads();
  if (tid < 32) {
    float v = tid < (int)(blockDim.x / 32) ? smax[tid] : 0.f;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (tid == 0) smax[0] = v;
  }
  __syncthreads();
  mx = smax[0];
  const float s = mx > 0.f ? mx / 16256.f : 1.f;
  // per-thread fp32 partial sums of <= K/blockDim (<= 20) squares; error <= 21*2^-24 relative, inflated below
  float fh = 0.f, fe = 0.f;
  for (int i = tid; i < K; i += blockDim.x) {
    const float h = __bfloat162float(x[i]);
    int qh = __float2int_rn(h / s);
    qh = max(-16319, min(16319, qh));
    int a = __float2int_rn((float)qh * (1.f / 128.f));
    a = max(-127, min(127, a));
    const int b = qh - 128 * a;  // |b| <= 64 for |qh| <= 16319
    const int kc = i >> 7, kk = i & 127;
    hq[((long)m * (K / 128) + kc) * 256 + kk] = (int8_t)a;
    hq[((long)m * (K / 128) + kc) * 256 + 128 + kk] = (int8_t)b;
    const float d = fabsf(fmaf(-s, (float)qh, h)) * (1.f + 1.2e-7f);  // |rn(h - s qh)| (1 + u) >= |h - s qh|
    fh = fmaf(h, h, fh);
    fe = fmaf(d, d, fe);
  }
  double sh = (double)fh * (1.0 + 1e-5) + 1e-30, se = (double)fe * (1.0 + 1e-5) + 1e-30;
  for (int o = 16; o; o >>= 1) { sh += __shfl_xor_sync(0xffffffffu, sh, o); se += __shfl_xor_sync(0xffffffffu, se, o); }
  if ((tid & 31) == 0) { s2[tid >> 5] = sh; e2[tid >> 5] = se; }
  __syncthreads();
  if (tid == 0) {
    double a = 0, b = 0;
    for (int w = 0; w < (int)(blockDim.x / 32); w++) { a += s2[w]; b += e2[w]; }
    // fp64 sums of <= 5120 squares: relative error < 1e-12; inflate by 1e-9 then round up
    hs[m] = s;
    hn[m] = __double2float_ru(sqrt(a * (1.0 + 1e-9)) * (1.0 + 1e-9));
    hqe[m] = __double2float_ru(sqrt(b * (1.0 + 1e-9)) * (1.0 + 1e-9));
  }
}

// ---------------- screen ----------------
// lb: per-row ordered running max of L (atomicMax); emission: (id, U) per row where bf16_ru(U) >= bf16_rd(thr_now)
template <int MT, int NT, int WARPS, int STAGES, bool MAGIC>
__global__ void __launch_bounds__(WARPS * 32)
screen_kernel(const int8_t* __restrict__ hq, int M, long K, const uint8_t* __restrict__ plane,
              const float* __restrict__ vA, const float* __restrict__ vB,
              const float* __restrict__ hs, const float* __restrict__ hn, const float* __restrict__ hqe, long N,
              unsigned* lb, int* ecnt, int2* ebuf, int ecap) {
  extern __shared__ uint4 smem[];  // [STAGES][MT*16 rows][16 vec] swizzled (a: vec 0-7, b: vec 8-15)
  constexpr int ROWS = MT * 16, VEC_PER_STAGE = ROWS * 16;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, tid = threadIdx.x;
  const long KC = K / 128;
  const long n8_base = ((long)blockIdx.x * WARPS + warp) * NT;
  const bool warp_active = n8_base * 8 < N;

  auto load_x = [&](int kc, int stage) {
    uint4* s = smem + stage * VEC_PER_STAGE;
    for (int v = tid; v < VEC_PER_STAGE; v += WARPS * 32) {
      const int r = v >> 4, c = v & 15;
      const bool ok = r < M;
      const int8_t* src = hq + ((long)(ok ? r : 0) * KC + kc) * 256 + c * 16;
      cp_async16(s + r * 16 + (c ^ (r & 7)), src, ok);
    }
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KC) load_x(s, s);
    asm volatile("cp.async.commit_group;");
  }
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; i++)
#pragma unroll
    for (int j = 0; j < NT; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  PRec cur[NT], nxt[NT];
  if (warp_active) {
#pragma unroll
    for (int nt = 0; nt < NT; nt++) load_rec(plane + ((n8_base + nt) * KC + 0) * REC, lane, cur[nt]);
  }
  for (long kc = 0; kc < KC; kc++) {
    if (warp_active && kc + 1 < KC) {
#pragma unroll
      for (int nt = 0; nt < NT; nt++) load_rec(plane + ((n8_base + nt) * KC + kc + 1) * REC, lane, nxt[nt]);
    }
    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
    __syncthreads();
    {
      const long kn = kc + STAGES - 1;
      if (kn < KC) load_x(kn, kn % STAGES);
      asm volatile("cp.async.commit_group;");
    }
    if (warp_active) {
      uint32_t bq[NT][8];
#pragma unroll
      for (int nt = 0; nt < NT; nt++) decode_rec(cur[nt], bq[nt]);
      int ia[MT][NT][4], ib[MT][NT][4];
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
          for (int e = 0; e < 4; e++) ia[i][j][e] = ib[i][j][e] = 0;
      const uint4* s = smem + (kc % STAGES) * VEC_PER_STAGE;
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          uint32_t aa[4], ab[4];
          const int q = lane >> 3, r = mt * 16 + (q & 1) * 8 + (lane & 7), c = 2 * j + (q >> 1);
          ldsm_x4(s + r * 16 + (c ^ (r & 7)), aa[0], aa[1], aa[2], aa[3]);
          ldsm_x4(s + r * 16 + ((8 + c) ^ (r & 7)), ab[0], ab[1], ab[2], ab[3]);
#pragma unroll
          for (int nt = 0; nt < NT; nt++) {
            mma_s8(ia[mt][nt], aa, bq[nt][2 * j], bq[nt][2 * j + 1]);
            mma_s8(ib[mt][nt], ab, bq[nt][2 * j], bq[nt][2 * j + 1]);
          }
        }
      }
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
          for (int e = 0; e < 4; e++) {
            float fI;
            if (MAGIC) {  // |ia|, |ib| < 2^22: exact floats; fmaf(fa, 128, fb) = rn(128 ia + ib) = (float)I bitwise
              const float fa = __int_as_float(ia[mt][nt][e] + 0x4B400000) - 12582912.f;
              const float fb = __int_as_float(ib[mt][nt][e] + 0x4B400000) - 12582912.f;
              fI = fmaf(fa, 128.f, fb);
            } else {
              fI = (float)(ia[mt][nt][e] * 128 + ib[mt][nt][e]);  // exact int: |I| < 2^28
            }
            acc[mt][nt][e] = fmaf(fI, (e & 1) ? cur[nt].sc.y : cur[nt].sc.x, acc[mt][nt][e]);
          }
#pragma unroll
      for (int nt = 0; nt < NT; nt++) cur[nt] = nxt[nt];
    }
  }
  asm volatile("cp.async.wait_group 0;");
  if (!warp_active) return;
  const int g = lane >> 2, t = lane & 3;
  float U[MT][NT][4], L[MT][NT][4];
  float lmax[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; mt++) {
    lmax[mt][0] = lmax[mt][1] = -INFINITY;
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
      const int r = mt * 16 + g + 8 * hf;
      const bool rv = r < M;
      const float shr = rv ? hs[r] : 0.f, hnr = rv ? hn[r] : 0.f, hqr = rv ? hqe[r] : 0.f;
#pragma unroll
      for (int nt = 0; nt < NT; nt++)
#pragma unroll
        for (int e1 = 0; e1 < 2; e1++) {
          const int e = 2 * hf + e1;
          const long col = (n8_base + nt) * 8 + 2 * t + e1;
          const bool ok = rv && col < N;
          const float a = acc[mt][nt][e] * shr;
          const float bnd = ok ? __fmaf_ru(hnr, vA[col], __fmul_ru(hqr, vB[col])) : 0.f;
          U[mt][nt][e] = ok ? __fadd_ru(a, bnd) : -INFINITY;
          L[mt][nt][e] = ok ? __fsub_rd(a, bnd) : -INFINITY;
          lmax[mt][hf] = fmaxf(lmax[mt][hf], L[mt][nt][e]);
        }
    }
  }
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
      float v = lmax[mt][hf];
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
      const int r = mt * 16 + g + 8 * hf;
      unsigned thr = 0;
      if (t == 0 && r < M) {
        const unsigned mine = f2ord(v);
        const unsigned now = __ldcg(lb + r);
        if (mine > now) atomicMax(lb + r, mine);
        thr = max(mine, now);
      }
      thr = __shfl_sync(0xffffffffu, thr, lane & ~3);
      const float T = bf_rd(ord2f(thr));
      if (r < M) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
          for (int e1 = 0; e1 < 2; e1++) {
            const float u = U[mt][nt][2 * hf + e1];
            if (bf_ru(u) >= T) {
              const int pos = atomicAdd(ecnt + r, 1);
              if (pos < ecap) ebuf[(long)r * ecap + pos] = make_int2((int)((n8_base + nt) * 8 + 2 * t + e1), __float_as_int(u));
            }
          }
      }
    }
}

// ---------------- finalize: filter emitted entries against the final LB ----------------
// keeps (m, v) with bf16_ru(U) >= bf16_rd(max(lbA[m], lbB[m])); dedupe owner via map[v] (-1 = free)
__global__ void finalize_kernel(int M, const unsigned* lbA, const unsigned* lbB, const int* ecnt, const int2* ebuf, int ecap,
                                int* pair_cnt, int2* pairs, int* src, int paircap, int* map, int* kept, int* fallback, int capr) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const int n = ecnt[m];
  if (n > ecap) { if (threadIdx.x == 0) fallback[m] = 1; return; }
  const float T = bf_rd(ord2f(max(lbA[m], lbB[m])));
  __shared__ int nk;
  if (threadIdx.x == 0) nk = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int2 e = ebuf[(long)m * ecap + i];
    if (bf_ru(__int_as_float(e.y)) >= T) {
      const int k = atomicAdd(&nk, 1);
      if (k >= capr) continue;
      const int p = atomicAdd(pair_cnt, 1);
      if (p >= paircap) { fallback[m] = 1; continue; }
      pairs[p] = make_int2(m, e.x);
      const int own = atomicCAS(map + e.x, -1, p);
      src[p] = own < 0 ? p : own;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) { kept[m] = nk; if (nk > capr) fallback[m] = 1; }
}

// ---------------- gather (R-HOST): copy owner rows from (host-mapped) W into staging ----------------
__global__ void gather_kernel(const int* pair_cnt, int paircap, const int2* pairs, const int* src, int* map,
                              const uint4* W, long K, uint4* stage) {
  const int np = min(*pair_cnt, paircap);
  const int vpr = K / 8;  // uint4 per row
  const long total = (long)np * vpr;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int p = i / vpr, c = i % vpr;
    if (src[p] != p) continue;
    stage[(long)p * vpr + c] = W[(long)pairs[p].y * vpr + c];
  }
}
__global__ void unmap_kernel(const int* pair_cnt, int paircap, const int2* pairs, int* map) {
  const int np = min(*pair_cnt, paircap);
  for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < np; p += gridDim.x * blockDim.x) map[pairs[p].y] = -1;
}

// ---------------- rescore: exact bf16 logits with the champion's MMA chain, diagonal trick ----------------
// warp per octet of pairs; A row g = h_{m_g}, B col g = w_{v_g}; D[g][g] = same per-element MMA chain as lmh_codec.
// mode 0: row ptr = Wdev + v*K ; mode 1: row ptr = stage + src[p]*K
__global__ void rescore_kernel(const __nv_bfloat16* __restrict__ X, long K, const int* pair_cnt, int paircap,
                               const int2* __restrict__ pairs, const int* __restrict__ src, const __nv_bfloat16* __restrict__ base,
                               int mode, long voff, unsigned long long* keys, float* vals) {
  const int np = min(*pair_cnt, paircap);
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int nwarps = gridDim.x * (blockDim.x / 32);
  for (int o = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; o * 8 < np; o += nwarps) {
    const int p = o * 8 + g;
    const bool valid = p < np;
    const int2 pr = valid ? pairs[p] : make_int2(0, 0);
    const uint32_t* hrow = reinterpret_cast<const uint32_t*>(X + (long)pr.x * K);
    const long ridx = mode == 0 ? (long)pr.y : (long)(valid ? src[p] : 0);
    const uint32_t* wrow = reinterpret_cast<const uint32_t*>(base + ridx * K);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const long KC = K / 128;  // KC % PF == 0 (checked on the host)
    constexpr int PF = 4;
    uint32_t rb[PF][16], ra[PF][16];
#pragma unroll
    for (int sl = 0; sl < PF; sl++) {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const long o0 = (sl * 128 + 16 * j + 2 * t) / 2, o1 = (sl * 128 + 16 * j + 2 * t + 8) / 2;
        rb[sl][2 * j] = wrow[o0]; rb[sl][2 * j + 1] = wrow[o1]; ra[sl][2 * j] = hrow[o0]; ra[sl][2 * j + 1] = hrow[o1];
      }
    }
    for (long kc0 = 0; kc0 < KC; kc0 += PF) {
#pragma unroll
      for (int sl = 0; sl < PF; sl++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {  // same chain as lmh_codec: kc ascending, k16 steps ascending, from 0
          uint32_t a[4] = {ra[sl][2 * j], 0u, ra[sl][2 * j + 1], 0u};
          mma16816(acc, a, rb[sl][2 * j], rb[sl][2 * j + 1]);
        }
        const long kn = kc0 + sl + PF;
        if (kn < KC) {
#pragma unroll
          for (int j = 0; j < 8; j++) {
            const long o0 = (kn * 128 + 16 * j + 2 * t) / 2, o1 = (kn * 128 + 16 * j + 2 * t + 8) / 2;
            rb[sl][2 * j] = wrow[o0]; rb[sl][2 * j + 1] = wrow[o1]; ra[sl][2 * j] = hrow[o0]; ra[sl][2 * j + 1] = hrow[o1];
          }
        }
      }
    }
    // diagonal element (n, n) lives in lane (g = n, t = n >> 1), c[n & 1]
    const int n = g;
    const bool mine = (t == (n >> 1)) && valid;
    if (mine) {
      const float v = (n & 1) ? acc[1] : acc[0];
      if (vals) vals[p] = v;
      const __nv_bfloat16 bv = __float2bfloat16_rn(v);
      float fv = __bfloat162float(bv);
      if (fv != fv) fv = -INFINITY;
      const unsigned long long key = ((unsigned long long)f2ord(fv) << 32) |
                                     (unsigned long long)(0xFFFFFFFFu - (unsigned)(pr.y + voff));
      atomicMax(keys + pr.x, key);
    }
  }
}

// ---------------- read flush (cold L2) ----------------
__global__ void read_flush_kernel(const uint4* __restrict__ p, long n, unsigned* sink) {
  uint32_t x = 0;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const uint4 v = ldg_nc(p + i);
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (x == 0x12345678u) atomicAdd(sink, 1u);
}

// ---------------- host ----------------
static cudaStream_t cur() { return at::cuda::getCurrentCUDAStream(); }

int64_t plane_verify(torch::Tensor Q, torch::Tensor plane) {
  const long N = Q.size(0), K = Q.size(1), ng = (N / 8) * (K / 128);
  auto bad = torch::zeros({1}, Q.options().dtype(torch::kInt64));
  plane_verify_kernel<<<(ng + 7) / 8, 256, 0, cur()>>>((const int8_t*)Q.data_ptr(), N, K, (const uint8_t*)plane.data_ptr(),
                                                       (unsigned long long*)bad.data_ptr());
  return bad.cpu().item<int64_t>();
}
void hprep(torch::Tensor X, torch::Tensor hq, torch::Tensor hs, torch::Tensor hn, torch::Tensor hqe) {
  const int M = X.size(0), K = X.size(1);
  hprep_kernel<<<M, 256, 0, cur()>>>((const __nv_bfloat16*)X.data_ptr(), K, (int8_t*)hq.data_ptr(), hs.data_ptr<float>(),
                                     hn.data_ptr<float>(), hqe.data_ptr<float>());
}
template <int MT, int NT, int WARPS, int STAGES, bool MAGIC>
static void launch_screen(const torch::Tensor& hq, int M, long K, const torch::Tensor& plane, const torch::Tensor& vA,
                          const torch::Tensor& vB, const torch::Tensor& hs, const torch::Tensor& hn, const torch::Tensor& hqe,
                          long N, torch::Tensor& lb, torch::Tensor& ecnt, torch::Tensor& ebuf, int ecap) {
  const long n8 = N / 8, per_cta = (long)WARPS * NT;
  const int grid = (n8 + per_cta - 1) / per_cta;
  const int smem = STAGES * MT * 16 * 16 * 16;
  auto k = screen_kernel<MT, NT, WARPS, STAGES, MAGIC>;
  static bool init = false;
  if (!init) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); init = true; }
  k<<<grid, WARPS * 32, smem, cur()>>>((const int8_t*)hq.data_ptr(), M, K, (const uint8_t*)plane.data_ptr(),
                                       vA.data_ptr<float>(), vB.data_ptr<float>(), hs.data_ptr<float>(), hn.data_ptr<float>(),
                                       hqe.data_ptr<float>(), N, (unsigned*)lb.data_ptr(), ecnt.data_ptr<int>(),
                                       (int2*)ebuf.data_ptr(), ecap);
}
void screen(torch::Tensor hq, int64_t M, torch::Tensor plane, torch::Tensor vA, torch::Tensor vB, torch::Tensor hs,
            torch::Tensor hn, torch::Tensor hqe, int64_t N, torch::Tensor lb, torch::Tensor ecnt, torch::Tensor ebuf, int64_t ecap,
            int64_t cfg) {
  TORCH_CHECK(M >= 1 && M <= 64);
  const long KK = hq.size(1) / 2;  // hq given as [Mmax][KC*256]
  switch (cfg) {
    case 0: launch_screen<4, 2, 4, 3, true>(hq, M, KK, plane, vA, vB, hs, hn, hqe, N, lb, ecnt, ebuf, ecap); break;
    case 1: launch_screen<4, 2, 4, 3, false>(hq, M, KK, plane, vA, vB, hs, hn, hqe, N, lb, ecnt, ebuf, ecap); break;
    case 2: launch_screen<4, 2, 8, 3, true>(hq, M, KK, plane, vA, vB, hs, hn, hqe, N, lb, ecnt, ebuf, ecap); break;
    case 3: launch_screen<4, 1, 8, 3, true>(hq, M, KK, plane, vA, vB, hs, hn, hqe, N, lb, ecnt, ebuf, ecap); break;
    case 4: launch_screen<3, 2, 4, 3, true>(hq, M, KK, plane, vA, vB, hs, hn, hqe, N, lb, ecnt, ebuf, ecap); break;
    case 5: launch_screen<1, 2, 4, 3, true>(hq, M, KK, plane, vA, vB, hs, hn, hqe, N, lb, ecnt, ebuf, ecap); break;
    default: TORCH_CHECK(false, "cfg");
  }
}
void finalize(int64_t M, torch::Tensor lbA, torch::Tensor lbB, torch::Tensor ecnt, torch::Tensor ebuf, int64_t ecap,
              torch::Tensor pair_cnt, torch::Tensor pairs, torch::Tensor src, torch::Tensor map, torch::Tensor kept,
              torch::Tensor fallback, int64_t capr) {
  finalize_kernel<<<M, 256, 0, cur()>>>(M, (const unsigned*)lbA.data_ptr(), (const unsigned*)lbB.data_ptr(), ecnt.data_ptr<int>(),
                                        (const int2*)ebuf.data_ptr(), ecap, pair_cnt.data_ptr<int>(), (int2*)pairs.data_ptr(),
                                        src.data_ptr<int>(), pairs.size(0), map.data_ptr<int>(), kept.data_ptr<int>(),
                                        fallback.data_ptr<int>(), capr);
}
void gather(torch::Tensor pair_cnt, torch::Tensor pairs, torch::Tensor src, torch::Tensor map, int64_t Wptr, int64_t K,
            torch::Tensor stage, int64_t blocks) {
  gather_kernel<<<blocks, 256, 0, cur()>>>(pair_cnt.data_ptr<int>(), pairs.size(0), (const int2*)pairs.data_ptr(), src.data_ptr<int>(),
                                           map.data_ptr<int>(), (const uint4*)Wptr, K, (uint4*)stage.data_ptr());
}
void unmap(torch::Tensor pair_cnt, torch::Tensor pairs, torch::Tensor map) {
  unmap_kernel<<<16, 256, 0, cur()>>>(pair_cnt.data_ptr<int>(), pairs.size(0), (const int2*)pairs.data_ptr(), map.data_ptr<int>());
}
void rescore(torch::Tensor X, torch::Tensor pair_cnt, torch::Tensor pairs, torch::Tensor src, int64_t base_ptr, int64_t mode,
             int64_t voff, torch::Tensor keys, int64_t blocks, int64_t vals_ptr) {
  TORCH_CHECK((X.size(1) / 128) % 4 == 0);
  rescore_kernel<<<blocks, 128, 0, cur()>>>((const __nv_bfloat16*)X.data_ptr(), X.size(1), pair_cnt.data_ptr<int>(), pairs.size(0),
                                            (const int2*)pairs.data_ptr(), src.data_ptr<int>(), (const __nv_bfloat16*)base_ptr,
                                            mode, voff, (unsigned long long*)keys.data_ptr(), (float*)vals_ptr);
}
void read_flush(torch::Tensor buf, torch::Tensor sink) {
  cudaCtxResetPersistingL2Cache();
  read_flush_kernel<<<1024, 256, 0, cur()>>>((const uint4*)buf.data_ptr(), buf.numel() * buf.element_size() / 16,
                                            (unsigned*)sink.data_ptr());
}
int64_t host_dev_ptr(int64_t hptr) {
  void* d = nullptr;
  TORCH_CHECK(cudaHostGetDevicePointer(&d, (void*)hptr, 0) == cudaSuccess, "cudaHostGetDevicePointer");
  return (int64_t)d;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plane_verify", &plane_verify); m.def("hprep", &hprep); m.def("screen", &screen); m.def("finalize", &finalize);
  m.def("gather", &gather); m.def("unmap", &unmap); m.def("rescore", &rescore); m.def("read_flush", &read_flush);
  m.def("host_dev_ptr", &host_dev_ptr);
}
