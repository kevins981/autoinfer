#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#define FSTRIDE 16

static __device__ __forceinline__ uint4 ld_cv(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x),"=r"(r.y),"=r"(r.z),"=r"(r.w) : "l"(p) : "memory");
  return r;
}
static __device__ __forceinline__ void st_cs(uint4* p, uint4 v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p),"r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w) : "memory");
}
static __device__ __forceinline__ int ld_vol(const int* p) {
  int r; asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(r) : "l"(p) : "memory"); return r;
}
static __device__ __forceinline__ uint32_t add_bf16x2(uint32_t a, uint32_t b) {
  __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(&a), y = *reinterpret_cast<__nv_bfloat162*>(&b);
  float2 fx = __bfloat1622float2(x), fy = __bfloat1622float2(y);
  __nv_bfloat162 r = __floats2bfloat162_rn(fx.x + fy.x, fx.y + fy.y);
  return *reinterpret_cast<uint32_t*>(&r);
}
static __device__ __forceinline__ uint4 add4(uint4 a, uint4 b) {
  return make_uint4(add_bf16x2(a.x,b.x), add_bf16x2(a.y,b.y), add_bf16x2(a.z,b.z), add_bf16x2(a.w,b.w));
}


// ---- Bitwise-lossless fixed-rate re-encoding of the bf16 payload (mode & 16) ----
// Per 32 values (64 B raw, a "block"): a header byte = max exponent field; per value sign+7 mantissa
// bits (1 B) and a 4-bit exponent offset (max - e; code 15 = exponent field 0, so zeros/denormals stay
// exact). A block with any offset > 14 (non-zero exponent), or max exponent 255 (Inf/NaN), is sent
// raw: header 0xFF, raw bytes 0-47 in the three planes, bytes 48-63 in an exception slot.
// Blocks are grouped by 32 (one warp, one block per lane); each group has a 64 B header region:
// 32 header bytes + CEX exception slots (16 B), filled in lane order of the group's raw blocks
// (warp ballot), read speculatively with the headers so a raw block costs no dependent PCIe read.
// A group with more than CEX raw blocks (0.008% on real payloads) puts the rest in a sparse
// overflow array O[block] that only those blocks touch. Layout (16 B units) for nb blocks, ng
// groups: A[nb] (sm bytes 0-15), B[nb] (sm 16-31), C[nb] (offset nibbles), GH[4 ng], O[nb].
// Wire bytes: 1600 per 2048 raw (0.781) + 16 B per raw block beyond CEX in its group.
#define CEX 2
static __device__ __forceinline__ uint8_t ld_cv_u8(const uint8_t* p) {
  unsigned short r; asm volatile("ld.global.cv.u8 %0, [%1];" : "=h"(r) : "l"(p) : "memory"); return (uint8_t)r;
}
static __device__ __forceinline__ void st_cs_u8(uint8_t* p, uint8_t v) {
  asm volatile("st.global.cs.u8 [%0], %1;" :: "l"(p), "h"((unsigned short)v) : "memory");
}
static __device__ __forceinline__ void enc_block(uint4 r0, uint4 r1, uint4 r2, uint4 r3,
                                                 uint4& a, uint4& b, uint4& c, uint8_t& h) {
  const uint32_t w[16] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w,
                          r2.x, r2.y, r2.z, r2.w, r3.x, r3.y, r3.z, r3.w};
  uint32_t sm[8], e[8], code[8];
  uint32_t m = 0;
#pragma unroll
  for (int q = 0; q < 8; q++) {
    uint32_t P0 = __byte_perm(w[2 * q], w[2 * q + 1], 0x6420);  // low bytes of values 4q..4q+3
    uint32_t P1 = __byte_perm(w[2 * q], w[2 * q + 1], 0x7531);  // high bytes
    sm[q] = (P1 & 0x80808080u) | (P0 & 0x7F7F7F7Fu);
    e[q] = ((P1 << 1) & 0xFEFEFEFEu) | ((P0 >> 7) & 0x01010101u);
    m = __vmaxu4(m, e[q]);
  }
  m = __vmaxu4(m, m >> 16); m = __vmaxu4(m, m >> 8);
  const uint32_t mx = m & 0xFF, mb = mx * 0x01010101u;
  uint32_t bad = (mx == 255u);
#pragma unroll
  for (int q = 0; q < 8; q++) {
    uint32_t off = mb - e[q];                       // bytewise mb >= e: no borrows
    uint32_t z = __vcmpeq4(e[q], 0u);               // 0xFF per zero-exponent byte
    bad |= __vcmpgtu4(off, 0x0E0E0E0Eu) & ~z;
    code[q] = (off & ~z) | (0x0F0F0F0Fu & z);
  }
  a = make_uint4(sm[0], sm[1], sm[2], sm[3]); b = make_uint4(sm[4], sm[5], sm[6], sm[7]);
  c = make_uint4(code[0] | (code[1] << 4), code[2] | (code[3] << 4), code[4] | (code[5] << 4), code[6] | (code[7] << 4));
  h = (uint8_t)mx;
  if (bad) { a = r0; b = r1; c = r2; h = 0xFF; }
}
static __device__ __forceinline__ void dec_quad(uint32_t smw, uint32_t code, uint32_t mb, uint32_t& w0, uint32_t& w1) {
  uint32_t z = __vcmpeq4(code, 0x0F0F0F0Fu);
  uint32_t e = (mb - (code & ~z)) & ~z;             // offset <= max for valid encodings: no borrows
  uint32_t P0 = (smw & 0x7F7F7F7Fu) | ((e << 7) & 0x80808080u);
  uint32_t P1 = (smw & 0x80808080u) | ((e >> 1) & 0x7F7F7F7Fu);
  w0 = __byte_perm(P0, P1, 0x5140); w1 = __byte_perm(P0, P1, 0x7362);
}
// decode a non-raw block into 4 uint4 (32 bf16 values, original order)
static __device__ __forceinline__ void dec_block(uint4 a, uint4 b, uint4 c, uint32_t h,
                                                 uint4& o0, uint4& o1, uint4& o2, uint4& o3) {
  const uint32_t mb = h * 0x01010101u;
  const uint32_t cw[4] = {c.x, c.y, c.z, c.w};
  const uint32_t sw[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  uint32_t w[16];
#pragma unroll
  for (int q = 0; q < 8; q++) {
    uint32_t code = (cw[q >> 1] >> (4 * (q & 1))) & 0x0F0F0F0Fu;
    dec_quad(sw[q], code, mb, w[2 * q], w[2 * q + 1]);
  }
  o0 = make_uint4(w[0], w[1], w[2], w[3]); o1 = make_uint4(w[4], w[5], w[6], w[7]);
  o2 = make_uint4(w[8], w[9], w[10], w[11]); o3 = make_uint4(w[12], w[13], w[14], w[15]);
}
// compressed-slot layout helpers (all in 16 B units from the slot base)
static __host__ __device__ __forceinline__ long comp_gh_off(long nb) { return 3 * nb; }
static __host__ __device__ __forceinline__ long comp_o_off(long nb) { return 3 * nb + 4 * ((nb + 31) / 32); }
static __host__ __device__ __forceinline__ long comp_total_vec(long nb) { return comp_o_off(nb) + nb; }

static __device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t;
}
#define MAXP 8
struct RArgs {
  const uint4* x; uint4* out; const uint4* peer; int* pflag;   // peer slot (device ptr), peer's flags (stride FSTRIDE per piece)
  int* state;       // [0]=call counter, [1]=done counter
  int n_vec, P, R, comp;
  unsigned long long* trace; int trace_cap;
  const uint4* pf; long pf_vec; const uint4* pf2; long pf2_vec; int pf_step;
};
// Reader: R CTAs poll the peer's piece flags (set by the peer's copy engine path), read each piece, add, write out.
__global__ void hyb_reader(RArgs p) {
  const int tid = threadIdx.x, nt = blockDim.x;
  if (blockIdx.x >= p.R) {
    long stride = (long)(gridDim.x - p.R) * nt;
    for (long i = ((long)(blockIdx.x - p.R) * nt + tid) * p.pf_step; i < p.pf_vec; i += stride * p.pf_step)
      asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p.pf + i));
    for (long i = ((long)(blockIdx.x - p.R) * nt + tid) * p.pf_step; i < p.pf2_vec; i += stride * p.pf_step)
      asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p.pf2 + i));
    return;
  }
  const int g = blockIdx.x, R = p.R, P = p.P;
  __shared__ int s_idx;
  if (tid == 0) s_idx = *(volatile int*)&p.state[0];
  __syncthreads();
  unsigned long long* tr = (p.trace && s_idx < p.trace_cap) ? p.trace + (size_t)s_idx * 4 : nullptr;
  if (tr && tid == 0 && g == 0) tr[0] = gtimer();
  const long n_units = p.comp ? (p.n_vec >> 2) : p.n_vec;
  long pu = (n_units + P - 1) / P; if (p.comp) pu = (pu + 31) & ~31L;
  for (int c = 0; c < P; c++) {
    long lo_c = c * pu, hi_c = lo_c + pu; if (lo_c > n_units) lo_c = n_units; if (hi_c > n_units) hi_c = n_units;
    long sub = (hi_c - lo_c + R - 1) / R; if (p.comp) sub = (sub + 31) & ~31L;
    long lo = lo_c + g * sub, hi = lo + sub; if (lo > hi_c) lo = hi_c; if (hi > hi_c) hi = hi_c;
    if (tid == 0) {
      { unsigned long long t0 = gtimer(); while (ld_vol(p.pflag + c * FSTRIDE) == 0) { if (gtimer() - t0 > 50000000ull) { p.state[2] += 1; break; } } }
      __threadfence_system();
      if (tr && g == 0) { unsigned long long t = gtimer(); if (c == 0) tr[1] = t; tr[3] = t; }
    }
    __syncthreads();
    if (p.comp) {
      const long nb = p.n_vec >> 2;
      const uint4* A = p.peer; const uint4* B = p.peer + nb; const uint4* Cc = p.peer + 2 * nb;
      const uint4* GH = p.peer + comp_gh_off(nb); const uint4* O = p.peer + comp_o_off(nb);
      const int lane = tid & 31; const long wstep = (long)(nt >> 5) * 32;
      const unsigned lt = (1u << lane) - 1;
      for (long gb = lo + (long)(tid >> 5) * 32; gb < hi; gb += 2 * wstep) {
        uint4 a[2], b[2], cc[2], ex[2]; uint32_t h[2]; bool v[2], gv[2];
#pragma unroll
        for (int u = 0; u < 2; u++) {
          const long g0 = gb + u * wstep, bk = g0 + lane;
          gv[u] = g0 < hi; v[u] = bk < hi;
          const uint4* gh = GH + (g0 >> 5) * 4;
          if (v[u]) { a[u] = ld_cv(A + bk); b[u] = ld_cv(B + bk); cc[u] = ld_cv(Cc + bk); h[u] = ld_cv_u8((const uint8_t*)gh + lane); }
          else h[u] = 0;
          ex[u] = (gv[u] && lane < CEX) ? ld_cv(gh + 2 + lane) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
          if (!gv[u]) break;
          const long bk = gb + u * wstep + lane;
          const bool raw = v[u] && h[u] == 0xFF;
          const unsigned bal = __ballot_sync(0xffffffffu, raw);
          const int r = __popc(bal & lt);
          const int src = (raw && r < CEX) ? r : 0;
          uint4 o;
          o.x = __shfl_sync(0xffffffffu, ex[u].x, src); o.y = __shfl_sync(0xffffffffu, ex[u].y, src);
          o.z = __shfl_sync(0xffffffffu, ex[u].z, src); o.w = __shfl_sync(0xffffffffu, ex[u].w, src);
          if (!v[u]) continue;
          uint4 v0, v1, v2, v3;
          if (raw) { v0 = a[u]; v1 = b[u]; v2 = cc[u]; v3 = (r < CEX) ? o : ld_cv(O + bk); }
          else dec_block(a[u], b[u], cc[u], h[u], v0, v1, v2, v3);
          p.out[4*bk] = add4(p.x[4*bk], v0); p.out[4*bk+1] = add4(p.x[4*bk+1], v1);
          p.out[4*bk+2] = add4(p.x[4*bk+2], v2); p.out[4*bk+3] = add4(p.x[4*bk+3], v3);
        }
      }
    } else {
      long i = lo + tid;
      for (; i + 3 * nt < hi; i += 4 * nt) {
        uint4 v0 = ld_cv(p.peer + i), v1 = ld_cv(p.peer + i + nt), v2 = ld_cv(p.peer + i + 2*nt), v3 = ld_cv(p.peer + i + 3*nt);
        p.out[i] = add4(p.x[i], v0); p.out[i+nt] = add4(p.x[i+nt], v1);
        p.out[i+2*nt] = add4(p.x[i+2*nt], v2); p.out[i+3*nt] = add4(p.x[i+3*nt], v3);
      }
      for (; i < hi; i += nt) p.out[i] = add4(p.x[i], ld_cv(p.peer + i));
    }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    int d = atomicAdd(&p.state[1], 1);
    if (d == R - 1) {
      // all my readers are done: re-arm the peer's flags (it sets them again two ARs later, other parity in between)
      for (int c = 0; c < P; c++) asm volatile("st.volatile.global.s32 [%0], %1;" :: "l"(p.pflag + c * FSTRIDE), "r"(0) : "memory");
      if (tr) tr[2] = gtimer();
      p.state[1] = 0; __threadfence(); *(volatile int*)&p.state[0] = s_idx + 1;
    }
  }
}
// Encoder (copy of hostar's compressed writer without flags/publication), writes a device staging buffer.
struct EArgs { const uint4* x; uint4* dst; int n_vec; int G; };
__global__ void hyb_encode(EArgs p) {
  const int tid = threadIdx.x, nt = blockDim.x, g = blockIdx.x;
  const long nb = p.n_vec >> 2;
  long cs = (nb + p.G - 1) / p.G; cs = (cs + 31) & ~31L;
  long lo = g * cs, hi = lo + cs; if (lo > nb) lo = nb; if (hi > nb) hi = nb;
  uint4* A = p.dst; uint4* B = p.dst + nb; uint4* Cc = p.dst + 2 * nb;
  uint4* GH = p.dst + comp_gh_off(nb); uint4* O = p.dst + comp_o_off(nb);
  const int lane = tid & 31; const long wstep = (long)(nt >> 5) * 32;
  const unsigned lt = (1u << lane) - 1;
  for (long gb = lo + (long)(tid >> 5) * 32; gb < hi; gb += 2 * wstep) {
    uint4 xr[2][4], a[2], b[2], cc[2]; uint8_t h[2]; bool v[2];
#pragma unroll
    for (int u = 0; u < 2; u++) {
      const long bk = gb + u * wstep + lane; v[u] = bk < hi;
#pragma unroll
      for (int k = 0; k < 4; k++) xr[u][k] = v[u] ? p.x[4 * bk + k] : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 2; u++) {
      const long g0 = gb + u * wstep, bk = g0 + lane;
      if (g0 >= hi) break;
      enc_block(xr[u][0], xr[u][1], xr[u][2], xr[u][3], a[u], b[u], cc[u], h[u]);
      const bool raw = v[u] && h[u] == 0xFF;
      const unsigned bal = __ballot_sync(0xffffffffu, raw);
      uint4* gh = GH + (g0 >> 5) * 4;
      if (v[u]) { A[bk] = a[u]; B[bk] = b[u]; Cc[bk] = cc[u]; ((uint8_t*)gh)[lane] = h[u]; }
      if (raw) { const int r = __popc(bal & lt); (r < CEX ? gh + 2 + r : O + bk)[0] = xr[u][3]; }
    }
  }
}
__global__ void flag_kernel(int* f) { __threadfence_system(); asm volatile("st.volatile.global.s32 [%0], %1;" :: "l"(f), "r"(1) : "memory"); }

struct HCtx {
  bool init = false; size_t slot_bytes; int rank;
  char* host; char* dev;      // base of the mapped region (host ptr, device ptr)
  size_t flags_off;
  int* state; unsigned long long* trace = nullptr; int trace_cap = 0;
  cudaStream_t cs; cudaEvent_t evf, evj;
  void* stage;                // device staging buffer for the compressed payload
  int dev_id;
  CUresult (*wv)(CUstream, CUdeviceptr, cuuint32_t, unsigned int) = nullptr;
};
static HCtx H;
static size_t hyb_total(size_t slot) { return 4 * slot + 2 * 2 * MAXP * FSTRIDE * 4; }
void hyb_init(std::string path, int64_t slot_bytes, int64_t rank, bool create) {
  size_t tot = hyb_total(slot_bytes);
  int fd = open(path.c_str(), O_RDWR | (create ? O_CREAT : 0), 0600);
  TORCH_CHECK(fd >= 0, "open");
  if (create) TORCH_CHECK(ftruncate(fd, tot) == 0, "ftruncate");
  void* m = mmap(nullptr, tot, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  TORCH_CHECK(m != MAP_FAILED, "mmap"); close(fd);
  TORCH_CHECK(cudaHostRegister(m, tot, cudaHostRegisterMapped) == cudaSuccess, "hostregister");
  void* d; TORCH_CHECK(cudaHostGetDevicePointer(&d, m, 0) == cudaSuccess, "devptr");
  H.host = (char*)m; H.dev = (char*)d; H.slot_bytes = slot_bytes; H.rank = rank; H.flags_off = 4 * slot_bytes;
  cudaMalloc((void**)&H.state, 64); cudaMemset(H.state, 0, 64);
  cudaStreamCreateWithFlags(&H.cs, cudaStreamNonBlocking);
  cudaEventCreateWithFlags(&H.evf, cudaEventDisableTiming); cudaEventCreateWithFlags(&H.evj, cudaEventDisableTiming);
  cudaMalloc(&H.stage, slot_bytes); cudaMemset(H.stage, 0, slot_bytes);
  cudaGetDevice(&H.dev_id);
  void* fn = nullptr; cudaDriverEntryPointQueryResult q;
  TORCH_CHECK(cudaGetDriverEntryPoint("cuStreamWriteValue32", &fn, cudaEnableDefault, &q) == cudaSuccess && fn, "no cuStreamWriteValue32");
  H.wv = (CUresult(*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int))fn;
  H.init = true;
}
int64_t hyb_timeouts() { cudaDeviceSynchronize(); int v; cudaMemcpy(&v, H.state + 2, 4, cudaMemcpyDeviceToHost); return v; }
void hyb_trace_enable(int64_t cap) {
  void* t = H.trace; if (!t) TORCH_CHECK(cudaMalloc(&t, cap * 32) == cudaSuccess); cudaMemset(t, 0, cap * 32); cudaDeviceSynchronize();
  H.trace = (unsigned long long*)t; H.trace_cap = cap; cudaMemset(H.state, 0, 8); cudaDeviceSynchronize();
}
torch::Tensor hyb_trace_dump() {
  cudaDeviceSynchronize();
  auto out = torch::empty({H.trace_cap, 4}, torch::kInt64);
  cudaMemcpy(out.data_ptr(), H.trace, H.trace_cap * 32, cudaMemcpyDeviceToHost);
  return out;
}
static char* slot_h(int r, int par) { return H.host + ((size_t)(r * 2 + par)) * H.slot_bytes; }
static char* slot_d(int r, int par) { return H.dev + ((size_t)(r * 2 + par)) * H.slot_bytes; }
static int* flag_h(int r, int par, int c) { return (int*)(H.host + H.flags_off) + ((size_t)((r * 2 + par) * MAXP + c)) * FSTRIDE; }
static int* flag_d(int r, int par, int c) { return (int*)(H.dev + H.flags_off) + ((size_t)((r * 2 + par) * MAXP + c)) * FSTRIDE; }

// comp: 0 raw, 1 compressed. flagmode: 0 = cuStreamWriteValue32 mem-op, 1 = 1-thread kernel.
void hyb_allreduce(torch::Tensor x, torch::Tensor out, int64_t P, int64_t R, int64_t threads, int64_t par,
                   int64_t comp, int64_t flagmode, int64_t enc_blocks,
                   std::optional<torch::Tensor> pf, int64_t pf_bytes, int64_t pf_blocks, int64_t pf_step,
                   std::optional<torch::Tensor> pf2) {
  TORCH_CHECK(H.init && P >= 1 && P <= MAXP);
  size_t bytes = x.numel() * 2; TORCH_CHECK(bytes % 16 == 0 && bytes <= H.slot_bytes);
  long n_vec = bytes / 16;
  if (comp) { TORCH_CHECK(n_vec % 4 == 0 && (size_t)comp_total_vec(n_vec / 4) * 16 <= H.slot_bytes); }
  cudaStream_t cur = at::cuda::getCurrentCUDAStream();
  const int me = H.rank, peer = 1 - H.rank;
  // fork: the copy branch starts when the GEMM (whatever precedes on `cur`) is done
  cudaEventRecord(H.evf, cur); cudaStreamWaitEvent(H.cs, H.evf, 0);
  // ---- reader on the main stream
  RArgs a; a.x = (const uint4*)x.data_ptr(); a.out = (uint4*)out.data_ptr();
  a.peer = (const uint4*)slot_d(peer, par); a.pflag = flag_d(peer, par, 0);
  a.state = H.state; a.n_vec = n_vec; a.P = P; a.R = R; a.comp = comp; a.trace = H.trace; a.trace_cap = H.trace_cap;
  a.pf = nullptr; a.pf_vec = 0; a.pf2 = nullptr; a.pf2_vec = 0; a.pf_step = pf_step;
  int grid = R;
  if (pf.has_value() && pf->defined() && pf_blocks > 0) {
    a.pf = (const uint4*)pf->data_ptr(); a.pf_vec = std::min<int64_t>(pf_bytes, pf->numel() * pf->element_size()) / 16;
    if (pf2.has_value() && pf2->defined()) { a.pf2 = (const uint4*)pf2->data_ptr(); a.pf2_vec = std::min<int64_t>(pf_bytes, pf2->numel() * pf2->element_size()) / 16; }
    grid += pf_blocks;
  }
  hyb_reader<<<grid, threads, 0, cur>>>(a);
  // ---- copy branch
  char* dst = slot_h(me, par);
  if (!comp) {
    long pv = (n_vec + P - 1) / P;
    for (int c = 0; c < P; c++) {
      long lo = std::min<long>(c * pv, n_vec), hi = std::min<long>(lo + pv, n_vec);
      if (hi > lo) cudaMemcpyAsync(dst + lo * 16, (const char*)x.data_ptr() + lo * 16, (hi - lo) * 16, cudaMemcpyDeviceToHost, H.cs);
      if (flagmode == 0) H.wv(H.cs, (CUdeviceptr)flag_d(me, par, c), 1, 0);
      else flag_kernel<<<1, 1, 0, H.cs>>>(flag_d(me, par, c));
    }
  } else {
    long nb = n_vec / 4;
    EArgs e; e.x = (const uint4*)x.data_ptr(); e.dst = (uint4*)H.stage; e.n_vec = n_vec; e.G = enc_blocks;
    hyb_encode<<<enc_blocks, threads, 0, H.cs>>>(e);
    long ng = (nb + 31) / 32;
    long pu = (nb + P - 1) / P; pu = (pu + 31) & ~31L;
    for (int c = 0; c < P; c++) {
      long lo = std::min<long>(c * pu, nb), hi = std::min<long>(lo + pu, nb);
      if (hi > lo) {
        long cnt = hi - lo;
        const char* sb = (const char*)H.stage;
        for (int k = 0; k < 3; k++)
          cudaMemcpyAsync(dst + (k * nb + lo) * 16, sb + (k * nb + lo) * 16, cnt * 16, cudaMemcpyDeviceToHost, H.cs);
        long glo = lo / 32, ghi = std::min<long>((hi + 31) / 32, ng);
        cudaMemcpyAsync(dst + (comp_gh_off(nb) + glo * 4) * 16, sb + (comp_gh_off(nb) + glo * 4) * 16, (ghi - glo) * 4 * 16, cudaMemcpyDeviceToHost, H.cs);
      }
      if (flagmode == 0) H.wv(H.cs, (CUdeviceptr)flag_d(me, par, c), 1, 0);
      else flag_kernel<<<1, 1, 0, H.cs>>>(flag_d(me, par, c));
    }
  }
  // join
  cudaEventRecord(H.evj, H.cs); cudaStreamWaitEvent(cur, H.evj, 0);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &hyb_init); m.def("timeouts", &hyb_timeouts); m.def("allreduce", &hyb_allreduce); m.def("trace_enable", &hyb_trace_enable); m.def("trace_dump", &hyb_trace_dump);
}
