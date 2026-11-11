#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>

#define MAXB 128
#define MAXK 8
#define FSTRIDE 16   // ints between flags (64B)

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
