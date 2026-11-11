// T9-001 ext: multi-transfer copy-engine send/recv (doorbells via stream mem-ops), raw copies, small kernels, core spin.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#define FS 16
#define NT 64
struct Ctx { char* hbase; char* dbase; size_t slot; int rank; };
static Ctx C;
static size_t tot(size_t slot){ return 2*(size_t)NT*slot + 4*NT*FS*4; }
static char* slotp(int r,int t){ return C.hbase + ((size_t)(r*NT+t))*C.slot; }
static size_t offF(int r,int t){ return 2*NT*C.slot + ((size_t)(r*NT+t))*FS*4; }
static size_t offA(int r,int t){ return 2*NT*C.slot + ((size_t)(2*NT + r*NT+t))*FS*4; }
static CUdeviceptr fF(int r,int t){ return (CUdeviceptr)(C.dbase+offF(r,t)); }
static CUdeviceptr fA(int r,int t){ return (CUdeviceptr)(C.dbase+offA(r,t)); }
#define CK(x) TORCH_CHECK((x)==CUDA_SUCCESS, #x)
#define CKR(x) TORCH_CHECK((x)==cudaSuccess, #x)
void init(std::string path, int64_t slot, int64_t rank, bool create){
  size_t t=tot(slot);
  int fd=open(path.c_str(), O_RDWR|(create?O_CREAT:0),0600); TORCH_CHECK(fd>=0,"open");
  if(create) TORCH_CHECK(ftruncate(fd,t)==0,"ftruncate");
  void* m=mmap(nullptr,t,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); TORCH_CHECK(m!=MAP_FAILED,"mmap"); close(fd);
  CKR(cudaHostRegister(m,t,cudaHostRegisterMapped|cudaHostRegisterPortable));
  void* d; CKR(cudaHostGetDevicePointer(&d,m,0));
  C.hbase=(char*)m; C.dbase=(char*)d; C.slot=slot; C.rank=rank;
  if(create) for(int r=0;r<2;r++) for(int k=0;k<NT;k++){ *(volatile int*)(C.hbase+offA(r,k))=1; *(volatile int*)(C.hbase+offF(r,k))=0; }
}
// send piece: wait slot free, D2H x[off:off+n] -> my slot t, doorbell
void send(torch::Tensor x, int64_t t, int64_t off, int64_t n){
  cudaStream_t s=at::cuda::getCurrentCUDAStream(); CUstream cs=(CUstream)s; int me=C.rank;
  CK(cuStreamWaitValue32(cs,fA(me,t),1,CU_STREAM_WAIT_VALUE_EQ));
  CK(cuStreamWriteValue32(cs,fA(me,t),0,CU_STREAM_WRITE_VALUE_DEFAULT));
  CKR(cudaMemcpyAsync(slotp(me,t),(char*)x.data_ptr()+off,n,cudaMemcpyDeviceToHost,s));
  CK(cuStreamWriteValue32(cs,fF(me,t),1,CU_STREAM_WRITE_VALUE_DEFAULT));
}
void recvwait(int64_t t){
  cudaStream_t s=at::cuda::getCurrentCUDAStream(); CUstream cs=(CUstream)s; int pr=1-C.rank;
  CK(cuStreamWaitValue32(cs,fF(pr,t),1,CU_STREAM_WAIT_VALUE_EQ));
  CK(cuStreamWriteValue32(cs,fF(pr,t),0,CU_STREAM_WRITE_VALUE_DEFAULT));
}
void recvcopy(torch::Tensor dst, int64_t t, int64_t off, int64_t n){
  cudaStream_t s=at::cuda::getCurrentCUDAStream(); CUstream cs=(CUstream)s; int pr=1-C.rank;
  CKR(cudaMemcpyAsync((char*)dst.data_ptr()+off,slotp(pr,t),n,cudaMemcpyHostToDevice,s));
  CK(cuStreamWriteValue32(cs,fA(pr,t),1,CU_STREAM_WRITE_VALUE_DEFAULT));
}
// raw copies (bandwidth test, no flags)
void d2h_raw(torch::Tensor x,int64_t t,int64_t n){ CKR(cudaMemcpyAsync(slotp(C.rank,t),x.data_ptr(),n,cudaMemcpyDeviceToHost,at::cuda::getCurrentCUDAStream())); }
void h2d_raw(torch::Tensor x,int64_t t,int64_t n){ CKR(cudaMemcpyAsync(x.data_ptr(),slotp(C.rank,t),n,cudaMemcpyHostToDevice,at::cuda::getCurrentCUDAStream())); }
std::string shm_range(){ return std::to_string((unsigned long)C.hbase)+" "+std::to_string(tot(C.slot)); }
__device__ __forceinline__ float bf(uint32_t w,int hi){ return __uint_as_float(hi? (w&0xffff0000u):(w<<16)); }
__device__ __forceinline__ uint32_t pk(float a,float b){ __nv_bfloat162 s=__floats2bfloat162_rn(a,b); return *reinterpret_cast<uint32_t*>(&s); }
__global__ void add2_k(const uint4* a,const uint4* b,uint4* o,long n){
  for(long i=blockIdx.x*(long)blockDim.x+threadIdx.x;i<n;i+=(long)gridDim.x*blockDim.x){
    uint4 x=a[i],y=b[i],r; uint32_t*px=(uint32_t*)&x,*py=(uint32_t*)&y,*pr=(uint32_t*)&r;
    #pragma unroll
    for(int k=0;k<4;k++) pr[k]=pk(bf(px[k],0)+bf(py[k],0),bf(px[k],1)+bf(py[k],1));
    o[i]=r;
  }
}
void add2(torch::Tensor a,torch::Tensor b,torch::Tensor o){
  long n=o.numel()/8; cudaStream_t s=at::cuda::getCurrentCUDAStream(); int blocks=(int)std::min<long>((n+255)/256,256);
  add2_k<<<blocks,256,0,s>>>((const uint4*)a.data_ptr(),(const uint4*)b.data_ptr(),(uint4*)o.data_ptr(),n);
}
__global__ void stamp_k(long long* buf,int idx){ long long t; asm volatile("mov.u64 %0, %globaltimer;":"=l"(t)); buf[idx]=t; }
void stamp(torch::Tensor buf,int64_t idx){ stamp_k<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>((long long*)buf.data_ptr(),(int)idx); }
// "core": occupies every SM (1 block/SM via 64KB smem, 1024 threads) for ns of wall time
__global__ void core_k(long long ns){
  extern __shared__ char sm[]; if(threadIdx.x==0) sm[0]=1;
  long long t0; asm volatile("mov.u64 %0, %globaltimer;":"=l"(t0));
  while(true){ long long t; asm volatile("mov.u64 %0, %globaltimer;":"=l"(t)); if(t-t0>=ns) break; __nanosleep(200); }
}
void core(int64_t ns){
  static bool set=false; if(!set){ cudaFuncSetAttribute(core_k,cudaFuncAttributeMaxDynamicSharedMemorySize,64*1024); set=true; }
  int nsm; cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,0);
  core_k<<<nsm,1024,64*1024,at::cuda::getCurrentCUDAStream()>>>(ns);
}
// ---- codec: copied from host_staged_ar.cu lines 14-109 (unchanged) ----
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
// ---- T9-007: device-side encode / fused decode+add for the copy-engine path ----
// payload = A[nb] B[nb] C[nb] GH[4*ng] (16 B units), nb % 32 == 0. O (overflow) region NOT transferred: blocks that are raw
// beyond CEX per group are counted in ovf[0] (encode) / ovf[1] (decode) and would be lossy here (microbench only).
__global__ void enc_k(const uint4* __restrict__ x, uint4* __restrict__ enc, long nb, int* ovf) {
  long bk = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const bool v = bk < nb;
  uint4 r0 = make_uint4(0,0,0,0), r1 = r0, r2 = r0, r3 = r0;
  if (v) { r0 = x[4*bk]; r1 = x[4*bk+1]; r2 = x[4*bk+2]; r3 = x[4*bk+3]; }
  uint4 a, b, c; uint8_t h;
  enc_block(r0, r1, r2, r3, a, b, c, h);
  const bool raw = v && h == 0xFF;
  const unsigned bal = __ballot_sync(0xffffffffu, raw);
  const unsigned lt = (1u << lane) - 1;
  uint4* gh = enc + comp_gh_off(nb) + (bk >> 5) * 4;
  if (v) { enc[bk] = a; enc[nb + bk] = b; enc[2*nb + bk] = c; ((uint8_t*)gh)[lane] = h; }
  if (raw) { int r = __popc(bal & lt); if (r < CEX) gh[2 + r] = r3; else atomicAdd(ovf, 1); }
}
__global__ void dec_add_k(const uint4* __restrict__ x, const uint4* __restrict__ rx, uint4* __restrict__ out, long nb, int* ovf) {
  long bk = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const bool v = bk < nb;
  const uint4* gh = rx + comp_gh_off(nb) + (bk >> 5) * 4;
  uint4 a = make_uint4(0,0,0,0), b = a, c = a; uint32_t h = 0;
  if (v) { a = rx[bk]; b = rx[nb + bk]; c = rx[2*nb + bk]; h = ((const uint8_t*)gh)[lane]; }
  uint4 ex = (lane < CEX && (bk - lane) < nb) ? gh[2 + lane] : make_uint4(0,0,0,0);
  const bool raw = v && h == 0xFF;
  const unsigned bal = __ballot_sync(0xffffffffu, raw);
  const unsigned lt = (1u << lane) - 1;
  const int r = __popc(bal & lt);
  const int src = (raw && r < CEX) ? r : 0;
  uint4 o; o.x = __shfl_sync(0xffffffffu, ex.x, src); o.y = __shfl_sync(0xffffffffu, ex.y, src);
  o.z = __shfl_sync(0xffffffffu, ex.z, src); o.w = __shfl_sync(0xffffffffu, ex.w, src);
  if (!v) return;
  uint4 v0, v1, v2, v3;
  if (raw) { v0 = a; v1 = b; v2 = c; if (r < CEX) v3 = o; else { v3 = make_uint4(0,0,0,0); atomicAdd(ovf + 1, 1); } }
  else dec_block(a, b, c, h, v0, v1, v2, v3);
  out[4*bk] = add4(x[4*bk], v0); out[4*bk+1] = add4(x[4*bk+1], v1); out[4*bk+2] = add4(x[4*bk+2], v2); out[4*bk+3] = add4(x[4*bk+3], v3);
}
int64_t enc_bytes(int64_t numel) { long nb = numel / 32; return comp_o_off(nb) * 16; }
void enc(torch::Tensor x, torch::Tensor e, torch::Tensor ovf) {
  long nb = x.numel() / 32; TORCH_CHECK(nb % 32 == 0 && (long)e.numel() * e.element_size() >= comp_o_off(nb) * 16);
  enc_k<<<(nb + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)x.data_ptr(), (uint4*)e.data_ptr(), nb, (int*)ovf.data_ptr());
}
void dec_add(torch::Tensor x, torch::Tensor rx, torch::Tensor o, torch::Tensor ovf) {
  long nb = x.numel() / 32;
  dec_add_k<<<(nb + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)x.data_ptr(), (const uint4*)rx.data_ptr(), (uint4*)o.data_ptr(), nb, (int*)ovf.data_ptr());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("init",&init);m.def("send",&send);m.def("recvwait",&recvwait);m.def("recvcopy",&recvcopy);m.def("d2h_raw",&d2h_raw);m.def("h2d_raw",&h2d_raw);
  m.def("shm_range",&shm_range);m.def("add2",&add2);m.def("stamp",&stamp);m.def("core",&core);
  m.def("enc",&enc);m.def("dec_add",&dec_add);m.def("enc_bytes",&enc_bytes);
}
