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


static __device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t;
}

#define MAXP 8
#define TS 8
struct RArgs {
  const uint4* x; uint4* out; const uint4* peer; int* pflag;
  int* state;       // [0]=call counter, [1]=done counter, [2]=timeouts, u64 at byte 32 = producer end time
  int n_vec, P, R, nowait, one, nsl;
  unsigned long long* trace; int trace_cap;
};
__global__ void hyb_reader(RArgs p) {
  const int tid = threadIdx.x, nt = blockDim.x;
  const int g = blockIdx.x, R = p.R, P = p.P;
  __shared__ int s_idx;
  if (tid == 0) s_idx = *(volatile int*)&p.state[0];
  __syncthreads();
  unsigned long long* tr = (p.trace && s_idx < p.trace_cap) ? p.trace + (size_t)s_idx * TS : nullptr;
  if (tr && tid == 0 && g == 0) { tr[0] = gtimer(); tr[4] = *(volatile unsigned long long*)(p.state + 8); }
  const long n_units = p.n_vec;
  long pu = (n_units + P - 1) / P;
  for (int c = 0; c < P; c++) {
    long lo_c = c * pu, hi_c = lo_c + pu; if (lo_c > n_units) lo_c = n_units; if (hi_c > n_units) hi_c = n_units;
    long sub = (hi_c - lo_c + R - 1) / R;
    long lo = lo_c + g * sub, hi = lo + sub; if (lo > hi_c) lo = hi_c; if (hi > hi_c) hi = hi_c;
    if (!p.nowait) {
      if (!p.one) {
        if (tid == 0) {
          { unsigned long long t0 = gtimer(); while (ld_vol(p.pflag + c * FSTRIDE) == 0) { if (gtimer() - t0 > 100000000ull) { atomicAdd(&p.state[2], 1); break; } } }
          __threadfence_system();
          if (tr && g == 0) { unsigned long long t = gtimer(); if (c == 0) tr[1] = t; tr[3] = t; }
        }
      } else if (tid == 0) {
        volatile int* dw = (volatile int*)(p.state + 32 + c);
        if (g == 0) {   // the single poller: host flag (PCIe read) with back-off, then publish to a device word
          unsigned long long t0 = gtimer();
          while (ld_vol(p.pflag + c * FSTRIDE) == 0) { if (p.nsl) __nanosleep(p.nsl); if (gtimer() - t0 > 100000000ull) { atomicAdd(&p.state[2], 1); break; } }
          __threadfence_system();
          if (tr) { unsigned long long t = gtimer(); if (c == 0) tr[1] = t; tr[3] = t; }
          __threadfence(); *dw = 1;
        } else {
          unsigned long long t0 = gtimer();
          while (*dw == 0) { if (gtimer() - t0 > 100000000ull) { atomicAdd(&p.state[2], 1); break; } }
          __threadfence();
        }
      }
      __syncthreads();
    }
    long i = lo + tid;
    for (; i + 3 * nt < hi; i += 4 * nt) {
      uint4 v0 = ld_cv(p.peer + i), v1 = ld_cv(p.peer + i + nt), v2 = ld_cv(p.peer + i + 2*nt), v3 = ld_cv(p.peer + i + 3*nt);
      p.out[i] = add4(p.x[i], v0); p.out[i+nt] = add4(p.x[i+nt], v1);
      p.out[i+2*nt] = add4(p.x[i+2*nt], v2); p.out[i+3*nt] = add4(p.x[i+3*nt], v3);
    }
    for (; i < hi; i += nt) p.out[i] = add4(p.x[i], ld_cv(p.peer + i));
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    int d = atomicAdd(&p.state[1], 1);
    if (d == R - 1) {
      if (!p.nowait) for (int c = 0; c < P; c++) { if (p.one) *(volatile int*)(p.state + 32 + c) = 0; asm volatile("st.volatile.global.s32 [%0], %1;" :: "l"(p.pflag + c * FSTRIDE), "r"(0) : "memory"); }
      if (tr) tr[2] = gtimer();
      p.state[1] = 0; __threadfence(); *(volatile int*)&p.state[0] = s_idx + 1;
    }
  }
}
// producer: x[i] = out[i] * 0.5 for the first n elements (exact in bf16), stamps its end time
__global__ void produce(const __nv_bfloat16* out, __nv_bfloat16* x, int n, int* state) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] = __float2bfloat16(__bfloat162float(out[i]) * 0.5f);
  __syncthreads();
  if (threadIdx.x == 0) { __threadfence(); *(volatile unsigned long long*)(state + 8) = gtimer(); }
}
struct HCtx {
  bool init = false; size_t slot_bytes; int rank;
  char* host; char* dev; size_t flags_off;
  int* state; unsigned long long* trace = nullptr; int trace_cap = 0;
  cudaStream_t cs; cudaEvent_t evf, evj;
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
  cudaMalloc((void**)&H.state, 256); cudaMemset(H.state, 0, 256);
  cudaStreamCreateWithFlags(&H.cs, cudaStreamNonBlocking);
  cudaEventCreateWithFlags(&H.evf, cudaEventDisableTiming); cudaEventCreateWithFlags(&H.evj, cudaEventDisableTiming);
  void* fn = nullptr; cudaDriverEntryPointQueryResult q;
  TORCH_CHECK(cudaGetDriverEntryPoint("cuStreamWriteValue32", &fn, cudaEnableDefault, &q) == cudaSuccess && fn, "no cuStreamWriteValue32");
  H.wv = (CUresult(*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int))fn;
  H.init = true;
}
int64_t hyb_timeouts() { cudaDeviceSynchronize(); int v; cudaMemcpy(&v, H.state + 2, 4, cudaMemcpyDeviceToHost); return v; }
void hyb_trace_enable(int64_t cap) {
  void* t = H.trace; if (!t) TORCH_CHECK(cudaMalloc(&t, cap * TS * 8) == cudaSuccess); cudaMemset(t, 0, cap * TS * 8); cudaDeviceSynchronize();
  H.trace = (unsigned long long*)t; H.trace_cap = cap; cudaMemset(H.state, 0, 8); cudaDeviceSynchronize();
}
torch::Tensor hyb_trace_dump() {
  cudaDeviceSynchronize();
  auto out = torch::empty({H.trace_cap, TS}, torch::kInt64);
  cudaMemcpy(out.data_ptr(), H.trace, (size_t)H.trace_cap * TS * 8, cudaMemcpyDeviceToHost);
  return out;
}
static char* slot_h(int r, int par) { return H.host + ((size_t)(r * 2 + par)) * H.slot_bytes; }
static char* slot_d(int r, int par) { return H.dev + ((size_t)(r * 2 + par)) * H.slot_bytes; }
static int* flag_d(int r, int par, int c) { return (int*)(H.dev + H.flags_off) + ((size_t)((r * 2 + par) * MAXP + c)) * FSTRIDE; }

void hyb_produce(torch::Tensor out, torch::Tensor x, int64_t n) {
  produce<<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)out.data_ptr(), (__nv_bfloat16*)x.data_ptr(), (int)n, H.state);
}
// mode: 0 hybrid (CE D2H pieces + flags, reader polls); 1 CE D2H only; 2 SM read only (no wait, no CE); 3 both, reader does not wait
void hyb_allreduce(torch::Tensor x, torch::Tensor out, int64_t P, int64_t R, int64_t threads, int64_t par, int64_t mode, int64_t one, int64_t nsl) {
  TORCH_CHECK(H.init && P >= 1 && P <= MAXP);
  size_t bytes = x.numel() * 2; TORCH_CHECK(bytes % 16 == 0 && bytes <= H.slot_bytes);
  long n_vec = bytes / 16;
  cudaStream_t cur = at::cuda::getCurrentCUDAStream();
  const int me = H.rank, peer = 1 - H.rank;
  const bool ce = mode != 2, sm = mode != 1;
  if (ce) { cudaEventRecord(H.evf, cur); cudaStreamWaitEvent(H.cs, H.evf, 0); }
  if (sm) {
    RArgs a; a.x = (const uint4*)x.data_ptr(); a.out = (uint4*)out.data_ptr();
    a.peer = (const uint4*)slot_d(peer, par); a.pflag = flag_d(peer, par, 0);
    a.state = H.state; a.n_vec = n_vec; a.P = P; a.R = R; a.nowait = mode != 0; a.one = one; a.nsl = nsl; a.trace = H.trace; a.trace_cap = H.trace_cap;
    hyb_reader<<<R, threads, 0, cur>>>(a);
  }
  if (ce) {
    char* dst = slot_h(me, par);
    long pv = (n_vec + P - 1) / P;
    for (int c = 0; c < P; c++) {
      long lo = std::min<long>(c * pv, n_vec), hi = std::min<long>(lo + pv, n_vec);
      if (hi > lo) cudaMemcpyAsync(dst + lo * 16, (const char*)x.data_ptr() + lo * 16, (hi - lo) * 16, cudaMemcpyDeviceToHost, H.cs);
      if (mode == 0) H.wv(H.cs, (CUdeviceptr)flag_d(me, par, c), 1, 0);
    }
    cudaEventRecord(H.evj, H.cs); cudaStreamWaitEvent(cur, H.evj, 0);
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &hyb_init); m.def("timeouts", &hyb_timeouts); m.def("allreduce", &hyb_allreduce);
  m.def("trace_enable", &hyb_trace_enable); m.def("trace_dump", &hyb_trace_dump); m.def("produce", &hyb_produce);
}
