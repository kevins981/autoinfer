// T20-001: fence-free, flag-in-data ("tagged") host all-reduce. Same 0.781 codec bits as the champion
// (codec.cuh = verbatim champion lines), carried in 63-bit payloads + 1 epoch-tag bit per 8 B word.
#include "codec.cuh"
#include <vector>
#define MASK63 0x7FFFFFFFFFFFFFFFull
#define UNITS 102      // 16 B units per group (204 words * 63 bits = 12852 >= 12800 logical bits)
#define WORDS 204
#define SW 404         // smem u32 per warp (1600 B logical + zero pad)
struct TP {
  const uint4* x; uint4* out;
  uint4* slot[2][2];
  int* state;
  int rank, G, mode, delay_rank; long n_vec; int delay_us, backoff_ns;
  unsigned long long* trace; long trace_cap;
  unsigned long long* stat;       // [0]=poll retries, [1]=main words accepted, [2]=O words accepted
  const uint4* pf; long pf_vec; int pf_step; const uint4* pf2; long pf2_vec;
};
static __device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
static __device__ __forceinline__ void st_cs_v2(uint4* p, unsigned long long a, unsigned long long b) {
  asm volatile("st.global.cs.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(a), "l"(b) : "memory");
}
static __device__ __forceinline__ unsigned long long pack63(const uint32_t* s, int j) {
  const int bp = 63 * j, w = bp >> 5, sh = bp & 31;
  unsigned long long v = (((unsigned long long)s[w + 1] << 32) | s[w]) >> sh;
  if (sh >= 2) v |= (unsigned long long)s[w + 2] << (64 - sh);
  return v & MASK63;
}
static __device__ __forceinline__ void deposit63(uint32_t* s, int j, unsigned long long pv) {
  const int bp = 63 * j, w = bp >> 5, sh = bp & 31;
  const unsigned long long lo = pv << sh;
  atomicOr(&s[w], (uint32_t)lo); atomicOr(&s[w + 1], (uint32_t)(lo >> 32));
  if (sh >= 2) atomicOr(&s[w + 2], (uint32_t)(pv >> (64 - sh)));
}
static __device__ __forceinline__ bool tag2ok(const uint4& v, uint32_t tg) { return ((v.y >> 31) == tg) && ((v.w >> 31) == tg); }

__global__ void __launch_bounds__(512) tag_kernel(TP p) {
  __shared__ int s_epoch;
  __shared__ __align__(16) uint32_t S[16][SW];
  const int G = p.G, tid = threadIdx.x, nt = 512;
  if (blockIdx.x >= 2 * G) {  // L2 prefetch blocks, identical to champion
    long stride = (long)(gridDim.x - 2 * G) * nt;
    for (long i = ((long)(blockIdx.x - 2 * G) * nt + tid) * p.pf_step; i < p.pf_vec; i += stride * p.pf_step)
      asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p.pf + i));
    for (long i = ((long)(blockIdx.x - 2 * G) * nt + tid) * p.pf_step; i < p.pf2_vec; i += stride * p.pf_step)
      asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(p.pf2 + i));
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      int d = atomicAdd(&p.state[1], 1);
      if (d == (int)gridDim.x - 1) {
        int e = *(volatile int*)&p.state[0];
        if (p.trace && e < p.trace_cap) atomicMax(p.trace + (size_t)e * 8 + 5, gtimer());
        p.state[1] = 0; __threadfence(); *(volatile int*)&p.state[0] = e + 1;
      }
    }
    return;
  }
  const bool writer = blockIdx.x < G;
  if (tid == 0) s_epoch = *(volatile int*)&p.state[0];
  __syncthreads();
  const int epoch = s_epoch, par = epoch & 1;
  const uint32_t tg = (epoch >> 1) & 1;
  unsigned long long* tr = (p.trace && epoch < p.trace_cap) ? p.trace + (size_t)epoch * 8 : nullptr;
  const int warp = tid >> 5, lane = tid & 31;
  const int wid = (writer ? blockIdx.x : blockIdx.x - G) * 16 + warp, W = G * 16;
  const long nb = p.n_vec >> 2, ng = nb >> 5;
  const unsigned lt = (1u << lane) - 1;
  uint32_t* s = S[warp];
  if (tr && tid == 0) atomicMax(tr + (writer ? 0 : 1), ~gtimer());   // inverted: max == earliest start
  if (writer) {
    uint4* mine = p.slot[p.rank][par];
    uint4* Ow = mine + ng * UNITS;     // sparse overflow words: 4 x 8 B per block, [payload32|epoch31|tag]
    if (p.delay_us && p.rank == p.delay_rank) {   // test-only: delay producer BEFORE its writes
      unsigned long long t0 = gtimer();
      while (gtimer() - t0 < (unsigned long long)p.delay_us * 1000ull) { }
    }
    for (long gi = wid; gi < ng; gi += W) {
      const long bk = gi * 32 + lane;
      uint4 xr[4];
#pragma unroll
      for (int k = 0; k < 4; k++) xr[k] = p.x[4 * bk + k];
      uint4 a, b, c; uint8_t h;
      enc_block(xr[0], xr[1], xr[2], xr[3], a, b, c, h);
      const bool raw = (h == 0xFF);
      const unsigned bal = __ballot_sync(0xffffffffu, raw);
      const int r = __popc(bal & lt);
      ((uint8_t*)s)[lane] = h;
      *(uint4*)&s[16 + 4 * lane] = a; *(uint4*)&s[144 + 4 * lane] = b; *(uint4*)&s[272 + 4 * lane] = c;
      if (lane < 4) s[400 + lane] = 0;
      if (raw && r < CEX) *(uint4*)&s[8 + 4 * r] = xr[3];
      if (raw && r >= CEX) {
        const unsigned long long hi = ((unsigned long long)(tg) << 63) | ((unsigned long long)(epoch & 0x7FFFFFFF) << 32);
        st_cs_v2(Ow + bk * 2, hi | xr[3].x, hi | xr[3].y);
        st_cs_v2(Ow + bk * 2 + 1, hi | xr[3].z, hi | xr[3].w);
      }
      __syncwarp();
      if (tr && lane == 0) atomicMax(tr + 7, ~gtimer());   // writer first-store stamp (inverted)
      for (int u = lane; u < UNITS; u += 32) {
        st_cs_v2(mine + gi * UNITS + u, pack63(s, 2 * u) | ((unsigned long long)tg << 63),
                 pack63(s, 2 * u + 1) | ((unsigned long long)tg << 63));
      }
      __syncwarp();
    }
    if (tr && tid == 0) atomicMax(tr + 6, gtimer());
  } else {
    const uint4* peer = p.slot[1 - p.rank][par];
    const uint4* Or = peer + ng * UNITS;
    const bool tagoff = p.mode & 1;
    uint4 v[4]; bool pend[4]; bool val[4];
#pragma unroll
    for (int k = 0; k < 4; k++) { val[k] = (lane + 32 * k) < UNITS; v[k] = (val[k] && wid < ng) ? ld_cv(peer + (long)wid * UNITS + lane + 32 * k) : make_uint4(0, 0, 0, 0); }
    for (long gi = wid; gi < ng; gi += W) {
      for (int i = lane; i < SW; i += 32) s[i] = 0;
      const uint4* src = peer + gi * UNITS;
      bool first = true;
      unsigned long long nretry = 0;
      while (true) {
        bool any = false;
#pragma unroll
        for (int k = 0; k < 4; k++) pend[k] = val[k] && !tagoff && !tag2ok(v[k], tg);
        for (int k = 0; k < 4; k++) any |= pend[k];
        if (tr && first) { bool some = false;
          for (int k = 0; k < 4; k++) some |= (val[k] && !pend[k]);
          if (some) { atomicMax(tr + 2, ~gtimer()); first = false; } }
        if (!any) break;
        if (p.backoff_ns > 0) __nanosleep(p.backoff_ns);
#pragma unroll
        for (int k = 0; k < 4; k++) if (pend[k]) v[k] = ld_cv(src + lane + 32 * k);
        nretry++;
      }
      if (tr) {
        unsigned long long t = gtimer();   // time at which this lane saw its group's last unit fresh
        atomicMax(tr + 3, t);
      }
      if (p.stat) { atomicAdd(p.stat, nretry); if (lane == 0) atomicAdd(p.stat + 1, (unsigned long long)WORDS); }
      uint4 vc[4];
#pragma unroll
      for (int k = 0; k < 4; k++) vc[k] = v[k];
      if (gi + W < ng) {
#pragma unroll
        for (int k = 0; k < 4; k++) v[k] = val[k] ? ld_cv(peer + (gi + W) * UNITS + lane + 32 * k) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int k = 0; k < 4; k++) if (val[k]) {
        const int u = lane + 32 * k;
        deposit63(s, 2 * u, (((unsigned long long)vc[k].y << 32) | vc[k].x) & MASK63);
        deposit63(s, 2 * u + 1, (((unsigned long long)vc[k].w << 32) | vc[k].z) & MASK63);
      }
      __syncwarp();
      const long bk = gi * 32 + lane;
      const uint32_t h = ((uint8_t*)s)[lane];
      const uint4 a = *(uint4*)&s[16 + 4 * lane], b = *(uint4*)&s[144 + 4 * lane], cc = *(uint4*)&s[272 + 4 * lane];
      const bool raw = (h == 0xFF);
      const unsigned bal = __ballot_sync(0xffffffffu, raw);
      const int r = __popc(bal & lt);
      uint4 v0, v1, v2, v3;
      if (raw) {
        v0 = a; v1 = b; v2 = cc;
        if (r < CEX) v3 = *(uint4*)&s[8 + 4 * r];
        else {
          const unsigned long long want = ((unsigned long long)(tg) << 63) | ((unsigned long long)(epoch & 0x7FFFFFFF) << 32);
          uint4 o0, o1;
          while (true) {
            o0 = ld_cv(Or + bk * 2); o1 = ld_cv(Or + bk * 2 + 1);
            if (tagoff) break;
            const unsigned long long m0 = ((unsigned long long)o0.y << 32) & 0xFFFFFFFF00000000ull, m1 = ((unsigned long long)o0.w << 32) & 0xFFFFFFFF00000000ull;
            const unsigned long long m2 = ((unsigned long long)o1.y << 32) & 0xFFFFFFFF00000000ull, m3 = ((unsigned long long)o1.w << 32) & 0xFFFFFFFF00000000ull;
            if (m0 == want && m1 == want && m2 == want && m3 == want) break;
            if (p.stat) atomicAdd(p.stat, 1ull);
          }
          if (p.stat) atomicAdd(p.stat + 2, 4ull);
          v3 = make_uint4(o0.x, o0.z, o1.x, o1.z);
        }
      } else dec_block(a, b, cc, h, v0, v1, v2, v3);
      p.out[4 * bk] = add4(p.x[4 * bk], v0); p.out[4 * bk + 1] = add4(p.x[4 * bk + 1], v1);
      p.out[4 * bk + 2] = add4(p.x[4 * bk + 2], v2); p.out[4 * bk + 3] = add4(p.x[4 * bk + 3], v3);
      __syncwarp();
    }
  }
  __syncthreads();
  if (tr && tid == 0 && !writer) atomicMax(tr + 4, gtimer());
  if (tid == 0) {
    __threadfence();
    int d = atomicAdd(&p.state[1], 1);
    if (d == (int)gridDim.x - 1) {
      if (tr) atomicMax(tr + 5, gtimer());
      p.state[1] = 0; __threadfence(); *(volatile int*)&p.state[0] = epoch + 1;
    }
  }
}

struct Ctx { size_t slot_bytes; TP p; bool init = false; int dev = 0; unsigned long long* stat = nullptr; };
static Ctx C;
void tag_init(std::string path, int64_t slot_bytes, int64_t rank, bool create) {
  size_t tot = 4 * (size_t)slot_bytes;
  int fd = open(path.c_str(), O_RDWR | (create ? O_CREAT : 0), 0600);
  TORCH_CHECK(fd >= 0, "open");
  if (create) TORCH_CHECK(ftruncate(fd, tot) == 0, "ftruncate");
  void* m = mmap(nullptr, tot, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  TORCH_CHECK(m != MAP_FAILED, "mmap"); close(fd);
  if (create) {   // parity-1 slots (first used by epoch 1, tag 0) start with tag 1; parity-0 slots (epoch 2, tag 1) with tag 0
    for (int r = 0; r < 2; r++) {
      unsigned long long* q = (unsigned long long*)((char*)m + (size_t)(r * 2 + 1) * slot_bytes);
      for (size_t i = 0; i < slot_bytes / 8; i++) q[i] = 0x8000000000000000ull;
    }
  }
  TORCH_CHECK(cudaHostRegister(m, tot, cudaHostRegisterMapped) == cudaSuccess, "hostregister");
  void* d; TORCH_CHECK(cudaHostGetDevicePointer(&d, m, 0) == cudaSuccess, "getdevptr");
  char* b = (char*)d;
  for (int r = 0; r < 2; r++) for (int q = 0; q < 2; q++) C.p.slot[r][q] = (uint4*)(b + (size_t)(r * 2 + q) * slot_bytes);
  void* st; cudaMalloc(&st, 64); cudaMemset(st, 0, 64);
  int one = 1; cudaMemcpy(st, &one, 4, cudaMemcpyHostToDevice);
  C.p.state = (int*)st; C.p.rank = rank; C.slot_bytes = slot_bytes; C.init = true;
  C.p.trace = nullptr; C.p.trace_cap = 0; cudaGetDevice(&C.dev);
}
void tag_trace_enable(int64_t cap) {
  void* t; TORCH_CHECK(cudaMalloc(&t, cap * 64) == cudaSuccess); cudaMemset(t, 0, cap * 64); cudaDeviceSynchronize();
  C.p.trace = (unsigned long long*)t; C.p.trace_cap = cap;
}
torch::Tensor tag_trace_dump() {
  cudaSetDevice(C.dev);
  auto out = torch::empty({C.p.trace_cap, 8}, torch::kInt64);
  cudaDeviceSynchronize();
  cudaMemcpy(out.data_ptr(), C.p.trace, C.p.trace_cap * 64, cudaMemcpyDeviceToHost);
  return out;
}
void tag_stat_enable() { void* t; cudaMalloc(&t, 64); cudaMemset(t, 0, 64); C.p.stat = (unsigned long long*)t; }
torch::Tensor tag_stat_get(bool reset) {
  auto out = torch::empty({8}, torch::kInt64); cudaDeviceSynchronize();
  cudaMemcpy(out.data_ptr(), C.p.stat, 64, cudaMemcpyDeviceToHost);
  if (reset) cudaMemset(C.p.stat, 0, 64);
  return out;
}
void tag_allreduce(torch::Tensor x, torch::Tensor out, int64_t G, int64_t mode, int64_t delay_us, int64_t delay_rank, int64_t backoff_ns,
                   std::optional<torch::Tensor> pf, int64_t pf_bytes, int64_t pf_blocks, int64_t pf_step, std::optional<torch::Tensor> pf2) {
  TORCH_CHECK(C.init);
  size_t bytes = x.numel() * 2;
  TORCH_CHECK(bytes % 2048 == 0 && G >= 1 && G <= 32);
  long nb = bytes / 64, ng = nb / 32;
  TORCH_CHECK((size_t)(ng * UNITS + nb * 2) * 16 <= C.slot_bytes, "slot too small");
  TP p = C.p; p.x = (const uint4*)x.data_ptr(); p.out = (uint4*)out.data_ptr();
  p.n_vec = bytes / 16; p.G = G; p.mode = mode; p.delay_us = delay_us; p.delay_rank = delay_rank; p.backoff_ns = backoff_ns;
  p.pf = nullptr; p.pf_vec = 0; p.pf2 = nullptr; p.pf2_vec = 0; p.pf_step = pf_step;
  int nblk = 2 * G;
  if (pf.has_value() && pf->defined() && pf_blocks > 0) {
    int64_t b1 = std::min<int64_t>(pf_bytes, pf->numel() * pf->element_size());
    p.pf = (const uint4*)pf->data_ptr(); p.pf_vec = b1 / 16;
    if (pf2.has_value() && pf2->defined()) { int64_t b2 = std::min<int64_t>(pf_bytes, pf2->numel() * pf2->element_size()); p.pf2 = (const uint4*)pf2->data_ptr(); p.pf2_vec = b2 / 16; }
    nblk += pf_blocks;
  }
  tag_kernel<<<nblk, 512, 0, at::cuda::getCurrentCUDAStream()>>>(p);
}
// ---- skew / load helpers (stress) ----
__global__ void spin_kernel(const int* d, int idx) {
  unsigned long long t0 = gtimer(), w = (unsigned long long)d[idx] * 1000ull;
  while (gtimer() - t0 < w) { }
}
void spin(torch::Tensor d, int64_t idx) { spin_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(d.data_ptr<int>(), (int)idx); }
__global__ void hbm_kernel(const float4* a, float4* b, long n, int iters) {
  for (int it = 0; it < iters; it++)
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) { float4 v = a[i]; v.x += 1.f; b[i] = v; }
}
void hbm(torch::Tensor a, torch::Tensor b, int64_t blocks, int64_t iters) {
  hbm_kernel<<<blocks, 512, 0, at::cuda::getCurrentCUDAStream()>>>((const float4*)a.data_ptr(), (float4*)b.data_ptr(), a.numel() * a.element_size() / 16, (int)iters);
}
__global__ void cmp_kernel(const uint32_t* a, const uint32_t* b, long n, unsigned long long* acc) {
  unsigned long long c = 0;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) c += (a[i] != b[i]);
  if (c) atomicAdd(acc, c);
}
void cmpacc(torch::Tensor a, torch::Tensor b, torch::Tensor acc) {
  long n = a.numel() * a.element_size() / 4;
  cmp_kernel<<<32, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint32_t*)a.data_ptr(), (const uint32_t*)b.data_ptr(), n, (unsigned long long*)acc.data_ptr());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &tag_init); m.def("trace_enable", &tag_trace_enable); m.def("trace_dump", &tag_trace_dump);
  m.def("stat_enable", &tag_stat_enable); m.def("stat_get", &tag_stat_get);
  m.def("allreduce", &tag_allreduce); m.def("spin", &spin); m.def("hbm", &hbm); m.def("cmpacc", &cmpacc);
}
