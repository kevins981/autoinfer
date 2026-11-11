// T47-002 part 1: DRAM streaming-read bandwidth vs active SMs (one persistent block per SM, forced by smem).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>

// (a) plain vectorized ldg: each thread U independent 16-B loads per tile; tile = blockDim*U uint4.
template <int U>
__global__ void k_ldg(const uint4* __restrict__ p, long n_tiles, unsigned long long* out) {
  extern __shared__ char sm_[];  // only to force 1 block/SM
  const long tile_elems = (long)blockDim.x * U;
  uint32_t acc = 0;
  for (long t = blockIdx.x; t < n_tiles; t += gridDim.x) {
    const uint4* b = p + t * tile_elems + threadIdx.x;
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) v[u] = __ldg(b + (long)u * blockDim.x);
#pragma unroll
    for (int u = 0; u < U; u++) acc += v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x9e3779b9u) atomicAdd(out, 1ull);  // defeats DCE, never true in practice
  if (threadIdx.x == 0 && sm_[0] == 0x7f) atomicAdd(out, 2ull);
}

// (b) cp.async.cg ring: per thread V 16-B chunks per stage, S stages; each thread consumes only its own slots.
__device__ __forceinline__ void cp16(void* s, const void* g) {
  unsigned sa = (unsigned)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(g));
}
template <int S, int V>
__global__ void k_cpa(const uint4* __restrict__ p, long n_tiles, unsigned long long* out) {
  extern __shared__ uint4 ring[];  // S * blockDim * V uint4
  const long tile_elems = (long)blockDim.x * V;
  const int stage_elems = blockDim.x * V;
  uint32_t acc = 0;
  long t0 = blockIdx.x;
  // number of tiles this block handles
  long my_n = t0 < n_tiles ? (n_tiles - 1 - t0) / gridDim.x + 1 : 0;
  auto issue = [&](long i) {  // i-th tile of this block into slot i % S
    if (i < my_n) {
      const uint4* g = p + (t0 + i * gridDim.x) * tile_elems + threadIdx.x;
      uint4* s = ring + (i % S) * stage_elems + threadIdx.x;
#pragma unroll
      for (int v = 0; v < V; v++) cp16(s + v * blockDim.x, g + (long)v * blockDim.x);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  for (int i = 0; i < S - 1; i++) issue(i);
  for (long i = 0; i < my_n; i++) {
    issue(i + S - 1);
    asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 1));
    const uint4* s = ring + (i % S) * stage_elems + threadIdx.x;
#pragma unroll
    for (int v = 0; v < V; v++) { uint4 x = s[v * blockDim.x]; acc += x.x ^ x.y ^ x.z ^ x.w; }
  }
  asm volatile("cp.async.wait_all;\n" ::);
  if (acc == 0x9e3779b9u) atomicAdd(out, 1ull);
}

static void setsmem(const void* f, int bytes) {
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

// variant: 0 = ldg 1024 thr U=8 (128 KB in flight/SM), 1 = ldg 512 thr U=4 (32 KB), 2 = ldg 1024 thr U=16 (256 KB regs-bound)
//          3 = cp.async 512 thr V=2 S=6 (96 KB ring), 4 = cp.async 256 thr V=4 S=4 (64 KB ring), 5 = cp.async 512 thr V=2 S=3 (48 KB)
void run(torch::Tensor buf, int64_t variant, int64_t nblocks, torch::Tensor out) {
  const uint4* p = (const uint4*)buf.data_ptr();
  long n16 = buf.numel() * buf.element_size() / 16;
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  auto* o = (unsigned long long*)out.data_ptr();
  const int FORCE = 80 * 1024;  // > half of the 100 KB/SM carve-out: 1 block per SM
  switch (variant) {
    case 0: { setsmem((void*)k_ldg<8>, FORCE); long tiles = n16 / (1024 * 8); k_ldg<8><<<nblocks, 1024, FORCE, st>>>(p, tiles, o); break; }
    case 1: { setsmem((void*)k_ldg<4>, FORCE); long tiles = n16 / (512 * 4); k_ldg<4><<<nblocks, 512, FORCE, st>>>(p, tiles, o); break; }
    case 2: { setsmem((void*)k_ldg<16>, FORCE); long tiles = n16 / (1024 * 16); k_ldg<16><<<nblocks, 1024, FORCE, st>>>(p, tiles, o); break; }
    case 3: { int sm = 6 * 512 * 2 * 16; int b = sm > FORCE ? sm : FORCE; setsmem((void*)k_cpa<6, 2>, b); long tiles = n16 / (512 * 2); k_cpa<6, 2><<<nblocks, 512, b, st>>>(p, tiles, o); break; }
    case 4: { int sm = 4 * 256 * 4 * 16; int b = sm > FORCE ? sm : FORCE; setsmem((void*)k_cpa<4, 4>, b); long tiles = n16 / (256 * 4); k_cpa<4, 4><<<nblocks, 256, b, st>>>(p, tiles, o); break; }
    case 5: { int sm = 3 * 512 * 2 * 16; int b = sm > FORCE ? sm : FORCE; setsmem((void*)k_cpa<3, 2>, b); long tiles = n16 / (512 * 2); k_cpa<3, 2><<<nblocks, 512, b, st>>>(p, tiles, o); break; }
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
}
void reset_persisting() { cudaCtxResetPersistingL2Cache(); }
int blocks_per_sm(int64_t variant) {
  int n = 0; const int FORCE = 80 * 1024;
  switch (variant) {
    case 0: setsmem((void*)k_ldg<8>, FORCE); cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k_ldg<8>, 1024, FORCE); break;
    case 3: setsmem((void*)k_cpa<6, 2>, 98304); cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k_cpa<6, 2>, 512, 98304); break;
    case 4: setsmem((void*)k_cpa<4, 4>, FORCE); cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k_cpa<4, 4>, 256, FORCE); break;
  }
  return n;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run); m.def("reset_persisting", &reset_persisting); m.def("blocks_per_sm", &blocks_per_sm);
}
