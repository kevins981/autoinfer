// Copy-engine 2-rank all-reduce: D2H memcpy into my mapped-host slot, stream write flag,
// stream wait peer flag, H2D memcpy of peer slot, small add kernel (rank0 + rank1 order).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdexcept>

__global__ void add2(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                     __nv_bfloat16* __restrict__ o, int64_t n) {
  int64_t i = (int64_t)(blockIdx.x * blockDim.x + threadIdx.x) * 8;
  for (; i < n; i += (int64_t)gridDim.x * blockDim.x * 8) {
    uint4 va = *(const uint4*)(a + i), vb = *(const uint4*)(b + i), vo;
    const __nv_bfloat16* pa = (const __nv_bfloat16*)&va; const __nv_bfloat16* pb = (const __nv_bfloat16*)&vb;
    __nv_bfloat16* po = (__nv_bfloat16*)&vo;
#pragma unroll
    for (int k = 0; k < 8; k++) po[k] = __float2bfloat16(__bfloat162float(pa[k]) + __bfloat162float(pb[k]));
    *(uint4*)(o + i) = vo;
  }
}

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s; cuGetErrorString(r, &s); throw std::runtime_error(s); } } while (0)

// dp: device ptr of mapped region; layout: 4 slots [parity][rank] of `slot` bytes, then flags (2 x u32)
void ce_allreduce(torch::Tensor inp, torch::Tensor out, torch::Tensor tmp, int64_t dp, int64_t slot,
                  int64_t rank, int64_t epoch, int64_t add_blocks) {
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  size_t n = inp.numel() * 2;
  int par = epoch & 1;
  CUdeviceptr flags = dp + 4 * slot;
  CUdeviceptr myf = flags + 4 * rank, pf = flags + 4 * (1 - rank);
  char* mys = (char*)dp + (2 * par + rank) * slot;
  char* ps = (char*)dp + (2 * par + 1 - rank) * slot;
  // peer must have finished reading my slot of this parity (epoch-2): it wrote flag epoch-1 after that read
  if (epoch >= 2) CK(cuStreamWaitValue32(stream, pf, (cuuint32_t)(epoch - 1), CU_STREAM_WAIT_VALUE_GEQ));
  cudaMemcpyAsync(mys, inp.data_ptr(), n, cudaMemcpyDefault, stream);
  CK(cuStreamWriteValue32(stream, myf, (cuuint32_t)epoch, 0));
  CK(cuStreamWaitValue32(stream, pf, (cuuint32_t)epoch, CU_STREAM_WAIT_VALUE_GEQ));
  cudaMemcpyAsync(tmp.data_ptr(), ps, n, cudaMemcpyDefault, stream);
  auto* a = (const __nv_bfloat16*)(rank == 0 ? inp.data_ptr() : tmp.data_ptr());
  auto* b = (const __nv_bfloat16*)(rank == 0 ? tmp.data_ptr() : inp.data_ptr());
  add2<<<add_blocks, 256, 0, stream>>>(a, b, (__nv_bfloat16*)out.data_ptr(), inp.numel());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("ce_allreduce", &ce_allreduce); }
