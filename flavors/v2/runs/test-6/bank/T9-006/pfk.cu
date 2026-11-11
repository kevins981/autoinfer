// standalone L2 prefetch kernel: same asm / stride scheme as the hostar prefetch blocks (host_staged_ar.cu)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void pf_kernel(const uint4* pf, long pf_vec, const uint4* pf2, long pf2_vec, int step) {
  long stride = (long)gridDim.x * blockDim.x;
  for (long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * step; i < pf_vec; i += stride * step)
    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(pf + i));
  for (long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * step; i < pf2_vec; i += stride * step)
    asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(pf2 + i));
}
void l2pf(torch::Tensor pf, int64_t pf_bytes, int64_t blocks, int64_t step, std::optional<torch::Tensor> pf2) {
  int64_t b1 = std::min<int64_t>(pf_bytes, pf.numel() * pf.element_size());
  const uint4* p2 = nullptr; long v2 = 0;
  if (pf2.has_value() && pf2->defined()) { int64_t b2 = std::min<int64_t>(pf_bytes, pf2->numel() * pf2->element_size()); p2 = (const uint4*)pf2->data_ptr(); v2 = b2 / 16; }
  pf_kernel<<<blocks, 512, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)pf.data_ptr(), b1 / 16, p2, v2, (int)step);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("l2pf", &l2pf); }
