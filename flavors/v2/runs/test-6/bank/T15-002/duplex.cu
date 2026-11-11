// PCIe duplex test with NUMA-node-0 pinned host buffers, SM-driven and copy-engine traffic, 1 or 2 GPUs at once.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s line %d\n",cudaGetErrorString(e),__LINE__); exit(1);}}while(0)
__device__ __forceinline__ void st_cs(uint4* p, uint4 v){asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"::"l"(p),"r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w):"memory");}
__device__ __forceinline__ uint4 ld_cv(const uint4* p){uint4 r; asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(r.x),"=r"(r.y),"=r"(r.z),"=r"(r.w):"l"(p):"memory"); return r;}
// mode 1: write only, 2: read only, 3: half CTAs write / half read
__global__ void k(int mode, uint4* hw, const uint4* hr, uint4* dsrc, uint4* ddst, size_t n) {
  int nb = gridDim.x, b = blockIdx.x; bool wr;
  int nbw = nb, bw = b;
  if (mode == 1) wr = true; else if (mode == 2) wr = false;
  else { wr = b < nb/2; nbw = nb/2; bw = wr ? b : b - nb/2; }
  size_t stride = (size_t)nbw * blockDim.x;
  if (wr) { for (size_t i = (size_t)bw*blockDim.x + threadIdx.x; i < n; i += stride) st_cs(hw + i, dsrc[i]); }
  else { uint4 acc = {0,0,0,0}; for (size_t i = (size_t)bw*blockDim.x + threadIdx.x; i < n; i += stride) { uint4 v = ld_cv(hr + i); acc.x ^= v.x; acc.y += v.y; } ddst[(size_t)bw*blockDim.x+threadIdx.x] = acc; }
}
struct Dev { int id; uint4 *hA, *hB, *dA, *dB, *dsrc, *ddst; uint4 *hAd, *hBd; cudaStream_t s1, s2; cudaEvent_t e0, e1; };
void* alloc_node0(size_t bytes) {
  void* p = mmap(nullptr, bytes, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
  unsigned long mask = 1; long r = syscall(SYS_mbind, p, bytes, 2 /*MPOL_BIND*/, &mask, 64, 0);
  if (r) printf("mbind failed\n");
  memset(p, 1, bytes); CK(cudaHostRegister(p, bytes, cudaHostRegisterMapped)); return p;
}
int main() {
  const size_t MAXB = 8 << 20;
  std::vector<Dev> D(2);
  for (int d = 0; d < 2; d++) {
    Dev& x = D[d]; x.id = d; CK(cudaSetDevice(d));
    x.hA = (uint4*)alloc_node0(MAXB); x.hB = (uint4*)alloc_node0(MAXB);
    CK(cudaHostGetDevicePointer((void**)&x.hAd, x.hA, 0)); CK(cudaHostGetDevicePointer((void**)&x.hBd, x.hB, 0));
    CK(cudaMalloc(&x.dsrc, MAXB)); CK(cudaMalloc(&x.ddst, MAXB)); CK(cudaMalloc(&x.dA, MAXB)); CK(cudaMalloc(&x.dB, MAXB));
    CK(cudaMemset(x.dsrc, 2, MAXB));
    CK(cudaStreamCreateWithFlags(&x.s1, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&x.s2, cudaStreamNonBlocking));
    CK(cudaEventCreate(&x.e0)); CK(cudaEventCreate(&x.e1));
  }
  system("grep -c N1= /proc/$PPID/numa_maps >/dev/null; true");
  char cmd[128]; snprintf(cmd, 128, "grep -E 'N[01]=' /proc/%d/numa_maps | grep -E 'shared|anon' | awk '{print $NF, $(NF-1), $(NF-2)}' | sort | uniq -c | tail -5", getpid()); system(cmd);
  size_t sizes[] = {393216, 491520, 1572864, 4194304};
  const char* names[] = {"SM-W", "SM-R", "SM-W+R(1 kernel, half CTAs each)", "CE-D2H", "CE-H2D", "CE-D2H+H2D", "SM-W + CE-H2D"};
  int grids[] = {32, 64, 128};
  const int REPS = 200;
  for (int ngpu = 1; ngpu <= 2; ngpu++)
  for (size_t bytes : sizes)
  for (int g : grids)
  for (int t = 0; t < 7; t++) {
    if (t >= 3 && g != 64) continue;
    size_t n = bytes / 16;
    float ms[2] = {0, 0};
    for (int pass = 0; pass < 2; pass++) {
      for (int d = 0; d < ngpu; d++) { Dev& x = D[d]; CK(cudaSetDevice(d)); CK(cudaEventRecord(x.e0, x.s1)); CK(cudaStreamWaitEvent(x.s2, x.e0)); }
      for (int r = 0; r < REPS; r++)
        for (int d = 0; d < ngpu; d++) {
          Dev& x = D[d]; CK(cudaSetDevice(d));
          int gg = (t == 2) ? 2*g : g;
          if (t <= 2) k<<<gg, 256, 0, x.s1>>>(t + 1, x.hAd, x.hBd, x.dsrc, x.ddst, n);
          if (t == 3 || t == 5) CK(cudaMemcpyAsync(x.hA, x.dA, bytes, cudaMemcpyDeviceToHost, x.s1));
          if (t == 4) CK(cudaMemcpyAsync(x.dB, x.hB, bytes, cudaMemcpyHostToDevice, x.s1));
          if (t == 5) CK(cudaMemcpyAsync(x.dB, x.hB, bytes, cudaMemcpyHostToDevice, x.s2));
          if (t == 6) { k<<<g, 256, 0, x.s1>>>(1, x.hAd, x.hBd, x.dsrc, x.ddst, n); CK(cudaMemcpyAsync(x.dB, x.hB, bytes, cudaMemcpyHostToDevice, x.s2)); }
        }
      for (int d = 0; d < ngpu; d++) { Dev& x = D[d]; CK(cudaSetDevice(d)); cudaEvent_t e2; CK(cudaEventCreate(&e2)); CK(cudaEventRecord(e2, x.s2)); CK(cudaStreamWaitEvent(x.s1, e2)); CK(cudaEventRecord(x.e1, x.s1)); }
      for (int d = 0; d < ngpu; d++) { Dev& x = D[d]; CK(cudaSetDevice(d)); CK(cudaEventSynchronize(x.e1)); CK(cudaEventElapsedTime(&ms[d], x.e0, x.e1)); }
    }
    double us0 = ms[0] * 1000 / REPS, us1 = ms[1] * 1000 / REPS;
    double dirs = (t == 2 || t == 5 || t == 6) ? 2 : 1;
    printf("ngpu=%d bytes=%7zu grid=%3d %-34s gpu0 %7.1f us (%5.1f GB/s/dir)  gpu1 %7.1f us\n", ngpu, bytes, g, names[t], us0, bytes / us0 / 1e3, us1);
  }
  return 0;
}
