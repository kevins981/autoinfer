// Premise test: does host page size / allocation type change PCIe transfer speed and
// PCIe-vs-HBM interference on this machine?  Single process, one GPU (argv[1]).
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include <cctype>
#include <cstdint>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static const size_t HBUF = 64ull<<20;   // host buffer
static const size_t HBM = 2ull<<30;     // device stream buffer

__global__ void hbm_stream(const int4* __restrict__ p, size_t n, int iters, int4* out){
  int4 acc = make_int4(0,0,0,0);
  for (int it=0; it<iters; ++it)
    for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x){
      int4 v = __ldcs(p+i); acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
  if (acc.x == 0x12345 && acc.y == 7) out[0] = acc;
}
// SM-driven host round trip like hostar: write `bytes` from dev to host slot, then read `bytes` from another host slot.
__global__ void sm_rt(int4* __restrict__ hw, const int4* __restrict__ hr, const int4* __restrict__ d, int4* dout, size_t n){
  for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x) hw[i] = d[i];
  __threadfence_system();
  for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x*blockDim.x) dout[i] = __ldcv(hr+i);
}

struct HB { std::string name; char* p; };

static long thp_kb(void* p){ // AnonHugePages of the mapping containing p
  FILE* f=fopen("/proc/self/smaps","r"); char line[512]; unsigned long a,b; bool in=false; long kb=-1;
  while(fgets(line,sizeof line,f)){
    if (isxdigit(line[0]) && sscanf(line,"%lx-%lx ",&a,&b)==2) in = ((unsigned long)p>=a && (unsigned long)p<b);
    else if (in && !strncmp(line,"AnonHugePages:",14)) sscanf(line+14,"%ld",&kb);
  }
  fclose(f); return kb;
}

int main(int argc, char** argv){
  int dev = argc>1? atoi(argv[1]) : 0; CK(cudaSetDevice(dev));
  std::vector<HB> bufs;
  { void* p; CK(cudaHostAlloc(&p, HBUF, cudaHostAllocMapped|cudaHostAllocPortable)); bufs.push_back({"hostalloc",(char*)p}); }
  { void* p = mmap(nullptr,HBUF,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); madvise(p,HBUF,MADV_NOHUGEPAGE);
    memset(p,1,HBUF); CK(cudaHostRegister(p,HBUF,cudaHostRegisterMapped|cudaHostRegisterPortable)); bufs.push_back({"anon4k",(char*)p}); }
  { size_t sz=HBUF+(2<<20); char* r=(char*)mmap(nullptr,sz,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    char* p=(char*)(((uintptr_t)r+(2<<20)-1)&~((uintptr_t)(2<<20)-1)); int rc=madvise(p,HBUF,MADV_HUGEPAGE);
    memset(p,1,HBUF); CK(cudaHostRegister(p,HBUF,cudaHostRegisterMapped|cudaHostRegisterPortable));
    printf("thp madvise rc=%d AnonHugePages(kB) in mapping: %ld\n", rc, thp_kb(p)); bufs.push_back({"anonTHP",(char*)p}); }
  { char nm[128]; snprintf(nm,sizeof nm,"/dev/shm/l8pg_%d",getpid()); int fd=open(nm,O_RDWR|O_CREAT,0600); ftruncate(fd,HBUF);
    char* p=(char*)mmap(nullptr,HBUF,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd); unlink(nm); memset(p,1,HBUF);
    CK(cudaHostRegister(p,HBUF,cudaHostRegisterMapped|cudaHostRegisterPortable)); bufs.push_back({"devshm",p}); }

  int4 *dbig, *dsmall, *dout; CK(cudaMalloc(&dbig,HBM)); CK(cudaMemset(dbig,1,HBM));
  CK(cudaMalloc(&dsmall,64<<20)); CK(cudaMalloc(&dout,64<<20));
  cudaStream_t s1,s2; CK(cudaStreamCreateWithFlags(&s1,cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s2,cudaStreamNonBlocking));
  cudaEvent_t a,b,c,d; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventCreate(&c); cudaEventCreate(&d);
  int sms; cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
  const int HB_IT=1; // 2 GB -> ~2.2 ms
  auto hbm_launch=[&](cudaStream_t s){ hbm_stream<<<sms*4,512,0,s>>>(dbig,HBM/16,HB_IT,dout); };
  // warm
  hbm_launch(s1); CK(cudaDeviceSynchronize());
  float t; std::vector<float> v;
  auto med=[&](std::vector<float>& x){ std::sort(x.begin(),x.end()); return x[x.size()/2]; };
  // HBM alone
  v.clear(); for(int r=0;r<5;r++){ cudaEventRecord(a,s1); hbm_launch(s1); cudaEventRecord(b,s1); cudaEventSynchronize(b); cudaEventElapsedTime(&t,a,b); v.push_back(t);}
  float hbm_alone=med(v); printf("HBM stream alone: %.3f ms (%.0f GB/s)\n", hbm_alone, HB_IT*(double)HBM/hbm_alone/1e6);

  size_t sizes[] = {384<<10, 655<<10, 4<<20};
  
  for (auto& hb : bufs){
    for (size_t sz : sizes){
      const int NC = sz<500000?100:(sz<1000000?60:12);
      for (int kind=0; kind<3; kind++){ // 0 H2D, 1 D2H, 2 SM round trip (write sz, read sz)
        for (int rot=0; rot<2; rot++){ // fixed slot vs rotating through 64 MB
          float res[2]; float hbmt=0;
          for (int conc=0; conc<2; conc++){
            CK(cudaDeviceSynchronize());
            if (conc){ cudaEventRecord(c,s1); hbm_launch(s1); cudaEventRecord(d,s1); }
            cudaEventRecord(a,s2);
            for (int i=0;i<NC;i++){
              size_t off = rot? ((size_t)i*sz*2) % (HBUF - 2*sz) : 0; off &= ~(size_t)4095;
              char* h = hb.p + off;
              if (kind==0) CK(cudaMemcpyAsync(dsmall, h, sz, cudaMemcpyHostToDevice, s2));
              else if (kind==1) CK(cudaMemcpyAsync(h, dsmall, sz, cudaMemcpyDeviceToHost, s2));
              else { int4 *hw, *hr; CK(cudaHostGetDevicePointer((void**)&hw, h, 0)); CK(cudaHostGetDevicePointer((void**)&hr, h+sz, 0));
                     sm_rt<<<16,512,0,s2>>>(hw,hr,dsmall,dout,sz/16); }
            }
            cudaEventRecord(b,s2); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&t,a,b); res[conc]=t*1000/NC;
            if (conc){ CK(cudaEventSynchronize(d)); cudaEventElapsedTime(&hbmt,c,d); }
          }
          const char* kn = kind==0?"H2D":(kind==1?"D2H":"SMrt");
          double gbs0 = (kind==2?2:1)*sz/(res[0]*1e3), gbs1=(kind==2?2:1)*sz/(res[1]*1e3);
          printf("%-9s %-4s %5zuKB %s  alone %7.1f us (%5.1f GB/s) | beside HBM %7.1f us (%5.1f GB/s) x%.2f | HBM kernel %.3f ms (x%.3f; copies cover %.0f%% of it)\n",
            hb.name.c_str(), kn, sz>>10, rot?"rot":"fix", res[0], gbs0, res[1], gbs1, res[1]/res[0], hbmt, hbmt/hbm_alone, 100.0*res[1]*NC/1000/hbmt);
        }
      }
    }
  }
  return 0;
}
