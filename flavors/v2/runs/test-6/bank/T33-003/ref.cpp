// CPU reference + compare. usage: ref dir name
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
static float dec(uint8_t b) { int s = b >> 7, e = (b >> 3) & 15, m = b & 7; float v = e == 0 ? m * 0.125f * powf(2.f, -6) : (1.f + m / 8.f) * powf(2.f, e - 7); return s ? -v : v; }
int main(int argc, char** argv) {
  const int M = 48, N = 17408, K = 5120, BL = 40;
  std::vector<uint8_t> W((size_t)N * K), A((size_t)M * K); std::vector<float> xs(M * BL), ws(136 * BL); std::vector<uint16_t> out((size_t)M * N);
  auto rd = [&](const char* f, void* p, size_t sz) { char fn[512]; snprintf(fn, sizeof fn, "%s/%s", argv[1], f); FILE* x = fopen(fn, "rb"); if (!x || fread(p, 1, sz, x) != sz) { printf("read fail %s\n", fn); exit(1);} fclose(x); };
  rd("W.bin", W.data(), W.size()); rd("A.bin", A.data(), A.size()); rd("xs.bin", xs.data(), xs.size() * 4); rd("ws.bin", ws.data(), ws.size() * 4);
  char nm[256]; snprintf(nm, sizeof nm, "out_%s.bin", argv[2]); rd(nm, out.data(), out.size() * 2);
  float tab[256]; for (int i = 0; i < 256; ++i) tab[i] = dec((uint8_t)i);
  std::vector<float> Af((size_t)M * K); for (size_t i = 0; i < Af.size(); ++i) Af[i] = tab[A[i]];
  double maxrel = 0, maxabs = 0, refmax = 0; long bad = 0, cnt = 0;
  std::vector<double> ref((size_t)M * N);
  #pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    std::vector<float> wf(K); for (int k = 0; k < K; ++k) wf[k] = tab[W[(size_t)n * K + k]];
    for (int m = 0; m < M; ++m) {
      double tot = 0;
      for (int kb = 0; kb < BL; ++kb) {
        float t = 0; const float* a = &Af[(size_t)m * K + kb * 128]; const float* w = &wf[kb * 128];
        for (int k = 0; k < 128; ++k) t += a[k] * w[k];
        tot += (double)t * ((double)xs[m * BL + kb] * ws[(n / 128) * BL + kb]);
      }
      ref[(size_t)m * N + n] = tot;
    }
  }
  for (size_t i = 0; i < ref.size(); ++i) refmax = fmax(refmax, fabs(ref[i]));
  for (size_t i = 0; i < ref.size(); ++i) {
    uint32_t u = (uint32_t)out[i] << 16; float o; memcpy(&o, &u, 4);
    double d = fabs(o - ref[i]); maxabs = fmax(maxabs, d);
    ++cnt; if (d > 0.01 * fabs(ref[i]) + 1e-3 * refmax) ++bad;
  }
  printf("%s: refmax %.4g maxabs %.4g (%.2e of refmax) bad %ld / %ld\n", argv[2], refmax, maxabs, maxabs / refmax, bad, cnt);
}
