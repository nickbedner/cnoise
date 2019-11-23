#pragma once
#ifndef COMMON_H
#define COMMON_H

#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(__WIN32__) || defined(__WINDOWS__)
#define PLATFORM_WIN32
#include <intrin.h>
#define cpuid(info, x) __cpuidex(info, x, 0)
#else
#define PLATFORM_OTHER
#include <cpuid.h>
#define cpuid(info, x) __cpuid_count(x, 0, info[0], info[1], info[2], info[3])
#endif

#if defined(__arm__) || defined(__aarch64__)
#define ARCH_ARM
#include <arm_neon.h>
#else
#define ARCH_32_64
#include <immintrin.h>
#include <smmintrin.h>
#endif

enum NoiseQuality {
  QUALITY_FAST,
  QUALITY_STANDARD,
  QUALITY_BEST
};

#define SQRT_3 1.7320508075688772935
#define X_NOISE_GEN 1619
#define Y_NOISE_GEN 31337
#define Z_NOISE_GEN 6971

enum SIMDType {
  SIMD_FALLBACK = 0,
  SIMD_SSE2 = 1,
  SIMD_SSE4_1 = 2,
  SIMD_AVX = 3,
  SIMD_AVX2 = 4,
  SIMD_AVX512F = 5,
  SIMD_NEON = 6
};

static inline void *noise_allocate(size_t alignment, size_t size) {
  void *mem = malloc(size + alignment + sizeof(void *));
  void **data = (void **)(((uintptr_t)mem + alignment + sizeof(void *)) & ~(alignment - 1));
  data[-1] = mem;
  return data;
}

static inline void noise_free(float *data) {
  free(((void **)data)[-1]);
}

static inline int detect_simd_support() {
#ifdef ARCH_32_64
  int cpu_info[4];
  cpuid(cpu_info, 1);

  //bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;

  bool sse2_supported = cpu_info[3] & (1 << 26) || false;
  bool sse4_1_supported = cpu_info[2] & (1 << 19) || false;
  bool avx_supported = cpu_info[2] & (1 << 28) || false;

  cpuid(cpu_info, 7);

  bool avx2_supported = cpu_info[1] & (1 << 5) || false;
  bool avx512f_supported = cpu_info[1] & (1 << 16) || false;

  if (avx512f_supported)
    return SIMD_AVX512F;
  else if (avx2_supported)
    return SIMD_AVX2;
  else if (avx_supported)
    return SIMD_AVX;
  else if (sse4_1_supported)
    return SIMD_SSE4_1;
  else if (sse2_supported)
    return SIMD_SSE2;
  else
    return SIMD_FALLBACK;
#else
// ARM stuff
#endif
}

static inline float make_int_32_range(float n) {
  if (n >= 1073741824.0)
    return (2.0 * fmod(n, 1073741824.0)) - 1073741824.0;
  else if (n <= -1073741824.0)
    return (2.0 * fmod(n, 1073741824.0)) + 1073741824.0;
  else
    return n;
}

static inline __m256 make_int_32_range_vec_256(__m256 n) {
  //__m256 comp_ge = _mm256_set1_ps(1073741824.0);
  //__m256 comp_le = _mm256_set1_ps(-1073741824.0);
  return n;

  //__m256 mask_ge = _mm256_cmp_ps(n, comp_ge, _CMP_GE_OQ);
  //*val = _mm256_blendv_ps(y, x, mask_ge);
  //
  ////////////////////////////////////////////////////
  //if (n >= 1073741824.0)
  //  return (2.0 * fmod(n, 1073741824.0)) - 1073741824.0;
  //else if (n <= -1073741824.0)
  //  return (2.0 * fmod(n, 1073741824.0)) + 1073741824.0;
  //else
  //  return n;
}

static inline float cubic_interp(float n0, float n1, float n2, float n3, float a) {
  float p = (n3 - n2) - (n0 - n1);
  float q = (n0 - n1) - p;
  float r = n2 - n0;
  float s = n1;
  return p * a * a * a + q * a * a + r * a + s;
}

static inline float s_curve3(float a) {
  return (a * a * (3.0 - 2.0 * a));
}

static inline __m256 s_curve3_vec_256(__m256 a) {
  return _mm256_mul_ps(a, _mm256_mul_ps(a, _mm256_sub_ps(_mm256_set1_ps(3.0), _mm256_mul_ps(_mm256_set1_ps(2.0), a))));
}

static inline float s_curve5(float a) {
  float a3 = a * a * a;
  float a4 = a3 * a;
  float a5 = a4 * a;
  return (6.0 * a5) - (15.0 * a4) + (10.0 * a3);
}

static inline __m256 s_curve5_vec_256(__m256 a) {
  __m256 a3 = _mm256_mul_ps(a, _mm256_mul_ps(a, a));
  __m256 a4 = _mm256_mul_ps(a3, a);
  __m256 a5 = _mm256_mul_ps(a4, a);

  return _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(6.0), a5), _mm256_mul_ps(_mm256_set1_ps(15.0), a4)), _mm256_mul_ps(_mm256_set1_ps(10.0), a3));
}

static inline float linear_interp(float n0, float n1, float a) {
  return ((1.0 - a) * n0) + (a * n1);
}

static inline __m256 linear_interp_vec_256(__m256 n0, __m256 n1, __m256 a) {
  return _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0), a), n0), _mm256_mul_ps(a, n1));
}

static inline int fast_floor(float x) {
  int xi = (int)x;
  return x < xi ? xi - 1 : xi;
}

static inline int int_value_noise_3d(int x, int y, int z, int seed) {
  // All constants are primes and must remain prime in order for this noise function to work correctly.
  //int n = (X_NOISE_GEN * x + Y_NOISE_GEN * y + Z_NOISE_GEN * z + SEED_NOISE_GEN * seed) & 0x7fffffff;
  int n = 1;
  n = (n >> 13) ^ n;

  return (n * (n * n * 60493 + 19990303) + 1376312589) & 0x7fffffff;
}

static inline float value_noise_3d(int x, int y, int z, int seed) {
  return 1.0 - ((float)int_value_noise_3d(x, y, z, seed) / 1073741824.0);
}

static inline float gradient_noise_3d(float fx, float fy, float fz, int ix, int iy, int iz, int seed) {
  int random_x = (*(int *)&fx) ^ (*(int *)&fx >> 16);
  int random_y = (*(int *)&fy) ^ (*(int *)&fy >> 16);
  int random_z = (*(int *)&fz) ^ (*(int *)&fz >> 16);

  random_x = seed ^ (X_NOISE_GEN * random_x);
  random_y = seed ^ (Y_NOISE_GEN * random_y);
  random_z = seed ^ (Z_NOISE_GEN * random_z);

  float xv_gradient = (random_x * random_x * random_x * 60493) / 2147483648.0;
  float yv_gradient = (random_y * random_y * random_y * 60493) / 2147483648.0;
  float zv_gradient = (random_z * random_z * random_z * 60493) / 2147483648.0;

  float xv_point = (fx - (float)ix);
  float yv_point = (fy - (float)iy);
  float zv_point = (fz - (float)iz);

  return ((xv_gradient * xv_point) + (yv_gradient * yv_point) + (zv_gradient * zv_point)) * 2.12;
}

static inline __m256 gradient_noise_3d_vec_256(__m256 fx, float fy, float fz, __m256i ix, int iy, int iz, int seed) {
  __m256i random_x = _mm256_xor_si256(_mm256_castps_si256(fx), _mm256_srlv_epi32(_mm256_castps_si256(fx), _mm256_set1_epi32(16)));
  int random_y = (*(int *)&fy) ^ (*(int *)&fy >> 16);
  int random_z = (*(int *)&fz) ^ (*(int *)&fz >> 16);

  // TODO: _mm256_mullo_epi32 can probably be optomized out
  random_x = _mm256_xor_si256(_mm256_set1_epi32(seed), _mm256_mullo_epi32(_mm256_set1_epi32(X_NOISE_GEN), random_x));
  random_y = seed ^ (Y_NOISE_GEN * random_y);
  random_z = seed ^ (Z_NOISE_GEN * random_z);

  //printf("Extracted value: %d\n", _mm256_extract_epi32(random_x, 0));

  __m256 xv_gradient = _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_mullo_epi32(random_x, _mm256_mullo_epi32(random_x, _mm256_mullo_epi32(random_x, _mm256_set1_epi32(60493))))), _mm256_set1_ps(2147483648.0));
  float yv_gradient = (random_y * random_y * random_y * 60493) / 2147483648.0;
  float zv_gradient = (random_z * random_z * random_z * 60493) / 2147483648.0;

  __m256 xv_point = _mm256_sub_ps(fx, _mm256_cvtepi32_ps(ix));
  float yv_point = (fy - (float)iy);
  float zv_point = (fz - (float)iz);

  return _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(xv_gradient, xv_point), _mm256_mul_ps(_mm256_set1_ps(yv_gradient), _mm256_set1_ps(yv_point))), _mm256_mul_ps(_mm256_set1_ps(zv_gradient), _mm256_set1_ps(zv_point))), _mm256_set1_ps(2.12));
}

static inline float gradient_coherent_noise_3d(float x, float y, float z, int seed, enum NoiseQuality noise_quality) {
  //int x0 = (x > 0.0 ? lrint(x) : lrint(x) - 1);
  int x0 = (x > 0.0 ? (int)x : (int)x - 1);
  int x1 = x0 + 1;
  int y0 = (y > 0.0 ? (int)y : (int)y - 1);
  int y1 = y0 + 1;
  int z0 = (z > 0.0 ? (int)z : (int)z - 1);
  int z1 = z0 + 1;

  float xs = 0, ys = 0, zs = 0;
  switch (noise_quality) {
    case QUALITY_FAST:
      xs = (x - (float)x0);
      ys = (y - (float)y0);
      zs = (z - (float)z0);
      break;
    case QUALITY_STANDARD:
      xs = s_curve3(x - (float)x0);
      ys = s_curve3(y - (float)y0);
      zs = s_curve3(z - (float)z0);
      break;
    case QUALITY_BEST:
      xs = s_curve5(x - (float)x0);
      ys = s_curve5(y - (float)y0);
      zs = s_curve5(z - (float)z0);
      break;
  }

  float n0 = gradient_noise_3d(x, y, z, x0, y0, z0, seed);
  float n1 = gradient_noise_3d(x, y, z, x1, y0, z0, seed);
  float ix0 = linear_interp(n0, n1, xs);
  n0 = gradient_noise_3d(x, y, z, x0, y1, z0, seed);
  n1 = gradient_noise_3d(x, y, z, x1, y1, z0, seed);
  float ix1 = linear_interp(n0, n1, xs);
  float iy0 = linear_interp(ix0, ix1, ys);
  n0 = gradient_noise_3d(x, y, z, x0, y0, z1, seed);
  n1 = gradient_noise_3d(x, y, z, x1, y0, z1, seed);
  ix0 = linear_interp(n0, n1, xs);
  n0 = gradient_noise_3d(x, y, z, x0, y1, z1, seed);
  n1 = gradient_noise_3d(x, y, z, x1, y1, z1, seed);
  ix1 = linear_interp(n0, n1, xs);
  float iy1 = linear_interp(ix0, ix1, ys);

  return linear_interp(iy0, iy1, zs);
}

static inline __m256 gradient_coherent_noise_3d_vec_256(__m256 x, float y, float z, int seed, enum NoiseQuality noise_quality) {
  __m256i x0 = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_blendv_ps(_mm256_sub_ps(x, _mm256_set1_ps(1.0)), x, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OQ))));
  __m256i x1 = _mm256_add_epi32(x0, _mm256_set1_epi32(1));
  int y0 = (y > 0.0 ? (int)y : (int)y - 1);
  int y1 = y0 + 1;
  int z0 = (z > 0.0 ? (int)z : (int)z - 1);
  int z1 = z0 + 1;

  __m256 xs;
  float ys, zs;
  switch (noise_quality) {
    case QUALITY_FAST:
      xs = _mm256_sub_ps(x, _mm256_cvtepi32_ps(x0));
      ys = (y - (float)y0);
      zs = (z - (float)z0);
      break;
    case QUALITY_STANDARD:
      xs = s_curve3_vec_256(_mm256_sub_ps(x, _mm256_cvtepi32_ps(x0)));
      ys = s_curve3(y - (float)y0);
      zs = s_curve3(z - (float)z0);
      break;
    case QUALITY_BEST:
      xs = s_curve5_vec_256(_mm256_sub_ps(x, _mm256_cvtepi32_ps(x0)));
      ys = s_curve5(y - (float)y0);
      zs = s_curve5(z - (float)z0);
      break;
  }

  __m256 n0 = gradient_noise_3d_vec_256(x, y, z, x0, y0, z0, seed);
  __m256 n1 = gradient_noise_3d_vec_256(x, y, z, x1, y0, z0, seed);
  __m256 ix0 = linear_interp_vec_256(n0, n1, xs);
  n0 = gradient_noise_3d_vec_256(x, y, z, x0, y1, z0, seed);
  n1 = gradient_noise_3d_vec_256(x, y, z, x1, y1, z0, seed);
  __m256 ix1 = linear_interp_vec_256(n0, n1, xs);
  __m256 iy0 = linear_interp_vec_256(ix0, ix1, _mm256_set1_ps(ys));
  n0 = gradient_noise_3d_vec_256(x, y, z, x0, y0, z1, seed);
  n1 = gradient_noise_3d_vec_256(x, y, z, x1, y0, z1, seed);
  ix0 = linear_interp_vec_256(n0, n1, xs);
  n0 = gradient_noise_3d_vec_256(x, y, z, x0, y1, z1, seed);
  n1 = gradient_noise_3d_vec_256(x, y, z, x1, y1, z1, seed);
  ix1 = linear_interp_vec_256(n0, n1, xs);
  __m256 iy1 = linear_interp_vec_256(ix0, ix1, _mm256_set1_ps(ys));

  return linear_interp_vec_256(iy0, iy1, _mm256_set1_ps(zs));
}

#endif  // COMMON_H
