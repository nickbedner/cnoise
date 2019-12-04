#include <math.h>
#include <stdio.h>
#include <time.h>
#include "../include/cnoise/cnoise.h"

static inline float run_benchmark(float* (*perlin_func)(struct PerlinNoise*, size_t, size_t, size_t), struct PerlinNoise* perlin_noise, size_t size_x, size_t size_y, size_t size_z, bool parallel);

#undef _mm_extract_epi32
#define _mm_extract_epi32(v, n) *(((int32_t*)&v) + n)
#undef _mm_extract_epi64
#define _mm_extract_epi64(v, n) *(((int64_t*)&v) + n)
//#undef _mm256_set_m128i
//#define _mm256_set_m128i(xmm1, xmm2) _mm256_set_epi32(_mm_extract_epi32(xmm1, 3), _mm_extract_epi32(xmm1, 2), _mm_extract_epi32(xmm1, 1), _mm_extract_epi32(xmm1, 0), _mm_extract_epi32(xmm2, 3), _mm_extract_epi32(xmm2, 2), _mm_extract_epi32(xmm2, 1), _mm_extract_epi32(xmm2, 0))

int main(int argc, char* argv[]) {
  const int size_x = 8, size_y = 1, size_z = 1;

  struct PerlinNoise perlin_noise;
  perlin_noise_init(&perlin_noise);

  // Warm-up tests to prevent delay on first allocation
  printf("Warm-up time 1: %f\n", run_benchmark(&perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, false));
  printf("Warm-up time 2: %f\n", run_benchmark(&perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, true));
  printf("///////////////////////////////////////////////////////////////////\n");

#ifdef ARCH_32_64
  // The following are tests to find Apple's missing intrinsics
  // All intrinsics on this line are good
  __m256i x0 = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_blendv_ps(_mm256_sub_ps(_mm256_set1_ps(1.0), _mm256_set1_ps(1.0)), _mm256_set1_ps(1.0), _mm256_cmp_ps(_mm256_set1_ps(1.0), _mm256_setzero_ps(), _CMP_GT_OQ))));
  // Fine
  //__m256i x0 = _mm256_set1_epi32(1);
  // Fine
  __m128i x1_low = _mm_set_epi64x(_mm256_extract_epi64(x0, 1), _mm256_extract_epi64(x0, 0));
  //_mm256_extractf128_si256(x0, 0);
  //x1_low = _mm_add_epi32(x1_low, _mm_set1_epi32(1));
  //__m128i x1_high = _mm256_extractf128_si256(x0, 1);
  //x1_high = _mm_add_epi32(x1_high, _mm_set1_epi32(1));
  // TODO: Figure out what instruction causes problem here on osx
  __m256i x1 = _mm256_set_epi64x(_mm_extract_epi64(x1_low, 1), _mm_extract_epi64(x1_low, 0), _mm_extract_epi64(x1_low, 1), _mm_extract_epi64(x1_low, 0));
  printf("Test: %d", x1);
  //printf("Test: %d", x1_low);
  //printf("Test: %d %d", x1_low, x1_high);

//  if (check_simd_support(SIMD_AVX2)) {
//    printf("AVX2 parallel time: %f\n", run_benchmark(&perlin_noise_eval_3d_avx2, &perlin_noise, size_x, size_y, size_z, true));
//    printf("AVX2 single thread time: %f\n", run_benchmark(&perlin_noise_eval_3d_avx2, &perlin_noise, size_x, size_y, size_z, false));
//  } else
//    printf("AVX2 support not detected!\n");
//  printf("///////////////////////////////////////////////////////////////////\n");
//
//  if (check_simd_support(SIMD_AVX)) {
//    printf("AVX parallel time: %f\n", run_benchmark(&perlin_noise_eval_3d_avx, &perlin_noise, size_x, size_y, size_z, true));
//    printf("AVX single thread time: %f\n", run_benchmark(&perlin_noise_eval_3d_avx, &perlin_noise, size_x, size_y, size_z, false));
//  } else
//    printf("AVX support not detected!\n");
//  printf("///////////////////////////////////////////////////////////////////\n");
//
//  if (check_simd_support(SIMD_SSE4_1)) {
//    printf("SSE4.1 parallel time: %f\n", run_benchmark(&perlin_noise_eval_3d_avx, &perlin_noise, size_x, size_y, size_z, true));
//    printf("SSE4.1 single thread time: %f\n", run_benchmark(&perlin_noise_eval_3d_avx, &perlin_noise, size_x, size_y, size_z, false));
//  } else
//    printf("SSE4.1 support not detected!\n");
//  printf("///////////////////////////////////////////////////////////////////\n");
#else
// ARM Neon
#endif
  printf("Fallback parallel time: %f\n", run_benchmark(&perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, true));
  printf("Fallback single thread time: %f\n", run_benchmark(&perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, false));
  return 0;
}

static inline float run_benchmark(float* (*perlin_func)(struct PerlinNoise*, size_t, size_t, size_t), struct PerlinNoise* perlin_noise, size_t size_x, size_t size_y, size_t size_z, bool parallel) {
  perlin_noise->parallel = parallel;

  float start_time = (float)clock() / CLOCKS_PER_SEC;
  float* noise_set = perlin_func(perlin_noise, size_x, size_y, size_z);
  float end_time = (float)clock() / CLOCKS_PER_SEC;
  noise_free(noise_set);

  return end_time - start_time;
}