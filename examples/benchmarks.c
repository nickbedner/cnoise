#include <math.h>
#include <stdio.h>
#include <time.h>
#include "../include/cnoise/cnoise.h"

static inline float run_benchmark(float* (*perlin_func)(struct PerlinNoise*, size_t, size_t, size_t), struct PerlinNoise* perlin_noise, size_t size_x, size_t size_y, size_t size_z, bool parallel);

int main(int argc, char* argv[]) {
  const int size_x = 128, size_y = 128, size_z = 128;

  struct PerlinNoise perlin_noise;
  perlin_noise_init(&perlin_noise);

  // Warm-up tests to prevent delay on first allocation
  printf("Warm-up time 1: %f\n", run_benchmark(&perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, false));
  printf("Warm-up time 2: %f\n", run_benchmark(&perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, true));
  printf("///////////////////////////////////////////////////////////////////\n");

#ifdef ARCH_32_64
  if (check_simd_support(SIMD_AVX2)) {
    printf("AVX2 parallel time: %f\n", run_benchmark(&perlin_noise_eval_3d_avx2, &perlin_noise, size_x, size_y, size_z, true));
    printf("AVX2 single thread time: %f\n", run_benchmark(&perlin_noise_eval_3d_avx2, &perlin_noise, size_x, size_y, size_z, false));
  } else
    printf("AVX2 support not detected!\n");
  printf("///////////////////////////////////////////////////////////////////\n");

  if (check_simd_support(SIMD_AVX)) {
    printf("AVX parallel time: %f\n", run_benchmark(&perlin_noise_eval_3d_avx, &perlin_noise, size_x, size_y, size_z, true));
    printf("AVX single thread time: %f\n", run_benchmark(&perlin_noise_eval_3d_avx, &perlin_noise, size_x, size_y, size_z, false));
  } else
    printf("AVX support not detected!\n");
  printf("///////////////////////////////////////////////////////////////////\n");

  if (check_simd_support(SIMD_SSE4_1)) {
    printf("SSE4.1 parallel time: %f\n", run_benchmark(&perlin_noise_eval_3d_sse4_1, &perlin_noise, size_x, size_y, size_z, true));
    printf("SSE4.1 single thread time: %f\n", run_benchmark(&perlin_noise_eval_3d_sse4_1, &perlin_noise, size_x, size_y, size_z, false));
  } else
    printf("SSE4.1 support not detected!\n");
  printf("///////////////////////////////////////////////////////////////////\n");

  if (check_simd_support(SIMD_SSE2)) {
    printf("SSE2 parallel time: %f\n", run_benchmark(&perlin_noise_eval_3d_sse2, &perlin_noise, size_x, size_y, size_z, true));
    printf("SSE2 single thread time: %f\n", run_benchmark(&perlin_noise_eval_3d_sse2, &perlin_noise, size_x, size_y, size_z, false));
  } else
    printf("SSE2 support not detected!\n");
  printf("///////////////////////////////////////////////////////////////////\n");
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