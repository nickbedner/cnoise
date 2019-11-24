#include <math.h>
#include <stdio.h>
#include <time.h>
#include "../include/cnoise/cnoise.h"

static inline float run_benchmark(float* (*perlin_func)(struct PerlinNoise*, size_t, size_t, size_t), struct PerlinNoise* perlin_noise, size_t size_x, size_t size_y, size_t size_z, bool parallel);

int main(int argc, char* argv[]) {
  const int size_x = 256, size_y = 256, size_z = 256;

  struct PerlinNoise perlin_noise;
  perlin_noise_init(&perlin_noise);

#ifdef ARCH_32_64
  printf("AVX2 parallel time: %f\n", run_benchmark(perlin_noise_eval_3d_avx2, &perlin_noise, size_x, size_y, size_z, true));
  printf("AVX2 single thread time: %f\n", run_benchmark(perlin_noise_eval_3d_avx2, &perlin_noise, size_x, size_y, size_z, false));
  printf("///////////////////////////////////////////////////////////////////\n");
#else
// ARM Neon
#endif
  printf("Fallback parallel time: %f\n", run_benchmark(perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, true));
  printf("Fallback single thread time: %f\n", run_benchmark(perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, false));
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