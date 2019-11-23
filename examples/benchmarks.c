#include <math.h>
#include <stdio.h>
#include <time.h>
#include "../include/cnoise/cnoise.h"

static inline float run_benchmark(float* (*perlin_func)(struct PerlinNoise*, size_t, size_t, size_t), struct PerlinNoise* perlin_noise, size_t size_x, size_t size_y, size_t size_z, bool parallel);

int main(int argc, char* argv[]) {
  printf("OpenMP threads available: %d\n", omp_get_num_procs());
  omp_set_num_threads(omp_get_num_procs());

  const int size_x = 128, size_y = 128, size_z = 128;

  struct PerlinNoise perlin_noise;
  perlin_noise_init(&perlin_noise);

  printf("AVX2 parallel time: %f\n", run_benchmark(perlin_noise_eval_3d_avx2, &perlin_noise, size_x, size_y, size_z, true));
  printf("AVX2 single thread time: %f\n", run_benchmark(perlin_noise_eval_3d_avx2, &perlin_noise, size_x, size_y, size_z, false));
  printf("///////////////////////////////////////////////////////////////////\n");
  printf("Fallback parallel time: %f\n", run_benchmark(perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, true));
  printf("Fallback single thread time: %f\n", run_benchmark(perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, false));

  return 0;
}

static inline float run_benchmark(float* (*perlin_func)(struct PerlinNoise*, size_t, size_t, size_t), struct PerlinNoise* perlin_noise, size_t size_x, size_t size_y, size_t size_z, bool parallel) {
  perlin_noise->parallel = parallel;

  float start_time = (float)clock() / CLOCKS_PER_SEC;
  float* data = perlin_func(perlin_noise, size_x, size_y, size_z);
  float end_time = (float)clock() / CLOCKS_PER_SEC;
  noise_free(data);

  return end_time - start_time;
}