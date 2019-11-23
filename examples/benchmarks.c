#include <math.h>
#include <stdio.h>
#include <time.h>
#include "../include/cnoise/cnoise.h"

static inline float run_benchmark(float* (*perlin_func)(struct PerlinNoise*, size_t, size_t, size_t), struct PerlinNoise* perlin_noise, size_t size_x, size_t size_y, size_t size_z, bool parallel);

int main(int argc, char* argv[]) {
  omp_set_num_threads(4);
  // omp_set_num_threads(omp_get_num_procs());

  const int size_x = 256, size_y = 256, size_z = 256;

  struct PerlinNoise perlin_noise;
  perlin_noise_init(&perlin_noise);

  printf("Parallel AVX2 time: %f\n", run_benchmark(perlin_noise_eval_3d_avx2, &perlin_noise, size_x, size_y, size_z, true));
  printf("Single thread AVX2 time: %f\n", run_benchmark(perlin_noise_eval_3d_avx2, &perlin_noise, size_x, size_y, size_z, false));

  printf("Parallel fallback time: %f\n", run_benchmark(perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, true));
  printf("Single thread fallback time: %f\n", run_benchmark(perlin_noise_eval_3d_fallback, &perlin_noise, size_x, size_y, size_z, false));

  return 0;
}

static inline float run_benchmark(float* (*perlin_func)(struct PerlinNoise*, size_t, size_t, size_t), struct PerlinNoise* perlin_noise, size_t size_x, size_t size_y, size_t size_z, bool parallel) {
  float start_time, end_time;
  float* data;

  perlin_noise->parallel = parallel;

  start_time = (float)clock() / CLOCKS_PER_SEC;
  data = perlin_func(perlin_noise, size_x, size_y, size_z);
  end_time = (float)clock() / CLOCKS_PER_SEC;
  noise_free(data);

  printf("///////////////////////////////////////////////////////////////////\n");

  return end_time - start_time;
}