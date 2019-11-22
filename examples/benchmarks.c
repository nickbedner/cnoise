#include <math.h>
#include <stdio.h>
#include <time.h>
#include "../include/cnoise/cnoise.h"

int main(int argc, char* argv[]) {
  struct PerlinNoise perlin_noise;
  perlin_noise_init(&perlin_noise);

  const int x = 256, y = 256, z = 256;
  float start_time, end_time;

  float* test_simd = calloc(1, sizeof(float) * x * y * z);

  start_time = (float)clock() / CLOCKS_PER_SEC;

  for (int loop_x = 0; loop_x < x; loop_x++) {
    for (int loop_y = 0; loop_y < y; loop_y++) {
      for (int loop_z = 0; loop_z < z; loop_z++) {
        *(test_simd + (loop_x + (loop_y * x) + (loop_z * (x * y)))) = perlin_noise_eval_3d(&perlin_noise, loop_x, loop_y, loop_z);
      }
    }
  }

  end_time = (float)clock() / CLOCKS_PER_SEC;
  printf("Standard elapsed time: %f\n", end_time - start_time);

  start_time = (float)clock() / CLOCKS_PER_SEC;
  perlin_noise_eval_3d_vec_256(&perlin_noise, (float*)test_simd, 0.0, 0.0, 0.0, x, y, z);
  end_time = (float)clock() / CLOCKS_PER_SEC;
  printf("AVX2 elapsed time: %f\n", end_time - start_time);
  free(test_simd);

  return 0;
}
