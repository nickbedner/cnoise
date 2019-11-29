#include <math.h>
#include <stdio.h>
#include "../include/cnoise/cnoise.h"

int main(int argc, char* argv[]) {
  const int x_size = 16, y_size = 4, z_size = 4;

  struct PerlinNoise perlin_noise;
  perlin_noise_init(&perlin_noise);

  float* noise_set = perlin_noise_eval_3d(&perlin_noise, x_size, y_size, z_size);

  int index = 0;
  for (int loop_x = 0; loop_x < x_size; loop_x++) {
    for (int loop_y = 0; loop_y < y_size; loop_y++) {
      for (int loop_z = 0; loop_z < z_size; loop_z++) {
        printf("Perlin noise value at X:%d Y:%d Z:%d is %f\n", loop_x, loop_y, loop_z, noise_set[index++]);
      }
    }
  }

  noise_free(noise_set);

  return 0;
}
