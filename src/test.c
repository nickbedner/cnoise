#include <math.h>
#include <stdio.h>
#include "../include/cnoise/cnoise.h"

int main(int argc, char *argv[]) {
  struct PerlinNoise perlin_noise;
  perlin_noise_init(&perlin_noise);

  for (int loop_num = 0; loop_num < 10; loop_num++)
    printf("Value from Perlin noise: %f\n", perlin_noise_eval_3d(&perlin_noise, cos(loop_num), sin(loop_num), tan(loop_num)));

  struct VoronoiNoise voronoi_noise;
  voronoi_noise_init(&voronoi_noise);

  for (int loop_num = 0; loop_num < 10; loop_num++)
    printf("Value from Voronoi noise: %f\n", voronoi_noise_eval_3d(&voronoi_noise, cos(loop_num), sin(loop_num), tan(loop_num)));

  struct RidgedFractalNoise ridged_fractal_noise;
  ridged_fractal_noise_init(&ridged_fractal_noise);

  for (int loop_num = 0; loop_num < 10; loop_num++)
    printf("Value from Ridged Fractal noise: %f\n", ridged_fractal_noise_eval_3d(&ridged_fractal_noise, cos(loop_num), sin(loop_num), tan(loop_num)));

  struct BillowNoise billow_noise;
  billow_noise_init(&billow_noise);

  for (int loop_num = 0; loop_num < 10; loop_num++)
    printf("Value from Billow noise: %f\n", billow_noise_eval_3d(&billow_noise, cos(loop_num), sin(loop_num), tan(loop_num)));

  struct WhiteNoise white_noise;
  white_noise_init(&white_noise);

  for (int loop_num = 0; loop_num < 10; loop_num++)
    printf("Value from White noise: %f\n", white_noise_eval_3d(&white_noise, cos(loop_num), sin(loop_num), tan(loop_num)));

  struct OpenSimplexNoise open_simplex_noise;
  open_simplex_noise_init(&open_simplex_noise);

  for (int loop_num = 0; loop_num < 10; loop_num++)
    printf("Value from Open Simplex noise: %f\n", open_simplex_noise_eval_3d(&open_simplex_noise, cos(loop_num), sin(loop_num), tan(loop_num)));

  return 0;
}
