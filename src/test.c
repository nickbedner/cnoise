#include "../include/noise/noise.h"
#include <stdio.h>
#include <math.h>

int open_simplex_noise_test();

int main(int argc, char *argv[])
{
    if (open_simplex_noise_test() == 0)
        printf("Open simplex noise test passed!\n");

    return 0;
}

int open_simplex_noise_test()
{
    struct PerlinNoise perlin_noise;
    perlin_noise_init(&perlin_noise);

    for (int loop_num = 0; loop_num < 10; loop_num++)
        printf("Value from Perlin noise: %f\n", perlin_noise_eval_3d(&perlin_noise, cos(loop_num), sin(loop_num), tan(loop_num)));

    struct VoronoiNoise voronoi_noise;
    voronoi_noise_init(&voronoi_noise);

    for (int loop_num = 0; loop_num < 10; loop_num++)
        printf("Value from Voronoi noise: %f\n", voronoi_noise_eval_3d(&voronoi_noise, cos(loop_num), sin(loop_num), tan(loop_num)));

    struct OpenSimplexNoise open_simplex_noise;
    open_simplex_noise_init(&open_simplex_noise, 0);

    for (int loop_num = 0; loop_num < 10; loop_num++)
        printf("Value from Open Simplex noise: %f\n", open_simplex_noise_eval_2d(&open_simplex_noise, cos(loop_num), sin(loop_num)));

    return 0;
}
