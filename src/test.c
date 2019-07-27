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
    struct OpenSimplexNoise open_simplex_noise;
    open_simplex_noise_init(&open_simplex_noise, 0);

    for (int loop_num = 0; loop_num < 10; loop_num++)
        printf("Value from open simplex noise: %f\n", open_simplex_noise_eval2d(&open_simplex_noise, cos(loop_num), sin(loop_num)));

    return 0;
}
