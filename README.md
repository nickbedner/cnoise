# CNoise

[![Build Status](https://github.com/Zalrioth/cnoise/workflows/CI/badge.svg)](https://github.com/Zalrioth/cnoise/commits/master)
[![Build Status](https://travis-ci.org/Zalrioth/cnoise.svg?branch=master)](https://travis-ci.org/Zalrioth/cnoise)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/cec66d7aa0304d15ade4ac7b8a0aff95)](https://www.codacy.com/gh/nickbedner/cnoise/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nickbedner/cnoise&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Header only C noise library with SIMD, multithreading, and more. Will automatically select the best instruction set to use at runtime.

## Setting up a Project

Include the cnoise header and you're good to go. OpenMP is required for multithreading support but can work without it.

```c
#include <stdio.h>
#include <cnoise/cnoise.h>

int main(int argc, char* argv[]) {
  const int x_size = 16, y_size = 4, z_size = 4;

  struct PerlinNoise perlin_noise;
  perlin_noise_init(&perlin_noise);

  float* noise_set = perlin_noise_eval_3d(&perlin_noise, x_size, y_size, z_size);

  int index = 0;
  for (int loop_z = 0; loop_z < z_size; loop_z++)
    for (int loop_y = 0; loop_y < y_size; loop_y++)
      for (int loop_x = 0; loop_x < x_size; loop_x++)
        printf("Perlin noise value at X:%d Y:%d Z:%d is %f\n", loop_x, loop_y, loop_z, noise_set[index++]);

  noise_free(noise_set);

  return 0;
}


```

### Implemented

* AVX
* AVX2
* Benchmarks
* Billow Noise
* Perlin Noise
* Ridged Fractal Noise
* Runtime instruction select
* SSE2
* SSE4.1
* Voronoi Noise

### In Progress

* ARM Neon
* AVX-512F
* Doc
* OpenSimplex2 Noise
* White Noise

### Planned

* Checkerboard Noise
* Cubic Noise
* Hydraulic Noise
* GLSL implementations
* Sphere Noise
* Cube Noise
