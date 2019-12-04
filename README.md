# CNoise

[![Build Status](https://github.com/Zalrioth/cnoise/workflows/CI/badge.svg)](https://github.com/Zalrioth/cnoise/commits/master)
[![Build Status](https://travis-ci.org/Zalrioth/cnoise.svg?branch=master)](https://travis-ci.org/Zalrioth/cnoise)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cec66d7aa0304d15ade4ac7b8a0aff95)](https://www.codacy.com/manual/Zalrioth/cnoise?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Zalrioth/cnoise&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Header only C noise library with SIMD, multithreading, and more. Will automatically select the best instruction set to use at runtime. Based on libnoise and FastNoise.

## Settings up a Project

Include the cnoise header and you're good to go. OpenMP is required for multithreading support but can work without it.

```c
#include <cnoise/cnoise.h>

int main(int argc, char* argv[]) {
  struct PerlinNoise perlin_noise;
  perlin_noise_init(&perlin_noise);

  float* noise_set = perlin_noise_eval_3d(&perlin_noise, 64, 64, 64);
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
* SSE4.1
* Voronoi Noise

### In Progress

* ARM Neon
* AVX-512F
* Doc
* SSE2
* White Noise

### Planned

* Hydraulic Noise
* GLSL implementations
* Open Simplex Noise
