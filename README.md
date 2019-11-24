# CNoise

[![Build Status](https://travis-ci.org/Zalrioth/cnoise.svg?branch=master)](https://travis-ci.org/Zalrioth/cnoise)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cec66d7aa0304d15ade4ac7b8a0aff95)](https://www.codacy.com/manual/Zalrioth/cnoise?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Zalrioth/cnoise&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Header only C noise library with SIMD, multithreading, and more. Will automatically select the best instruction set to use at runtime. Based on libnoise and FastNoise.

## Settings up a Project

Include the cnoise header and you're good to go. OpenMP required for multithreading support but can work without it.

```c
struct PerlinNoise perlin_noise;
perlin_noise_init(&perlin_noise);

float* data = perlin_noise_eval_3d(&perlin_noise, 64, 64, 64);
```

## Platforms tested

* Clang: Windows, Linux, ARM Linux, MacOSX
* GCC: Linux

### Implemented

* AVX2
* Billow Noise
* Perlin Noise
* Ridged Fractal Noise
* Runtime instruction select
* Voronoi Noise
* White Noise

### In Progress

* ARM Neon
* AVX
* AVX-512F
* Benchmarks
* Doc
* SSE2
* SSE4.1

### Planned

* Hydraulic Noise
* GLSL implementations
* Open Simplex Noise
