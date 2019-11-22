# CNoise

[![Build Status](https://travis-ci.org/Zalrioth/cnoise.svg?branch=master)](https://travis-ci.org/Zalrioth/cnoise)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cec66d7aa0304d15ade4ac7b8a0aff95)](https://www.codacy.com/manual/Zalrioth/cnoise?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Zalrioth/cnoise&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Header only C noise library with SIMD, multithreading, and more. Will automatically select the best instruction set to use at runtime. Based on libnoise and FastNoise

## Settings up a Project

Include the cnoise header and you're good to go. OpenMP required for multithreading support but can still work without it.

### Implemented

Billow Noise<br/>
Perlin Noise<br/>
Ridged Fractal Noise<br/>
Runtime instruction select<br/>
Voronoi Noise<br/>
White Noise<br/>

### In Progress

AVX2<br/>
Benchmarks<br/>
Doc<br/>

### Planned

ARM Neon<br/>
AVX<br/>
AVX-512F<br/>
Hydraulic Noise<br/>
GLSL implementations<br/>
Open Simplex Noise<br/>
SSE2<br/>
SSE4.1<br/>
