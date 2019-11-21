# CNoise

[![Build Status](https://travis-ci.org/Zalrioth/cnoise.svg?branch=master)](https://travis-ci.org/Zalrioth/noise-in-c-and-glsl)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/c41a5345402f4831a1f09af4f2961b74)](https://www.codacy.com/app/Zalrioth/data-structures-in-c?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Zalrioth/data-structures-in-c&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Header only C port of libnoise with SIMD, multithreading, and extras. Will automatically select the best instruction set to use at runtime.

## Settings up a Project

Include the cnoise header and you're good to go

### Implemented

Auto detect best instruction set at runtime<br/>
Billow Noise<br/>
Perlin Noise<br/>
Ridged Fractal Noise<br/>
Open Simplex Noise<br/>
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
SSE2<br/>
SSE4.1<br/>
