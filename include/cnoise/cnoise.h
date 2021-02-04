#pragma once
#ifndef CNOISE_H
#define CNOISE_H

#include "billownoise.h"
#include "perlinnoise.h"
#include "ridgedfractalnoise.h"
#include "spherenoise.h"
#include "voronoinoise.h"
#include "whitenoise.h"

enum NoiseType {
  BILLOW_NOISE,
  PERLIN_NOISE,
  RIDGED_FRACTAL_NOISE,
  VORONOI_NOISE,
  WHITE_NOISE,
  SPHERE_NOISE
};

struct Noise {
  enum NoiseType noise_type;
  union {
    struct BillowNoise billow_noise;
    struct PerlinNoise perlin_noise;
    struct RidgedFractalNoise ridged_fractal_noise;
    struct VoronoiNoise voronoi_noise;
    struct WhiteNoise white_noise;
    struct SphereNoise sphere_noise;
  };
};

#endif  // CNOISE_H
