#pragma once
#ifndef RIDGID_FRACTAL_NOISE_H
#define RIDGID_FRACTAL_NOISE_H

#include "common.h"

#define DEFAULT_RIDGED_FREQUENCY 1.0
#define DEFAULT_RIDGED_LACUNARITY 2.0
#define DEFAULT_RIDGED_OCTAVE_COUNT 6
#define DEFAULT_RIDGED_SEED 0
#define DEFAULT_RIDGED_QUALITY QUALITY_STANDARD
#define RIDGED_MAX_OCTAVE 30

struct RidgedFractalNoise {
  float frequency;
  float lacunarity;
  float spectral_weights[RIDGED_MAX_OCTAVE];
  unsigned char octave_count;
  int seed;
  enum NoiseQuality noise_quality;
};

static inline void ridged_fractal_noise_calc_spectral_weights(struct RidgedFractalNoise *ridged_fractal_noise) {
  float h = 1.0;

  float frequency = 1.0;
  for (int i = 0; i < RIDGED_MAX_OCTAVE; i++) {
    ridged_fractal_noise->spectral_weights[i] = pow(frequency, -h);
    frequency *= ridged_fractal_noise->lacunarity;
  }
}

static inline void ridged_fractal_noise_init(struct RidgedFractalNoise *ridged_fractal_noise) {
  ridged_fractal_noise->frequency = DEFAULT_RIDGED_FREQUENCY;
  ridged_fractal_noise->lacunarity = DEFAULT_RIDGED_LACUNARITY;
  ridged_fractal_noise->octave_count = DEFAULT_RIDGED_OCTAVE_COUNT;
  ridged_fractal_noise->seed = DEFAULT_RIDGED_SEED;
  ridged_fractal_noise->noise_quality = DEFAULT_RIDGED_QUALITY;

  ridged_fractal_noise_calc_spectral_weights(ridged_fractal_noise);
}

static inline float ridged_fractal_noise_eval_3d(struct RidgedFractalNoise *ridged_fractal_noise, float x, float y, float z) {
  x *= ridged_fractal_noise->frequency;
  y *= ridged_fractal_noise->frequency;
  z *= ridged_fractal_noise->frequency;

  float signal = 0.0;
  float value = 0.0;
  float weight = 1.0;

  float offset = 1.0;
  float gain = 2.0;

  for (int cur_octave = 0; cur_octave < ridged_fractal_noise->octave_count; cur_octave++) {
    float nx, ny, nz;
    nx = make_int_32_range(x);
    ny = make_int_32_range(y);
    nz = make_int_32_range(z);

    int cur_seed = (ridged_fractal_noise->seed + cur_octave) & 0x7fffffff;
    signal = gradient_coherent_noise_3d(nx, ny, nz, cur_seed, ridged_fractal_noise->noise_quality);

    signal = fabs(signal);
    signal = offset - signal;

    signal *= signal;
    signal *= weight;

    weight = signal * gain;
    if (weight > 1.0)
      weight = 1.0;
    if (weight < 0.0)
      weight = 0.0;

    value += (signal * ridged_fractal_noise->spectral_weights[cur_octave]);

    x *= ridged_fractal_noise->lacunarity;
    y *= ridged_fractal_noise->lacunarity;
    z *= ridged_fractal_noise->lacunarity;
  }

  return (value * 1.25) - 1.0;
}

#endif  // RIDGID_FRACTAL_NOISE_H
