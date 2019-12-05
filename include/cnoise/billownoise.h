#pragma once
#ifndef BILLOW_NOISE_H
#define BILLOW_NOISE_H

#include "common.h"

#define DEFAULT_BILLOW_FREQUENCY 1.0
#define DEFAULT_BILLOW_LACUNARITY 2.0
#define DEFAULT_BILLOW_PERSISTENCE 0.5
#define DEFAULT_BILLOW_OCTAVE_COUNT 6
#define DEFAULT_BILLOW_SEED 0
#define DEFAULT_BILLOW_POSITION_X 0.0
#define DEFAULT_BILLOW_POSITION_Y 0.0
#define DEFAULT_BILLOW_POSITION_Z 0.0
#define DEFAULT_BILLOW_STEP 0.01
#define DEFAULT_BILLOW_PARALLEL false
#define DEFAULT_BILLOW_QUALITY QUALITY_STANDARD

struct BillowNoise {
  float frequency;
  float lacunarity;
  float persistence;
  unsigned char octave_count;
  int seed;
  float position[3];
  float step;
  bool parallel;
  float *(*billow_func)(struct PerlinNoise *, size_t, size_t, size_t);
  enum NoiseQuality noise_quality;
};

static inline void billow_noise_init(struct BillowNoise *billow_noise) {
  billow_noise->frequency = DEFAULT_BILLOW_FREQUENCY;
  billow_noise->lacunarity = DEFAULT_BILLOW_LACUNARITY;
  billow_noise->persistence = DEFAULT_BILLOW_PERSISTENCE;
  billow_noise->octave_count = DEFAULT_BILLOW_OCTAVE_COUNT;
  billow_noise->seed = DEFAULT_BILLOW_SEED;
  billow_noise->noise_quality = DEFAULT_BILLOW_QUALITY;
}

static inline float billow_noise_eval_3d(struct BillowNoise *billow_noise, float x, float y, float z) {
  float value = 0.0;
  float signal = 0.0;
  float cur_persistence = 1.0;
  float nx, ny, nz;
  int curSeed;

  x *= billow_noise->frequency;
  y *= billow_noise->frequency;
  z *= billow_noise->frequency;

  for (int cur_octave = 0; cur_octave < billow_noise->octave_count; cur_octave++) {
    nx = make_int_32_range(x);
    ny = make_int_32_range(y);
    nz = make_int_32_range(z);

    curSeed = (billow_noise->seed + cur_octave) & 0xffffffff;
    signal = gradient_coherent_noise_3d(nx, ny, nz, curSeed, billow_noise->noise_quality);
    signal = 2.0 * fabs(signal) - 1.0;
    value += signal * cur_persistence;

    x *= billow_noise->lacunarity;
    y *= billow_noise->lacunarity;
    z *= billow_noise->lacunarity;
    cur_persistence *= billow_noise->persistence;
  }

  value += 0.5;

  return value;
}

#endif  // BILLOW_NOISE_H
