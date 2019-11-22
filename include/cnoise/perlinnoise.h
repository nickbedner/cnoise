#pragma once
#ifndef PERLIN_NOISE_H
#define PERLIN_NOISE_H

#include "common.h"

#define DEFAULT_PERLIN_FREQUENCY 1.0
#define DEFAULT_PERLIN_LACUNARITY 2.0
#define DEFAULT_PERLIN_PERSISTENCE 0.5
#define DEFAULT_PERLIN_OCTAVE_COUNT 1
#define DEFAULT_PERLIN_SEED 0
#define DEFAULT_PERLIN_POSITION_X 0.0
#define DEFAULT_PERLIN_POSITION_Y 0.0
#define DEFAULT_PERLIN_POSITION_Z 0.0
#define DEFAULT_PERLIN_STEP 0.001
#define DEFAULT_PERLIN_PARALLEL false
#define DEFAULT_PERLIN_QUALITY QUALITY_STANDARD

struct PerlinNoise {
  float frequency;
  float lacunarity;
  float persistence;
  unsigned char octave_count;
  int seed;
  float position[3];
  float step;
  bool parallel;
  float *(*perlin_func)(struct PerlinNoise *, size_t, size_t, size_t);
  enum NoiseQuality noise_quality;
};

static inline float *perlin_noise_eval_3d_avx2(struct PerlinNoise *perlin_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *perlin_noise_eval_3d_fallback(struct PerlinNoise *perlin_noise, size_t x_size, size_t y_size, size_t z_size);

static inline void perlin_noise_init(struct PerlinNoise *perlin_noise) {
  perlin_noise->frequency = DEFAULT_PERLIN_FREQUENCY;
  perlin_noise->lacunarity = DEFAULT_PERLIN_LACUNARITY;
  perlin_noise->persistence = DEFAULT_PERLIN_PERSISTENCE;
  perlin_noise->octave_count = DEFAULT_PERLIN_OCTAVE_COUNT;
  perlin_noise->seed = DEFAULT_PERLIN_SEED;
  perlin_noise->noise_quality = DEFAULT_PERLIN_QUALITY;
  perlin_noise->position[0] = DEFAULT_PERLIN_POSITION_X;
  perlin_noise->position[1] = DEFAULT_PERLIN_POSITION_Y;
  perlin_noise->position[2] = DEFAULT_PERLIN_POSITION_Z;
  perlin_noise->step = DEFAULT_PERLIN_STEP;

  switch (detect_simd_support()) {
    case SIMD_AVX512F:
      break;
    case SIMD_AVX2:
      perlin_noise->perlin_func = &perlin_noise_eval_3d_avx2;
      break;
    case SIMD_AVX:
      break;
    case SIMD_SSE4_1:
      break;
    case SIMD_SSE2:
      break;
    case SIMD_FALLBACK:
      perlin_noise->perlin_func = &perlin_noise_eval_3d_fallback;
      break;
  }
}

static inline float *perlin_noise_eval_1d(struct PerlinNoise *perlin_noise, size_t x_size) {
  return perlin_noise->perlin_func(perlin_noise, x_size, 1, 1);
}

static inline float *perlin_noise_eval_2d(struct PerlinNoise *perlin_noise, size_t x_size, size_t y_size) {
  return perlin_noise->perlin_func(perlin_noise, x_size, y_size, 1);
}

static inline float *perlin_noise_eval_3d(struct PerlinNoise *perlin_noise, size_t x_size, size_t y_size, size_t z_size) {
  return perlin_noise->perlin_func(perlin_noise, x_size, y_size, z_size);
}

// Note: Every other value seems to be working and reversed
static inline float *perlin_noise_eval_3d_avx2(struct PerlinNoise *perlin_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *values = _aligned_malloc(sizeof(float) * x_size * y_size * z_size, sizeof(__m256));
#pragma omp parallel for collapse(3) if (perlin_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 8) {
        __m256 value = _mm256_set1_ps(0.0);
        __m256 signal = _mm256_set1_ps(0.0);
        float cur_persistence = 1.0;
        __m256 nx;
        float ny, nz;
        int cur_seed;

        __m256 x_vec = _mm256_add_ps(_mm256_set1_ps(perlin_noise->position[0]), _mm256_mul_ps(_mm256_set_ps(x_dim, x_dim + 1.0, x_dim + 2.0, x_dim + 3.0, x_dim + 4.0, x_dim + 5.0, x_dim + 6.0, x_dim + 7.0), _mm256_set1_ps(perlin_noise->step * perlin_noise->frequency)));
        float y = perlin_noise->position[1] + (y_dim * perlin_noise->step * perlin_noise->frequency);
        float z = perlin_noise->position[2] + (z_dim * perlin_noise->step * perlin_noise->frequency);

        for (int cur_octave = 0; cur_octave < perlin_noise->octave_count; cur_octave++) {
          // Note: Don't need this only need to check first and final
          // Check first and last if good continue else make in range
          make_int_32_range_vec_256(&nx, x_vec);
          ny = make_int_32_range(y);
          nz = make_int_32_range(z);

          cur_seed = (perlin_noise->seed + cur_octave) & 0xffffffff;
          gradient_coherent_noise_3d_vec_256(&signal, nx, ny, nz, cur_seed, perlin_noise->noise_quality);
          const __m256 cur_persistence_scalar = _mm256_set1_ps(cur_persistence);
          value = _mm256_add_ps(value, _mm256_mul_ps(signal, cur_persistence_scalar));

          const __m256 lacunarity_scalar = _mm256_set1_ps(perlin_noise->lacunarity);
          x_vec = _mm256_mul_ps(x_vec, lacunarity_scalar);
          y *= perlin_noise->lacunarity;
          z *= perlin_noise->lacunarity;

          cur_persistence = cur_persistence * perlin_noise->persistence;
        }

        _mm256_store_ps(values + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), value);
      }
    }
  }
  return values;
}

static inline float *perlin_noise_eval_3d_fallback(struct PerlinNoise *perlin_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *values = _aligned_malloc(sizeof(float) * x_size * y_size * z_size, sizeof(float) * 8);
#pragma omp parallel for collapse(3) if (perlin_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim++) {
        float value = 0.0;
        float cur_persistence = 1.0;

        float x = (perlin_noise->position[0] * perlin_noise->frequency) + (x_dim * perlin_noise->step);
        float y = (perlin_noise->position[1] * perlin_noise->frequency) + (y_dim * perlin_noise->step);
        float z = (perlin_noise->position[2] * perlin_noise->frequency) + (z_dim * perlin_noise->step);

        for (int cur_octave = 0; cur_octave < perlin_noise->octave_count; cur_octave++) {
          float nx = make_int_32_range(x);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (perlin_noise->seed + cur_octave) & 0xffffffff;
          float signal = gradient_coherent_noise_3d(nx, ny, nz, cur_seed, perlin_noise->noise_quality);
          value += signal * cur_persistence;

          x *= perlin_noise->lacunarity;
          y *= perlin_noise->lacunarity;
          z *= perlin_noise->lacunarity;
          cur_persistence *= perlin_noise->persistence;
        }

        *(values + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size)))) = value;
      }
    }
  }
  return values;
}

#endif  // PERLIN_NOISE_H
