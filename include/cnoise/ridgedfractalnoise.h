#pragma once
#ifndef RIDGID_FRACTAL_NOISE_H
#define RIDGID_FRACTAL_NOISE_H

#include "noisecommon.h"

#define DEFAULT_RIDGED_FREQUENCY 1.0
#define DEFAULT_RIDGED_LACUNARITY 2.0
#define DEFAULT_RIDGED_OCTAVE_COUNT 6
#define DEFAULT_RIDGED_SEED 0
#define DEFAULT_RIDGED_POSITION_X 0.0
#define DEFAULT_RIDGED_POSITION_Y 0.0
#define DEFAULT_RIDGED_POSITION_Z 0.0
#define DEFAULT_RIDGED_STEP 0.01
#define DEFAULT_RIDGED_PARALLEL false
#define DEFAULT_RIDGED_QUALITY QUALITY_STANDARD
#define RIDGED_MAX_OCTAVE 30

struct RidgedFractalNoise {
  float frequency;
  float lacunarity;
  float spectral_weights[RIDGED_MAX_OCTAVE];
  unsigned char octave_count;
  int seed;
  float position[3];
  float step;
  bool parallel;
  float *(*ridged_func)(struct RidgedFractalNoise *, size_t, size_t, size_t);
  enum NoiseQuality noise_quality;
};

static inline float *ridged_fractal_noise_eval_1d(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size);
static inline float *ridged_fractal_noise_eval_2d(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size);
static inline float *ridged_fractal_noise_eval_3d(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float ridged_fractal_noise_eval_3d_single(struct RidgedFractalNoise *ridged_fractal_noise);
static inline float *ridged_fractal_noise_eval_3d_fallback(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *ridged_fractal_noise_eval_3d_sse2(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *ridged_fractal_noise_eval_3d_sse4_1(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *ridged_fractal_noise_eval_3d_avx(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *ridged_fractal_noise_eval_3d_avx2(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *ridged_fractal_noise_eval_3d_avx512(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size);

// TODO: Calculate this to remove lookup
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
  ridged_fractal_noise->position[0] = DEFAULT_PERLIN_POSITION_X;
  ridged_fractal_noise->position[1] = DEFAULT_PERLIN_POSITION_Y;
  ridged_fractal_noise->position[2] = DEFAULT_PERLIN_POSITION_Z;
  ridged_fractal_noise->step = DEFAULT_PERLIN_STEP;
  ridged_fractal_noise->parallel = DEFAULT_PERLIN_PARALLEL;

  ridged_fractal_noise_calc_spectral_weights(ridged_fractal_noise);

  switch (detect_simd_support()) {
#ifdef ARCH_32_64
    case SIMD_AVX512F:
      ridged_fractal_noise->ridged_func = &ridged_fractal_noise_eval_3d_fallback;
      break;
    case SIMD_AVX2:
      ridged_fractal_noise->ridged_func = &ridged_fractal_noise_eval_3d_avx2;
      break;
    case SIMD_AVX:
      ridged_fractal_noise->ridged_func = &ridged_fractal_noise_eval_3d_avx;
      break;
    case SIMD_SSE4_1:
      ridged_fractal_noise->ridged_func = &ridged_fractal_noise_eval_3d_sse4_1;
      break;
    case SIMD_SSE2:
      ridged_fractal_noise->ridged_func = &ridged_fractal_noise_eval_3d_sse2;
      break;
#else
    case SIMD_NEON:
      ridged_fractal_noise->ridged_func = &ridged_fractal_noise_eval_3d_fallback;
      break;
#endif
    default:
      ridged_fractal_noise->ridged_func = &ridged_fractal_noise_eval_3d_fallback;
      break;
  }
}

static inline float *ridged_fractal_noise_eval_1d(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size) {
  return ridged_fractal_noise->ridged_func(ridged_fractal_noise, x_size, 1, 1);
}

static inline float *ridged_fractal_noise_eval_2d(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size) {
  return ridged_fractal_noise->ridged_func(ridged_fractal_noise, x_size, y_size, 1);
}

static inline float *ridged_fractal_noise_eval_3d(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size) {
  return ridged_fractal_noise->ridged_func(ridged_fractal_noise, x_size, y_size, z_size);
}

static inline float ridged_fractal_noise_eval_3d_single(struct RidgedFractalNoise *ridged_fractal_noise) {
  float x = ridged_fractal_noise->position[0] * ridged_fractal_noise->frequency;
  float y = ridged_fractal_noise->position[1] * ridged_fractal_noise->frequency;
  float z = ridged_fractal_noise->position[2] * ridged_fractal_noise->frequency;

  float value = 0.0;
  float weight = 1.0;

  float offset = 1.0;
  float gain = 2.0;

  for (int cur_octave = 0; cur_octave < ridged_fractal_noise->octave_count; cur_octave++) {
    float nx = make_int_32_range(x);
    float ny = make_int_32_range(y);
    float nz = make_int_32_range(z);

    int cur_seed = (ridged_fractal_noise->seed + cur_octave) & 0x7fffffff;
    float signal = gradient_coherent_noise_3d(nx, ny, nz, cur_seed, ridged_fractal_noise->noise_quality);

    signal = fabs(signal);
    signal = offset - signal;

    signal *= signal;
    signal *= weight;

    weight = signal * gain;
    if (weight > 1.0)
      weight = 1.0;
    else if (weight < 0.0)
      weight = 0.0;

    value += (signal * ridged_fractal_noise->spectral_weights[cur_octave]);

    x *= ridged_fractal_noise->lacunarity;
    y *= ridged_fractal_noise->lacunarity;
    z *= ridged_fractal_noise->lacunarity;
  }

  return (value * 1.25) - 1.0;
}

static inline float *ridged_fractal_noise_eval_3d_fallback(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(float), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (ridged_fractal_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim++) {
        float x = (ridged_fractal_noise->position[0] * ridged_fractal_noise->frequency) + (x_dim * ridged_fractal_noise->step);
        float y = (ridged_fractal_noise->position[1] * ridged_fractal_noise->frequency) + (y_dim * ridged_fractal_noise->step);
        float z = (ridged_fractal_noise->position[2] * ridged_fractal_noise->frequency) + (z_dim * ridged_fractal_noise->step);

        float value = 0.0;
        float weight = 1.0;

        float offset = 1.0;
        float gain = 2.0;

        for (int cur_octave = 0; cur_octave < ridged_fractal_noise->octave_count; cur_octave++) {
          float nx = make_int_32_range(x);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (ridged_fractal_noise->seed + cur_octave) & 0x7fffffff;
          float signal = gradient_coherent_noise_3d(nx, ny, nz, cur_seed, ridged_fractal_noise->noise_quality);

          signal = fabs(signal);
          signal = offset - signal;

          signal *= signal;
          signal *= weight;

          weight = signal * gain;
          if (weight > 1.0)
            weight = 1.0;
          else if (weight < 0.0)
            weight = 0.0;

          value += (signal * ridged_fractal_noise->spectral_weights[cur_octave]);

          x *= ridged_fractal_noise->lacunarity;
          y *= ridged_fractal_noise->lacunarity;
          z *= ridged_fractal_noise->lacunarity;
        }
        *(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size)))) = (value * 1.25) - 1.0;
      }
    }
  }
  return noise_set;
}

#ifdef ARCH_32_64
static inline float *ridged_fractal_noise_eval_3d_sse2(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m128), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (ridged_fractal_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 4) {
        __m128 x_vec = _mm_add_ps(_mm_set1_ps(ridged_fractal_noise->position[0]), _mm_mul_ps(_mm_set_ps(x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm_set1_ps(ridged_fractal_noise->step * ridged_fractal_noise->frequency)));
        float y = (ridged_fractal_noise->position[1] * ridged_fractal_noise->frequency) + (y_dim * ridged_fractal_noise->step);
        float z = (ridged_fractal_noise->position[2] * ridged_fractal_noise->frequency) + (z_dim * ridged_fractal_noise->step);

        __m128 value = _mm_set1_ps(0.0);
        __m128 weight = _mm_set1_ps(1.0);

        float offset = 1.0;
        float gain = 2.0;

        for (int cur_octave = 0; cur_octave < ridged_fractal_noise->octave_count; cur_octave++) {
          __m128 nx = make_int_32_range_sse2(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (ridged_fractal_noise->seed + cur_octave) & 0x7fffffff;
          __m128 signal = gradient_coherent_noise_3d_sse2(nx, ny, nz, cur_seed, ridged_fractal_noise->noise_quality);

          signal = _mm_andnot_ps(_mm_set1_ps(-0.0), signal);
          signal = _mm_sub_ps(_mm_set1_ps(offset), signal);

          signal = _mm_mul_ps(signal, signal);
          signal = _mm_mul_ps(signal, weight);

          weight = _mm_mul_ps(signal, _mm_set1_ps(gain));

          __m128 weight_gt_mask = _mm_cmpgt_ps(weight, _mm_set1_ps(1.0));
          weight = _mm_or_ps(_mm_and_ps(_mm_set1_ps(1.0), weight_gt_mask), _mm_andnot_ps(weight_gt_mask, weight));
          __m128 weight_lt_mask = _mm_cmplt_ps(weight, _mm_setzero_ps());
          weight = _mm_or_ps(_mm_and_ps(_mm_set1_ps(1.0), weight_lt_mask), _mm_andnot_ps(weight_lt_mask, weight));

          value = _mm_add_ps(value, _mm_mul_ps(signal, _mm_set1_ps(ridged_fractal_noise->spectral_weights[cur_octave])));

          x_vec = _mm_mul_ps(x_vec, _mm_set1_ps(ridged_fractal_noise->lacunarity));
          y *= ridged_fractal_noise->lacunarity;
          z *= ridged_fractal_noise->lacunarity;
        }
        _mm_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), _mm_sub_ps(_mm_mul_ps(value, _mm_set1_ps(1.25)), _mm_set1_ps(1.0)));
      }
    }
  }
  return noise_set;
}

static inline float *ridged_fractal_noise_eval_3d_sse4_1(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m128), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (ridged_fractal_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 4) {
        __m128 x_vec = _mm_add_ps(_mm_set1_ps(ridged_fractal_noise->position[0]), _mm_mul_ps(_mm_set_ps(x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm_set1_ps(ridged_fractal_noise->step * ridged_fractal_noise->frequency)));
        float y = (ridged_fractal_noise->position[1] * ridged_fractal_noise->frequency) + (y_dim * ridged_fractal_noise->step);
        float z = (ridged_fractal_noise->position[2] * ridged_fractal_noise->frequency) + (z_dim * ridged_fractal_noise->step);

        __m128 value = _mm_set1_ps(0.0);
        __m128 weight = _mm_set1_ps(1.0);

        float offset = 1.0;
        float gain = 2.0;

        for (int cur_octave = 0; cur_octave < ridged_fractal_noise->octave_count; cur_octave++) {
          __m128 nx = make_int_32_range_sse2(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (ridged_fractal_noise->seed + cur_octave) & 0x7fffffff;
          __m128 signal = gradient_coherent_noise_3d_sse4_1(nx, ny, nz, cur_seed, ridged_fractal_noise->noise_quality);

          signal = _mm_andnot_ps(_mm_set1_ps(-0.0), signal);
          signal = _mm_sub_ps(_mm_set1_ps(offset), signal);

          signal = _mm_mul_ps(signal, signal);
          signal = _mm_mul_ps(signal, weight);

          weight = _mm_mul_ps(signal, _mm_set1_ps(gain));

          weight = _mm_blendv_ps(weight, _mm_set1_ps(1.0), _mm_cmpgt_ps(weight, _mm_set1_ps(1.0)));
          weight = _mm_blendv_ps(weight, _mm_setzero_ps(), _mm_cmplt_ps(weight, _mm_setzero_ps()));

          value = _mm_add_ps(value, _mm_mul_ps(signal, _mm_set1_ps(ridged_fractal_noise->spectral_weights[cur_octave])));

          x_vec = _mm_mul_ps(x_vec, _mm_set1_ps(ridged_fractal_noise->lacunarity));
          y *= ridged_fractal_noise->lacunarity;
          z *= ridged_fractal_noise->lacunarity;
        }
        _mm_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), _mm_sub_ps(_mm_mul_ps(value, _mm_set1_ps(1.25)), _mm_set1_ps(1.0)));
      }
    }
  }
  return noise_set;
}

static inline float *ridged_fractal_noise_eval_3d_avx(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m256), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (ridged_fractal_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 8) {
        __m256 x_vec = _mm256_add_ps(_mm256_set1_ps(ridged_fractal_noise->position[0]), _mm256_mul_ps(_mm256_set_ps(x_dim + 7.0, x_dim + 6.0, x_dim + 5.0, x_dim + 4.0, x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm256_set1_ps(ridged_fractal_noise->step * ridged_fractal_noise->frequency)));
        float y = (ridged_fractal_noise->position[1] * ridged_fractal_noise->frequency) + (y_dim * ridged_fractal_noise->step);
        float z = (ridged_fractal_noise->position[2] * ridged_fractal_noise->frequency) + (z_dim * ridged_fractal_noise->step);

        __m256 value = _mm256_set1_ps(0.0);
        __m256 weight = _mm256_set1_ps(1.0);

        float offset = 1.0;
        float gain = 2.0;

        for (int cur_octave = 0; cur_octave < ridged_fractal_noise->octave_count; cur_octave++) {
          __m256 nx = make_int_32_range_avx(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (ridged_fractal_noise->seed + cur_octave) & 0x7fffffff;
          __m256 signal = gradient_coherent_noise_3d_avx(nx, ny, nz, cur_seed, ridged_fractal_noise->noise_quality);

          signal = _mm256_andnot_ps(_mm256_set1_ps(-0.0), signal);
          signal = _mm256_sub_ps(_mm256_set1_ps(offset), signal);

          signal = _mm256_mul_ps(signal, signal);
          signal = _mm256_mul_ps(signal, weight);

          weight = _mm256_mul_ps(signal, _mm256_set1_ps(gain));

          weight = _mm256_blendv_ps(weight, _mm256_set1_ps(1.0), _mm256_cmp_ps(weight, _mm256_set1_ps(1.0), _CMP_GT_OQ));
          weight = _mm256_blendv_ps(weight, _mm256_setzero_ps(), _mm256_cmp_ps(weight, _mm256_setzero_ps(), _CMP_LT_OQ));

          value = _mm256_add_ps(value, _mm256_mul_ps(signal, _mm256_set1_ps(ridged_fractal_noise->spectral_weights[cur_octave])));

          x_vec = _mm256_mul_ps(x_vec, _mm256_set1_ps(ridged_fractal_noise->lacunarity));
          y *= ridged_fractal_noise->lacunarity;
          z *= ridged_fractal_noise->lacunarity;
        }
        _mm256_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), _mm256_sub_ps(_mm256_mul_ps(value, _mm256_set1_ps(1.25)), _mm256_set1_ps(1.0)));
      }
    }
  }
  return noise_set;
}

static inline float *ridged_fractal_noise_eval_3d_avx2(struct RidgedFractalNoise *ridged_fractal_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m256), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (ridged_fractal_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 8) {
        __m256 x_vec = _mm256_add_ps(_mm256_set1_ps(ridged_fractal_noise->position[0]), _mm256_mul_ps(_mm256_set_ps(x_dim + 7.0, x_dim + 6.0, x_dim + 5.0, x_dim + 4.0, x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm256_set1_ps(ridged_fractal_noise->step * ridged_fractal_noise->frequency)));
        float y = (ridged_fractal_noise->position[1] * ridged_fractal_noise->frequency) + (y_dim * ridged_fractal_noise->step);
        float z = (ridged_fractal_noise->position[2] * ridged_fractal_noise->frequency) + (z_dim * ridged_fractal_noise->step);

        __m256 value = _mm256_set1_ps(0.0);
        __m256 weight = _mm256_set1_ps(1.0);

        float offset = 1.0;
        float gain = 2.0;

        for (int cur_octave = 0; cur_octave < ridged_fractal_noise->octave_count; cur_octave++) {
          __m256 nx = make_int_32_range_avx(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (ridged_fractal_noise->seed + cur_octave) & 0x7fffffff;
          __m256 signal = gradient_coherent_noise_3d_avx2(nx, ny, nz, cur_seed, ridged_fractal_noise->noise_quality);

          signal = _mm256_andnot_ps(_mm256_set1_ps(-0.0), signal);
          signal = _mm256_sub_ps(_mm256_set1_ps(offset), signal);

          signal = _mm256_mul_ps(signal, signal);
          signal = _mm256_mul_ps(signal, weight);

          weight = _mm256_mul_ps(signal, _mm256_set1_ps(gain));

          weight = _mm256_blendv_ps(weight, _mm256_set1_ps(1.0), _mm256_cmp_ps(weight, _mm256_set1_ps(1.0), _CMP_GT_OQ));
          weight = _mm256_blendv_ps(weight, _mm256_setzero_ps(), _mm256_cmp_ps(weight, _mm256_setzero_ps(), _CMP_LT_OQ));

          value = _mm256_add_ps(value, _mm256_mul_ps(signal, _mm256_set1_ps(ridged_fractal_noise->spectral_weights[cur_octave])));

          x_vec = _mm256_mul_ps(x_vec, _mm256_set1_ps(ridged_fractal_noise->lacunarity));
          y *= ridged_fractal_noise->lacunarity;
          z *= ridged_fractal_noise->lacunarity;
        }
        _mm256_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), _mm256_sub_ps(_mm256_mul_ps(value, _mm256_set1_ps(1.25)), _mm256_set1_ps(1.0)));
      }
    }
  }
  return noise_set;
}
#endif

#endif  // RIDGID_FRACTAL_NOISE_H
