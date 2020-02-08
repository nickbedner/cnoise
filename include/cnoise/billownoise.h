#pragma once
#ifndef BILLOW_NOISE_H
#define BILLOW_NOISE_H

#include "noisecommon.h"

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
  float *(*billow_func)(struct BillowNoise *, size_t, size_t, size_t);
  enum NoiseQuality noise_quality;
};

static inline float *billow_noise_eval_1d(struct BillowNoise *billow_noise, size_t x_size);
static inline float *billow_noise_eval_2d(struct BillowNoise *billow_noise, size_t x_size, size_t y_size);
static inline float *billow_noise_eval_3d(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float billow_noise_eval_3d_single(struct BillowNoise *billow_noise);
static inline float *billow_noise_eval_3d_fallback(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *billow_noise_eval_3d_sse2(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *billow_noise_eval_3d_sse4_1(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *billow_noise_eval_3d_avx(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *billow_noise_eval_3d_avx2(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *billow_noise_eval_3d_avx512(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size);

static inline void billow_noise_init(struct BillowNoise *billow_noise) {
  billow_noise->frequency = DEFAULT_BILLOW_FREQUENCY;
  billow_noise->lacunarity = DEFAULT_BILLOW_LACUNARITY;
  billow_noise->persistence = DEFAULT_BILLOW_PERSISTENCE;
  billow_noise->octave_count = DEFAULT_BILLOW_OCTAVE_COUNT;
  billow_noise->seed = DEFAULT_BILLOW_SEED;
  billow_noise->noise_quality = DEFAULT_BILLOW_QUALITY;
  billow_noise->position[0] = DEFAULT_BILLOW_POSITION_X;
  billow_noise->position[1] = DEFAULT_BILLOW_POSITION_Y;
  billow_noise->position[2] = DEFAULT_BILLOW_POSITION_X;
  billow_noise->step = DEFAULT_BILLOW_STEP;
  billow_noise->parallel = DEFAULT_BILLOW_PARALLEL;

  switch (detect_simd_support()) {
#ifdef ARCH_32_64
    case SIMD_AVX512F:
      billow_noise->billow_func = &billow_noise_eval_3d_fallback;
      break;
    case SIMD_AVX2:
      billow_noise->billow_func = &billow_noise_eval_3d_avx2;
      break;
    case SIMD_AVX:
      billow_noise->billow_func = &billow_noise_eval_3d_avx;
      break;
    case SIMD_SSE4_1:
      billow_noise->billow_func = &billow_noise_eval_3d_sse4_1;
      break;
    case SIMD_SSE2:
      billow_noise->billow_func = &billow_noise_eval_3d_sse2;
      break;
#else
    case SIMD_NEON:
      billow_noise->billow_func = &billow_noise_eval_3d_fallback;
      break;
#endif
    default:
      billow_noise->billow_func = &billow_noise_eval_3d_fallback;
      break;
  }
}

static inline float *billow_noise_eval_1d(struct BillowNoise *billow_noise, size_t x_size) {
  return billow_noise->billow_func(billow_noise, x_size, 1, 1);
}

static inline float *billow_noise_eval_2d(struct BillowNoise *billow_noise, size_t x_size, size_t y_size) {
  return billow_noise->billow_func(billow_noise, x_size, y_size, 1);
}

static inline float *billow_noise_eval_3d(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size) {
  return billow_noise->billow_func(billow_noise, x_size, y_size, z_size);
}

static inline float billow_noise_eval_3d_single(struct BillowNoise *billow_noise) {
  float value = 0.0;
  float cur_persistence = 1.0;

  float x = billow_noise->position[0] * billow_noise->frequency;
  float y = billow_noise->position[1] * billow_noise->frequency;
  float z = billow_noise->position[2] * billow_noise->frequency;

  for (int cur_octave = 0; cur_octave < billow_noise->octave_count; cur_octave++) {
    float nx = make_int_32_range(x);
    float ny = make_int_32_range(y);
    float nz = make_int_32_range(z);

    int cur_seed = (billow_noise->seed + cur_octave) & 0xffffffff;
    float signal = gradient_coherent_noise_3d(nx, ny, nz, cur_seed, billow_noise->noise_quality);
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

static inline float *billow_noise_eval_3d_fallback(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(float), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (billow_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim++) {
        float value = 0.0;
        float cur_persistence = 1.0;

        float x = (billow_noise->position[0] * billow_noise->frequency) + (x_dim * billow_noise->step);
        float y = (billow_noise->position[1] * billow_noise->frequency) + (y_dim * billow_noise->step);
        float z = (billow_noise->position[2] * billow_noise->frequency) + (z_dim * billow_noise->step);

        for (int cur_octave = 0; cur_octave < billow_noise->octave_count; cur_octave++) {
          float nx = make_int_32_range(x);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (billow_noise->seed + cur_octave) & 0xffffffff;
          float signal = gradient_coherent_noise_3d(nx, ny, nz, cur_seed, billow_noise->noise_quality);
          signal = 2.0 * fabs(signal) - 1.0;
          value += signal * cur_persistence;

          x *= billow_noise->lacunarity;
          y *= billow_noise->lacunarity;
          z *= billow_noise->lacunarity;

          cur_persistence *= billow_noise->persistence;
        }

        value += 0.5;
        *(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size)))) = value;
      }
    }
  }

  return noise_set;
}

#ifdef ARCH_32_64
static inline float *billow_noise_eval_3d_sse2(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m128), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (billow_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 4) {
        __m128 value = _mm_set1_ps(0.0);
        float cur_persistence = 1.0;

        __m128 x_vec = _mm_add_ps(_mm_set1_ps(billow_noise->position[0]), _mm_mul_ps(_mm_set_ps(x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm_set1_ps(billow_noise->step * billow_noise->frequency)));
        float y = (billow_noise->position[1] * billow_noise->frequency) + (y_dim * billow_noise->step);
        float z = (billow_noise->position[2] * billow_noise->frequency) + (z_dim * billow_noise->step);

        for (int cur_octave = 0; cur_octave < billow_noise->octave_count; cur_octave++) {
          __m128 nx = make_int_32_range_sse2(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (billow_noise->seed + cur_octave) & 0xffffffff;
          __m128 signal = gradient_coherent_noise_3d_sse2(nx, ny, nz, cur_seed, billow_noise->noise_quality);
          signal = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0), _mm_andnot_ps(_mm_set1_ps(-0.0), signal)), _mm_set1_ps(1.0));
          value = _mm_add_ps(value, _mm_mul_ps(signal, _mm_set1_ps(cur_persistence)));

          x_vec = _mm_mul_ps(x_vec, _mm_set1_ps(billow_noise->lacunarity));
          y *= billow_noise->lacunarity;
          z *= billow_noise->lacunarity;

          cur_persistence *= billow_noise->persistence;
        }

        value = _mm_add_ps(value, _mm_set1_ps(0.5));
        _mm_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), value);
      }
    }
  }
  return noise_set;
}

static inline float *billow_noise_eval_3d_sse4_1(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m128), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (billow_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 4) {
        __m128 value = _mm_set1_ps(0.0);
        float cur_persistence = 1.0;

        __m128 x_vec = _mm_add_ps(_mm_set1_ps(billow_noise->position[0]), _mm_mul_ps(_mm_set_ps(x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm_set1_ps(billow_noise->step * billow_noise->frequency)));
        float y = (billow_noise->position[1] * billow_noise->frequency) + (y_dim * billow_noise->step);
        float z = (billow_noise->position[2] * billow_noise->frequency) + (z_dim * billow_noise->step);

        for (int cur_octave = 0; cur_octave < billow_noise->octave_count; cur_octave++) {
          __m128 nx = make_int_32_range_sse2(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (billow_noise->seed + cur_octave) & 0xffffffff;
          __m128 signal = gradient_coherent_noise_3d_sse4_1(nx, ny, nz, cur_seed, billow_noise->noise_quality);
          signal = _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0), _mm_andnot_ps(_mm_set1_ps(-0.0), signal)), _mm_set1_ps(1.0));
          value = _mm_add_ps(value, _mm_mul_ps(signal, _mm_set1_ps(cur_persistence)));

          x_vec = _mm_mul_ps(x_vec, _mm_set1_ps(billow_noise->lacunarity));
          y *= billow_noise->lacunarity;
          z *= billow_noise->lacunarity;

          cur_persistence *= billow_noise->persistence;
        }

        value = _mm_add_ps(value, _mm_set1_ps(0.5));
        _mm_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), value);
      }
    }
  }
  return noise_set;
}

static inline float *billow_noise_eval_3d_avx(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m256), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (billow_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 8) {
        __m256 value = _mm256_set1_ps(0.0);
        float cur_persistence = 1.0;

        __m256 x_vec = _mm256_add_ps(_mm256_set1_ps(billow_noise->position[0]), _mm256_mul_ps(_mm256_set_ps(x_dim + 7.0, x_dim + 6.0, x_dim + 5.0, x_dim + 4.0, x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm256_set1_ps(billow_noise->step * billow_noise->frequency)));
        float y = (billow_noise->position[1] * billow_noise->frequency) + (y_dim * billow_noise->step);
        float z = (billow_noise->position[2] * billow_noise->frequency) + (z_dim * billow_noise->step);

        for (int cur_octave = 0; cur_octave < billow_noise->octave_count; cur_octave++) {
          __m256 nx = make_int_32_range_avx(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (billow_noise->seed + cur_octave) & 0xffffffff;
          __m256 signal = gradient_coherent_noise_3d_avx(nx, ny, nz, cur_seed, billow_noise->noise_quality);
          signal = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0), _mm256_andnot_ps(_mm256_set1_ps(-0.0), signal)), _mm256_set1_ps(1.0));
          value = _mm256_add_ps(value, _mm256_mul_ps(signal, _mm256_set1_ps(cur_persistence)));

          x_vec = _mm256_mul_ps(x_vec, _mm256_set1_ps(billow_noise->lacunarity));
          y *= billow_noise->lacunarity;
          z *= billow_noise->lacunarity;

          cur_persistence *= billow_noise->persistence;
        }

        value = _mm256_add_ps(value, _mm256_set1_ps(0.5));
        _mm256_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), value);
      }
    }
  }
  return noise_set;
}

static inline float *billow_noise_eval_3d_avx2(struct BillowNoise *billow_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m256), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (billow_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 8) {
        __m256 value = _mm256_set1_ps(0.0);
        float cur_persistence = 1.0;

        __m256 x_vec = _mm256_add_ps(_mm256_set1_ps(billow_noise->position[0]), _mm256_mul_ps(_mm256_set_ps(x_dim + 7.0, x_dim + 6.0, x_dim + 5.0, x_dim + 4.0, x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm256_set1_ps(billow_noise->step * billow_noise->frequency)));
        float y = (billow_noise->position[1] * billow_noise->frequency) + (y_dim * billow_noise->step);
        float z = (billow_noise->position[2] * billow_noise->frequency) + (z_dim * billow_noise->step);

        for (int cur_octave = 0; cur_octave < billow_noise->octave_count; cur_octave++) {
          __m256 nx = make_int_32_range_avx(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (billow_noise->seed + cur_octave) & 0xffffffff;
          __m256 signal = gradient_coherent_noise_3d_avx2(nx, ny, nz, cur_seed, billow_noise->noise_quality);
          signal = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0), _mm256_andnot_ps(_mm256_set1_ps(-0.0), signal)), _mm256_set1_ps(1.0));
          value = _mm256_add_ps(value, _mm256_mul_ps(signal, _mm256_set1_ps(cur_persistence)));

          x_vec = _mm256_mul_ps(x_vec, _mm256_set1_ps(billow_noise->lacunarity));
          y *= billow_noise->lacunarity;
          z *= billow_noise->lacunarity;

          cur_persistence *= billow_noise->persistence;
        }

        value = _mm256_add_ps(value, _mm256_set1_ps(0.5));
        _mm256_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), value);
      }
    }
  }
  return noise_set;
}

#endif

#endif  // BILLOW_NOISE_H
