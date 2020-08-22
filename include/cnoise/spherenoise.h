#pragma once
#ifndef SPHERE_NOISE_H
#define SPHERE_NOISE_H

#include "noisecommon.h"

#define DEFAULT_SPHERE_RADIUS 8.0
#define DEFAULT_SPHERE_SPHERE_ORIGIN_X 0.0
#define DEFAULT_SPHERE_SPHERE_ORIGIN_Y 0.0
#define DEFAULT_SPHERE_SPHERE_ORIGIN_Z 0.0
#define DEFAULT_SPHERE_POSITION_X 0.0
#define DEFAULT_SPHERE_POSITION_Y 0.0
#define DEFAULT_SPHERE_POSITION_Z 0.0
#define DEFAULT_SPHERE_STEP 1.0 / 256.0
#define DEFAULT_SPHERE_PARALLEL false

struct SphereNoise {
  float radius;
  float position[3];
  float sphere_origin[3];
  float step;
  bool parallel;
  float *(*sphere_func)(struct SphereNoise *, size_t, size_t, size_t);
  enum NoiseQuality noise_quality;
};

static inline float *sphere_noise_eval_1d(struct SphereNoise *sphere_noise, size_t x_size);
static inline float *sphere_noise_eval_2d(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size);
static inline float *sphere_noise_eval_3d(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float sphere_noise_eval_3d_single(struct SphereNoise *sphere_noise, float x_pos, float y_pos, float z_pos);
static inline float *sphere_noise_eval_3d_fallback(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *sphere_noise_eval_3d_sse2(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *sphere_noise_eval_3d_sse4_1(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *sphere_noise_eval_3d_avx(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *sphere_noise_eval_3d_avx2(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *sphere_noise_eval_3d_avx512(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size);

static inline void sphere_noise_init(struct SphereNoise *sphere_noise) {
  sphere_noise->radius = DEFAULT_SPHERE_RADIUS;
  sphere_noise->position[0] = DEFAULT_SPHERE_POSITION_X;
  sphere_noise->position[1] = DEFAULT_SPHERE_POSITION_Y;
  sphere_noise->position[2] = DEFAULT_SPHERE_POSITION_Z;
  sphere_noise->sphere_origin[0] = DEFAULT_SPHERE_SPHERE_ORIGIN_X;
  sphere_noise->sphere_origin[1] = DEFAULT_SPHERE_SPHERE_ORIGIN_Y;
  sphere_noise->sphere_origin[2] = DEFAULT_SPHERE_SPHERE_ORIGIN_Z;
  sphere_noise->step = DEFAULT_SPHERE_STEP;
  sphere_noise->parallel = DEFAULT_SPHERE_PARALLEL;

  switch (detect_simd_support()) {
#ifdef ARCH_32_64
    /*case SIMD_AVX512F:
      sphere_noise->sphere_func = &sphere_noise_eval_3d_fallback;
      break;
    case SIMD_AVX2:
      sphere_noise->sphere_func = &sphere_noise_eval_3d_avx2;
      break;
    case SIMD_AVX:
      sphere_noise->sphere_func = &sphere_noise_eval_3d_avx;
      break;
    case SIMD_SSE4_1:
      sphere_noise->sphere_func = &sphere_noise_eval_3d_sse4_1;
      break;
    case SIMD_SSE2:
      sphere_noise->sphere_func = &sphere_noise_eval_3d_sse2;
      break;*/
#else
    case SIMD_NEON:
      sphere_noise->sphere_func = &sphere_noise_eval_3d_fallback;
      break;
#endif
    default:
      sphere_noise->sphere_func = &sphere_noise_eval_3d_fallback;
      break;
  }
}

static inline float *sphere_noise_eval_1d(struct SphereNoise *sphere_noise, size_t x_size) {
  return sphere_noise->sphere_func(sphere_noise, x_size, 1, 1);
}

static inline float *sphere_noise_eval_2d(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size) {
  return sphere_noise->sphere_func(sphere_noise, x_size, y_size, 1);
}

static inline float *sphere_noise_eval_3d(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size) {
  return sphere_noise->sphere_func(sphere_noise, x_size, y_size, z_size);
}

static inline float sphere_noise_eval_3d_single(struct SphereNoise *sphere_noise, float x_pos, float y_pos, float z_pos) {
  float length_x = sphere_noise->position[0] + sphere_noise->sphere_origin[0] - x_pos;
  float length_y = sphere_noise->position[1] + sphere_noise->sphere_origin[1] - y_pos;
  float length_z = sphere_noise->position[2] + sphere_noise->sphere_origin[2] - z_pos;

  float magnitude = sqrtf((length_x * length_x) + (length_y * length_y) + (length_z * length_z));

  return sphere_noise->radius - magnitude;
}

static inline float *sphere_noise_eval_3d_fallback(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(float), sizeof(float) * x_size * y_size * z_size);

#pragma omp parallel for collapse(3) if (sphere_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim++) {
        float length_x = sphere_noise->position[0] + sphere_noise->sphere_origin[0] - x_dim;
        float length_y = sphere_noise->position[1] + sphere_noise->sphere_origin[1] - y_dim;
        float length_z = sphere_noise->position[2] + sphere_noise->sphere_origin[2] - z_dim;

        float magnitude = sqrtf((length_x * length_x) + (length_y * length_y) + (length_z * length_z));

        *(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size)))) = sphere_noise->radius - magnitude;
      }
    }
  }
  return noise_set;
}
/*
#ifdef ARCH_32_64
static inline float *sphere_noise_eval_3d_sse2(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m128), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (sphere_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 4) {
        __m128 x_vec = _mm_mul_ps(_mm_add_ps(_mm_set1_ps(sphere_noise->position[0]), _mm_mul_ps(_mm_set_ps(x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm_set1_ps(sphere_noise->step))), _mm_set1_ps(sphere_noise->frequency));
        float y = (sphere_noise->position[1] + (y_dim * sphere_noise->step)) * sphere_noise->frequency;
        float z = (sphere_noise->position[2] + (z_dim * sphere_noise->step)) * sphere_noise->frequency;

        __m128 value = _mm_set1_ps(0.0);
        float cur_persistence = 1.0;

        for (int cur_octave = 0; cur_octave < sphere_noise->octave_count; cur_octave++) {
          __m128 nx = make_int_32_range_sse2(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (sphere_noise->seed + cur_octave) & 0xffffffff;
          __m128 signal = gradient_coherent_noise_3d_sse2(nx, ny, nz, cur_seed, sphere_noise->noise_quality);
          value = _mm_add_ps(value, _mm_mul_ps(signal, _mm_set1_ps(cur_persistence)));

          x_vec = _mm_mul_ps(x_vec, _mm_set1_ps(sphere_noise->lacunarity));
          y *= sphere_noise->lacunarity;
          z *= sphere_noise->lacunarity;

          cur_persistence = cur_persistence * sphere_noise->persistence;
        }

        _mm_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), value);
      }
    }
  }
  return noise_set;
}

static inline float *sphere_noise_eval_3d_sse4_1(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m128), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (sphere_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 4) {
        __m128 x_vec = _mm_mul_ps(_mm_add_ps(_mm_set1_ps(sphere_noise->position[0]), _mm_mul_ps(_mm_set_ps(x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm_set1_ps(sphere_noise->step))), _mm_set1_ps(sphere_noise->frequency));
        float y = (sphere_noise->position[1] + (y_dim * sphere_noise->step)) * sphere_noise->frequency;
        float z = (sphere_noise->position[2] + (z_dim * sphere_noise->step)) * sphere_noise->frequency;

        __m128 value = _mm_set1_ps(0.0);
        float cur_persistence = 1.0;

        for (int cur_octave = 0; cur_octave < sphere_noise->octave_count; cur_octave++) {
          __m128 nx = make_int_32_range_sse2(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (sphere_noise->seed + cur_octave) & 0xffffffff;
          __m128 signal = gradient_coherent_noise_3d_sse4_1(nx, ny, nz, cur_seed, sphere_noise->noise_quality);
          value = _mm_add_ps(value, _mm_mul_ps(signal, _mm_set1_ps(cur_persistence)));

          x_vec = _mm_mul_ps(x_vec, _mm_set1_ps(sphere_noise->lacunarity));
          y *= sphere_noise->lacunarity;
          z *= sphere_noise->lacunarity;

          cur_persistence = cur_persistence * sphere_noise->persistence;
        }

        _mm_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), value);
      }
    }
  }
  return noise_set;
}

static inline float *sphere_noise_eval_3d_avx(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m256), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (sphere_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 8) {
        __m256 x_vec = _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(sphere_noise->position[0]), _mm256_mul_ps(_mm256_set_ps(x_dim + 7.0, x_dim + 6.0, x_dim + 5.0, x_dim + 4.0, x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm256_set1_ps(sphere_noise->step))), _mm256_set1_ps(sphere_noise->frequency));
        float y = (sphere_noise->position[1] + (y_dim * sphere_noise->step)) * sphere_noise->frequency;
        float z = (sphere_noise->position[2] + (z_dim * sphere_noise->step)) * sphere_noise->frequency;

        __m256 value = _mm256_set1_ps(0.0);
        float cur_persistence = 1.0;

        for (int cur_octave = 0; cur_octave < sphere_noise->octave_count; cur_octave++) {
          __m256 nx = make_int_32_range_avx(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (sphere_noise->seed + cur_octave) & 0xffffffff;
          __m256 signal = gradient_coherent_noise_3d_avx(nx, ny, nz, cur_seed, sphere_noise->noise_quality);
          value = _mm256_add_ps(value, _mm256_mul_ps(signal, _mm256_set1_ps(cur_persistence)));

          x_vec = _mm256_mul_ps(x_vec, _mm256_set1_ps(sphere_noise->lacunarity));
          y *= sphere_noise->lacunarity;
          z *= sphere_noise->lacunarity;

          cur_persistence = cur_persistence * sphere_noise->persistence;
        }

        _mm256_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), value);
      }
    }
  }
  return noise_set;
}

static inline float *sphere_noise_eval_3d_avx2(struct SphereNoise *sphere_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m256), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (sphere_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 8) {
        __m256 x_vec = _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(sphere_noise->position[0]), _mm256_mul_ps(_mm256_set_ps(x_dim + 7.0, x_dim + 6.0, x_dim + 5.0, x_dim + 4.0, x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm256_set1_ps(sphere_noise->step))), _mm256_set1_ps(sphere_noise->frequency));
        float y = (sphere_noise->position[1] + (y_dim * sphere_noise->step)) * sphere_noise->frequency;
        float z = (sphere_noise->position[2] + (z_dim * sphere_noise->step)) * sphere_noise->frequency;

        __m256 value = _mm256_set1_ps(0.0);
        float cur_persistence = 1.0;

        for (int cur_octave = 0; cur_octave < sphere_noise->octave_count; cur_octave++) {
          __m256 nx = make_int_32_range_avx(x_vec);
          float ny = make_int_32_range(y);
          float nz = make_int_32_range(z);

          int cur_seed = (sphere_noise->seed + cur_octave) & 0xffffffff;
          __m256 signal = gradient_coherent_noise_3d_avx2(nx, ny, nz, cur_seed, sphere_noise->noise_quality);
          value = _mm256_add_ps(value, _mm256_mul_ps(signal, _mm256_set1_ps(cur_persistence)));

          x_vec = _mm256_mul_ps(x_vec, _mm256_set1_ps(sphere_noise->lacunarity));
          y *= sphere_noise->lacunarity;
          z *= sphere_noise->lacunarity;

          cur_persistence = cur_persistence * sphere_noise->persistence;
        }

        _mm256_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), value);
      }
    }
  }
  return noise_set;
}
#endif
*/
#endif  // sphere_noise_H
