#pragma once
#ifndef VORONOI_NOISE_H
#define VORONOI_NOISE_H

#include "common.h"

#define DEFAULT_VORONOI_FREQUENCY 1.0
#define DEFAULT_VORONOI_DISPLACEMENT 1.0
#define DEFAULT_VORONOI_SEED 0
#define DEFAULT_VORONOI_ENABLE_DISTANCE true
#define DEFAULT_VORONOI_POSITION_X 0.0
#define DEFAULT_VORONOI_POSITION_Y 0.0
#define DEFAULT_VORONOI_POSITION_Z 0.0
#define DEFAULT_VORONOI_STEP 0.01
#define DEFAULT_VORONOI_PARALLEL false

struct VoronoiNoise {
  float frequency;
  float displacement;
  int seed;
  unsigned char enable_distance;
  float position[3];
  float step;
  bool parallel;
  float *(*voronoi_func)(struct VoronoiNoise *, size_t, size_t, size_t);
};

static inline float *voronoi_noise_eval_3d_fallback(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *voronoi_noise_eval_3d_sse2(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *voronoi_noise_eval_3d_sse4_1(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *voronoi_noise_eval_3d_avx(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *voronoi_noise_eval_3d_avx2(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float *voronoi_noise_eval_3d_avx512(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size);

static inline void voronoi_noise_init(struct VoronoiNoise *voronoi_noise) {
  voronoi_noise->frequency = DEFAULT_VORONOI_FREQUENCY;
  voronoi_noise->displacement = DEFAULT_VORONOI_DISPLACEMENT;
  voronoi_noise->seed = DEFAULT_VORONOI_SEED;
  voronoi_noise->enable_distance = DEFAULT_VORONOI_ENABLE_DISTANCE;
  voronoi_noise->position[0] = DEFAULT_VORONOI_POSITION_X;
  voronoi_noise->position[1] = DEFAULT_VORONOI_POSITION_Y;
  voronoi_noise->position[2] = DEFAULT_VORONOI_POSITION_Z;
  voronoi_noise->step = DEFAULT_VORONOI_STEP;
  voronoi_noise->parallel = DEFAULT_VORONOI_PARALLEL;

  switch (detect_simd_support()) {
#ifdef ARCH_32_64
    case SIMD_AVX512F:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_fallback;
      printf("Using AVX512\n");
      break;
    case SIMD_AVX2:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_avx2;
      printf("Using AVX2\n");
      break;
    case SIMD_AVX:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_fallback;
      printf("Using AVX\n");
      break;
    case SIMD_SSE4_1:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_fallback;
      printf("Using SSE4.1\n");
      break;
    case SIMD_SSE2:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_fallback;
      printf("Using SSE2\n");
      break;
#else
    case SIMD_NEON:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_fallback;
      break;
#endif
    default:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_fallback;
      printf("Using fallback\n");
      break;
  }
}

static inline float *voronoi_noise_eval_3d_fallback(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(float), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (voronoi_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim++) {
        float x = (voronoi_noise->position[0] * voronoi_noise->frequency) + (x_dim * voronoi_noise->step);
        float y = (voronoi_noise->position[1] * voronoi_noise->frequency) + (y_dim * voronoi_noise->step);
        float z = (voronoi_noise->position[2] * voronoi_noise->frequency) + (z_dim * voronoi_noise->step);

        int x_int = (x > 0.0 ? (int)x : (int)x - 1);
        int y_int = (y > 0.0 ? (int)y : (int)y - 1);
        int z_int = (z > 0.0 ? (int)z : (int)z - 1);

        float min_dist = 2147483647.0;
        float x_candidate = 0;
        float y_candidate = 0;
        float z_candidate = 0;

        for (int z_cur = z_int - 2; z_cur <= z_int + 2; z_cur++) {
          for (int y_cur = y_int - 2; y_cur <= y_int + 2; y_cur++) {
            for (int x_cur = x_int - 2; x_cur <= x_int + 2; x_cur++) {
              //printf("Value at x: %d is %d\n", x_dim, x_cur);

              float x_pos = x_cur + value_noise_3d(x_cur, y_cur, z_cur, voronoi_noise->seed);
              float y_pos = y_cur + value_noise_3d(x_cur, y_cur, z_cur, voronoi_noise->seed + 1);
              float z_pos = z_cur + value_noise_3d(x_cur, y_cur, z_cur, voronoi_noise->seed + 2);
              float x_dist = x_pos - x;
              float y_dist = y_pos - y;
              float z_dist = z_pos - z;
              float dist = x_dist * x_dist + y_dist * y_dist + z_dist * z_dist;

              if (dist < min_dist) {
                min_dist = dist;
                x_candidate = x_pos;
                y_candidate = y_pos;
                z_candidate = z_pos;
              }
            }
          }
        }

        float value;
        if (voronoi_noise->enable_distance) {
          float x_dist = x_candidate - x;
          float y_dist = y_candidate - y;
          float z_dist = z_candidate - z;
          value = sqrt(x_dist * x_dist + y_dist * y_dist + z_dist * z_dist) * SQRT_3 - 1.0;
        } else
          value = 0.0;

        *(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size)))) = value + (voronoi_noise->displacement * value_noise_3d((int)floor(x_candidate), (int)floor(y_candidate), (int)floor(z_candidate), voronoi_noise->seed));
      }
    }
  }
  return noise_set;
}

#ifdef ARCH_32_64
static inline float *voronoi_noise_eval_3d_avx2(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m256), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (voronoi_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 8) {
        __m256 x_vec = _mm256_add_ps(_mm256_set1_ps(voronoi_noise->position[0]), _mm256_mul_ps(_mm256_set_ps(x_dim + 7.0, x_dim + 6.0, x_dim + 5.0, x_dim + 4.0, x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm256_set1_ps(voronoi_noise->step * voronoi_noise->frequency)));
        float y = (voronoi_noise->position[1] * voronoi_noise->frequency) + (y_dim * voronoi_noise->step);
        float z = (voronoi_noise->position[2] * voronoi_noise->frequency) + (z_dim * voronoi_noise->step);

        __m256i x_int = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_blendv_ps(_mm256_sub_ps(x_vec, _mm256_set1_ps(1.0)), x_vec, _mm256_cmp_ps(x_vec, _mm256_setzero_ps(), _CMP_GT_OQ))));
        int y_int = (y > 0.0 ? (int)y : (int)y - 1);
        int z_int = (z > 0.0 ? (int)z : (int)z - 1);

        __m256 min_dist = _mm256_set1_ps(2147483647.0);
        __m256 x_candidate = _mm256_setzero_ps();
        __m256 y_candidate = _mm256_setzero_ps();
        __m256 z_candidate = _mm256_setzero_ps();

        for (int z_cur = z_int - 2; z_cur <= z_int + 2; z_cur++) {
          for (int y_cur = y_int - 2; y_cur <= y_int + 2; y_cur++) {
            for (int x_cur = -2; x_cur <= 2; x_cur++) {
              __m256i x_cur_temp = _mm256_add_epi32(x_int, _mm256_set1_epi32(x_cur));

              __m256 x_pos = _mm256_add_ps(_mm256_cvtepi32_ps(x_cur_temp), value_noise_3d_avx2(x_cur_temp, y_cur, z_cur, voronoi_noise->seed));
              __m256 y_pos = _mm256_add_ps(_mm256_set1_ps((float)y_cur), value_noise_3d_avx2(x_cur_temp, y_cur, z_cur, voronoi_noise->seed + 1));
              __m256 z_pos = _mm256_add_ps(_mm256_set1_ps((float)z_cur), value_noise_3d_avx2(x_cur_temp, y_cur, z_cur, voronoi_noise->seed + 2));
              __m256 x_dist = _mm256_sub_ps(x_pos, x_vec);
              __m256 y_dist = _mm256_sub_ps(y_pos, _mm256_set1_ps(y));
              __m256 z_dist = _mm256_sub_ps(z_pos, _mm256_set1_ps(z));
              __m256 dist = _mm256_add_ps(_mm256_mul_ps(x_dist, x_dist), _mm256_add_ps(_mm256_mul_ps(y_dist, y_dist), _mm256_mul_ps(z_dist, z_dist)));

              __m256 dist_cmp_mask = _mm256_cmp_ps(dist, min_dist, _CMP_LT_OQ);
              min_dist = _mm256_blendv_ps(min_dist, dist, dist_cmp_mask);
              x_candidate = _mm256_blendv_ps(x_candidate, x_pos, dist_cmp_mask);
              y_candidate = _mm256_blendv_ps(y_candidate, y_pos, dist_cmp_mask);
              z_candidate = _mm256_blendv_ps(z_candidate, z_pos, dist_cmp_mask);
            }
          }
        }

        __m256 value;
        if (voronoi_noise->enable_distance) {
          __m256 x_dist = _mm256_sub_ps(x_candidate, x_vec);
          __m256 y_dist = _mm256_sub_ps(y_candidate, _mm256_set1_ps(y));
          __m256 z_dist = _mm256_sub_ps(z_candidate, _mm256_set1_ps(z));
          value = _mm256_sub_ps(_mm256_mul_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(x_dist, x_dist), _mm256_add_ps(_mm256_mul_ps(y_dist, y_dist), _mm256_mul_ps(z_dist, z_dist)))), _mm256_set1_ps(SQRT_3)), _mm256_set1_ps(1.0));
        } else
          value = _mm256_setzero_ps();

        _mm256_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), _mm256_add_ps(value, _mm256_mul_ps(_mm256_set1_ps(voronoi_noise->displacement), value_noise_3d_avx2_full(_mm256_cvtps_epi32(_mm256_floor_ps(x_candidate)), _mm256_cvtps_epi32(_mm256_floor_ps(y_candidate)), _mm256_cvtps_epi32(_mm256_floor_ps(z_candidate)), voronoi_noise->seed))));
      }
    }
  }
  return noise_set;
}
#endif

#endif  // VORONOI_NOISE_H
