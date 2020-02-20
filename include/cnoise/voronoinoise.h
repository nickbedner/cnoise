#pragma once
#ifndef VORONOI_NOISE_H
#define VORONOI_NOISE_H

#include "noisecommon.h"

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

static inline float *voronoi_noise_eval_1d(struct VoronoiNoise *voronoi_noise, size_t x_size);
static inline float *voronoi_noise_eval_2d(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size);
static inline float *voronoi_noise_eval_3d(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size);
static inline float voronoi_noise_eval_3d_single(struct VoronoiNoise *voronoi_noise, float x_pos, float y_pos, float z_pos);
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
      break;
    case SIMD_AVX2:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_avx2;
      break;
    case SIMD_AVX:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_avx;
      break;
    case SIMD_SSE4_1:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_sse4_1;
      break;
    case SIMD_SSE2:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_sse2;
      break;
#else
    case SIMD_NEON:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_fallback;
      break;
#endif
    default:
      voronoi_noise->voronoi_func = &voronoi_noise_eval_3d_fallback;
      break;
  }
}

static inline float *voronoi_noise_eval_1d(struct VoronoiNoise *voronoi_noise, size_t x_size) {
  return voronoi_noise->voronoi_func(voronoi_noise, x_size, 1, 1);
}

static inline float *voronoi_noise_eval_2d(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size) {
  return voronoi_noise->voronoi_func(voronoi_noise, x_size, y_size, 1);
}

static inline float *voronoi_noise_eval_3d(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size) {
  return voronoi_noise->voronoi_func(voronoi_noise, x_size, y_size, z_size);
}

static inline float voronoi_noise_eval_3d_single(struct VoronoiNoise *voronoi_noise, float x_pos, float y_pos, float z_pos) {
  float x = (voronoi_noise->position[0] + (x_pos * voronoi_noise->step)) * voronoi_noise->frequency;
  float y = (voronoi_noise->position[1] + (y_pos * voronoi_noise->step)) * voronoi_noise->frequency;
  float z = (voronoi_noise->position[2] + (z_pos * voronoi_noise->step)) * voronoi_noise->frequency;

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

  return value + (voronoi_noise->displacement * value_noise_3d((int)floor(x_candidate), (int)floor(y_candidate), (int)floor(z_candidate), voronoi_noise->seed));
}

static inline float *voronoi_noise_eval_3d_fallback(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size) {
#ifdef CUSTOM_ALLOCATOR
  float *noise_set = malloc(sizeof(float) * x_size * y_size * z_size);
#else
  float *noise_set = noise_allocate(sizeof(float), sizeof(float) * x_size * y_size * z_size);
#endif
#pragma omp parallel for collapse(3) if (voronoi_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim++) {
        float x = (voronoi_noise->position[0] + (x_dim * voronoi_noise->step)) * voronoi_noise->frequency;
        float y = (voronoi_noise->position[1] + (y_dim * voronoi_noise->step)) * voronoi_noise->frequency;
        float z = (voronoi_noise->position[2] + (z_dim * voronoi_noise->step)) * voronoi_noise->frequency;

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
static inline float *voronoi_noise_eval_3d_sse2(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m128), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (voronoi_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 4) {
        __m128 x_vec = _mm_mul_ps(_mm_add_ps(_mm_set1_ps(voronoi_noise->position[0]), _mm_mul_ps(_mm_set_ps(x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm_set1_ps(voronoi_noise->step))), _mm_set1_ps(voronoi_noise->frequency));
        float y = (voronoi_noise->position[1] * voronoi_noise->frequency) + (y_dim * voronoi_noise->step);
        float z = (voronoi_noise->position[2] * voronoi_noise->frequency) + (z_dim * voronoi_noise->step);

        __m128i x_int;
        for (int x_num = 0; x_num < 4; x_num++)
          *(((int32_t *)&x_int) + x_num) = (*(((float *)&x_vec) + x_num) > 0.0 ? (int)*(((float *)&x_vec) + x_num) : (int)*(((float *)&x_vec) + x_num) - 1);
        int y_int = (y > 0.0 ? (int)y : (int)y - 1);
        int z_int = (z > 0.0 ? (int)z : (int)z - 1);

        __m128 min_dist = _mm_set1_ps(2147483647.0);
        __m128 x_candidate = _mm_setzero_ps();
        __m128 y_candidate = _mm_setzero_ps();
        __m128 z_candidate = _mm_setzero_ps();

        for (int z_cur = z_int - 2; z_cur <= z_int + 2; z_cur++) {
          for (int y_cur = y_int - 2; y_cur <= y_int + 2; y_cur++) {
            for (int x_cur = -2; x_cur <= 2; x_cur++) {
              __m128i x_cur_temp = _mm_add_epi32(x_int, _mm_set1_epi32(x_cur));

              __m128 x_pos = _mm_add_ps(_mm_cvtepi32_ps(x_cur_temp), value_noise_3d_sse2(x_cur_temp, y_cur, z_cur, voronoi_noise->seed));
              __m128 y_pos = _mm_add_ps(_mm_set1_ps((float)y_cur), value_noise_3d_sse2(x_cur_temp, y_cur, z_cur, voronoi_noise->seed + 1));
              __m128 z_pos = _mm_add_ps(_mm_set1_ps((float)z_cur), value_noise_3d_sse2(x_cur_temp, y_cur, z_cur, voronoi_noise->seed + 2));
              __m128 x_dist = _mm_sub_ps(x_pos, x_vec);
              __m128 y_dist = _mm_sub_ps(y_pos, _mm_set1_ps(y));
              __m128 z_dist = _mm_sub_ps(z_pos, _mm_set1_ps(z));
              __m128 dist = _mm_add_ps(_mm_mul_ps(x_dist, x_dist), _mm_add_ps(_mm_mul_ps(y_dist, y_dist), _mm_mul_ps(z_dist, z_dist)));

              for (int check_num = 0; check_num < 4; check_num++) {
                float dist_extract = *(((float *)&dist) + check_num);
                float min_dist_extract = *(((float *)&min_dist) + check_num);

                if (dist_extract < min_dist_extract) {
                  *(((float *)&min_dist) + check_num) = dist_extract;
                  *(((float *)&x_candidate) + check_num) = *(((float *)&x_pos) + check_num);
                  *(((float *)&y_candidate) + check_num) = *(((float *)&y_pos) + check_num);
                  *(((float *)&z_candidate) + check_num) = *(((float *)&z_pos) + check_num);
                }
              }
            }
          }
        }

        __m128 value;
        if (voronoi_noise->enable_distance) {
          __m128 x_dist = _mm_sub_ps(x_candidate, x_vec);
          __m128 y_dist = _mm_sub_ps(y_candidate, _mm_set1_ps(y));
          __m128 z_dist = _mm_sub_ps(z_candidate, _mm_set1_ps(z));
          value = _mm_sub_ps(_mm_mul_ps(_mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(x_dist, x_dist), _mm_add_ps(_mm_mul_ps(y_dist, y_dist), _mm_mul_ps(z_dist, z_dist)))), _mm_set1_ps(SQRT_3)), _mm_set1_ps(1.0));
        } else
          value = _mm_setzero_ps();

        for (int floor_candidate_num = 0; floor_candidate_num < 4; floor_candidate_num++) {
          *(((float *)&x_candidate) + floor_candidate_num) = floor(*(((float *)&x_candidate) + floor_candidate_num));
          *(((float *)&y_candidate) + floor_candidate_num) = floor(*(((float *)&y_candidate) + floor_candidate_num));
          *(((float *)&z_candidate) + floor_candidate_num) = floor(*(((float *)&z_candidate) + floor_candidate_num));
        }
        _mm_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), _mm_add_ps(value, _mm_mul_ps(_mm_set1_ps(voronoi_noise->displacement), value_noise_3d_sse2_full(_mm_cvtps_epi32(x_candidate), _mm_cvtps_epi32(y_candidate), _mm_cvtps_epi32(z_candidate), voronoi_noise->seed))));
      }
    }
  }
  return noise_set;
}

static inline float *voronoi_noise_eval_3d_sse4_1(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m128), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (voronoi_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 4) {
        __m128 x_vec = _mm_mul_ps(_mm_add_ps(_mm_set1_ps(voronoi_noise->position[0]), _mm_mul_ps(_mm_set_ps(x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm_set1_ps(voronoi_noise->step))), _mm_set1_ps(voronoi_noise->frequency));
        float y = (voronoi_noise->position[1] * voronoi_noise->frequency) + (y_dim * voronoi_noise->step);
        float z = (voronoi_noise->position[2] * voronoi_noise->frequency) + (z_dim * voronoi_noise->step);

        __m128i x_int = _mm_cvtps_epi32(_mm_floor_ps(_mm_blendv_ps(_mm_sub_ps(x_vec, _mm_set1_ps(1.0)), x_vec, _mm_cmp_ps(x_vec, _mm_setzero_ps(), _CMP_GT_OQ))));
        int y_int = (y > 0.0 ? (int)y : (int)y - 1);
        int z_int = (z > 0.0 ? (int)z : (int)z - 1);

        __m128 min_dist = _mm_set1_ps(2147483647.0);
        __m128 x_candidate = _mm_setzero_ps();
        __m128 y_candidate = _mm_setzero_ps();
        __m128 z_candidate = _mm_setzero_ps();

        for (int z_cur = z_int - 2; z_cur <= z_int + 2; z_cur++) {
          for (int y_cur = y_int - 2; y_cur <= y_int + 2; y_cur++) {
            for (int x_cur = -2; x_cur <= 2; x_cur++) {
              __m128i x_cur_temp = _mm_add_epi32(x_int, _mm_set1_epi32(x_cur));

              __m128 x_pos = _mm_add_ps(_mm_cvtepi32_ps(x_cur_temp), value_noise_3d_sse4_1(x_cur_temp, y_cur, z_cur, voronoi_noise->seed));
              __m128 y_pos = _mm_add_ps(_mm_set1_ps((float)y_cur), value_noise_3d_sse4_1(x_cur_temp, y_cur, z_cur, voronoi_noise->seed + 1));
              __m128 z_pos = _mm_add_ps(_mm_set1_ps((float)z_cur), value_noise_3d_sse4_1(x_cur_temp, y_cur, z_cur, voronoi_noise->seed + 2));
              __m128 x_dist = _mm_sub_ps(x_pos, x_vec);
              __m128 y_dist = _mm_sub_ps(y_pos, _mm_set1_ps(y));
              __m128 z_dist = _mm_sub_ps(z_pos, _mm_set1_ps(z));
              __m128 dist = _mm_add_ps(_mm_mul_ps(x_dist, x_dist), _mm_add_ps(_mm_mul_ps(y_dist, y_dist), _mm_mul_ps(z_dist, z_dist)));

              __m128 dist_cmp_mask = _mm_cmp_ps(dist, min_dist, _CMP_LT_OQ);
              min_dist = _mm_blendv_ps(min_dist, dist, dist_cmp_mask);
              x_candidate = _mm_blendv_ps(x_candidate, x_pos, dist_cmp_mask);
              y_candidate = _mm_blendv_ps(y_candidate, y_pos, dist_cmp_mask);
              z_candidate = _mm_blendv_ps(z_candidate, z_pos, dist_cmp_mask);
            }
          }
        }

        __m128 value;
        if (voronoi_noise->enable_distance) {
          __m128 x_dist = _mm_sub_ps(x_candidate, x_vec);
          __m128 y_dist = _mm_sub_ps(y_candidate, _mm_set1_ps(y));
          __m128 z_dist = _mm_sub_ps(z_candidate, _mm_set1_ps(z));
          value = _mm_sub_ps(_mm_mul_ps(_mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(x_dist, x_dist), _mm_add_ps(_mm_mul_ps(y_dist, y_dist), _mm_mul_ps(z_dist, z_dist)))), _mm_set1_ps(SQRT_3)), _mm_set1_ps(1.0));
        } else
          value = _mm_setzero_ps();

        _mm_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), _mm_add_ps(value, _mm_mul_ps(_mm_set1_ps(voronoi_noise->displacement), value_noise_3d_sse4_1_full(_mm_cvtps_epi32(_mm_floor_ps(x_candidate)), _mm_cvtps_epi32(_mm_floor_ps(y_candidate)), _mm_cvtps_epi32(_mm_floor_ps(z_candidate)), voronoi_noise->seed))));
      }
    }
  }
  return noise_set;
}

static inline float *voronoi_noise_eval_3d_avx(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m256), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (voronoi_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 8) {
        __m256 x_vec = _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(voronoi_noise->position[0]), _mm256_mul_ps(_mm256_set_ps(x_dim + 7.0, x_dim + 6.0, x_dim + 5.0, x_dim + 4.0, x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm256_set1_ps(voronoi_noise->step))), _mm256_set1_ps(voronoi_noise->frequency));
        float y = (voronoi_noise->position[1] + (y_dim * voronoi_noise->step)) * voronoi_noise->frequency;
        float z = (voronoi_noise->position[2] + (z_dim * voronoi_noise->step)) * voronoi_noise->frequency;

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
              __m128i x_cur_temp_low = _mm_add_epi32(_mm256_extractf128_si256(x_int, 0), _mm_set1_epi32(x_cur));
              __m128i x_cur_temp_high = _mm_add_epi32(_mm256_extractf128_si256(x_int, 1), _mm_set1_epi32(x_cur));
              __m256i x_cur_temp = _mm256_set_m128i(x_cur_temp_high, x_cur_temp_low);

              __m256 x_pos = _mm256_add_ps(_mm256_cvtepi32_ps(x_cur_temp), value_noise_3d_avx(x_cur_temp, y_cur, z_cur, voronoi_noise->seed));
              __m256 y_pos = _mm256_add_ps(_mm256_set1_ps((float)y_cur), value_noise_3d_avx(x_cur_temp, y_cur, z_cur, voronoi_noise->seed + 1));
              __m256 z_pos = _mm256_add_ps(_mm256_set1_ps((float)z_cur), value_noise_3d_avx(x_cur_temp, y_cur, z_cur, voronoi_noise->seed + 2));
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

        _mm256_store_ps(noise_set + (x_dim + (y_dim * x_size) + (z_dim * (x_size * y_size))), _mm256_add_ps(value, _mm256_mul_ps(_mm256_set1_ps(voronoi_noise->displacement), value_noise_3d_avx_full(_mm256_cvtps_epi32(_mm256_floor_ps(x_candidate)), _mm256_cvtps_epi32(_mm256_floor_ps(y_candidate)), _mm256_cvtps_epi32(_mm256_floor_ps(z_candidate)), voronoi_noise->seed))));
      }
    }
  }
  return noise_set;
}

static inline float *voronoi_noise_eval_3d_avx2(struct VoronoiNoise *voronoi_noise, size_t x_size, size_t y_size, size_t z_size) {
  float *noise_set = noise_allocate(sizeof(__m256), sizeof(float) * x_size * y_size * z_size);
#pragma omp parallel for collapse(3) if (voronoi_noise->parallel)
  for (int z_dim = 0; z_dim < z_size; z_dim++) {
    for (int y_dim = 0; y_dim < y_size; y_dim++) {
      for (int x_dim = 0; x_dim < x_size; x_dim += 8) {
        __m256 x_vec = _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(voronoi_noise->position[0]), _mm256_mul_ps(_mm256_set_ps(x_dim + 7.0, x_dim + 6.0, x_dim + 5.0, x_dim + 4.0, x_dim + 3.0, x_dim + 2.0, x_dim + 1.0, x_dim), _mm256_set1_ps(voronoi_noise->step))), _mm256_set1_ps(voronoi_noise->frequency));
        float y = (voronoi_noise->position[1] + (y_dim * voronoi_noise->step)) * voronoi_noise->frequency;
        float z = (voronoi_noise->position[2] + (z_dim * voronoi_noise->step)) * voronoi_noise->frequency;

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
