#pragma once
#ifndef VORONOI_NOISE_H
#define VORONOI_NOISE_H

#include "common.h"

#define DEFAULT_VORONOI_FREQUENCY 1.0
#define DEFAULT_VORONOI_DISPLACEMENT 1.0
#define DEFAULT_VORONOI_SEED 0
#define DEFAULT_VORONOI_ENABLE_DISTANCE 0

struct VoronoiNoise {
  float frequency;
  float displacement;
  int seed;
  unsigned char enable_distance;
};

static inline void voronoi_noise_init(struct VoronoiNoise *voronoi_noise) {
  voronoi_noise->frequency = DEFAULT_VORONOI_FREQUENCY;
  voronoi_noise->displacement = DEFAULT_VORONOI_DISPLACEMENT;
  voronoi_noise->seed = DEFAULT_VORONOI_SEED;
  voronoi_noise->enable_distance = DEFAULT_VORONOI_ENABLE_DISTANCE;
}

static inline float voronoi_noise_eval_3d(struct VoronoiNoise *voronoi_noise, float x, float y, float z) {
  x *= voronoi_noise->frequency;
  y *= voronoi_noise->frequency;
  z *= voronoi_noise->frequency;

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
    value = (sqrt(x_dist * x_dist + y_dist * y_dist + z_dist * z_dist)) * SQRT_3 - 1.0;
  } else
    value = 0.0;

  return value + (voronoi_noise->displacement * (float)value_noise_3d(fast_floor(x_candidate), fast_floor(y_candidate), fast_floor(z_candidate), voronoi_noise->seed));
}

#endif  // VORONOI_NOISE_H
